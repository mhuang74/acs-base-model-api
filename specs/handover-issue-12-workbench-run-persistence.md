# Handover — Issue #12: Workbench Run persistence, Restore chain, v2 Session export/import

Repo: `acs_workbench_export_import` (ACS base-model platform).
Spec: GitHub issue #12 on `<owner>/acs-base-model-api`; vocabulary in root `CONTEXT.md`; decisions in `docs/adr/0001-best-effort-reproduction-and-v2-session-format.md`. Read both before touching anything.

Two phases on one data model:
- **Phase 1** — Run persistence (extended Sampling settings per Run), Restore chain (Draft → last successful Run → defaults), compare lane prefill.
- **Phase 2** — v2 Export documents + `POST /workbench/import`.

---

## 1. Status dashboard

### DONE (code written, on disk)

| File | Change |
|---|---|
| `base_model_wrapper/alembic/versions/0046_run_sampling_settings.py` | NEW migration. Additive columns on `chat_snapshots`: `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (Float/Integer), `seed` (BigInteger), `stop` (Text), `logprobs_count` (Integer), `recorded_incomplete` (Boolean NOT NULL server_default false), `model_echo` (Text). **Applied to dev DB — currently at 0046 head.** |
| `base_model_wrapper/src/wrapper/models.py` | `ChatSnapshot` gains the 11 columns (with comments). Docstring reworded to "One Run (single-pane Continue)". `ChatGeneration` deliberately NOT changed. |
| `base_model_wrapper/src/wrapper/workbench_generations.py` | `_absorb_chunk_text` now returns 3-tuple `(delta, saw_done, model_echo)` — last non-empty top-level `model` in the SSE chunks wins. `_run_generation_task` tracks `upstream_model_echo` and writes the extended fields on the `ChatSnapshot` (values read back off the clamped `body`; seed only-when-set; `stop = body["stop"][0] or None`; `recorded_incomplete=False`; `model_echo`). |
| `base_model_wrapper/src/wrapper/services/workbench.py` | New helpers: `composer_settings_from_run(snap)` (shape one Run's settings for the composer; seed → str or ""), `latest_restore_run(session, session_id)` (newest `ChatSnapshot` with `cancelled.is_(False)`, `ts desc, id desc`), `latest_compare_snapshot(session, session_id)`. `CompareSnapshot` added to imports. |
| `base_model_wrapper/src/wrapper/routes/workbench.py` | `chat_open`: fetches `latest_restore_run` → `composer_settings`, `latest_compare_snapshot.lanes` → `compare_prefill`; passes both to `_render_chat`. `_render_chat` signature extended with `composer_settings: dict \| None = None` and `compare_prefill: list \| None = None`; both forwarded into the template context as `composer_settings` / `compare_prefill_lanes`. |
| `base_model_wrapper/src/wrapper/templates/workbench.html` | Max-tokens + temperature inputs prefill from `composer_settings` (fallback `chat.last_*`). Sampling panel rebuilt as ONE grid; every input prefills from `{% set cs = composer_settings if composer_settings else {} %}` with built-in defaults as fallback (never empty — empty strings would 422 the float Form params). Single logprobs toggle wired to `cs.logprobs_on` / `cs.logprobs_count` (checked + disabled state server-rendered). Snapshot history meta now shows `· incomplete record` / `· reproducible` (temp==0 or seed set) / `· non-reproducible`. URL Draft JS added after the logprobs wiring (~line 1653): `DRAFT_KEYS`, `draftParams()`, `writeDraft()` (`history.replaceState`, preserves `mode` param), `applyDraft()` on load (Draft beats server prefill), form-level `input`/`change` listeners (150 ms debounce), re-encode at submit. |
| `base_model_wrapper/src/wrapper/templates/workbench_compare.html` | `cmp-config` JSON gains `"prefillLanes": compare_prefill_lanes or []`. Lane-seed block builds lanes from `cfg.prefillLanes` (config-only, capped at `maxLanes`) else falls back to the default two lanes / one inert lane when `noKey`. Diff verified minimal (`git diff` clean except those two hunks). |
| `base_model_wrapper/tests/test_workbench_generations.py` | `_absorb_chunk_text` unit tests re-pinned for the 3-tuple (old 2-tuple tests deleted per contract change). |

### Verification so far
- `pytest tests/test_workbench_generations.py` (file alone): **52 passed** after the persistence + absorb changes.
- Migration applied: dev DB at `0046_run_sampling_settings (head)`; columns verified via `information_schema`.

### IN FLIGHT / BROKEN — see §3 before doing anything else

1. Migration 0046 does **not** mark pre-migration rows (defect, must fix + re-apply).
2. Combined-run failure: `test_upstream_unreachable_threads_through_sse_and_api_requests` (diagnosis in §3.3).

### NOT STARTED (Phase 2 + remaining Phase 1 tests)
See §5.

---

## 2. Design decisions — LOCKED, do not re-litigate

1. **Run record = `chat_snapshots`** (single-pane Continue). Compare lanes keep their existing persistence (`chat_generations.compare_config` JSONB + `CompareSnapshot.lanes`); the spec says that path "already persists full configs today". Table names unchanged.
2. **`recorded_incomplete`** lives on `chat_snapshots`: `False` = fully recorded, `True` = pre-migration / v1-imported. Pre-migration rows are marked by a migration `UPDATE` (see §3.1) — the *settings* columns stay NULL (that is the forbidden-to-backfill part).
3. **Model echo**: `_absorb_chunk_text`'s 3rd return value; last non-empty `model` string wins; stored on `snapshot.model_echo` alongside the requested short id (`snapshot.model`). **No** `model_echo` on `chat_generations` / compare lanes (out of the spec's compare-snapshot field list; lanes travel verbatim as records).
4. **stop** stored as a single string (first element of the body's stop list) — matches `_sanitize_compare_lane` / `_lane_from_generation` convention.
5. **seed** only-when-set: blank seed → NULL (upstream drew randomly; vLLM never echoes the effective seed). Reproducible ⇔ `seed is not None or temperature == 0`.
6. **logprobs toggle/count** = `logprobs_count` nullable int (requested top-k when on, NULL when off). The existing `logprobs` JSONB stays the heatmap payload only. Normalize `body.get("logprobs") <= 0 → None` at write (see §3.2).
7. **Restore chain**: server owns leg 2 (newest non-cancelled `ChatSnapshot` → composer values; cancelled excluded; errored generations never commit so they fall out). Defaults are built-ins (`_CHAT_DEFAULT_MAX_TOKENS=200`, `_CHAT_DEFAULT_TEMPERATURE=1.0`, top_p 1 / top_k −1 / min_p 0 / penalties 0,0,1). Model + API key keep the existing chat-row restore. **Composer prefill comes from the snapshot, NOT `chat.last_max_tokens/last_temperature`** (cancelled-with-text runs update those but must not feed the chain). Keep writing `chat.last_*` in the task (v1-era readers).
8. **Draft** = client-side JS only (per spec: "pure client-side JS… verified manually in the browser; every server-side leg is covered at the primary seam"). Draft beats server-rendered values; submit re-encodes (the submit is fetch-based, never navigates).
9. **v2 Export document** shape:
   ```
   {"version": 2,
    "session": {"id", "title", "model", "created_at", "updated_at", "prompt"},
    "runs": [{"ts", "prompt_before", "completion_text", "cancelled", "model",
              "model_echo", "n_completion",
              "sampling": {"max_tokens", "temperature", "top_p", "top_k", "min_p",
                           "presence_penalty", "frequency_penalty", "repetition_penalty",
                           "seed", "stop", "logprobs_count"},
              "recorded_incomplete"}],
    "compare_snapshots": [{"ts", "prompt", "n_lanes", "lanes": [<as stored>]}]}
   ```
   - `n_completion` IS included (decision: keeps the "+N tok" display honest for imported runs and makes round-trips exact; no identity leak).
   - `logprobs` JSONB, `compare_run_id`, user identity, API keys, request ids: NOT in the document.
   - runs in `ts` asc order; `runs` may be empty.
10. **Cutover**: `GET /workbench/{session_id}/export.json` (v2, `application/json` attachment) REPLACES `GET /workbench/{session_id}/export.jsonl` (v1). Delete `chat_export_jsonl` + `_session_to_jsonl_record` + template link + `main.py` compat entries (`__all__` too). `export-all.jsonl` keeps its path/media type but now emits **ndjson of v2 documents**. v1 tolerance exists only on IMPORT.
11. **Import** `POST /workbench/import`, raw JSON body, cookie-authed. Error contract mirrors Loom's importer: 401 `{"error":{"code":"unauthorized"}}`, 400 `bad_json` / `bad_shape`, 413 `too_large`. Success: 200 `{"session_id": "<uuid>"}`.
    - Shape sniff: `version == 2` and dict `session` and list `runs` → v2; else dict with list `snapshots` → v1 record; else `bad_shape`. Exactly one document per call (ndjson files fail `request.json()` → `bad_json`).
    - **Whitelist only**; fresh UUIDs; incoming ids never read; imported session owned by the importing user; no origin-ownership check (interchange).
    - **Clamp-on-import = run the sanitized sampling dict through `_build_lane_body` and read the clamped values back** — zero drift vs the wire clamp. Pre-normalize first: bool-guard ints (`seed`/`top_k`/`max_tokens`/`logprobs` — bool is an int subclass, exclude), floats via try/except → None, stop str-only.
    - NULL sampling fields in a v2 doc stay NULL (unset) — do NOT fabricate defaults; `max_tokens`/`temperature` NULL → 200 / 1.0 (NOT NULL columns).
    - **v1 imports**: extended settings → NULL + `recorded_incomplete=True` (honest; "defaulted extended parameters" describes restore-as-defaults behavior). Session `model` → None. v1 `max_tokens`/`temperature` clamped too.
    - Timestamps preserved (`datetime.fromisoformat`, fallback `now()`); title capped 120 (loom precedent); `prompt_before`/`completion_text` capped 1,000,000 chars on import (pathological-file defense, round-trip safe).
    - Cap: **2000 Runs per Session** (`too_large` over). Count `len(runs) + total compare lanes` ≤ 2000. Lanes per imported snapshot truncated to `_COMPARE_MAX_LANES` (6); imported `CompareSnapshot.compare_run_id = None`.
12. **Pure-function seam**: v2 serializer + import parser live in a NEW `services/session_exchange.py`; it imports `_build_lane_body` from `routes.workbench` **lazily inside a function** (import-cycle avoidance; precedent: `_maybe_write_compare_snapshot` imports `_COMPARE_SNAPSHOT_LIMIT` the same way). Unit tests import them via `wrapper.main` compat re-exports (house pattern).

---

## 3. Outstanding defects — fix FIRST

### 3.1 Migration 0046 is missing the completeness-marker backfill (dev DB already applied!)
`recorded_incomplete` has `server_default false` with no backfill, so every pre-existing row reads as *fully recorded* while its extended settings are NULL — the exact lie user story 11 forbids ("pre-migration Runs visibly marked `recorded_incomplete`").

Fix: in `upgrade()`, after the column adds, add:
```python
op.execute("UPDATE chat_snapshots SET recorded_incomplete = true")
```
The marker backfill is the honest label; the forbidden backfill is the settings values (they stay NULL). Also fix the migration docstring, which currently implies "no data backfill" covers the marker — it doesn't.

Then re-apply (the DB is at 0046 without the backfill):
```bash
cd base_model_wrapper
DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  uv run --python 3.12 --extra dev alembic downgrade 0045
DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  uv run --python 3.12 --extra dev alembic upgrade head
```
(Note: `alembic -x url=...` fails in this repo — env.py reads `DATABASE_URL`; use the env-var form above.)

### 3.2 Normalize `logprobs_count` 0 → NULL at the snapshot write
`_build_lane_body` can emit `body["logprobs"] == 0` (its clamp is `max(0, min(int, 20))`). At the `ChatSnapshot(...)` write in `_run_generation_task`, change `logprobs_count=body.get("logprobs")` to `logprobs_count=(body.get("logprobs") or None)` so restore never renders an "on" toggle with count 0. (`logprobs_requested` already gates on truthiness, so this only fixes the recorded value.)

### 3.3 Known test failure — `test_upstream_unreachable_threads_through_sse_and_api_requests`

Facts:
- `pytest tests/test_workbench_generations.py` alone: **passes** (52).
- `pytest tests/test_chat_history.py tests/test_workbench_generations.py` combined: fails with
  `AssertionError: assert 'upstream_4xx' == 'upstream_unreachable'`.

Verified intact: the task's error branch (`workbench_generations.py:527-548`) and the test's fake (`_make_gated_error_upstream`, tests file :1460-1470) which yields exactly one event `{"kind": "error", "status": 0, "upstream_kind": "upstream_unreachable"}`. That event makes `upstream_error_kind = upstream_kind or (...)` → `"upstream_unreachable"` **unconditionally** — this test's own task cannot produce `upstream_4xx`.

Leading hypothesis: the helper `_wait_for_terminal_row` (tests file :1430-1457) fetches the **globally most-recent** `/workbench` `ApiRequest` row (`order_by(ApiRequest.ts.desc())`, no scoping). With combined-run timing, some other generation task's row (or a stray row in the shared dev DB — it accumulates across runs and any dev server traffic) can shadow this test's row. This is a test-isolation flaw, not a product bug.

Next steps:
1. Reproduce: run the combined command again (§6). Then run `test_workbench_generations.py` alone again.
2. Check for stray writers: `pgrep -fl uvicorn` / `pgrep -fl "uv run"` against the shared DB; `docker ps` (only `acs-hiring-devdb` should be up).
3. Fix the helper to scope the query — e.g. capture `gen`'s `started_at` and filter `ApiRequest.ts >= started_at` scoped to the test's `key_id`, or take `ApiRequest` rows for that key and pick the newest with `ts >= started_at`. Keep it deterministic and isolated.

Do NOT "fix" this by weakening the product error-threading code — it is verified correct for this input.

---

## 4. File-by-file map (current working tree)

Modified: `models.py`, `workbench_generations.py`, `routes/workbench.py`, `services/workbench.py`, `templates/workbench.html`, `templates/workbench_compare.html`, `tests/test_workbench_generations.py`. New: `alembic/versions/0046_run_sampling_settings.py`.

Untouched so far: `main.py` compat, export routes (`export.jsonl`, `export-all.jsonl`, `_session_to_jsonl_record`), template export menu, `workbench.md` docs, `tests/test_chat_history.py`.

**Template gotchas (already burned twice — re-read after every edit):**
- `workbench.html`: composer form is ~line 1146-1271; sampling `<details>` block 1220-1250 (single grid, `{% set cs %}` at 1240); logprobs toggle 1252-1262; URL Draft JS 1653-1726; snapshot meta line ~1297.
- The template `{% set cs = ... %}` sits inside the `<details>` block but at template scope (no enclosing for-loop), so the logprobs label after `</details>` can see it — verify with a render.

---

## 5. Remaining work

### Phase 1 leftovers
1. §3 defects.
2. Route/render tests (extend existing files, per-file preamble convention):
   - `test_workbench_generations.py`: Run persistence — POST a generation with the full panel + fake upstream echoing `"model"`; assert `ChatSnapshot` carries clamped settings, seed only-when-set, `logprobs_count`, `model_echo`, `recorded_incomplete=False`; clamping case (top_p 5 → 1.0, temperature 500 → 100); cancelled-with-text commits `cancelled=True` snapshot.
   - `test_chat_history.py` (has client + `_make_chat_session`): restore chain render — insert snapshots directly; bare load prefills newest successful Run's values; cancelled-only session → built-in defaults; `recorded_incomplete=True` + NULL extended → defaults + "incomplete record" marker visible; compare prefill: insert `CompareSnapshot`, assert `cmp-config` JSON contains its lanes.
3. Compare lane logprobs (small, in-scope): `compare_start`'s `compare_config` gains `"logprobs": body.get("logprobs")`; `_sanitize_compare_lane` whitelists `logprobs` (int, bool-guard, else None); `_lane_from_generation` emits it. This makes snapshot restore/prefill carry the heatmap toggle.

### Phase 2 build order
1. `services/session_exchange.py` — `session_to_v2_document(chat, snapshots, compare_snapshots)` + `parse_export_document(payload)` (sniff, whitelist, clamp, cap) + lazy `_build_lane_body` clamp helper. (§2.9-12.)
2. `routes/workbench.py`:
   - `GET /workbench/{session_id}/export.json` (new) — ownership-gated like `export.txt` (redirect 303 to /login unauth; 404 not-owned); fetch snapshots ts asc (ALL, not the 20-cap) + compare snapshots ts asc; `Response(json.dumps(doc, indent=2), media_type="application/json", attachment filename f"{safe_title or 'chat'}.json")`.
   - `GET /workbench/export-all.jsonl` — ndjson of v2 docs (per session: snapshots + compare snapshots).
   - DELETE `chat_export_jsonl` + `_session_to_jsonl_record`.
   - `POST /workbench/import` — `chat_import` (§2.11).
3. `main.py`: `__all__` + re-exports — remove `_session_to_jsonl_record`, `chat_export_jsonl`; add `_session_to_v2_document`, `_parse_import_document` (or chosen names), `chat_export_json`, `chat_import` (plain assignments, house style).
4. `templates/workbench.html`: export dropdown item → `/workbench/{{ chat.id }}/export.json` ("Full session `.json` — v2 Export document: every Run with full sampling settings + compare snapshots."). Sidebar Import affordance copying `loom.html`'s pattern (hidden `<input type="file" id="wb-import-file" accept="application/json,.json">` + button; JS reads file, `fetch('/workbench/import', {method:'POST', headers:{'Content-Type':'application/json'}, body})`, on success `window.location.href = '/workbench/' + data.session_id`; error → show message; reset input).
5. `templates/workbench.html` export-all link stays (content now v2 docs).
6. Tests: new `tests/test_workbench_export_import.py` (preamble copied from `test_chat_history.py`):
   - Unit (no DB): serializer shape (version 2, no user id/keys/request ids, runs fields, compare snapshots); parser — v2 parse + clamp (top_p 5→1.0, max_tokens 999999→4000, seed bool→None), whitelist (extra fields dropped), v1 sniff (recorded_incomplete True, extended NULL), bad shape, cap 2000.
   - Route: export v2 shape + ownership (user B 404 on A's session); export-all ndjson; import round-trip export→import→export equality **except `session.id`**; fresh ids; cross-user interchange; 401/400 bad json/400 bad shape/413; clamp-on-import observable in the re-export; v1 import marker; whitelisted-field enforcement (crafted `user_id`/`session_id`/`id` keys ignored).
7. Update `tests/test_chat_history.py`: delete `test_session_to_jsonl_record_shape` (pins the deleted v1 serializer); rewrite `test_chat_export_jsonl_includes_snapshots` → v2 `export.json`; update `test_chat_export_all_returns_only_own_sessions` (titles now at `doc["session"]["title"]`).
8. Docs: add a short "Export & import" + settings-persistence section to `base_model_wrapper/src/wrapper/docs/workbench.md` (tutorial tests pin slugs/copy anchors only — additive content safe). Use glossary terms (Run, Export document, Import, Restore chain).

---

## 6. Environment & commands

- Dev DB already up: docker container `acs-hiring-devdb` (postgres:16, pg/pg, host port 5434). Shared database: rows accumulate across runs.
- Migrations (env-var form only):
  ```bash
  cd base_model_wrapper
  DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
    uv run --python 3.12 --extra dev alembic upgrade head
  ```
- Tests (whole files, never node ids; `TEST_DATABASE_URL` mandatory):
  ```bash
  cd base_model_wrapper
  TEST_DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
    uv run --python 3.12 --extra dev pytest tests/test_workbench_generations.py -q
  ```
  Full suite ~3–5 min. DB must be migrated first (§3.1 re-apply). A trailing `RuntimeError: No active exception to reraise` after the summary is a dependency's shutdown handler — exit code is what counts.
- Lint: `uv run --python 3.12 --extra dev ruff check .` (line-length 100). NEVER `ruff format .`; format only touched files.
- `wrapper.main` import needs Settings env (`DATABASE_URL`, `MODAL_BASE_URL`, `VLLM_API_KEY`, `ADMIN_TOKEN`) — unit tests set them at module top / fixtures; bare imports fail (conftest note ACS-309).
- basedpyright noise on alembic files (`Import "sqlalchemy" could not be resolved` etc.) is spurious — all sibling migrations show it; ruff is the gate.

---

## 7. Process gotchas (this session's scars)

- The line-anchored edit tool corrupted adjacent regions four times (docstring close-quote, `upstream_status`/`upstream_error_kind`/`upstream_error_msg` initializers, the ChatSnapshot insert block, the sampling-panel comment/details boundary). **After every multi-line edit: re-read the whole edited region end-to-end, verify balanced tags/brackets, and `ast.parse`/render before moving on.** Anchors go stale within 2-3 edits; prefer content-anchored greps over remembered line numbers.
- The test DB is shared and long-lived: rows persist across pytest runs; helpers that query "latest row" globally (like `_wait_for_terminal_row`) are ordering-sensitive.
- House conventions to keep: comments cite ticket/issue IDs (`ACS-###`, here issue #12); comments explain "why"; snake_case; per-file test fixtures; `from wrapper.main import X` inside test functions.
