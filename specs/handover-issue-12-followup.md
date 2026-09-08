# Handover 2 — Issue #12: Run persistence (follow-up for the next agent)

Continues `specs/handover-issue-12-workbench-run-persistence.md` — read it FIRST:
its §2 decisions are LOCKED, §5 lists the remaining build order, §6 the commands.
This file records the delta since that handover: all three §3 defects are fixed
and verified; Phase 1 test coverage + all of Phase 2 remain.

## 1. Status after session 2

### FIXED & VERIFIED (was §3 in the previous handover)

1. **Migration 0046** (`base_model_wrapper/alembic/versions/0046_run_sampling_settings.py`):
   - The file arrived corrupted (stray `)` after the `op.execute`, duplicated
     `model_echo` add_column). Repaired; `ast.parse` clean.
   - Docstring now states the completeness-marker backfill explicitly.
   - Re-applied: `alembic downgrade 0045` → `alembic upgrade head` (env-var form,
     `DATABASE_URL=postgresql://pg:pg@localhost:5434/acs`).
   - Verified in the dev DB: `alembic current` = `0046_run_sampling_settings (head)`;
     `SELECT count(*), count(*) FILTER (WHERE recorded_incomplete) FROM chat_snapshots`
     → `106|106` (every pre-existing row marked, settings columns stay NULL).
2. **§3.2 logprobs_count normalization**: `workbench_generations.py:667` now reads
   `logprobs_count=(body.get("logprobs") or None)`.
3. **§3.3 test isolation**: `tests/test_workbench_generations.py` —
   `_wait_for_terminal_row(gen_uuid, key_id)` (~:1430) now scopes the ApiRequest
   query with `ApiRequest.key_id == key_id` alongside the `/workbench` endpoint
   filter; `key_id` is plumbed from `_make_chat_and_key` through
   `_start_generation_and_subscribe`, which now returns a 4-tuple
   `(gen_uuid, drain_task, captured, key_id)` (~:1427); all four error-threading
   tests updated to pass it.
   - Verification: `pytest tests/test_chat_history.py tests/test_workbench_generations.py -q`
     → **68 passed** (run twice after the fix; the combined pair previously
     flaked with `assert 'upstream_4xx' == 'upstream_unreachable'`).
   - The product error-threading code was NOT touched — it was verified correct.

### Not yet done
- Full suite (`pytest` all files) not run this session; lint status of the three
  touched files is in the PR description / last session notes.
- Everything in the previous handover §5 except §3 (defects): Phase 1 leftovers
  #2 and #3, and all of Phase 2. Details below with the research already done.

## 2. Next work — Phase 1 leftovers

### 2.1 Run persistence tests → extend `tests/test_workbench_generations.py`

Append after `test_snapshot_logprobs_null_when_logprobs_off` (~:2155). Facts
verified this session so the next agent does not re-derive them:

- `POST /workbench/{chat_id}/generations` accepts Form fields: `max_tokens`,
  `temperature`, `top_p` (1.0), `top_k` (-1), `min_p` (0.0),
  `presence_penalty` (0.0), `frequency_penalty` (0.0), `repetition_penalty`
  (1.0), `seed` (""), `stop` (""), `logprobs` (""), `model` ("") —
  `routes/workbench.py` ~:680-696; the body is shaped by `_build_lane_body`
  (~:775-788) so the snapshot records exactly what went on the wire.
- The ChatSnapshot write reads the CLAMPED body:
  `workbench_generations.py` ~:656-673. `top_p/top_k/min_p/penalties` from
  `body`; `seed=body.get("seed")` (only-when-set); `stop=body["stop"][0]` when
  the body carries a stop list; `logprobs_count=(body.get("logprobs") or None)`
  (:667); `recorded_incomplete=False` (:671); `model_echo=upstream_model_echo`
  (:673).
- Clamp ranges (`_build_lane_body` ~:1080-1130): top_p [0,1] default 1.0;
  min_p [0,1]; presence/frequency [-2,2]; repetition [0,2]; top_k int →
  (>=1 else -1); seed int only when non-empty; max_tokens cap 4000
  (`_CHAT_MAX_MAX_TOKENS` ~:64); temperature cap 100.
- Reusable helpers in the test file: `_patch_upstream_chunks` (:488),
  `_await_generation_done` (:2046), `_only_snapshot` (:2065), `_make_user`
  (:402), `_make_chat_and_key` (:416), `_login` (:462).

Tests to write (whole-file runs only, never node ids):

1. Full sampling panel POST (fake upstream chunks include a `"model"` echo,
   e.g. `data: {"model":"gpt2-served","choices":[{"text":"hi"}]}\n\n`) → assert
   `ChatSnapshot` carries the exact in-range values, `seed == 123`,
   `stop == "END"`, `logprobs_count == 5`, `model_echo == "gpt2-served"`,
   `recorded_incomplete is False`.
2. Clamping case: `top_p=5` → snapshot `top_p == 1.0`; `temperature=500` →
   `temperature == 100` (the Run records what the wire carried, not what was
   typed).
3. Only-when-set semantics: blank seed → `snap.seed is None`; `logprobs=0` →
   `snap.logprobs_count is None` (exercises the §3.2 `or None` branch).
4. Cancelled-with-text run commits a snapshot with `cancelled=True` (gated
   upstream + cancel-endpoint pattern exists ~:803). Per LOCKED decision 7 this
   snapshot must NOT feed the restore chain — that negative is covered by the
   `test_chat_history.py` cancelled-only case (2.2 case 2).

### 2.2 Restore-chain render tests → extend `tests/test_chat_history.py`

That file already has the client fixture + `_make_chat_session`. Facts:

- `chat_open` (`routes/workbench.py` ~:280-330) fetches `latest_restore_run`
  and `latest_compare_snapshot` and passes them into the template context as
  `composer_settings` / `compare_prefill_lanes` (~:316-320).
- Service helpers on disk, untested: `composer_settings_from_run`
  (`services/workbench.py` :281-303; seed → str or ""), `latest_restore_run`
  (:306-324; excludes cancelled, `ts desc, id desc`), `latest_compare_snapshot`
  (:327-339).
- Template anchors (re-grep before editing; anchors burn): `workbench.html`
  `{% set cs = composer_settings if composer_settings else {} %}` ~:1240,
  sampling inputs prefill `cs.X` with built-in defaults as fallback, logprobs
  toggle ~:1252, snapshot meta markers (`· incomplete record` /
  `· reproducible` / `· non-reproducible`) ~:1297; `workbench_compare.html`
  `cmp-config` JSON gains `prefillLanes`.
- Insert snapshots directly via DB — `ChatSnapshot` (`src/wrapper/models.py`)
  carries the 11 new columns; `recorded_incomplete` is NOT NULL default false,
  so pre-migration-style rows should be inserted with
  `recorded_incomplete=True` + NULL extended settings.

Cases:
1. Two snapshots inserted; newest successful Run's values prefill the composer
   (older one must not win).
2. Cancelled-only session → built-in defaults (max_tokens 200, temperature
   1.0, top_p 1.0, top_k -1, min_p 0.0, penalties 0.0/0.0/1.0).
3. `recorded_incomplete=True` + NULL extended → defaults render AND
   "incomplete record" marker visible.
4. `CompareSnapshot` inserted → rendered `cmp-config` JSON contains its lanes.

### 2.3 Compare lane logprobs passthrough (small, in-scope)

- `compare_start`'s `compare_config` dict (`routes/workbench.py` :1303-1313)
  gains `"logprobs": body.get("logprobs")`.
- `_sanitize_compare_lane` (~:1390-1445): READ the whole function first — the
  whitelist tuple is at ~:1403-1410 and the int coercion at ~:1434 currently
  covers `("max_tokens", "top_k", "seed")`; add `logprobs` to the whitelist
  with the bool-guard the previous handover specifies (bool is an int
  subclass), else None.
- `_lane_from_generation` (near :1445+) emits it.
- Effect: compare snapshot restore/prefill carries the heatmap toggle.

## 3. Phase 2 — unchanged from the previous handover §5

Nothing of Phase 2 has been started; the v1 jsonl surface (routes, `main.py`
compat entries, template link) is untouched on disk. Follow the previous
handover §5 build order 1-8 verbatim (session_exchange.py → routes → main.py
`__all__` → templates → new `tests/test_workbench_export_import.py` →
`test_chat_history.py` v2 rewrites → `workbench.md` docs), with document shape
and import rules from its §2.9-12.

## 4. Commands (previous handover §6 applies unchanged)

- Migrations: env-var form only (`DATABASE_URL=... alembic upgrade head`);
  `alembic -x url=` fails in this repo.
- Tests: whole files only; `TEST_DATABASE_URL` mandatory; dev DB at 5434 is
  shared and accumulates rows.
- Lint gate: `uv run --python 3.12 --extra dev ruff check .` (line 100);
  never `ruff format .`, format only touched files.

## 5. Scars to keep (both sessions)

- The line-anchored edit tool corrupts adjacent regions: the migration file
  arrived corrupted this session and one test-file docstring hunk triggered an
  auto-repair warning. After EVERY multi-line edit, re-read the edited region
  end-to-end and verify balanced tags/brackets.
- The dev DB is shared and long-lived: any helper querying "the latest row"
  globally is ordering-sensitive — the `_wait_for_terminal_row` key_id fix is
  the template for isolation fixes.
- House conventions: comments cite issue #12 / ACS-###; snake_case; per-file
  test fixtures; `from wrapper.main import X` inside test functions.
