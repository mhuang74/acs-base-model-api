# SOLUTION NOTES

Repo: https://github.com/mhuang74/acs-base-model-api — browse Issues, PRs, and contextual files referenced below.

## Compare lanes run on the prompt baseline (resolves #5)

**Design.** Root cause: the session prompt is a rolling continuation buffer (server appends each completion); both panes render it, so Compare ran on prompt + completion. Targeted fix, no schema changes: the boundary is recovered from the latest *roll-forward snapshot*, not the latest generation (failed/cancelled runs don't write). Render (`routes/workbench.py::_render_chat`): Compare renders the snapshot's `prompt_before` only when the session prompt exactly equals baseline + completion; otherwise as-is — restores, edits, fresh typing, write-back carry untouched. Toggles read `<script id="cmp-roll-forward">` `{baseline, completion}`.

**Tests.** `tests/test_workbench_compare_render.py` (DB-gated): regression failing pre-fix; no-generations; cancelled-with-partial; restore-older-snapshot (pins equality over latest-generation anchoring); post-run edit. Toggles: manual only — no JS test infra.

**Residuals.** Root refactor out of scope; append-without-run carries whole; cancelled-with-partial leaves a stale in-page boundary until refresh.

## Monthly token usage report per model (implements #2)

**Design.** CSV on /usage: month dropdown → `usage-report-YYYY-MM.csv`, UTF-8 BOM. Cookie auth, own usage only. One row per (model, key) across all keys; `?key=` filter ignored. Header: `key_name,key_prefix,model,prompt_tokens,completion_tokens,total_tokens,requests`; grouped by key, models by tokens desc; NULL model → `(unknown)`. UTC months; default previous; invalid falls back. Tokenized successes only — ties out to /usage page. Source: request log aggregated by (model, key_id), not rollups — ADR 0001. No migration.

**Tests.** Service + route seam (redirect, headers, CSV bytes), DB-gated; prior art: /usage tab test file.

**Follow-ups.** #7 model-dimension rollup + backfill; #8 admin export; key deletion shrinks past months — documented limitation (ADR 0001).

## Workbench run persistence + v2 export/import (#6) — NOT finished

**Status.** Open P1; spec'd as issue #12 (`ready-for-agent`). Unfinished work is detailed in `specs/handover-issue-12-followup.md` — all three prior defects are fixed and verified, but Phase 1 test coverage and all of Phase 2 remain. Today's exports (.txt, v1 `.jsonl`, export-all) record only temperature/max tokens; extended settings reset on reload; no import exists; the spec's ADR is unwritten.

**Design (per #12).** One data model, two phases. Phase 1 — Run persistence: additive columns for full sampling settings plus a `recorded_incomplete` marker; NULL = unset, no backfill. Restore chain on load: URL Draft → most recent successful Run → defaults (cancelled/errored excluded); Compare lanes prefill from latest Compare snapshot. Phase 2 — versioned v2 export document (runs with full settings, both model identities, compare snapshots as records, no identity/keys/request ids) plus `POST /workbench/import`: cookie-authed, fresh UUIDs, whitelisted fields, clamp to ranges, v1 shape-sniffing → `recorded_incomplete`, 2000-Run cap.

**Still missing.** The entire build: Phase 1 tests, compare prefill, v2 serializer, import endpoint — plus the handover doc and ADR.
