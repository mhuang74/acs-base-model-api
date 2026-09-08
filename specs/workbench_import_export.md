# Spec: Workbench Run persistence, Restore chain, and v2 Session export/import

> Published as [<owner>/acs-base-model-api#12](https://github.com/<owner>/acs-base-model-api/issues/12); supersedes the raw complaint in #6.

Resolves #6. Vocabulary per root `CONTEXT.md` (Session, Run, Sampling settings, Draft, Compare lane, Compare snapshot, Reproducible Run, Export document, Import, Restore chain); decisions per `docs/adr/0001-best-effort-reproduction-and-v2-session-format.md`.

## Problem Statement

Researchers lose work and cannot reproduce results on the Workbench. When they reload the page, every Sampling setting outside temperature and max tokens resets to defaults — they must remember and re-enter top_p, top_k, min_p, the three penalties, seed, stop sequences, and the logprobs toggle for every single Run. The existing session export omits nearly all of this: it carries only max tokens and temperature per Run, so a file given to a colleague (or kept as an archive) describes nothing reproducible. And even if a file were complete, there is no way to bring it back in — the import capability that Loom already has does not exist for Workbench Sessions.

## Solution

Every Run records the full Sampling settings that produced it, along with both model identities (the requested model short id and the model string the upstream echoed back). On page load the composer restores settings through the Restore chain: the URL Draft if present, else the Session's most recent successful Run, else built-in defaults. Compare panel lanes prefill from the most recent Compare snapshot. Sessions export as versioned v2 Export documents containing Runs with full Sampling settings, prompts, completions, timestamps, and Compare snapshots as records; any authenticated user can Import such a document as a new Session owned by them, with settings clamped to this deployment's parameter ranges. Reproduction is best-effort exact: greedy Runs (temperature 0) must match; sampled Runs attempt to match using the recorded seed; Runs without a recorded seed — and legacy Runs marked `recorded_incomplete` — are honestly labeled non-reproducible while their settings still restore.

Delivered in two phases: Phase 1 = Run persistence + Restore chain (fixes settings loss); Phase 2 = v2 export/import (fixes reproduction and interchange).

## User Stories

**Persistence and restore (Phase 1)**

1. As a researcher, I want every Run to record the full Sampling settings used (temperature, max tokens, top_p, top_k, min_p, presence/frequency/repetition penalties, seed, stop sequences, logprobs toggle and count), so that each completion's recipe is known forever.
2. As a researcher, I want the composer to restore from the URL Draft on reload, so that settings I've edited but not yet run with survive an accidental refresh.
3. As a researcher, I want the Draft to update as I change each control, so that I never have to remember to "save" my settings.
4. As a researcher, I want the Draft to survive Run submission (the redirect re-encodes the submitted settings), so that running an experiment doesn't reset the panel.
5. As a researcher, I want the composer to fall back to my Session's most recent successful Run's Sampling settings when the URL carries no Draft, so that returning to a Session restores what I last ran.
6. As a researcher, I want brand-new Sessions to start from sensible built-in defaults, so that a fresh Session is never broken by stale values.
7. As a researcher, I want cancelled Runs to be excluded from the last-successful-Run fallback, so that abandoning a bad attempt doesn't change my defaults.
8. As a researcher, I want the compare panel's lanes prefilled from the most recent Compare snapshot, so that a compare setup survives reload.
9. As a researcher, I want model and API key choice restored exactly as today, so that Session continuity is preserved.
10. As a researcher, I want pre-migration Runs displayed with defaults-with-placeholder for unset parameters, so that legacy history remains readable and usable.
11. As a researcher, I want pre-migration Runs visibly marked `recorded_incomplete`, so that I know their extended settings were never captured.
12. As a researcher, I want Runs with an explicitly set seed marked Reproducible, so that I know which sampled Runs I can attempt to reproduce.
13. As a researcher, I want blank-seed Runs marked non-reproducible, so that I'm never misled into expecting identical output from them.
14. As a researcher, I want temperature-0 Runs marked Reproducible regardless of seed, so that greedy experiments are clearly identified.

**Export (Phase 2)**

15. As a researcher, I want to export a Session as a single v2 Export document containing all Runs with full Sampling settings, prompts, completions, timestamps, and cancel flags, so that I can archive or hand off a complete transcript.
16. As a researcher, I want the export to carry both the requested model short id and the upstream-echoed model string per Run, so that the receiving side knows which model actually served.
17. As a researcher, I want Compare snapshots included in the export as historical records (lane configs and per-lane completions), so that compare experiments travel with their groupings intact.
18. As a researcher, I want export-all to remain available as ndjson of v2 documents, so that I can bulk-archive my work.
19. As a researcher, I want the export format versioned, so that future format evolution stays parseable.
20. As a researcher, I want exports to contain no user identity, API keys, or request ids, so that sharing a file leaks nothing sensitive.

**Import (Phase 2)**

21. As a researcher, I want to import an Export document as a new Session owned by me, so that I can receive and continue a colleague's experiment.
22. As any authenticated user, I want to import any valid Export document regardless of who exported it, so that Sessions genuinely interchange between accounts.
23. As a researcher, I want imported Sampling settings clamped to this deployment's parameter ranges, so that re-running an imported Run sends exactly what this deployment would send.
24. As a researcher, I want legacy v1 exports to import successfully with defaulted extended parameters and the `recorded_incomplete` marker, so that previously exported archives remain useful and honestly labeled.
25. As a researcher, I want imported Sessions and Runs to receive fresh identities, so that imported data can never collide with or overwrite my existing Sessions.
26. As a researcher, I want the importer to accept exactly one Export document per call, so that bulk ndjson files don't silently duplicate Sessions.
27. As a researcher, I want imports size-capped (2000 Runs per Session, mirroring the Loom limit), so that pathological files can't overwhelm the server.
28. As a researcher, I want the importer to accept only whitelisted fields, so that crafted payloads can't inject unexpected state.
29. As a researcher, I want the import response to give me the new Session's id, so that I can navigate straight to it.

## Implementation Decisions

**Phasing.** Two phases on one data model. Phase 1: Run persistence + Restore chain + compare prefill. Phase 2: v2 Export documents + Import. Export/import builds on the Phase 1 Run record; the settings fix ships first because it is the higher-frequency complaint.

**Run record.** The existing chat snapshot table gains additive columns for the extended Sampling settings: top_p, top_k, min_p, presence_penalty, frequency_penalty, repetition_penalty, seed, stop, and the logprobs toggle/count. NULL means unset (historical rows keep NULLs; no backfill — backfilled defaults would lie about what actually ran). Compare lanes route through the same Run record (the compare-lane path already persists full configs today; it consolidates onto the unified record), while the Compare snapshot table remains the lane grouping. The generation/compare-generation tables' role narrows to in-flight tracking and lane grouping as appropriate. Table names in the database do not change; the glossary term for all of these records is Run.

**Completeness marker.** Each Run carries a completeness flag distinguishing fully-recorded Runs from pre-migration rows and v1-imported Runs (`recorded_incomplete`). The marker surfaces in the UI and travels in exports.

**Seed policy.** Seed is recorded only when the user explicitly sets one. Blank seed keeps today's meaning (upstream draws randomly; effective seed unknown — vLLM never echoes it back). No auto-minting. A Run is Reproducible iff it has a recorded seed or temperature 0. This aligns with the proxy's existing seed→deterministic retry assumption.

**Model identity.** Each Run records both the requested model short id and the model string echoed by the upstream completion response. Model strings import verbatim (no registry validation), matching the Loom precedent: identity travels as data, so documents from deployments with different model ids still import.

**Restore chain.** On page load, composer settings resolve: URL Draft → the Session's most recent successful Run's Sampling settings → built-in defaults. The Draft covers the Sampling settings panel only; model and API key keep their existing session-row restore. The last-successful-Run fallback excludes cancelled Runs (they commit records — existing behavior — but don't feed the fallback) and naturally excludes errored generations (they never commit).

**Draft mechanics.** The Draft is the URL-encoded Sampling settings panel, rewritten via `history.replaceState` on every panel change (no history spam), and re-encoded by the post-Submit redirect (which carries the client-clamped values the user sees). Draft values are intent; the Run records the server-clamped recipe actually sent upstream — the two can differ slightly by design, and the Run is authoritative for Reproduction.

**Compare prefill.** On page load the compare panel's lanes prefill from the most recent Compare snapshot of the Session (falls back to defaults when none exists). Ships in Phase 1 — lane configs are persisted by the same migration work, so prefill is pure render logic.

**v2 Export document.** Per-Session download, JSON attachment. Shape: `version: 2`; `session{}` (id, title, model, created/updated timestamps, prompt); `runs[]` — one entry per committed Run in order: timestamps, prompt prefix, completion text, cancel flag, model short id, echoed model string, the full Sampling settings (NULL/unset where unknown), completeness marker; `compare_snapshots[]` — grouped lanes with per-lane Sampling settings and completions, as historical records. No user identity, API keys, or request ids anywhere in the document. Cancelled Runs travel as records; errored generations don't exist to export. Export-all stays ndjson of v2 documents. The prompt-only txt export is unchanged.

**v1 compatibility.** The importer shape-sniffs: a document without v2 structure is treated as a v1 record — Runs import with defaulted extended parameters and the `recorded_incomplete` marker. (Loom writes a version field its importer ignores; we deliberately validate ours.)

**Import endpoint.** `POST /workbench/import`, raw JSON body, cookie-authed. Error contract mirrors Loom's importer: 401 unauthenticated, 400 bad JSON / bad shape, 413 over the 2000-Run cap. Mechanics mirror Loom's: fresh UUIDs for the Session and every Run (incoming ids never trusted), whitelisted fields only, imported Session owned by the importing user. Success returns the new Session id. Semantics: copy-in, new Session — no merge, no overwrite. Single document per call; export-all ndjson files are not importable. Interchange policy: any authenticated user may import any valid document; no ownership enforcement on the file's origin.

**Clamp-on-import.** Imported Sampling settings are clamped to this deployment's parameter ranges at import time. This preserves rather than breaks the reproduction contract: every historical Run's recorded recipe was already clamped by the request builder before the upstream saw it, so the clamped values are the true recipe of the recorded outcome.

**Naming.** Documentation, tests, and the export format use the glossary term Run. Database table names are unchanged.

**Schema change.** One additive migration for the extended Run columns plus the completeness marker. No data backfill.

**Explicit non-goals.** Public `/v1/completions` traffic is untouched (no Run persistence for API calls; existing request telemetry unchanged). Loom's format and importer are untouched (including its ignored version field). The prompt-only txt export is untouched. No re-run-on-import batch action; import restores state only. No merge/overwrite import semantics.

## Testing Decisions

**What makes a good test here.** Tests assert external behavior a consumer observes — HTTP status codes, response bodies, document shapes, and DB-observable outcomes after a request — never internal call graphs, module globals, or incidental defaults. The importer, for example, is tested by what it returns and what a subsequent export/page render shows, not by which helper it called.

**Primary seam: the cookie-authed route surface (existing).** TestClient route tests against four surfaces:

- Generation endpoint → Run records the full Sampling settings (including seed only-when-set and server-side clamping); cancelled Runs commit with their flag and don't feed the last-successful fallback; compare-lane Runs carry their lane config through the unified record.
- Session page render → Restore chain prefill: bare load prefills the most recent successful Run's settings; `recorded_incomplete` rows render defaults-with-placeholder; cancelled-only Sessions fall through to defaults.
- Export endpoints → v2 document shape (version, session, runs with full Sampling settings, both model identities, compare snapshots as records), ownership scoping, export-all as ndjson of v2 documents.
- Import endpoint → the full contract: export→import→export round-trip equality; fresh identity minting; whitelisted-field enforcement; clamp-on-import; v1 shape-sniffing producing `recorded_incomplete` Runs; the 2000-Run cap (413); cross-user interchange; ownership of the imported Session.

**Secondary seam: pure-function unit tests (existing pattern).** The v2 document serializer and the import parser/validator are unit-tested directly — shape, whitelisting, clamping, v1 detection — the same way the current JSONL record serializer has a focused unit test.

**Prior art.** The chat-history test file (route-level export tests plus a serializer unit test) and the Loom test file (export/import round-trip, ownership enforcement, size caps, topology preservation) are the templates: same per-file fixture preamble (test database URL + skip-if marker + client fixture), same ownership assertions, same round-trip style.

**Not seamed.** The URL Draft is pure client-side JS in a server-rendered, no-JS-build repo; there is no browser test infrastructure. Draft behavior is verified manually in the browser; every server-side leg of the Restore chain is covered at the primary seam. Tests run per whole file, never by node id, per repo convention.

## Out of Scope

- Persisting Runs for public-API `/v1/completions` traffic; any change to existing request telemetry tables.
- Loom's export/import format, including fixing its version-field-ignoring importer.
- The prompt-only txt export (kept as-is).
- Bit-exact reproduction of sampled Runs across upstream deployments; the guarantee is greedy-only.
- Seed auto-minting; requiring a seed to submit.
- Merge/overwrite import semantics; importable export-all ndjson; re-run-on-import batch actions.
- Browser/E2E test infrastructure for the Draft.

## Further Notes

- Root cause note: the two complaints share one fix. Persisting full Sampling settings per Run makes reload-restore and complete exports the same data story — the Restore chain reads Runs, exports serialize Runs.
- Clamp-on-import consistency: the request builder clamps every Run's parameters before the upstream sees them, so the clamped value is what the recorded outcome actually came from; imported documents are clamped for the same reason.
- The v1-tolerant importer is a deliberate improvement over the Loom precedent, which writes a version field and then ignores it — ours validates the version/shape and maps legacy documents to `recorded_incomplete` Runs.
- The upstream never echoes the effective seed back; this is the hard floor of the reproduction contract and the reason Runs without an explicit seed are marked non-reproducible rather than silently trusted.
- Glossary and ADR produced alongside this spec: `CONTEXT.md` (Session, Run, Sampling settings, Draft, Compare lane/snapshot, Reproducible Run, Export document, Import, Restore chain) and `docs/adr/0001-best-effort-reproduction-and-v2-session-format.md`.