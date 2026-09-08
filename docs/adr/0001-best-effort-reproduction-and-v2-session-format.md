# Best-effort-exact reproduction: Run persistence, seed policy, and v2 export/import format

Researchers could not reproduce Workbench runs: extended sampling settings (top_p, top_k, min_p, penalties, seed, stop, logprobs toggle) lived only in uncontrolled form inputs and reset on reload, and exports omitted them entirely. We decided to (1) persist full sampling settings per Run (extended `chat_snapshots` rows; compare lanes already persist theirs), (2) restore composer settings via the Restore chain — URL Draft, then the Session's most recent successful Run, then defaults, and (3) export Sessions as versioned v2 JSON documents carrying the full Sampling settings plus both requested and upstream-echoed model identity, with an importer that mirrors loom's (fresh UUIDs, whitelisted fields, size cap) and additionally clamps imported settings to this deployment's parameter ranges.

The reproduction promise is best-effort exact: greedy decoding (temperature 0) must match; sampled Runs attempt to match using a recorded seed. Seed is recorded only when the user explicitly sets one — a blank seed means the upstream drew randomly and the effective seed is unknown (vLLM never echoes it back), so those Runs are non-reproducible by contract, as are all pre-migration/v1-legacy Runs (marked `recorded_incomplete`); their settings still restore. Imported settings are clamped on the way in because the recorded recipe of any historical Run is already the clamped value (`_build_lane_body`), so clamping preserves — not distorts — the reproduction contract.

## Considered Options

- **URL-only settings persistence**: rejected as the sole mechanism — it misses the `/workbench` redirect path and resets the moment a link is reshared without params; kept as the Draft layer on top of Run persistence.
- **New unified runs table**: rejected — cleanest model but rewrote render/revert/export and migrated three live tables for no behavioral gain over extending `chat_snapshots`.
- **Seed auto-minting when blank**: rejected — silently changes what the user ran today; "record only when set" keeps blank seed meaning random while honestly marking such Runs non-reproducible.
- **v2-only import**: rejected — breaks every previously exported file; the v1-tolerant importer shape-sniffs instead (loom writes a version field its importer ignores; we deliberately do validate ours).
- **Store imported out-of-range params raw**: rejected — clamping at import time matches what the deployment would actually send.

## Consequences

- Only non-cancelled committed Runs update the last-successful-Run fallback; cancelled Runs travel in exports as historical records.
- `POST /workbench/import` accepts exactly one Export document; `export-all.jsonl` ndjson files are not importable.
- Public `/v1/completions` traffic, loom's export/import, and the prompt-only `export.txt` are explicitly out of scope.