# Usage report aggregates the request log; rollups keep no model dimension

Issue #2 asks for a downloadable monthly token-usage report per model, but the usage rollups are keyed by (key, period) with no model column — per-model data lives only in the request log. We decided to build the usage report by aggregating the request log over the usage month instead of extending the rollups with a model dimension: a hot-path schema change (new rollup dimension, usage-commit changes across every call site, backfill) is disproportionate to a P1 human-facing report, and request-log retention is intentionally unpruned for now.

## Consequences

- The report depends on request-log retention: if pruning ever ships, a model-dimension rollup must exist first, or the report loses history (tracked as a follow-up issue).
- Deleting an API key cascades away its request-log and rollup rows, so a previously downloaded month can silently shrink on re-download. Known limitation; revisit only if it bites.