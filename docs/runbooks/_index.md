---
title: Runbooks
status: current
updated: 2026-07-10
owner: platform@example.org
---

# runbooks/ — how to operate it

Operational how-to: smoke, triage, stress. Living procedures.

- [`error-troubleshooting.md`](error-troubleshooting.md) — triage a 4xx/5xx spike on the API; pairs with the Metabase dashboard and Sentry.
- [`activation-go-live.md`](activation-go-live.md) — bringing a model's activation/harvest apps live: deploy order, version guards, validation.
- [`stress-test.md`](stress-test.md) — how to run the pre-beta stress suites (`s4_param_matrix`, `fuzz_api`).
- [`metabase-dashboard-queries.sql`](metabase-dashboard-queries.sql) — the dashboard tile queries for Metabase over the wrapper Postgres (ACS-38), grouped into 3 pages: **A. Operations**, **B. Usage**, **C. Users**.
