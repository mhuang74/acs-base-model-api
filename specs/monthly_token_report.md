# Spec: Monthly Token Usage Report (usage per model)

Published as [issue #10](https://github.com/mhuang74/acs-base-model-api/issues/10) — implementation spec for [#2](https://github.com/mhuang74/acs-base-model-api/issues/2). Labels: `ready-for-agent`.

## Problem Statement

A researcher on the platform is writing a grant report and needs last month's token usage broken out per model, in a form they can download and hand to funders. Today the /usage page shows only live, this-month numbers on screen — there is no way to get a prior month, no per-model breakdown, and nothing downloadable.

## Solution

A **usage report**: a CSV download offered on the /usage page. The user picks a month from a dropdown (months that have usage, defaulting to the previous month) and downloads `usage-report-YYYY-MM.csv`. Each row is one (model, key) pair for the chosen usage month, covering all the user's keys — including revoked ones, so the numbers tie out to how budgets are accounted. Users see only their own usage; there is no admin surface in this feature.

## User Stories

1. As a researcher on the platform, I want to download last month's token usage as a CSV, so that I can file my grant report without transcribing numbers from a web page.
2. As a researcher, I want the report broken out per model, so that I can attribute usage to the specific base models named in my proposal.
3. As a researcher, I want prompt and completion tokens as separate columns, so that I can report input vs output usage the way grant forms ask for it.
4. As a researcher, I want a total-tokens column per row, so that I can eyeball relative model usage without summing two columns.
5. As a researcher, I want a request count per row, so that I can distinguish "expensive few" from "cheap many" usage patterns.
6. As a researcher, I want the report to include usage from keys I revoked during the month, so that my totals match what my budget was charged.
7. As a researcher, I want each row to name the API key it belongs to (name and prefix), so that I can attribute usage to projects sharing the account.
8. As a researcher, I want the file to open directly in Excel/Sheets, so that I don't fight encoding problems during a deadline.
9. As a researcher, I want the download to default to the previous month, so that "last month's report" is one click.
10. As a researcher, I want to pick any earlier month that has usage, so that I can backfill reports for months I missed.
11. As a researcher, I want months with no successful usage excluded from the picker, so that I never download an empty report by mistake.
12. As a researcher, I want rows grouped by key with models ordered by usage within each key, so that the file reads like a per-project breakdown.
13. As a researcher, I want a row labeled (unknown) for usage with no recorded model, so that the file's totals always reconcile with the /usage page.
14. As a researcher, I want the file to omit a grand-total row, so that spreadsheet pivot tables and filters work unimpeded.
15. As a key-owning user, I want the download to always cover all my keys, so that a report can never be silently scoped down by whatever key filter I had selected on the page.
16. As a key-owning user, I want the month picker and download control placed near the page's filter controls, so that the feature feels like part of the usage page rather than a hidden URL.
17. As a logged-out visitor, I want the download to require login like the rest of the page, so that usage numbers are not exposed without authentication.
18. As a user, I want the CSV to count only successful tokenized requests, so that its totals tie out to the monthly numbers the /usage page shows me.
19. As an administrator, I want the permission policy to leave room for admin exports later, so that supporting a user's report request doesn't require redesign.
20. As the platform operator, I want the report to read from the request log rather than mutate the rollup schema, so that the token-accounting hot path stays untouched.
21. As the platform operator, I want the report's dependence on request-log retention recorded in an ADR, so that a future pruning change cannot silently break historical reporting.
22. As the platform operator, I want the durable per-model rollup tracked as a separate issue, so that the retention gap has a planned fix.
23. As the platform operator, I want key deletion's effect on historical reports documented, so that support conversations about shrunk past months have a written answer.
24. As a developer, I want the report built as a service function returning a plain data structure, so that the route stays thin and future admin exports can reuse it.

## Implementation Decisions

- **Surface**: a download control (month dropdown + button) rendered on the /usage page, next to the existing key filter. It submits a GET to a new route on the usage router; the response is a CSV attachment named `usage-report-YYYY-MM.csv` with `Content-Disposition: attachment`. CSV is UTF-8 with a UTF-8 BOM so Excel/Sheets open it correctly. Follows the repo's existing attachment-download idiom.
- **Auth**: the route requires a logged-in user via the same cookie-session dependency as the /usage page; anonymous requests redirect to login. Users can only ever download their own usage; no admin route or admin bypass in this feature.
- **Scope of rows**: every API key owned by the requesting user, including revoked and disabled keys, so report totals match budget accounting (which counts revoked keys' usage). The download ignores the page's `?key=` filter by design — the file always covers all the user's keys.
- **Row grain and layout**: one row per (model, key). Header: `key_name,key_prefix,model,prompt_tokens,completion_tokens,total_tokens,requests`. Rows grouped by key, keys ordered by key creation date (oldest first — same order the /usage page's per-key table uses), models within a key ordered by total tokens descending. Rows with NULL model render as `(unknown)`. No grand-total row.
- **Month semantics**: a usage month is a UTC calendar month, consistent with how the rollups key `period_start`. The month picker offers the distinct months present in the user's usage rollups ∪ the current (partial) month, defaulting to the previous month. An invalid/unknown month parameter falls back to the default rather than erroring.
- **Row semantics**: counts are tokenized successes only — a request contributes to the report only when it has a positive prompt/completion token count. This matches how the usage rollups are written, so CSV column totals equal the monthly totals the /usage page displays for the same month (except the rollups lack the model dimension). Request count per row = number of contributing requests for that (model, key) pair. `total_tokens = prompt_tokens + completion_tokens`.
- **Data source**: aggregate the request log (`GROUP BY model, key_id` over the chosen UTC month, summing prompt/completion token columns) — NOT the usage rollups, which have no model column. This is the decision recorded in ADR 0001; its consequences (retention dependency; durable rollup fix tracked in #7) apply. Harvest submits log token-less rows by design, so harvest token usage does not appear in this report — consistent with the rollups.
- **Module layout**: a new service function (in the usage-reports service area) takes (user, month) and returns a plain row structure (rows, offered months, chosen month); the usage route renders it to CSV text; the /usage template gains the picker+button wired to the route. The route stays thin (auth + call service + CSV response).
- **No schema changes, no new tables, no migration.** All data read from the request log and the user's key list.

## Testing Decisions

- **What makes a good test here**: assert observable contract — request the CSV route as a logged-in user and assert response content-type/disposition, filename, and parsed CSV rows; assert the month picker's offered months and default via the service's return structure. Do not assert SQL text, internal call ordering, or template markup beyond what a user observes.
- **Modules tested**: the report service (row grain, grouping/order, (unknown) bucket, month list, default month, revoked-key inclusion, month-window boundaries) and the route (auth redirect, attachment headers, CSV content for a known fixture month).
- **Prior art**: the DB-gated test file covering the /usage tab and `build_usage_context` (per-key filter scoping, zero-state rendering) is the direct template — same client fixture style, same login helpers, same direct service-level assertions. The new tests extend that file or clone its preamble. DB-gated tests follow the repo convention: `TEST_DATABASE_URL` + per-file `dbtest` skipif marker.
- **Seam (highest available, no new seams needed)**: test at the service function (user, month → rows/months) for report logic, and at the HTTP route (TestClient GET → CSV bytes/headers) for the user-visible contract. Both seams already exist in the codebase — the service seam mirrors the existing direct-call tests; the route seam mirrors existing /usage TestClient tests. Zero new seams.

## Out of Scope

- Any admin-facing export (tracked as #8), platform-wide reports, and any other-user data access.
- A bearer-token API endpoint for reports.
- JSON or other non-CSV formats; grand-total rows; date-range picking beyond a single month.
- Extending usage rollups with a model dimension / backfill (tracked as #7).
- Changing key-deletion semantics (documented as a known limitation in ADR 0001).
- Cost/pricing columns — no estimated cost is persisted anywhere today.
- Changes to the /usage page's existing sections (totals, budget bars, daily chart, per-key table).

## Further Notes

- Vocabulary per CONTEXT.md: *usage month* (UTC calendar month), *usage rollup*, *request log*, *usage report* (avoid "export"/"statement").
- ADR 0001 (docs/adr/0001-usage-report-aggregates-request-log.md) is the decision record for the data source; its consequences bind this feature: the report depends on request-log retention staying unpruned, and key deletion cascades away historical rows.
- Follow-up issues: #7 (model-dimension rollup + backfill), #8 (admin export). Design interview record: the #2 comment.
- The month dropdown defaults to the previous month even when the previous month has no rollup rows (a researcher asking on the 2nd still gets one click); downloading it yields a header-only CSV.