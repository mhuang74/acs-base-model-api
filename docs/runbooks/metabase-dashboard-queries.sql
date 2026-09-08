-- ACS-38 — beta dashboard queries (Metabase over the Railway Postgres)
--
-- Each query below = one Metabase "SQL question". The queries are grouped to
-- mirror the dashboard's three pages:
--   A. OPERATIONS — reliability & health (requests, errors, latency, cold boot)
--   B. USAGE      — what's consumed (tokens, interactive/batch, GPU cost, feature mix)
--   C. USERS      — who's testing & how engaged (per-user summary, per-key detail)
-- All read-only over `api_requests` (+ joins to api_keys/users and the
-- signup-invite tables). No prompt/response bodies exist in this data —
-- metadata only (PR #23). Postgres dialect.
--
-- SLO definitions these implement live on Linear ACS-54. The "wrapper overhead"
-- query (A3) is also the basis for the wrapper-overhead alert.
--
-- ⓘ CARD DESCRIPTIONS (ACS-348): each tile carries a plain-English
-- `-- ⓘ DESCRIPTION (paste into Metabase card):` block — that text is the
-- source-of-truth for the card's Metabase Description (the little "ⓘ" tooltip).
-- Metabase descriptions live in the hosted UI, not in this repo, so changing the
-- text here is a documented hand-off: paste each block into the matching card.
-- Multi-series / gauge tiles also carry a one-line series/colour legend.
--
-- Setup: connect Metabase to the wrapper Postgres via the Railway PRIVATE host
-- (postgres.railway.internal) — same region as Postgres (US-West). A read-only
-- DB role is recommended but optional for beta.
--
-- Internal-account exclusion: the Users tiles (C1, C2) and the feature-adoption
-- tile (B8) exclude internal team accounts so the numbers reflect real testers,
-- not us. Membership is the `internal` USER TAG (ACS-374), not an email-domain
-- match: edit it in /admin/users like any other tag. That fixes the case the
-- old rule couldn't — a team member testing from a PERSONAL address — which
-- previously needed an uncommittable `NOT IN (...)` list maintained by hand in
-- the hosted Metabase UI. The tag was seeded from the old `@acsresearch.org`
-- rule (migration 0045), so the numbers did not move when this changed.
--
--   ⚠ A NEW @acsresearch.org signup is NOT tagged automatically. Nothing
--     applies the tag on signup/approve/invite — tag the account when you
--     approve it, or it counts as a real external tester in B8, C1 and C2.
--     (This is the cost of curated membership; the domain rule was automatic
--     but could not express the personal-address case. See ACS-375.)
--
--   ⚠ Hosted Metabase cards last re-pasted: NEVER — this file is the source of
--     truth but the cards are updated by hand (no Metabase API creds in the
--     repo). Until they are, the tiles still run the OLD domain-match SQL, so
--     editing the tag in /admin/users will not move them. Update this line
--     with the date when you re-paste.
--
-- The Operations tiles (A1–A7) and the volume tiles (B1–B7) intentionally keep
-- ALL traffic — internal load is still real load — but note that pre-beta
-- internal stress-testing inflates error/latency there until real beta traffic
-- dominates.
--
-- start_date parameter: every query carries an extra `AND <time> >= {{start_date}}`
-- floor (on top of its own rolling window) so the whole dashboard can be anchored
-- to a single cutoff from one control — e.g. the beta launch date. {{start_date}}
-- is a Metabase **plain Date variable** — in each question's Variables panel set
-- type=Date and a default (use 2026-06-15 for testing now; bump to the real launch
-- date when beta opens). To drive all cards from one knob: on the dashboard click
-- "Add a filter" → Time (Single Date), then connect it to each card's `start_date`
-- variable. A past default just makes the rolling window the binding constraint.
--
-- MODEL-ID NORMALIZATION (ACS-147): the model `trinity-base` was renamed to
-- `trinity-truebase` on 2026-06-26. Append-only rows written before the rename
-- (api_requests.model, gpu_cost_sample.model_id) keep the OLD id, so the
-- per-model tiles (A3b, A5, B2, B4, B5, and C1's models_used) fold the two together with
--   CASE WHEN <id> = 'trinity-base' THEN 'trinity-truebase' ELSE <id> END
-- so Trinity isn't split into two series across the rename boundary. Add a new
-- arm here if another model is ever renamed.


-- =========================================================================
-- A. OPERATIONS — reliability & health
-- =========================================================================

-- A1. Request volume + error rate over time (daily) ------------------------
-- "real_5xx" = server errors worth alarming on (upstream_unreachable / vllm_* /
-- unclassified upstream_5xx). cold_boot now usually appears as successful
-- held-open requests with cold_boot=true; late cold-start failures and
-- circuit_open (the breaker *protecting* a failing upstream — a derived symptom,
-- not a root cause) are split into their own columns so they don't inflate the
-- headline.
-- NOTE: truly unhandled wrapper 500s (error_kind would be 'internal_error') do
-- NOT appear here — the catch-all handler is Sentry-only and writes no
-- api_requests row. See error-troubleshooting.md for how to triage each kind.
-- VIZ: line/bar — X=day. Series legend: real_5xx=alarmable server errors ·
-- breaker_open_503=circuit-breaker protective 503s · cold_boot_503=cold-start
-- 503s · client_4xx=caller (4xx) errors · pct_real_5xx=% of traffic that was a
-- real server error.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Daily server-error health. The headline is pct_real_5xx: the % of requests
--   that failed with a real server error we'd alarm on — near 0 is healthy. The
--   other columns break out expected noise (breaker and cold-boot 503s) and
--   client mistakes (4xx), so they don't inflate the headline. Pre-beta numbers
--   are inflated by our own stress tests.
SELECT date_trunc('day', ts)                                           AS day,
       -- count(*)                                                     AS requests,  -- dropped on the live tile: it's error-focused (uncomment for total volume)
       count(*) FILTER (
         WHERE status >= 500
           AND coalesce(error_kind,'') NOT IN ('cold_boot','circuit_open')) AS real_5xx,
       count(*) FILTER (WHERE error_kind = 'circuit_open')             AS breaker_open_503,
       count(*) FILTER (WHERE error_kind = 'cold_boot')                AS cold_boot_503,
       count(*) FILTER (WHERE status >= 400 AND status < 500)          AS client_4xx,
       round(100.0 * count(*) FILTER (
         WHERE status >= 500
           AND coalesce(error_kind,'') NOT IN ('cold_boot','circuit_open'))
             / nullif(count(*), 0), 2)                                 AS pct_real_5xx
FROM api_requests
WHERE ts > now() - interval '14 days'
  AND ts >= {{start_date}}
GROUP BY 1 ORDER BY 1;

-- A2. Latency p50/p95 + TTFT p95 (warm only) -------------------------------
-- ⓘ DESCRIPTION (paste into Metabase card):
--   How fast warm requests are, per day. p50 is the typical request; p95 is the
--   slow tail (95% of requests finish faster than this). p95_ttft_ms is how long
--   streaming callers wait for the first token. Lower is better. Cold-boot
--   requests are excluded so a container spin-up doesn't look like slowness.
SELECT date_trunc('day', ts)                                          AS day,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms)        AS p50_total_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)        AS p95_total_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms)
         FILTER (WHERE ttft_ms IS NOT NULL)                           AS p95_ttft_ms
FROM api_requests
WHERE ts > now() - interval '14 days'
  AND ts >= {{start_date}}
  AND NOT cold_boot
GROUP BY 1 ORDER BY 1;

-- A3. Wrapper overhead p95 — PLAIN path (SLO ≤ 50ms; alert basis > 200ms) ---
-- latency_ms - upstream_latency_ms = auth + proxy framing (our controllable
-- part). Only meaningful for unary requests (streaming leaves upstream NULL).
-- SCOPED TO THE PLAIN PATH (ACS-325): `AND NOT activation` excludes
-- activation/steering requests. Their base64 steering-vector payload gets
-- serialized into vllm_xargs, so their wrapper overhead is structurally ~15× a
-- plain completion (measured over 6 prod days: plain p95 37ms/n=1165 vs
-- activation p95 556ms/n=11). Mixing them made the ≤50ms SLO read as "breached"
-- on low-traffic days that happened to be all steering (2026-08-05: 9 requests,
-- all steering → p95 557ms). Activation-path overhead now lives in A3b, tracked
-- but not held to this SLO.
-- SMALL-SAMPLE CAVEAT: p95 over fewer than ~20 requests in an hour is noise —
-- read the `n` column before trusting a spike.
-- UPSTREAM MEASUREMENT (ACS-325 "worth a look", verified 2026-08-10): the
-- activation path records upstream_latency_ms the SAME way as the plain path.
-- Both unary paths go through run_completion_nonstream (base_model_wrapper/src/
-- wrapper/routes/api.py:953) → proxymod.post_nonstream (called at :1053) and
-- store upstream_latency_ms=upstream_ms (:1165); only the upstream TARGET differs (activation
-- engine vs workbench engine). So overhead = latency_ms - upstream_latency_ms is
-- apples-to-apples: the steering overhead is real wrapper CPU (payload
-- serialization), not mismeasured upstream time.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   The wrapper's own added time on ordinary (non-activation) completions — our
--   auth + request-framing cost, on top of the model's time. Target ≤50ms at
--   p95; we alert above 200ms. Read the `n` column first: a p95 spike over fewer
--   than ~20 requests in an hour is noise, not a regression. Activation/steering
--   requests are excluded here — see A3b for those.
SELECT date_trunc('hour', ts)                                         AS hour,
       percentile_cont(0.95) WITHIN GROUP (
         ORDER BY latency_ms - upstream_latency_ms)                   AS p95_overhead_ms,
       count(*)                                                       AS n
FROM api_requests
WHERE ts > now() - interval '72 hours'
  AND ts >= {{start_date}}
  AND upstream_latency_ms IS NOT NULL
  AND NOT stream
  AND NOT cold_boot
  AND NOT activation                                                  -- plain path only (ACS-325)
GROUP BY 1 ORDER BY 1;

-- A3b. Activation-path overhead p50/p95 by model (ACS-325) -----------------
-- Same metric as A3 (latency_ms - upstream_latency_ms = OUR wrapper time), but
-- over activation/steering requests only (`AND activation`). Tracked to watch
-- the steering-payload serialization cost as activation traffic grows — NOT held
-- to the 50ms plain-path SLO. In practice this is STEERING overhead: capture
-- requests stream, which leaves upstream_latency_ms NULL, so `NOT stream` +
-- `upstream_latency_ms IS NOT NULL` exclude them (overhead is uncomputable for a
-- stream). Window is 14 days (not A3's 72h) because activation traffic is sparse
-- and a shorter window would give too few samples for a stable p95.
-- SMALL-SAMPLE CAVEAT: p50/p95 over fewer than ~20 requests is noise — read `n`.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   The wrapper's own added time on activation/steering requests, per model —
--   tracked but NOT held to the 50ms plain-path SLO. In practice this is steering
--   overhead: the base64 steering payload is heavy to serialize, and capture
--   requests stream so they don't appear here. p50 is typical, p95 the slow tail.
--   A reading over fewer than ~20 requests (see `n`) is noise.
SELECT CASE WHEN model = 'trinity-base' THEN 'trinity-truebase'
            ELSE coalesce(model, '(unknown)') END                     AS model,  -- ACS-147 normalize
       percentile_cont(0.5)  WITHIN GROUP (
         ORDER BY latency_ms - upstream_latency_ms)                   AS p50_overhead_ms,
       percentile_cont(0.95) WITHIN GROUP (
         ORDER BY latency_ms - upstream_latency_ms)                   AS p95_overhead_ms,
       count(*)                                                       AS n
FROM api_requests
WHERE ts > now() - interval '14 days'
  AND ts >= {{start_date}}
  AND upstream_latency_ms IS NOT NULL
  AND NOT stream
  AND NOT cold_boot
  AND activation                                                      -- activation/steering path only (ACS-325)
GROUP BY 1 ORDER BY 1;

-- A4. Warm request success rate (SLO ≥ 99%) --------------------------------
-- success = 2xx / (2xx + real 5xx). Excludes 4xx (client), late cold-start
-- failures, and circuit_open.
-- NOTE: pre-beta this is depressed by our own stress/incident testing (e.g. the
-- June-16 llama-405b incident). Once invites go out, consider windowing this to
-- "since beta launch" so test noise doesn't mask the real signal.
-- GAUGE COLOUR BANDS (ACS-348): the SLO line is 99% (ACS-54). For a 3-band gauge,
-- bands must be contiguous, so:
--   green  ≥ 99%      — healthy (meets SLO)
--   amber  95 – <99%  — watch (below SLO but not critical)
--   red    < 95%      — breach (critical)
-- (ACS-348 phrased the amber/red boundary loosely as "95–99% watch / <99%
-- breach"; 99% is the SLO line — anything under it is out of SLO — and <95% is
-- the hard-breach cutoff.) In Metabase set these on the gauge's two thresholds.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   The share of warm requests that succeeded — our top-line reliability number,
--   target ≥99%. It counts successes (2xx) against genuine server failures only;
--   client mistakes (4xx), cold-boot 503s and breaker 503s are ignored so they
--   don't count against us. Gauge colours: green ≥99% healthy · amber 95–99%
--   watch (under SLO) · red <95% breach. Pre-beta this is dragged down by our own
--   incident testing.
SELECT round(100.0 * count(*) FILTER (WHERE status < 400)
             / nullif(count(*) FILTER (WHERE status < 400
                        OR (status >= 500 AND coalesce(error_kind,'')
                            NOT IN ('cold_boot','circuit_open'))), 0), 3)         AS warm_success_pct,
       count(*) FILTER (WHERE status < 400)                                       AS ok,
       count(*) FILTER (WHERE status >= 500
                        AND coalesce(error_kind,'') NOT IN ('cold_boot','circuit_open')) AS warm_failures
FROM api_requests
WHERE ts > now() - interval '7 days'
  AND ts >= {{start_date}}
  AND NOT cold_boot;

-- A5. Cold-start frequency by model (beta-learning input, no SLO target) ---
-- coalesce(model,...) so requests with no model tag don't render as a blank
-- "(empty)" series (early/health rows can have a NULL model).
-- LATENCY DATA-QUALITY WATCH (re A2): if p95_total_ms shows multi-hundred-
-- second "warm" spikes, check whether those rows are mis-tagged cold-boots
-- (cold_boot=false but really a container start) vs genuine long batch
-- generations — the former would understate this tile and inflate A2.
-- VIZ: line/bar — X=day. Series legend: one series per model (ACS-147-normalized);
-- cold_boot_pct is that model's cold-start share for the day.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   How often each model had to cold-start (spin up a fresh container) per day,
--   and what share of its requests that was (cold_boot_pct). A high share means
--   users are waiting on spin-ups for that model. No SLO target — it's an input
--   to warm-pool tuning.
SELECT date_trunc('day', ts)                                          AS day,
       CASE WHEN model = 'trinity-base' THEN 'trinity-truebase'
            ELSE coalesce(model, '(unknown)') END                     AS model,  -- ACS-147 normalize
       count(*) FILTER (WHERE cold_boot)                              AS cold_boot_reqs,
       count(*)                                                       AS total,
       round(100.0 * count(*) FILTER (WHERE cold_boot)
             / nullif(count(*), 0), 2)                                AS cold_boot_pct
FROM api_requests
WHERE ts > now() - interval '14 days'
  AND ts >= {{start_date}}
GROUP BY 1, 2 ORDER BY 1, 2;

-- A6. Cap-pressure 429s over time (inline + bulk harvesting) ---------------
-- "Do I need to raise a cap?" — daily counts of every capacity/quota 429 in
-- api_requests, one column per cap. Column ↔ error_kind ↔ knob (triage detail
-- lives in the error-troubleshooting.md taxonomy table; live values in
-- settings.py / Railway env — deliberately not restated here):
--   bulk_big_model_capacity  → harvest_capacity_exceeded — GLOBAL cross-key
--     big-model harvest cap (HARVEST_MAX_RUNNING_BIG_MODEL). Raising it also
--     requires redeploying the big-model harvest apps with a matching
--     HARVEST_MAX_CONTAINERS (serving/harvest_offline.py).
--   bulk_per_key_concurrency → harvest_concurrency_exceeded — per-key running-
--     job cap (HARVEST_MAX_RUNNING_PER_KEY; ONE global setting applied to
--     every key — there is no per-key override).
--   bulk_monthly_quota   → harvest_quota_exceeded — api_keys.monthly_harvest_budget.
--   inline_monthly_quota → activation_quota_exceeded — api_keys.
--     monthly_activation_budget. Only emitted on the inline /v1/completions
--     activation path, so it IS the inline split even though _error rows don't
--     carry the `activation` flag.
--   token_monthly_budget → budget_exceeded — api_keys.monthly_token_budget.
--   completions_queue_full → queue_full — fires only when the per-key WAITER
--     QUEUE overflows (MAX_QUEUED_PER_KEY waiting behind MAX_INFLIGHT_PER_KEY
--     active; all /v1/completions traffic incl. inline activation): a nonzero
--     count means ~80 outstanding requests for one key, and keys pinned at the
--     in-flight cap with a shorter queue wait silently — they never show here.
-- Read with care — these are REJECTION counts, not unmet demand: one client
-- retry loop inflates a column by hundreds/day, while clients that serialize
-- submissions never 429 even when a cap pins their throughput all day (a
-- harvest_jobs-based utilization tile is the direct signal — ACS-275).
-- Zeros in the quota/budget columns can also mean the budget is
-- simply unset (0 = unlimited → the gate is skipped entirely). These rows are
-- a subset of A1's client_4xx — one incident, two tiles, don't double-count.
-- `rate_limited` (slowapi) writes no api_requests row and cannot appear here;
-- _error rows log model=NULL, so no per-model split is possible.
-- KEEP IN SYNC: the FILTER kinds below ↔ A7's CASE + IN list.
-- VIZ: line (or stacked bar) — X=day, Y=each count column as a series. Series
-- legend: one line per cap — see the column↔knob mapping above for what each is.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   How often each capacity or quota limit turned callers away per day, one line
--   per cap. A rising line means that specific knob may need raising. These are
--   rejection counts, not unmet demand — one client retry loop can inflate a
--   line, and callers who submit slowly never show up even when a cap pins them.
--   A flat-zero line is healthy (or the budget is simply unset = unlimited).
SELECT date_trunc('day', ts)                                          AS day,
       count(*) FILTER (WHERE error_kind = 'harvest_capacity_exceeded')    AS bulk_big_model_capacity,
       count(*) FILTER (WHERE error_kind = 'harvest_concurrency_exceeded') AS bulk_per_key_concurrency,
       count(*) FILTER (WHERE error_kind = 'harvest_quota_exceeded')       AS bulk_monthly_quota,
       count(*) FILTER (WHERE error_kind = 'activation_quota_exceeded')    AS inline_monthly_quota,
       count(*) FILTER (WHERE error_kind = 'budget_exceeded')              AS token_monthly_budget,
       count(*) FILTER (WHERE error_kind = 'queue_full')                   AS completions_queue_full
FROM api_requests
WHERE ts > now() - interval '30 days'
  AND ts >= {{start_date}}
GROUP BY 1 ORDER BY 1;
-- (No error_kind WHERE filter on purpose: every day with ANY traffic gets a
-- row of real zeros, so the line chart shows flat-zero series when healthy
-- instead of gaps / "No results" — same pattern as A1.)

-- A7. Who is hitting which cap (drill-down behind A6) ----------------------
-- Grain = user × key × cap. Decides WHICH knob to raise: one key dominating a
-- quota/budget column → raise THAT key's monthly budget (budgets are per-key
-- api_keys columns; the concurrency caps are global settings affecting every
-- key — raising those is a fleet-wide change); several DIFFERENT keys hitting
-- bulk_big_model_capacity → the global big-model cap is the bottleneck.
-- Internal accounts intentionally INCLUDED (ops view, like A1–A6 — our own
-- harvest runs are real cap pressure). cap_hit uses the SAME names as A6's
-- columns. KEEP IN SYNC: the CASE + IN list below ↔ A6's FILTER kinds.
-- VIZ: table, sorted by hits.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Drill-down behind A6: which user + API key hit which cap, how many times, and
--   when (first/last). Use it to decide whose budget to raise — one key
--   dominating a quota column → raise that key's budget; several different keys
--   hitting the same global cap → the fleet-wide cap is the bottleneck. Sorted by
--   hit count.
SELECT u.email,
       ak.key_prefix,
       CASE ar.error_kind
         WHEN 'harvest_capacity_exceeded'    THEN 'bulk_big_model_capacity'
         WHEN 'harvest_concurrency_exceeded' THEN 'bulk_per_key_concurrency'
         WHEN 'harvest_quota_exceeded'       THEN 'bulk_monthly_quota'
         WHEN 'activation_quota_exceeded'    THEN 'inline_monthly_quota'
         WHEN 'budget_exceeded'              THEN 'token_monthly_budget'
         WHEN 'queue_full'                   THEN 'completions_queue_full'
       END                                        AS cap_hit,
       count(*)                                   AS hits,
       min(ar.ts)                                 AS first_hit,
       max(ar.ts)                                 AS last_hit
FROM api_requests ar
JOIN api_keys ak ON ak.id = ar.key_id
JOIN users    u  ON u.id  = ak.user_id
WHERE ar.ts > now() - interval '30 days'
  AND ar.ts >= {{start_date}}
  AND ar.error_kind IN ('harvest_capacity_exceeded', 'harvest_concurrency_exceeded',
                        'harvest_quota_exceeded', 'activation_quota_exceeded',
                        'budget_exceeded', 'queue_full')
GROUP BY u.email, ak.key_prefix, ar.error_kind
ORDER BY hits DESC;


-- =========================================================================
-- B. USAGE — what's consumed (tokens, interactive/batch, GPU cost, feature mix)
-- =========================================================================

-- B1. Interactive vs batch mix --------------------------------------------
-- VIZ: pie/bar — one slice per workload class. Series legend: each row is a
-- workload class — declared workload_type when present, else inferred from
-- whether the request streamed (stream=interactive, unary=batch).
-- ⓘ DESCRIPTION (paste into Metabase card):
--   The mix of interactive (streaming) vs batch (unary) requests over the window.
--   We use the caller's declared workload_type when they set it, otherwise we
--   infer it from whether the request streamed. A bigger interactive share means
--   more live workbench use; a bigger batch share means more scripted pulls.
SELECT coalesce(workload_type,
                CASE WHEN stream THEN 'stream (undeclared)'
                     ELSE 'unary (undeclared)' END)                   AS workload,
       count(*)                                                       AS requests
FROM api_requests
WHERE ts > now() - interval '14 days'
  AND ts >= {{start_date}}
GROUP BY 1 ORDER BY requests DESC;

-- B2. Token volume (in/out) per day per model ------------------------------
-- The "overall usage" headline: tokens processed each day, split by model.
-- Long format (one row per day×model) so Metabase can stack the models as
-- series. n_prompt/n_completion are nullable (early/health/errored rows) =>
-- coalesce to 0; a 4xx/429/cold-boot row contributes ~0 tokens naturally.
-- Counts ALL traffic (token volume = real GPU load, like the Operations A1–A5
-- tiles). To measure tester adoption only, uncomment the internal-exclusion
-- join below (mirrors C2). Pair with B5 ($/model/day) for tokens-vs-spend.
-- VIZ: stacked bar — X=day, Y=total_tokens, Series breakout=model. Series
-- legend: one stacked series per model (ACS-147-normalized).
-- STREAM WATCH: an API stream without stream_options.include_usage can log a
-- NULL n_completion, so stream-path output is slightly undercounted here.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Tokens processed each day, stacked by model — the overall load headline.
--   input_tokens are the prompt, output_tokens are what the model generated. It
--   counts all traffic, because token volume is real GPU load. Streaming requests
--   that don't report usage can slightly undercount output.
SELECT date_trunc('day', ar.ts)                                      AS day,
       CASE WHEN ar.model = 'trinity-base' THEN 'trinity-truebase'
            ELSE coalesce(ar.model, '(unknown)') END                 AS model,  -- ACS-147 normalize
       sum(coalesce(ar.n_prompt, 0))                                 AS input_tokens,
       sum(coalesce(ar.n_completion, 0))                             AS output_tokens,
       sum(coalesce(ar.n_prompt, 0) + coalesce(ar.n_completion, 0))  AS total_tokens,
       count(*)                                                      AS requests
FROM api_requests ar
-- JOIN api_keys ak ON ak.id = ar.key_id            -- uncomment 2 lines + WHERE for adoption view
-- JOIN users    u  ON u.id  = ak.user_id
WHERE ar.ts >= {{start_date}}
  AND ar.ts > now() - interval '30 days'
  -- AND NOT EXISTS (SELECT 1 FROM user_tags t WHERE t.user_id = u.id AND t.tag = 'internal')  -- see header
GROUP BY 1, 2
ORDER BY 1, 2;

-- B3. Token volume in vs out per day (all models) --------------------------
-- The input/output split over time. Wide format: plot both numeric columns as
-- stacked series. Watch the ratio — base-model research is often prompt-heavy
-- (scoring / prompt_logprobs), so input can dominate, which loads capacity/cost
-- differently than output-heavy generation.
-- VIZ: stacked bar — X=day, Y=[input_tokens, output_tokens]. Series legend:
-- input_tokens=prompt tokens · output_tokens=generated tokens.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Daily tokens split into prompt (input) vs generated (output), all models
--   combined. Watch the ratio: base-model research is often prompt-heavy
--   (scoring / logprobs), which loads cost differently than output-heavy
--   generation.
SELECT date_trunc('day', ts)                                   AS day,
       sum(coalesce(n_prompt, 0))                              AS input_tokens,
       sum(coalesce(n_completion, 0))                          AS output_tokens,
       sum(coalesce(n_prompt, 0) + coalesce(n_completion, 0))  AS total_tokens
FROM api_requests
WHERE ts >= {{start_date}}
  AND ts > now() - interval '30 days'
GROUP BY 1 ORDER BY 1;

-- B4. Token volume by model over the window (leaderboard) ------------------
-- At-a-glance "which models carry the load" for the selected window.
-- VIZ: row bar (Y=model, X=total_tokens) or a table.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Which models carry the load over the window — total tokens, request count,
--   and average output length per model, ranked by total tokens. A quick "what's
--   actually being used" leaderboard.
SELECT CASE WHEN model = 'trinity-base' THEN 'trinity-truebase'
            ELSE coalesce(model, '(unknown)') END              AS model,  -- ACS-147 normalize
       count(*)                                                AS requests,
       sum(coalesce(n_prompt, 0))                              AS input_tokens,
       sum(coalesce(n_completion, 0))                          AS output_tokens,
       sum(coalesce(n_prompt, 0) + coalesce(n_completion, 0))  AS total_tokens,
       round(avg(coalesce(n_completion, 0)), 1)                AS avg_output_per_req
FROM api_requests
WHERE ts >= {{start_date}}
  AND ts > now() - interval '14 days'
GROUP BY 1 ORDER BY total_tokens DESC;

-- B5. GPU spend per model per day (estimated) ------------------------------
-- From gpu_cost_sample (written by the cost-monitor scheduler job). est_usd
-- per sample = running_containers × per-container hourly rate × interval, so
-- summing per day ≈ $/model/day. Approximate (count-at-tick sampling) — it's a
-- spend trend + runaway tripwire, not exact billing. The cost-spike alert lives
-- in cost_monitor.py (Sentry, threshold = settings.cost_alert_daily_usd).
-- Since ACS-221 the sampler also writes a row per activation-enabled model for
-- its separate Modal activation engine, under model_id '<model>::activation'
-- (e.g. 'llama-8b::activation'), and since ACS-281 a row per harvest-capable
-- model for its bulk-harvester app, under '<model>::harvest' — so each model
-- shows up as (up to) three series here: serving, activation, harvest. No SQL
-- change needed.
-- VIZ: stacked bar/line — X=day. Series legend: one series per model_id, where a
-- model appears as up to three: '<model>' (serving), '<model>::activation',
-- '<model>::harvest'.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Estimated GPU dollars per model per day. Each model can show up as three
--   series — serving, ::activation, ::harvest — for its separate engines. The
--   number is sampled at intervals, so read it as a spend trend and a runaway
--   tripwire, not an exact bill.
SELECT date_trunc('day', ts)                  AS day,
       CASE WHEN model_id = 'trinity-base' THEN 'trinity-truebase'
            ELSE model_id END                  AS model_id,  -- ACS-147 normalize
       round(sum(est_usd)::numeric, 2)         AS est_usd
FROM gpu_cost_sample
WHERE ts >= {{start_date}}
GROUP BY 1, 2 ORDER BY 1, 2;

-- B6. Total GPU spend per day (all models, estimated) ----------------------
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Estimated total GPU dollars per day across all models and engines. Sampled,
--   so it's a trend line, not exact billing.
SELECT date_trunc('day', ts)                  AS day,
       round(sum(est_usd)::numeric, 2)         AS est_usd_total
FROM gpu_cost_sample
WHERE ts >= {{start_date}}
GROUP BY 1 ORDER BY 1;


-- -------------------------------------------------------------------------
-- FEATURE-MIX ENGAGEMENT (ACS-333) — plain vs advanced-feature usage
-- -------------------------------------------------------------------------
-- "Which features do users actually use?" Splits traffic into three feature
-- classes from the api_requests activation telemetry (migration 0031, ACS-199;
-- columns activation bool, activation_layers int|null, activation_steering_vectors
-- int|null). Discriminator — mirrors
-- base_model_wrapper/src/wrapper/services/request_log.py:53-95:
--   PLAIN    = NOT activation                                        (ordinary completion)
--   CAPTURE  = activation AND activation_layers IS NOT NULL          (residual-stream harvest)
--   STEERING = activation AND activation_steering_vectors IS NOT NULL (steering vectors applied)
-- OVERLAP: a single request can be BOTH capture and steering. It then counts in
-- BOTH the capture and steering columns/counts, so capture + steering can exceed
-- the activation total. The clean partition is plain + activation-total = grand
-- total; capture and steering are overlapping subsets of activation, not a
-- partition of it.
-- SUCCESS-ONLY (status < 400): a failed/rejected activation request is logged via
-- the _error path, which writes activation=false (see the A6 inline_monthly_quota
-- note + request_log.py) — so a failed activation attempt would masquerade as
-- PLAIN. Restricting to successful requests keeps the split honest (adoption =
-- features that actually ran) and avoids inflating PLAIN with failed activations.

-- B7. Daily request counts by feature class (ACS-333) ----------------------
-- VIZ: stacked bar / line — X=day, Y=[plain, capture, steering] as series.
-- Series legend: plain=ordinary completions · capture=residual-stream harvest ·
-- steering=steering vectors applied. capture+steering rows are counted in BOTH
-- the capture and steering series (see subsection OVERLAP note).
-- Counts ALL traffic incl. internal team (volume view, like A1–A6 / B2). For
-- external-tester adoption only, uncomment the join + WHERE (mirrors B2/C2).
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Daily request counts split by feature: plain completions vs capture
--   (residual-stream harvest) vs steering. Shows whether the advanced features
--   are being picked up over time. A request that both captured and steered is
--   counted in both the capture and steering lines. Successful requests only.
SELECT date_trunc('day', ar.ts)                                      AS day,
       count(*) FILTER (WHERE NOT ar.activation)                     AS plain,
       count(*) FILTER (WHERE ar.activation
                        AND ar.activation_layers IS NOT NULL)        AS capture,
       count(*) FILTER (WHERE ar.activation
                        AND ar.activation_steering_vectors IS NOT NULL) AS steering
FROM api_requests ar
-- JOIN api_keys ak ON ak.id = ar.key_id            -- uncomment 2 lines + WHERE for adoption view
-- JOIN users    u  ON u.id  = ak.user_id
WHERE ar.ts > now() - interval '30 days'
  AND ar.ts >= {{start_date}}
  AND ar.status < 400                               -- successful feature use only (see subsection note)
  -- AND NOT EXISTS (SELECT 1 FROM user_tags t WHERE t.user_id = u.id AND t.tag = 'internal')  -- see header
GROUP BY 1 ORDER BY 1;

-- B8. Distinct users per feature over the window (ACS-333) ------------------
-- "How many people actually use each advanced feature?" Three counts of DISTINCT
-- users (by user id) who made ≥1 successful request in each class over the
-- window. A user who both captured and steered is counted in BOTH advanced
-- columns (same OVERLAP rule as B7). Internal team EXCLUDED — this is an adoption
-- question (like C1/C2): we want real testers, not our own activation testing.
-- One row (no GROUP BY — pure aggregate), so it renders as 3 big-number cards or
-- a one-row table.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   How many distinct people actually used each feature over the window — plain
--   vs capture vs steering. This is the adoption headline for the advanced
--   features: a huge token count from one power user still counts as one user
--   here. Someone who did both capture and steering is counted in both advanced
--   columns. Internal team excluded.
SELECT count(DISTINCT u.id) FILTER (WHERE NOT ar.activation)             AS plain_users,
       count(DISTINCT u.id) FILTER (WHERE ar.activation
                        AND ar.activation_layers IS NOT NULL)            AS capture_users,
       count(DISTINCT u.id) FILTER (WHERE ar.activation
                        AND ar.activation_steering_vectors IS NOT NULL)  AS steering_users
FROM api_requests ar
JOIN api_keys ak ON ak.id = ar.key_id
JOIN users    u  ON u.id  = ak.user_id
WHERE ar.ts > now() - interval '30 days'
  AND ar.ts >= {{start_date}}
  AND ar.status < 400                               -- successful feature use only
  AND NOT EXISTS (SELECT 1 FROM user_tags t WHERE t.user_id = u.id AND t.tag = 'internal')  -- see header
;


-- =========================================================================
-- C. USERS — who's testing & how engaged (customer360, ACS-139)
-- =========================================================================

-- C1. Per-user summary — customer360 --------------------------------------
-- One row per approved tester, consolidating the old activation / roster /
-- churn tiles: identity, onboarding (access_granted → first request +
-- time-to-first), recency + engagement bucket, and lifetime usage.
-- LEFT JOINs from `users` so an invited-but-never-called tester still appears
-- (first_request_at NULL, engagement = 'never_activated' — the activation gap).
-- Invite info comes from a LATERAL aggregate (one row per user) so a user with
-- >1 invite can't fan out and double the usage sums; multiple api_keys don't
-- double either (each request has one key_id). Internal team excluded.
-- Engagement: active ≤7d · cooling 8–14d · churned >14d · never_activated.
-- ⚠ The in-app admin roster/person page ports this bucket definition — keep it
-- in sync with base_model_wrapper/src/wrapper/services/customer360.py (ACS-300).
-- Identity beyond name/org (Discord handle, structured affiliation, multi-email
-- /account merge) isn't captured yet (→ ACS-155). GPU $ is shared infra, not
-- per-user-attributable (see B5 for $/model/day).
-- Columns are ordered important-first (Metabase shows ~7 before horizontal
-- scroll; dates are pushed to the end — reorder in the table viz settings if you
-- like). VIZ: table. Default sort = most-recently-active; click days_since_last
-- or filter engagement IN ('cooling','churned','never_activated') for outreach.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   One row per approved external tester: who they are, how engaged they are
--   (active / cooling / churned / never_activated), when they first and last
--   called the API, and their lifetime usage. Sorted most-recently-active first.
--   Use it for outreach — filter to cooling / churned / never_activated. Internal
--   team excluded.
SELECT u.email,
       u.name,
       CASE
         WHEN count(ar.id) = 0                          THEN 'never_activated'
         WHEN max(ar.ts) >= now() - interval '7 days'   THEN 'active'
         WHEN max(ar.ts) >= now() - interval '14 days'  THEN 'cooling'
         ELSE 'churned'
       END                                                          AS engagement,
       date_part('day', now() - max(ar.ts))::int                     AS days_since_last,
       sum(coalesce(ar.n_prompt, 0) + coalesce(ar.n_completion, 0))  AS total_tokens,
       -- Most-used model (statistical mode); NULL for never-activated. Normalized.
       mode() WITHIN GROUP (ORDER BY CASE WHEN ar.model = 'trinity-base'
                            THEN 'trinity-truebase' ELSE ar.model END) AS favorite_model,
       -- % of the user's requests that streamed. The workbench always streams,
       -- so this is the interactive-vs-batch proxy (workload_type is too sparse).
       round(100.0 * count(*) FILTER (WHERE ar.stream)
             / nullif(count(ar.id), 0), 0)                           AS pct_interactive,
       count(ar.id)                                                  AS requests,
       count(DISTINCT ak.id)                                         AS keys,
       count(DISTINCT CASE WHEN ar.model = 'trinity-base' THEN 'trinity-truebase'
                           ELSE ar.model END)                        AS models_used,  -- ACS-147 normalize
       u.org,
       sum(coalesce(ar.n_prompt, 0))                                 AS input_tokens,
       sum(coalesce(ar.n_completion, 0))                            AS output_tokens,
       -- Activation latency as decimal days (the raw interval renders too verbosely).
       round((extract(epoch FROM (min(ar.ts) - coalesce(inv.invited_at, u.approved_at)))
              / 86400.0)::numeric, 2)                                AS days_to_first_request,
       coalesce(inv.invited_at, u.approved_at)                       AS access_granted_at,
       min(ar.ts)                                                    AS first_request_at,
       max(ar.ts)                                                    AS last_request_at,
       inv.redeemed_at                                              AS redeemed_at
FROM users u
LEFT JOIN LATERAL (
    -- Collapse a user's invite(s) to one row so the joins below don't fan out.
    SELECT min(si.created_at) AS invited_at,
           max(r.redeemed_at) AS redeemed_at
    FROM signup_invite_redemptions r
    LEFT JOIN signup_invites si ON si.id = r.invite_id
    WHERE r.user_id = u.id
) inv ON true
LEFT JOIN api_keys     ak ON ak.user_id = u.id
LEFT JOIN api_requests ar ON ar.key_id  = ak.id
-- 'suspended' counts as approved-for-history here (ACS-353): suspension is a
-- reversible access park, and a user who onboarded and made requests really did
-- activate. Dropping them on suspend would retroactively rewrite past cohort
-- denominators — the activation rate for a closed cohort would change months
-- later because of an admin action today. Excluding a cohort from reporting is
-- a job for tags, not for the access state.
WHERE u.status IN ('approved', 'suspended')
  -- Floor on the access-granted date = the onboarding cohort since the cutoff
  -- (not request ts, so a tester onboarded post-cutoff still counts even if
  -- they haven't called the API yet — the activation gap we want to see).
  AND coalesce(inv.invited_at, u.approved_at) >= {{start_date}}
  AND NOT EXISTS (SELECT 1 FROM user_tags t WHERE t.user_id = u.id AND t.tag = 'internal')  -- see header
GROUP BY u.id, u.email, u.name, u.org, inv.invited_at, inv.redeemed_at
ORDER BY last_request_at DESC NULLS LAST;

-- C2. Per-key activity detail (drill-down behind C1's `keys`) --------------
-- Grain = user × key: which key is doing the work. INNER JOIN from
-- api_requests, so only keys with traffic in the window appear. Recency-windowed
-- (last 30d) — a "who's active right now" view, not C1's lifetime totals.
-- ⓘ DESCRIPTION (paste into Metabase card):
--   Drill-down behind C1's key count: per API key, how recently it was used and
--   how much traffic and how many tokens it drove in the last 30 days. Shows
--   which specific key is doing a user's work. Internal team excluded.
SELECT u.email,
       ak.key_prefix,
       ak.name                                    AS key_name,
       date_part('day', now() - max(ar.ts))::int  AS last_seen_days_ago,
       count(*)                                   AS requests,
       sum(coalesce(ar.n_prompt, 0))              AS input_tokens,
       sum(coalesce(ar.n_completion, 0))          AS output_tokens,
       max(ar.ts)                                 AS last_seen
FROM api_requests ar
JOIN api_keys ak ON ak.id = ar.key_id
JOIN users    u  ON u.id  = ak.user_id
WHERE ar.ts >= {{start_date}}
  AND ar.ts > now() - interval '30 days'
  AND NOT EXISTS (SELECT 1 FROM user_tags t WHERE t.user_id = u.id AND t.tag = 'internal')  -- see header
GROUP BY u.email, ak.key_prefix, ak.name
ORDER BY requests DESC;
