---
title: GPU cost monitoring & runner-count reads
status: current
updated: 2026-08-10
owner: platform@example.org
---

# GPU cost monitoring & runner-count reads — how it works now

How the wrapper estimates GPU spend, and how it reads live container counts from
Modal's control plane (the input both this and the admin Models page depend on).

## Cost monitor

- **Sampler.** An APScheduler job (`cost_monitor.run_cost_sample`, wired in
  `lifespan.py` alongside the probe/warm-window jobs) runs per
  `COST_SAMPLE_INTERVAL_MINUTES` (default 60) and writes one `gpu_cost_sample`
  row per *live* model (migration `0020`, model in `models.py`) — plus, since
  ACS-221, one row per activation-enabled model (those with
  `activation_upstream_url` set) for its separate `acs-<id>-activation` Modal
  app, recorded under `model_id = "<id>::activation"` (same suffix convention
  as the breaker key). The activation engine runs on the same GPU shape as
  serving, so it reuses the model's `gpu_shape_label`/rate. The activation
  app name derives from the wrapper model id (`modal_ops.activation_app_name`;
  a registry entry's explicit `activation_app_name` overrides) — NOT from
  `modal_app_name`, because Trinity's serving app kept the pre-rename name
  while its activation app is `acs-trinity-truebase-activation`. The web
  function's tag differs by lifecycle path (`serve_activation` on multi-GPU,
  `ActivationSnap.*` on the single-GPU snapshot path), so the fetch tries both
  (`ACTIVATION_FUNCTION_TAGS`, remembering the winning tag per app). Since
  ACS-281 the sampler also writes a third row per harvest-capable model (those
  with a Modal serving app, or an explicit `harvest_app_name` override) for its
  `acs-<id>-harvest` bulk-harvester app (ACS-245), under
  `model_id = "<id>::harvest"` — same shape/rate reuse and same
  model-id-derived naming (`modal_ops.resolve_harvest_app`, shared with the
  /v1/harvest route). Harvest containers are ephemeral batch jobs: the tick
  sampler catches the long multi-GPU harvests (the spend that matters); a
  short 8B harvest can finish between ticks and go uncounted. All engines
  sample independently: an undeployed app (kimi's activation today; any
  not-yet-deployed harvester) logs `cost_sample_runner_count_failed` and skips
  that row only — the other samples still land — and a serving control-plane
  blip likewise doesn't drop the sidecar rows.
- **Suspicious-zero retry (serving)** (ACS-112). The serving read uses
  `fetch_runner_count_fresh` (uncached), so a returned 0 is Modal affirmatively
  reporting `num_total_tasks == 0`, not a swallowed failure. But a single
  control-plane RPC can briefly under-report while containers exist, so the
  sampler re-reads once when a fresh 0 comes back for an app that isn't
  explicitly *stopped* (`_read_serving_count` → `_app_maybe_running`, which
  consults the SWR-cached `get_app_state`). This is belt-and-braces on top of
  the earlier cached-0 fix (ACS-50): it narrows, not eliminates, the
  transient-zero window (a second read can still return the same 0). Only the
  serving engine retries; activation/harvest apps legitimately sit at 0.
- **Boot-time rate-config validation** (ACS-112). `parsed_gpu_hourly_rates()`
  returns `{}` on malformed JSON, so a *set-but-broken* `GPU_HOURLY_USD_BY_TYPE_JSON`
  makes every `per_container` rate 0 and silently writes `est_usd = 0` rows.
  `lifespan` calls `cost_monitor.rate_config_problem(settings)` at boot and logs
  `gpu_rate_config_invalid` when the raw var is non-empty but parses to no usable
  rates (an *intentionally* empty var is fine → no warning). Partial drops (one
  bad value in an otherwise-good object) are not flagged at boot — those still
  surface per-tick as `cost_sample_no_rate_for_gpu_type`.
- **Estimate.** `est_usd = running_containers × per_container_$/hr × elapsed_hours`.
  Per-container rate = the gpu_type's per-GPU rate (`GPU_HOURLY_USD_BY_TYPE_JSON`,
  default `{"H200": 4.54, "L40S": 1.95}`) × the model's GPU count parsed from
  `gpu_shape_label`. So an 8×H200 container ≈ $36/hr; Trinity always-on ≈ $870/day.
- **Cadence robustness.** The job also fires once at boot (`next_run_time`, after
  `COST_SAMPLE_WARMUP_SECONDS` so Modal's client is ready) — `IntervalTrigger`
  alone first-fires at boot+interval, and frequent Railway redeploys reset that
  clock, so a pure interval job would ~never run. Each sample bills the **actual
  elapsed time since the last sample** (clamped to one interval), so firing on
  every redeploy doesn't over-count and a long outage records ≤ one interval.
- **Spike alert.** After each write, rolling-24h `sum(est_usd)` over
  `COST_ALERT_DAILY_USD` (default 1200, `0` disables) →
  `sentry_sdk.capture_message(level="warning", fingerprint=["gpu-cost-spike"])`.
  The stable fingerprint groups all spikes into one Sentry issue (fires the
  Discord notification once, then increments occurrences — no spam). This is a
  direct SDK capture, so it reaches Sentry even though the stdlib
  `LoggingIntegration` is disabled; routing to Discord depends on the Sentry
  alert rule covering **warning-level** issues.
- **Surface.** Metabase tiles B5 ($/model/day) and B6 (total $/day) in
  [`../runbooks/metabase-dashboard-queries.sql`](../runbooks/metabase-dashboard-queries.sql),
  windowed by `{{start_date}}`.

Approximate by design (count-at-tick × rate) — a spend trend + runaway tripwire,
not exact billing. Accurate per-run reconciliation is `benchmarks/billing.py`
(container-seconds from the lifetime-log Volume).

**Deferred — effective-utilization scrape (ACS-112, Part 1).** A warm-only scrape
of each container's vLLM `/metrics` (`gpu_cache_usage_perc`,
`num_requests_running`, tokens/s) to chart how *hard* the paid GPUs are working,
not just that they're on. Not yet built, and it is a genuine design spike, not a
one-line add: (1) vLLM `/metrics` is **per-replica**, and hitting a model's Modal
web endpoint load-balances to *one arbitrary container* — there is no trivial way
to address and scrape *each* warm container, so per-container utilization needs a
real fan-out/aggregation design; (2) it needs a new table + additive migration
and a new chart surface (there is no read path for `gpu_cost_sample` today — spend
is surfaced only via Metabase + the Sentry spike alert). Scoped as its own ticket
rather than half-built here.

## Runner-count reads — two prod-only gotchas

The cost sampler and the admin Models "live container count" both read the count
from Modal. Two things bit us hard (invisible locally, only failed in the
deployed container):

1. **Use the internal async client, not the high-level API.**
   `modal.Function.from_name(app, "serve").get_current_stats.aio()` logs
   `RPC request … made outside of task context` and **hangs until timeout for
   every model in the Railway container** (a laptop probe returns in ~0.3s —
   environment-specific). Read instead via the internal client, the same path
   `_resolve_app` uses for app state and which works reliably in prod:
   `await _ModalAsyncClient.from_env()` → `client.stub.FunctionGet(...)` →
   `client.stub.FunctionGetCurrentStats(function_id=…)`; the count is the
   response's `num_total_tasks` (what the SDK exposes as `num_total_runners`).
   This is `modal_ops._fetch_runner_count`. **Rule: any Modal control-plane RPC
   from the wrapper (especially from background/scheduler jobs) goes through
   `_ModalAsyncClient` + `client.stub`, never the high-level `.aio()` helpers.**
2. **`gpu_shape_label` uses "×" (U+00D7)**, e.g. `"8×H200"` — not ASCII `"x"`.
   The cost sampler's shape parser must accept both (`[x×]`); a regex matching
   only `"x"` skips every model. (`services/completions.py` already had a ×→x
   helper — the convention predates the cost monitor.)

Cache semantics (`modal_ops`): `get_active_runner_count` is stale-while-revalidate
and caches a failed read as **`None` (unknown)**, never a misleading `0`, so the
admin page renders "live (count unavailable)" rather than a false "scaled to
zero". The cost sampler instead uses `fetch_runner_count_fresh` (uncached,
write-through, raises on failure) and **skips** a model on failure — better a
gap than a billed-as-idle always-on model. Runner-fetch failures log
`runner_count_fetch_failed` (added so the silent-swallow path is diagnosable).

## Operating it

- **Tune the alert:** set `COST_ALERT_DAILY_USD` in Railway. Steady state is
  ≈ Trinity ($870) + llama-8b serving ($47) + llama-8b activation warm floor
  ($47, tracked since ACS-221 — before that it was invisible here) ≈ $965/day,
  so 1200 leaves headroom for occasional on-demand 405B/Kimi; lower it for
  earlier warning, `0` to disable.
- **Retune rates** if Modal pricing moves: `GPU_HOURLY_USD_BY_TYPE_JSON`.
- A flat/empty `gpu_cost_sample` after deploy → check Railway logs for
  `runner_count_fetch_failed` (the read is failing) before suspecting the sampler.
