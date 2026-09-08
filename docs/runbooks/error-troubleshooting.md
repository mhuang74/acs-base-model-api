---
title: Error / failure troubleshooting runbook
status: current
updated: 2026-08-05
owner: platform@example.org
---

# Error / failure troubleshooting runbook

How to triage a 4xx/5xx spike on the API. Pairs with the Metabase usage
dashboard (`metabase-dashboard-queries.sql`, ACS-38) and Sentry.

## First: classify, don't panic

`status >= 500` lumps together expected, derived, and real failures. The
`error_kind` column on `api_requests` is what separates them. **Always break a
spike down by `error_kind` before treating it as an incident:**

```sql
SELECT date_trunc('hour', ts) AS hour, status, error_kind, model, count(*) AS n
FROM api_requests
WHERE ts >= 'YYYY-MM-DD' AND ts < 'YYYY-MM-DD'::date + 1
  AND status >= 500
GROUP BY 1, 2, 3, 4
ORDER BY hour, n DESC;
```

Add `key_id` (join `api_keys`/`users`, see dashboard query C2) to tell **your own
test traffic / `/admin/debug/boom`** from a real tester.

## The taxonomy (what actually lands in `error_kind`)

| `error_kind` | HTTP | Class | Meaning | Set at |
|---|---|---|---|---|
| `cold_boot` | 200* | **Expected / allocation incident** | Normal cold starts complete in one held-open request with `cold_boot=true`; `error_kind=cold_boot` means the 14-minute startup budget was exceeded after keepalive bytes committed the transport status. | `routes/api.py` (ColdBootError late payload) |
| `budget_exceeded` | 429 | **Expected** | Key/account over monthly token budget. | `routes/api.py` |
| `activation_quota_exceeded` | 429 | **Expected** | Key over its monthly activation-request quota (`api_keys.monthly_activation_budget`, 0 = unlimited). Separate from token budget; resets at the start of next UTC month. | `routes/api.py` (ACS-199) |
| `harvest_quota_exceeded` | 429 | **Expected** | Key over its monthly bulk-harvest job quota (`api_keys.monthly_harvest_budget`, 0 = unlimited). Counted against `harvest_jobs` rows started this UTC month; resets next month. | `routes/api.py` (ACS-245) |
| `harvest_concurrency_exceeded` | 429 | **Expected / abuse guard** | Key already has its max concurrently-running harvest jobs **in that model's lane** (ACS-321): big multi-GPU models use `HARVEST_MAX_RUNNING_PER_KEY` (default 1), small single-GPU models use `HARVEST_MAX_RUNNING_PER_KEY_SMALL` (default 3), counted independently so a running 405B job never blocks an 8B submit. The message names the lane that is full. Note `HARVEST_MAX_RUNNING_PER_KEY` changed meaning in ACS-321 — it now bounds only the big-model lane. Carries a `Retry-After: 30` header (ACS-344). Poll `GET /v1/harvest/<id>` until the running job finishes, then resubmit — or `DELETE /v1/harvest/<id>` to cancel the running job and free the slot immediately. | `routes/api.py` (ACS-245, ACS-344) |
| `harvest_capacity_exceeded` | 429 | **Expected** | GLOBAL (cross-key) cap on concurrently-running big-model (multi-GPU) harvest jobs is full (`HARVEST_MAX_RUNNING_BIG_MODEL`). A transient queueing signal, not a per-key quota — retry shortly (carries a `Retry-After: 30` header, ACS-344). Small 1-GPU harvests are uncapped. Raising the cap also requires redeploying the big-model harvest apps with a matching `HARVEST_MAX_CONTAINERS` (`serving/harvest_offline.py::_harvest_max_containers_default`). | `routes/api.py` (ACS-265) |
| `queue_full` | 429 | **Expected / abuse guard** | Key's bounded waiter queue overflowed: `MAX_QUEUED_PER_KEY` (64) requests already waiting behind `MAX_INFLIGHT_PER_KEY` (16, since ACS-249) active; caller should back off. Fires only on queue *overflow* — a key pinned at the in-flight cap with a shorter queue waits silently. | `routes/api.py` |
| `rate_limited` | 429 | **Expected / abuse guard** | Key exceeded 600 request starts/minute. Rate limiting happens before the request-log row is created, so inspect access logs for this code. | `main.py` / SlowAPI |
| `circuit_open` | 503 | **Derived** | Breaker is open after repeated upstream failures — a *symptom*, chase the root cause below, not this. | `breaker.py` / `routes/api.py` |
| `upstream_unreachable` | 502 | **Real** | Wrapper couldn't reach the Modal app at all (down / not deployed / network / failing to boot). | `proxy.py` → `routes/api.py` |
| `harvest_unavailable` | 503 | **Real** | `POST /v1/harvest` could not spawn the Modal harvest app (`acs-<model>-harvest` not deployed, or Modal credentials/control-plane failure). Deploy the harvest app for that model (bucket go-live checklist in `docs/design/activation-offline-harvest.md`). | `modal_ops.py` → `routes/api.py` (ACS-245) |
| `vllm_oom` | 502 | **Real** | vLLM ran out of GPU memory. | `proxy.classify_upstream_error_body` |
| `vllm_engine_dead` | 502 | **Real** | vLLM engine/worker crashed. | `proxy.classify_upstream_error_body` |
| `upstream_5xx` | 502 | **Real** | Upstream 5xx we couldn't classify more precisely. | `routes/api.py` |
| `upstream_error` | 502 | **Real** | Workbench-path upstream failure that couldn't be classified further (the workbench doesn't split 4xx/5xx like `/v1` does). | `workbench_generations.py` |
| `vllm_context_length` | 400 | Client-ish | Prompt + max_tokens exceeds the model's context, detected from the upstream error body. Since ACS-322 a client-ish upstream kind returns 400 (not 502), isn't retried, and does NOT count against the breaker. | `proxy.classify_upstream_error_body` |
| `vllm_invalid_request` | 400 | Client-ish | Bad request reflected from upstream's error body — includes an out-of-range steering `layer_index` (vLLM-Lens raises `ValueError` → 500; classified here so it returns a 400, not a mislabelled 5xx). Pre-flight (`routes/api.py`, when the model publishes `n_layers`) catches most of these as `invalid_request` before dispatch; this is the safety net (ACS-322). Returns 400, not retried, doesn't trip the breaker. | `proxy.classify_upstream_error_body` |
| `upstream_4xx` | 4xx | Client | Non-5xx error passed through from upstream. | `routes/api.py` |
| `invalid_request` | 400 | Client | Unknown field / bad type / out-of-range param. | `routes/api.py` |
| `activations_unsupported` | 400 | Client | Request sent `output_residual_stream` / `apply_steering_vectors` to a model with no activation engine configured (`activation_upstream_url`). Check `GET /v1/models` `activations`. | `routes/api.py` (ACS-199) |
| `harvest_unsupported` | 400 | Client | `POST /v1/harvest` for a model with no Modal-backed harvest path (registry entry has no `modal_app_name`). | `routes/api.py` (ACS-245) |
| `activation_prompt_too_long` | 400 | Client | Activation capture (`output_residual_stream`) covers more positions than the per-model `max_activation_prompt_tokens` cap allows. The cap counts **prompt + generated − 1** captured positions (ACS-255), so a short prompt with a large `max_tokens` can also trip it — the streamed response scales with total positions × captured layers. Use a shorter prompt, a smaller `max_tokens`, request fewer layers, or the offline bulk path (`POST /v1/harvest`). | `routes/api.py` (ACS-199, ACS-255) |
| `harvest_job_not_found` | 404 | Client | `GET`/`DELETE /v1/harvest/<id>` for a job id that doesn't exist **or belongs to another key/user** — deliberately indistinguishable so job ids don't leak existence. | `routes/api.py` (ACS-245) |
| `harvest_not_cancellable` | 409 | Client | `DELETE /v1/harvest/<id>` for a job already in a terminal state (`done`/`failed`/`cancelled`) — there is nothing to cancel. Also returned if a poll finalized the job in the race window between the ownership read and the cancel write (the committed terminal state wins). | `routes/api.py` (ACS-344) |
| `request_too_large` | 413 | Client | Request body exceeds the accepted Content-Length (currently the `POST /v1/harvest` corpus cap, `HARVEST_MAX_BODY_BYTES`, default 50 MB). Split the corpus into multiple jobs. | `routes/api.py` (ACS-245) |
| `bad_json` | 400 | Client | Request body wasn't valid JSON. **In `api_requests`** — recorded via `_error` once the caller has authenticated (`routes/api.py:780`). | `routes/api.py` |
| `context_length_exceeded` | 400 | Client | Caught pre-flight by the wrapper. | `services/completions.py` |
| `chat_completions_unsupported` | 400 | Client | Someone called `/v1/chat/completions`. **Not in `api_requests`** — the 400 is returned directly from the route handler before any DB row is written; it shows up in Railway logs / access logs only. | `routes/api.py` / `main.py` |
| `invalid_api_key` | 401 | Client | Missing / wrong / revoked key. Rejected in `auth.py` **before** a request row exists, so it's **not necessarily an `api_requests` row** — it now emits a structured log line (being improved, ACS-…); look in Railway logs, not Metabase, for these. | `auth.py` |
| `account_not_approved` | 401 | Client | Key is live but its owner's account status isn't `approved` — e.g. a previously-approved user who was rejected (ACS-212). Structured log only, **not in `api_requests`** (same auth-path family as `invalid_api_key`). | `auth.py` |
| `account_suspended` | 401 | Client | Narrower sibling of `account_not_approved`: the owner's account is specifically `suspended` — a reversible, usually time-boxed park (ACS-353), not a decision about them. An admin unsuspends from `/admin/users/{id}`; access resumes on the next request (there is no auth cache). Structured log only, **not in `api_requests`**. | `auth.py` |
| `internal_error` | 500 | **Real (wrapper bug)** | Unhandled exception. **Not in `api_requests`** — Sentry-only (the catch-all in `main.py` returns without writing a row). | `main.py` |

**Where rows land.** Anything recorded through `_record_request` / `_error` (after auth, including `bad_json`, `invalid_request`, `budget_exceeded`, `context_length_exceeded`, the `cold_boot`/`upstream_*`/`vllm_*` family) gets an `api_requests` row. Successful held-open cold starts are normal success rows with `cold_boot=true`; late cold-start failures use `error_kind=cold_boot`. Errors raised **before** a request row exists — `invalid_api_key` (auth rejection), `chat_completions_unsupported` (route-level 400) — and `internal_error` (catch-all) do **not** land in `api_requests`; chase those in Railway logs / Sentry instead.

**Alarm only on the Real rows.** Isolated `cold_boot=true` successes, `budget_exceeded`, `queue_full`, and `rate_limited` are normal;
`circuit_open` means "go look at what the breaker was protecting against."

## Per-kind: where to look, what to do

### `upstream_unreachable` (502) — the most common real one
The Modal app for that model wasn't reachable. Check, in order:
1. **Modal dashboard** for that model's app (e.g. `acs-llama-405b`): is it deployed? containers crash-looping? stuck booting? The 405B is the usual suspect (big, slow, expensive to boot).
2. **Modal logs** for the app around the timestamp — boot failures, image pull errors, GPU unavailable.
3. **Railway logs** (wrapper): structured `upstream_error` / `error_kind=upstream_unreachable` lines carry the `request_id`.
4. If it was a brief blip (deploy, scale event) and recovered, no action beyond noting it. If sustained → the model is down; redeploy / fix boot.

### `circuit_open` (503) — derived
The breaker opened because `upstream_unreachable`/`vllm_*` crossed the failure
threshold (`breaker.py`). **Find the matching real errors in the same hour +
model** (they'll be right next to it in the drill-down). Fix those; the breaker
closes on the next success. Only touch breaker thresholds if it's opening too
eagerly on transient blips.

### `vllm_oom` (502)
Model OOMed on the GPU. Check **Modal logs** (vLLM stderr) for the CUDA OOM.
Likely causes: GPU shape too small for the model, `max_model_len` too high, or a
huge prompt/batch. Fix in `serving/modal_config` (GPU shape / max-len) or via
request limits.

### `vllm_engine_dead` (502)
vLLM engine/worker crashed. **Modal logs** for the stack trace; usually needs a
container restart / redeploy. Watch for a crash-loop (repeated across the hour).

### `cold_boot` (late JSON/SSE error)
The wrapper now keeps one request open through normal scale-from-zero and logs a
successful cold request with `cold_boot=true`. `error_kind=cold_boot` means the
model missed the 14-minute safety deadline or exhausted Modal continuations.
Repeated rows indicate an allocation/startup incident; isolated successful
`cold_boot=true` rows are expected usage.

### Expected client 429s / `invalid_api_key` (401)
For `budget_exceeded`, if the user is legit, raise their cap in
`/admin/users/{id}`. For `queue_full` or `rate_limited`, have them honor
`Retry-After` and add client-side concurrency/rate backpressure. For
`invalid_api_key`, the caller is using a wrong/revoked key — point them at
`/dashboard`.

### `internal_error` (500) — Sentry only
A real wrapper bug, and it **won't show in Metabase**. Go to **Sentry**
(`environment:production`, filter the date): each event has the exception, stack
trace (PII/body-scrubbed), endpoint, and `request_id`. Fix the code path. If you
saw the count in Sentry but not in `api_requests`, that's expected — they're
disjoint sources.

## Cross-referencing Sentry ↔ the DB
Every request carries a `request_id` (surfaced as `X-Request-Id`, ACS-41). It's
in the structured Railway logs and in Sentry events, so you can pivot from a
Metabase row → Railway log line → Sentry event for the same request.

## Quick decision tree
1. Break the spike down by `error_kind` (query above).
2. Mostly `cold_boot=true` successes / isolated `cold_boot` late errors / `budget_exceeded`? → not an incident unless repeated cold-start failures cluster by model.
3. `circuit_open` present? → find the `upstream_*`/`vllm_*` in the same window; that's the root cause.
4. `upstream_unreachable`/`vllm_*`? → Modal dashboard + Modal logs for that model's app.
5. Count in Sentry but not in the DB? → unhandled wrapper bug (`internal_error`); fix from the Sentry stack trace.
6. Same `key_id` as your test key / `/admin/debug/boom`? → self-inflicted, ignore.
