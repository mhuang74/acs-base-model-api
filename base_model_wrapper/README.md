# acs-base-model-wrapper

Auth + key-management wrapper in front of the Modal/vLLM base-model endpoint.

Plan: [`wiki/common-projects/base-model-hosting/wrapper-implementation-plan.md`](../../wiki/wiki-content/common-projects/base-model-hosting/wrapper-implementation-plan.md).

## What it does

```
client (curl / openai SDK / python)        Authorization: Bearer acs-bm-<prefix>-<secret>
        │
        ▼
[Railway] FastAPI wrapper  →  Postgres (users, api_keys, api_requests, usage_monthly)
        │                       structured JSON log to stdout (no prompt bodies)
        │
        │   Authorization: Bearer $VLLM_API_KEY (server-side only)
        ▼
[Modal] vLLM /v1/completions (Llama-3.1-405B, 8×H200)
```

Surface:

- `GET  /v1/models` — registry of live models with capability flags. Auth required.
- `POST /v1/completions` — validated request body, pre-flight budget clamp, SSE streaming supported.
- `POST /v1/chat/completions` — **400 by design** (base-model API; use `/v1/completions`).
- `GET  /health` — wrapper health, no auth.
- `POST /admin/users`, `POST /admin/keys`, `GET /admin/keys`, `POST /admin/keys/{id}/revoke` — admin only, `X-Admin-Token` header.

## Base-model API contract

This wrapper serves **base** (foundation) models, not chat-tuned models. The
contract is intentionally tight so reproducibility doesn't depend on what
sampler defaults vLLM happens to ship that week.

### Raw-text prompts (no chat templating)

`/v1/completions` accepts `prompt` as a raw string (or a list of raw strings
for batch). The wrapper does **not** apply a chat template, does **not** add
BOS/EOS or instruction wrappers, and does **not** insert role markers. What
you send is what the tokenizer sees. `/v1/chat/completions` returns a loud
400 to keep this contract obvious.

The `chat_template` field in `/v1/models` is always `null` to advertise this.

### Validated sampling parameters

Unknown fields and out-of-range values are rejected with an explicit 400
rather than silently dropped. Accepted parameters (all optional except
`prompt`):

| Param | Range | Notes |
|---|---|---|
| `prompt` | string or list of strings (required) | raw text; no templating |
| `model` | string | short id from `/v1/models`; HF repo id is rejected with a hint |
| `max_tokens` | int ≥ 1 | optional only for one prompt with `n=1`; omission is bounded to `min(32k, remaining context)` |
| `n` | int 1..16 | number of completions; `n>1` requires explicit `max_tokens` |
| `temperature` | float 0..2 | 0 = greedy |
| `top_p` | float (0, 1] | nucleus sampling |
| `top_k` | int (-1 disables, else ≥ 1) | |
| `min_p` | float [0, 1] | |
| `presence_penalty` | float -2..2 | |
| `frequency_penalty` | float -2..2 | |
| `repetition_penalty` | float (0, 2] | |
| `seed` | int ≥ 0 | for reproducibility |
| `stop` | string or list (≤ 4) | stop sequences |
| `logprobs` | int 0..100 | top-k logprobs per generated token |
| `prompt_logprobs` | int 0..100 | top-k logprobs per prompt token |
| `echo` | bool | include prompt in output |
| `stream` | bool | SSE response |
| `stream_options` | object | passed through to vLLM |
| `user` | string | client-side bookkeeping tag |

Anything else returns `400 {"error":{"code":"invalid_request", ...}}` with the
offending field name in the message.

### Logprobs & token IDs

Responses surface vLLM's native logprobs format unchanged, including `tokens`,
`token_ids`, `token_logprobs`, and `top_logprobs` per generated position, plus
`prompt_logprobs` for the prompt side when requested. The capability is
advertised per model in `/v1/models.capabilities`. `logprobs=-1` (full vocab)
is **not** supported — set a top-k in `1..20`.

### Token limits & overflow behavior

Three independent limits:

1. **Model context window** (`max_model_len`, per model — published in
   `/v1/models.capabilities.max_model_len`). The wrapper rejects
   `prompt_tokens + max_tokens > max_model_len` pre-flight with
   `400 {"error":{"code":"context_length_exceeded", ...}}` containing both
   numbers, before any GPU time is spent.
2. **Per-key budget** (`monthly`, `daily`, `input`, `output` — see
   `acs-keys create --help`). Exhausted budgets return `429
   {"error":{"code":"budget_exceeded", ...}}`. If the requested `max_tokens`
   exceeds the smallest remaining output-side budget, the wrapper clamps
   `max_tokens` down and surfaces the clamp via response header:

   ```
   X-Acs-Max-Tokens-Clamped: requested=500,applied=98,reason=budget
   ```

   Absence of the header means no clamping happened.
3. **Request work + admission**. Worst-case output work is capped at
   `prompt_count × n × max_tokens ≤ 32,000`; prompt lists and `n>1` require an
   explicit `max_tokens`. Each key gets 16 active requests, 64 queued waiters,
   and 600 request starts/minute. Excess queue/rate traffic returns structured
   retryable `429` responses with `Retry-After`.

### `/v1/models` shape

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-405b",
      "object": "model",
      "owned_by": "acs",
      "served_model_name": "meta-llama/Llama-3.1-405B",
      "gpu_shape": "8×H200",
      "status": "live",
      "capabilities": {
        "max_model_len": 32768,
        "max_logprobs": 100,
        "logprobs": true,
        "prompt_logprobs": true,
        "chat_template": null
      }
    }
  ]
}
```

The `capabilities` block is the discoverable contract — clients should read
`max_model_len` from here rather than hardcoding per-model values.

## Reliability surface

The wrapper is the boundary between clients and a stack with variable
behavior (Modal scale-to-zero, vLLM occasional OOM, Railway transients).
The following features make those failure modes legible to clients and
shield the system from cascading failures.

### Structured upstream errors

Every upstream failure is mapped to a structured error code. Clients can
key off `error.code` for programmatic handling; the tutorial page lists
the full set with HTTP status mappings.

| HTTP | `error.code` | When |
|---|---|---|
| 502 | `upstream_unreachable` | Wrapper couldn't reach upstream (DNS / connect / read timeout) after `RETRY_5XX_MAX_RETRIES` attempts |
| 502 | `vllm_oom` | Upstream returned 5xx with body matching out-of-memory pattern |
| 502 | `vllm_context_length` | Upstream returned 5xx with context-length pattern (rare; usually caught pre-flight) |
| 502 | `vllm_engine_dead` | Upstream returned 5xx with engine-crashed pattern |
| 502 | `upstream_server_error` | Other 5xx after retries |
| 200* | `modal_cold_boot` | Late JSON/SSE error after a cold boot exceeds the 14-minute safety deadline; normal cold boots finish in the original request |
| 503 | `circuit_open` | Per-backend circuit breaker tripped; includes `retry_after_seconds` |

Every response (success and error) also carries:

- `X-Acs-Upstream-Model` — short model id of the backend (e.g. `llama-405b`).
- `X-Acs-Upstream-Gpu` — GPU shape (e.g. `8×H200`).
- `X-Acs-Upstream-Error-Kind` — mirrors `error.code` on upstream-error 5xx responses (greppable in logs).

### Retry policy

The wrapper retries idempotently on transient upstream failures so a
single Modal-router blip doesn't surface to the caller:

| Trigger | Retries | Backoff |
|---|---|---|
| Modal cold-boot continuation | One POST, then same-invocation 303 result-URL GETs, capped at 14 min | JSON whitespace / SSE comment every 5s toward the caller |
| Network error (ConnectError, TimeoutException, …) | `RETRY_5XX_MAX_RETRIES = 3` | Exponential 1s, 2s, 4s + jitter up to 0.5s, cap 16s |
| Upstream 5xx | `RETRY_5XX_MAX_RETRIES = 3` | Same as network |
| Upstream 4xx | **0** (never) | — |

Public requests send keepalive bytes while the upstream call is pending. Modal's
150-second Web Function continuation redirects are followed as GETs to the result
URL; the wrapper never re-POSTs and never cancels a known cold start after 25/60s.

### Per-model timeout

Each `MODELS_REGISTRY_JSON` entry can override the global `upstream_timeout_s`
via an `upstream_timeout_s` field. Useful for tight bounds on small models
(an 8B should never sit on the wire for 20 minutes) and generous bounds for
405B / Kimi-K2 cold boots. Omit to inherit the global.

Known model ids from `acs_model_registry` can use concise wrapper entries that
only supply the deployed `upstream_url`; identity fields such as
`served_model_name`, `tokenizer_repo`, `gpu_shape_label`, `modal_app_name`, and
`max_model_len` are filled from the shared registry. Custom model ids still
need the full shape in `MODELS_REGISTRY_JSON`.

### Per-backend circuit breaker

`wrapper/breaker.py` tracks per-model upstream health in-memory. After
`FAILURE_THRESHOLD = 5` consecutive `upstream_unreachable` /
`upstream_server_error` failures, the breaker trips for `OPEN_DURATION_S = 60s`
and short-circuits requests for that model with `503 circuit_open`. After
the window it moves to half-open: the next request probes the upstream;
success closes the breaker, failure re-opens it.

**Cold-boot is not a failure** — Modal scale-to-zero is expected.

Admin endpoints:
- `GET /admin/breakers` — JSON snapshot of every model's breaker state + the active policy.
- `POST /admin/breakers/{model_id}/reset` — force-close after an incident is resolved.

### Aggregated `/health`

Public, no auth. Aggregates the wrapper's own readiness (DB ping) with
per-backend health derived from in-memory state (circuit breaker +
`last_completion_at`). No active probing of upstreams — those would either
trigger cold boots on idle 8×H200 containers or return 303 without
distinguishing healthy-but-cold from broken.

```
GET /health
→ 200 (or 503 if the wrapper itself is broken)
{
  "status": "ok" | "degraded" | "down",
  "wrapper": {
    "uptime_seconds": 123456,
    "database": {"ok": true, "latency_ms": 4}
  },
  "backends": [
    {
      "model_id": "llama-405b",
      "status": "live",
      "gpu_shape": "8×H200",
      "breaker_state": "closed",       // closed | open | half_open
      "breaker_consecutive_failures": 0,
      "breaker_last_failure_kind": null,
      "last_completion_at": "2026-06-03T11:55:00+00:00" | null,
      "last_completion_seconds_ago": 240 | null,
      "warm_estimate": "warm" | "cold"
    }
  ],
  "degraded_backends": []  // model_ids whose breakers are open
}
```

Top-level status semantics:
- `down` → HTTP 503. Wrapper-self failure (DB unreachable). Railway will restart on this.
- `degraded` → HTTP 200. One or more backends' breakers are open. Wrapper is fine; don't restart.
- `ok` → HTTP 200. All backends' breakers closed.

`warm_estimate` is a cheap derivation: `warm` when a completion came back
within that model's serving scaledown window (`scaledown_window_s`, per-model
in the registry — 30 min for most, shorter for some), else `cold`. It's a
hint, not authoritative — Modal's own runner state is the truth, but querying
that on every healthcheck would be wasteful.

The legacy `{"status": "ok"}` body is preserved as a subset of the new
shape — existing consumers don't break.

## Local dev

```bash
# 1. Postgres (one option: docker)
docker run -d --name acs-pg -p 5432:5432 \
  -e POSTGRES_USER=acs -e POSTGRES_PASSWORD=acs -e POSTGRES_DB=acs_wrapper postgres:16

# 2. Env
cp .env.example .env
export $(grep -v '^#' .env | xargs)
export DATABASE_URL=postgresql://acs:acs@localhost:5432/acs_wrapper

# 3. Install + migrate
pip install -e ".[dev]"
alembic upgrade head

# 4. Run
uvicorn wrapper.main:app --reload --port 8000
```

## Admin CLI

```bash
# Create a user + key (one shot)
acs-keys create alice@lab.org --create-user --name "rollout batch" --budget 100000000

# List + revoke
acs-keys list
acs-keys revoke <key_id>
```

`acs-keys create` prints the plaintext key **once**. Save it; there is no recovery path.

## Smoke test from a collaborator's POV

```bash
export ACS_API_BASE=https://<railway-url>/v1
export ACS_API_KEY=acs-bm-...

curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is", "max_tokens": 8}'
```

## Privacy commitment

The wrapper logs one structured JSON line per request:

```json
{"ts":"...","key_id":"...","key_prefix":"...","user_email":"...","ip":"...",
 "endpoint":"/v1/completions","model":"...","n_prompt":123,"n_completion":456,
 "status":200,"latency_ms":437,"upstream_latency_ms":401,"error_kind":null,
 "stream":false,"cold_boot":false,"ttft_ms":null,"workload_type":null}
```

It **never** logs request or response bodies (`prompt`, `completion`, `messages`,
`logprobs`, …). Enforced by funnelling every per-request log line through a
single typed helper in `wrapper/logging.py` that has no body-content parameters
(`tests/test_logging.py` asserts the signature stays body-free).

### Beta event logging

Each request also writes an `api_requests` row carrying **request-shape
metadata only** (never body content): `stream`, `cold_boot`, `ttft_ms`,
`workload_type`, requested `req_max_tokens`, `temperature`/`top_p`, and the
`logprobs_set`/`prompt_logprobs_set`/`seed_set`/`echo_set` adoption flags. These
power the beta usage-pattern dashboards (cold-start pain, interactive-vs-batch
mix, sampling-feature adoption). Clients may self-declare workload by sending an
optional `X-Acs-Workload: batch|interactive` request header; an unrecognized
value is ignored, never an error. Behavioral metrics (session length,
inter-request gap, invite→first-request, return-after-cold-start) are derived in
SQL from `(key_id, ts)` — no extra storage. Retention is intentionally not
pruned during the beta so the multi-week curves survive.

## Deployment

Railway-native. Push the repo + add Postgres plugin. Required env vars:

| Var | Required | Note |
|---|---|---|
| `DATABASE_URL` | yes | Railway sets this when Postgres is attached |
| `MODAL_BASE_URL` | yes | e.g. `https://<workspace>--acs-llama-405b-serve.example.modal.run` |
| `VLLM_API_KEY` | yes | Matches the `vllm-api` Modal Secret |
| `ADMIN_TOKEN` | yes | Long random |
| `HF_TOKEN` | yes (for gated tokenizer) | HF read token |
| `SERVED_MODEL_NAME` | no | Defaults to `meta-llama/Llama-3.1-405B` |
| `LOG_LEVEL` | no | Defaults to INFO |
| `LOG_IP` | no | `false` to disable per-request IP logging |
| `SENTRY_DSN` | no | Enables Sentry error tracking. Unset = no-op. |
| `SENTRY_ENVIRONMENT` | no | Defaults to `production` |
| `SENTRY_TRACES_SAMPLE_RATE` | no | 0–1, defaults to `1.0` (dial down at scale) |
| `SENTRY_ENABLE_LOGS` | no | Forward stdlib logs to Sentry; `false` to disable |

Migrations run automatically at boot via `entrypoint.sh`.

### Error tracking (Sentry)

Set `SENTRY_DSN` (prod only) to turn on error tracking; without it, Sentry init
is a no-op (local/dev/tests never phone home). Privacy is enforced in
`wrapper/observability.py` regardless of project settings: `send_default_pii=False`,
request bodies never captured (`max_request_body_size="never"`), and a
`before_send` scrub drops `user.email`/`ip_address` and recursively redacts
prompt-bearing keys — so **prompts/completions/logprobs never reach Sentry**.
Unhandled exceptions are caught by a global handler that returns a generic
`500 {"error":…,"request_id":…}` (no internals leaked) and reports the exception
to Sentry with the `request_id` for correlation.
