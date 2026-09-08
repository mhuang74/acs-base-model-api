# API reference

OpenAI-compatible `POST /v1/completions` for raw next-token generation against a base model. No chat endpoint — see [What's not supported](#whats-not-supported).

## Endpoint

**`POST {{API_BASE}}/completions`**

Sends a prompt to a base model and returns one or more continuations. The request body follows the OpenAI Completions shape with a handful of vLLM-native extras (`top_k`, `min_p`, `repetition_penalty`, `prompt_logprobs`); responses pass through verbatim from vLLM.

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is", "max_tokens": 8}'
```

## Using the OpenAI Python SDK

Five params are vLLM-native and aren't on the SDK's typed signature, so the SDK rejects them as keyword arguments *before the request leaves your machine*: **`top_k`, `min_p`, `repetition_penalty`, `prompt_logprobs`, `add_special_tokens`**. Pass them through `extra_body={...}`. Everything else — `temperature`, `top_p`, `max_tokens`, `n`, the penalties, `seed`, `logprobs`, `echo`, `stream`, `stop`, `user` — works as a normal keyword argument.

```python
resp = client.completions.create(
    model="llama-8b",
    prompt="The capital of France is",
    max_tokens=1,
    temperature=0.7,            # standard kwarg
    logprobs=5,                 # standard kwarg
    extra_body={                # vLLM-native — must go here
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "prompt_logprobs": 5,
    },
)
```

> **`extra_body` is an SDK construct, not a wire field.** The SDK merges everything under `extra_body` into the top-level request JSON before it goes over the wire. Over **raw HTTP** (curl, `requests`, etc.) there's no SDK to do that merge — put these parameters at the **top level** of the JSON body alongside `model` and `prompt`. Sending a literal `{"extra_body": {...}}` over raw HTTP gets a `400 Unknown field 'extra_body'`.

```bash
# Raw HTTP: top_k / min_p / repetition_penalty / prompt_logprobs are just top-level fields.
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is",
       "max_tokens": 1, "temperature": 0.7, "logprobs": 5,
       "top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1, "prompt_logprobs": 5}'
```

## Strict validation

Unknown fields are rejected with a `400`, and out-of-range values fail loudly instead of silently clamping — `temperature=200` is a `400`, not a quiet reset to the default.

Every one of these constraints is published as a machine-readable OpenAPI spec at [`{{SITE_BASE}}/openapi.json`]({{SITE_BASE}}/openapi.json) (browsable UI at [`{{SITE_BASE}}/docs`]({{SITE_BASE}}/docs)). It covers the public API surface — the `/v1/*` endpoints plus `/health` — and lists the exact bounds — `seed` minimum `0`, `temperature` `0`–`100`, `top_p` in `(0, 1]`, `stop` at most 4 items, and so on — so you can check a value against the schema instead of discovering the limit from a `400`.

## Request body

**Routing and prompt**

### model

**Type:** `string` &nbsp;·&nbsp; **Default:** none

Short id from [`GET /v1/models`](/tutorial/models#what-get-v1models-returns) — e.g. `llama-8b`, `llama-405b`. Use the short id, **not** the HF repo name. Unknown ids return `400 model_not_found`.

### prompt

**Type:** `string`, `string[]`, `int[]` (token ids), or `int[][]` (batch of token-id prompts) &nbsp;·&nbsp; **Required**

Raw text — no chat template is applied (these are base models, so the prompt passes through verbatim and the model continues it). A `string[]` is a batch: one completion per prompt, subject to the output-work cap below.

**Pre-tokenized prompts.** You can pass token ids directly instead of text — `int[]` for one prompt, `int[][]` for a batch. Ids are forwarded to the model **verbatim**, with no detokenize→retokenize round-trip (which on a non-bijective tokenizer can quietly change the token sequence). Use this when the prompt must be byte-for-byte exact — interpretability / eval work, or replaying an already-tokenized dataset. `usage.prompt_tokens` for a token-id prompt is just the number of ids. Mixed lists (e.g. `[1, "two"]`) and empty prompts return `400`.

```python
# OpenAI SDK — prompt takes token ids directly (no extra_body needed)
resp = client.completions.create(model="llama-8b", prompt=[1, 2, 3, 4, 5], max_tokens=8)
```

**Length and stop**

### max_tokens

**Type:** `int`, `1 ≤ n ≤ 1_000_000` (schema bound) &nbsp;·&nbsp; **Default:** auto-injected for a single prompt with `n=1`; required otherwise

Maximum tokens to generate **per completion**. A single prompt with `n=1` may omit it — the wrapper injects `min(32_000, remaining model context)` rather than letting vLLM fill an unbounded context. Batched prompts or `n > 1` **must** set it explicitly because the total worst-case output work is capped:

> **`prompt_count × n × max_tokens ≤ 32_000`** &nbsp;—&nbsp; exceed this and the request returns `400 invalid_request`.
>
> `prompt_count` is the number of prompts the request carries — **not** a token count: `1` for a single string (or a single pre-tokenized `list[int]` prompt), the list length for a batch (`list[str]` or `list[list[int]]`).

`max_tokens=0` is **not** supported. To score existing text rather than generate, use `prompt_logprobs` with `max_tokens=1` and `echo=true`; you get the prompt-position logprobs and ignore the one generated token. See the [prompt_logprobs](/tutorial/examples/prompt-logprobs) and [echo](/tutorial/examples/echo) examples.

The wrapper may further reduce `max_tokens` to fit a per-key budget; when that happens the response carries an `X-Acs-Max-Tokens-Clamped` header.

### n

**Type:** `int`, `1 ≤ n ≤ 16` &nbsp;·&nbsp; **Default:** `1`

Number of independently-sampled completions per prompt. Counts against the `prompt_count × n × max_tokens` cap above.

### stop

**Type:** `string` or `string[]` (up to 4 entries) &nbsp;·&nbsp; **Default:** none

Strings that, when produced, stop sampling. Matched stop content is **not** included in the response. Lists longer than 4 return `400 invalid_request`.

**Sampling**

### temperature

**Type:** `float`, `0.0 ≤ x ≤ 100` &nbsp;·&nbsp; **Default:** `1.0`

Sampling temperature. `0.0` is greedy decoding (top-1 at every step) and is fully deterministic regardless of `seed`. Higher values flatten the distribution toward uniform; the effect scales as ~1/T, so it's already ~90% of the way to uniform by `10` and negligible beyond. (The Workbench slider stops at `5` — the common range — but you can pass higher here.) Pair a high temperature with `top_p` / `top_k` / `min_p` to keep the flattened sampling inside a plausible set.

### top_p

**Type:** `float`, `0.0 < x ≤ 1.0` &nbsp;·&nbsp; **Default:** `1.0`

Nucleus sampling threshold — keep the smallest set of tokens whose cumulative probability reaches `top_p`. `1.0` disables. Must be strictly positive (`0.0` returns `400`).

### top_k

**Type:** `int`, `-1` (disabled) or `n ≥ 1` &nbsp;·&nbsp; **Default:** `-1`

Keep only the top-*k* tokens at each step. `-1` disables.

> **vLLM-native:** with the OpenAI Python SDK, pass via `extra_body={"top_k": ...}`.

### min_p

**Type:** `float`, `0.0 ≤ x ≤ 1.0` &nbsp;·&nbsp; **Default:** `0.0`

Minimum probability — relative to the top token — a token must have to be sampled. `0.0` disables.

> **vLLM-native:** with the OpenAI Python SDK, pass via `extra_body={"min_p": ...}`.

**Penalties**

### presence_penalty

**Type:** `float`, `-2.0 ≤ x ≤ 2.0` &nbsp;·&nbsp; **Default:** `0.0`

Penalty applied once a token has appeared anywhere in the generation so far. Positive values push the model away from repeating any token; negative encourage it.

### frequency_penalty

**Type:** `float`, `-2.0 ≤ x ≤ 2.0` &nbsp;·&nbsp; **Default:** `0.0`

Like `presence_penalty`, but scaled by how often the token has already appeared.

### repetition_penalty

**Type:** `float`, `0.0 < x ≤ 2.0` &nbsp;·&nbsp; **Default:** `1.0`

Multiplicative repetition penalty (vLLM-native, distinct from the additive presence / frequency penalties). `1.0` disables; values above `1.0` discourage repeats; values below `1.0` encourage them. Must be strictly positive.

> **vLLM-native:** with the OpenAI Python SDK, pass via `extra_body={"repetition_penalty": ...}`.

**Determinism**

### seed

**Type:** `int`, `n ≥ 0` &nbsp;·&nbsp; **Default:** none (random)

PRNG seed for sampling. Honored — a fixed seed with identical params on the same underlying checkpoint gives a reproducible draw. Greedy decoding (`temperature=0`) is deterministic without `seed`.

**Token-level inspection**

### logprobs

**Type:** `int`, `0 ≤ n ≤ 100` &nbsp;·&nbsp; **Default:** none

Top-*k* logprobs **per generated token**. `0` / unset disables. The per-model cap is exposed in `GET /v1/models` as `capabilities.max_logprobs` — read that rather than hard-coding `100`. See the [logprobs example](/tutorial/examples/logprobs).

### prompt_logprobs

**Type:** `int`, `n = -1` or `0 ≤ n ≤ 100` &nbsp;·&nbsp; **Default:** none

Top-*k* logprobs at each **prompt** position — the model's predictions over the prompt, not the actual prompt tokens unless they happened to be in the top *k*. Useful for likelihood / surprisal / interpretability work. See the [prompt_logprobs example](/tutorial/examples/prompt-logprobs).

Pass **`-1` for the full vocabulary** — the model's entire next-token distribution at every prompt position, not just the top *k*. This is prompt-only (completion `logprobs` can't be `-1`) and, because the payload is ~`vocab_size` values *per token* (tens of MB), prompt length is capped — see `capabilities.full_vocab_max_prompt_tokens` in `GET /v1/models`, currently **1024 tokens** for every model; a longer full-vocab prompt returns `400 full_vocab_prompt_too_long`. For longer prompts, use a fixed top-*k*. Send `Accept-Encoding: gzip` (most HTTP clients do by default) — the wrapper compresses the response, which shrinks these logprobs bodies several-fold. Expect a large download either way, and note the response streams: the first bytes can take a while on long prompts while the model server serializes the body.

> **vLLM-native:** with the OpenAI Python SDK, pass via `extra_body={"prompt_logprobs": ...}`.

### echo

**Type:** `bool` &nbsp;·&nbsp; **Default:** `false`

If `true`, the prompt is prepended to `choices[i].text` (no separator). See the [echo example](/tutorial/examples/echo).

**Streaming**

### stream

**Type:** `bool` &nbsp;·&nbsp; **Default:** `false`

If `true`, the response is `text/event-stream` (SSE) with one chunk per token, terminated by `data: [DONE]`. See the [stream example](/tutorial/examples/stream).

### stream_options

**Type:** `object` &nbsp;·&nbsp; **Default:** none

Optional streaming flags. Currently honored:

- `include_usage: bool` — emit a terminal chunk with `choices: []` and a populated `usage` object **before** `[DONE]`. Without this, streamed responses skip the usage frame.

**Bookkeeping**

### user

**Type:** `string` &nbsp;·&nbsp; **Default:** none

Free-form tag echoed back for your own bookkeeping (matches the OpenAI shape). Telemetry only — does not affect routing, scheduling, or pricing.

### add_special_tokens

**Type:** `boolean` &nbsp;·&nbsp; **Default:** `true`

Whether the tokenizer prepends the model's special tokens — for these base models, the BOS (`<|begin_of_text|>`). Leave it `true` (the default) for plain text prompts.

Set it to `false` when your prompt text **already** starts with a BOS — most commonly when you render a chat template client-side, which emits `<|begin_of_text|>` itself. With the default `true`, the engine adds a **second** BOS, and every token position shifts by one: logprobs, `echo` offsets, and any activation capture no longer line up with the tokens you think you sent. Passing `add_special_tokens=false` tokenizes your text verbatim so exactly one BOS is present.

This is a vLLM-native field (not on the OpenAI SDK's typed signature), so pass it via `extra_body={...}` from the SDK, or as a top-level field over raw HTTP.

## What's not supported

- **No chat endpoint.** `POST /v1/chat/completions` returns `400 chat_completions_unsupported` — these are base models, no chat template. The `openai` SDK defaults to chat, so call `client.completions.create(...)` (not `client.chat.completions.create(...)`).

## Returns

A `text_completion` object (JSON for non-streaming; SSE frames for `stream: true`):

```json
{
  "id": "cmpl-91ea94c6ed0e5b75",
  "object": "text_completion",
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": " a",
      "logprobs": {
        "text_offset": [0],
        "tokens": [" a"],
        "token_logprobs": [-1.7437],
        "top_logprobs": [
          {" a": -1.7437, " Paris": -1.9937, " one": -2.5562, " the": -2.5562, " also": -3.1187}
        ]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7}
}
```

**Top-level**

- `id` — unique per response. Distinct from the `X-Request-Id` response header (which is the wrapper's trace id — quote that one in bug reports).
- `object` — always `"text_completion"`.
- `model` — the **checkpoint** the backend served (e.g. `"meta-llama/Llama-3.1-8B"`), not the short id you passed. Use this for reproducibility; the short id is a routing alias and can re-point.
- `choices` — array of length `prompt_count × n`, ordered first by prompt then by fan-out index.
- `usage` — `prompt_tokens`, `completion_tokens`, `total_tokens`. Streaming responses include `usage` **only** if you set `stream_options.include_usage: true`.

**Per choice**

- `text` — the generated continuation (or prompt + continuation, if `echo: true`).
- `index` — the choice's position in the response.
- `finish_reason` — `"stop"` (a stop string matched), `"length"` (hit `max_tokens` or the model's context limit), or `null` mid-stream.
- `stop_reason` — the matched stop string when `finish_reason == "stop"`, else `null`. vLLM-specific; omitted from the abbreviated example above but present on the wire.
- `logprobs` — `null` unless you set `logprobs`. When present:
  - `tokens[]` — generated token strings.
  - `token_logprobs[]` — logprob of each generated token.
  - `top_logprobs[]` — array of `{token: logprob}` dicts, one per position; each dict's size is the `logprobs` value you asked for.
  - `text_offset[]` — byte offsets of each token in `text`.

When you set `prompt_logprobs`, vLLM surfaces it under `choices[i].prompt_logprobs` — see the [prompt_logprobs example](/tutorial/examples/prompt-logprobs) for the full shape.

> **Responses are passed through verbatim from vLLM.** The wrapper validates request bodies strictly but does not re-shape responses, so `logprobs`, `prompt_logprobs`, and token ids surface exactly as the engine produced them.

## Limits

- Each key has a **monthly token budget** (`prompt + completion` tokens, reset on the 1st, UTC). Exceed it and requests return `429 budget_exceeded`. [Email](mailto:infra@acsresearch.org) to raise it.
- Up to **16 active requests per key**; another **64 may wait server-side**. Beyond that, requests return retryable `429 queue_full` rather than consuming unbounded memory.
- Each key may start up to **600 requests/minute**. Sustained floods above that return `429 rate_limited`.
- A single request may ask for at most **32,000 worst-case output tokens**, calculated as `prompt_count × n × max_tokens` (`prompt_count` = prompts in the request: 1 for a single prompt, the list length for a batch). Batched prompts and `n > 1` must set `max_tokens` explicitly.

## Errors

Every error body follows the OpenAI shape: `{"error": {"code": "...", "message": "...", "type": "..."}}` with extra fields tagged where useful. Read `error.code` for programmatic handling.

<table>
  <thead>
    <tr><th>HTTP</th><th>Code</th><th>Meaning</th><th>What to do</th></tr>
  </thead>
  <tbody>
    <tr><td><code>400</code></td><td><code>invalid_request</code></td><td>Unknown field, bad type, or out-of-range sampling param (e.g. <code>temperature&gt;2</code>, <code>top_p&gt;1</code>, <code>logprobs&gt;100</code>)</td><td>The message names the offending field; fix and retry</td></tr>
    <tr><td><code>400</code></td><td><code>invalid_request</code></td><td><code>prompt_count × n × max_tokens &gt; 32,000</code>, or a fan-out request omitted <code>max_tokens</code></td><td>Reduce the prompt batch, <code>n</code>, or <code>max_tokens</code></td></tr>
    <tr><td><code>400</code></td><td><code>context_length_exceeded</code></td><td><code>prompt_tokens + max_tokens &gt; max_model_len</code></td><td>Reduce the prompt or <code>max_tokens</code>; check <code>/v1/models</code> for the per-model limit</td></tr>
    <tr><td><code>400</code></td><td><code>chat_completions_unsupported</code></td><td>You hit <code>/v1/chat/completions</code></td><td>Use <code>/v1/completions</code> — these are base models, no chat template</td></tr>
    <tr><td><code>400</code></td><td><code>model_not_found</code></td><td>Unknown <code>model</code> id</td><td>Use a short id from <code>/v1/models</code> (not the HF repo name)</td></tr>
    <tr><td><code>400</code></td><td><code>bad_json</code></td><td>Request body wasn't valid JSON</td><td>Fix the JSON</td></tr>
    <tr><td><code>401</code></td><td><code>invalid_api_key</code></td><td>Key missing, wrong, paused, or revoked</td><td>Check <code>ACS_API_KEY</code> in your account settings</td></tr>
    <tr><td><code>429</code></td><td><code>budget_exceeded</code></td><td>Monthly / daily / input / output token budget hit</td><td>Wait for the reset, or ask for more. See <a href="/tutorial/examples/budget-cap">Budget-cap recovery</a></td></tr>
    <tr><td><code>429</code></td><td><code>queue_full</code></td><td>This key already has 16 active + 64 queued requests</td><td>Honor <code>Retry-After</code>, reduce client fan-out, and use a client-side <code>Semaphore(16)</code></td></tr>
    <tr><td><code>429</code></td><td><code>rate_limited</code></td><td>This key exceeded 600 request starts/minute</td><td>Honor <code>Retry-After</code> and reduce sustained request rate</td></tr>
    <tr><td><code>502</code></td><td><code>upstream_unreachable</code></td><td>Wrapper couldn't reach the model server (DNS / connection / read timeout) after retries</td><td>Retry shortly; persistent failures are an outage — report it</td></tr>
    <tr><td><code>502</code></td><td><code>vllm_oom</code></td><td>Upstream model server ran out of GPU memory</td><td>Retry with a smaller prompt / <code>max_tokens</code> / lower <code>n</code></td></tr>
    <tr><td><code>400</code></td><td><code>vllm_context_length</code></td><td>Upstream enforced its context-length limit (rare — the wrapper usually catches this as <code>context_length_exceeded</code> first)</td><td>Reduce prompt / <code>max_tokens</code></td></tr>
    <tr><td><code>400</code></td><td><code>vllm_invalid_request</code></td><td>The model server rejected the request as invalid — e.g. a steering <code>layer_index</code> outside the model's range. The wrapper usually catches an out-of-range layer index first with <code>invalid_request</code>; this is the fallback when the model's layer count isn't known to the wrapper</td><td>Fix the request (for steering, use a <code>layer_index</code> in <code>0..n_layers-1</code>; see <code>GET /v1/models</code>)</td></tr>
    <tr><td><code>502</code></td><td><code>vllm_engine_dead</code></td><td>Upstream vLLM engine crashed</td><td>Retry; if it persists the model is down — check status or report</td></tr>
    <tr><td><code>502</code></td><td><code>upstream_server_error</code></td><td>Other upstream 5xx after retries</td><td>Retry; check <code>error.upstream_status</code> for the original code</td></tr>
    <tr><td><code>200*</code></td><td><code>modal_cold_boot</code></td><td>The model didn't become ready before the 14-minute safety deadline. The <code>200</code> status was already sent (keepalive bytes had committed it), so the error arrives in the response body, not as an HTTP error code.</td><td>Treat the JSON/SSE <code>error</code> payload as a failed request; report repeated allocation failures. Normal cold boots complete in the original request. See <a href="/tutorial/examples/cold-boot">Cold-boot waiting</a></td></tr>
    <tr><td><code>503</code></td><td><code>circuit_open</code></td><td>Backend is in a circuit-breaker open state after repeated failures</td><td>Use <code>retry_after_seconds</code>; if the model is critical, contact us</td></tr>
  </tbody>
</table>

## Request headers

- `Authorization: Bearer <ACS_API_KEY>` — required.
- `Content-Type: application/json` — for the JSON body.
- `X-Acs-Workload: interactive | batch` — **please set this on your requests**: `batch` for bulk / offline / eval jobs, `interactive` for live, latency-sensitive calls. It's telemetry only — it does **not** change scheduling or priority, and unrecognised values are ignored — but it's the main signal we use to understand usage and plan capacity / keep-warm policy, so setting it accurately genuinely helps us run the service well. See [Batch rollouts](/tutorial/examples/batch-rollouts).

## Response headers

Reliability hints on the response itself:

- `X-Request-Id` — unique trace id for the request; quote it in bug reports so we can find it in our logs.
- `X-Acs-Upstream-Model` / `X-Acs-Upstream-Gpu` — which backend served (or failed) this request. Cite this in bug reports.
- `X-Acs-Upstream-Error-Kind` — set on upstream-error 5xx responses (mirrors `error.code`).
- `X-Acs-Max-Tokens-Clamped` — present when the wrapper reduced your `max_tokens` to fit a budget; format `requested=N,applied=M,reason=budget`.
- `X-Acs-Longpoll: declined` — set on a `GET /v1/harvest/<id>?wait=` that was answered **early** (as a normal `200` with the current job state) because too many long-polls were already waiting. Not an error — just poll again; the accompanying `Retry-After` tells you when.
- `Retry-After` — RFC-7231 header on retryable `429` and `503` responses, and on an early-declined harvest long-poll (the `X-Acs-Longpoll: declined` `200` above).

## Privacy

API requests to `/v1/completions` are logged as **metadata only** — key prefix, email, IP, endpoint, model, token counts, status, latency — which we use to run the service and prevent abuse. We don't log your prompts or completions. Note this might change: we may start logging API requests for abuse-prevention reasons. We will not train on them. (Keep your own copies if you need them for reproducibility.)

The one exception is the Workbench: prompts you save there are stored server-side, so your sessions persist across browsers. Don't keep anything there you wouldn't want stored.

## Getting help

Email [infra@acsresearch.org](mailto:infra@acsresearch.org), or use the in-app **Feedback** button if you're signed in. To help us trace a specific request, include the rough time you made it, your key prefix or name (found in your dashboard — *not* the secret), and the error body.
