# Bulk harvest

Harvest **residual-stream activations over a whole prompt corpus** as downloadable
files — the batch counterpart to [inline harvesting](/tutorial/activation-harvesting),
for when you want thousands to millions of prompts (SAE training, large probing
datasets) rather than a few activations in an HTTP response.

You submit a job, poll it, and download `safetensors` shards by URL. Same API key
as everything else. The GPU work runs on a dedicated batch app per model; nothing
you do here slows the completions engines.

## Submit a job

`POST {{API_BASE}}/harvest` with your prompts:

```bash
curl -s "$ACS_API_BASE/harvest" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-8b",
    "prompts": ["The quick brown fox jumps over the lazy dog.",
                "Interpretability research reads the residual stream."],
    "layers": [8, 16, 24]
  }'
```

```jsonc
{ "job_id": "071fa547-…", "status": "running", "model": "llama-8b", "n_prompts": 2 }
```

(Responses below are abridged — a real one carries a few more bookkeeping keys
like `run_id` and `created_at`.)

Fields beyond `model` and `prompts` (a typo'd field name returns a `400` rather
than silently using a default on a paid GPU job):

- **`layers`** — which decoder-block outputs to keep, e.g. `[8, 16, 24]`. Omit
  it for the default quartile subset (blocks at ~25/50/75% depth — `[8, 16, 24]`
  on llama-8b); pass the string `"all"` (lowercase, exact) to keep every block.
  Unlike the inline endpoint, the harvester subsets **server-side**, so the
  3-layer default on a 32-layer model writes ~10× less data than `"all"`.
  Same convention as inline: layer `k` is the output of block `k`.
- **`shard_size`** — prompts per output file (default 32).
- **`batch_size`** — forward-pass batch inside the job. Leave it unset: the
  default vLLM-Lens backend batches continuously on its own; the value only
  shapes the HF reference path (`HARVEST_USE_VLLM=0`).

## Poll it

`GET {{API_BASE}}/harvest/<job_id>` with the same key. `status` moves from
`running` to `done` (or `failed`, with a short error message):

```bash
curl -s "$ACS_API_BASE/harvest/071fa547-…" -H "Authorization: Bearer $ACS_API_KEY"
```

```jsonc
{
  "job_id": "071fa547-…",
  "status": "done",
  "model": "llama-8b",
  "n_prompts": 2,
  "result": {
    "n_shards": 1,
    "layer_indices": [8, 16, 24],
    "manifest_url": "https://…/manifest.json?…",      // presigned — no key needed
    "shard_urls":   ["https://…/shard_00000.safetensors?…"],
    "stats_url":    "https://…/stats.safetensors?…",
    "timings": { "forward_s": 1.59, "upload_s": 4.05, "tokens": 21, /* … */ }
  },
  "urls_expire_at": "2026-07-24T20:31:37+00:00"
}
```

A small llama-8b job goes submit → `done` in about a minute, most of it engine
start-up. Poll every 20–30 s; there's no webhook.

## Cancel a job

`DELETE {{API_BASE}}/harvest/<job_id>` with the same key cancels a job that is
still `pending` or `running`. Because your key allows only so many jobs at once
(see the concurrency lanes below), a slow or stuck job otherwise blocks the slot
for its whole run — cancelling frees the slot immediately so your next submit
goes through, and asks the GPU container to stop.

```bash
curl -s -X DELETE "$ACS_API_BASE/harvest/071fa547-…" \
  -H "Authorization: Bearer $ACS_API_KEY"
```

```jsonc
{
  "job_id": "071fa547-…",
  "status": "cancelled",
  "model": "llama-8b",
  "n_prompts": 2,
  "created_at": "2026-08-07T20:14:00+00:00",
  "completed_at": "2026-08-07T20:14:12+00:00"
}
```

- `404` — no such job for your key (unknown id, or someone else's job).
- `409 harvest_not_cancellable` — the job already finished (`done`/`failed`) or
  was already cancelled; there's nothing to cancel.

Cancellation frees the slot the moment the `200` returns. Stopping the container
is best-effort: if the wrapper can't reach Modal, the job still shows
`cancelled` and the slot is free, but that container may keep running until it
finishes on its own or hits its timeout.

## Download and read the shards

The URLs are presigned and work in a plain browser, `curl`, or `wget` — no API
key. They expire at `urls_expire_at` (7 days after completion; the response adds
`"urls_expired": true` once they have).

Each shard holds two tensors per prompt — the activations and the **input token
ids**, so you can line activations up with tokens without re-tokenizing:

```python
import os, requests
from safetensors.torch import load_file

API_BASE = os.environ["ACS_API_BASE"]           # e.g. https://…/v1
KEY = os.environ["ACS_API_KEY"]

job = requests.get(f"{API_BASE}/harvest/{job_id}",
                   headers={"Authorization": f"Bearer {KEY}"}).json()

open("shard0.safetensors", "wb").write(requests.get(job["result"]["shard_urls"][0]).content)
shard = load_file("shard0.safetensors")

shard["prompt_0"]   # bf16, [n_layers_kept, n_tokens, hidden] — e.g. [3, 11, 4096]
shard["tokens_0"]   # int32, [n_tokens] — same order as the tensor's token axis
```

> **`tokens_i` is a stable part of the format.** It's what lets you verify
> alignment against your own tokenization rather than trusting ours, and it's
> how the double-BOS trap below gets caught. We won't remove it.

### Watch out: the engine adds BOS to text prompts

Harvest tokenizes each prompt with `add_special_tokens=True`, so it prepends the
model's BOS token (Llama: `<|begin_of_text|>`). If your text **already** starts
with one — which it does whenever you render a chat template client-side — you
get **two**, and every position in your analysis is off by one against your own
tokenization. Nothing errors; the numbers just quietly refer to the wrong tokens.

- **Detect it.** Compare `shard["tokens_i"]` against your client-side render
  before you compare a single activation. A length mismatch of exactly one, with
  a duplicated first id, is this bug. Make this the first assertion in any
  parity harness — it is the cheapest check that catches the widest class of
  alignment mistakes, including ones on your side.
- **Fix it.** Set `add_special_tokens: false` in the request. Harvest then
  tokenizes your text verbatim and adds no BOS, so a prompt that already carries
  one ends up with exactly one. This is the clean fix — `/v1/harvest` takes text
  only (`prompts` is a list of strings), so unlike `/v1/completions` there is no
  pre-tokenized escape hatch here. (Default stays `true`: plain text prompts
  still get their BOS.)

`manifest_url` points to a JSON index of every shard (which prompts are in which
file, shapes, the layer convention spelled out). `stats_url` is a small
safetensors file with per-layer token `mean` and `std` over the whole run — the
normalization statistics SAE training wants, with each prompt's BOS token left
out of the statistics because its outlier norms would skew them.

## Get projections instead of activations

If you already know which directions you care about — probe vectors, SAE
features, a trait axis — send them with the job and the shards come back holding
**per-token projections onto those directions** instead of the raw residual
stream:

```jsonc
{
  "model": "llama-8b",
  "prompts": ["…"],
  "layers": [14],
  "project_onto": {                 // shape (n_directions, hidden)
    "data": "<base64>",             // raw little-endian buffer, no compression
    "dtype": "float32",             // or float16 / bfloat16
    "shape": [8, 4096],
    "compression": "none"
  }
}
```

`prompt_i` then has shape `[n_layers_kept, n_tokens, n_directions]` in float32,
and `tokens_i` is unchanged. For 8 directions on llama-8b that's **512× less
data** than the raw stream — the difference between downloading gigabytes and
downloading megabytes for the same corpus.

Details worth knowing:

- **Directions are L2-normalized server-side**, so values are components along
  each unit direction. Projecting onto a non-unit vector would silently rescale
  every number, and you couldn't see it in the output. The manifest records
  `projection.normalized: true`.
- The manifest's **`contents`** field reads `"projections"` (raw runs say
  `"activations"`), so a shard can't be mistaken for the other kind. Its
  **`dtype`** reads `"float32"` accordingly — projections are always float32,
  whatever dtype you sent the directions in.
- **`stats.safetensors` and `mean_token_norm` describe the projections**, not
  the residual stream: `mean`/`std` are `[n_layers_kept, n_directions]` and
  `mean_token_norm` is the mean L2 norm of the projection vectors. The manifest's
  `statistics` string says which one you have.
- Directions must have the model's `hidden` size — a mismatch is rejected
  *before* the model loads, so a wrong shape costs seconds rather than a long
  GPU load.
- Up to 64 directions per job. The buffer must be exactly
  `n_directions × hidden × itemsize` bytes — a truncated or mis-shaped payload
  is rejected at submit time with the byte counts in the message, not hours
  later on a GPU.
- Projection happens **on the GPU, before the activations are copied to host
  memory**, so the reduction is what makes long-context and `layers: "all"`
  jobs affordable rather than just making the download smaller.

Raw-activation harvests are unchanged and remain the default; omit
`project_onto` and nothing about your pipeline changes.

## Limits & practicalities

- **Concurrent jobs per key, in two lanes.** Small single-GPU models (llama-8b)
  allow **3** jobs at once; the large multi-GPU models (llama-405b,
  trinity-truebase) allow **1**, because each one occupies a whole 8×H200
  container. The lanes are counted separately, so a running 405B job never
  blocks a quick 8B one — you can keep an interactive session going while a
  corpus run grinds away. Over the limit returns
  `429 harvest_concurrency_exceeded`, and the message names the lane you filled.
  That 429 carries a `Retry-After: 30` header, so back off ~30 s between retries
  rather than hammering — or `DELETE` the in-flight job (see above) to free the
  slot now. Your key may also carry a monthly job quota
  (`429 harvest_quota_exceeded`).
- **Corpus caps.** Up to 4,096 prompts and ~2M estimated tokens per job, and
  request bodies up to 50 MB (`413` beyond that). For more, split into several
  jobs.
- **Prompts longer than the model's context window are truncated** to fit, and
  the token axis you get back matches what actually ran.
- **Big models are slow to start.** llama-405b streams ~810 GB of weights on a
  cold start — the first job can spend ~50 minutes loading before any prompt
  runs. The load is paid once per job, so batch your corpus into one large job
  instead of many small ones.
- **Waiting for a job: add `?wait=`.** `GET /v1/harvest/<job_id>?wait=30` holds
  the request open for up to 30 seconds (60 max) and returns the moment the job
  reaches a terminal state — the same payload an immediate poll gives you, just
  without the polling loop. It is optional in every sense: omit it and nothing
  changes. Jobs routinely outlive the window, so keep your loop, just make it
  one request per minute instead of dozens.
  - **It can also return early *without* a terminal state.** To protect
    `/v1/completions` latency, only so many long-polls may wait at once (4 per
    key, 8 overall). Over that ceiling your `?wait=` is answered immediately with
    the current (possibly still-`running`) state, an `X-Acs-Longpoll: declined`
    header and a `Retry-After` — a normal `200`, not an error. Just poll again
    after `Retry-After`; keeping a plain one-request-per-minute loop handles this
    transparently.
- **`503 harvest_unavailable`** means the harvest app for that model isn't
  reachable right now; the response records a job id you can show us.

## Inline or bulk?

Inline (`output_residual_stream`) answers in the same HTTP response and is right
for probing and steering workflows up to ~10k examples. Bulk trades latency for
throughput: the capture itself runs ~13× faster than the inline path's transport
ceiling, keeps only the layers you ask for, and hands you files you can re-download
for a week.

## See also

- [Activation harvesting](/tutorial/activation-harvesting) — the inline path.
- [Activation steering](/tutorial/activation-steering) — steer with vectors you
  build from harvested activations.
