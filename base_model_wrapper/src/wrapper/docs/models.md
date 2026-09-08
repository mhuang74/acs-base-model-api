# Models

The wrapper currently fronts three base-model checkpoints. Pick `llama-8b` if you're just trying it out — it's kept always-on, so there's no cold-boot wait.

## Available models

<table>
  <thead>
    <tr><th><code>model</code></th><th>Checkpoint</th><th>Precision</th></tr>
  </thead>
  <tbody>
    <tr><td><code>llama-8b</code></td><td><code>meta-llama/Llama-3.1-8B</code></td><td>bf16</td></tr>
    <tr><td><code>llama-405b</code></td><td><code>meta-llama/Llama-3.1-405B</code></td><td>bf16</td></tr>
    <tr><td><code>trinity-truebase</code></td><td><code>arcee-ai/Trinity-Large-TrueBase</code></td><td>bf16</td></tr>
  </tbody>
</table>

The set above can change; **`GET /v1/models` is the source of truth** for what's actually live right now. Pass the short `model` id (e.g. `llama-8b`), never the HF repo name.

## What `GET /v1/models` returns

Each entry carries more than the checkpoint — query it programmatically rather than hard-coding numbers that drift:

- `id` — the short id you pass as `model`.
- `served_model_name` — the underlying checkpoint the backend serves.
- `gpu_shape` — the GPU shape backing this model (e.g. which accelerator / how many), useful for reasoning about cold-start cost and throughput.
- `status` — `"live"` for models you can call (only `live` models are listed).
- `capabilities`:
  - `max_model_len` — the context length: total `prompt + completion` tokens the model accepts. Read this instead of memorising a per-model window. (`null` means the registry hasn't declared it, so the pre-flight length check is skipped.)
  - `max_logprobs` — the cap on a **positive** (top-*k*) `logprobs` / `prompt_logprobs` count.
  - `logprobs`, `prompt_logprobs` — feature flags (both `true` today).
  - `prompt_logprobs_full_vocab` — whether `prompt_logprobs=-1` (the model's **whole** next-token distribution at each prompt position) is supported. Prompt-only — completion `logprobs` stays top-*k*.
  - `full_vocab_max_prompt_tokens` — max prompt length for a full-vocab `prompt_logprobs=-1` request. The full distribution is ~`vocab_size` values per position, so longer prompts must use a fixed top-*k*.
  - `n_layers` — the model's decoder-layer count (activation models only; the field is **absent**, not `null`, when a model doesn't declare it, so read it with a `.get()`). Two uses: layer indices run `0 … n_layers-1`, and it's what lets you compute your own capture cap when you request a subset of layers — see [Activation harvesting](/tutorial/activation-harvesting). The two paths differ on an out-of-range index: a steering `layer_index` outside the range is rejected up front with a `400`, while out-of-range **capture** indices are quietly dropped by the engine and you get fewer layers back than you asked for — so check the shape you received.
  - `chat_template` — always `null`: these are base models, prompts pass through verbatim with no templating.

## Availability & cold starts

Models that get steady use are usually kept warm, so requests start immediately. Less-used or larger models sleep when idle to save GPU time and cold-start on the first request after a quiet spell — usually a few minutes, worst case up to ~10 minutes (occasionally longer for the largest models when GPUs are scarce). Once warm, a model answers in seconds and stays warm as long as you keep using it.

The wrapper holds the first request open and sends keepalive bytes while the model starts, so callers should set a timeout of at least 15 minutes and wait for that request to finish. See [Cold-boot waiting](/tutorial/examples/cold-boot). For interactive work, send one short request first to warm the model before a session.

## Always-on (warm windows)

If you'd rather have a model kept always-on for a stretch — a work session, a deadline — tell us via the **Feedback** button or [email](mailto:infra@acsresearch.org), and we'll schedule a warm window.
