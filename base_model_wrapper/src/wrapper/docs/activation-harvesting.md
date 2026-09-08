# Activation harvesting

Get a model's **internal residual-stream activations** back alongside a normal
completion — the values flowing between the model's layers, for interpretability
and probing work (SAEs, linear probes, logit lens, …).

It's the same `POST {{API_BASE}}/completions` endpoint and the same API key as
ordinary completions — you just add one field. Activation requests run on a
**separate engine** per model, kept apart from ordinary completions. One
consequence for seeded work: a seed reproduces a draw *within* an engine, so
don't use a plain completion as a control for an activation-engine request —
see [Baselines on the steering page](/tutorial/activation-steering) for the
clean recipe.

## Turn it on

Add **`"output_residual_stream": true`** to the request body:

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-8b",
    "prompt": "The capital of France is",
    "max_tokens": 1,
    "temperature": 0,
    "output_residual_stream": true
  }'
```

`true` captures the residual stream at **every** decoder layer. If you only need
a few layers, pass a **list of layer indices** instead — the subset is applied on
the engine, so only those layers cross the wire (much smaller responses):

```jsonc
"output_residual_stream": [15, 20]   // capture only layers 15 and 20
```

`n_layers` in the returned `shape` is then the number of layers you asked for
(in ascending order), not the model's full count. Pass `true` for all layers.

## What comes back

The normal completion JSON, plus a top-level **`activations`** object:

```jsonc
{
  "choices": [ /* the normal completion */ ],
  "usage":   { /* the normal usage */ },
  "activations": {
    "residual_stream": {
      "data": "<base64 string>",       // zstd-compressed, int16-encoded bf16 bytes
      "dtype": "int16",                // storage dtype (bf16 reinterpreted as int16)
      "original_dtype": "torch.bfloat16",
      "shape": [n_layers, n_tokens, hidden],   // n_layers = full count for `true`, else how many you requested
      "compression": "zstd"
    }
  }
}
```

`shape[0]` is the model's **full** layer count (e.g. 32 for llama-8b, 126 for
llama-405b, 60 for trinity-truebase). Slice the layers you want after decoding.

## Which token positions are captured

`shape[1]` covers every token that went through a forward pass:

```
n_tokens = prompt tokens + generated tokens − 1
```

A 6-token prompt with `max_tokens: 5` returns 10 positions: rows 0–5 are the
prompt, rows 6–9 are the generated tokens that were fed back through the model.
The **final sampled token is not included** — it's drawn from the last forward
pass but never processed itself, so it has no residual stream. With
`max_tokens: 1` (the probing default, as in the example above) that reduces to
exactly the prompt positions.

So a single request captures most of the generation trajectory too, not just
the prompt — useful when you want activations *while the model writes*, at the
cost of a larger payload (the prompt-length cap below applies to the prompt).

## Decode it

`dtype: "int16"` holds the bfloat16 **bit pattern** stored as int16. Reinterpret
those bits back to bfloat16 with `.view(...)`, as the code below does — a numeric
`.astype()` would convert the integer values and give you garbage:

```python
import base64, numpy as np, ml_dtypes, zstandard as zstd

def decode_residual_stream(resp):
    d = resp["activations"]["residual_stream"]
    raw = base64.b64decode(d["data"])
    if d.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    # reinterpret the 16 bits as bfloat16 with .view (an .astype would corrupt them)
    arr = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    return arr.reshape(d["shape"])          # (n_layers, n_tokens, hidden)

x = decode_residual_stream(resp)            # e.g. shape (32, 6, 4096) for llama-8b
layer16 = x[16]                             # slice the layer(s) you want, client-side
```

## Always assert the shape

A capture hook that fails is skipped **server-side with only a log warning**, so
a partial capture can come back *looking* like valid data. Assert before you
trust it:

```python
x = decode_residual_stream(resp)
assert x.ndim == 3, x.shape
assert x.shape[0] == n_layers_for_model, f"partial capture: {x.shape[0]} layers"
assert np.isfinite(x.astype(np.float32)).all(), "non-finite — broken capture"
```

## Which models support this?

All three models we serve support harvesting (and steering): **llama-8b**,
**llama-405b**, and **trinity-truebase**.

## Limits & practicalities

- **Prompt-length cap — and how to raise it.** The cap exists because the
  response is large: payload ≈ `captured_layers × prompt_tokens × hidden × 2 B`.
  Each model advertises **`max_activation_prompt_tokens`** under its
  `capabilities` in `/v1/models`, and a longer capture prompt gets a
  `400 activation_prompt_too_long`. That published number is for capturing
  **every** layer.

  **Ask for fewer layers and the cap scales up in proportion**, since what's
  bounded is the response size:

  ```
  cap_for_your_request = ⌊max_activation_prompt_tokens × n_layers ÷ len(layers you asked for)⌋
                         capped at max_model_len
  ```

  The division rounds down, and the result is capped at the model's
  `max_model_len` — you can't exceed the context window whatever the arithmetic
  says. `n_layers` and `max_model_len` are published in the same `capabilities`
  block, so read your ceiling from `/v1/models` rather than hard-coding it: the
  caps are operational settings and they change.

  Worked example, with llama-8b's values at the time of writing
  (`max_activation_prompt_tokens: 512`, `n_layers: 32`, `max_model_len: 8192`).
  A single-layer capture — `"output_residual_stream": [14]` —
  gets `512 × 32 ÷ 1 = 16384`, **clamped to the 8192-token context window**: a
  full conversation captured inline instead of a bulk job. A 4-layer capture
  gets `512 × 32 ÷ 4 = 4096`. The `400` tells you the effective cap for the
  request you actually sent, so you never have to guess.

  Two practical notes. The context window covers the prompt **and** what you
  generate, so a prompt at exactly the cap plus `max_tokens: 50` still gets a
  `400 context_length_exceeded` — leave room. And the cap scales with the layer
  count, not with the payload: one layer of llama-405b at its scaled cap is
  still a ~130 MB response, so on the big models prefer a short prompt or the
  bulk path.

  **This inline endpoint is still meant for probing.** It comfortably handles up
  to ~10k examples; for corpus-scale harvesting, use
  [Bulk harvest](/tutorial/bulk-harvest) — a batch job that writes shards you
  download by URL.
- **Cold boot.** The llama-8b activation engine is kept warm (one always-on
  container), so it is interactive from the first request; if it ever does have
  to come up, it restores from a GPU snapshot in ~30–40 s. The big engines
  (llama-405b, trinity-truebase) scale to zero when idle and rebuild from
  scratch — several minutes on the 8×H200
  models. The wrapper holds that request open and sends keepalive bytes while
  the engine starts, so it normally completes in one shot — set a client timeout
  of at least 15 minutes rather than retrying. If the engine can't come up (e.g.
  the 14-minute safety deadline passes), you get a retryable `modal_cold_boot`
  error — note it can arrive inside a `200` response body once keepalive bytes
  have committed the status, so check the body for `error`, don't rely on the
  HTTP code alone. See [Cold-boot waiting](/tutorial/examples/cold-boot).
- **Usage & quota.** Activation requests are attributed to your key and may count
  against a per-key monthly activation quota (a heavier operation than a plain
  completion); over quota returns `429 activation_quota_exceeded`.

## See also

- [Bulk harvest](/tutorial/bulk-harvest) — the batch pipeline for corpus-scale
  harvesting (submit a job, download shards).
- [Activation steering](/tutorial/activation-steering) — inject your own
  direction into the residual stream.
- [API reference](/tutorial/api) — the base `/v1/completions` contract.
