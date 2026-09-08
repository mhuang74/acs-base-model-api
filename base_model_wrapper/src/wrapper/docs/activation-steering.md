# Activation steering

Nudge a model's behaviour at inference time by **adding a direction to its
residual stream** — the same mechanism behind "concept vectors" / representation
engineering. You supply one or more steering vectors; the engine adds each to the
hidden state at the layer(s) you name, every forward pass, and generation
continues from the perturbed state.

Same `POST {{API_BASE}}/completions` endpoint and key as a normal completion —
you add an **`apply_steering_vectors`** field. Steering runs on the same
activation engine as [harvesting](/tutorial/activation-harvesting).

## The shape of a request

```jsonc
{
  "model": "llama-8b",
  "prompt": "I think that",
  "max_tokens": 32,
  "apply_steering_vectors": [
    {
      "activations": { /* the vector, in the codec below */ },
      "layer_indices": [16],     // which layer(s) to add it at
      "scale": 8.0,              // multiplier on the vector
      "norm_match": false,       // rescale to the hidden state's norm first?
      "position_indices": null   // null = all positions; or a list of token positions
    }
  ]
}
```

You can send up to **8** vectors in one request; each is applied independently.

### The `activations` codec

The vector is encoded the **same way as harvested activations** — bfloat16 values
reinterpreted as int16, zstd-compressed, then base64-encoded:

```python
import base64, numpy as np, ml_dtypes, zstandard as zstd

def encode_vector(vec: np.ndarray) -> dict:
    """vec: float array with ONE ROW PER STEERED LAYER.
    (n_layers, hidden)              -> each layer's vector broadcast to all positions
    (n_layers, n_positions, hidden) -> position-specific
    n_layers must equal len(layer_indices).
    """
    a = vec.astype(ml_dtypes.bfloat16).view(np.int16)     # bf16 bits as int16
    packed = zstd.ZstdCompressor(level=1).compress(a.tobytes())
    return {
        "data": base64.b64encode(packed).decode(),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(a.shape),
        "compression": "zstd",
    }
```

**The leading axis is layers.** `shape[0]` must equal `len(layer_indices)` — one
direction per layer you're steering (a mismatch returns `400`). So:

- steer **one** layer → `layer_indices: [16]`, vector `(1, hidden)`
- steer **two** layers → `layer_indices: [16, 20]`, vector `(2, hidden)` (row 0 → layer 16, row 1 → layer 20)
- target **specific token positions** → make it 3-D `(n_layers, n_positions, hidden)` and set `position_indices`
- target a **contiguous span** (e.g. a whole message) → keep the 2-D vector and set `position_spans` (see [below](#steering-a-span-of-positions))

`hidden` is the model's width — **4096** (llama-8b), **16384** (llama-405b),
**3072** (trinity-truebase).

## A full example

```python
import base64, json, numpy as np, ml_dtypes, zstandard as zstd, urllib.request

BASE, KEY = "$ACS_API_BASE", "$ACS_API_KEY"

def post(body):
    req = urllib.request.Request(f"{BASE}/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    if "error" in out:   # late cold-boot failures arrive inside a 200 body
        raise RuntimeError(out["error"].get("message", str(out["error"])))
    return out

# In practice this vector is a learned/derived direction (a probe weight, a
# difference-of-means between two prompt sets, an SAE decoder column, …).
vec = my_direction.reshape(1, 4096)          # (n_layers=1, hidden) — one row for layer 16

sv = {"activations": encode_vector(vec), "layer_indices": [16],
      "scale": 8.0, "norm_match": False, "position_indices": None}

gen = {"model": "llama-8b", "prompt": "I think that", "max_tokens": 32, "temperature": 0}
print("baseline:", post(gen)["choices"][0]["text"])
print("steered :", post({**gen, "apply_steering_vectors": [sv]})["choices"][0]["text"])
```

Sanity check while wiring it up: **`scale: 0.0` is the identity** — a
zero-scale steer must reproduce the baseline text exactly. If it doesn't, your
request isn't reaching the steering path.

## Steering a span of positions

To steer a contiguous run of positions — say every token of a message — you'd
otherwise have to spell out each index in `position_indices` *and* build a 3-D
tensor that repeats one direction across all of them. `position_spans` does that
for you: send the **2-D** `(n_layers, hidden)` vector plus `[start, end)` pairs,
and the wrapper broadcasts the vector across every position in the spans before
it reaches the engine.

```python
# same encode_vector / post helpers as above
vec = my_direction.reshape(1, 4096)          # 2-D: one row for layer 16

sv = {"activations": encode_vector(vec), "layer_indices": [16],
      "scale": 8.0, "norm_match": False,
      "position_spans": [[10, 25]]}          # steer positions 10..24 (half-open)

print(post({**gen, "apply_steering_vectors": [sv]})["choices"][0]["text"])
```

Rules:

- Spans are **half-open**: `[10, 25]` covers positions 10 through 24, not 25.
  Endpoints are absolute and 0-indexed, `end` must be `> start`, and negatives
  are rejected for the same reason as `position_indices`.
- Pass the **2-D** vector — the wrapper tiles it. A 3-D tensor with
  `position_spans` returns `400` (use `position_indices` for per-position
  directions).
- `position_spans` and `position_indices` are **mutually exclusive** — set one,
  not both (`400` otherwise).
- Multiple spans are fine (`[[0, 4], [20, 30]]`); they must not overlap.

Under the hood this expands to exactly the `position_indices` + 3-D path below,
so the engine behaviour is identical — it's an ergonomics shortcut, not a new
steering mode.

## Field reference

| Field | Meaning |
|---|---|
| `activations` | The direction(s), in the codec above — **one row per steered layer**; `shape[0]` must equal `len(layer_indices)`. |
| `layer_indices` | Decoder layer(s) to steer (0-indexed). Row *i* of `activations` is added at `layer_indices[i]`. |
| `scale` | Multiplier. Bigger = stronger effect; too big collapses into gibberish/repetition. |
| `norm_match` | If `true`, the vector is rescaled to the hidden state's norm at each position *before* `scale` — makes a good `scale` transfer across layers/prompts. If `false`, the raw vector is used. |
| `position_indices` | `null` steers every token position; a list steers only those positions. Positions are **absolute and 0-indexed** — for the last token of an `n`-token prompt pass `[n-1]`, not `[-1]` (negative indices are rejected: the engine's semantics for them are unverified, and mis-targeted steering fails silently). Requires a 3-D `activations` shape with `shape[1] == len(position_indices)` — combined with a 2-D vector the engine would ignore it, so that returns `400`. |
| `position_spans` | Convenience for steering contiguous runs: a list of half-open `[start, end)` pairs, applied by broadcasting a **2-D** vector across every position in the spans (see [Steering a span of positions](#steering-a-span-of-positions)). Endpoints are absolute, 0-indexed, non-negative, with `end > start`; spans must not overlap. Mutually exclusive with `position_indices` (`400` if both set) and requires a 2-D vector (`400` with a 3-D tensor). Expands server-side to the `position_indices` + 3-D path. |

## Which models support this?

All three models we serve support steering (and harvesting): **llama-8b**,
**llama-405b**, and **trinity-truebase**.

## Practicalities

- **One sequence per steered request:** steering requires a single prompt and
  `n: 1`; a batched (list-valued) prompt or `n > 1` returns `400`. The engine
  applies vectors to the first sequence of a request only, so before this
  guard those shapes came back part-steered with a `200`. To fan out, send
  separate requests (vary `seed` for sampled diversity).
- **Cold boot & quota:** same as harvesting — llama-8b wakes from a GPU
  snapshot in ~30–40 s after an idle gap; the big engines (llama-405b,
  trinity-truebase) cold-boot for several minutes on the first request
  after idle, and the wrapper holds that request open through the boot (set a
  client timeout of at least 15 minutes — see
  [Cold-boot waiting](/tutorial/examples/cold-boot)). Requests are attributed
  to your key against the activation quota.
- **Baselines: same seed ≠ same engine.** Ordinary completions and
  activation/steering requests run on **separate engines** per model. A seed
  reproduces a draw *within* an engine, so comparing a steered request against
  a same-seed plain completion is not a valid control — the two engines sample
  different streams. Run the control on the activation engine too: a
  `scale: 0.0` steering vector is an exact identity (verified — it reproduces
  the same-engine unsteered output bit-for-bit), so `{**req,
  "apply_steering_vectors": [{**sv, "scale": 0.0}]}` is the clean baseline.
- **Finding a good `scale`:** start small and sweep up. Two things interact — the
  `scale` **and how many layers you steer**: pushing a wide band compounds
  fast. The worked example below (a single mid layer, raw vector) has its sweet
  spot around `1.5–2`; with `norm_match: true` over a small band, useful scales
  are typically below `1`. Watch for the output degrading into repetition —
  that's the sign you've overshot.

## Worked example: a "pirate" contrast vector

A full end-to-end demonstration — done entirely through this API — lives as a
**marimo notebook** at [`pirate_steering.py`](/examples/pirate_steering.py)
(run: `ACS_API_KEY=… uvx marimo edit --sandbox --watch pirate_steering.py`),
with a rendered run **including all cell outputs** at
[`pirate_steering.html`](/examples/pirate_steering.html) — readable without an
API key. It:

1. **harvests** residual-stream activations for six pirate-styled sentences and
   six plain paraphrases (one capture each; mean over content tokens, BOS
   position excluded),
2. **builds** the direction: `mean(pirate) − mean(plain)`, per layer —
   contrasting **actual text in the target style**,
3. **steers** layer 16 of `llama-8b` with the raw vector, sweeping `scale`, and
4. **scores** the effect two ways — a pirate-word count and a logprob
   preference metric (`echo` + `prompt_logprobs`, shared prefix cancelling).

The measured dose–response (temperature 0.8, pinned seed):

| scale | result |
|---|---|
| `0` | ordinary completions — verified **identical** to a same-engine unsteered baseline |
| `1.5` | *"**Arr, me hearties**, 'tis a fair day for ye walk the plank, **matey**!"* — coherent pirate |
| `2.0` | the weather forecast: *"…I'm **hoisting the main 'n jib** and **shiver me arrgh**! we'll be a-fer shore by 3 bells!"* |
| `≥3` | degenerates into repetition — the expected overshoot |

The same vector **generalizes** to new prompts: at `2.0`, job-day advice
becomes *"go in with a **swashbucklin'** attitude and a **bountiful chest o'
booty**, **Mateys**!"* — the direction carries the style itself, so it transfers.

## Steer and harvest in the same request

Steering and [activation harvesting](/tutorial/activation-harvesting) run on the
same engine, so one request can do both: add your steering vectors **and** ask
for the residual stream back. You intervene and measure in the same forward
pass — the core setup for causal-interpretability work, where you want the
activations from the *steered* generation, not from a separate run.

Add `output_residual_stream` (from the [harvesting](/tutorial/activation-harvesting)
page) to a steering request. The single-feature rules still apply and compose:
the request must be **non-streaming**, a **single prompt**, and **`n: 1`** —
captured activations ride only on the non-streaming response body, and steering
touches only the first sequence.

```python
# builds on `post`, `encode_vector`, `sv`, and `gen` from the full example above
resp = post({
    **gen,                                    # single prompt; n defaults to 1
    "apply_steering_vectors": [sv],           # intervene: add the direction at layer 16
    "output_residual_stream": [16],           # measure: capture that same layer back
})

text = resp["choices"][0]["text"]              # the steered completion
acts = resp["activations"]["residual_stream"]  # decode as on the harvesting page
```

The response carries both the completion (`choices`) and the captured tensor
(`activations`) — decode the latter with the helper on the
[Activation harvesting](/tutorial/activation-harvesting) page. A steering
vector's `layer_indices` and the `output_residual_stream` indices address the
**same** decoder layers: in beta testing a steer applied at layers 0, 1, 16, 30
and 31 came back captured at those same indices, so you can request the exact
layer you steer. Because both happen in one forward pass, the activations you
get back are from the same steered generation, not a separate baseline call.

## See also

- [Activation harvesting](/tutorial/activation-harvesting) — capture the residual
  stream (and derive steering directions from it).
- [API reference](/tutorial/api) — the base `/v1/completions` contract.
