# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "ml-dtypes",
#     "zstandard",
#     "matplotlib",
# ]
# ///
"""Pirate-vector activation steering — a worked end-to-end example.

Runs entirely against the public ACS base-model API (no local GPU, no model
weights): harvest activations, build a steering vector, steer, measure.

    ACS_API_KEY=... uvx marimo edit --sandbox --watch pirate_steering.py

(--watch keeps the browser session in sync if the file changes on disk.)
Static export with outputs:  uvx marimo export html pirate_steering.py
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        r"""
    # Steering a base model with a pirate vector 🏴‍☠️

    This notebook walks the **full activation-steering loop** against the ACS
    base-model API — the same `/v1/completions` endpoint you already use, no local
    GPU required:

    1. **Harvest** residual-stream activations for pirate-styled vs. plain sentences
       ([harvesting tutorial](https://infra.acsresearch.org/tutorial/activation-harvesting));
    2. build a **pirate direction** — the difference of means between the two sets;
    3. **steer** with it ([steering tutorial](https://infra.acsresearch.org/tutorial/activation-steering))
       and watch a weather forecast turn into a sea shanty;
    4. **quantify** the effect — a pirate-word score, a logprob preference metric,
       and the characteristic *inverted-U* (steer too hard and fluency collapses).

    At the end: a short note on a popular recipe that does **not** transfer well to
    base models, so you don't spend an afternoon rediscovering that.

    You need an ACS API key in `ACS_API_KEY` and a model with
    `"activations": true` in `GET /v1/models` (we use `llama-8b`). The first
    activation request cold-boots a dedicated engine (~2 min); the whole notebook
    is ~50 small requests (a couple of minutes warm).
    """
    )
    return (mo,)


@app.cell
def _(mo):
    import base64
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    import ml_dtypes
    import numpy as np
    import zstandard as zstd

    # Accept the base URL with or without the /v1 suffix (and note: marimo
    # auto-loads a .env from the working directory, which can override this).
    BASE = os.environ.get("ACS_API_BASE", "https://infra.acsresearch.org/v1").rstrip("/")
    if not BASE.endswith("/v1"):
        BASE += "/v1"
    KEY = os.environ.get("ACS_API_KEY", "")
    MODEL = "llama-8b"

    mo.stop(
        not KEY,
        mo.md("⚠️ **Set `ACS_API_KEY` in your environment and re-run.**"),
    )

    def post(body: dict, timeout: int = 1200, retries: int = 8) -> dict:
        """POST /v1/completions, riding out cold boots and deploy blips.

        The activation engine scales to zero when idle; the first request
        returns a retryable `503 modal_cold_boot`. Brief 404/502/503 blips
        (a deploy in flight) get a few quick retries too.
        """
        req = urllib.request.Request(
            f"{BASE}/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        transient_left = 3
        for attempt in range(retries):
            try:
                out = json.load(urllib.request.urlopen(req, timeout=timeout))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                msg = body[:300]
                if e.code == 503 and "cold_boot" in msg:
                    time.sleep(30)  # engine cold boot: be patient
                    continue
                if e.code in (404, 502, 503) and transient_left > 0:
                    transient_left -= 1
                    time.sleep(10)
                    continue
                # Surface the API's reason (error.message) rather than raw JSON;
                # fall back to the raw body if it isn't the expected shape.
                try:
                    reason = json.loads(body)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    reason = msg
                raise RuntimeError(f"HTTP {e.code} on {req.full_url}: {reason}") from None
            # A late cold-boot failure arrives INSIDE a 200 body (keepalive
            # bytes commit the status before the engine gives up), so a bare
            # out["choices"] would die on a KeyError instead of the real reason.
            err = out.get("error")
            if err is None:
                return out
            if "cold_boot" in str(err.get("code", "")) and attempt < retries - 1:
                time.sleep(30)  # same patience as the 503 cold-boot path
                continue
            raise RuntimeError(f"API error on {req.full_url}: {err.get('message', err)}")
        raise RuntimeError(f"gave up waiting for the engine ({req.full_url})")

    mo.md(f"Configured for **`{MODEL}`** at `{BASE}` — key found ✓")
    return MODEL, base64, ml_dtypes, np, post, zstd


@app.cell
def _(mo):
    mo.md(r"""
    ## The activation codec

    Both directions of the API use one tensor envelope: **bf16 values, bit-cast to
    int16, zstd-compressed, base64-encoded**. Two helpers straight from the
    tutorial — decode what we harvest, encode what we inject:
    """)
    return


@app.cell
def _(base64, ml_dtypes, np, zstd):
    def decode_residual_stream(resp: dict):
        """activations.residual_stream -> np array (n_layers, n_tokens, hidden)."""
        d = resp["activations"]["residual_stream"]
        raw = base64.b64decode(d["data"])
        if d.get("compression") == "zstd":
            raw = zstd.ZstdDecompressor().decompress(raw)
        arr = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
        return arr.reshape(d["shape"])

    def encode_vector(vec) -> dict:
        """float array (n_layers, hidden) -> vLLM-Lens tensor codec dict."""
        a = vec.astype(ml_dtypes.bfloat16).view(np.int16)
        packed = zstd.ZstdCompressor(level=1).compress(a.tobytes())
        return {
            "data": base64.b64encode(packed).decode(),
            "dtype": "int16",
            "original_dtype": "torch.bfloat16",
            "shape": list(a.shape),
            "compression": "zstd",
        }

    return decode_residual_stream, encode_vector


@app.cell
def _(MODEL, decode_residual_stream, np, post):
    # ---- experiment harness --------------------------------------------------

    def capture(prompt: str):
        """Full residual stream for a prompt: (n_layers, n_tokens, hidden) f32.

        Trust nothing: a failed capture hook is skipped server-side with only a
        log warning, so assert the shape before using the data.
        """
        r = post(
            {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "output_residual_stream": True,
            }
        )
        x = decode_residual_stream(r)
        assert x.ndim == 3 and x.shape[0] == 32, f"partial capture: {x.shape}"
        x = x.astype(np.float32)
        assert np.isfinite(x).all(), "non-finite — broken capture"
        return x

    def gen(prompt: str, sv: dict, max_tokens: int = 45) -> str:
        """One steered completion (temperature 0.8, pinned seed → reproducible).

        The vector is sent at EVERY strength, zero included — a zero-scale steer
        is a proven identity (see the sanity check) and it keeps every data
        point on the SAME engine (see the same-engine note below).
        """
        body = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
            "seed": 7,
            "apply_steering_vectors": [sv],
        }
        return post(body)["choices"][0]["text"]

    PIRATE_WORDS = set(
        "arr arrr aye ye yer matey ahoy avast booty plunder treasure doubloon "
        "scallywag scurvy landlubber buccaneer privateer sail sailed sailing sails "
        "ship ships vessel schooner sloop frigate sea seas ocean waves captain "
        "cap'n crew mateys grog rum anchor helm mast deck harbor port voyage "
        "sailor pirate pirates parrot shanty seadog swab hearties yo-ho tis".split()
    )

    def pirate_score(text: str) -> int:
        """Crude but legible: how many pirate/nautical words in the text."""
        return sum(t.strip(".,!?;:'\"—").lower() in PIRATE_WORDS for t in text.split())

    return capture, gen, pirate_score


@app.cell
def _(mo):
    mo.md(r"""
    ## Build the pirate direction

    Six pirate-*styled* sentences vs. six semantically-matched plain paraphrases —
    we contrast **actual text in the target style**, which is the design that works
    on a base model (see the note at the end for what doesn't). Each sentence's
    residuals are averaged over its content tokens, **skipping position 0**: the
    BOS token is an attention sink with a huge, style-irrelevant norm. One capture
    per sentence returns *all* layers at once; the direction is the difference of
    means.
    """)
    return


@app.cell
def _(capture, mo, np):
    PIRATE_SENTS = [
        "Arr, matey, hand over the treasure!",
        "Ahoy, ye scurvy dogs, hoist the sails!",
        "Shiver me timbers, there be a storm brewin'!",
        "Avast, ye landlubbers, walk the plank!",
        "Yo ho ho, a pirate's life be grand!",
        "Batten down the hatches, me hearties!",
    ]
    PLAIN_SENTS = [
        "Please hand over the valuables.",
        "Hello everyone, raise the sails.",
        "Goodness, a storm is coming.",
        "Stop, you inexperienced people, leave the ship.",
        "Honestly, a sailor's life is great.",
        "Secure the ship, my friends.",
    ]

    def _mean_content(prompt):
        return capture(prompt)[:, 1:, :].mean(axis=1)  # (32, 4096), BOS excluded

    print("harvesting 12 sentences…")
    _pir = np.stack([_mean_content(p) for p in PIRATE_SENTS])
    _pla = np.stack([_mean_content(p) for p in PLAIN_SENTS])
    V = _pir.mean(axis=0) - _pla.mean(axis=0)  # (32, 4096): per-layer direction

    mo.md(
        f"Pirate direction `V` built: shape `{V.shape}`, "
        f"‖V₁₆‖ = {np.linalg.norm(V[16]):.2f}."
    )
    return (V,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Sanity check before steering anything

    A steering vector rides along in the completions request under
    **`apply_steering_vectors`**. Before trusting any result: **`scale: 0` must
    reproduce the baseline exactly** — if it doesn't, your request isn't reaching
    the steering path at all.

    One subtlety: activation requests run on a **separate engine** from plain
    completions, and two engines aren't bit-identical — even at `temperature: 0`
    their outputs can differ by a token. So compare like with like: the baseline
    below forces the activation engine too (via `output_residual_stream`), making
    the zero-scale check exact. For the same reason, every curve in this notebook
    sends the vector at every strength — zero included.
    """)
    return


@app.cell
def _(MODEL, V, encode_vector, mo, post):
    def pirate_sv(scale: float) -> dict:
        """Steering directive: layer 16 (justified by the sweep below), raw scale."""
        return {
            "activations": encode_vector(V[16].reshape(1, -1)),
            "layer_indices": [16],
            "scale": float(scale),
            "norm_match": False,
            "position_indices": None,
        }

    _gen0 = {
        "model": MODEL,
        "prompt": 'I looked out at the sea, turned to the crew, and said, "',
        "max_tokens": 45,
        "temperature": 0,
        # Force the activation engine for the baseline too (same-engine compare):
        "output_residual_stream": True,
    }
    _baseline = post(_gen0)["choices"][0]["text"]
    _zero = post({**_gen0, "apply_steering_vectors": [pirate_sv(0.0)]})["choices"][0]["text"]
    assert _baseline == _zero, "scale-0 steer must be the identity!"

    mo.md("**Zero-scale identity holds** — the steering path is engaged and inert at 0. ✓")
    return (pirate_sv,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Which layer? Sweep it

    One good way to pick the layer: steer **each candidate layer alone** — same
    scale, same seed — and read the outputs. Each layer is steered with its own
    row of `V`, i.e. the direction *measured at that same layer*. Mid-stack
    layers carry the style; very early layers are too token-bound, and late
    layers too output-specific, to move much. That's why the rest of this
    notebook fixes **layer 16**.
    """)
    return


@app.cell
def _(V, encode_vector, gen, mo, pirate_score):
    def _sv_at(layer: int, scale: float) -> dict:
        return {
            "activations": encode_vector(V[layer].reshape(1, -1)),
            "layer_indices": [layer],
            "scale": float(scale),
            "norm_match": False,
            "position_indices": None,
        }

    _rows = []
    for _layer in [4, 8, 12, 16, 20, 24, 28]:
        _t = gen("The weather forecast for tomorrow says", _sv_at(_layer, 2.0))
        _rows.append(
            {
                "layer": _layer,
                "🏴‍☠️": pirate_score(_t),
                "completion (scale 2.0)": _t.replace("\n", " ")[:110],
            }
        )
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dose–response

    Same prompts, same seed, rising `scale`. Watch the completions go *normal →
    pirate → heavy dialect → collapse*: the **inverted-U** that is the fundamental
    trade-off of activation steering. The `🏴‍☠️` column counts pirate words across
    all three completions — and the third prompt (**job advice**, nothing nautical
    about it) is the *generalization probe*: a direction that pirates *that* has
    captured a concept, not just memorized our twelve harvest sentences.

    The **sweet spot** — the scale band where the style shift is strong but
    fluency still intact — is found exactly this way: sweep and read. Here it
    is ≈ **1.5–2.0**.
    """)
    return


@app.cell
def _(gen, mo, pirate_score, pirate_sv):
    SCALES = [0.0, 1.5, 2.0, 2.5, 3.0]
    PROMPT_SPEECH = 'I looked out at the sea, turned to the crew, and said, "'
    PROMPT_WEATHER = "The weather forecast for tomorrow says"
    PROMPT_JOB = "My advice for your first day at the new job is"

    _rows = []
    for _s in SCALES:
        _t1 = gen(PROMPT_SPEECH, pirate_sv(_s))
        _t2 = gen(PROMPT_WEATHER, pirate_sv(_s))
        _t3 = gen(PROMPT_JOB, pirate_sv(_s))
        _rows.append(
            {
                "scale": _s,
                "🏴‍☠️": pirate_score(_t1) + pirate_score(_t2) + pirate_score(_t3),
                "speech": _t1.replace("\n", " ")[:80],
                "weather": _t2.replace("\n", " ")[:80],
                "job advice (generalization)": _t3.replace("\n", " ")[:80],
            }
        )
    mo.ui.table(_rows, selection=None)
    return PROMPT_WEATHER, SCALES


@app.cell
def _(mo):
    mo.md(r"""
    ## A second metric: the distribution shift

    The `🏴‍☠️` word count above already quantifies the *sampled generations* —
    simple and direct, though lexicon-bound and noisy per completion. A
    complementary, finer-grained measure looks at the **distribution** instead:
    under each scale we score two fixed continuations of the same prefix with
    `echo` + `prompt_logprobs` — one pirate-y, one plain — and compare **mean
    logprob per token**. The shared prefix cancels in the difference:

    > preference = lp/tok(pirate continuation) − lp/tok(plain continuation)

    Read it by its **slope**: steering shifts the preference steadily toward
    pirate. The zero crossing — the point where pirate-speak becomes outright
    *more likely than plain English*, token for token — comes later (here
    ≈ 2.5) than the first visibly pirate samples (≈ 1.5–2.0). That's expected:
    sampling at temperature surfaces pirate tokens well before the average
    likelihood favors an exaggerated dialect string. The two metrics agree on
    direction and together bracket the effect.
    """)
    return


@app.cell
def _(MODEL, PROMPT_WEATHER, SCALES, mo, pirate_sv, post):
    import matplotlib.pyplot as plt

    # NB: continuations must START WITH A SPACE — that keeps the prefix's BPE
    # tokenization stable when we tokenize prefix+continuation together, so
    # slicing by the bare prefix's token count is exact.
    _CONT_PIRATE = " arr, 'tis be a fine day for sailin', me hearties!"
    _CONT_PLAIN = " it will be sunny with light winds in the afternoon."

    _n_prefix = post(
        {"model": MODEL, "prompt": PROMPT_WEATHER, "max_tokens": 1, "temperature": 0}
    )["usage"]["prompt_tokens"]

    def _lp_per_token(cont: str, sv: dict) -> float:
        body = {
            "model": MODEL,
            "prompt": PROMPT_WEATHER + cont,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "prompt_logprobs": 0,
            "apply_steering_vectors": [sv],  # sent at every strength, 0 included
        }
        pl = post(body)["choices"][0]["prompt_logprobs"]
        xs = [list(d.values())[0]["logprob"] for d in pl[_n_prefix:] if d]
        return sum(xs) / len(xs)

    _pref = [
        (s, _lp_per_token(_CONT_PIRATE, pirate_sv(s)) - _lp_per_token(_CONT_PLAIN, pirate_sv(s)))
        for s in SCALES
    ]

    _fig, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(*zip(*_pref), marker="o", color="#7c3aed")
    _ax.axhline(0, lw=1, color="gray", ls=":")
    _ax.set_xlabel("steering scale")
    _ax.set_ylabel("pirate − plain (lp/token)")
    _ax.set_title("Preference for pirate continuations vs. steering scale")
    _ax.spines[["top", "right"]].set_visible(False)
    mo.vstack(
        [
            mo.ui.table(
                [{"scale": s, "preference (lp/token)": round(p, 2)} for s, p in _pref],
                selection=None,
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ⚠️ What did *not* work: instruction-pair contrast on a base model

    The classic [repeng/CAA](https://vgel.me/posts/representation-engineering/)
    recipe builds the direction from **paired instructions** — `"Talk like a
    pirate."` vs. `"Talk like a normal person."` over shared neutral suffixes,
    reading the residual at the last prompt token. It's standard and strong on
    **instruct** models, where the instruction heavily conditions that state.

    We measured it here so you don't have to: on this **base** model it produced
    only mild nautical drift with no persona shift (pirate-word score ≤5, vs. ~19
    for the text contrast above at its sweet spot, scale 2). A base model doesn't *obey*
    the instruction — the prefix is merely weak document context. A document-framed
    variant (`"The following is a transcript of a pirate captain speaking:"`)
    failed differently: the "transcript of someone speaking" framing injects a
    dominant first-person direction, and steering collapses into `"I, I, I…"`.

    **Rule of thumb: on a base model, contrast actual text in the target style** —
    the distribution you want to induce — rather than states downstream of an
    instruction the model was never trained to follow.

    ## Takeaways

    - The whole loop — capture, decode, encode, steer, score — is plain
      `/v1/completions` calls. No GPU, no model surgery.
    - **Always keep a `scale: 0` identity check** in your harness, compared
      same-engine (activation requests run on their own engine).
    - **Calibrate strength from small** and watch for repetition/degeneration —
      the useful window here was scale ≈1.5–2.0; by 3 the output collapses.
    - Practicalities: activation-capable models advertise `"activations": true` in
      `GET /v1/models`; capture prompts are length-capped per model; the engine
      cold-boots after idle; activation requests count against a per-key quota.

    **Docs:** [activation harvesting](https://infra.acsresearch.org/tutorial/activation-harvesting) ·
    [activation steering](https://infra.acsresearch.org/tutorial/activation-steering) ·
    [API reference](https://infra.acsresearch.org/tutorial/api)
    """)
    return


if __name__ == "__main__":
    app.run()
