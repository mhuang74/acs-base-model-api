#!/usr/bin/env python3
"""repeng-style activation steering, end-to-end through the ACS API.

A worked example of the technique from https://vgel.me/posts/representation-engineering/,
done entirely over the public `/v1/completions` endpoint — proof that reading
activations and steering with them both work:

  1. READ    capture residual-stream activations for paired prompts
             ("talk like a pirate" vs "talk like a normal person") over many
             neutral suffixes.
  2. BUILD    per-layer contrast vector = mean(pirate) - mean(neutral), unit-normed
             (the robust difference-of-means form of a repeng control vector).
  3. STEER    add the vector (x a coefficient) across a small mid-layer band while
             generating a neutral prompt; sweep the coefficient.
  4. OBSERVE  a pirate-word score that rises as you steer positive, and coherent
             pirate/nautical prose at the sweet spot.

Run:
  ACS_API_KEY=sk-... [ACS_API_BASE=https://infra.acsresearch.org/v1] \
    uv run --with numpy --with ml_dtypes --with zstandard python \
    scripts/activation/pirate_steering_example.py

Notes:
- Uses llama-8b (hidden size 4096, 32 layers). First call cold-boots (~2 min).
- ~60 capture calls to build the vector, then a short generation sweep.
- The effect saturates if you steer too many layers or too hard — that's expected;
  the sweet spot is a small mid band at a modest coefficient (see SWEEP below).
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

import ml_dtypes
import numpy as np
import zstandard as zstd

BASE = os.environ.get("ACS_API_BASE", "https://infra.acsresearch.org/v1")
KEY = os.environ["ACS_API_KEY"]
MODEL = "llama-8b"


def _post(body: dict, timeout: int = 600, retries: int = 4) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE}/completions", data=data,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        try:
            out = json.load(urllib.request.urlopen(req, timeout=timeout))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            # 503 = engine cold-booting after scale-to-zero; retry a few times.
            if e.code == 503 and attempt < retries - 1:
                print(f"  cold boot ({e.code}); retrying in 20s...", flush=True)
                time.sleep(20)
                continue
            # Surface the API's reason (error.message); fall back to the raw body.
            try:
                reason = json.loads(body)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                reason = body[:300]
            raise SystemExit(f"HTTP {e.code} from {BASE}: {reason}")
        # A late cold-boot failure lands INSIDE a 200 body (keepalive bytes
        # commit the status first); surface it instead of a KeyError downstream.
        err = out.get("error")
        if err is None:
            return out
        if "cold_boot" in str(err.get("code", "")) and attempt < retries - 1:
            print("  cold boot failed mid-request; retrying in 20s...", flush=True)
            time.sleep(20)
            continue
        raise SystemExit(f"API error from {BASE}: {err.get('message', err)}")
    raise SystemExit("exhausted retries waiting for the engine")


def decode_resid(resp: dict) -> np.ndarray:
    """(n_layers, n_tokens, hidden) float32 from the activations codec."""
    d = resp["activations"]["residual_stream"]
    raw = base64.b64decode(d["data"])
    if d.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    # int16 storage is the bf16 bit-pattern reinterpreted — view, don't cast.
    arr = np.frombuffer(raw, dtype=np.uint16).view(ml_dtypes.bfloat16)
    return arr.reshape(d["shape"]).astype(np.float32)


def capture_last_token(prompt: str) -> np.ndarray:
    """(n_layers, hidden): residual at the LAST prompt token, per layer."""
    r = _post({"model": MODEL, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "output_residual_stream": True})
    return decode_resid(r)[:, -1, :]


def encode_vec(mat: np.ndarray) -> dict:
    """(n_layers_selected, hidden) float -> vLLM-Lens steering-vector codec dict."""
    a = mat.astype(ml_dtypes.bfloat16).view(np.int16)
    packed = zstd.ZstdCompressor(level=1).compress(a.tobytes())
    return {"data": base64.b64encode(packed).decode(), "dtype": "int16",
            "original_dtype": "torch.bfloat16", "shape": list(a.shape),
            "compression": "zstd"}


# ---- 1. contrastive dataset -------------------------------------------------
POS = "Talk like a pirate.\n"
NEG = "Talk like a normal person.\n"
SUFFIXES = [
    "I think that", "The weather today is", "My favorite food is", "When I woke up",
    "Let me tell you about", "The best part of my day was", "I walked into the room and",
    "Honestly, the thing is", "We should probably", "After a long day I like to",
    "The meeting went", "I opened the door and", "She looked at me and said",
    "The plan for tomorrow is", "I can't believe that", "Every morning I",
    "The strangest thing happened", "He picked up the phone and", "In my opinion",
    "The town was quiet until", "I ordered a coffee and", "They gathered around to",
    "Nothing beats a good", "The road ahead was", "I remember when",
    "It all started when", "The captain of the team", "Out on the water",
    "We set off early because", "The old map showed",
]

# ---- 2. build the contrast vector (difference of means, unit-normed) --------
print(f"reading activations: {2 * len(SUFFIXES)} capture calls "
      "(first one cold-boots ~2 min)...", flush=True)
pos = np.stack([capture_last_token(POS + s) for s in SUFFIXES])   # (pairs, layers, hidden)
neg = np.stack([capture_last_token(NEG + s) for s in SUFFIXES])
control = pos.mean(0) - neg.mean(0)                               # (layers, hidden)
control /= np.linalg.norm(control, axis=1, keepdims=True) + 1e-8
print(f"built contrast vector: {control.shape} (layers, hidden)", flush=True)

# ---- 3 & 4. steer a small mid band, sweep the coefficient, score ------------
# Steering many layers at once compounds and saturates into "pirate pirate...".
# A small mid band at a modest coefficient gives coherent pirate/nautical prose.
LAYERS = [11, 12, 13, 14, 15]
vec = np.stack([control[L] for L in LAYERS])                      # (len(LAYERS), hidden)
SWEEP = [-0.5, -0.25, 0.0, 0.4, 0.6, 0.8, 1.0]
PROMPTS = ["Yesterday I went down to the docks and",
           "My advice for your first day at the new job is"]

PIRATE_WORDS = set(
    "arr arrr aye ye yer matey ahoy avast booty plunder salty windward treasure "
    "doubloon scallywag scurvy landlubber buccaneer privateer sail sailed sailing "
    "sails ship ships vessel schooner sloop frigate sea seas ocean waves captain "
    "cap'n crew mateys grog rum anchor helm mast deck harbor port voyage sailor "
    "pirate pirates parrot shanty seadog swab hearties yo-ho".split()
)


def pirate_score(text: str) -> int:
    return sum(t.strip(".,!?;:'\"").lower() in PIRATE_WORDS for t in text.split())


def generate(prompt: str, coeff: float) -> str:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 40, "temperature": 0}
    if coeff != 0.0:
        body["apply_steering_vectors"] = [{
            "activations": encode_vec(vec), "layer_indices": LAYERS,
            "scale": float(coeff), "norm_match": True, "position_indices": None}]
    return _post(body)["choices"][0]["text"].strip()


for prompt in PROMPTS:
    print(f"\nsteering layers {LAYERS[0]}..{LAYERS[-1]}  |  prompt: {prompt!r}")
    print("=" * 74)
    for coeff in SWEEP:
        txt = generate(prompt, coeff)
        print(f"  coeff={coeff:+.2f}  pirate={pirate_score(txt):2d}  | {txt[:150]!r}")
print("\ncoeff 0 = ordinary; moderate positive = coherent pirate/nautical; "
      "too high = it saturates. Reading + steering both verified.")
