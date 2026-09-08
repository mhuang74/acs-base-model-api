---
title: Modal deploy phases (stage_weights → preshard → serve)
status: current
updated: 2026-07-02
owner: platform@example.org
---

# Why the deploy is split into three phases

Reference note for the `stage_weights` → `preshard` → `serve` pipeline in `modal_app.py`.
The short version: Llama-3.1-405B is ~810 GB in bf16 and needs 8 GPUs to run, so the
slow/expensive setup work (download, shard) is done **once** into persistent Modal Volumes,
and only the cheap runtime (`serve`) touches those artifacts on every container start.

## Phase 1 — `stage_weights` (one-shot, ~30–60 min)

**Purpose: pull the gated HF checkpoint onto fast Modal-local storage, exactly once.**

- ~810 GB download — you don't want this on every cold start (it would hammer HF and make
  cold starts 20+ min). A Modal **Volume** (`weights`) is persistent network storage: write
  once, mount read-only from many containers later.
- `HF_HUB_ENABLE_HF_TRANSFER=1` + `snapshot_download` → the Rust `hf_transfer` parallel
  downloader, much faster than the default Python client for files this size.
- Gated by an `HF_TOKEN` Modal **Secret** because 405B is a gated repo on HF.
- **Idempotent**: if the snapshot is already present and the checksum matches, return
  immediately. This is the safety net so re-running a deploy never re-downloads 810 GB.

**Output:** `/vol/weights/llama-3.1-405b` — the canonical model, in HF format.

## Phase 2 — `preshard` (one-shot, ~5–10 min)

**Purpose: pre-compute the tensor-parallel (TP=8) split so serving containers don't redo it on every cold start.**

- 405B only fits across 8 GPUs (tensor parallelism). Normally vLLM loads the *full* HF
  checkpoint and shards it in memory at startup — slow and memory-heavy on every boot.
- Boot vLLM once with `tensor_parallel_size=8` and call `save_sharded_state` (in the V1
  engine: `llm.llm_engine.engine_core.save_sharded_state(...)`, pattern
  `ShardedStateLoader.DEFAULT_PATTERN`, `max_size=5 GiB`) to write per-worker shard files
  (`model-rank-{rank}-part-{part}.safetensors`) into the `acs-sharded` Volume. Also copies the
  non-weight metadata (`config.json`, `tokenizer.*`, `generation_config.json`, …) alongside
  so `vllm serve <SHARDED_DIR>` finds them. Each serving container then reads only its own
  shard → faster load, lower peak RAM.
- This is why Phase 3 serves with `--load-format runai_streamer_sharded`: "weights are
  already sharded for TP=8 — stream them in." `runai_streamer_sharded` is the Run:ai Model
  Streamer variant of vLLM's `sharded_state` loader: same `model-rank-*-part-*` file naming,
  but faster reads (tunable via `--model-loader-extra-config '{"concurrency":N,"memory_limit":B}'`).
  Falls back to loading from the plain HF cache if no shards are present (fine for the small
  dev model, slow for 405B).
- Separate Volume from Phase 1 so the original stays the source of truth; if you ever change
  TP size or engine, re-run Phase 2 from Phase 1's output — no re-download.
- Skip if shards already present (same idempotency idea as Phase 1).

**Output:** `/vol/weights-sharded/llama-3.1-405b-tp8` — serving-format derivative tied to TP=8.

**Why two phases, not one:** different artifacts with different lifecycles — Phase 1's output
is the canonical model, Phase 2's is a disposable, engine/TP-specific cache.

**MoE / expert-parallel models (Trinity) — the EP-correctness constraint.** `save_sharded_state`
has **no expert-parallel awareness**: it writes one file per *tensor-parallel rank* capturing
whatever `state_dict()` slice lives on that rank at save time (vLLM 0.19.1). For an
expert-parallel MoE like Trinity (`--enable-expert-parallel`), each rank holds a subset of
experts — so the preshard `LLM(...)` **must be built with the same `enable_expert_parallel`
config as serve**, or a rank loads a shard whose expert-to-rank mapping doesn't match and the
model generates garbage *with no crash*. `ModelSpec.enable_expert_parallel` (derived from
`vllm_extra_args`) is the single source of truth passed to both preshard and serve so they
can't drift. Trinity also loads its source weights from a staged Volume path
(`/cache/trinity-base`, not the HF cache) with `trust_remote_code=True`, so `preshard` and
`copy_metadata` mount `acs-trinity-cache` and copy the custom `configuration_afmoe.py` beside
the shards for the loader to rebuild the skeleton.

**Per-model sharded Volume.** The sharded copy lives on `ModelSpec.sharded_volume_name`
(default `acs-sharded`). Trinity uses a dedicated `acs-trinity-sharded` because `acs-sharded`
already holds 405B's ~810 GB against Modal's 1 TB per-Volume cap.

## Phase 3 — `serve` (long-running + web endpoint)

**Purpose: the actual inference service.**

- 8×H200 (`gpu="H200:8"`) — fits 405B weights + KV cache. `scaledown_window` (currently
  `2*MINUTES` in the code; dev value, raise once stable) controls how long an idle container
  stays warm before Modal kills it: short saves money, too short means constant cold starts.
  Cold start here = pull image + mount Volume + stream ~100 GB shard per GPU + CUDA-graph
  capture → minutes (measured ~7 min on 2026-05-12), so this is worth tuning / possibly
  keeping one warm. Function `timeout=60*MINUTES`, `@modal.web_server` `startup_timeout=40*MINUTES`.
- Mounts: HF cache Volume (`acs-hf-cache`), vLLM compile-cache Volume (`acs-vllm-cache`,
  holds torch.compile / CUDA-graph artifacts so cold boots don't recompile), and the
  pre-sharded Volume (`acs-sharded`). Secrets: `huggingface-secret`, `vllm-api`.
- vLLM runs as a **subprocess** — `vllm serve <SHARDED_DIR or MODEL_NAME>
  --tensor-parallel-size 8 [--load-format runai_streamer_sharded] ...` — exposing its built-in
  OpenAI-compatible HTTP server on `localhost:VLLM_PORT`.
- Exposed via **`@modal.web_server(port=VLLM_PORT)`** + **`@modal.concurrent(max_inputs=32)`**
  — i.e. Modal proxies straight to vLLM's own server; there is no FastAPI layer yet.
- **Auth (Week 1 / current):** vLLM's native `--api-key`, value from the `vllm-api` Modal
  Secret (`VLLM_API_KEY`); every request carries `Authorization: Bearer <VLLM_API_KEY>`.
- **Planned follow-up (not yet built):** a thin **FastAPI shim** in front of vLLM —
  per-collaborator keys checked against a Secret-sourced table, structured request logging
  (ts, ip, key_id, endpoint, n_in, n_out, status; no bodies), proxy to localhost vLLM, own
  health endpoint. When that lands the serve fn would move to `@modal.asgi_app()`. Tracked in
  the design doc §Auth. The original Phase-3 plan (pasted above this doc's origin) described
  that end state, not today's code.

## Cold-start knobs (CUDA graphs, torch.compile, FAST_BOOT)

These are vLLM/PyTorch concepts, not Modal features — but they're the bulk of the ~7-min
405B cold boot, so they live here. One coupled story: **cold-start latency ↔ steady-state
decode throughput.**

- **CUDA graph** = a recorded, replayable sequence of GPU kernels. At small batch sizes,
  per-kernel *launch overhead* dominates and the GPU idles between kernels; capturing the
  sequence once and replaying it as a single launch removes most of that → faster decode. A
  captured graph is fixed to one input shape, so vLLM captures one per batch size —
  `--cuda-graph-sizes 1,2,4,8,16,32,64` in `serve` pre-captures those; other sizes fall back
  to eager kernel launches. Cost: capture happens at startup, takes time + extra GPU memory.
- **torch.compile** — vLLM also compiles model code at startup. The compiled artifacts can be
  cached to disk; that's what the `acs-vllm-cache` Volume (`/root/.cache/vllm`) is for — a
  second cold boot skips recompilation. (Graph *capture* still re-runs each boot.)
- **`enforce_eager` / `--enforce-eager`** — skips torch.compile and CUDA-graph capture
  entirely. Fast boot, slower decode (every step pays launch overhead). `preshard` uses
  `LLM(..., enforce_eager=True)` — correct, a one-shot weight dump has no steady state to
  optimize. `serve` does *not* set it (and passes `--cuda-graph-sizes`), i.e. it accepts the
  long capture for fast decode — which only pays off if the container stays warm, hence the
  `scaledown_window` tuning.
- **`FAST_BOOT`** — *not* a vLLM or Modal flag; it's an env var that **Modal's vLLM example
  app** invented as a single switch for the above tradeoff: `FAST_BOOT=1` → pass
  `--enforce-eager` (quick boot, churn-friendly); `FAST_BOOT=0` → full compile + graph capture
  (slow boot, fast serving). Our `modal_app.py` doesn't use this var — `preshard` is hard-wired
  to the eager side, `serve` to the compiled side.

## The through-line

Phases 1→2 are build-time, run-once, idempotent steps that turn a gated HF repo into a
fast-loading local artifact. Phase 3 is the runtime that consumes it and wraps it with the
operational concerns (auth, logging, scaling) a bare vLLM server doesn't provide. Result: a
deploy/restart costs seconds-to-minutes, not an 810 GB download.

## Status check vs. current Modal / vLLM docs (checked 2026-05-12)

- **`gpu="H200:8"`** — valid. Modal lists `H200` (and `B200`) and supports up to 8 GPUs /
  1,536 GB per container. Caveat from the docs: *"requesting more than 2 GPUs per container
  will usually result in larger wait times."*
- **`save_sharded_state`** — still present in current vLLM (`examples/offline_inference/
  save_sharded_state.py` + `load_sharded_state.py` on `main`), not deprecated. Writes shards
  named `model-rank-{rank}-part-{part}.safetensors`. Known rough edge: the saved state
  doesn't include the GPU P2P-access cache (vllm-project/vllm #10967), so you may still pay a
  P2P probe at boot.
- **`--load-format runai_streamer_sharded`** — current vLLM extension (Run:ai Model Streamer).
  Loads the same `model-rank-*-part-*` shards but streams them faster; tune with
  `--model-loader-extra-config '{"concurrency":16,"memory_limit":5368709120}'`. Requires the
  Run:ai streamer dep (`pip install vllm[runai]` / `runai-model-streamer`) in the image — the
  design doc flags this flag's availability as version-dependent; if it's ever missing, fall
  back to plain `--load-format sharded_state`.
- **Modal's own vLLM example** (`modal.com/docs/examples/vllm_inference`) uses a simpler
  shape than ours: weights cached in a Volume (no pre-shard step), vLLM launched via
  `subprocess.Popen` behind `@modal.web_server`, single H200, a `FAST_BOOT` toggle trading
  torch.compile/CUDA-graph capture against latency, JIT artifacts cached in a Volume. Our
  extra `preshard` phase (and the eventual `@modal.asgi_app` auth/logging shim) are deliberate
  additions for the 405B size and the multi-tenant requirement — not things Modal's example
  covers.

Sources: [Modal vLLM example](https://modal.com/docs/examples/vllm_inference) ·
[Modal GPU guide](https://modal.com/docs/guide/gpu) ·
[vLLM Run:ai Model Streamer](https://docs.vllm.ai/en/stable/models/extensions/runai_model_streamer/) ·
[vLLM save_sharded_state.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/save_sharded_state.py) ·
[vLLM #10967](https://github.com/vllm-project/vllm/issues/10967)
