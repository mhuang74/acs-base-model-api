---
title: Modal app architecture
status: current
updated: 2026-07-22
owner: platform@example.org
---

# Modal app architecture — how a `/v1/completions` request returns text

Reference walkthrough of every layer a prompt traverses from a laptop to a generated
continuation on a Modal-hosted Llama server. (Moved here from `researchlog.md` —
this is reference documentation, not a progress log.)

## 1. HF account plumbing (one-time, outside the stack)
- Accept the Meta license on https://huggingface.co/meta-llama/Llama-3.2-1B for your HF account.
- Generate an HF access token ("Read" type so it has gated-repo access).
- Put the token in `tokens.txt` locally and into a Modal Secret named `huggingface-secret` so both local and remote code can authenticate to HF.

## 2. Modal account + API-key plumbing (one-time)
- `pip install -e .` → gets the `modal` Python SDK on your laptop.
- `modal token new` → Modal CLI stores an OAuth token for your workspace.
- `modal secret create vllm-api VLLM_API_KEY=<random>` → a server-side shared bearer token the vLLM process will accept; same value lives in your local `tokens.txt`.

## 3. Container image (one-time per change)
Built declaratively in `modal_app.py`:
- Base: `debian_slim` with Python 3.12.
- `pip install vllm==0.19.1 huggingface_hub[hf_transfer]` with PyTorch CUDA 12 wheels.
- Env vars baked in: `HF_HUB_ENABLE_HF_TRANSFER=1` (fast downloader), `DEV_MODE=<0 or 1>` (model selector).

Modal caches the image; subsequent deploys are fast.

## 4. Persistent storage (Modal Volumes — one-time creation)
- `acs-hf-cache` → HF snapshot of the model weights.
- `acs-vllm-cache` → vLLM's torch-compile / CUDA-graph cache so cold boots don't recompile.

These survive across container restarts; the container mounts them at `/root/.cache/huggingface` and `/root/.cache/vllm`.

## 5. Weight staging (one-time per model)
`modal run modal_app.py::stage_weights`:
- Spawns a CPU container with the image.
- Mounts `acs-hf-cache` and the HF Secret.
- Calls `huggingface_hub.snapshot_download("meta-llama/Llama-3.2-1B")` — auth'd by `HF_TOKEN`, accelerated by `hf_transfer`.
- Blobs land on the Volume. Subsequent calls are no-ops (already cached).

## 6. Serving (per-deploy)
`modal serve modal_app.py` (or `modal deploy` for persistent):
- Modal provisions a container with 1×L40S (because `DEV_MODE=1`).
- Image starts; the `@modal.web_server(port=8000)` decorator tells Modal to forward HTTPS traffic on the ephemeral `*.modal.run` URL to port 8000 inside the container.
- The serve function body launches `vllm serve meta-llama/Llama-3.2-1B ...` as a subprocess inside the container. vLLM reads `VLLM_API_KEY` from the container's env and enables Bearer-token auth.
- vLLM boots:
  - Reads tokenizer/config from the HF cache on the mounted Volume (local FS, no network).
  - Loads weights into VRAM (bf16, ~2.3 GB).
  - Initializes the v1 engine, picks FlashAttention-3 backend.
  - Allocates paged KV cache (~32 GB on L40S after 8B BF16 weights, good for ~244k tokens).
  - Starts FastAPI on `0.0.0.0:8000`; registers `/v1/completions`, `/v1/models`, `/health`, etc.
- Modal's edge now proxies `https://<workspace>--acs-base-model-api-dev-serve-dev.example.modal.run/*` → `http://<container>:8000/*`.

### Boot-stage publishing during cold boot (ACS-272)

While vLLM boots, the container publishes authored stage events — `container_started` →
`weights_loading` → `weights_loaded` → `engine_ready` → `serving`, or `failed` if vLLM
exits mid-boot — into the shared Modal Dict `acs-boot-status`, keyed by app name
(`serving/boot_status.py`; stages derive from the same stdout markers as the ACS-17
lifetime CSV; the per-model activation apps publish the same way, keyed on their
own app name — ACS-276). The wrapper polls that Dict during a cold-boot wait
(`modal_ops.get_boot_status`, interpreted by `wrapper/boot_stage.py`) and surfaces the
stage in three places: the workbench cold-boot banner (replacing the old elapsed-time
guess), `: cold_boot stage=…` SSE keepalive comments on public streaming requests, and
`GET /v1/models/{id}/status`. No fresh Dict entry during a cold wait is reported as
"Waiting for Modal to allocate GPUs" — the pre-container phase nothing else can see.
Raw container logs never reach clients; only these authored stage keys/labels do.
Publishing and polling are both best-effort: on a Dict outage the SSE comments fall back
to plain `: keepalive`, the status endpoint reports `boot: null`, and the banner shows
the inferred waiting-for-GPU label (also what users see until the serving apps are
redeployed with the publishing code).

## 7. A request round-trip (per request)
Client:
```
POST https://<workspace>--acs-base-model-api-dev-serve-dev.example.modal.run/v1/completions
Authorization: Bearer <VLLM_API_KEY>
{"model": "meta-llama/Llama-3.2-1B", "prompt": "The capital of France is", "max_tokens": 8, "logprobs": 5}
```
Server side:
1. Modal edge terminates TLS, routes to the container.
2. FastAPI (inside vLLM) receives the POST.
3. vLLM auth middleware checks `Authorization: Bearer <key>` against `VLLM_API_KEY` env — 401s if missing/wrong.
4. Request body is parsed into a `SamplingParams` + a raw prompt.
5. vLLM tokenizes the prompt with the cached HF tokenizer → token IDs.
6. Engine schedules the request; prompt tokens go through prefill on the GPU.
7. Decoding loop: for each of 8 tokens, run a forward pass, apply sampling (temperature/top-p/seed), emit one token.
8. Response assembles: generated text, token IDs, top-k logprobs (completion side capped at 100 by the wrapper), `prompt_logprobs` if asked — including **full-vocabulary** `prompt_logprobs=-1` on vLLM 0.23 (the server runs `--max-logprobs -1` uncapped; the wrapper enforces the real bounds — ACS-191).
9. JSON response → FastAPI → Modal edge → HTTPS → your laptop.

Total time after warm boot: ~100–300 ms for ~10 tokens on a 1B model.

## 8. Validation (per test)
`scripts/validate.py` hits the live endpoint with the same auth path and runs 7 checks: basic completion, top-k logprobs, `prompt_logprobs`, logprobs cap probe, tokenizer parity (HF `AutoTokenizer` vs server token IDs byte-for-byte), sampling-param acceptance, seed reproducibility.

## 9. Teardown
Ctrl-C on `modal serve` → Modal stops the container → GPU is released → the `scaledown_window` countdown never has to matter. A new `modal serve` starts a fresh container; weights come from the Volume in ~15 s (not from HF), torch.compile is skipped (`FAST_BOOT=True`).

## What made this hard in practice
- `DEV_MODE` not propagating into the remote container — fixed by baking it into the image's env.
- Accidentally running preshard on 405B because of the default-to-production bug — fixed by flipping the default to `DEV_MODE=1`.
- logprobs cap evolution: vLLM 0.19.1 had no full-vocab support, so the top-k cap was **100**, set via `--max-logprobs 100` and kept in lockstep with the wrapper's `MAX_LOGPROBS` (ACS-84). On the 0.23 cutover this changed (ACS-191): `prompt_logprobs=-1` (full vocab) works on the V1 engine, the server now runs `--max-logprobs -1` (uncapped), and the wrapper — not the serve flag — enforces the real bounds (positive top-k ≤ 100, a full-vocab prompt-length gate, gzip on the response). The lockstep invariant is now "server uncapped or ≥ wrapper cap". Since ACS-198 the wrapper *streams* full-vocab bodies through without parsing them (`proxy.passthrough_post`, gzip on the fly), so the prompt-length gate (`FULL_VOCAB_MAX_PROMPT_TOKENS`, now 1024) protects the *model server's* serialization memory and the transfer window, not wrapper RAM.
- Local `HF_TOKEN` in `tokens.txt` differed from the Modal Secret's `HF_TOKEN`, causing tokenizer parity to 401 on our laptop while Modal was fine.
- Stale `~/.cache/huggingface/token` CLI-login cache fighting our env var; pinned by passing `token=` explicitly to `AutoTokenizer`.
