---
title: Activation harvesting — inline probing + bulk harvest pipeline
status: current
updated: 2026-08-10
owner: platform@example.org
---

# Activation harvesting — how researchers get residual-stream activations

Two capture paths, split by scale. Both use the same layer convention: captured
layer `k` == HF `hidden_states[k+1]` == the output of decoder block `k`.

- **Inline** (probing, ≲10k examples, and all steering): `POST /v1/completions`
  with `output_residual_stream` / `apply_steering_vectors`, served by the
  per-model vLLM-Lens activation engine. Contract:
  [`../design/activation-api-contract.md`](../design/activation-api-contract.md).
  Ceiling ~384 tok/s (the HTTP fp32→zstd→base64 codec, not the GPU).
- **Bulk** (SAE-scale corpora): the offline harvester below, reachable
  self-serve via `POST /v1/harvest` or operator-run via `modal run`.

## Bulk harvester (`serving/harvest_offline.py`)

One Modal GPU function per model (apps `acs-llama-8b-harvest`,
`acs-trinity-truebase-harvest`, `acs-llama-405b-harvest`, all deployed
2026-07-17) captures via **offline vLLM-Lens** (`LLM.generate` +
`output_residual_stream`) — the default for every model since 2026-07-23
(big models since ACS-268; 8B switched after a parity gate vs the HF
reference: worst per-layer mean cosine 0.99992 on layers 8/16/24). One
PRE-final-norm capture convention across all models and the inline engine.
The plain HF-transformers `output_hidden_states=True` path remains as the
reference implementation (`HARVEST_USE_VLLM=0`) — parity ground truth, and
the only source of the HF-style POST-final-norm last layer. vLLM's native
`extract_hidden_states` was tried first and abandoned: fast on 8B but blocked
upstream on both production big models. _(researchlog 2026-07-15, 2026-07-21,
parity 2026-07-23)_

Pipeline, in order:

1. **Batched padded forwards** — right padding + attention mask, longest-first
   packing, padded footprint capped by `HARVEST_MAX_BATCH_TOKENS` (16384).
   Default batch size 8 on single-GPU, 1 (sequential) on the multi-GPU big
   models, where batching is opt-in because parity was verified on 8B only
   (batched-vs-sequential worst per-layer cosine 0.9997). Only the requested
   layers are stacked — stacking all layers first cost ~10× the needed peak
   GPU memory. _(from researchlog 2026-07-17)_
2. **Storage backend — one of two, chosen by `HARVEST_UPLOAD`** (ACS-278;
   `select_storage_backend()` is the single seam). Both overlap storage with the
   next batch's forward and back-pressure the capture loop so shards can't pile up
   in RAM.
   - **Go-live (`HARVEST_UPLOAD=1`) — straight to S3, no Volume.** Each shard is
     serialized in memory and streamed **directly** to the Railway bucket
     (`activation-harvest`, iad; Modal secret `activation-harvest-bucket`) by a
     **pool of `HARVEST_UPLOAD_CONCURRENCY` uploaders** (default 4), each doing
     boto3 multipart (`HARVEST_MULTIPART_CHUNK_MB` part size,
     `HARVEST_MULTIPART_CONCURRENCY` parts in flight per object) with
     botocore retries/timeouts. The Modal Volume and its `.commit()` are skipped
     entirely — S3 is the **only** durable copy, so a hard upload failure loses
     the GPU pass (transient blips are absorbed by the retries). Optional gzip
     (`HARVEST_COMPRESS=1`, off by default — bf16 is near-incompressible): shard
     keys gain a `.gz` suffix and the manifest records `compression`.
   - **Smoke (no bucket).** A single worker thread writes shards to the
     `acs-activation-harvest` Volume with one final `.commit()` after the manifest
     (ACS-270); the Volume is then the only durable store. `verify_shard` reads it
     back. This path exists so a smoke needs no bucket provisioned.
3. **Finalize + manifest-last invariant.** stats then `manifest.json` are written
   **last**, and only after every shard has landed with no errors, so "manifest
   present ⇒ run complete" holds for both backends. On the S3 path a failed shard
   aborts before the manifest is written (a manifest referencing a missing shard
   is silent corruption; orphaned shards with no manifest are harmless — a re-run
   with the same `run_id` overwrites the keys). Presigned GET URLs (7-day TTL) are
   returned for researcher download — verified credential-free. Returned timings
   include `upload_wall_s` and a wall-clock `upload_mb_s` so throughput is
   measurable (the bottleneck at scale, per Tilde Activault).

**Shard layout (v2, Activault-style):** per prompt `prompt_{i}`
(`[n_layers_kept, n_tokens, hidden]` bf16) **and** `tokens_{i}` (int32 input
ids, so autointerp never re-tokenizes); run-level `stats.safetensors` with
token mean/std per kept layer (BOS excluded — attention-sink outlier) plus
per-layer `mean_token_norm` in `manifest.json` (`layout_version: 2`).

**Big-model loading** (8×H200): `device_map="auto"` with a 110 GiB per-GPU cap
+ CPU/disk offload spill (Trinity's MoE expert-weight conversion OOMs without
headroom); `trust_remote_code=False` (transformers ≥5.13 ships native
`AfmoeForCausalLM`); Trinity loads from its local `/cache/trinity-base` staged
copy; 150-min timeout (405B streams ~810 GB off the cache Volume, ~50-min cold
load). _(from researchlog 2026-07-15)_

**Deploy caveat:** `HARVEST_UPLOAD`, `HARVEST_MAX_BATCH_TOKENS`,
`HARVEST_URL_TTL_S`, and the ACS-278 uploader knobs (`HARVEST_UPLOAD_CONCURRENCY`,
`HARVEST_MULTIPART_CONCURRENCY`, `HARVEST_MULTIPART_CHUNK_MB`, `HARVEST_COMPRESS`)
are read at module load and baked into the image env — local and remote module
evaluation must agree or Modal's dependency reconstruction fails with an
object-count mismatch. A deploy-time override of any of them therefore reaches the
container. Deploy with
`HARVEST_UPLOAD=1 MODEL_ID=<model> modal deploy serving/harvest_offline.py`.
_(from researchlog 2026-07-17)_

## Self-serve API (ACS-245)

`POST /v1/harvest` on the wrapper (bearer key, completions scope) validates
prompts (count + estimated-token + body-size caps), enforces a per-key monthly
job quota (`api_keys.monthly_harvest_budget`) and a running-job cap (default 1,
race-free via `SELECT … FOR UPDATE` + pending-row-before-spawn), then spawns
the deployed harvest app. `GET /v1/harvest/{job_id}` polls with lazy Modal
reconciliation and returns the presigned URLs plus `urls_expire_at` on
completion. An optional `?wait=<0..60>` holds the request open until the job is
terminal; the number of *waiting* long-polls is bounded (ACS-321) — per key
(`harvest_max_longpoll_per_key`, default 4 = small lane 3 + big lane 1) and
overall (`harvest_max_longpoll_total`, default 8) — because `poll_harvest` runs
on the event loop's default thread pool shared with `/v1/completions` work.
Over the ceiling the poll returns early with the current state and an
`X-Acs-Longpoll: declined` header (a normal `200`, paced by `Retry-After`), so a
flood of waiters can't starve completions. The harvest app name derives from the
registry (`harvest_app_name` override where the alias differs — Trinity). Jobs
live in the `harvest_jobs` table (migrations 0035 + 0041).

`DELETE /v1/harvest/{job_id}` (ACS-344) cancels an owned `pending`/`running`
job: it marks the row `cancelled` (a terminal state, so it no longer occupies a
concurrency slot — the owner can resubmit at once) under a guarded
`WHERE status IN ('pending','running')` UPDATE (a poll that finalized the job in
the race window wins → 409), then best-effort calls
`modal_ops.cancel_harvest(call_id)` (`FunctionCall.cancel(terminate_containers=True)`)
to free the GPU. The DB commit precedes the Modal RPC, so the slot frees even if
the terminate call is slow or fails; on failure the job is still `cancelled` and
the container merely runs to its own timeout. A cancel of a still-`pending` job
(no `modal_call_id`) never engaged a GPU and does not count against monthly
quota; a cancel after the spawn does. The two retryable 429s
(`harvest_concurrency_exceeded`, `harvest_capacity_exceeded`) carry a
`Retry-After: 30` header (`HARVEST_RETRY_AFTER_S`).

**Wall-clock accounting (ACS-343).** A done job's `GET` payload carries an
`accounting` block — `wall_clock_s` (wrapper-observed spawn→finalize),
`in_container_s` (sum of the harvest function's phase timings), and
`unaccounted_s` (the difference). The harvest `timings` dict is assembled inside
the Modal function, so it is blind to time spent BEFORE the function body runs —
the FunctionCall enqueued waiting for a container/GPU slot, plus container
scheduling/boot — and to the wrapper's poll cadence. `accounting` surfaces that
gap so a long wall clock on trivial work (a one-prompt job showed ~1025s wall
clock with `load_s: 0` and ~1.6s of timings) is explicable; a large
`unaccounted_s` with `load_s≈0` points at Modal queue/scheduling, not GPU work.

## Measured performance profile (8B, 1×L40S)

| Stage | Rate / share |
|---|---|
| forward (sequential) | 5,012 tok/s — ~10% of pipeline; 13× the inline ceiling |
| shard save (local disk) | ~1.1 GB/s |
| Volume commit | ~195 MB/s — ~70% of pipeline pre-overlap; **skipped on the `HARVEST_UPLOAD=1` path (ACS-278)** |
| bucket upload | 23.8 MB/s single-threaded — was the end-to-end bottleneck; now a concurrent multipart pool (ACS-278, TB-scale gain not yet measured) |

The forward pass is not the bottleneck; storage and above all upload are —
which is why the go-live path skips the Volume double-write + commit entirely and
streams straight to S3 through a concurrent uploader pool (ACS-278), and why a
Goodfire-style instrumented-engine fork is not worth building at our scale.
The layer-subsetting lever is now the default: an unspecified `layers`
harvests the quartile subset (blocks at ~25/50/75% depth,
`default_layer_indices()` in `serving/harvest_offline.py`); pass `"all"`
(API) / `--layers all` (CLI) for every block. The uploader-pool + multipart
levers are now shipped (ACS-278, `HARVEST_UPLOAD_CONCURRENCY` /
`HARVEST_MULTIPART_*`); remaining levers if 100M+-token harvests materialize:
bucket-region locality, and re-measuring the pool's sustained MB/s at TB scale
(`upload_mb_s` / `upload_wall_s` in the returned timings).
_(from researchlog 2026-07-17; upload path ACS-278, 2026-08-10)_
