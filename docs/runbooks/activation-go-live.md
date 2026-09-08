---
title: Activation harvesting/steering — go-live runbook
status: current
updated: 2026-07-23
owner: platform@example.org
---

# Activation harvesting/steering — go-live runbook (ACS-199)

The wrapper-side code to expose activation harvesting + steering through the API
(per-user key, routing, quota, telemetry, `/v1/models` flag) is in the
consolidated PR **#199** (contract verified against the live vLLM-Lens engine),
plus the shard-guard fix **#198** (item 5) that makes Trinity/405B activation
redeploys safe. **Merging that code does NOT turn the feature on** — the wrapper
only routes activation requests for a model whose registry entry has an
`activation_upstream_url`, and prod doesn't have one yet.

This runbook enables it for **all three** models (8B, 405B, Trinity) — all now
unblocked. Note the vLLM version: the activation engines run **vLLM 0.19.1 +
vLLM-Lens** (0.23 breaks capture — vllm-lens reads `GPUModelRunner.input_batch`,
removed in 0.23; researchlog 2026-07-06). Item 5's guard keeps the 0.19.1
activation engine from loading incompatible 0.23 shards (Trinity → HF fallback,
405B dense → its shards are cross-version-loadable).

## Preconditions
- #199 + #198 merged to `main` and deployed to Railway.
- The activation Modal apps are deployed (`max_containers=1`, #184). Confirm:
  `modal app list | grep activation`. Redeploy each with the item-5 code first
  (`MODEL_ID=<id> modal deploy modal_app_activation.py`) — safe now (scale-to-zero;
  no forced boot).

## Step 1 — wire the activation URLs into the wrapper registry
The wrapper builds its model table from `MODELS_REGISTRY_JSON` on Railway
(`settings.py:parsed_models_registry`). Add `activation_upstream_url` **and** a
per-model `activation_max_prompt_tokens` **and** `n_layers` to each model's entry.

The prompt cap is load-bearing. Capture returns the residual stream for every
requested layer, and since ACS-250 the response is **streamed** through the
wrapper — so the binding constraint is the RESPONSE SIZE the client downloads
(payload ≈ layers × prompt_tokens × d_model × 2 B), not wrapper RAM. Size the
cap per model for an **all-layer** capture:

```jsonc
"llama-8b": {   // 32 × 4096 ≈ 256 KB/token → generous cap
  "activation_upstream_url": "https://<workspace>--acs-llama-8b-activation-serve-activation.example.modal.run",
  "activation_max_prompt_tokens": 512,
  "n_layers": 32
},
"llama-405b": { // 126 × 16384 ≈ 4 MB/token → TIGHT cap
  "activation_upstream_url": "https://<workspace>--acs-llama-405b-activation-serve-activation.example.modal.run",
  "activation_max_prompt_tokens": 32,
  "n_layers": 126
},
"trinity-truebase": { // 60 × 3072 ≈ 369 KB/token
  "activation_upstream_url": "https://<workspace>--acs-trinity-truebase-activation-serv-f7a95c.example.modal.run",
  "activation_max_prompt_tokens": 256,
  "n_layers": 60
}
```

`n_layers` lets the wrapper scale the cap when a request asks for a **subset**
of layers (ACS-317): payload scales with the layer count, so a 1-of-32 capture
on llama-8b is allowed 32× the prompt at the same response size, clamped to
`max_model_len`. Leave `n_layers` unset and the flat cap applies to every
request — safe, but subset requests get no benefit, and `/v1/models` won't
publish the layer count clients need to compute their own ceiling.

> ⚠️ **After enabling `n_layers`, re-check activation latency.** The flat cap
> also incidentally bounded prefill work: llama-8b's engine runs
> `@modal.concurrent(max_inputs=16)` and was load-tested (ACS-249) at a
> 512-token cap. A 1-of-32 request can now admit an 8192-token prefill, 16 of
> them concurrently. vLLM queues rather than OOMs, so the risk is tail latency,
> not failure — but the shared `upstream_timeout_s` gap below makes it worth a
> measurement.

> ⚠️ **Verify each exact URL first** — the values above are from
> `notebooks/activation_tour.ipynb` (Trinity's has a hash suffix
> `…-serv-f7a95c.modal.run` — don't assume the naming pattern). Confirm with
> `modal app list` / each app's web endpoint before pasting.

> **405B egress:** even at the 32-token cap, an all-layers 405B capture is ~130 MB
> — fine for RAM (bounded) but heavy on the wire. It's for **probing**; bulk /
> long-context 405B capture is the offline path (ACS-198), not inline. Start 405B
> with a conservative cap and raise it once real usage + RAM headroom are observed.

## Step 2 — deploy the wrapper (runs migration 0031)
Railway auto-deploys `main`. The release phase runs `alembic upgrade head`, which
applies **migration 0031** (activation quota + telemetry columns — additive,
reversible, reviewer-verified on a fresh Postgres with zero autogenerate drift).

**Confirm the migration ran** (the one thing to actually check):
```bash
# in the Railway shell / against the prod DB:
alembic current          # should show 0031_activation_quota_telemetry
# or:
psql "$DATABASE_URL" -c "\d api_requests" | grep activation
```
If Railway's release command is not already `alembic upgrade head`, run it once
manually after deploy.

## Step 3 — (optional) set per-user activation quotas
`api_keys.monthly_activation_budget` defaults to `0` = unlimited, so nothing is
capped until you set a value. Activation requests each wake an 8×H200-class
engine, so consider a modest cap per beta key, e.g.:
```sql
UPDATE api_keys SET monthly_activation_budget = 500 WHERE …;   -- 500 activation reqs/month
```
(Idle cost is already $0 via scale-to-zero + `max_containers=1`; this caps the
active side.)

## Step 4 — smoke it (author's pre-announce check)
Run the smoke against the **running wrapper** with a real per-user key — this is
the faithful end-to-end check (real ASGI + real cold boot), not TestClient:

```bash
WRAPPER_URL=https://infra.acsresearch.org \
API_KEY=sk-… \
scripts/activation/wrapper_smoke.sh
```
It exercises: `/v1/models` advertises `activations:true` · a capture request
(`output_residual_stream: true`, all layers) returns the `activations` payload
through the wrapper · a plain completion carries none · a non-activation model is
a clean `400 activations_unsupported`. Run it per model (`MODEL=llama-8b`, then
`405b`, `trinity-truebase`). **First capture after idle pays a cold boot** (8B: ~30–40 s
GPU-snapshot restore, or ~165 s full rebuild on a snapshot-miss region — ACS-218;
~5 min 405B, ~20 min Trinity) — the script uses a long timeout.

## Step 5 — tell the testers
Once smoke passes, the wrapper-mediated path replaces the raw-Modal-URL +
shared-`VLLM_API_KEY` path in `notebooks/activation_tour.ipynb`. Point beta users
at the wrapper contract (`docs/design/activation-api-contract.md`) and kick off
the survey-ungated second-wave recruitment that was gated on this landing
(ACS-172).

## Known gaps / follow-ups (not blockers for probing-scale v1)
- Layer subsets over HTTP work since vllm-lens 1.2.0 (ACS-266; send the list
  as a `json.dumps`'d string in `vllm_xargs`). Long-context / bulk (SAE-scale)
  capture still needs the two-stage download (ACS-198), not inline — the
  base64-in-JSON transport and per-request RAM stay the bottleneck. The per-model `activation_max_prompt_tokens` cap keeps inline safe
  for probing; 405B is the tight one.
- Activation-engine health isn't surfaced in `/health` yet (isolated out by the
  breaker split — a deliberate follow-up).
- Big-model activation wants a separate `activation_upstream_timeout_s` (the
  request timeout is still shared with the workbench entry).
- `vllm-lens` 0.23 compatibility: RESOLVED 2026-07-23 (ACS-296) — 1.2.0 passes
  the full parity gate on vLLM 0.23.0 (llama-8b), so retiring the 0.19.1
  activation image is unblocked; per-model parity reruns (Trinity/405B) gate
  the actual cutover — unification tracked in ACS-295.


## Harvest new-kwarg deploys — deploy order (ACS-320, ACS-319)

`POST /v1/harvest` has gained new keyword arguments forwarded to the deployed
`harvest()` Modal function: `project_onto` (ACS-320) and `add_special_tokens`
(ACS-319). **Deploy the harvest apps before the wrapper** — or at least before
anyone sends the new field:

```bash
HARVEST_UPLOAD=1 MODEL_ID=llama-8b         modal deploy serving/harvest_offline.py
HARVEST_UPLOAD=1 MODEL_ID=llama-405b       modal deploy serving/harvest_offline.py
HARVEST_UPLOAD=1 MODEL_ID=trinity-truebase modal deploy serving/harvest_offline.py
```

**`HARVEST_UPLOAD=1` is required, not optional.** `harvest_offline.py` reads
`HARVEST_UPLOAD` **at module load** — while building the app, not inside the
function — to pick the bucket-secret list attached to the image and to bake the
`UPLOAD` flag the function later checks (see the `activation-offline-harvest.md`
go-live note). Deploy *without* it and the app bakes `UPLOAD=0`: no bucket secret
is attached and shards stop uploading, so `/v1/harvest` returns no downloadable
URLs, silently. Always set it.

The wrapper omits each new kwarg entirely when a job doesn't use it (projections
off; `add_special_tokens` only sent when the caller set it), so ordinary
harvests keep working against an older deployment. A job that *does* use the new
field against an old app fails at Modal with an unexpected-keyword error — loud,
not silent, and the fix is the deploy above.

**Straight-to-S3 uploader (ACS-278) — recovery + tuning.** On a `HARVEST_UPLOAD=1`
deploy shards now stream **directly to the bucket** with no Modal Volume copy, so
S3 is the only durable store. botocore absorbs transient blips (4 retries); a
*hard* upload failure fails the job with "the whole GPU pass must be re-run" —
there is **no pass-2 recovery from the Volume** any more, so re-submit the harvest.
The manifest is uploaded last and only on a fully-successful run, so a bucket that
has a `manifest.json` for a `run_id` is complete. Optional deploy-time knobs (all
baked into the image, all safe to omit — sensible defaults): `HARVEST_UPLOAD_CONCURRENCY`
(uploader threads, 4), `HARVEST_MULTIPART_CHUNK_MB` (64), `HARVEST_MULTIPART_CONCURRENCY`
(4), `HARVEST_COMPRESS` (0 — gzip shards; leave off, bf16 is near-incompressible).
`verify_shard` only works for no-bucket smoke runs (it reads the Volume); to check
an uploaded run, GET the presigned URLs the job returned. Throughput is in the
job's `timings`: `upload_mb_s` + `upload_wall_s`.
