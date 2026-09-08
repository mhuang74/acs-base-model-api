"""Offline / bulk activation harvesting — in-process forward over a corpus to storage.

The inline ``/v1/completions?output_residual_stream=true`` path is for *probing*:
it returns every layer over HTTP as a base64 blob, fine for a few prompts but
capped ~384 tok/s (ACS-196) and RAM-bounded per request. For SAE-scale collection
you want to run the model in-process over a corpus and write raw bf16 tensors to
storage a researcher downloads by URL.

**One default capture backend: offline vLLM-Lens, for every model** (since the
8B parity gate of 2026-07-23; big models since ACS-268): batched ``LLM.generate``
+ ``output_residual_stream`` at true TP. vLLM-Lens forces ``enforce_eager``
(fine — prefill-only). It captures the PRE-final-norm residual for every layer —
one convention across all models AND matching the inline engine, so
harvest-derived steering vectors are consistent with the injection point.
8B parity vs the HF reference (layers 8/16/24, identical tokenization): worst
per-layer mean cosine 0.99992, in line with the 0.9997 batching-parity
precedent. Forward was never the 8B bottleneck (upload is, per Tilde Activault)
— the switch is for convention consistency, not speed (though it measured
128.6 vs 43.1 tok/s sequential-shape on the parity run).

The **HF ``transformers`` ``output_hidden_states`` path remains as the reference
implementation** (``HARVEST_USE_VLLM=0``): it's the parity ground truth, and the
only way to get the HF-style POST-final-norm last layer (its last layer is HF
``hidden_states[n]`` — see the manifest convention; vLLM-Lens is pre-norm).

Both write the SAME shard/stats/manifest layout, so downloaders don't care which
backend produced a run. The earlier vLLM-*native* ``extract_hidden_states`` harvester
was fast on 8B but blocked on the big models (405B couldn't load its drafter through
the sharded-state loader; Trinity hit a non-uniform-KV-page assertion); the offline
vLLM-Lens path (hook-based, no drafter/extra-KV-page) sidesteps both. ``HARVEST_USE_VLLM``
overrides the per-model default. Residual convention: captured layer k == block-k
output (== HF ``hidden_states[k+1]`` for the middle layers).

- One Modal GPU function loads the model (mounting the shared ``acs-hf-cache``
  weights Volume — no re-download), forwards prompts in **padded batches**
  (``--batch-size``: default 8 on the single-GPU 8B, 1 = sequential on the
  multi-GPU big models where batching is opt-in; padded footprint capped by
  ``HARVEST_MAX_BATCH_TOKENS``; right-padding + attention mask; per-prompt
  tensors are sliced back to their unpadded length), and keeps the requested
  ``hidden_states[k+1]`` block outputs.
- Two mutually-exclusive storage paths (``select_storage_backend``; ACS-278):
  * **Smoke (no bucket):** bf16 ``safetensors`` shards + ``stats.safetensors`` +
    ``manifest.json`` land in the ``acs-activation-harvest`` Modal Volume with one
    final ``.commit()``. The Volume is the only durable store.
  * **Go-live (``HARVEST_UPLOAD=1``):** shards are serialized IN MEMORY and
    streamed **STRAIGHT to the S3-compatible Railway bucket** by a POOL of
    concurrent multipart uploaders overlapped with the forward pass — the Volume
    is skipped entirely (no second copy, no commit). S3 is then the ONLY durable
    copy, so a hard upload failure loses the GPU pass (botocore retries transient
    blips; the manifest is uploaded LAST and only after every shard lands, so
    "manifest present in the bucket ⇒ run complete"). Returned as presigned GET
    URLs. Tunable via ``HARVEST_UPLOAD_CONCURRENCY`` / ``HARVEST_MULTIPART_*`` /
    ``HARVEST_COMPRESS``.
  Both layouts are identical for downloaders: each prompt's **input token ids
  travel with its activations** (``tokens_{i}``) and run-level token mean/std/norm
  land in ``stats.safetensors`` (SAE normalization + autointerp without
  re-tokenizing — the Activault layout).
- Every run returns phase ``timings``. Measured 2026-07-17 (8B, 1×L40S, 234
  prompts / 81.5k tok, all 32 layers, 21.4 GB): load 12s, forward 16s
  (~5k tok/s *sequential*), shard save 19s, Volume commit 110s — storage I/O is
  ~82% of the pipeline, the forward pass ~10% (ACS-218: the bottleneck is
  storage/upload, not compute; layer subsetting is the biggest lever).

Layer convention: captured layer ``k`` == HF ``hidden_states[k+1]`` == the output
of decoder block ``k`` (the post-block residual stream) — the same convention as
the inline ``output_residual_stream`` path. Select with ``--layers "8,16,24"``
(block indices); default is the quartile subset (~25/50/75% depth, see
``default_layer_indices``); ``--layers all`` keeps every block.

Run (smoke, no bucket needed):
    modal run serving/harvest_offline.py --prompts scripts/activation/sample_prompts.txt
    MODEL_ID=trinity-truebase modal run serving/harvest_offline.py --prompts prompts.jsonl

Go-live (operator, after provisioning the bucket + secret):
    HARVEST_UPLOAD=1 modal run serving/harvest_offline.py --prompts prompts.jsonl

The pure helpers (chunk_prompts / build_manifest / shard_key / parse_layers /
select_storage_backend) and the S3 uploader orchestration (S3ShardUploader /
finalize_s3_run / _put_bytes / _maybe_compress) stay torch-free (boto3 only lazily
inside the real network call) and are unit-tested in
``tests/test_harvest_offline.py`` with a fake S3 client.
"""

from __future__ import annotations

import json
import os

import modal

from acs_model_registry import get_model_config, get_model_spec
from serving.modal_resources import create_volumes, serve_volumes, sharded_dir_for_model
from serving.vllm_runtime import activation_can_load_sharded

# SigV4 presigned GET max is 7 days; clamp so an override can't mint URLs S3 rejects.
URL_TTL_S = min(int(os.environ.get("HARVEST_URL_TTL_S", str(7 * 24 * 3600))), 7 * 24 * 3600)
MINUTES = 60

# Upload is opt-in (operator go-live). Read LOCALLY at module load so it can gate
# the bucket Secret in the function definition — the Railway bucket + secret are
# provisioned as a separate step, so a plain smoke run must NOT require them.
UPLOAD = os.environ.get("HARVEST_UPLOAD", "0") == "1"

# Cap on a forward batch's PADDED token footprint (n_prompts × longest member):
# batch memory scales with this product, so a corpus of near-context-length
# prompts degrades to smaller batches instead of OOMing a paid multi-hour run.
MAX_BATCH_TOKENS = int(os.environ.get("HARVEST_MAX_BATCH_TOKENS", "16384"))

# Straight-to-S3 uploader tuning (ACS-278). Baked into the image below (like
# MAX_BATCH_TOKENS) so a deploy-time override reaches the container. Read here at
# module load only so the value can be baked; the behavior is per-run.
#   UPLOAD_CONCURRENCY        — shard-uploader worker threads (cross-shard parallelism)
#   UPLOAD_MULTIPART_CONCURRENCY — parts in flight per object (intra-shard parallelism)
#   MULTIPART_CHUNK_BYTES     — multipart threshold + part size (S3 floor is 5 MiB)
#   COMPRESS                  — gzip shards before upload (off: bf16 ~incompressible)
UPLOAD_CONCURRENCY = max(1, int(os.environ.get("HARVEST_UPLOAD_CONCURRENCY", "4")))
UPLOAD_MULTIPART_CONCURRENCY = max(1, int(os.environ.get("HARVEST_MULTIPART_CONCURRENCY", "4")))
MULTIPART_CHUNK_BYTES = max(
    5 * 1024 * 1024, int(os.environ.get("HARVEST_MULTIPART_CHUNK_MB", "64")) * 1024 * 1024
)
COMPRESS = os.environ.get("HARVEST_COMPRESS", "0") == "1"


# --------------------------------------------------------------------------- #
# Pure helpers (no torch / no boto3) — unit-testable.
# --------------------------------------------------------------------------- #
def chunk_prompts(prompts: list[str], shard_size: int) -> list[list[int]]:
    """Split prompt indices into shards of at most ``shard_size`` (>=1)."""
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    return [
        list(range(i, min(i + shard_size, len(prompts))))
        for i in range(0, len(prompts), shard_size)
    ]


def shard_key(run_id: str, shard_idx: int, compress: bool = False) -> str:
    """Object key for a shard. With ``compress`` the safetensors bytes are
    gzip-wrapped, so the key gains a ``.gz`` suffix (self-describing on the wire)."""
    return f"harvest/{run_id}/shard_{shard_idx:05d}.safetensors" + (".gz" if compress else "")


def manifest_key(run_id: str) -> str:
    return f"harvest/{run_id}/manifest.json"


def stats_key(run_id: str) -> str:
    return f"harvest/{run_id}/stats.safetensors"


def select_storage_backend(upload: bool) -> dict:
    """Path selection for a harvest run (ACS-278) — the single source of truth for
    where a run persists, so the two paths can never both run or both skip.

    ``HARVEST_UPLOAD=1`` streams shards STRAIGHT to the S3 bucket and skips the
    Modal Volume entirely (no shard write, no ``.commit()``): S3 is then the ONLY
    durable copy, so an upload failure loses the GPU pass. A no-bucket smoke keeps
    the Volume path (the Volume is then the only durable store) with the single
    final commit. Exactly one of ``to_s3`` / ``to_volume`` is True.
    """
    return {"to_s3": upload, "to_volume": not upload, "commit": not upload}


def parse_layers(spec: str | None) -> list[int] | str | None:
    """Parse a "8,12,16" layer spec; None/'' -> default subset, "all" -> every block."""
    if not spec:
        return None
    if spec.strip().lower() == "all":
        return "all"
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def default_layer_indices(n_layers: int) -> list[int]:
    """Default harvest subset: blocks at ~25/50/75% depth.

    Interp work concentrates in middle depths (SAE/steering/probing sweet spots
    sit roughly between ¼ and ¾ of the stack), and storage/upload — not the
    forward pass — is the bulk-harvest bottleneck (ACS-218), so a 3-layer
    default cuts the shipped bytes ~10–40× vs every block. Pass "all" (CLI) or
    an explicit list (API) to override.
    """
    return sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4})


def batch_prompt_indices(
    idxs: list[int], sizes: list[int], batch_size: int, token_budget: int | None = None
) -> list[list[int]]:
    """Group a shard's prompt indices into forward batches.

    Longest-first (by ``sizes``, estimated tokens per prompt, indexed by GLOBAL
    prompt index) so each padded batch wastes as little compute as possible — the
    Activault length-bucketing idea at shard granularity. ``token_budget`` caps a
    batch's PADDED footprint (n_prompts × longest member, the quantity GPU memory
    actually scales with), so a corpus of near-context-length prompts degrades to
    smaller batches instead of OOMing the whole run. Order within the shard
    doesn't matter: tensors are keyed per prompt index.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = sorted(idxs, key=lambda i: -sizes[i])
    batches: list[list[int]] = []
    for i in order:
        # longest-first ⇒ the current batch's padded length is fixed by its first
        # member, so the padded footprint of adding one more is (len+1) * first.
        if batches and len(batches[-1]) < batch_size:
            first = sizes[batches[-1][0]]
            if token_budget is None or (len(batches[-1]) + 1) * max(first, 1) <= token_budget:
                batches[-1].append(i)
                continue
        batches.append([i])
    return batches


_PROJECTION_TOKEN_CHUNK = 4096


def project_activations(x, directions):
    """``[n_kept, n_tok, hidden]`` -> ``[n_kept, n_tok, n_dirs]`` float32.

    Two things matter here and both are about memory, not math (ACS-320 review):

    * Run this **on whatever device the activations are already on**, before the
      host copy. Projecting after ``.cpu()`` would send the full hidden width
      over the D2H link — the 512x reduction would never reach the wire, which
      is the entire point of the feature.
    * Widen a **chunk of tokens at a time**. ``x.float()`` on the whole tensor
      allocates a second copy at 2x the width: for 405B at full context that is
      ~3.2 GB -> ~9.6 GB peak, and ``layers: "all"`` would end up *worse* than a
      raw harvest. Widening a whole layer at a time is not enough either — one
      405B layer at 32k is still 2.1 GB, and here that lands in **VRAM**, on top
      of vLLM's KV reservation. A token chunk bounds the transient to
      ``TOKEN_CHUNK x hidden x 4`` (~270 MB at 405B) regardless of context
      length, and writing into a preallocated output avoids holding both a list
      of per-layer results and the stacked copy.
    """
    import torch

    n_kept, n_tok, _ = x.shape
    d = directions.to(device=x.device, dtype=torch.float32)
    out = torch.empty((n_kept, n_tok, d.shape[0]), dtype=torch.float32, device=x.device)
    dT = d.T
    for k in range(n_kept):
        for start in range(0, n_tok, _PROJECTION_TOKEN_CHUNK):
            end = min(start + _PROJECTION_TOKEN_CHUNK, n_tok)
            out[k, start:end] = x[k, start:end].float() @ dT
    return out


def decode_projection_directions_np(codec: dict, hidden_size: int):
    """Validate + decode projection directions to a normalized float32 ndarray.

    numpy-only so the risky parts — dtype handling, the bf16 bit-pattern widen,
    shape/size agreement, normalization — are unit-testable without torch on the
    machine (the torch wrapper below is a one-liner). Returns ``[n_dirs, hidden]``.

    Directions are L2-normalized: projecting onto a NON-unit vector silently
    rescales every number and the caller can't see it in the output, so we
    normalize and record that in the manifest rather than trusting the input
    (ACS-320).

    Every failure here is a ValueError with an actionable message, and this runs
    BEFORE the model loads — a bad payload costs seconds, not a 50-minute 405B
    load.
    """
    import base64

    import numpy as np

    if not isinstance(codec, dict):
        raise ValueError("project_onto must be a tensor-codec object")
    shape = codec.get("shape")
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError(
            f"project_onto.shape must be 2-D (n_directions, hidden); got {shape!r}"
        )
    n_dirs, hidden = int(shape[0]), int(shape[1])
    if hidden != hidden_size:
        raise ValueError(
            f"project_onto hidden dim {hidden} != this model's hidden size "
            f"{hidden_size}; directions must come from the same model"
        )
    if n_dirs < 1:
        raise ValueError("project_onto needs at least one direction")
    if codec.get("compression") not in (None, "none", ""):
        raise ValueError(
            f"project_onto compression {codec.get('compression')!r} is not supported; "
            "send raw (uncompressed) codec data"
        )
    dtype = str(codec.get("dtype", "")).lower()
    np_dtype = {"float32": "float32", "float16": "float16", "bfloat16": "uint16"}.get(dtype)
    if np_dtype is None:
        raise ValueError(
            f"project_onto dtype {dtype!r} unsupported; use float32, float16 or bfloat16"
        )
    try:
        # Whitespace-tolerant, and it must stay in sync with the wrapper's check:
        # the wrapper forwards the caller's codec dict verbatim, so anything it
        # accepts has to decode here too — otherwise a line-wrapped payload
        # passes validation at submit time and then dies on a GPU.
        raw = base64.b64decode("".join(str(codec["data"]).split()), validate=True)
    except Exception as exc:
        raise ValueError(f"project_onto.data is not valid base64: {exc}") from exc

    arr = np.frombuffer(raw, dtype=np_dtype)
    if arr.size != n_dirs * hidden:
        raise ValueError(
            f"project_onto.data holds {arr.size} values but shape {shape} needs "
            f"{n_dirs * hidden}"
        )
    if dtype == "bfloat16":
        # bf16 bit pattern stored as uint16 → widen into the high half of an f32.
        arr = (arr.astype("uint32") << 16).view("float32")
    out = arr.astype("float32").reshape(n_dirs, hidden)
    if not np.all(np.isfinite(out)):
        raise ValueError("project_onto contains non-finite values")
    # float64 norm: a direction big enough to overflow float32 would give
    # norm=inf, and out/inf is a silent all-zero direction that reports no error
    # and returns 0.0 for the whole run (ACS-320 review).
    norms = np.linalg.norm(out.astype("float64"), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("project_onto contains a zero-norm direction")
    # No non-finite check on `norms`: `out` is already all-finite float32, and a
    # float64 sum of at most `hidden` squares below 3.4e38 cannot overflow — it
    # would take hidden ~1e540. The float64 accumulation is what buys that.
    return (out.astype("float64") / norms).astype("float32")


def decode_projection_directions(codec: dict, hidden_size: int):
    """``decode_projection_directions_np`` as a torch tensor (harvest-side use)."""
    import torch

    return torch.from_numpy(decode_projection_directions_np(codec, hidden_size).copy())


def build_manifest(
    *,
    run_id: str,
    model_id: str,
    hf_repo: str,
    dtype: str,
    layer_indices: list[int],
    prompts: list[str],
    shards: list[dict],
    batch_size: int = 1,
    mean_token_norm: dict[str, float] | None = None,
    capture: str = "hf-transformers output_hidden_states",
    n_projection_dirs: int | None = None,
    add_special_tokens: bool = True,
    compression: str = "none",
) -> dict:
    """Assemble the manifest a downloader reads to locate + decode each prompt.

    ``compression`` (ACS-278) records how the SHARD objects are encoded on the
    wire: ``"none"`` = raw ``safetensors`` (``load_file`` directly); ``"gzip"`` =
    the ``safetensors`` bytes were gzip-wrapped before upload (keys end ``.gz``;
    gunzip before ``load_file``). stats + manifest are always uncompressed.
    """
    _is_vllm = capture.startswith("vllm")
    # The two backends agree on every middle layer (captured k == HF
    # hidden_states[k+1] == block-k residual, pre-norm) but differ on the LAST
    # layer: HF's hidden_states[n] is POST-final-norm, while vLLM-Lens captures the
    # PRE-norm residual for every layer (ACS-268). The vLLM last layer matches the
    # inline engine (also vLLM-Lens); the HF last layer does not.
    _last_layer = (
        "the LAST captured layer is the PRE-final-norm residual stream (consistent "
        "with the inline vLLM-Lens engine; ACS-268 offline path)"
        if _is_vllm
        else "the LAST captured layer is HF hidden_states[n_layers] (POST-final-norm), "
        "which differs from the inline vLLM-Lens engine's pre-norm last layer"
    )
    return {
        "run_id": run_id,
        # v2 (2026-07-17): shards gained tokens_{i} entries, stats.safetensors,
        # mean_token_norm. v1 runs on the Volume predate all three.
        "layout_version": 2,
        "model_id": model_id,
        "hf_repo": hf_repo,
        "dtype": dtype,
        # How the SHARD objects are encoded (ACS-278). "none" = raw safetensors;
        # "gzip" = gzip-wrapped safetensors, keys end ".gz". stats/manifest are
        # never compressed.
        "compression": compression,
        "capture": capture,
        # Whether the tokenizer prepended special tokens (BOS) to each prompt
        # (ACS-319). False means the prompt text was tokenized verbatim — the
        # right choice when the caller already emitted a BOS client-side.
        "add_special_tokens": add_special_tokens,
        "layer_indices": layer_indices,
        "residual_stream_convention": (
            f"captured layer k == HF hidden_states[k+1] == output of decoder block k "
            f"(post-block residual stream), EXCEPT: {_last_layer}. "
            f"hidden_states[0] (embeddings) is dropped."
        ),
        "n_prompts": len(prompts),
        "batch_size": batch_size,
        "prompts": prompts,
        "shards": shards,
        # A projected run must never be mistakable for a raw-activation one:
        # same key names, different last axis and meaning (ACS-320).
        "contents": "projections" if n_projection_dirs else "activations",
        **(
            {
                "projection": {
                    "n_directions": n_projection_dirs,
                    "normalized": True,
                    "note": (
                        "Directions were L2-normalized server-side before "
                        "projecting, so values are components along each unit "
                        "direction. dtype float32."
                    ),
                }
            }
            if n_projection_dirs
            else {}
        ),
        "tensor_layout": (
            (
                "per prompt: key 'prompt_{i}', shape [n_layers_kept, n_tokens, "
                "n_directions] float32 — the residual stream PROJECTED onto the "
                "directions you supplied, not the raw stream. "
            )
            if n_projection_dirs
            else "per prompt: key 'prompt_{i}', shape [n_layers_kept, n_tokens, hidden]. "
        )
        + (
            "n_tokens = the tokenized prompt (with a leading BOS at token 0 when "
            "add_special_tokens=true; verbatim otherwise), truncated to the model's "
            "max context length."
        ),
        "tokens_layout": (
            "per prompt: key 'tokens_{i}', int32 input token ids (leading BOS present "
            "iff add_special_tokens=true), same length and order as the tensor's seq "
            "axis — stored with the data so autointerp never re-tokenizes (Activault "
            "pattern)."
        ),
        "statistics": (
            "stats.safetensors: 'mean'/'std' float32 "
            + (
                "[n_layers_kept, n_directions] statistics of the PROJECTIONS "
                "(not of the residual stream), and manifest.mean_token_norm is "
                "the mean L2 norm of the projection vectors, not of the "
                "residual stream. "
                if n_projection_dirs
                else "[n_layers_kept, hidden] token statistics over the whole run "
                "(SAE normalization), and manifest.mean_token_norm is the "
                "per-layer mean token L2 norm. "
            )
            + "'layer_indices' int32. Position 0 of each prompt is EXCLUDED (the "
            "BOS attention-sink outlier when add_special_tokens=true)."
        ),
        "mean_token_norm": mean_token_norm or {},
    }


def s3_client():
    """boto3 S3 client for the Railway (S3-compatible) bucket.

    Uses the addressing style the provider advertises (virtual-host by default)
    and SigV4 — the exact config validated in the #203 put/presign/get round-trip.
    """
    import boto3
    from botocore.config import Config

    style = os.environ.get("BUCKET_URL_STYLE", "virtual-host")
    addressing = "path" if "path" in style else "virtual"
    return boto3.client(
        "s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("BUCKET_REGION", "auto"),
        config=Config(
            s3={"addressing_style": addressing},
            signature_version="s3v4",
            # botocore retries transient errors only (a permanent AccessDenied /
            # NoSuchBucket fails immediately instead of sleeping through backoff),
            # with bounded socket timeouts so a stalled connection can't hang the
            # uploader past the drain deadline.
            retries={"max_attempts": 4, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


# --------------------------------------------------------------------------- #
# Straight-to-S3 streaming uploader (ACS-278). On HARVEST_UPLOAD=1 shards are
# serialized in memory and uploaded directly to the bucket — no Volume copy, no
# commit — by a POOL of workers (concurrency, per-object multipart, optional
# gzip), overlapped with the forward pass. The bottleneck at SAE scale is upload
# bandwidth to object storage (Tilde Activault), so this is where the effort goes.
#
# The orchestration below is torch/boto3-free so the no-data-loss invariant
# ("manifest present in the bucket ⇒ every shard present") is unit-testable with a
# fake client: shard uploads run through the pool, and the manifest is written
# LAST and ONLY after a clean drain with zero errors.
# --------------------------------------------------------------------------- #
def _maybe_compress(data: bytes, compress: bool) -> bytes:
    """Optionally gzip the serialized shard bytes (ACS-278).

    bf16 activations are near-incompressible and gzip competes with the forward
    pass for CPU, so compression is an opt-in operator knob (HARVEST_COMPRESS=1),
    off by default. Level 1: at this entropy the higher levels buy almost nothing
    for a lot more CPU.
    """
    if not compress:
        return data
    import gzip

    return gzip.compress(data, compresslevel=1)


def serialize_shard(tensors: dict, *, compress: bool = False) -> bytes:
    """Serialize a shard's tensors to (optionally gzip-wrapped) safetensors bytes.

    In-memory (``safetensors.torch.save``, not ``save_file``) so nothing touches
    the Volume on the upload path — the bytes go straight from RAM to S3.
    """
    from safetensors.torch import save

    return _maybe_compress(save(tensors), compress)


def _multipart_config(chunk_bytes: int, max_concurrency: int):
    """boto3 TransferConfig: automatic multipart over ``chunk_bytes`` with
    ``max_concurrency`` parts in flight per object. Shards larger than the
    threshold upload as concurrent parts; smaller ones as a single PUT."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=chunk_bytes,
        multipart_chunksize=chunk_bytes,
        max_concurrency=max_concurrency,
        use_threads=True,
    )


def _put_bytes(s3, bucket: str, key: str, data: bytes, *, transfer_config=None) -> int:
    """Stream ``data`` to ``s3://bucket/key`` from memory (no temp file). boto3's
    transfer manager splits it into concurrent multipart uploads per
    ``transfer_config``. Returns the byte count."""
    import io

    s3.upload_fileobj(io.BytesIO(data), bucket, key, Config=transfer_config)
    return len(data)


class S3ShardUploader:
    """A pool of worker threads that serialize + upload shards to S3 concurrently.

    ``submit(key, payload)`` hands one shard to the pool and BLOCKS when the bounded
    queue is full — backpressure that stops the forward pass from racing ahead and
    piling serialized shards in RAM. Each worker builds its OWN boto3 client (S3
    clients are not thread-safe). Per-shard upload errors are collected (not raised
    in the worker) and surfaced at ``drain`` so the caller can gate the manifest.

    Metrics (read after ``drain`` joins the threads): ``bytes`` uploaded,
    ``serialize_s`` / ``upload_s`` summed thread time, and ``upload_wall_s`` — the
    first-start→last-end window, i.e. the effective MB/s the pool sustained.
    """

    def __init__(
        self, *, s3_factory, bucket, serialize, concurrency, transfer_config=None, maxsize=None
    ):
        import queue
        import threading

        self._s3_factory = s3_factory
        self._bucket = bucket
        self._serialize = serialize
        self._transfer_config = transfer_config
        self._concurrency = max(1, int(concurrency))
        self._q: queue.Queue = queue.Queue(maxsize=maxsize or self._concurrency)
        self._lock = threading.Lock()
        self.errors: list[str] = []
        self.bytes = 0
        self.serialize_s = 0.0
        self.upload_s = 0.0
        self._first_start = None
        self._last_end = 0.0
        self._threads = [
            threading.Thread(target=self._worker, daemon=True) for _ in range(self._concurrency)
        ]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def submit(self, key: str, payload) -> None:
        """Enqueue a shard; blocks when the pool is saturated (backpressure)."""
        self._q.put((key, payload))

    def _worker(self) -> None:
        import time

        try:
            s3 = self._s3_factory()
        except Exception as e:  # noqa: BLE001 — a dead worker must surface, not hang silently
            with self._lock:
                self.errors.append(f"upload worker init failed: {e}")
            # Drain the queue so a blocked submit() doesn't deadlock the run.
            while self._q.get() is not None:
                pass
            return
        while True:
            item = self._q.get()
            if item is None:
                return
            key, payload = item
            try:
                ts0 = time.perf_counter()
                data = self._serialize(payload)
                ts1 = time.perf_counter()
                _put_bytes(s3, self._bucket, key, data, transfer_config=self._transfer_config)
                t1 = time.perf_counter()
                with self._lock:
                    self.bytes += len(data)
                    self.serialize_s += ts1 - ts0
                    self.upload_s += t1 - ts1
                    if self._first_start is None:
                        self._first_start = ts1
                    self._last_end = max(self._last_end, t1)
            except Exception as e:  # noqa: BLE001 — collected, gated at drain
                with self._lock:
                    self.errors.append(f"{key}: {e}")

    def drain(self, timeout: float = 600) -> bool:
        """Signal end-of-input, join every worker, and report a CLEAN drain
        (all threads exited within ``timeout``)."""
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join(timeout=timeout)
        return all(not t.is_alive() for t in self._threads)

    @property
    def upload_wall_s(self) -> float:
        if self._first_start is None:
            return 0.0
        return max(0.0, self._last_end - self._first_start)


def finalize_s3_run(
    uploader: S3ShardUploader,
    put,
    *,
    stats_key: str,
    stats_bytes: bytes,
    manifest_key: str,
    manifest_bytes: bytes,
    drain_timeout: float = 600,
) -> None:
    """Drain the shard pool, GATE on success, then write stats + manifest LAST.

    This encodes the no-data-loss invariant (ACS-278): the manifest is uploaded
    only after every shard has landed, so "manifest present in the bucket" always
    implies "the run is complete". If any shard failed (or a worker hung), raise
    WITHOUT touching the manifest — a manifest that references a missing shard is
    silent corruption for the downloader; orphaned shards with no manifest are
    harmless (a re-run with the same run_id overwrites the keys). On the upload
    path there is NO Volume fallback, so the error says the GPU pass must be re-run.
    """
    clean = uploader.drain(timeout=drain_timeout)
    if uploader.errors or not clean:
        raise RuntimeError(
            "harvest upload failed — HARVEST_UPLOAD=1 keeps NO Volume copy, so the "
            "whole GPU pass must be re-run: "
            + (str(uploader.errors) if uploader.errors else "uploader hung past drain deadline")
        )
    put(stats_key, stats_bytes)
    put(manifest_key, manifest_bytes)  # LAST — its presence certifies a complete run


# --------------------------------------------------------------------------- #
# Modal app (registry-driven, same selector as modal_app.py).
# --------------------------------------------------------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", "llama-8b")
_CFG = get_model_config(MODEL_ID)
_SPEC = get_model_spec(MODEL_ID)
HF_REPO = _CFG["hf_repo"]
N_GPU = _CFG["n_gpu"]
GPU_TYPE = _CFG["gpu_type"]
MAX_MODEL_LEN = _CFG["max_model_len"]
LOCAL_MODEL_PATH = _CFG.get("local_model_path")
# vLLM offline harvest path (ACS-268; default for ALL models since 2026-07-23) — see the image split below.
DTYPE = _CFG["dtype"]
PRESHARD = _CFG.get("preshard", False)
TRUST_REMOTE_CODE = _CFG.get("trust_remote_code", False) or False
SHARDED_DIR = sharded_dir_for_model(model_name=HF_REPO, n_gpu=N_GPU)


def _harvest_max_containers_default(n_gpu: int) -> int:
    """Default harvest fleet cap (ACS-265), N_GPU-aware.

    Big models (>1 GPU) match the wrapper's cross-key big-model cap
    (``Settings.harvest_max_running_big_model``, default 2): the wrapper admits at
    most that many concurrent big-model jobs, so a fleet of the same size gives
    each admitted job its OWN container immediately — no queuing. Queuing would be
    unsafe here: a queued job's ``created_at`` ages toward the wrapper's 180-min
    stale janitor (``expire_stale_harvest_jobs``) while it's still legitimately
    waiting/running, and would be false-failed. Coupling: if you raise the wrapper
    big-model cap, raise this in tandem. A tighter cap buys no real amortization —
    the per-key cap (1) already forces one key's jobs to be sequential, so they
    reuse the warm container at any max_containers>=1.

    8B is uncapped at the wrapper, cheap (1×L40S), and wants throughput for the
    "10 researchers at once" scenario, so it gets a generous fleet, not a tight cap.
    """
    return 2 if n_gpu > 1 else 16


# vLLM-Lens is the default backend for EVERY model (2026-07-23; 8B parity-gated
# at worst per-layer mean cosine 0.99992 vs the HF reference) — one PRE-final-norm
# capture convention across all models and the inline engine. HARVEST_USE_VLLM=0
# selects the HF-transformers reference path (parity ground truth; POST-final-norm
# last layer). The two need different stacks, so the image is chosen per deploy
# (the module is deployed per-MODEL_ID). HF image notes: arch-agnostic, no vLLM;
# accelerate drives the multi-GPU device_map; transformers floored at 5.13 — the
# `trust_remote_code=False` load relies on its NATIVE AfmoeForCausalLM (added
# there), so an older resolve would break Trinity.
_USE_VLLM = os.environ.get("HARVEST_USE_VLLM", "1") == "1"
if _USE_VLLM:
    _base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "vllm==0.19.1",
        "vllm-lens==1.2.0",  # pinned; forces enforce_eager (fine — prefill-only harvest)
        # Floor transformers so the shared `AutoConfig(trust_remote_code=False)` layer
        # count resolves Trinity's NATIVE AfmoeForCausalLM/AfmoeConfig (added in 5.13);
        # a lower transitive pin would abort every Trinity harvest before the load.
        "transformers>=5.13",
        "safetensors",
        "boto3",
        "huggingface_hub[hf_transfer]",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
else:
    _base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch",
        "transformers>=5.13",
        "accelerate",
        "safetensors",
        "boto3",
        "huggingface_hub[hf_transfer]",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
image = (
    _base_image
    # expandable_segments avoids the allocator fragmentation that OOM'd one H200
    # during Trinity's MoE expert-weight conversion.
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODEL_ID": MODEL_ID,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Bake module-load config into the image so the REMOTE module
            # evaluation agrees with the local one. UPLOAD gates the function's
            # secret list at definition time — without this the local app has 8
            # dependency objects and the container reconstructs 7, and Modal
            # aborts with "Function has 7 dependencies but container got 8
            # object ids" (hit on the first HARVEST_UPLOAD=1 run, 2026-07-17).
            # These are read at module load too and would silently ignore
            # operator overrides remotely. The app name, container cap, and
            # scaledown window (ACS-265) are all resolved at module/decorator
            # eval, so bake them for remote↔local agreement — a name mismatch
            # would leave the container reconstructing a differently-named app.
            "HARVEST_UPLOAD": "1" if UPLOAD else "0",
            "HARVEST_MAX_BATCH_TOKENS": str(MAX_BATCH_TOKENS),
            # ACS-278 straight-to-S3 uploader knobs — baked so a deploy-time
            # override reaches the container (read at module load, used per-run).
            "HARVEST_UPLOAD_CONCURRENCY": str(UPLOAD_CONCURRENCY),
            "HARVEST_MULTIPART_CONCURRENCY": str(UPLOAD_MULTIPART_CONCURRENCY),
            "HARVEST_MULTIPART_CHUNK_MB": str(MULTIPART_CHUNK_BYTES // (1024 * 1024)),
            "HARVEST_COMPRESS": "1" if COMPRESS else "0",
            "HARVEST_URL_TTL_S": str(URL_TTL_S),
            "HARVEST_APP_NAME": os.environ.get("HARVEST_APP_NAME", f"acs-{MODEL_ID}-harvest"),
            "HARVEST_MAX_CONTAINERS": os.environ.get(
                "HARVEST_MAX_CONTAINERS", str(_harvest_max_containers_default(N_GPU))
            ),
            "HARVEST_SCALEDOWN_S": os.environ.get("HARVEST_SCALEDOWN_S", str(20 * 60)),
            # ACS-268: which capture path. Read at module load (picks the image +
            # the harvest branch), so bake it for remote↔local agreement.
            "HARVEST_USE_VLLM": "1" if _USE_VLLM else "0",
        }
    )
    .add_local_python_source("acs_model_registry", "serving")
)

_VOLUMES = create_volumes()
_SERVE_VOLUMES = serve_volumes(spec=_SPEC, volumes=_VOLUMES)
# Durable output volume — shards + manifest land here even without bucket upload.
_OUT_VOL = modal.Volume.from_name("acs-activation-harvest", create_if_missing=True)
_OUT_DIR = "/harvest_out"
_ALL_VOLUMES = {**_SERVE_VOLUMES, _OUT_DIR: _OUT_VOL}

_secrets = [modal.Secret.from_name("huggingface-secret")]
if UPLOAD:
    _secrets.append(modal.Secret.from_name("activation-harvest-bucket"))

app = modal.App(os.environ.get("HARVEST_APP_NAME", f"acs-{MODEL_ID}-harvest"))


# 405B streams ~810 GB off the cache Volume — the load alone is ~50 min, so give
# the big models generous headroom (single-GPU 8B finishes in seconds regardless).
_TIMEOUT_MIN = 150 if N_GPU > 1 else 60


# Warm-container model cache (ACS-265). Modal reuses a container across the inputs
# it drains, so loading the model ONCE per container — not per job — is what makes
# concurrent big-model harvests feasible: the ~50-min 405B load is paid once and
# every subsequent job the warm container serves reuses the resident model. This
# file already assumed container reuse (the "/tmp/offload … reused container"
# cleanup below). The amortization that actually applies under the wrapper caps is
# a single key's SEQUENTIAL jobs (per-key cap = 1 forces them serial): job 2..N
# land on the same warm container and skip the load. ``max_containers`` bounds the
# fleet so concurrent jobs from different keys can't each spawn a fresh 8×H200 that
# re-loads from scratch. SAFE ONLY under one-input-per-container (the Modal
# default — do NOT add @modal.concurrent): the shared Volume commit + the global
# tokenizer padding state would corrupt across jobs run concurrently in one
# container. Parallelism comes from more containers (max_containers=k), not
# concurrent inputs.
_MODEL = None
_TOK = None
_IN_DEVICE = None
_VLLM = None
_VLLM_TOK = None


def _load_vllm_once():
    """Load an offline vLLM engine once per container (ACS-268; all models by default).

    Mirrors the activation engine's proven load path (presharded → ``sharded_state``,
    else the HF repo from the cache volume). vLLM-Lens auto-registers on ``import
    vllm`` and forces ``enforce_eager`` (fine — prefill-only harvest). Reused across
    jobs like ``_load_model_once``; returns ``(llm, load_s)`` (~0 on warm reuse).
    """
    global _VLLM
    if _VLLM is not None:
        return _VLLM, 0.0
    import time

    from vllm import LLM

    use_sharded = (
        PRESHARD
        and activation_can_load_sharded(_SPEC.sharded_volume_name, _SPEC.activation_sharded_ok)
        and os.path.isdir(SHARDED_DIR)
        and os.listdir(SHARDED_DIR)
    )
    kwargs: dict = dict(
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        trust_remote_code=TRUST_REMOTE_CODE,
        enable_prefix_caching=False,  # stale-KV corrupts capture (shipped capture config)
    )
    if use_sharded:
        model_path = SHARDED_DIR
        kwargs["load_format"] = "sharded_state"
    else:
        model_path = (
            LOCAL_MODEL_PATH if (LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH)) else HF_REPO
        )
    print(f"[harvest] vLLM offline load {model_path} (tp={N_GPU} sharded={bool(use_sharded)})", flush=True)
    t0 = time.perf_counter()
    _VLLM = LLM(model=model_path, **kwargs)
    return _VLLM, time.perf_counter() - t0


def _capture_vllm(
    idxs, llm, prompts, layer_indices, timings, counters, directions=None, add_special_tokens=True
):
    """Offline vLLM-Lens capture for one prompt subset (ACS-268; default backend).

    Returns ``{prompt_idx: (acts[n_kept, n_tok, hidden] bf16 cpu, ids[n_tok] int32 cpu)}``
    — the SAME structure the HF ``capture_batch`` produces, so the shard/stats/upload
    pipeline downstream is identical. vLLM batches internally (continuous batching),
    so one ``generate()`` per shard replaces the HF token-budget batching. Layer index
    ``k`` == HF ``hidden_states[k+1]`` (block output) for both paths — the shipped
    vLLM-Lens convention — so the kept-layer indexing matches the HF harvest.
    """
    import time

    import torch
    from vllm import SamplingParams, TokensPrompt

    global _VLLM_TOK
    if _VLLM_TOK is None:
        from transformers import AutoTokenizer

        _src = LOCAL_MODEL_PATH if (LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH)) else HF_REPO
        _VLLM_TOK = AutoTokenizer.from_pretrained(_src, trust_remote_code=False)

    t0 = time.perf_counter()
    sp = SamplingParams(
        max_tokens=1,
        temperature=0,
        extra_args={"output_residual_stream": list(layer_indices)},
    )
    # Pre-truncate over-length corpus prompts and feed token ids (0.19.1
    # SamplingParams has no truncate option). Mirrors the HF path's
    # tok(truncation=True, max_length=...); the SAME tokenizer means the ids equal
    # vLLM's own text tokenization when not truncated, so capture is identical to
    # passing text — a single too-long prompt no longer aborts the shard. Cap at
    # MAX_MODEL_LEN-1, NOT MAX_MODEL_LEN: unlike the HF forward, vLLM runs
    # generate(max_tokens=1), so prompt_len + 1 must fit the context window (the HF
    # path has no generated token and caps at the full length). Only maximally-long
    # (already-truncated) prompts see the 1-token difference.
    _cap = max(1, MAX_MODEL_LEN - 1)
    token_prompts = [
        TokensPrompt(
            prompt_token_ids=_VLLM_TOK(
                prompts[i],
                truncation=True,
                max_length=_cap,
                add_special_tokens=add_special_tokens,
            ).input_ids
        )
        for i in idxs
    ]
    outs = llm.generate(token_prompts, sp)  # returned in input order
    result: dict[int, tuple] = {}
    for b, i in enumerate(idxs):
        raw = outs[b].activations["residual_stream"].to(torch.bfloat16)
        acts = (
            project_activations(raw, directions).cpu()
            if directions is not None
            else raw.contiguous().cpu()
        )
        ids = torch.tensor(list(outs[b].prompt_token_ids), dtype=torch.int32)
        # acts: [n_kept, n_tok, hidden] — or [n_kept, n_tok, n_dirs] when projecting.
        result[i] = (acts, ids)  # captures prompt positions
        counters["tokens"] += int(acts.shape[1])
    timings["forward_s"] += time.perf_counter() - t0
    return result


def _load_model_once():
    """Load model + tokenizer once per container; reuse on every later job.

    Returns ``(model, tok, in_device, load_s)`` where ``load_s`` is THIS call's
    load time — ~0 on a warm reuse — so a job's ``timings.load_s`` reflects the
    amortization (job 1 pays the full load; jobs 2..N see ~0).
    """
    global _MODEL, _TOK, _IN_DEVICE
    if _MODEL is not None:
        return _MODEL, _TOK, _IN_DEVICE, 0.0
    import pathlib
    import shutil
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = LOCAL_MODEL_PATH if (LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH)) else HF_REPO
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False)
    # Right padding + attention mask: each prompt's real tokens sit at [:len] with
    # correct 0..len-1 positions, so slicing the mask length recovers exactly the
    # unpadded per-prompt tensor. (Left padding would shift positions — force right.)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=False)
    if N_GPU > 1:
        # accelerate shards the model across the GPUs; inputs go to the first.
        # Big-model loading needs headroom + spill targets: transformers' MoE
        # expert-weight conversion (Trinity) needs transient memory on top of the
        # placed weights and OOMs a single H200 without a per-GPU cap; and if the
        # cap forces any offload it errors unless given an offload_folder. So cap
        # each GPU below its real capacity, give the loader CPU + a disk folder to
        # spill into, and offload the state dict during load to cut the peak.
        if pathlib.Path("/tmp/offload").exists():
            shutil.rmtree("/tmp/offload")  # clean any prior run on a reused container
        pathlib.Path("/tmp/offload").mkdir(parents=True, exist_ok=True)
        max_mem = {i: "110GiB" for i in range(N_GPU)}  # H200 = ~140 GB; ~30 GB headroom
        max_mem["cpu"] = "1000GiB"
        model = AutoModelForCausalLM.from_pretrained(
            src,
            device_map="auto",
            max_memory=max_mem,
            offload_folder="/tmp/offload",
            offload_state_dict=True,
            **load_kwargs,
        ).eval()
        in_device = model.get_input_embeddings().weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs).eval().to("cuda")
        in_device = torch.device("cuda")
    _MODEL, _TOK, _IN_DEVICE = model, tok, in_device
    return _MODEL, _TOK, _IN_DEVICE, time.perf_counter() - t0


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    timeout=_TIMEOUT_MIN * MINUTES,
    volumes=_ALL_VOLUMES,
    secrets=_secrets,
    # ACS-265: cap the harvest fleet. Without this, N concurrent jobs each spawn a
    # fresh container that re-loads the model (10 × 8×H200 405B = 80 H200, each a
    # ~50-min load). Default is N_GPU-aware (see _harvest_max_containers_default):
    # big models match the wrapper's big-model cap (2) so admitted jobs never queue
    # behind a warm container (queuing would trip the 180-min stale janitor); 8B
    # gets a generous fleet for throughput. Load-once still amortizes a single key's
    # sequential jobs (they reuse the warm container). scaledown_window holds the
    # container warm across the inter-job gap so a key's jobs 2..N skip the load.
    max_containers=int(
        os.environ.get("HARVEST_MAX_CONTAINERS", str(_harvest_max_containers_default(N_GPU)))
    ),
    scaledown_window=int(os.environ.get("HARVEST_SCALEDOWN_S", str(20 * MINUTES))),
)
def harvest(
    prompts: list[str],
    layer_indices: list[int] | str | None,
    shard_size: int,
    run_id: str,
    batch_size: int | None = None,
    project_onto: dict | None = None,
    add_special_tokens: bool = True,
) -> dict:
    import pathlib
    import queue
    import shutil
    import threading
    import time

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    # Phase timings returned with the result (ACS-218): the forward-vs-storage-vs-
    # upload split is what decides where optimization effort goes, so every run
    # reports it rather than us re-instrumenting ad hoc.
    timings = {
        "load_s": 0.0,
        "forward_s": 0.0,
        "save_s": 0.0,
        "commit_s": 0.0,
        "upload_s": 0.0,
    }
    counters = {"tokens": 0, "bytes_written": 0}

    # Fail fast on misconfiguration BEFORE the model loads / forwards run — a
    # bad flag must not surface after a ~50-min 405B load.
    if not prompts:
        raise ValueError("no prompts (empty corpus after filtering blank lines)")
    # Default batch size: 8 on the single-GPU model; 1 (sequential) on the
    # multi-GPU big models, whose memory envelope and batched-vs-sequential
    # parity were verified sequential-only — batching there is operator opt-in.
    if batch_size is None:
        batch_size = 8 if N_GPU == 1 else 1
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if UPLOAD:
        missing = [
            k
            for k in ("BUCKET_ENDPOINT_URL", "BUCKET_NAME", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
            if not os.environ.get(k)
        ]
        if missing:
            raise RuntimeError(f"HARVEST_UPLOAD=1 but bucket secret is missing {missing}")
        # Probe the bucket now: a bad endpoint/credential would otherwise ride
        # silently in the uploader thread and surface only after the GPU pass.
        s3_client().head_bucket(Bucket=os.environ["BUCKET_NAME"])

    # Weight source: prefer the registry's local staged copy when present — it
    # holds the full HF-format weights, so no re-download (Trinity: all 31
    # safetensors live in /cache/trinity-base). Else the HF repo, resolved from
    # the shared acs-hf-cache. transformers ships NATIVE archs for all our models
    # (LlamaForCausalLM; AfmoeForCausalLM since transformers 5.13), so load with
    # trust_remote_code OFF — the registry's `trust_remote_code=True` (Trinity)
    # is for vLLM's loader; with it ON, transformers hunts for a remote
    # modeling_afmoe.py that neither the repo nor the staged copy ships, and fails.
    src = LOCAL_MODEL_PATH if (LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH)) else HF_REPO
    trust_remote = False
    _cfg = AutoConfig.from_pretrained(src, trust_remote_code=trust_remote)
    n_layers_all = _cfg.num_hidden_layers
    # Decode + validate the projection directions BEFORE the model loads: a bad
    # payload must cost seconds, not a ~50-minute 405B load (ACS-320).
    directions = None
    if project_onto is not None:
        directions = decode_projection_directions(project_onto, int(_cfg.hidden_size))
        print(
            f"[harvest] projecting onto {directions.shape[0]} unit direction(s) "
            f"(hidden={directions.shape[1]}) — shards hold projections, not activations",
            flush=True,
        )
    # Layer ids index the decoder blocks (0..n_layers-1); default = quartile
    # subset (~25/50/75% depth), "all" = every block.
    if layer_indices == "all":
        layer_indices = list(range(n_layers_all))
    elif layer_indices is None:
        layer_indices = default_layer_indices(n_layers_all)
    layer_indices = sorted(set(layer_indices))
    bad = [L for L in layer_indices if not 0 <= L < n_layers_all]
    if bad:
        raise ValueError(f"layer_indices {bad} out of range for {MODEL_ID} (0..{n_layers_all - 1})")

    print(f"[harvest] loading {src} (n_gpu={N_GPU}) n_layers_kept={len(layer_indices)}", flush=True)
    # Load once per container, reuse across jobs (ACS-265). All models default to
    # the offline vLLM-Lens path; HARVEST_USE_VLLM=0 selects the HF reference loop.
    vllm_engine = None
    if _USE_VLLM:
        vllm_engine, timings["load_s"] = _load_vllm_once()
    else:
        # load_s ~0 on a warm reuse; the tokenizer (pad token + right padding) is set
        # up inside the loader.
        model, tok, in_device, timings["load_s"] = _load_model_once()

    # ---- capture (HF path): batched padded forward, keep hidden_states[1:] (block outputs) ----

    def capture_batch(idxs: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Forward one padded batch; return {prompt_idx: (activations, token_ids)}."""
        # Truncate to the model's context window: a corpus prompt longer than that
        # would overrun position embeddings / OOM and abort the whole harvest. With
        # add_special_tokens=True (default) token 0 is the tokenizer's BOS; the
        # captured seq axis matches the tokenized prompt either way.
        t0 = time.perf_counter()
        enc = tok(
            [prompts[i] for i in idxs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_MODEL_LEN,
            add_special_tokens=add_special_tokens,
        ).to(in_device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states: (n_layers+1) × [B, T, H]; k+1 = output of block k. Stack
        # ONLY the kept layers — stacking all layers first would transiently
        # allocate n_layers/n_kept × the needed memory (≈10× for a 3-layer 405B
        # subset) on top of the tuple HF already holds, and OOM long batches.
        hs = torch.stack([out.hidden_states[k + 1] for k in layer_indices], dim=0).to(
            torch.bfloat16
        )
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        result = {}
        for b, i in enumerate(idxs):
            n = int(lengths[b])
            _slice = hs[:, b, :n, :]  # [n_layers_kept, n, hidden], still on device
            _a = (
                project_activations(_slice, directions).cpu()
                if directions is not None
                else _slice.contiguous().cpu()
            )
            result[i] = (
                _a,
                enc["input_ids"][b, :n].to(torch.int32).cpu(),
            )
        # .cpu() synchronizes, so this wall time covers tokenize + forward + D2H.
        timings["forward_s"] += time.perf_counter() - t0
        counters["tokens"] += int(sum(lengths))
        return result

    # ---- storage pipeline (ACS-278): two MUTUALLY-EXCLUSIVE backends chosen by
    # `plan` (HARVEST_UPLOAD). The measured bottleneck at scale is storage/upload,
    # not compute (ACS-218), so both overlap storage with the forward pass and
    # back-pressure the capture loop so shards can't pile up in RAM.
    #
    #   to_s3   (HARVEST_UPLOAD=1): serialize each shard IN MEMORY and stream it
    #           STRAIGHT to the bucket via a pool of concurrent multipart uploaders
    #           — no Volume write, no `.commit()`. S3 is the ONLY durable copy, so
    #           an upload failure loses the GPU pass (see finalize_s3_run). This
    #           removes the redundant second copy + the Volume-commit cost that
    #           dominated the old pipeline.
    #   to_volume (no-bucket smoke): write shards to the Volume on a single worker
    #           thread, one final `.commit()` after the manifest (ACS-270). The
    #           Volume is then the only durable store. No per-shard commit: Modal
    #           background-commits attached Volumes, the run is not resumable
    #           (a re-run rmtree's the dir), and the manifest is written last, so a
    #           crashed run's shards are orphaned + discarded regardless.
    plan = select_storage_backend(UPLOAD)

    s3_uploader: S3ShardUploader | None = None
    transfer_cfg = None
    out_run_dir = None
    storage_q: queue.Queue | None = None
    storage_thread = None
    storage_errors: list[str] = []
    storage_done = threading.Event()

    if plan["to_s3"]:
        transfer_cfg = _multipart_config(MULTIPART_CHUNK_BYTES, UPLOAD_MULTIPART_CONCURRENCY)
        s3_uploader = S3ShardUploader(
            s3_factory=s3_client,
            bucket=os.environ["BUCKET_NAME"],
            serialize=lambda tensors: serialize_shard(tensors, compress=COMPRESS),
            concurrency=UPLOAD_CONCURRENCY,
            transfer_config=transfer_cfg,
        )
        s3_uploader.start()

        def _submit(shard_idx: int, key: str, tensors: dict) -> None:
            s3_uploader.submit(key, tensors)  # blocks when the pool is saturated
    else:
        out_run_dir = pathlib.Path(_OUT_DIR) / "harvest" / run_id
        if out_run_dir.exists():
            shutil.rmtree(out_run_dir)  # no stale shards from a shorter prior run
        out_run_dir.mkdir(parents=True, exist_ok=True)
        storage_q = queue.Queue(maxsize=2)  # ≤2 shards of tensors wait in CPU RAM

        def _volume_worker() -> None:
            try:
                while True:
                    item = storage_q.get()
                    if item is None:
                        storage_done.set()
                        return
                    vol_path, tensors = item
                    t0 = time.perf_counter()
                    save_file(tensors, str(vol_path))
                    timings["save_s"] += time.perf_counter() - t0
                    counters["bytes_written"] += vol_path.stat().st_size
                    print(f"[harvest] stored {vol_path.name}", flush=True)
            except Exception as e:  # noqa: BLE001 - main loop aborts on next check
                storage_errors.append(str(e))

        storage_thread = threading.Thread(target=_volume_worker, daemon=True)
        storage_thread.start()

        def _submit(shard_idx: int, key: str, tensors: dict) -> None:
            storage_q.put((out_run_dir / f"shard_{shard_idx:05d}.safetensors", tensors))

    def _backend_failing() -> bool:
        return bool(storage_errors) or bool(s3_uploader and s3_uploader.errors)

    # ---- capture loop: batched forwards feed the storage pipeline.
    # Running token statistics accumulate in float64 (fp32 running sums lose
    # bits at SAE scale and E[x^2]-E[x]^2 cancels catastrophically) and EXCLUDE
    # each prompt's BOS token — a Llama-style attention sink whose huge residual
    # norms would contaminate the normalization stats (Activault excludes it too).
    est_tokens = [len(p) // 4 + 1 for p in prompts]  # ~4 chars/token proxy
    stat_count = 0
    stat_sum = stat_sumsq = stat_normsum = 0.0
    shard_meta: list[dict] = []
    for shard_idx, idxs in enumerate(chunk_prompts(prompts, shard_size)):
        if _backend_failing():
            break  # abort early: storage is failing, don't burn more GPU time
        captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if _USE_VLLM:  # vLLM batches internally — one generate() per shard
            captured = _capture_vllm(
                idxs,
                vllm_engine,
                prompts,
                layer_indices,
                timings,
                counters,
                directions,
                add_special_tokens,
            )
        else:
            for batch in batch_prompt_indices(idxs, est_tokens, batch_size, MAX_BATCH_TOKENS):
                captured.update(capture_batch(batch))
        tensors: dict[str, torch.Tensor] = {}
        entries: list[dict] = []
        for i in idxs:
            t, ids = captured[i]
            tensors[f"prompt_{i}"] = t
            tensors[f"tokens_{i}"] = ids
            entries.append({"name": f"prompt_{i}", "shape": list(t.shape)})
            entries.append({"name": f"tokens_{i}", "shape": list(ids.shape)})
            if t.shape[1] > 1:
                # Drop position 0 from the normalization stats: it is the BOS
                # (attention-sink outlier) when add_special_tokens=True. With
                # add_special_tokens=False it is a real content token, so the stats
                # then exclude one genuine token per prompt — an accepted minor
                # imprecision in the SAE-normalization aid; the saved activations
                # and tokens_i are unaffected.
                x = t[:, 1:, :].double()  # [n_kept, n_tokens-1, hidden], position 0 dropped
                stat_sum = stat_sum + x.sum(dim=1)
                stat_sumsq = stat_sumsq + (x * x).sum(dim=1)
                stat_normsum = stat_normsum + x.norm(dim=-1).sum(dim=1)
                stat_count += x.shape[1]
        # Compressed shards carry a .gz suffix so the object is self-describing.
        key = shard_key(run_id, shard_idx, compress=COMPRESS)
        shard_meta.append({"key": key, "prompt_indices": idxs, "tensors": entries})
        _submit(shard_idx, key, tensors)  # blocks for backpressure

    # ---- finalize statistics (token mean/std per kept layer, mean L2 norm).
    if stat_count == 0:  # degenerate corpus: every prompt was a lone BOS
        z = torch.zeros(len(layer_indices), int(shard_meta[0]["tensors"][0]["shape"][2]))
        mean, std, mean_norm = z, z.clone(), z[:, 0].clone()
    else:
        mean = (stat_sum / stat_count).float()
        std = (stat_sumsq / stat_count - (stat_sum / stat_count) ** 2).clamp(min=0.0).sqrt().float()
        mean_norm = (stat_normsum / stat_count).float()
    stats_tensors = {
        "mean": mean.contiguous(),
        "std": std.contiguous(),
        "layer_indices": torch.tensor(layer_indices, dtype=torch.int32),
    }
    mean_token_norm = {str(k): round(mean_norm[j].item(), 4) for j, k in enumerate(layer_indices)}

    manifest = build_manifest(
        n_projection_dirs=(int(directions.shape[0]) if directions is not None else None),
        run_id=run_id,
        model_id=MODEL_ID,
        hf_repo=HF_REPO,
        # Projected shards are float32; claiming bfloat16 would contradict
        # tensor_layout and mis-decode for anyone trusting the field.
        dtype="float32" if directions is not None else "bfloat16",
        layer_indices=layer_indices,
        prompts=prompts,
        shards=shard_meta,
        batch_size=batch_size,
        mean_token_norm=mean_token_norm,
        add_special_tokens=add_special_tokens,
        compression="gzip" if COMPRESS else "none",
        capture=(
            "vllm-lens offline output_residual_stream"
            if _USE_VLLM
            else "hf-transformers output_hidden_states"
        ),
    )

    urls: list[str] = []
    manifest_url = None
    stats_url = None
    upload_wall_s = 0.0
    if plan["to_s3"]:
        # Drain the shard pool, GATE on a clean drain (no errors), then upload
        # stats + manifest LAST — "manifest present ⇒ every shard present". A
        # failed shard raises WITHOUT writing the manifest (no Volume fallback on
        # this path: the whole GPU pass must re-run). Stats + manifest go
        # uncompressed (small; JSON must stay readable).
        s3 = s3_client()
        bucket = os.environ["BUCKET_NAME"]
        skey, mkey = stats_key(run_id), manifest_key(run_id)
        finalize_s3_run(
            s3_uploader,
            lambda k, b: _put_bytes(s3, bucket, k, b, transfer_config=transfer_cfg),
            stats_key=skey,
            stats_bytes=serialize_shard(stats_tensors, compress=False),
            manifest_key=mkey,
            manifest_bytes=json.dumps(manifest, indent=2).encode(),
        )
        # Roll the pool's metrics into the phase timings: save_s = in-memory
        # serialization, upload_s = network (commit_s stays 0 — no Volume).
        timings["save_s"] += s3_uploader.serialize_s
        timings["upload_s"] += s3_uploader.upload_s
        counters["bytes_written"] += s3_uploader.bytes
        upload_wall_s = s3_uploader.upload_wall_s

        def _presign(k: str) -> str:
            return s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": k}, ExpiresIn=URL_TTL_S
            )

        urls = [_presign(sm["key"]) for sm in shard_meta]
        manifest_url = _presign(mkey)
        stats_url = _presign(skey)
    else:
        storage_q.put(None)
        storage_thread.join(timeout=600)
        if storage_errors or not storage_done.is_set():
            raise RuntimeError(
                f"storage worker failed/hung: {storage_errors or 'no clean drain'}"
            )
        save_file(stats_tensors, str(out_run_dir / "stats.safetensors"))
        (out_run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # The ONE explicit commit (ACS-270): makes the whole completed run durable
        # before we return, regardless of Modal's background commit cadence.
        _t0 = time.perf_counter()
        _OUT_VOL.commit()
        timings["commit_s"] += time.perf_counter() - _t0

    timings = {k: round(v, 2) for k, v in timings.items()}
    stats = {
        **timings,
        "tokens": counters["tokens"],
        "gb_written": round(counters["bytes_written"] / 1e9, 3),
        "forward_tok_s": round(counters["tokens"] / timings["forward_s"], 1)
        if timings["forward_s"]
        else None,
        # Effective uploader throughput: bytes over the first-start→last-end window
        # (overlap-aware — accounts for concurrency AND idle gaps waiting on the
        # forward). None on the Volume smoke path (nothing uploaded).
        "upload_wall_s": round(upload_wall_s, 2) if plan["to_s3"] else None,
        "upload_mb_s": round(counters["bytes_written"] / 1e6 / upload_wall_s, 1)
        if plan["to_s3"] and upload_wall_s
        else None,
    }
    print(f"[harvest] timings: {json.dumps(stats)}", flush=True)
    return {
        "run_id": run_id,
        "n_prompts": len(prompts),
        "n_shards": len(shard_meta),
        "layer_indices": layer_indices,
        "batch_size": batch_size,
        "uploaded": UPLOAD,
        # Straight-to-S3 runs write nothing to the Volume (ACS-278); the run prefix
        # is the same logical path in the bucket.
        "volume": None if plan["to_s3"] else "acs-activation-harvest",
        "volume_path": f"harvest/{run_id}/",
        "manifest_url": manifest_url,
        "shard_urls": urls,
        "stats_url": stats_url,
        "timings": stats,
    }


@app.function(image=image, volumes={_OUT_DIR: _OUT_VOL}, timeout=10 * MINUTES)
def verify_shard(run_id: str) -> dict:
    """Read shard_00000 + manifest back from the Volume; assert valid bf16 tensors.

    Volume-only: a ``HARVEST_UPLOAD=1`` run writes nothing to the Volume (ACS-278),
    so this verifier applies to smoke runs. To verify an uploaded run, GET the
    presigned shard/manifest URLs the ``harvest`` result returned.
    """
    import pathlib

    import torch
    from safetensors.torch import load_file

    run_dir = pathlib.Path(_OUT_DIR) / "harvest" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    shard = load_file(str(run_dir / "shard_00000.safetensors"))
    pkeys = sorted(k for k in shard if k.startswith("prompt_"))
    first = shard[pkeys[0]]
    tokens = shard.get(pkeys[0].replace("prompt_", "tokens_"))
    out = {
        "run_id": run_id,
        "n_layers_in_manifest": len(manifest["layer_indices"]),
        "shard0_prompt_keys": pkeys,
        "first_tensor_shape": list(first.shape),
        "first_tensor_dtype": str(first.dtype),
        "all_finite": bool(torch.isfinite(first.float()).all()),
        "abs_mean": round(first.float().abs().mean().item(), 4),
        # tokens travel with the data (Activault pattern); length must match seq axis
        "tokens_match_seq_axis": bool(
            tokens is not None and tokens.dtype == torch.int32 and tokens.shape[0] == first.shape[1]
        ),
        "has_stats": (run_dir / "stats.safetensors").exists(),
    }
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main(
    prompts: str,
    layers: str = "",
    shard_size: int = 32,
    run_id: str = "run",
    batch_size: int = 0,
    add_special_tokens: bool = True,
):
    """batch_size 0 (default) = model-appropriate: 8 on single-GPU, sequential on multi-GPU.

    add_special_tokens (default true) prepends the tokenizer's BOS; pass
    --no-add-special-tokens for prompts that already carry one client-side (ACS-319).
    """
    import pathlib

    lines = [ln for ln in pathlib.Path(prompts).read_text().splitlines() if ln.strip()]
    parsed: list[str] = []
    for ln in lines:  # accept plain prompt-per-line OR JSONL {"prompt": "..."}
        try:
            parsed.append(json.loads(ln)["prompt"])
        except (json.JSONDecodeError, KeyError, TypeError):
            parsed.append(ln)
    result = harvest.remote(
        parsed,
        parse_layers(layers),
        shard_size,
        run_id,
        batch_size or None,
        add_special_tokens=add_special_tokens,
    )
    print(json.dumps(result, indent=2))
