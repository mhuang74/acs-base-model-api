"""Implementation helpers for Modal weight staging phases."""

from __future__ import annotations

import os
import shutil


def stage_weights_impl(*, model_name: str, hf_cache_dir: str, hf_cache_vol: object) -> None:
    """Download a model snapshot into the HF cache Volume."""
    from huggingface_hub import snapshot_download

    print(f"[stage_weights] downloading {model_name} to {hf_cache_dir}")
    path = snapshot_download(
        model_name,
        ignore_patterns=["*.pth", "original/*"],
    )
    print(f"[stage_weights] complete: {path}")
    hf_cache_vol.commit()


def preshard_impl(
    *,
    model_id: str,
    model_name: str,
    n_gpu: int,
    preshard: bool,
    sharded_dir: str,
    sharded_vol: object,
    source_model_path: str | None = None,
    trust_remote_code: bool = False,
    enable_expert_parallel: bool = False,
) -> None:
    """Pre-shard a model to the configured tensor-parallel size.

    ``source_model_path`` is what vLLM loads to shard: an HF repo id (default,
    resolved via the HF cache — 405B) or a local path into a staged Volume
    (Trinity — /cache/trinity-base). ``enable_expert_parallel`` MUST match the
    serve-time EP config: ShardedStateLoader saves one shard per TP rank
    capturing that rank's in-memory expert slice, so save-time and load-time EP
    must agree or the loaded experts are silently wrong.
    """
    if not preshard:
        print(f"[preshard] MODEL_ID={model_id} preshard=False - skipping")
        return

    if os.path.isdir(sharded_dir) and os.listdir(sharded_dir):
        print(f"[preshard] shards already present at {sharded_dir}, skipping")
        return

    from vllm import LLM
    from vllm.model_executor.model_loader import ShardedStateLoader

    os.makedirs(sharded_dir, exist_ok=True)
    load_from = source_model_path or model_name
    print(
        f"[preshard] sharding {load_from} -> {sharded_dir} "
        f"(TP={n_gpu}, expert_parallel={enable_expert_parallel}, "
        f"trust_remote_code={trust_remote_code})"
    )

    llm = LLM(
        model=load_from,
        tensor_parallel_size=n_gpu,
        dtype="bfloat16",
        enforce_eager=True,
        trust_remote_code=trust_remote_code,
        enable_expert_parallel=enable_expert_parallel,
    )

    # vLLM 0.19+ V1 engine: save_sharded_state lives on engine_core, not
    # model_executor.
    llm.llm_engine.engine_core.save_sharded_state(
        path=sharded_dir,
        pattern=ShardedStateLoader.DEFAULT_PATTERN,
        max_size=5 * 1024**3,
    )

    sharded_vol.commit()
    print(f"[preshard] complete: {sharded_dir} - now run `modal run modal_app.py::copy_metadata`")


def copy_metadata_impl(
    *,
    model_id: str,
    model_name: str,
    preshard: bool,
    hf_cache_dir: str,
    sharded_dir: str,
    sharded_vol: object,
    metadata_source_dir: str | None = None,
) -> None:
    """Copy tokenizer/config metadata (incl. custom-arch .py) next to shards.

    ``metadata_source_dir`` overrides the source: locally-staged models
    (Trinity — /cache/trinity-base) keep their config/tokenizer/afmoe-.py in a
    flat dir rather than the HF-hub cache layout. The sharded_state loader needs
    these — including ``configuration_afmoe.py`` for trust_remote_code — to
    rebuild the model skeleton before streaming the rank shards.
    """
    if not preshard:
        print(f"[copy_metadata] MODEL_ID={model_id} preshard=False - skipping")
        return

    if not (os.path.isdir(sharded_dir) and os.listdir(sharded_dir)):
        raise RuntimeError(f"{sharded_dir} is empty; run preshard first")

    if metadata_source_dir:
        source_dir = metadata_source_dir
        if not os.path.isdir(source_dir):
            raise RuntimeError(f"metadata_source_dir {source_dir} does not exist")
    else:
        # Walk the HF cache directly to avoid snapshot_download verification
        # while the large sharded write has recently flushed through Modal
        # Volume FUSE.
        cache_root = (
            f"{hf_cache_dir}/hub/models--{model_name.replace('/', '--')}/snapshots"
        )
        revs = sorted(os.listdir(cache_root))
        if not revs:
            raise RuntimeError(f"No revisions at {cache_root}")
        source_dir = os.path.join(cache_root, revs[-1])
    print(f"[copy_metadata] source: {source_dir}")
    print(f"[copy_metadata] dest:   {sharded_dir}")

    # ".cache" — HF snapshot_download(local_dir=...) bookkeeping (locks, blob
    # refs), not model metadata; copying it is useless and its lock files can be
    # stale. "original" — Llama's raw-checkpoint mirror.
    skip_dirs = {"original", ".cache"}
    skip_exts = {".bin", ".pt", ".safetensors"}

    copied = 0
    for entry in os.listdir(source_dir):
        src = os.path.join(source_dir, entry)
        if os.path.isdir(src):
            if entry in skip_dirs:
                print(f"[copy_metadata] skipping dir {entry}/")
                continue
        elif os.path.splitext(entry)[1] in skip_exts:
            continue
        dst = os.path.join(sharded_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy(src, dst)
        copied += 1
        print(f"[copy_metadata] copied {entry}")

    print(f"[copy_metadata] complete: {copied} entries copied to {sharded_dir}")
    sharded_vol.commit()
