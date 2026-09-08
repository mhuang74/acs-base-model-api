"""Modal image, volume, and path definitions for the serving entrypoint."""

from __future__ import annotations

from dataclasses import dataclass

import modal

from acs_model_registry import ModelSpec

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
SHARDED_CACHE_DIR = "/root/.cache/sharded"
LIFETIME_DIR = "/lifetime"
LIFETIME_LOG_PATH = f"{LIFETIME_DIR}/lifetime.csv"
VLLM_PORT = 8000


@dataclass(frozen=True)
class ModalVolumes:
    hf_cache: modal.Volume
    vllm_cache: modal.Volume
    sharded: modal.Volume
    kimi_cache: modal.Volume
    trinity_cache: modal.Volume
    trinity_sharded: modal.Volume
    trinity_sharded_v023: modal.Volume
    lifetime: modal.Volume


# Production serving pins. Cutover 0.19.1 → 0.23.0 landed 2026-07-03 (ACS-114/
# ACS-121) — all three live models verified serving on 0.23 in prod.
PROD_VLLM_VERSION = "0.23.0"
# PyTorch CUDA wheel index. MUST match the CUDA version the pinned vLLM wheel
# was compiled against, or `import vllm._C` fails with a missing libcudart.so.
# vllm 0.19.1 → CUDA 12.8 (cu128). vllm 0.23.0 → torch 2.11.0 + CUDA 13 (cu130):
# the 0.23 wheel links libcudart.so.13, which only the cu130 torch build ships.
PROD_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"


def build_vllm_image(
    *,
    dev_mode_env: str,
    model_id: str,
    load_format_dummy_env: str = "0",
    vllm_version: str = PROD_VLLM_VERSION,
    torch_cuda_index: str = PROD_TORCH_CUDA_INDEX,
    extra_vllm_env: dict[str, str] | None = None,
) -> modal.Image:
    """Build the vLLM image shared by staging and serving functions.

    ``vllm_version`` defaults to the pinned production version
    (``PROD_VLLM_VERSION``, now **0.23.0** after the 2026-07-03 cutover). An env
    override (``VLLM_VERSION=…``) builds a different version — e.g. rolling back
    to 0.19.1, or testing a future candidate — WITHOUT changing the default.
    ``torch_cuda_index`` must move in lockstep with the vLLM version (its wheel is
    compiled against a specific CUDA): cu130 for 0.23.0, cu128 for 0.19.1. These
    are build-time values read locally during image construction, so overriding
    them via env at the call-site is correct (they are NOT one of the "reads
    don't reach the container" traps).

    ``VLLM_USE_FLASHINFER_SAMPLER=0`` is baked as a default (below) because 0.23
    defaults its top-k/top-p sampler to FlashInfer, which JIT-compiles a kernel
    at boot needing ``nvcc``; the lean debian_slim image has none, so the engine
    core dies. Forcing the native sampler avoids the JIT. ``extra_vllm_env`` can
    override it (e.g. if a future image ships nvcc). Forcing the native
    PyTorch sampler avoids the JIT entirely (also better for cold boot — no
    per-boot compile).
    """
    image_env = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        # Propagate to the remote container; Modal re-imports the
        # module there and the local shell env is not inherited.
        "DEV_MODE": dev_mode_env,
        "MODEL_ID": model_id,
        "LOAD_FORMAT_DUMMY": load_format_dummy_env,
        # Baked in for observability — the staging smoke logs this to
        # confirm which vLLM the container actually built with.
        "ACS_VLLM_VERSION": vllm_version,
        # REQUIRED for the 0.23 prod default: 0.23's top-k/top-p sampler defaults
        # to FlashInfer, which JIT-compiles a kernel at boot needing nvcc (absent
        # in debian_slim) → engine core dies. Native sampler avoids the JIT.
        # Harmless on 0.19.1 (native was already effectively used). Overridable
        # via extra_vllm_env if a future image ships nvcc.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    if extra_vllm_env:
        image_env.update(extra_vllm_env)
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            f"vllm=={vllm_version}",
            "huggingface_hub[hf_transfer]",
            extra_index_url=torch_cuda_index,
        )
        .env(image_env)
        .add_local_python_source("acs_model_registry", "serving")
    )


def create_volumes() -> ModalVolumes:
    """Create or attach all Modal Volumes used by this entrypoint."""
    return ModalVolumes(
        hf_cache=modal.Volume.from_name("acs-hf-cache", create_if_missing=True),
        vllm_cache=modal.Volume.from_name("acs-vllm-cache", create_if_missing=True),
        sharded=modal.Volume.from_name("acs-sharded", create_if_missing=True),
        kimi_cache=modal.Volume.from_name("acs-kimi-cache", create_if_missing=True),
        trinity_cache=modal.Volume.from_name(
            "acs-trinity-cache", create_if_missing=True
        ),
        # Dedicated sharded-weights Volume for Trinity — acs-sharded is at ~810 GB
        # (405B) against Modal's 1 TB per-Volume cap, so Trinity's ~797 GB sharded
        # copy cannot share it.
        trinity_sharded=modal.Volume.from_name(
            "acs-trinity-sharded", create_if_missing=True
        ),
        # Isolated sharded Volume for Trinity weights re-generated on vLLM 0.23
        # (ACS-197). trinity-truebase now points sharded_volume_name here; the
        # 0.19.1-format shards on acs-trinity-sharded crash 0.23's native afmoe
        # sharded_state load (KeyError), so they stay only as rollback.
        trinity_sharded_v023=modal.Volume.from_name(
            "acs-trinity-sharded-v023", create_if_missing=True
        ),
        lifetime=modal.Volume.from_name("acs-lifetime-log", create_if_missing=True),
    )


def sharded_dir_for_model(*, model_name: str, n_gpu: int) -> str:
    safe_model_name = model_name.replace("/", "__")
    return f"{SHARDED_CACHE_DIR}/{safe_model_name}-tp{n_gpu}"


def _volume_by_name(volumes: ModalVolumes, name: str) -> modal.Volume:
    named_volumes = {
        "acs-hf-cache": volumes.hf_cache,
        "acs-vllm-cache": volumes.vllm_cache,
        "acs-sharded": volumes.sharded,
        "acs-kimi-cache": volumes.kimi_cache,
        "acs-trinity-cache": volumes.trinity_cache,
        "acs-trinity-sharded": volumes.trinity_sharded,
        "acs-trinity-sharded-v023": volumes.trinity_sharded_v023,
        "acs-lifetime-log": volumes.lifetime,
    }
    try:
        return named_volumes[name]
    except KeyError as exc:
        raise RuntimeError(
            f"Unknown Modal Volume name in model registry: {name!r}"
        ) from exc


def sharded_volume_for(spec: ModelSpec, volumes: ModalVolumes) -> modal.Volume:
    """Return the Volume holding this model's pre-sharded weights.

    Defaults to the shared acs-sharded Volume; models with a dedicated one
    (Trinity) route here so preshard/copy_metadata/serve all mount the SAME
    Volume at SHARDED_CACHE_DIR.
    """
    return _volume_by_name(volumes, spec.sharded_volume_name)


def _maybe_extra_mount(
    mounts: dict[str, modal.Volume], *, spec: ModelSpec, volumes: ModalVolumes
) -> None:
    if spec.serve_extra_volume_mount and spec.serve_extra_volume_name:
        mounts[spec.serve_extra_volume_mount] = _volume_by_name(
            volumes,
            spec.serve_extra_volume_name,
        )


def preshard_volumes(
    *, spec: ModelSpec, volumes: ModalVolumes
) -> dict[str, modal.Volume]:
    """Volume mounts for the preshard() GPU function.

    Reads source weights (HF cache for 405B, or the extra Volume — e.g.
    /cache/trinity-base — for locally-staged models) and writes shards to the
    model's sharded Volume.
    """
    mounts = {
        HF_CACHE_DIR: volumes.hf_cache,
        VLLM_CACHE_DIR: volumes.vllm_cache,
        SHARDED_CACHE_DIR: sharded_volume_for(spec, volumes),
    }
    _maybe_extra_mount(mounts, spec=spec, volumes=volumes)
    return mounts


def copy_metadata_volumes(
    *, spec: ModelSpec, volumes: ModalVolumes
) -> dict[str, modal.Volume]:
    """Volume mounts for the copy_metadata() CPU function.

    Reads non-weight metadata (config, tokenizer, custom-arch .py) from the HF
    cache or the extra Volume, writes it beside the shards.
    """
    mounts = {
        HF_CACHE_DIR: volumes.hf_cache,
        SHARDED_CACHE_DIR: sharded_volume_for(spec, volumes),
    }
    _maybe_extra_mount(mounts, spec=spec, volumes=volumes)
    return mounts


def serve_volumes(*, spec: ModelSpec, volumes: ModalVolumes) -> dict[str, modal.Volume]:
    """Return the per-model Volume mounts for the single-node serve function."""
    mounts = {
        HF_CACHE_DIR: volumes.hf_cache,
        VLLM_CACHE_DIR: volumes.vllm_cache,
        SHARDED_CACHE_DIR: sharded_volume_for(spec, volumes),
        LIFETIME_DIR: volumes.lifetime,
    }
    _maybe_extra_mount(mounts, spec=spec, volumes=volumes)
    return mounts


def clustered_serve_volumes(
    *, spec: ModelSpec, volumes: ModalVolumes
) -> dict[str, modal.Volume]:
    """Return Volume mounts for multi-node clustered serving."""
    mounts = {
        HF_CACHE_DIR: volumes.hf_cache,
        VLLM_CACHE_DIR: volumes.vllm_cache,
        LIFETIME_DIR: volumes.lifetime,
    }
    if spec.clustered_extra_volume_mount and spec.clustered_extra_volume_name:
        mounts[spec.clustered_extra_volume_mount] = _volume_by_name(
            volumes,
            spec.clustered_extra_volume_name,
        )
    return mounts
