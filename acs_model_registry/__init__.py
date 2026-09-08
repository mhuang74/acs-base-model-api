"""Shared model registry for Modal serving and wrapper defaults.

This package is intentionally stdlib-only: Modal imports it while evaluating
decorators at module import time, and the wrapper imports it while parsing
settings. Do not add env reads, network calls, Modal imports, or FastAPI/Pydantic
dependencies here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_SCALEDOWN_WINDOW_S = 30 * 60
DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S = 5 * 60


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_repo: str
    served_model_name: str
    n_gpu: int
    gpu_type: str
    max_model_len: int
    app_name: str
    preshard: bool
    dtype: str
    tokenizer_repo: str | None = None
    status: str = "live"
    trust_remote_code: bool = False
    local_model_path: str | None = None
    n_nodes: int = 1
    rdma: bool = False
    coordination: str = "mp"
    topology: str = "tp_only"
    min_containers: int = 0
    max_containers: int = 4
    scaledown_window_s: int = DEFAULT_SCALEDOWN_WINDOW_S
    # Serve this model through the class-based GPU-memory-snapshot lifecycle
    # (Modal enable_memory_snapshot + enable_gpu_snapshot + vLLM sleep-mode)
    # instead of the shared function-based serve() (ACS-200). ONLY valid for
    # single-GPU models: a GPU snapshot captures one device's memory, so TP>1
    # models (405B/Trinity/Kimi) are snapshot-incompatible and MUST leave this
    # False. Lets the small/common model scale to zero yet cold-boot in ~20s.
    snapshot: bool = False
    # Volume that holds the pre-sharded (save_sharded_state) weights for this
    # model. Defaults to the shared "acs-sharded" Volume; models whose sharded
    # copy would push a shared Volume past Modal's 1 TB cap get a dedicated one
    # (e.g. Trinity — acs-sharded already holds 405B's ~810 GB).
    sharded_volume_name: str = "acs-sharded"
    # Whether the **0.19.1-pinned activation engine** may load this model's
    # sharded_state weights (ACS-199 item 5). Default ``False`` = fall back to the
    # HF-format ``local_model_path``/repo, which is always version-safe. Set
    # ``True`` ONLY for models whose current shards are verified loadable by
    # vLLM 0.19.1 — i.e. **dense** models (Llama), whose param names are stable
    # across 0.19↔0.23. MoE models (Trinity/afmoe) MUST stay ``False``: 0.23
    # renamed their params, so a sharded_state load under 0.19.1 dies with a
    # KeyError AFTER a paid multi-GPU boot (the ACS-197 crash). Default-deny is
    # deliberate: a forgotten flag costs a slow HF boot, never a crash. The
    # ``-v023`` volume-name hard-block in ``activation_can_load_sharded`` is a
    # second line of defence. Enforced MoE⇒False by a registry invariant test.
    activation_sharded_ok: bool = False
    # Activation engine (modal_app_activation.py) runtime — SEPARATE from the
    # serving knobs above, because harvesting/steering runs on its own
    # scale-to-zero engine with a heavier, burstier per-request cost. Env
    # ``ACT_SCALEDOWN_S`` / ``ACT_MAX_CONTAINERS`` still override at deploy time.
    # Defaults are conservative (short window, single container = the cost
    # guardrail); set a longer window per model where the GPU is cheap enough
    # that a warm engine beats repeated cold boots for interactive research.
    activation_scaledown_window_s: int = DEFAULT_ACTIVATION_SCALEDOWN_WINDOW_S
    activation_max_containers: int = 1
    # Warm-container floor for the activation engine. Default ``0`` = scale-to-
    # zero ($0 when no research runs). Set ``1`` for a cheap model where the
    # ~2-min activation cold boot is too disruptive for interactive
    # harvesting/steering and a warm GPU 24/7 is worth the cost (ACS-223, llama-
    # 8b). Env ``ACT_MIN_CONTAINERS`` overrides at deploy time.
    activation_min_containers: int = 0
    # Modal per-container request concurrency for the ACTIVATION engine
    # (``@modal.concurrent(max_inputs=...)`` in modal_app_activation.py) — separate
    # from the serving ``max_inputs`` below. Each concurrent capture request holds
    # an all-layers hidden-state blob, so the safe ceiling is memory/prompt-length
    # bound; keep the default conservative and raise per-model only after a load
    # test (ACS-249). Env ``ACT_MAX_INPUTS`` overrides at deploy time.
    activation_max_inputs: int = 8
    # Per-model cap on the PROMPT length of an activation-CAPTURE request
    # (``output_residual_stream``). ``None`` → the wrapper's
    # ``DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS`` (64). Since ACS-250 streams the
    # capture response (no wrapper-RAM materialization), the binding constraint is
    # the RESPONSE SIZE the client downloads (all layers × prompt × d_model),
    # which is per-model: ~0.27 MB/token on 8B (32×4096 bf16) vs ~4 MB/token on
    # 405B (126×16384). So raise it only where the response stays reasonable (8B)
    # and leave the big models tight until layer subsetting (ACS-227) lands.
    activation_max_prompt_tokens: int | None = None
    max_inputs: int = 128
    target_inputs: int | None = None
    upstream_timeout_s: float | None = None
    vllm_extra_args: list[str] = field(default_factory=list)
    serve_extra_volume_mount: str | None = None
    serve_extra_volume_name: str | None = None
    clustered_extra_volume_mount: str | None = None
    clustered_extra_volume_name: str | None = None

    @property
    def gpu_shape_label(self) -> str:
        total_gpus = self.n_gpu * self.n_nodes
        return f"{total_gpus}x{self.gpu_type}"

    @property
    def wrapper_tokenizer_repo(self) -> str:
        return self.tokenizer_repo or self.served_model_name

    @property
    def enable_expert_parallel(self) -> bool:
        """True when this model serves with vLLM expert-parallel (MoE).

        Single source of truth derived from ``vllm_extra_args`` so the preshard
        step and the serve step CANNOT drift: ``save_sharded_state`` writes one
        shard per TP rank capturing whatever expert slice lives on that rank, so
        the sharded weights are only correct if the EP config at save time
        matches the EP config at load time. Mismatch loads silently-wrong
        experts (no crash) → garbage generations.
        """
        return "--enable-expert-parallel" in self.vllm_extra_args

    def to_modal_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hf_repo": self.hf_repo,
            "served_model_name": self.served_model_name,
            "n_gpu": self.n_gpu,
            "gpu_type": self.gpu_type,
            "max_model_len": self.max_model_len,
            "app_name": self.app_name,
            "preshard": self.preshard,
            "dtype": self.dtype,
            "min_containers": self.min_containers,
            "max_containers": self.max_containers,
            "scaledown_window_s": self.scaledown_window_s,
            "max_inputs": self.max_inputs,
            "target_inputs": self.target_inputs or self.max_inputs,
        }
        if self.trust_remote_code:
            out["trust_remote_code"] = True
        if self.local_model_path:
            out["local_model_path"] = self.local_model_path
        if self.n_nodes != 1:
            out["n_nodes"] = self.n_nodes
        if self.rdma:
            out["rdma"] = True
        if self.coordination != "mp":
            out["coordination"] = self.coordination
        if self.topology != "tp_only":
            out["topology"] = self.topology
        if self.vllm_extra_args:
            out["vllm_extra_args"] = list(self.vllm_extra_args)
        if self.snapshot:
            out["snapshot"] = True
        out["activation_scaledown_window_s"] = self.activation_scaledown_window_s
        out["activation_max_containers"] = self.activation_max_containers
        out["activation_min_containers"] = self.activation_min_containers
        out["activation_max_inputs"] = self.activation_max_inputs
        return out

    def wrapper_defaults(self) -> dict[str, Any]:
        return {
            "served_model_name": self.served_model_name,
            "tokenizer_repo": self.wrapper_tokenizer_repo,
            "gpu_shape_label": self.gpu_shape_label,
            "status": self.status,
            "modal_app_name": self.app_name,
            "max_model_len": self.max_model_len,
            "upstream_timeout_s": self.upstream_timeout_s,
            # Per-model idle scaledown windows, so the wrapper's RPC-free
            # cold-hint (ModelEntry.scaledown_window_s / activation_scaledown_window_s)
            # matches the real Modal teardown per breaker key instead of a single
            # global 30-min guess (ACS-226). The activation engine scales down on
            # its own (usually shorter) window; both flow through unchanged.
            "scaledown_window_s": self.scaledown_window_s,
            "activation_scaledown_window_s": self.activation_scaledown_window_s,
            "activation_max_prompt_tokens": self.activation_max_prompt_tokens,
        }


SPECS: dict[str, ModelSpec] = {
    "llama-8b": ModelSpec(
        model_id="llama-8b",
        hf_repo="meta-llama/Llama-3.1-8B",
        served_model_name="meta-llama/Llama-3.1-8B",
        n_gpu=1,
        gpu_type="L40S",
        max_model_len=8192,
        app_name="acs-llama-8b",
        preshard=False,
        dtype="bfloat16",
        # ACS-200: served through the class-based GPU-memory-snapshot lifecycle
        # (single-GPU only; snapshot=True routes it to the class-based serve
        # path). ACS-223 puts it BACK on an always-on warm container
        # (min_containers=1), reversing the ACS-17/#193 scale-to-zero: the
        # snapshot restore (~20s warm-region, ~140s fresh per-region build,
        # ACS-201) is still too disruptive for interactive workbench use, so we
        # accept one warm L40S 24/7. Takes effect on the next
        # `MODEL_ID=llama-8b modal deploy modal_app.py` (human-gated prod flip —
        # see docs/design/gpu-snapshot-8b-cold-boot.md).
        snapshot=True,
        min_containers=1,
        max_containers=3,
        scaledown_window_s=DEFAULT_SCALEDOWN_WINDOW_S,
        # Activation engine: one always-on warm L40S (activation_min_containers=1,
        # ≈ $1.4k/mo) so the FIRST capture/steer is instant rather than paying the
        # ~30s snapshot wake — re-armed for daily interactive activation research
        # (operator decision 2026-07-21, reversing the ACS-248 scale-to-zero that
        # followed the ACS-214 workshop). Bursts to 5 containers under load
        # (activation_max_containers=5; the extra 4 only bill while scaled, 10-min
        # tail). snapshot=True stays for fast restores of those scale-ups.
        activation_scaledown_window_s=10 * 60,
        activation_max_containers=5,
        activation_min_containers=1,
        # Per-container concurrency, load-tested 2026-07-20 (ACS-249): 1×L40S with
        # ~64-tok captures took NO OOM through 32 concurrent (each response is
        # ~23 MB of all-layers activations), but p50 latency is flat to ~24 then
        # ~doubles at 32 (5.3s → 11.7s). 16 is the safe interactive value (2× the
        # default 8, p95 ~5.5s); raise once ACS-250 layer-subsetting cuts payload.
        activation_max_inputs=16,
        # ACS-250: capture streams now (no wrapper-RAM wall), so the bound is the
        # streamed response the client pulls — ~0.27 MB/token on 8B → 256 tokens
        # ≈ 68 MB. 4× the default 64; unblocks paragraph-length capture. Tunable.
        activation_max_prompt_tokens=256,
        max_inputs=128,
        target_inputs=100,
    ),
    # --- Staging upgrade candidate (ACS-114 / ACS-17) — NOT a production model.
    # A vLLM-0.23.0 clone of llama-8b for the dependency-version-census staging
    # track. Deploy it as its own Modal app (acs-llama-8b-v023), isolated from
    # the live acs-llama-8b (0.19.1), with:
    #   VLLM_VERSION=0.23.0 MODEL_ID=llama-8b-v023 modal deploy modal_app.py
    # status="staging" keeps it out of every live/wrapper listing (the wrapper's
    # live set is GET /v1/models over status=="live"); min_containers=0 → scales
    # to zero, so it carries no warm cost when idle. Delete the Modal app once
    # the upgrade is validated or abandoned.
    "llama-8b-v023": ModelSpec(
        model_id="llama-8b-v023",
        hf_repo="meta-llama/Llama-3.1-8B",
        served_model_name="meta-llama/Llama-3.1-8B",
        n_gpu=1,
        gpu_type="L40S",
        max_model_len=8192,
        app_name="acs-llama-8b-v023",
        preshard=False,
        dtype="bfloat16",
        status="staging",
        min_containers=0,
        max_containers=1,
        scaledown_window_s=DEFAULT_SCALEDOWN_WINDOW_S,
        max_inputs=128,
        target_inputs=100,
    ),
    # --- Staging sibling for the ACS-200 snapshot serve path — NOT production.
    # A snapshot-enabled clone of llama-8b used to validate the class-based
    # GPU-memory-snapshot lifecycle end-to-end on its OWN Modal app
    # (acs-llama-8b-snapprod), fully isolated from the live acs-llama-8b. Deploy:
    #   MODEL_ID=llama-8b-snapprod modal deploy modal_app.py
    # status="staging" keeps it out of the wrapper's live listing; min=0 → scales
    # to zero (no warm cost). Kept in-registry (like the -v023 siblings) so the
    # snapshot path can be re-validated later; delete the Modal app when idle.
    "llama-8b-snapprod": ModelSpec(
        model_id="llama-8b-snapprod",
        hf_repo="meta-llama/Llama-3.1-8B",
        served_model_name="meta-llama/Llama-3.1-8B",
        n_gpu=1,
        gpu_type="L40S",
        max_model_len=8192,
        app_name="acs-llama-8b-snapprod",
        preshard=False,
        dtype="bfloat16",
        snapshot=True,
        status="staging",
        min_containers=0,
        max_containers=1,
        # Short window (2 min) so the staging app idles to zero quickly between
        # measurements — lets a clean natural cold-restore be timed without
        # `modal container stop` (which triggers Modal's immediate re-prime).
        # Prod llama-8b keeps the DEFAULT 30-min window.
        scaledown_window_s=2 * 60,
        max_inputs=128,
        target_inputs=100,
    ),
    # Staging candidate (ACS-114 / ACS-17) — 405B on 0.23, 8xH200. Reuses the
    # EXISTING presharded weights on acs-sharded (same hf_repo + n_gpu=8 → same
    # sharded dir as prod llama-405b; shard format is 0.19.1<->0.23 compatible),
    # so no re-preshard. Deploy: VLLM_VERSION=0.23.0 TORCH_CUDA_INDEX=<cu130>
    # VLLM_USE_FLASHINFER_SAMPLER=0 MODEL_ID=llama-405b-v023 modal deploy ...
    "llama-405b-v023": ModelSpec(
        model_id="llama-405b-v023",
        hf_repo="meta-llama/Llama-3.1-405B",
        served_model_name="meta-llama/Llama-3.1-405B",
        n_gpu=8,
        gpu_type="H200",
        max_model_len=32768,
        app_name="acs-llama-405b-v023",
        preshard=True,
        dtype="bfloat16",
        status="staging",
        min_containers=0,
        max_containers=1,
        max_inputs=128,
        target_inputs=100,
    ),
    # Staging candidate (ACS-114 / ACS-17) — Trinity on 0.23, 8xH200. Reuses the
    # pre-staged weights on acs-trinity-cache (/cache/trinity-base). NOT presharded
    # (same as prod) → long boot. trust_remote_code + its vllm_extra_args carried
    # over verbatim (watch for 0.23 flag-compat).
    "trinity-truebase-v023": ModelSpec(
        model_id="trinity-truebase-v023",
        hf_repo="arcee-ai/Trinity-Large-TrueBase",
        served_model_name="arcee-ai/Trinity-Large-TrueBase",
        n_gpu=8,
        gpu_type="H200",
        max_model_len=8192,
        app_name="acs-trinity-base-v023",
        preshard=False,
        dtype="bfloat16",
        trust_remote_code=True,
        local_model_path="/cache/trinity-base",
        status="staging",
        min_containers=0,
        max_containers=1,
        max_inputs=128,
        target_inputs=100,
        serve_extra_volume_mount="/cache",
        serve_extra_volume_name="acs-trinity-cache",
        vllm_extra_args=[
            "--enable-expert-parallel",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "128",
            "--max-num-batched-tokens",
            "16384",
            "--kv-cache-dtype",
            "auto",
            "--enable-prefix-caching",
        ],
    ),
    "llama-405b": ModelSpec(
        model_id="llama-405b",
        hf_repo="meta-llama/Llama-3.1-405B",
        served_model_name="meta-llama/Llama-3.1-405B",
        n_gpu=8,
        gpu_type="H200",
        max_model_len=32768,
        app_name="acs-llama-405b",
        preshard=True,
        # Dense Llama: shards are cross-version-loadable, so the 0.19.1 activation
        # engine may use the presharded weights (fast cold boot). Verified by the
        # 405B activation capture-parity PASS (all 126 layers, #188). ACS-199 §5.
        activation_sharded_ok=True,
        dtype="bfloat16",
        max_containers=2,
        # ACS-315 (Matthew Sweeney / Modal cost review): tighten the serve idle
        # window from the 30-min default to 5 min. 405B is scale-to-zero
        # (min_containers=0), so a 30-min warm tail kept an idle 8×H200 billing for
        # 25 min after every spiky research burst. Tradeoff: requests spaced >5 min
        # apart re-pay the ~150-180s cold boot. Accepted for spiky low-traffic use.
        scaledown_window_s=5 * 60,
        # Activation engine: 8×H200 is expensive to keep warm, so a short 10-min
        # window (vs 8b's 30) and the single-container guardrail.
        activation_scaledown_window_s=10 * 60,
        max_inputs=128,
        target_inputs=100,
    ),
    "trinity-truebase": ModelSpec(
        model_id="trinity-truebase",
        hf_repo="arcee-ai/Trinity-Large-TrueBase",
        served_model_name="arcee-ai/Trinity-Large-TrueBase",
        n_gpu=8,
        gpu_type="H200",
        max_model_len=8192,
        app_name="acs-trinity-base",
        # Pre-shard to TP=8 with expert-parallel (ACS-17): Trinity's non-sharded
        # cold boot is weight-load-dominated (~340s, 31 HF shards resharded at
        # load). save_sharded_state must run with the SAME EP config as serve —
        # see ModelSpec.enable_expert_parallel. Sharded weights land in the
        # dedicated acs-trinity-sharded Volume; serve() prefers them over the
        # HF-format local_model_path fallback. Prerequisite for scale-to-zero.
        #
        # RE-ENABLED 2026-07-03 (ACS-197 fixed): the earlier crash was that 0.23
        # ships a NATIVE AfmoeForCausalLM whose parameter names differ from the
        # 0.19.1-saved shards (trust-remote-code afmoe), so ShardedStateLoader hit
        # KeyError: 'model.layers.6.mlp.experts._shared_experts.down_proj.weight'
        # and all 8 workers died in load_model. Fix = re-preshard on 0.23 (keys the
        # shards to the native names). Verified: presharded Trinity on the NEW 0.23
        # shards boots clean (no KeyError) and serves a coherent completion
        # ("The capital of France is" → " Paris, and the official language is
        # French …"), 2026-07-03. So point at the 0.23-generated volume below.
        # NOTE the 0.19.1 shards on acs-trinity-sharded are NOT 0.23-loadable —
        # do not repoint back without re-presharding.
        preshard=True,
        sharded_volume_name="acs-trinity-sharded-v023",
        dtype="bfloat16",
        trust_remote_code=True,
        local_model_path="/cache/trinity-base",
        # Always-on (operator decision 2026-07-13): keep one warm 8×H200 so the
        # first request never eats Trinity's long cold boot (~150-180s at this
        # size). Reverses the ACS-17/ACS-203 scale-to-zero (2026-07-06) — that
        # deploy was never actually rolled out, so the live app had stayed warm;
        # this makes the warm floor the committed intent rather than leftover
        # state. The activation engine stays scale-to-zero (activation_min_
        # containers=0 below). Expensive to keep warm — watch ACS-50 cost alerts.
        # Takes effect on the next `MODEL_ID=trinity-truebase modal deploy
        # modal_app.py` (the live app is already warm, so no change in behavior).
        min_containers=1,
        max_containers=2,
        # ACS-315 (Matthew Sweeney / Modal cost review): tighten the serve idle
        # window from the 30-min default to 5 min. While min_containers=1 keeps one
        # 8×H200 always warm, this window governs how long a BURST 2nd container
        # (max_containers=2) stays warm after load drops — 5 min instead of 30.
        # It also becomes the full idle window if min_containers is ever dropped to
        # 0 (the gated cost decision — see docs/design/acs-315-modal-cost-quick-wins.md).
        scaledown_window_s=5 * 60,
        # Activation engine: 8×H200 — short 10-min window + single container.
        activation_scaledown_window_s=10 * 60,
        max_inputs=128,
        target_inputs=100,
        serve_extra_volume_mount="/cache",
        serve_extra_volume_name="acs-trinity-cache",
        vllm_extra_args=[
            "--enable-expert-parallel",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "128",
            "--max-num-batched-tokens",
            "16384",
            "--kv-cache-dtype",
            "auto",
            "--enable-prefix-caching",
        ],
    ),
    "kimi-k2-base": ModelSpec(
        model_id="kimi-k2-base",
        hf_repo="moonshotai/Kimi-K2-Base",
        served_model_name="moonshotai/Kimi-K2-Base",
        n_gpu=8,
        n_nodes=2,
        rdma=True,
        coordination="ray",
        topology="tp16",
        gpu_type="H200",
        max_model_len=131072,
        app_name="acs-kimi-k2-base",
        preshard=False,
        dtype="fp8",
        trust_remote_code=True,
        max_containers=4,
        max_inputs=128,
        target_inputs=128,
        clustered_extra_volume_mount="/root/.cache/kimi",
        clustered_extra_volume_name="acs-kimi-cache",
        vllm_extra_args=[
            "--quantization",
            "fp8",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.85",
        ],
    ),
}

MODELS: dict[str, dict[str, Any]] = {
    model_id: spec.to_modal_config() for model_id, spec in SPECS.items()
}


def default_model_id(dev_mode: bool) -> str:
    """Return the legacy DEV_MODE-selected model id."""
    return "llama-8b" if dev_mode else "llama-405b"


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return SPECS[model_id]
    except KeyError as exc:
        raise RuntimeError(
            f"MODEL_ID={model_id!r} not in shared model registry. "
            f"Available: {sorted(SPECS)}"
        ) from exc


def maybe_model_spec(model_id: str) -> ModelSpec | None:
    return SPECS.get(model_id)


def get_model_config(model_id: str) -> dict[str, Any]:
    return get_model_spec(model_id).to_modal_config()


def wrapper_defaults(model_id: str) -> dict[str, Any] | None:
    spec = maybe_model_spec(model_id)
    if spec is None:
        return None
    return spec.wrapper_defaults()
