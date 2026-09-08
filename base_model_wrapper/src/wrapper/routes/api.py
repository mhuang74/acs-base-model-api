"""OpenAI-compatible API and health routes."""

from __future__ import annotations

import asyncio
import datetime as dt
import gzip
import json
import time
import uuid
import zlib
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import boot_stage as boot_stage_mod
from .. import breaker as breakermod
from .. import modal_ops
from .. import proxy as proxymod
from .. import workbench_generations as genmod
from ..cost_monitor import parse_gpu_shape
from ..db import get_session
from ..dependencies import get_http, get_settings
from ..logging import get_logger
from ..model_resolution import _resolve_model, _unknown_model_response
from ..models import ApiKey as ApiKeyRow
from ..models import HarvestJob
from ..rate_limit import limiter
from ..schemas import (
    DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS,
    FULL_VOCAB_MAX_PROMPT_TOKENS,
    FULL_VOCAB_SENTINEL,
    MAX_LOGPROBS,
    MAX_OUTPUT_WORK_TOKENS,
    MAX_STEERING_VECTORS,
    CompletionsRequest,
    HarvestRequest,
)
from ..services import completions as completion_svc
from ..services import health as health_svc
from ..services import request_log as request_log_svc
from ..settings import Settings
from ..tokenizer import get_token_counter
from ..warm_state import MODAL_SCALEDOWN_WINDOW, _is_model_warm, _mark_model_warm

log = get_logger()
router = APIRouter()
SSE_HEADERS = genmod.SSE_HEADERS
PUBLIC_KEEPALIVE_INTERVAL_S = 5.0
PUBLIC_RESPONSE_GRACE_S = 1.0
# Suffix that turns a serving model id into its activation-engine breaker /
# warm-state key (ACS-199). Built at the routing site and parsed back in
# ``_scaledown_window_for_key``. Single-sourced from modal_ops (ACS-221) so the
# runtime keys and the persisted gpu_cost_sample series can't drift.
ACTIVATION_KEY_SUFFIX = modal_ops.ACTIVATION_KEY_SUFFIX
# A 1 KiB JSON-whitespace heartbeat is still negligible over a 13-minute boot
# (~160 KiB total) and is less likely than a single byte to be buffered by an
# intermediary. RFC 8259 permits whitespace before the JSON value.
JSON_KEEPALIVE_CHUNK = b" " * 1024

# Response compression for the non-streaming completion JSON body. Full-vocab
# logprobs (ACS-191) make this body tens of MB; gzip is the standard payload
# mitigation and shrinks the highly-repetitive logprobs floats ~10-15×. Only the
# buffered ``application/json`` completion body is compressed — never the
# StreamingResponse paths (SSE + the cold-boot keepalive stream), whose
# incremental flushing the pure-ASGI middleware stack is built to preserve.
# 500 bytes matches Starlette GZipMiddleware's default floor: below it the gzip
# header/overhead outweighs the saving.
GZIP_MIN_SIZE = 500


def _out_of_range_layers(indices, n_layers: int) -> list[int]:
    """Return the sorted, deduped indices that fall outside ``[0, n_layers)``.

    Single-sourced so the pre-flight layer-range 400s — steering
    (``apply_steering_vectors``, ACS-322), inline activation *capture*
    (``output_residual_stream`` list, ACS-342), and bulk harvest
    (``layers`` list, ACS-342) — share one definition of "out of range" and
    one error vocabulary. The engine (vLLM-Lens) would otherwise raise a
    ValueError that surfaces as a mislabelled upstream 5xx / a 200 with an
    embedded 500, after burning GPU attempts.
    """
    return sorted({idx for idx in indices if not 0 <= idx < n_layers})


def _accepts_gzip(request: Request | None) -> bool:
    if request is None:
        return False
    return "gzip" in request.headers.get("accept-encoding", "").lower()


def _serialize_completion_json(payload: dict[str, Any]) -> tuple[bytes, bool]:
    """Serialize a completion body, gzipping it once it clears ``GZIP_MIN_SIZE``.

    Returns ``(bytes, is_gzipped)``. Byte-for-byte the same JSON as
    ``JSONResponse`` (compact separators, ``ensure_ascii=False``,
    ``allow_nan=False``) before compression. **CPU-bound** — a full-vocab body is
    tens of MB, so ``json.dumps`` + ``gzip`` here is hundreds of ms; callers run
    it off the event loop via ``asyncio.to_thread``. ``compresslevel=1`` gets
    within a few percent of the max ratio on the highly-repetitive logprobs
    floats at a fraction of the CPU of level 6.
    """
    body = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(body) >= GZIP_MIN_SIZE:
        return gzip.compress(body, compresslevel=1), True
    return body, False


async def _render_completion_json(
    status_code: int, payload: dict[str, Any], *, accept_gzip: bool
) -> Response:
    """Build the completion/error JSON response, gzipping it when worthwhile.

    When the client sent ``Accept-Encoding: gzip`` the serialize+compress runs in
    a worker thread (``asyncio.to_thread``) so a tens-of-MB full-vocab body never
    blocks the single event loop — which would stall every other in-flight
    request's keepalive/health flush. A body under ``GZIP_MIN_SIZE`` is returned
    uncompressed (reusing the bytes already serialized in the thread — no second
    ``json.dumps``). The common non-gzip path returns a plain ``JSONResponse``,
    unchanged. ``Vary: Accept-Encoding`` is always set so caches key on the
    client's capability.
    """
    if accept_gzip:
        data, gzipped = await asyncio.to_thread(_serialize_completion_json, payload)
        headers = {"Vary": "Accept-Encoding"}
        if gzipped:
            headers["Content-Encoding"] = "gzip"
        return Response(
            content=data,
            status_code=status_code,
            media_type="application/json",
            headers=headers,
        )
    response: Response = JSONResponse(status_code=status_code, content=payload)
    response.headers["Vary"] = "Accept-Encoding"
    return response


# Per-key safeguards on /v1/completions. Sixteen active requests (8→16 in
# ACS-249) leave room for many concurrent users against vLLM's --max-num-seqs
# 128. A bounded waiter
# queue still supports ordinary batch fan-out without allowing one key to
# accumulate unbounded asyncio futures. The generous per-minute cap catches
# sustained request floods while leaving normal eval/sweep traffic alone.
MAX_INFLIGHT_PER_KEY = 16  # per-key in-flight cap; 8→16 for scripted activation/completions pulls (ACS-249)
MAX_QUEUED_PER_KEY = 64
COMPLETIONS_RATE_LIMIT_PER_MINUTE = 600
COMPLETIONS_RATE_LIMIT = f"{COMPLETIONS_RATE_LIMIT_PER_MINUTE}/minute"
QUEUE_FULL_RETRY_AFTER_S = 5


class KeyConcurrencyGate:
    """Per-key active-slot semaphore with a bounded waiter queue."""

    def __init__(
        self,
        max_inflight: int = MAX_INFLIGHT_PER_KEY,
        max_queued: int = MAX_QUEUED_PER_KEY,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_inflight)
        self.max_queued = max_queued
        self.queued = 0

    async def acquire(self) -> bool:
        """Acquire an active slot; return False when the waiter cap is full."""
        if not self._semaphore.locked():
            await self._semaphore.acquire()
            return True
        if self.queued >= self.max_queued:
            return False
        self.queued += 1
        try:
            await self._semaphore.acquire()
        finally:
            self.queued -= 1
        return True

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> KeyConcurrencyGate:
        if not await self.acquire():
            raise RuntimeError("per-key concurrency queue is full")
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()


def _get_key_semaphore(app_state, key_id: uuid.UUID) -> KeyConcurrencyGate:
    """Lazy-create one bounded concurrency gate per key id.

    Safe to call from multiple coroutines because dict.setdefault is atomic
    under the GIL and the gate constructor has no awaits.
    """
    gate = app_state.key_semaphores.get(key_id)
    if gate is None:
        gate = app_state.key_semaphores.setdefault(key_id, KeyConcurrencyGate())
    return gate


def _scaledown_window_for_key(app_state, breaker_key: str) -> dt.timedelta:
    """Resolve the idle scaledown window for a warm-state / breaker key.

    ``breaker_key`` is either a serving ``model_id`` or a ``<model_id>::activation``
    key (ACS-199). The two point at *different* Modal apps that scale to zero on
    their own schedules — the activation side-car is usually shorter (10 min for
    the deployed models) than serving (30 min). Using a single global constant
    for both (ACS-226) judged an activation engine "warm" for up to 30 min when
    it had already torn down at 10, so a cold boot skipped the keepalive path and
    sent zero bytes during the multi-minute boot (silent-socket failure).

    Per-model windows come from the registry entry (``ModelEntry.scaledown_window_s``
    / ``activation_scaledown_window_s``). Every routed key resolves to a live
    ``app_state.models`` entry today (breaker keys derive from ``_resolve_model``,
    and the legacy single-model fallback still builds a real entry), so the
    ``entry is None`` branch is defence-in-depth for a future caller passing an
    unregistered key: fall back to the global ``MODAL_SCALEDOWN_WINDOW`` rather
    than raise on the hot path.
    """
    is_activation = breaker_key.endswith(ACTIVATION_KEY_SUFFIX)
    model_id = breaker_key.removesuffix(ACTIVATION_KEY_SUFFIX)
    entry = getattr(app_state, "models", {}).get(model_id)
    if entry is None:
        return MODAL_SCALEDOWN_WINDOW
    seconds = entry.activation_scaledown_window_s if is_activation else entry.scaledown_window_s
    return dt.timedelta(seconds=seconds)


def _model_is_cold(app_state, model_id: str | None) -> bool:
    """Best-effort, RPC-free guess of whether ``model_id`` is scaled-to-zero.

    Used to set ``BackendContext.cold_hint`` so the proxy can classify a
    stalled upstream socket (the 8×H200 cold-boot signature) as a cold boot
    instead of an ``upstream_unreachable`` 502. A completion within the key's
    scaledown window means Modal still has a warm container; otherwise the next
    request pays a cold boot. The window is per breaker key
    (``_scaledown_window_for_key``) so an ``<model>::activation`` key uses the
    activation engine's shorter teardown, not the serving engine's (ACS-226).
    ``/health``'s ``warm_estimate`` applies the same per-model idea on the
    serving key (``backend_health_entry`` uses ``entry.scaledown_window_s``).

    Deliberately in-memory only (the ``last_completion_at`` map) — NO Modal
    RPC on the hot completion path. This is a *hint*: a false "cold" only
    changes which classification a real stall gets (503 vs 502, both honest
    for a model that isn't answering), and a false "warm" just preserves the
    old 502-after-retries behaviour. The proxy still requires an actual
    timeout/stall before raising ColdBootError, so a warm-but-flagged-cold
    model that answers normally is unaffected.
    """
    if not model_id:
        return False
    last_at = getattr(app_state, "last_completion_at", {}).get(model_id)
    if last_at is None:
        # Never served a completion this process lifetime → assume cold. After
        # a wrapper restart this over-reports cold, but the only effect is that
        # the first stall yields a (correct) 503 instead of a 502.
        return True
    now = dt.datetime.now(tz=dt.UTC)
    return (now - last_at) >= _scaledown_window_for_key(app_state, model_id)


# --- dependencies ------------------------------------------------------------


def _validation_message(exc: Exception) -> str:
    """Compose a user-facing message from a Pydantic ValidationError.

    Pydantic's default str() is verbose and includes internal noise (the
    ``pydantic.dev/2.x/v/...`` URLs and ``input_value=`` echoes). For the
    beta API we want one line per problem with the field path + the human
    reason, so callers can fix typos at a glance.
    """
    return completion_svc.validation_message(exc)


async def _check_sequence_length(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
) -> JSONResponse | None:
    """Return a 400 JSONResponse if prompt + max_tokens would exceed
    ``entry.max_model_len``, else None.

    Skipped when:
      - the registry entry has no ``max_model_len`` declared (legacy fallback)
      - the prompt is empty (vLLM accepts empty prompts; nothing to enforce)

    Tokenizer cost: this adds one tokenizer pass for streaming requests that
    previously skipped tokenization. For non-streaming the budget enforcement
    inside ``run_completion_nonstream`` will tokenize again — that's two
    passes, but the tokenizer is in-process and the cost is in microseconds
    for typical prompts (and the cache means it's a no-op the second time
    for the same string). Premature de-duplication is not worth the
    code-complexity here.
    """
    return await completion_svc.check_sequence_length(
        session,
        request_id,
        caller,
        ip,
        t0,
        entry,
        parsed,
        settings,
        error_response=_error,
        token_counter_factory=get_token_counter,
    )


async def _check_full_vocab_prompt_logprobs(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
) -> JSONResponse | None:
    """Return a 400 if a full-vocab ``prompt_logprobs=-1`` prompt is too long.

    Thin route-level wrapper around the service guard (ACS-191). No-op unless
    the caller asked for the ``-1`` full-vocab sentinel.
    """
    return await completion_svc.check_full_vocab_prompt_logprobs(
        session,
        request_id,
        caller,
        ip,
        t0,
        entry,
        parsed,
        settings,
        error_response=_error,
        token_counter_factory=get_token_counter,
    )


async def _check_activation_prompt_length(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
    effective_max_tokens: int | None = None,
) -> JSONResponse | None:
    """Return a 400 if an activation-capture response would exceed the per-model cap.

    Thin route-level wrapper around the service guard (ACS-199). No-op unless the
    request sets ``output_residual_stream``. ``effective_max_tokens`` is the
    generated-token count the request will actually use (after the None →
    per-model default is resolved), so the cap counts prompt + generated − 1
    positions (ACS-255).
    """
    return await completion_svc.check_activation_prompt_length(
        session,
        request_id,
        caller,
        ip,
        t0,
        entry,
        parsed,
        settings,
        error_response=_error,
        token_counter_factory=get_token_counter,
        effective_max_tokens=effective_max_tokens,
    )


def _default_max_tokens(
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
) -> int:
    """Return a bounded default for a single-prompt request.

    OpenAI-compatible clients commonly omit ``max_tokens``. Preserve that
    convenience without forwarding vLLM's potentially context-filling
    default: cap generated output at 32k and, when the model context is known,
    at the remaining context window.
    """
    if entry.max_model_len is None:
        return MAX_OUTPUT_WORK_TOKENS
    prompt = parsed.prompt
    counter = get_token_counter(entry.tokenizer_repo, settings.hf_token)
    prompt_tokens = completion_svc.count_prompt_tokens(prompt, counter)
    remaining_context = max(1, entry.max_model_len - prompt_tokens)
    return min(MAX_OUTPUT_WORK_TOKENS, remaining_context)


def _apply_extras_headers(response: Response, extras: dict[str, Any]) -> None:
    """Translate the run_completion_nonstream extras dict into response headers.

    Currently surfaces the budget-driven ``max_tokens`` clamp so callers can
    detect that their requested ``max_tokens`` was reduced rather than have
    it disappear silently. Format chosen so a single header value is greppable
    in logs: ``X-Acs-Max-Tokens-Clamped: requested=N,applied=M,reason=budget``.

    Also surfaces ``upstream_error_kind`` (when set by the upstream-error
    handlers in run_completion_nonstream) as ``X-Acs-Upstream-Error-Kind`` so
    on-call triage can grep for the failure mode without parsing the JSON
    body.
    """
    completion_svc.apply_extras_headers(response, extras)


def _apply_backend_headers(response: Response, ctx: proxymod.BackendContext) -> None:
    """Tag every response with the backend identity so triage can answer
    "which upstream served / failed this request?" from the response alone.

    Set on success AND error paths. Cheap to add; invaluable during incidents.
    The values come from the wrapper-side registry, not the upstream — so
    they're stable across upstream churn and safe to expose.
    """
    completion_svc.apply_backend_headers(response, ctx)


async def _record_request(
    session: AsyncSession,
    *,
    request_id: str,
    caller: authmod.AuthedCaller | None,
    ip: str | None,
    endpoint: str,
    model: str | None,
    n_prompt: int | None,
    n_completion: int | None,
    status_code: int,
    latency_ms: int,
    upstream_latency_ms: int | None,
    error_kind: str | None,
    telemetry: request_log_svc.RequestTelemetry | None = None,
    cold_boot: bool = False,
    ttft_ms: int | None = None,
) -> None:
    """Persist one `api_requests` row AND emit the privacy-safe stdout line.

    Per the plan, every request lands in `api_requests` for cost attribution +
    abuse detection (write-heavy table; rolled into `usage_monthly` by a cron
    later). Retention is intentionally not pruned during the beta so the
    multi-week usage-pattern curves survive. Unauthenticated requests are
    skipped here because the schema requires a key_id FK — those still log to
    stdout. ``telemetry`` / ``cold_boot`` / ``ttft_ms`` carry the beta
    event-logging fields (pure metadata, never body content).
    """
    await request_log_svc.record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        model=model,
        n_prompt=n_prompt,
        n_completion=n_completion,
        status_code=status_code,
        latency_ms=latency_ms,
        upstream_latency_ms=upstream_latency_ms,
        error_kind=error_kind,
        telemetry=telemetry,
        cold_boot=cold_boot,
        ttft_ms=ttft_ms,
    )


_VALID_WORKLOADS = {"batch", "interactive"}


def _parse_workload(raw: str | None) -> str | None:
    """Validate the optional ``X-Acs-Workload`` header.

    Telemetry only — an unrecognized value is dropped to ``None`` rather than
    rejected, so a stray header never fails an otherwise-valid request.
    """
    if not raw:
        return None
    v = raw.strip().lower()
    return v if v in _VALID_WORKLOADS else None


# --- public endpoints --------------------------------------------------------


async def _ping_database(engine) -> tuple[bool, int]:
    """Quick ``SELECT 1`` round-trip. Returns (ok, latency_ms).

    Tight timeout: Railway's healthcheck calls /health every few seconds, so
    a slow DB check would itself become a self-imposed outage. We bound at
    ~500 ms — if Postgres can't answer within that we'd rather call the
    wrapper "down" and let Railway restart than pretend everything's fine.
    """
    return await health_svc.ping_database(engine)


def _backend_health_entry(
    entry,
    breaker_status,
    last_completion_at,
    now: dt.datetime,
) -> dict[str, Any]:
    """Per-backend block for /health. Pure derivation from in-memory state —
    no Modal RPCs, no DB queries, so /health stays cheap under load.

    ``warm`` = a completion came back within ``MODAL_SCALEDOWN_WINDOW``, so
    the upstream container is likely still in Modal's pool. ``cold`` =
    never seen a completion, or last one is older than the window — Modal
    would have scaled the container down and the next request will pay a
    cold-boot.
    """
    return health_svc.backend_health_entry(entry, breaker_status, last_completion_at, now)


async def aggregate_health(app_state) -> tuple[int, dict[str, Any]]:
    """Build the /health response from in-memory state + a single DB ping.

    Returns ``(http_status, body)``:
      - 503 when the wrapper itself is broken (DB unreachable) — Railway's
        healthcheck reads the status code, so 503 triggers restart-on-degraded.
      - 200 with ``status: "degraded"`` when any backend breaker is open.
        The wrapper is fine; one model is sick. Don't restart for this.
      - 200 with ``status: "ok"`` when every backend's breaker is closed.

    Active probing of upstreams is deliberately NOT done. For Modal-backed
    8×H200 models a probe either (a) triggers a $108/hr cold boot or (b)
    returns 303 which doesn't distinguish "healthy but cold" from "broken".
    The breaker already learns from real traffic at zero cost; that's the
    long-term-correct signal.
    """
    now = dt.datetime.now(tz=dt.UTC)
    boot_time = getattr(app_state, "boot_time", now)
    uptime_s = int((now - boot_time).total_seconds())

    db_ok, db_latency_ms = await _ping_database(app_state.engine)

    registry = app_state.models
    breakers = app_state.breakers
    last_completion_map = app_state.last_completion_at

    backends: list[dict[str, Any]] = []
    open_backends: list[str] = []
    for model_id in sorted(registry.keys()):
        entry = registry[model_id]
        # Skip disabled entries — they're not part of the active surface.
        if entry.status == "disabled":
            continue
        snap = breakers.snapshot(model_id)
        backends.append(
            _backend_health_entry(
                entry,
                snap,
                last_completion_map.get(model_id),
                now,
            )
        )
        if snap.state == "open":
            open_backends.append(model_id)

    if not db_ok:
        overall = "down"
        http_status = 503
    elif open_backends:
        overall = "degraded"
        http_status = 200
    else:
        overall = "ok"
        http_status = 200

    body = {
        "status": overall,
        "wrapper": {
            "uptime_seconds": uptime_s,
            "database": {
                "ok": db_ok,
                "latency_ms": db_latency_ms,
            },
        },
        "backends": backends,
        "degraded_backends": open_backends,
    }
    return http_status, body


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Aggregated health/readiness for the wrapper + every live backend.

    Public (no auth) so external monitors and Railway can hit it. The body
    deliberately omits secrets and request bodies — same privacy stance as
    every other public endpoint.

    Response shape (stable contract):
    ```
    {
      "status": "ok" | "degraded" | "down",
      "wrapper": {
        "uptime_seconds": int,
        "database": {"ok": bool, "latency_ms": int}
      },
      "backends": [
        {
          "model_id": str,
          "status": "live" | "staging",
          "gpu_shape": str,
          "breaker_state": "closed" | "open" | "half_open",
          "breaker_consecutive_failures": int,
          "breaker_last_failure_kind": str | null,
          "last_completion_at": ISO-8601 | null,
          "last_completion_seconds_ago": int | null,
          "warm_estimate": "warm" | "cold"
        },
        ...
      ],
      "degraded_backends": [str]   // model_ids whose breakers are open
    }
    ```

    HTTP status: 503 when the wrapper itself can't serve (DB down), else 200
    (including ``degraded`` — one bad model doesn't mean the wrapper should
    be restarted). Backward compatible: the legacy ``{"status": "ok"}`` shape
    is preserved as a subset.
    """
    http_status, body = await aggregate_health(request.app.state)
    return JSONResponse(status_code=http_status, content=body)


@router.get("/v1/models")
@limiter.limit("60/minute")
async def models(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Return the wrapper's model registry in OpenAI-shape, with capability flags.

    Replaces the previous upstream proxy: each upstream Modal app only knows
    about its own served model, so aggregating from the wrapper registry is
    the source of truth across all live models.

    The ``capabilities`` block per model is the beta-grade contract: it tells
    callers what this model supports without round-tripping. Fields:
      - ``max_model_len``: total prompt+completion tokens vLLM will accept.
        ``None`` if unknown (registry didn't declare it; pre-flight check is
        skipped for this model).
      - ``max_logprobs``: cap on a POSITIVE (top-k) ``logprobs`` /
        ``prompt_logprobs`` count.
      - ``prompt_logprobs_full_vocab``: whether ``prompt_logprobs=-1`` (the whole
        distribution per prompt position) is supported.
      - ``full_vocab_max_prompt_tokens``: max prompt length for a full-vocab
        ``prompt_logprobs=-1`` request.
      - ``max_output_work_tokens``: cap on
        ``prompt_count × n × max_tokens``.
      - ``max_inflight_per_key`` / ``max_queued_per_key``: per-key active and
        waiting request limits.
      - ``rate_limit_per_minute``: per-key request-rate ceiling.
      - ``logprobs``, ``prompt_logprobs``: bool feature flags (always True
        today; reserved for future backends without these capabilities).
      - ``chat_template``: always ``null`` — these are base models, prompts
        pass through verbatim without templating. Surfaced explicitly so
        clients can detect the no-chat-template contract.
    """
    await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    data = []
    for entry in request.app.state.models.values():
        if entry.status != "live":
            continue
        data.append(
            {
                "id": entry.model_id,
                "object": "model",
                "owned_by": "acs",
                "served_model_name": entry.served_model_name,
                "gpu_shape": entry.gpu_shape_label,
                "status": entry.status,
                "capabilities": {
                    "max_model_len": entry.max_model_len,
                    "max_logprobs": MAX_LOGPROBS,
                    "max_output_work_tokens": MAX_OUTPUT_WORK_TOKENS,
                    "max_inflight_per_key": MAX_INFLIGHT_PER_KEY,
                    "max_queued_per_key": MAX_QUEUED_PER_KEY,
                    "rate_limit_per_minute": COMPLETIONS_RATE_LIMIT_PER_MINUTE,
                    "logprobs": True,
                    "prompt_logprobs": True,
                    # Full-vocabulary logprobs (ACS-191): pass prompt_logprobs=-1
                    # to get the model's whole next-token distribution at each
                    # prompt position. Prompt-only (completion logprobs stay
                    # top-k, capped at max_logprobs) and limited to short prompts
                    # because the payload is ~vocab_size values per token.
                    "prompt_logprobs_full_vocab": True,
                    "full_vocab_max_prompt_tokens": FULL_VOCAB_MAX_PROMPT_TOKENS,
                    # Activation harvesting + steering (ACS-199): true only for
                    # models with an activation engine configured. Clients check
                    # this before sending output_residual_stream (capture ALL
                    # layers; filter client-side) / apply_steering_vectors (a
                    # request to a non-activation model is a 400). Caps advertised
                    # only when supported: max steering vectors + the per-model
                    # activation prompt-token cap. The cap bounds RESPONSE SIZE,
                    # which scales with captured layers — so it applies to an
                    # all-layer capture, and a subset request is allowed
                    # proportionally more (ACS-317). ``n_layers`` is published so
                    # clients can compute their own ceiling.
                    "activations": entry.activation_upstream_url is not None,
                    **(
                        {
                            "max_steering_vectors": MAX_STEERING_VECTORS,
                            "max_activation_prompt_tokens": (
                                entry.activation_max_prompt_tokens
                                or DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS
                            ),
                            **(
                                {"n_layers": entry.n_layers}
                                if entry.n_layers
                                else {}
                            ),
                        }
                        if entry.activation_upstream_url is not None
                        else {}
                    ),
                    "chat_template": None,
                },
            }
        )
    return JSONResponse(status_code=200, content={"object": "list", "data": data})


@router.get("/v1/models/{model_id}/status")
@limiter.limit("60/minute")
async def model_status(
    model_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Warm/cold state + current cold-boot stage for one model (ACS-272).

    The non-streaming completion path can't carry progress text (its keepalive
    is JSON whitespace on a committed 200), so clients waiting out a cold boot
    poll this instead — e.g. from a second terminal:

        curl -H "Authorization: Bearer $KEY" .../v1/models/llama-405b/status

    ``boot`` is the latest authored stage the serving container published to
    the boot-status Dict (see boot_stage.py), or null when there's no fresh
    entry — for a cold model that means Modal hasn't started our container
    yet (GPU allocation), for a warm one simply that the last boot aged out.
    ``warm`` reuses the workbench warm-pill signal (Modal runner count with a
    recent-completion fallback). Both are advisory; the model's actual
    behaviour on a request is always the ground truth.

    For activation-enabled models the response also carries
    ``activation_boot`` — the ACTIVATION engine's stage (ACS-276). Activation
    *capture* requests are non-streaming by design (the whole residual-stream
    tensor comes back as one JSON body), so this field is the only progress
    surface while a capture waits out a cold activation boot; the engines
    scale independently, so ``boot``/``warm`` say nothing about it. The key is
    absent for models without an activation engine.
    """
    await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    entry = request.app.state.models.get(model_id)
    if entry is None or entry.status != "live":
        return _unknown_model_response(request, model_id)
    now = dt.datetime.now(tz=dt.UTC)
    last_at = request.app.state.last_completion_at.get(model_id)
    warm = await _is_model_warm(entry, last_at, now)
    st = None
    if entry.modal_app_name:
        st = await boot_stage_mod.snapshot(entry.modal_app_name)
    content: dict[str, Any] = {
        "id": entry.model_id,
        "object": "model.status",
        "warm": warm,
        "boot": st.as_payload() if st is not None else None,
    }
    if entry.activation_upstream_url is not None:
        act_st = await boot_stage_mod.snapshot(
            modal_ops.resolve_activation_app(entry, model_id)
        )
        content["activation_boot"] = act_st.as_payload() if act_st is not None else None
    return JSONResponse(status_code=200, content=content)


@router.post("/v1/chat/completions")
async def chat_completions_unsupported(
    request: Request, settings: Settings = Depends(get_settings)
) -> JSONResponse:
    """Loud 400 by design.

    This is a base-model API; we don't want researchers paste-from-OpenAI-tutorial
    to silently 404 — they need to be redirected to /v1/completions explicitly.
    """
    msg = "This is a base-model API. Use /v1/completions instead (raw prompt, no chat template)."
    if settings.chat_completions_handout_url:
        msg += f" See {settings.chat_completions_handout_url}."
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "message": msg,
                "type": "invalid_request_error",
                "code": "chat_completions_unsupported",
            }
        },
    )


# Human-readable error messages keyed by budget dimension (item 9). Used by
# ``run_completion_nonstream`` when refusing a call up-front. The "exhausted"
# variant fires when the dimension is already at zero; the "prompt alone"
# variant fires when the prompt's token count would push the dimension below
# zero on its own (no room for any completion).
_BUDGET_MESSAGES: dict[str, str] = {
    "key_monthly": "Monthly token budget exhausted for this API key.",
    "user_monthly": "Monthly token budget exhausted for this user (aggregate across keys).",
    "daily": "Daily token budget exhausted for this API key.",
    "input": "Monthly input-token budget exhausted for this API key.",
    "output": "Monthly output-token budget exhausted for this API key.",
}

_PROMPT_ALONE_MESSAGES: dict[str, str] = {
    "key_monthly": "Prompt alone would exceed remaining monthly budget for this key.",
    "user_monthly": "Prompt alone would exceed remaining monthly budget for this user.",
    "daily": "Prompt alone would exceed remaining daily budget for this key.",
}


async def _budget_preflight_clamp(
    *,
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    endpoint: str,
    t0: float,
    body: dict[str, Any],
    tokenizer_repo: str | None,
    settings: Settings,
    telemetry: request_log_svc.RequestTelemetry | None,
    extras: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    """Multi-dimensional budget enforcement (item 9), shared pre-flight.

    For each dimension that's actually configured (non-None), check the
    pre-flight headroom; if any is already at 0, 429 without calling
    upstream. The final ``max_tokens`` is clamped to the smallest
    output-side headroom.

    Returns the ``(status, payload)`` 429 tuple when a budget rejects the
    request, else None. Mutates ``body`` (the ``max_tokens`` clamp) and
    ``extras`` (clamp metadata for the ``X-Acs-Max-Tokens-Clamped`` headers).
    Extracted from ``run_completion_nonstream`` so the full-vocab
    pass-through path (ACS-198) enforces the same budgets.
    """
    remaining_by_dim = caller.effective_remaining()
    # Fast path: caller has no limits at all → skip tokenizer entirely (kept
    # explicit so the existing "unlimited budget skips tokenizer" test still
    # passes byte-identical to before).
    has_any_limit = any(v is not None for v in remaining_by_dim.values())
    if not has_any_limit:
        return None

    # Reject up-front when any non-NULL dimension is already exhausted.
    for dim, rem in remaining_by_dim.items():
        if rem is not None and rem <= 0:
            return await _budget_error(
                session,
                request_id,
                caller,
                ip,
                endpoint,
                t0,
                _BUDGET_MESSAGES[dim],
                telemetry=telemetry,
            )

    prompt = body.get("prompt") or ""
    tok_repo = tokenizer_repo or settings.served_model_name
    counter = get_token_counter(tok_repo, settings.hf_token)
    prompt_tokens_est = completion_svc.count_prompt_tokens(prompt, counter)

    # ``input`` is the only dimension that compares directly against
    # prompt tokens (output budget doesn't count the prompt). For every
    # *total* dimension the prompt eats into the same pool as the
    # completion, so headroom = remaining − prompt_tokens.
    if remaining_by_dim["input"] is not None and prompt_tokens_est > remaining_by_dim["input"]:
        return await _budget_error(
            session,
            request_id,
            caller,
            ip,
            endpoint,
            t0,
            "Prompt would exceed remaining input-token budget.",
            telemetry=telemetry,
        )

    output_headrooms: list[int] = []
    for dim in ("key_monthly", "user_monthly", "daily"):
        rem = remaining_by_dim[dim]
        if rem is None:
            continue
        hr = rem - prompt_tokens_est
        if hr <= 0:
            return await _budget_error(
                session,
                request_id,
                caller,
                ip,
                endpoint,
                t0,
                _PROMPT_ALONE_MESSAGES[dim],
                telemetry=telemetry,
            )
        output_headrooms.append(hr)
    if remaining_by_dim["output"] is not None:
        output_headrooms.append(remaining_by_dim["output"])

    if output_headrooms:
        original, clamped = proxymod.clamp_max_tokens(body, min(output_headrooms))
        # Loud-fail the silent clamp: when the caller's requested
        # ``max_tokens`` got reduced (or when they sent None and we filled
        # in a tighter value), record it so the /v1/completions handler
        # can set X-Acs-Max-Tokens-Clamped on the response.
        if clamped is not None and original != clamped:
            extras["max_tokens_clamped"] = {
                "requested": original,
                "applied": clamped,
                "reason": "budget",
            }
            log.info(
                "max_tokens_clamped",
                request_id=request_id,
                key_id=str(caller.key_id),
                requested=original,
                applied=clamped,
                reason="budget",
            )
    return None


async def run_completion_nonstream(
    *,
    session: AsyncSession,
    http: httpx.AsyncClient,
    settings: Settings,
    caller: authmod.AuthedCaller,
    body: dict[str, Any],
    ip: str | None,
    endpoint: str = "/v1/completions",
    request_id: str | None = None,
    upstream_url: str | None = None,
    tokenizer_repo: str | None = None,
    model_id: str | None = None,
    request: Request | None = None,
    backend_ctx: proxymod.BackendContext | None = None,
    breakers: breakermod.BackendBreakers | None = None,
    timeout_s: float | None = None,
    telemetry: request_log_svc.RequestTelemetry | None = None,
    transport_extras: dict[str, Any] | None = None,
    breaker_key: str | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Pre-flight budget clamp → upstream `/v1/completions` (non-streaming) →
    commit usage → record an `api_requests` row.

    Returns ``(status_code, payload, extras)`` where:
      - ``payload`` is either the upstream response body on success, or an
        OpenAI-shaped ``{"error": {...}}`` dict on failure.
      - ``extras`` is a dict for the caller's transport layer to surface as
        response headers / logs without round-tripping through the body.
        Currently used for budget-driven ``max_tokens`` clamping (the
        ``X-Acs-Max-Tokens-Clamped`` family of headers); empty on the happy
        path where nothing was clamped.

    Callers convert the tuple to whatever response form they want:
    - ``/v1/completions``: wrap in ``JSONResponse(status_code=..., content=...)``
      and merge ``extras`` into response headers.
    - ``/chat`` POST: extract ``payload["choices"][0]["text"]`` on 2xx, flash
      ``payload["error"]["message"]`` on >=400; ``extras`` is ignored.
    """
    t0 = time.monotonic()
    if request_id is None:
        request_id = (
            getattr(request.state, "request_id", None) if request is not None else None
        ) or f"req_{uuid.uuid4().hex[:20]}"

    extras: dict[str, Any] = transport_extras if transport_extras is not None else {}

    budget_err = await _budget_preflight_clamp(
        session=session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        t0=t0,
        body=body,
        tokenizer_repo=tokenizer_repo,
        settings=settings,
        telemetry=telemetry,
        extras=extras,
    )
    if budget_err is not None:
        status, payload = budget_err
        return status, payload, extras
    resolved_upstream = upstream_url or f"{settings.modal_base_url.rstrip('/')}/v1/completions"
    # ``model`` recorded in api_requests is the wrapper-side short id when
    # we have one (multi-model path); otherwise whatever the body carried.
    model_name = model_id or body.get("model")
    # Breaker + warm-state key. The activation path passes a distinct
    # ``breaker_key`` (``<model_id>::activation``) so its health can't
    # contaminate the workbench engine's (ACS-199); ordinary completions default
    # to ``model_id``. ``model_name`` (above) stays the real id for logging.
    bkey = breaker_key or model_id
    effective_timeout = timeout_s or settings.upstream_timeout_s

    # Circuit breaker pre-check: short-circuit if the backend is in OPEN state.
    # The breaker is per-backend; the chat/workbench paths (which don't call us)
    # do their own breaker checks via _serve_stream.
    if breakers is not None and bkey is not None:
        if not await breakers.allow(bkey):
            status_snap = breakers.snapshot(bkey)
            payload = breakermod.circuit_open_payload(status_snap)
            await _record_request(
                session,
                request_id=request_id,
                caller=caller,
                ip=ip,
                endpoint=endpoint,
                model=model_name,
                n_prompt=None,
                n_completion=None,
                status_code=503,
                latency_ms=int((time.monotonic() - t0) * 1000),
                upstream_latency_ms=None,
                error_kind="circuit_open",
                telemetry=telemetry,
            )
            extras["upstream_error_kind"] = "circuit_open"
            return 503, payload, extras

    try:
        status_code, payload, upstream_ms = await proxymod.post_nonstream(
            http,
            resolved_upstream,
            settings.vllm_api_key,
            body,
            effective_timeout,
            ctx=backend_ctx,
        )
    except proxymod.ColdBootError as exc:
        # Cold-boot is NOT a breaker failure — Modal is doing its thing.
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=503,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind="cold_boot",
            telemetry=telemetry,
            cold_boot=True,
        )
        return (
            503,
            proxymod.cold_boot_error_payload(
                exc.upstream_status, retry_after_s=proxymod.COLD_BOOT_BACKOFF_S
            ),
            extras,
        )
    except proxymod.UpstreamUnreachable as exc:
        if breakers is not None and bkey is not None:
            await breakers.record_failure(bkey, "upstream_unreachable")
        extras["upstream_error_kind"] = "upstream_unreachable"
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=502,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind="upstream_unreachable",
            telemetry=telemetry,
        )
        return 502, proxymod.upstream_unreachable_payload(exc), extras
    except proxymod.UpstreamServerError as exc:
        kind = exc.upstream_kind or "upstream_5xx"
        # A client-ish upstream error (e.g. an out-of-range steering layer_index
        # reflected by vLLM-Lens as a 500, ACS-322) is the caller's fault, not a
        # server fault: return 400 (keeps it out of the real_5xx alarm) and do
        # NOT record a breaker failure — otherwise repeated bad requests would
        # open the circuit and 503 every caller on this model.
        client_ish = exc.upstream_kind in proxymod.CLIENT_ISH_UPSTREAM_KINDS
        resp_status = 400 if client_ish else 502
        if breakers is not None and bkey is not None and not client_ish:
            await breakers.record_failure(bkey, kind)
        extras["upstream_error_kind"] = kind
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=resp_status,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind=kind,
            telemetry=telemetry,
        )
        return resp_status, proxymod.upstream_server_error_payload(exc), extras

    n_prompt = n_completion = None
    if status_code < 400 and isinstance(payload, dict):
        usage = payload.get("usage") or {}
        n_prompt = int(usage.get("prompt_tokens", 0) or 0)
        n_completion = int(usage.get("completion_tokens", 0) or 0)
        if n_prompt + n_completion > 0:
            await authmod.commit_usage(session, caller.key_id, n_prompt, n_completion)
        if request is not None:
            _mark_model_warm(request, bkey)
        # Successful response → close any half-open breaker / reset counter.
        if breakers is not None and bkey is not None:
            await breakers.record_success(bkey)

    error_kind = None
    if status_code >= 500:
        error_kind = "upstream_5xx"
    elif status_code >= 400:
        error_kind = "upstream_4xx"

    await _record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        model=model_name,
        n_prompt=n_prompt,
        n_completion=n_completion,
        status_code=status_code,
        latency_ms=int((time.monotonic() - t0) * 1000),
        upstream_latency_ms=upstream_ms,
        error_kind=error_kind,
        telemetry=telemetry,
        cold_boot=bool(backend_ctx and backend_ctx.cold_hint and status_code < 400),
    )
    return status_code, payload or {}, extras


async def _budget_error(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    endpoint: str,
    t0: float,
    message: str,
    *,
    telemetry: request_log_svc.RequestTelemetry | None = None,
) -> tuple[int, dict[str, Any]]:
    await _record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        model=None,
        n_prompt=None,
        n_completion=None,
        status_code=429,
        latency_ms=int((time.monotonic() - t0) * 1000),
        upstream_latency_ms=None,
        error_kind="budget_exceeded",
        telemetry=telemetry,
    )
    return 429, {
        "error": {"message": message, "type": "invalid_request_error", "code": "budget_exceeded"}
    }


@router.post("/v1/completions")
@limiter.limit(COMPLETIONS_RATE_LIMIT)
async def completions(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    t0 = time.monotonic()
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None
    caller = await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    # Authentication opens a transaction (and may update last_used_at). Close
    # it before waiting on the per-key gate so queued requests do not retain a
    # Postgres connection for the duration of the upstream backlog.
    await session.commit()

    # Per-key in-flight cap: block until a slot frees up, but bound the waiter
    # queue so a runaway client cannot pin unlimited asyncio futures. For
    # streams the slot is held by _serve_stream until the generator finishes.
    sem = _get_key_semaphore(request.app.state, caller.key_id)
    if not await sem.acquire():
        response = await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/completions",
            t0,
            429,
            "queue_full",
            (
                f"Too many queued requests for this API key "
                f"(maximum {MAX_QUEUED_PER_KEY} waiting behind "
                f"{MAX_INFLIGHT_PER_KEY} active). Retry later."
            ),
        )
        response.headers["Retry-After"] = str(QUEUE_FULL_RETRY_AFTER_S)
        return response
    sem_released = False
    try:
        try:
            raw_body: dict[str, Any] = await request.json()
        except Exception:
            return await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/completions",
                t0,
                400,
                "bad_json",
                "Request body must be valid JSON.",
            )

        # Validate against the beta schema BEFORE resolving the model or
        # branching into stream/non-stream. ``extra="forbid"`` catches typos
        # like ``temprature=0.5`` that would otherwise be silently dropped by
        # vLLM; range checks catch values vLLM would 400 on with a less clear
        # message. The model dump preserves only the fields the caller
        # explicitly set, so vLLM's own defaults still apply downstream.
        try:
            parsed = CompletionsRequest.model_validate(raw_body)
        except Exception as exc:  # pydantic.ValidationError + the rare TypeError
            return await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/completions",
                t0,
                400,
                "invalid_request",
                _validation_message(exc),
            )

        requested_model = parsed.model
        resolved = _resolve_model(request, requested_model)
        if resolved is None:
            return _unknown_model_response(request, requested_model)
        model_id, entry = resolved

        # Activation harvesting / steering routing (ACS-199). A request carrying
        # activation params is dispatched to the model's separate vLLM-Lens
        # activation engine (acs-<id>-activation), never the workbench upstream,
        # so research instrumentation can't slow production completions. Reject
        # when the model has no activation engine configured — the schema
        # accepts the fields shape-wise; per-model support is a routing concern.
        wants_activation = parsed.has_activation_params
        if wants_activation and not entry.activation_upstream_url:
            return await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/completions",
                t0,
                400,
                "activations_unsupported",
                f"model {model_id!r} does not support activation harvesting/steering; "
                "omit output_residual_stream / apply_steering_vectors "
                "(GET /v1/models lists activation-capable models)",
            )

        # Reject an out-of-range steering OR capture ``layer_index`` at the
        # boundary with a clear 400, instead of letting vLLM-Lens raise
        # ValueError -> upstream 500. For steering that surfaces as a mislabelled
        # 5xx (ACS-322); for activation *capture* it's worse — capture routes
        # through the streamed ``passthrough_post`` (ACS-250), so the 200 status
        # line + keepalive bytes are already committed by the time the upstream
        # 500 lands, yielding a 200 body with no ``choices`` after 4 wasted GPU
        # attempts (ACS-342). Runs before the stream/non-stream fork, so it
        # covers both. Only when ``n_layers`` is known (published per-model,
        # ACS-317); otherwise the proxy's error-body classification is the safety
        # net. Negative *capture* indices are already rejected at schema parse
        # (schemas.py); this adds the model-specific UPPER bound the schema
        # deliberately punts here because it can't see ``n_layers``.
        if wants_activation and entry.n_layers is not None:
            if parsed.apply_steering_vectors:
                bad_layers = _out_of_range_layers(
                    (idx for vec in parsed.apply_steering_vectors for idx in vec.layer_indices),
                    entry.n_layers,
                )
                if bad_layers:
                    return await _error(
                        session,
                        request_id,
                        caller,
                        ip,
                        "/v1/completions",
                        t0,
                        400,
                        "invalid_request",
                        f"apply_steering_vectors layer_index {bad_layers} out of range "
                        f"for {model_id!r}: valid decoder layer indices are "
                        f"0-{entry.n_layers - 1}",
                    )
            if isinstance(parsed.output_residual_stream, list):
                bad_capture = _out_of_range_layers(
                    parsed.output_residual_stream, entry.n_layers
                )
                if bad_capture:
                    return await _error(
                        session,
                        request_id,
                        caller,
                        ip,
                        "/v1/completions",
                        t0,
                        400,
                        "invalid_request",
                        f"output_residual_stream layer index {bad_capture} out of range "
                        f"for {model_id!r}: valid decoder layer indices are "
                        f"0-{entry.n_layers - 1}",
                    )

        # Per-key monthly activation quota (ACS-199). Activation requests each
        # wake an expensive activation GPU engine, so they get a separate cap
        # from ordinary completions (0 = unlimited). Checked pre-flight against
        # the count of already-committed activation requests this month; the
        # rejected request is NOT counted (it never reached the engine).
        # Soft cap by design: the count read and the enforcement aren't a single
        # atomic transaction (no row-lock), so a burst of concurrent requests
        # right at the boundary can each pass. Acceptable for a monthly quota —
        # the overrun is bounded by per-key concurrency (16), not unbounded.
        if wants_activation and caller.monthly_activation_budget:
            used = await request_log_svc.count_activation_requests_this_month(
                session, caller.key_id
            )
            if used >= caller.monthly_activation_budget:
                return await _error(
                    session,
                    request_id,
                    caller,
                    ip,
                    "/v1/completions",
                    t0,
                    429,
                    "activation_quota_exceeded",
                    f"monthly activation-request quota reached "
                    f"({used}/{caller.monthly_activation_budget}); it resets at the "
                    "start of next month",
                )

        # Build the upstream body for the chosen engine: the activation body
        # nests the controls under ``vllm_xargs`` (what vLLM-Lens reads); the
        # normal body excludes them so a plain completion never carries them.
        body: dict[str, Any] = (
            parsed.to_activation_upstream_body() if wants_activation else parsed.to_upstream_body()
        )
        # Resolve the generated-token count once: an omitted max_tokens becomes
        # the per-model default. Reused for the upstream body AND the activation
        # capture cap (which counts prompt + generated − 1 positions, ACS-255).
        effective_max_tokens = (
            parsed.max_tokens
            if parsed.max_tokens is not None
            else _default_max_tokens(entry, parsed, settings)
        )
        if parsed.max_tokens is None:
            # Single-prompt requests may omit max_tokens for OpenAI
            # compatibility, but never inherit vLLM's context-filling default.
            body["max_tokens"] = effective_max_tokens
        # Substitute the upstream's served-model name into the body so vLLM
        # accepts it (vLLM matches on its --served-model-name flag).
        body["model"] = entry.served_model_name
        upstream_base = entry.activation_upstream_url if wants_activation else entry.upstream_url
        upstream_url = f"{upstream_base.rstrip('/')}/v1/completions"
        # Isolate the activation engine's health + warm state from the workbench
        # engine (ACS-199 PR3). The activation engine is a separate Modal app, so
        # it gets a distinct circuit-breaker + warm-tracking key: a flaky/cold
        # activation container can no longer trip the breaker that gates
        # production completions on the same model (and vice-versa). The real
        # ``model_id`` is still what's logged to api_requests — only the breaker,
        # warm-state, and backend identity switch. (Remaining shared knob:
        # ``upstream_timeout_s`` — fine for the 8B-only v1; a separate activation
        # timeout is a small registry follow-up when big-model activation lands.)
        # Known follow-up: the /health sweep iterates registry models, so the
        # ``::activation`` breaker keys aren't surfaced there yet — an activation
        # engine outage no longer shows in /health (tracked separately).
        breaker_key = f"{model_id}{ACTIVATION_KEY_SUFFIX}" if wants_activation else model_id
        # Backend identity (BackendContext → logs/headers, and since ACS-272
        # also the boot-stage Dict key the streaming keepalive polls): resolve
        # via the shared helper — deriving from ``modal_app_name`` mislabels
        # Trinity, whose serving app kept the pre-rename name while its
        # activation app is ``acs-trinity-truebase-activation`` (ACS-221).
        backend_app_name = (
            modal_ops.resolve_activation_app(entry, model_id)
            if wants_activation
            else (entry.modal_app_name or "")
        )

        # Pre-flight sequence-length check: catch prompt+max_tokens > max_model_len
        # at the wrapper boundary with a clear 400, rather than queueing the
        # request to vLLM and surfacing whatever the upstream error layer
        # produces (often opaque, sometimes mid-stream). Skipped when the
        # registry entry didn't declare ``max_model_len`` (legacy fallback).
        seq_err = await _check_sequence_length(
            session,
            request_id,
            caller,
            ip,
            t0,
            entry,
            parsed,
            settings,
        )
        if seq_err is not None:
            return seq_err

        # Full-vocab prompt_logprobs (-1) blows up as prompt_len × vocab_size —
        # reject an over-long full-vocab prompt at the boundary before it OOMs
        # the model server (ACS-191). No-op for ordinary top-k requests.
        full_vocab_err = await _check_full_vocab_prompt_logprobs(
            session,
            request_id,
            caller,
            ip,
            t0,
            entry,
            parsed,
            settings,
        )
        if full_vocab_err is not None:
            return full_vocab_err

        # Activation capture streams the residual-stream tensor back (since
        # ACS-250 — no wrapper-RAM materialization), so the binding constraint is
        # the RESPONSE SIZE the client downloads. Bound it by a per-model cap on
        # captured positions — prompt + generated − 1 (ACS-199, ACS-255). No-op
        # unless output_residual_stream is set.
        activation_len_err = await _check_activation_prompt_length(
            session,
            request_id,
            caller,
            ip,
            t0,
            entry,
            parsed,
            settings,
            effective_max_tokens=effective_max_tokens,
        )
        if activation_len_err is not None:
            return activation_len_err

        # Build the backend identity context once — passed into proxy +
        # breaker so error payloads / logs / response headers all name the
        # same upstream consistently.
        backend_ctx = proxymod.BackendContext(
            model_id=model_id,
            gpu_shape=entry.gpu_shape_label or "",
            # Name the actual backend app (``…-activation`` for activation
            # requests) so logs/headers point at the engine that served it.
            modal_app_name=backend_app_name,
            # RPC-free warm-state guess so the proxy can turn a stalled cold
            # boot into a fast 503 modal_cold_boot instead of a 502 past the
            # Railway edge window. Keyed on ``breaker_key`` so the activation
            # engine's warmth is tracked separately. See _model_is_cold.
            cold_hint=_model_is_cold(request.app.state, breaker_key),
        )
        effective_timeout = entry.upstream_timeout_s or settings.upstream_timeout_s
        breakers: breakermod.BackendBreakers = request.app.state.breakers

        # Beta event-logging: capture request shape once (stream flag + declared
        # workload + sampling params) and attach to every api_requests row for
        # this request. Pure metadata — no prompt/completion content.
        is_stream = bool(body.get("stream"))
        # Whether we may gzip the (non-streaming) JSON response body.
        accept_gzip = _accepts_gzip(request)
        telemetry = request_log_svc.RequestTelemetry.from_completions_request(
            parsed,
            stream=is_stream,
            workload_type=_parse_workload(request.headers.get("x-acs-workload")),
        )

        if is_stream:
            response = await _serve_stream(
                request_id,
                caller,
                ip,
                upstream_url,
                body,
                settings,
                session,
                http,
                t0,
                model_id,
                request=request,
                key_semaphore=sem,
                backend_ctx=backend_ctx,
                breakers=breakers,
                effective_timeout=effective_timeout,
                telemetry=telemetry,
                breaker_key=breaker_key,
            )
            # _serve_stream owns the slot iff it actually returned a StreamingResponse;
            # the error-path JSONResponse it can return on cold-boot does not.
            if isinstance(response, StreamingResponse):
                sem_released = True
            # Identity headers on whatever the stream path returned (JSONResponse
            # on cold-boot/breaker, StreamingResponse on success).
            _apply_backend_headers(response, backend_ctx)
            return response

        # Tens-of-MB bodies stream through as bytes instead of materializing in
        # wrapper RAM (ACS-198 full-vocab logprobs; ACS-250 activation capture).
        # _passthrough_usage_mode decides which shapes qualify and where their
        # ``usage`` block sits ("head" for capture, "tail" for full-vocab).
        # Steering-only requests and the full-vocab+capture combo deliberately
        # stay on the buffered path (see the helper's docstring).
        _passthrough_mode = _passthrough_usage_mode(parsed)
        if _passthrough_mode is not None:
            response = await _serve_fullvocab_passthrough(
                request_id=request_id,
                caller=caller,
                ip=ip,
                upstream_url=upstream_url,
                body=body,
                settings=settings,
                session=session,
                http=http,
                t0=t0,
                model_id=model_id,
                request=request,
                key_semaphore=sem,
                backend_ctx=backend_ctx,
                breakers=breakers,
                effective_timeout=effective_timeout,
                tokenizer_repo=entry.tokenizer_repo,
                telemetry=telemetry,
                accept_gzip=accept_gzip,
                breaker_key=breaker_key,
                usage_at_head=_passthrough_mode == "head",
            )
            # The streaming generator owns the per-key slot; error-path JSON
            # responses leave release to this handler's finally.
            if isinstance(response, StreamingResponse):
                sem_released = True
            _apply_backend_headers(response, backend_ctx)
            return response

        if backend_ctx.cold_hint:
            response = await _serve_nonstream_with_keepalive(
                request_id=request_id,
                caller=caller,
                ip=ip,
                upstream_url=upstream_url,
                body=body,
                settings=settings,
                session=session,
                http=http,
                model_id=model_id,
                request=request,
                key_semaphore=sem,
                backend_ctx=backend_ctx,
                breakers=breakers,
                effective_timeout=effective_timeout,
                tokenizer_repo=entry.tokenizer_repo,
                telemetry=telemetry,
                accept_gzip=accept_gzip,
                breaker_key=breaker_key,
            )
            if isinstance(response, StreamingResponse):
                sem_released = True
        else:
            status_code, payload, extras = await run_completion_nonstream(
                session=session,
                http=http,
                settings=settings,
                caller=caller,
                body=body,
                ip=ip,
                endpoint="/v1/completions",
                request_id=request_id,
                upstream_url=upstream_url,
                tokenizer_repo=entry.tokenizer_repo,
                model_id=model_id,
                request=request,
                backend_ctx=backend_ctx,
                breakers=breakers,
                timeout_s=effective_timeout,
                telemetry=telemetry,
                breaker_key=breaker_key,
            )
            response = await _completion_json_response(
                status_code, payload, extras, accept_gzip=accept_gzip
            )
        _apply_backend_headers(response, backend_ctx)
        return response
    finally:
        if not sem_released:
            sem.release()


# --- Self-serve bulk activation harvest (ACS-245) ----------------------------
#
# POST /v1/harvest spawns the offline bulk harvester
# (serving/harvest_offline.py, deployed per-model as ``acs-<model>-harvest``)
# for the caller's own prompts; GET /v1/harvest/<id> polls the job and, once
# finished, returns the manifest/shard URLs. Replaces the operator-run
# ``modal run serving/harvest_offline.py`` for external users — see
# docs/design/activation-offline-harvest.md ("Self-serve trigger").

# Presigned bucket URLs in a harvest result carry a 7-day TTL (see the design
# doc). ``urls_expire_at`` in the GET payload is completed_at + this window;
# the wrapper holds no bucket credentials, so expired URLs are re-minted by an
# operator re-running the upload pass, never here.
HARVEST_URL_TTL = dt.timedelta(days=7)

# Long-poll bounds for GET /v1/harvest/<id>?wait=<s> (ACS-321). 60s is short
# enough to sit inside every proxy idle timeout in the path (Railway's is
# minutes, but clients and corporate proxies are the unknown), and long enough
# that a client polling loop becomes one request per minute instead of dozens.
# The interval is what a waiting request costs us: one Modal poll every 3s.
HARVEST_MAX_WAIT_S = 60
HARVEST_WAIT_POLL_INTERVAL_S = 3.0

# How many requests may be WAITING at once (ACS-321 follow-up). `poll_harvest`
# runs via `asyncio.to_thread`, i.e. the event loop's DEFAULT ThreadPoolExecutor
# — shared with full-vocab logprob serialization, zstd compression and
# `spawn_harvest`. Unbounded waiters therefore do not just cost harvest: at 500
# concurrent, `to_thread` latency was measured going from 0.5ms to ~3s, which
# lands on /v1/completions traffic that has nothing to do with harvesting. (The
# DB pool is fine — the connection is released around each sleep.)
#
# Over either ceiling the poll is answered EARLY rather than refused: returning
# current state early is within the endpoint's contract, whereas a 429 could
# break a client loop that treats it as fatal. "Early" is not "instantly" — a
# declined poll still costs one `poll_harvest`, and the loop our docs recommend
# has no sleep of its own, so an instant answer would turn a declined client
# into a hot loop bounded only by the rate limit (~10 req/s, i.e. far MORE
# thread-pool pressure than the wait it was denied) exactly when the pool is
# busiest. Declined responses are paced by one poll interval and carry
# Retry-After. The ceilings live in Settings: they bound a shared resource on
# numbers that are estimates, so they must be tunable without a deploy.
_LONGPOLL_INFLIGHT: dict[uuid.UUID, int] = {}

# Retry-After (seconds) advertised on the retryable harvest 429s
# (``harvest_concurrency_exceeded`` / ``harvest_capacity_exceeded``, ACS-344).
# The QUEUE_FULL_RETRY_AFTER_S sibling is 5s, but a harvest job runs for
# minutes, not the sub-second a queued completion waits — polling every 5s just
# burns 429s (a tester saw 44 of them over 915s with no header to pace them).
# 30s is a sane back-off floor; the client can always poll GET /v1/harvest/<id>
# for the actual finish, or DELETE it to free the slot immediately. The monthly
# ``harvest_quota_exceeded`` 429 gets no header — that resets next month, not in
# seconds.
HARVEST_RETRY_AFTER_S = 30


def _harvest_job_payload(job: Any) -> dict[str, Any]:
    """Client-facing job dict shared by the POST 202 and the GET responses."""
    out: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
        "model": job.model_id,
        "run_id": job.run_id,
        "n_prompts": (job.params or {}).get("n_prompts"),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    if job.status == "done":
        out["result"] = job.result
        if job.completed_at is not None:
            expires = job.completed_at + HARVEST_URL_TTL
            out["urls_expire_at"] = expires.isoformat()
            if dt.datetime.now(tz=dt.UTC) > expires:
                # Still served (the stored objects persist — S3 on an upload run,
                # the Volume on a smoke; an operator can re-mint), but flagged so
                # clients don't chase dead links.
                out["urls_expired"] = True
        accounting = _harvest_accounting(job)
        if accounting is not None:
            out["accounting"] = accounting
    elif job.status == "failed":
        out["error"] = job.error
    return out


# The phase timings the Modal harvest() function reports, in seconds. Only these
# are summed into ``in_container_s`` — ``forward_tok_s`` / ``upload_mb_s`` are
# RATES (also *_s-suffixed) and would corrupt the sum if included (ACS-343).
_HARVEST_TIMING_KEYS = ("load_s", "forward_s", "save_s", "commit_s", "upload_s")


def _harvest_accounting(job: Any) -> dict[str, float] | None:
    """Attribute a done job's wall clock vs. the work its own timings measured.

    ACS-343: a one-prompt job showed a ~1025s wall clock with ``load_s: 0`` and
    ~1.6s of ``timings`` — ~99.8% unaccounted. The harvest ``timings`` dict is
    assembled INSIDE the Modal function (``serving/harvest_offline.py``) and only
    clocks in-container phases (load/forward/save/commit/upload). It is
    structurally blind to the time BEFORE the function body runs: the Modal
    FunctionCall sitting enqueued waiting for a container/GPU slot, plus
    container scheduling/boot — and to the wrapper's poll cadence in OBSERVING
    completion. This surfaces that gap so a long wall clock on trivial work is
    explicable without guessing:

      * ``wall_clock_s`` — wrapper-observed spawn→finalize (created_at →
        completed_at); includes up to one poll interval of observation lag.
      * ``in_container_s`` — sum of the reported in-container phase timings.
      * ``unaccounted_s`` — the difference: queue wait + container
        scheduling/boot before the function body + poll cadence; NOT GPU work.

    Returns ``None`` when the timestamps or a well-formed ``timings`` dict are
    missing (never fabricate a number).
    """
    if job.created_at is None or job.completed_at is None:
        return None
    result = job.result if isinstance(job.result, dict) else None
    timings = result.get("timings") if isinstance(result, dict) else None
    if not isinstance(timings, dict):
        return None
    in_container = 0.0
    for k in _HARVEST_TIMING_KEYS:
        v = timings.get(k)
        if isinstance(v, (int, float)):
            in_container += float(v)
    wall_s = (job.completed_at - job.created_at).total_seconds()
    return {
        "wall_clock_s": round(wall_s, 2),
        "in_container_s": round(in_container, 2),
        # Clamp at 0: observation lag can make in_container marginally exceed a
        # rounded wall clock; a negative "unaccounted" would only confuse.
        "unaccounted_s": round(max(0.0, wall_s - in_container), 2),
    }


def _resolve_harvest_app(entry: Any, model_id: str) -> str | None:
    """Modal app name of this model's bulk harvester, or None if unsupported.

    Thin alias of ``modal_ops.resolve_harvest_app`` (single source shared with
    the cost sampler since ACS-281 — same pattern as ``resolve_activation_app``).
    See that docstring for the derivation rules and the ACS-273 rename trap.
    """
    return modal_ops.resolve_harvest_app(entry, model_id)


def _harvest_is_big_model(entry: Any) -> bool:
    """True when a harvest on this model runs on a multi-GPU (expensive) container.

    Uses the canonical ``parse_gpu_shape`` — which matches BOTH ``"8xH200"`` and
    the prod/dev-local ``"8×H200"`` (U+00D7) form. An ASCII-only ``"x"`` split
    silently no-ops the cap on the ``×`` labels (that exact bug once left
    gpu_cost_sample empty — see cost_monitor). >1 GPU ⇒ big: the 8×H200 (405B,
    Trinity) and 16×H200 (Kimi) harvests that must not fan out unbounded across
    keys; the cheap 1×L40S (8B) harvest stays uncapped. Unparseable label ⇒ small
    (fail-open — never block a harvest over a display-string quirk).
    """
    shape = parse_gpu_shape(getattr(entry, "gpu_shape_label", None))
    return shape is not None and shape[0] > 1


@router.post("/v1/harvest")
@limiter.limit(COMPLETIONS_RATE_LIMIT)
async def harvest_submit(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Start one bulk activation-harvest job; returns 202 + a job id to poll.

    Guard rails for untrusted input:
      - Content-Length cap (``harvest_max_body_bytes``) rejected 413 before the
        body is even read;
      - prompt COUNT cap (``harvest_max_prompts``) and a chars/4
        estimated-token cap (``harvest_max_est_tokens``) — cheap bounds on GPU
        time + activation storage, no tokenizer pass over thousands of prompts;
      - per-key monthly job quota (``api_keys.monthly_harvest_budget``,
        0 = unlimited) counted against ``harvest_jobs`` rows;
      - per-key running-job cap (``harvest_max_running_per_key``) so one key
        can't fan out GPU containers.

    The quota gate is RACE-FREE: one transaction takes ``SELECT ... FOR
    UPDATE`` on the caller's api_keys row (serializing this key's submits),
    ages out stale jobs, runs both counts, and inserts the job as ``pending``
    — so a concurrent submit blocks on the lock and then sees the pending row.
    The transaction commits BEFORE the Modal spawn (no DB connection held
    across the await); the row then flips to ``running`` (or ``failed`` on a
    spawn error — such rows never engaged a GPU and don't burn monthly quota).
    """
    t0 = time.monotonic()
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None
    # Same auth + scope as /v1/completions — a completions-scoped key may
    # harvest; the budget knobs (not a new scope) are the abuse control.
    caller = await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    # Close the auth transaction (may have updated last_used_at) before doing
    # any slow work, mirroring completions().
    await session.commit()

    # Body-size guard BEFORE request.json() materializes the corpus in RAM.
    # A missing/unparseable Content-Length proceeds — the est-token cap still
    # bounds what a parsed body may cost.
    raw_len = request.headers.get("content-length")
    if raw_len is not None:
        try:
            content_length = int(raw_len)
        except ValueError:
            content_length = None
        if content_length is not None and content_length > settings.harvest_max_body_bytes:
            return await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/harvest",
                t0,
                413,
                "request_too_large",
                f"request body of {content_length} bytes exceeds the "
                f"{settings.harvest_max_body_bytes}-byte cap for /v1/harvest — "
                "split the corpus into multiple jobs",
            )

    try:
        raw_body: dict[str, Any] = await request.json()
    except Exception:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            400,
            "bad_json",
            "Request body must be valid JSON.",
        )
    try:
        parsed = HarvestRequest.model_validate(raw_body)
    except Exception as exc:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            400,
            "invalid_request",
            _validation_message(exc),
        )

    resolved = _resolve_model(request, parsed.model)
    if resolved is None:
        return _unknown_model_response(request, parsed.model)
    model_id, entry = resolved
    # Harvest support = a Modal-backed model (the harvest sibling app is named
    # after the MODEL ID, or set explicitly via ``harvest_app_name``).
    # Registry entries with neither (legacy / ad-hoc upstreams) have no harvest
    # path at all — distinct from the 503 below, which means "supported but the
    # harvest app isn't deployed yet".
    harvest_app = _resolve_harvest_app(entry, model_id)
    if harvest_app is None:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            400,
            "harvest_unsupported",
            f"model {model_id!r} does not support bulk activation harvest",
        )

    # Reject out-of-range harvest ``layers`` pre-flight (ACS-342), mirroring the
    # inline steering/capture guard in completions(). The schema caps indices at
    # a model-agnostic [0, 512] sanity bound; this adds the per-model UPPER bound
    # so an index that's valid-shaped but out of range for THIS model fails fast
    # here instead of after a Modal spawn (a wasted GPU boot that then fails
    # remotely). ``layers`` is None (engine default subset) or "all" for the
    # common cases — only an explicit int list is range-checked. Only when
    # ``n_layers`` is known (ACS-317); otherwise the remote fail-fast stands.
    if entry.n_layers is not None and isinstance(parsed.layers, list):
        bad_layers = _out_of_range_layers(parsed.layers, entry.n_layers)
        if bad_layers:
            return await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/harvest",
                t0,
                400,
                "invalid_request",
                f"layers {bad_layers} out of range for {model_id!r}: valid decoder "
                f"layer indices are 0-{entry.n_layers - 1}",
            )

    if len(parsed.prompts) > settings.harvest_max_prompts:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            400,
            "invalid_request",
            f"too many prompts ({len(parsed.prompts)}); max {settings.harvest_max_prompts} "
            "per job — split the corpus into multiple jobs",
        )
    est_tokens = parsed.estimated_tokens
    if est_tokens > settings.harvest_max_est_tokens:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            400,
            "invalid_request",
            f"estimated {est_tokens} total tokens (chars/4) exceeds the per-job cap "
            f"of {settings.harvest_max_est_tokens} — split the corpus into multiple jobs",
        )

    # ---- Atomic quota gate + job insert (single transaction) ----------------
    # The FOR UPDATE row lock on the caller's api_keys row serializes this
    # key's submits: a concurrent POST blocks here until this transaction
    # commits, then counts the pending row we insert below — closing the
    # count-then-insert race. Different keys don't contend.
    await session.execute(
        select(ApiKeyRow.id).where(ApiKeyRow.id == caller.key_id).with_for_update()
    )
    # Lazy janitor: age out provably-dead pending/running rows so a stale row
    # can never wedge this key's concurrency cap (no background worker).
    expired = await request_log_svc.expire_stale_harvest_jobs(session, caller.key_id)
    if expired:
        log.info("harvest_jobs_expired_stale", key_id=str(caller.key_id), n=expired)
    if caller.monthly_harvest_budget:
        used = await request_log_svc.count_harvest_jobs_this_month(session, caller.key_id)
        if used >= caller.monthly_harvest_budget:
            response = await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/harvest",
                t0,
                429,
                "harvest_quota_exceeded",
                f"monthly harvest-job quota reached ({used}/{caller.monthly_harvest_budget}); "
                "it resets at the start of next month",
            )
            await session.commit()  # release the row lock promptly
            return response
    # Per-key concurrency, in two LANES (ACS-321). A single flat cap treated an
    # 8-prompt interactive job and a 4096-prompt corpus run as the same thing, so
    # the small one queued behind the big one and blocked a whole working
    # session — an availability problem dressed as fairness. Big-model jobs keep
    # the strict cap (they hold a whole multi-GPU container); small-model jobs
    # get their own, looser one. The lanes are counted independently, so a
    # running 405B job never blocks a cheap 8B submit, or vice versa.
    big_model_ids = {
        mid for mid, e in request.app.state.models.items() if _harvest_is_big_model(e)
    }
    if _harvest_is_big_model(entry):
        lane, lane_cap = "large-model", settings.harvest_max_running_per_key
        lane_filter = {"only_model_ids": big_model_ids}
    else:
        lane, lane_cap = "small-model", settings.harvest_max_running_per_key_small
        # Exclude rather than include: a job whose model has since left the
        # registry belongs to no lane, and it should hold a slot rather than
        # quietly become free capacity.
        lane_filter = {"exclude_model_ids": big_model_ids}
    running = await request_log_svc.count_running_harvest_jobs(
        session, caller.key_id, **lane_filter
    )
    if running >= lane_cap:
        response = await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            429,
            "harvest_concurrency_exceeded",
            f"this key already has {running} active {lane} harvest job(s) "
            f"(max {lane_cap}); poll GET /v1/harvest/<job_id> and resubmit when "
            "it finishes, or DELETE /v1/harvest/<job_id> to cancel it and free "
            "the slot now",
        )
        response.headers["Retry-After"] = str(HARVEST_RETRY_AFTER_S)
        await session.commit()  # release the row lock promptly
        return response

    # Global (cross-key) cap on concurrent BIG-MODEL harvests. The per-key cap
    # above bounds one key, but 10 different keys each spawning an 8×H200 405B
    # harvest = 80 H200 at once (~$360/hr, and more than Modal will allocate).
    # Bound the multi-GPU fleet across all keys; cheap 1×L40S (8B) harvests stay
    # uncapped. Best-effort — the FOR UPDATE lock is per-caller-key, so distinct
    # keys can race the count (overshoot ≈ the number of keys racing the gate at
    # once, not a hard bound): enough to stop a stampede, not a hard admission
    # queue. Reject (retryable) rather than queue,
    # so no job waits past HARVEST_STALE_AFTER and gets false-failed.
    if _harvest_is_big_model(entry):
        big_running = await request_log_svc.count_running_big_model_harvest_jobs(
            session, big_model_ids
        )
        if big_running >= settings.harvest_max_running_big_model:
            response = await _error(
                session,
                request_id,
                caller,
                ip,
                "/v1/harvest",
                t0,
                429,
                "harvest_capacity_exceeded",
                f"the GPU cluster is at capacity for large-model harvests "
                f"({big_running} running, max {settings.harvest_max_running_big_model}); "
                "retry shortly — a transient capacity limit, not a quota",
            )
            response.headers["Retry-After"] = str(HARVEST_RETRY_AFTER_S)
            await session.commit()  # release the row lock promptly
            return response

    run_id = f"hv-{uuid.uuid4().hex[:12]}"
    job_id = str(uuid.uuid4())
    session.add(
        HarvestJob(
            id=job_id,
            key_id=caller.key_id,
            user_id=caller.user_id,
            model_id=model_id,
            run_id=run_id,
            status="pending",
            # Metadata only — prompt COUNT and sizes, never prompt text (the
            # api_requests privacy stance carries over to harvest_jobs).
            params={
                "n_prompts": len(parsed.prompts),
                "est_tokens": est_tokens,
                "layers": parsed.layers,
                "shard_size": parsed.shard_size,
                "batch_size": parsed.batch_size,
                "add_special_tokens": parsed.add_special_tokens,
            },
        )
    )
    # Commit = pending row visible to concurrent submits + row lock released +
    # no DB connection/transaction held across the Modal await below.
    await session.commit()

    # Forward add_special_tokens to spawn ONLY when the caller explicitly set it
    # (ACS-319): existing traffic then sends the exact kwargs today's deployed
    # harvest apps accept, and the Modal harvest() function picks the parameter up
    # on its next redeploy — so this change is safe to ship before those redeploys.
    spawn_kwargs: dict[str, object] = {
        "app_name": harvest_app,
        "model_id": model_id,
        "prompts": parsed.prompts,
        "layer_indices": parsed.layers,
        "shard_size": parsed.shard_size,
        "batch_size": parsed.batch_size,
        "run_id": run_id,
        "project_onto": parsed.project_onto,
    }
    if "add_special_tokens" in parsed.model_fields_set:
        spawn_kwargs["add_special_tokens"] = parsed.add_special_tokens

    try:
        call_id = await modal_ops.spawn_harvest(**spawn_kwargs)
    except Exception as exc:
        # Not-deployed app, missing Modal creds, or a control-plane failure.
        # The pending row is marked failed (client-safe class name only; full
        # detail in logs) — it stays pollable and, having no modal_call_id,
        # never burns monthly quota. Surface a retryable 503.
        log.warning(
            "harvest_spawn_failed",
            request_id=request_id,
            job_id=job_id,
            model=model_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        now = dt.datetime.now(tz=dt.UTC)
        await session.execute(
            update(HarvestJob)
            .where(HarvestJob.id == job_id, HarvestJob.status == "pending")
            .values(
                status="failed",
                error=f"harvest spawn failed ({type(exc).__name__}); see server logs",
                completed_at=now,
                updated_at=now,
            )
        )
        response = await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest",
            t0,
            503,
            "harvest_unavailable",
            f"bulk harvest is not currently available for model {model_id!r} "
            "(harvest backend not deployed or unreachable); the job was recorded "
            f"as failed (job_id {job_id!r}) and does not count against your quota",
        )
        await session.commit()
        return response

    await session.execute(
        update(HarvestJob)
        # Guard on 'pending' (matching the spawn-failure path above): the row is
        # always pending here in the happy path, but if a concurrent DELETE
        # cancelled it during the spawn await we must NOT resurrect it to
        # 'running' with a live call_id the owner believes is cancelled. Today
        # the job_id isn't observable until the 202 below, so this can't race —
        # the guard keeps it safe if a "list my jobs" surface is ever added.
        .where(HarvestJob.id == job_id, HarvestJob.status == "pending")
        .values(
            status="running",
            modal_call_id=call_id,
            updated_at=dt.datetime.now(tz=dt.UTC),
        )
    )
    # One api_requests row per submit for cost attribution / abuse detection —
    # record_request is endpoint-generic, so the harvest submit reuses it
    # (token counts stay NULL; harvest_jobs carries the job-level detail).
    await _record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint="/v1/harvest",
        model=model_id,
        n_prompt=None,
        n_completion=None,
        status_code=202,
        latency_ms=int((time.monotonic() - t0) * 1000),
        upstream_latency_ms=None,
        error_kind=None,
    )
    await session.commit()
    log.info(
        "harvest_job_started",
        request_id=request_id,
        job_id=job_id,
        model=model_id,
        run_id=run_id,
        n_prompts=len(parsed.prompts),
        est_tokens=est_tokens,
    )
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "running",
            "model": model_id,
            "n_prompts": len(parsed.prompts),
        },
    )


@router.get("/v1/harvest/{job_id}")
@limiter.limit(COMPLETIONS_RATE_LIMIT)
async def harvest_status(
    job_id: str,
    request: Request,
    wait: int = Query(
        0,
        description=(
            "Seconds to hold the request open waiting for the job to finish "
            "(long-poll), 0..60. 0 (the default) returns the current state "
            "immediately."
        ),
    ),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Poll one harvest job; on completion returns the harvest result dict.

    ``?wait=<seconds>`` long-polls (ACS-321): the request is held until the job
    reaches a terminal state or the budget runs out, then returns exactly the
    same payload an immediate poll would. It saves the client a polling loop; it
    is not a different response shape, and it is never required. **The DB
    connection is released between reconcile attempts** — the pool is small
    (SQLAlchemy's default 5+10), so a handful of clients holding connections for
    the whole wait would starve every other request in the app, login pages
    included.

    Lazy reconciliation: a ``running`` row triggers one non-blocking Modal poll
    (``FunctionCall.get(timeout=0)`` via modal_ops) and persists the outcome —
    no background worker to operate. The read transaction commits BEFORE the
    Modal await (no idle-in-transaction connection), and the reconcile UPDATE
    is guarded ``WHERE status = 'running'`` so a racing poll can never
    overwrite a committed terminal state. Any unexpected poll error serves the
    stored row as-is rather than 500ing. Terminal rows are served straight
    from Postgres. Only the owning key or another key of the same user may
    read a job; anyone else gets the same 404 as a nonexistent id.
    """
    t0 = time.monotonic()
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None
    caller = await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    await session.commit()

    # Range-checked here rather than with Query(ge=, le=): FastAPI's own
    # parameter validation raises RequestValidationError, which bypasses this
    # API's error envelope and would make /v1/harvest the one endpoint that
    # answers a bad request with a bare `{"detail": [...]}` 422.
    if not 0 <= wait <= HARVEST_MAX_WAIT_S:
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest/{job_id}",
            t0,
            400,
            "invalid_request",
            f"wait must be between 0 and {HARVEST_MAX_WAIT_S} seconds; got {wait}",
        )

    job = (
        await session.execute(select(HarvestJob).where(HarvestJob.id == job_id))
    ).scalar_one_or_none()
    if job is None or (job.key_id != caller.key_id and job.user_id != caller.user_id):
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest/{job_id}",
            t0,
            404,
            "harvest_job_not_found",
            f"no harvest job {job_id!r} for this key",
        )
    # Release the read transaction before any Modal round trip (attributes stay
    # loaded: expire_on_commit=False).
    await session.commit()

    # Admission control for WAITING requests (see the ceilings above). Counted
    # after auth and ownership, so an unauthenticated or wrong-key request can
    # never occupy a slot. The dict is only touched from the event-loop thread
    # between awaits, so no lock is needed; the check-and-increment has no await
    # between its halves.
    declined = False
    requested_wait = wait
    if wait > 0:
        in_key = _LONGPOLL_INFLIGHT.get(caller.key_id, 0)
        in_total = sum(_LONGPOLL_INFLIGHT.values())
        if (
            in_key >= settings.harvest_max_longpoll_per_key
            or in_total >= settings.harvest_max_longpoll_total
        ):
            log.info(
                "harvest_longpoll_declined",
                request_id=request_id,
                job_id=job.id,
                in_flight_key=in_key,
                in_flight_total=in_total,
            )
            wait, declined = 0, True
        else:
            _LONGPOLL_INFLIGHT[caller.key_id] = in_key + 1

    try:
        job = await _reconcile_harvest_job_until(
            session, job, job_id, time.monotonic() + wait, request_id, request
        )
    finally:
        # `wait` is 0 on the declined path, which never incremented — so this
        # correctly skips a decrement it did not earn.
        if wait > 0:
            remaining = _LONGPOLL_INFLIGHT.get(caller.key_id, 1) - 1
            if remaining > 0:
                _LONGPOLL_INFLIGHT[caller.key_id] = remaining
            else:
                _LONGPOLL_INFLIGHT.pop(caller.key_id, None)

    headers = None
    if declined:
        # Pace the caller (see the note above) — but NEVER hold the request
        # longer than the wait they asked for. Someone using `?wait=1` as a
        # cheap "give it a beat", with a 2s client timeout, would otherwise
        # start timing out exactly when the service is busiest.
        pace = min(HARVEST_WAIT_POLL_INTERVAL_S, float(requested_wait))
        headers = {
            "X-Acs-Longpoll": "declined",
            "Retry-After": str(max(1, int(pace))),
        }
        await session.close()
        await asyncio.sleep(pace)
    return JSONResponse(status_code=200, content=_harvest_job_payload(job), headers=headers)


async def _reconcile_harvest_job_until(
    session: AsyncSession,
    job: Any,
    job_id: str,
    deadline: float,
    request_id: str | None,
    request: Request,
) -> Any:
    """Reconcile a harvest job with Modal until terminal, disconnected, or out of budget.

    Extracted so the admission accounting above stays readable; the body is the
    original loop unchanged, and with ``deadline`` already past (the ``wait=0``
    default) it runs exactly one reconcile.
    """
    while True:
        now = dt.datetime.now(tz=dt.UTC)
        if (
            job.status in ("pending", "running")
            and job.created_at < now - request_log_svc.HARVEST_STALE_AFTER
        ):
            # Provably dead (Modal hard-kills at the function timeout) — same lazy
            # age-out the submit path applies, scoped to this key.
            await request_log_svc.expire_stale_harvest_jobs(session, job.key_id)
            await session.commit()
        elif job.status == "running" and job.modal_call_id:
            try:
                status_now, payload = await modal_ops.poll_harvest(job.modal_call_id)
            except Exception as exc:
                # poll_harvest already classifies expected failures; anything that
                # still escapes is a wrapper bug or truly unexpected infra state —
                # serve the stored row rather than 500, and leave the job running.
                log.warning(
                    "harvest_poll_unexpected_error",
                    request_id=request_id,
                    job_id=job.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                status_now, payload = "running", None
            if status_now == "done":
                await session.execute(
                    update(HarvestJob)
                    # Guard: only a still-running row may be finalized — a racing
                    # GET that already committed a terminal state wins.
                    .where(HarvestJob.id == job.id, HarvestJob.status == "running")
                    .values(
                        status="done",
                        result=payload if isinstance(payload, dict) else {"value": payload},
                        completed_at=now,
                        updated_at=now,
                    )
                )
                await session.commit()
            elif status_now == "failed":
                await session.execute(
                    update(HarvestJob)
                    .where(HarvestJob.id == job.id, HarvestJob.status == "running")
                    .values(
                        status="failed",
                        error=str(payload),  # already client-safe (modal_ops classifies)
                        completed_at=now,
                        updated_at=now,
                    )
                )
                await session.commit()
            # "running" → nothing persisted.

        # Re-read: serves whatever state actually won (this poll's update, a
        # racing poll's earlier terminal write, or the unchanged row).
        job = (await session.execute(select(HarvestJob).where(HarvestJob.id == job_id))).scalar_one()
        if job.status not in ("pending", "running"):
            break
        if time.monotonic() >= deadline or await request.is_disconnected():
            # Out of budget, or the client hung up — either way stop burning
            # Modal polls on a request nobody is waiting for.
            break
        # Hand the pooled connection back for the sleep. Holding it would let
        # a few long-polls exhaust a 5+10 pool and stall the whole app.
        await session.close()
        await asyncio.sleep(
            min(HARVEST_WAIT_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
        )
    return job


@router.delete("/v1/harvest/{job_id}")
@limiter.limit(COMPLETIONS_RATE_LIMIT)
async def harvest_cancel(
    job_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Cancel one of the caller's in-flight harvest jobs (ACS-344).

    The one-job-per-key concurrency cap means a stuck or slow job blocks the key
    for its entire wall clock (a tester saw 44 consecutive 429s over 915s with
    no way out). This gives the owner an exit: mark the job ``cancelled`` — a
    terminal state that no longer occupies a concurrency slot, so the very next
    submit succeeds — and best-effort ask Modal to stop the container so the GPU
    is freed too.

    Ownership + lifecycle:
      * only the owning key (or another key of the same user) may cancel; anyone
        else gets the same 404 as a nonexistent id, so job ids don't leak
        existence (mirrors ``harvest_status``);
      * a job already in a terminal state (done / failed / cancelled) → 409
        ``harvest_not_cancellable`` — there is nothing to cancel;
      * the cancel UPDATE is guarded ``WHERE status IN ('pending','running')``;
        if a concurrent poll finalized the job between the read and the write
        (rowcount 0) the committed terminal state wins and we return 409, never
        clobbering a ``done`` result with ``cancelled``.

    The DB write commits BEFORE the Modal round trip: the slot is freed durably
    even if the (best-effort, never-raising) container-terminate RPC is slow or
    fails. On such a failure the job is still ``cancelled`` and the slot free;
    only the Modal container may keep running until it finishes or hits its
    function timeout. Returns 200 with the cancelled job payload.
    """
    t0 = time.monotonic()
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None
    caller = await authmod.authenticate(
        request, session, settings.last_used_throttle_s, log_ip=settings.log_ip
    )
    await session.commit()

    job = (
        await session.execute(select(HarvestJob).where(HarvestJob.id == job_id))
    ).scalar_one_or_none()
    if job is None or (job.key_id != caller.key_id and job.user_id != caller.user_id):
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest/{job_id}",
            t0,
            404,
            "harvest_job_not_found",
            f"no harvest job {job_id!r} for this key",
        )
    if job.status not in ("pending", "running"):
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest/{job_id}",
            t0,
            409,
            "harvest_not_cancellable",
            f"harvest job {job_id!r} is already {job.status} and cannot be cancelled",
        )

    now = dt.datetime.now(tz=dt.UTC)
    call_id = job.modal_call_id
    result = await session.execute(
        update(HarvestJob)
        # Only a still-live row may be cancelled — a racing poll that already
        # committed done/failed between our read and here wins (rowcount 0).
        .where(HarvestJob.id == job_id, HarvestJob.status.in_(("pending", "running")))
        .values(
            status="cancelled",
            error="cancelled by the key owner via DELETE /v1/harvest/<job_id>",
            completed_at=now,
            updated_at=now,
        )
    )
    if result.rowcount == 0:
        # Lost the race to a poll finalization — serve the terminal state as 409.
        return await _error(
            session,
            request_id,
            caller,
            ip,
            "/v1/harvest/{job_id}",
            t0,
            409,
            "harvest_not_cancellable",
            f"harvest job {job_id!r} reached a terminal state before it could be "
            "cancelled",
        )
    # Commit = slot freed durably + no DB connection held across the Modal await.
    await session.commit()

    # Best-effort: tell Modal to terminate the container so the GPU frees now,
    # not at the function timeout. Never raises — the DB is already authoritative.
    modal_cancelled = False
    if call_id:
        modal_cancelled = await modal_ops.cancel_harvest(call_id)

    job = (
        await session.execute(select(HarvestJob).where(HarvestJob.id == job_id))
    ).scalar_one()
    await _record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint="/v1/harvest/{job_id}",
        model=job.model_id,
        n_prompt=None,
        n_completion=None,
        status_code=200,
        latency_ms=int((time.monotonic() - t0) * 1000),
        upstream_latency_ms=None,
        error_kind=None,
    )
    log.info(
        "harvest_job_cancelled",
        request_id=request_id,
        job_id=job_id,
        model=job.model_id,
        had_modal_call=bool(call_id),
        modal_cancelled=modal_cancelled,
    )
    return JSONResponse(status_code=200, content=_harvest_job_payload(job))


async def _completion_json_response(
    status_code: int,
    payload: dict[str, Any],
    extras: dict[str, Any],
    *,
    accept_gzip: bool = False,
) -> Response:
    response = await _render_completion_json(status_code, payload, accept_gzip=accept_gzip)
    _apply_extras_headers(response, extras)
    if status_code == 503:
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            retry_after = err.get("retry_after_seconds")
            if isinstance(retry_after, int) and retry_after > 0:
                response.headers["Retry-After"] = str(retry_after)
    return response


async def _serve_nonstream_with_keepalive(
    *,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    upstream_url: str,
    body: dict[str, Any],
    settings: Settings,
    session: AsyncSession,
    http: httpx.AsyncClient,
    model_id: str,
    request: Request,
    key_semaphore: asyncio.Semaphore,
    backend_ctx: proxymod.BackendContext,
    breakers: breakermod.BackendBreakers,
    effective_timeout: float,
    tokenizer_repo: str,
    telemetry: request_log_svc.RequestTelemetry | None = None,
    accept_gzip: bool = False,
    breaker_key: str | None = None,
):
    """Return a real-status JSON response quickly, otherwise stream heartbeats.

    Once a whitespace byte is sent the HTTP status is necessarily committed to
    200. Late failures therefore use the normal OpenAI-shaped JSON error body;
    ``run_completion_nonstream`` still records the true logical status.
    """
    # Authentication may have updated last_used_at. Close that transaction
    # before the potentially 14-minute upstream wait starts.
    await session.commit()
    shared_extras: dict[str, Any] = {}
    task = asyncio.create_task(
        run_completion_nonstream(
            session=session,
            http=http,
            settings=settings,
            caller=caller,
            body=body,
            ip=ip,
            endpoint="/v1/completions",
            request_id=request_id,
            upstream_url=upstream_url,
            tokenizer_repo=tokenizer_repo,
            model_id=model_id,
            request=request,
            backend_ctx=backend_ctx,
            breakers=breakers,
            timeout_s=effective_timeout,
            telemetry=telemetry,
            transport_extras=shared_extras,
            breaker_key=breaker_key,
        )
    )
    ownership_transferred = False

    async def gen():
        try:
            # Commit body bytes immediately; subsequent ticks keep every proxy
            # in the Railway → client chain from treating the request as idle.
            yield JSON_KEEPALIVE_CHUNK
            while True:
                done, _ = await asyncio.wait({task}, timeout=PUBLIC_KEEPALIVE_INTERVAL_S)
                if done:
                    _status_code, payload, _extras = await task
                    yield json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    return
                if await request.is_disconnected():
                    return
                yield JSON_KEEPALIVE_CHUNK
        finally:
            cancelled_child = False
            try:
                if not task.done():
                    cancelled_child = True
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                if cancelled_child:
                    await session.rollback()
            finally:
                key_semaphore.release()

    try:
        done, _ = await asyncio.wait({task}, timeout=PUBLIC_RESPONSE_GRACE_S)
        if done:
            status_code, payload, extras = await task
            return await _completion_json_response(
                status_code, payload, extras, accept_gzip=accept_gzip
            )

        response = StreamingResponse(
            gen(),
            status_code=200,
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
        _apply_extras_headers(response, shared_extras)
        ownership_transferred = True
        return response
    finally:
        # If the handler is cancelled during the grace period, the response
        # generator never takes ownership. Cancel and roll back here so the
        # task cannot outlive its request-scoped AsyncSession.
        if not ownership_transferred and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await session.rollback()


def _passthrough_usage_mode(parsed: CompletionsRequest) -> str | None:
    """Which unbuffered pass-through window a request qualifies for, or None.

    Big bodies stream through as bytes (ACS-198 / ACS-250) instead of being
    materialized in wrapper RAM, but ``usage`` sits in a different spot per shape,
    so the pass-through must be told which window to keep:

      - ``"head"`` — activation **capture** (``output_residual_stream``): usage
        precedes the multi-MB all-layers ``activations`` blob.
      - ``"tail"`` — full-vocab ``prompt_logprobs=-1``: usage follows the
        tens-of-MB logprobs.
      - ``None`` — everything else stays on the buffered path, INCLUDING:
        * **steering-only** (``apply_steering_vectors`` without capture): a
          normal-sized completion with usage in the tail and no blob to stream —
          buffering keeps its usage exact (a head window would under-bill a long
          steered generation, whose usage is nowhere near the head).
        * the full-vocab **+** capture combo: usage is sandwiched between two huge
          blobs, so no fixed head/tail window reliably catches it.
    """
    fullvocab = parsed.prompt_logprobs == FULL_VOCAB_SENTINEL
    capture = bool(parsed.output_residual_stream)
    if capture and not fullvocab:
        return "head"
    if fullvocab and not parsed.has_activation_params:
        return "tail"
    return None


async def _serve_fullvocab_passthrough(
    *,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    upstream_url: str,
    body: dict[str, Any],
    settings: Settings,
    session: AsyncSession,
    http: httpx.AsyncClient,
    t0: float,
    model_id: str,
    request: Request,
    key_semaphore: KeyConcurrencyGate,
    backend_ctx: proxymod.BackendContext,
    breakers: breakermod.BackendBreakers,
    effective_timeout: float,
    tokenizer_repo: str | None,
    telemetry: request_log_svc.RequestTelemetry | None = None,
    accept_gzip: bool = False,
    breaker_key: str | None = None,
    usage_at_head: bool = False,
) -> Response:
    """Stream a big completion body through unbuffered (ACS-198 / ACS-250).

    Two body shapes qualify, differing only in where the streamable ``usage``
    block sits — selected by ``usage_at_head``: full-vocab ``prompt_logprobs=-1``
    (ACS-198) keeps usage in the TAIL, after the tens-of-MB logprobs; activation
    capture (``output_residual_stream``, ACS-250) puts usage in the HEAD, before
    the ~MB all-layers ``activations`` blob. Everything else — byte streaming,
    gzip-on-the-fly, keepalive/cold-boot contract, per-key slot ownership — is
    identical for both. Original ACS-198 rationale:

    The buffered path (``run_completion_nonstream`` → ``proxy.post_nonstream``)
    materializes the whole upstream body in wrapper RAM — ~22 MB of parsed dict
    per prompt position at a 128k vocab — which forced the old 16-token
    full-vocab prompt cap. Here the body is passed through as bytes
    (``proxy.passthrough_post``), gzipped on the fly when the client accepts
    it, so wrapper memory stays O(chunk) regardless of prompt length.

    Response-shape contract:
      - First upstream event within ``PUBLIC_RESPONSE_GRACE_S`` → the response
        carries the real status (200 streams the body; upstream 4xx/cold-boot/
        unreachable return the same JSON errors as the buffered path).
      - Slower (long prefill + serialization, or a cold boot) → commit 200 and
        emit JSON-whitespace keepalives until upstream bytes arrive — the same
        contract as ``_serve_nonstream_with_keepalive``: a late failure becomes
        an OpenAI-shaped JSON error body while ``api_requests`` records the true
        logical status.

    ``usage`` accounting: the body is never parsed, so token counts are
    plucked from the final ``PASSTHROUGH_USAGE_TAIL_BYTES`` of the stream
    after it completes; an unparseable tail logs and skips the commit rather
    than failing the response.
    """
    extras: dict[str, Any] = {}
    bkey = breaker_key or model_id
    # Same api_requests fallback as run_completion_nonstream: the wrapper-side
    # short id when we have one, else whatever the body carried.
    model_name = model_id or body.get("model")
    endpoint = "/v1/completions"

    budget_err = await _budget_preflight_clamp(
        session=session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        t0=t0,
        body=body,
        tokenizer_repo=tokenizer_repo,
        settings=settings,
        telemetry=telemetry,
        extras=extras,
    )
    if budget_err is not None:
        status_code, payload = budget_err
        return await _completion_json_response(
            status_code, payload, extras, accept_gzip=accept_gzip
        )

    # Circuit breaker pre-check (same logic as run_completion_nonstream).
    if breakers is not None and bkey and not await breakers.allow(bkey):
        status_snap = breakers.snapshot(bkey)
        payload = breakermod.circuit_open_payload(status_snap)
        extras["upstream_error_kind"] = "circuit_open"
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=503,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind="circuit_open",
            telemetry=telemetry,
        )
        # Via _completion_json_response so the 503 carries Retry-After (from
        # the payload's retry_after_seconds) exactly like the buffered path.
        return await _completion_json_response(503, payload, extras, accept_gzip=accept_gzip)

    upstream_iter = proxymod.passthrough_post(
        http,
        upstream_url,
        settings.vllm_api_key,
        body,
        effective_timeout,
        ctx=backend_ctx,
    )
    # Wall clock to the first upstream body byte — recorded as both
    # upstream_latency_ms and ttft_ms (the buffered path's upstream_ms measures
    # to the *whole* body; first-byte is the honest equivalent here).
    ttfb_holder: dict[str, int | None] = {"ms": None}

    enc = zlib.compressobj(1, zlib.DEFLATED, 16 + zlib.MAX_WBITS) if accept_gzip else None

    def _enc_tick(data: bytes) -> bytes:
        """Encode a keepalive tick, sync-flushed so the bytes hit the wire now.

        Only ticks flush: they are tiny and exist purely to keep intermediaries
        from timing the connection out, so they must not sit in zlib's buffer.
        Body chunks skip the flush (see ``_enc_body``) — flushing every body
        chunk resets the deflate block and costs a few percent of ratio across
        a multi-GB stream for no benefit once bytes are flowing anyway.
        """
        if enc is None:
            return data
        return enc.compress(data) + enc.flush(zlib.Z_SYNC_FLUSH)

    async def _enc_body(data: bytes) -> bytes:
        """Encode a body chunk off the event loop (zlib releases the GIL).

        At the 1024-token cap a body is multi-GB; compressing it inline would
        occupy the single event loop for tens of seconds in aggregate and
        starve every other in-flight request — the same reason the buffered
        path runs its serialize+gzip in ``asyncio.to_thread``.
        """
        if enc is None:
            return data
        return await asyncio.to_thread(enc.compress, data)

    def _enc_final(data: bytes) -> bytes:
        """Encode the stream's last bytes, closing the gzip container."""
        if enc is None:
            return data
        return enc.compress(data) + enc.flush()

    async def _record_upstream_error(exc: Exception) -> tuple[int, dict[str, Any]]:
        """Map a proxy error to (status, payload); record + breaker bookkeeping.

        Mirrors run_completion_nonstream's except blocks so the pass-through
        path degrades to byte-identical error responses.
        """
        if isinstance(exc, proxymod.ColdBootError):
            await _record_request(
                session,
                request_id=request_id,
                caller=caller,
                ip=ip,
                endpoint=endpoint,
                model=model_name,
                n_prompt=None,
                n_completion=None,
                status_code=503,
                latency_ms=int((time.monotonic() - t0) * 1000),
                upstream_latency_ms=None,
                error_kind="cold_boot",
                telemetry=telemetry,
                cold_boot=True,
            )
            return 503, proxymod.cold_boot_error_payload(
                exc.upstream_status, retry_after_s=proxymod.COLD_BOOT_BACKOFF_S
            )
        if isinstance(exc, proxymod.UpstreamServerError):
            kind = exc.upstream_kind or "upstream_5xx"
            payload = proxymod.upstream_server_error_payload(exc)
        else:  # UpstreamUnreachable (incl. the defensive empty-body case)
            kind = "upstream_unreachable"
            payload = proxymod.upstream_unreachable_payload(exc)
        if breakers is not None and bkey:
            await breakers.record_failure(bkey, kind)
        extras["upstream_error_kind"] = kind
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=502,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind=kind,
            telemetry=telemetry,
        )
        return 502, payload

    async def _record_4xx(chunk: bytes, status_code: int) -> dict[str, Any]:
        try:
            payload = json.loads(chunk)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": chunk.decode("utf-8", "replace")}
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=status_code,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=ttfb_holder["ms"],
            error_kind="upstream_4xx",
            telemetry=telemetry,
        )
        return payload

    async def _record_aborted(exc: Exception) -> None:
        """Mid-body upstream failure: the client sees a truncated stream, but
        the row + breaker bookkeeping must not be lost (the buffered path
        recorded every upstream error)."""
        if breakers is not None and bkey:
            await breakers.record_failure(bkey, "passthrough_aborted")
        log.warning(
            "fullvocab_passthrough_aborted",
            request_id=request_id,
            key_id=str(caller.key_id),
            error=f"{type(exc).__name__}: {exc}",
        )
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=None,
            n_completion=None,
            status_code=502,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=ttfb_holder["ms"],
            error_kind="passthrough_aborted",
            telemetry=telemetry,
        )

    async def _finish_success(tail: bytes) -> None:
        usage = proxymod.extract_usage_from_json_tail(tail)
        if usage is None:
            # Never let a tail-parse miss turn multi-GB GPU work into
            # unmetered usage: floor with a wrapper-side prompt count
            # (completion side is at most max_tokens ≈ 1 for this shape).
            try:
                counter = get_token_counter(
                    tokenizer_repo or settings.served_model_name, settings.hf_token
                )
                n_prompt = completion_svc.count_prompt_tokens(body.get("prompt") or "", counter)
            except Exception:
                n_prompt = 0
            n_completion = 0
            log.warning(
                "fullvocab_usage_parse_failed",
                request_id=request_id,
                key_id=str(caller.key_id),
                fallback_prompt_tokens=n_prompt,
            )
        else:
            n_prompt = usage["prompt_tokens"]
            n_completion = usage["completion_tokens"]
        if n_prompt + n_completion > 0:
            try:
                await authmod.commit_usage(session, caller.key_id, n_prompt, n_completion)
            except Exception:
                log.warning("usage_commit_failed", key_id=str(caller.key_id))
        _mark_model_warm(request, bkey)
        if breakers is not None and bkey:
            await breakers.record_success(bkey)
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint=endpoint,
            model=model_name,
            n_prompt=n_prompt,
            n_completion=n_completion,
            status_code=200,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=ttfb_holder["ms"],
            error_kind=None,
            telemetry=telemetry,
            ttft_ms=ttfb_holder["ms"],
            cold_boot=bool(backend_ctx.cold_hint),
        )

    def _keep_tail(tail: bytearray, chunk: bytes) -> None:
        """Retain the last PASSTHROUGH_USAGE_TAIL_BYTES with O(tail) copying."""
        limit = proxymod.PASSTHROUGH_USAGE_TAIL_BYTES
        if len(chunk) >= limit:
            tail[:] = chunk[-limit:]
            return
        tail.extend(chunk)
        if len(tail) > limit:
            del tail[: len(tail) - limit]

    def _keep_head(head: bytearray, chunk: bytes) -> None:
        """Retain the FIRST PASSTHROUGH_USAGE_TAIL_BYTES (ACS-250: activation
        bodies put ``usage`` before the big ``activations`` blob, so the usable
        window is the head). Stops copying once full — O(head)."""
        limit = proxymod.PASSTHROUGH_USAGE_TAIL_BYTES
        if len(head) < limit:
            head.extend(chunk[: limit - len(head)])

    async def _stream_body(first_chunk: bytes):
        """Yield encoded body bytes, keep the usage window, run success bookkeeping."""
        tail = bytearray()
        _keep = _keep_head if usage_at_head else _keep_tail
        try:
            _keep(tail, first_chunk)
            out = await _enc_body(first_chunk)
            if out:
                yield out
            while True:
                try:
                    chunk, _status = await upstream_iter.__anext__()
                except StopAsyncIteration:
                    break
                _keep(tail, chunk)
                out = await _enc_body(chunk)
                if out:
                    yield out
            if enc is not None:
                final = enc.flush()
                if final:
                    yield final
        except Exception as exc:
            # CancelledError/GeneratorExit (client disconnect) are
            # BaseException and skip this: only genuine upstream/encoding
            # failures get recorded as aborted.
            await _record_aborted(exc)
            raise
        await _finish_success(bytes(tail))

    async def _cleanup(*, cancelled: bool, pending: asyncio.Task | None = None) -> None:
        try:
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await upstream_iter.aclose()
            except Exception:
                log.warning("passthrough_close_failed", request_id=request_id)
            if cancelled:
                await session.rollback()
        finally:
            key_semaphore.release()

    async def gen_direct(first_chunk: bytes):
        cancelled = False
        try:
            async for out in _stream_body(first_chunk):
                yield out
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            await _cleanup(cancelled=cancelled)

    async def gen_keepalive(first_task: asyncio.Task):
        """Committed-200 variant: whitespace until upstream bytes, then the body."""
        pending: asyncio.Task | None = first_task
        cancelled = False
        try:
            # Commit body bytes immediately; subsequent ticks keep every proxy
            # in the Railway → client chain from treating the request as idle.
            yield _enc_tick(JSON_KEEPALIVE_CHUNK)
            while True:
                done, _ = await asyncio.wait({pending}, timeout=PUBLIC_KEEPALIVE_INTERVAL_S)
                if done:
                    break
                if await request.is_disconnected():
                    return
                yield _enc_tick(JSON_KEEPALIVE_CHUNK)
            try:
                first_chunk, status_code = await pending
                pending = None
            except StopAsyncIteration:
                pending = None
                _status, payload = await _record_upstream_error(
                    proxymod.UpstreamUnreachable(
                        reason="upstream body ended before any bytes",
                        attempts=1,
                        ctx=backend_ctx,
                    )
                )
                yield _enc_final(_json_error_bytes(payload))
                return
            except (
                proxymod.ColdBootError,
                proxymod.UpstreamUnreachable,
                proxymod.UpstreamServerError,
            ) as exc:
                pending = None
                _status, payload = await _record_upstream_error(exc)
                yield _enc_final(_json_error_bytes(payload))
                return
            ttfb_holder["ms"] = int((time.monotonic() - t0) * 1000)
            if status_code is not None and status_code >= 400:
                # Status is committed to 200; forward the upstream's JSON error
                # body while api_requests records the true logical status.
                await _record_4xx(first_chunk, status_code)
                yield _enc_final(first_chunk)
                return
            async for out in _stream_body(first_chunk):
                yield out
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            await _cleanup(cancelled=cancelled, pending=pending)

    # Close the auth transaction before a potentially minutes-long upstream
    # wait (same rationale as _serve_stream's docstring).
    await session.commit()

    first_task = asyncio.create_task(upstream_iter.__anext__())
    ownership_transferred = False
    try:
        done, _ = await asyncio.wait({first_task}, timeout=PUBLIC_RESPONSE_GRACE_S)
        if done:
            try:
                first_chunk, status_code = await first_task
            except StopAsyncIteration:
                status, payload = await _record_upstream_error(
                    proxymod.UpstreamUnreachable(
                        reason="upstream body ended before any bytes",
                        attempts=1,
                        ctx=backend_ctx,
                    )
                )
                return await _completion_json_response(
                    status, payload, extras, accept_gzip=accept_gzip
                )
            except (
                proxymod.ColdBootError,
                proxymod.UpstreamUnreachable,
                proxymod.UpstreamServerError,
            ) as exc:
                status, payload = await _record_upstream_error(exc)
                return await _completion_json_response(
                    status, payload, extras, accept_gzip=accept_gzip
                )
            ttfb_holder["ms"] = int((time.monotonic() - t0) * 1000)
            if status_code is not None and status_code >= 400:
                payload = await _record_4xx(first_chunk, status_code)
                return await _completion_json_response(
                    status_code, payload, extras, accept_gzip=accept_gzip
                )
            response = StreamingResponse(
                gen_direct(first_chunk),
                status_code=200,
                media_type="application/json",
                headers=_passthrough_headers(gzipped=enc is not None),
            )
        else:
            response = StreamingResponse(
                gen_keepalive(first_task),
                status_code=200,
                media_type="application/json",
                headers=_passthrough_headers(gzipped=enc is not None),
            )
        _apply_extras_headers(response, extras)
        ownership_transferred = True
        return response
    finally:
        if not ownership_transferred:
            # Handler cancelled (or an error response path already recorded):
            # the generator never took ownership of the task/iterator/slot.
            # Error paths return real Response objects above — for those the
            # route's own finally releases the semaphore, so only unwind the
            # upstream side here.
            if not first_task.done():
                first_task.cancel()
                try:
                    await first_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await upstream_iter.aclose()
            except Exception:
                log.warning("passthrough_close_failed", request_id=request_id)


def _passthrough_headers(*, gzipped: bool) -> dict[str, str]:
    headers = {
        "Vary": "Accept-Encoding",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    if gzipped:
        headers["Content-Encoding"] = "gzip"
    return headers


def _json_error_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


async def _serve_stream(
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    upstream_url: str,
    body: dict[str, Any],
    settings: Settings,
    session: AsyncSession,
    http: httpx.AsyncClient,
    t0: float,
    model_name: str | None,
    request: Request | None = None,
    key_semaphore: KeyConcurrencyGate | None = None,
    backend_ctx: proxymod.BackendContext | None = None,
    breakers: breakermod.BackendBreakers | None = None,
    effective_timeout: float | None = None,
    telemetry: request_log_svc.RequestTelemetry | None = None,
    breaker_key: str | None = None,
):
    """Stream upstream SSE → client, capture `usage` for budget commit.

    Session lifecycle: ``authenticate()`` opened a Postgres transaction (SELECT
    + optional ``UPDATE api_keys.last_used_at``). We ``commit`` it before
    returning ``StreamingResponse`` so the connection sits idle — not "idle in
    transaction" — for the duration of the stream. Without this commit, a
    cold-boot retry that legitimately runs for minutes (workbench retry budget
    is effectively unbounded) would hold the auth tx + any row locks for the
    whole stream, blocking vacuum on ``api_keys`` and starving the pool.

    The captured ``session`` still lives past this function's return — FastAPI
    keeps ``yield``-style dependencies alive until the generator completes —
    so the post-stream usage commit happens on the same session, in a fresh
    transaction opened implicitly by the next ``execute``.
    """
    captured: dict[str, int] = {}
    status_holder: dict[str, int] = {"code": 200}
    # Time-to-first-token: wall clock from request arrival (t0) to the first SSE
    # chunk. Set once the first chunk is in hand; stays None if the stream
    # errors/cold-boots before yielding anything.
    ttft_holder: dict[str, int | None] = {"ms": None}
    backend_ctx = backend_ctx or proxymod.BackendContext()
    timeout_s = effective_timeout or settings.upstream_timeout_s
    # Breaker + warm-state key: the activation path passes a distinct key so its
    # health is tracked apart from the workbench engine (ACS-199). ``model_name``
    # stays the real id for api_requests logging.
    bkey = breaker_key or model_name

    # Circuit breaker pre-check (same logic as run_completion_nonstream).
    # Done BEFORE constructing StreamingResponse so we can still return a
    # 503 JSONResponse on a tripped breaker.
    if breakers is not None and bkey:
        if not await breakers.allow(bkey):
            status_snap = breakers.snapshot(bkey)
            payload = breakermod.circuit_open_payload(status_snap)
            await _record_request(
                session,
                request_id=request_id,
                caller=caller,
                ip=ip,
                endpoint="/v1/completions",
                model=model_name,
                n_prompt=None,
                n_completion=None,
                status_code=503,
                latency_ms=int((time.monotonic() - t0) * 1000),
                upstream_latency_ms=None,
                error_kind="circuit_open",
                telemetry=telemetry,
            )
            return JSONResponse(status_code=503, content=payload)

    upstream_iter = proxymod.stream_post(
        http, upstream_url, settings.vllm_api_key, body, timeout_s, ctx=backend_ctx
    )
    logical_error_kind: dict[str, str | None] = {"value": None}
    cold_boot_holder = {"seen": False}

    async def gen():
        pending: asyncio.Task | None = None
        cancelled_response = False
        # True while an in-place progress bar (CR-terminated comment,
        # ACS-280) is on the client's current terminal line and needs a
        # newline handoff before any real frame is written.
        bar_active = False
        try:
            while True:
                pending = asyncio.create_task(upstream_iter.__anext__())
                while True:
                    done, _ = await asyncio.wait({pending}, timeout=PUBLIC_KEEPALIVE_INTERVAL_S)
                    if done:
                        break
                    if request is not None and await request.is_disconnected():
                        return
                    # SSE comments are ignored by EventSource and OpenAI SDKs,
                    # so raw-stream users (`curl -N`) can watch progress while
                    # SDK users see nothing unusual. Before the first token of
                    # a believed-cold model, carry the container-reported boot
                    # stage (ACS-272); afterwards, plain keepalives.
                    if (
                        ttft_holder["ms"] is None
                        and backend_ctx.cold_hint
                        and backend_ctx.modal_app_name
                    ):
                        try:
                            st = await boot_stage_mod.for_cold_wait(
                                backend_ctx.modal_app_name
                            )
                            frame = boot_stage_mod.sse_comment(
                                st, int(time.monotonic() - t0)
                            )
                            # Flag BEFORE the yield: if a cancel lands inside
                            # the yield, the goodbye path still errs safe
                            # (at worst a harmless spurious handoff newline).
                            bar_active = True
                            yield frame
                        except Exception:  # noqa: BLE001 - never break keepalive
                            if bar_active:
                                # Terminate the dangling bar line so the plain
                                # keepalive doesn't overdraw it (review #284).
                                yield b"\n"
                                bar_active = False
                            yield b": keepalive\n\n"
                    else:
                        yield b": keepalive\n\n"

                if bar_active:
                    # Newline handoff: terminate the bar's comment line (SSE)
                    # and drop the cursor below the finished bar (terminal)
                    # before whatever comes next — token or error frame.
                    yield b"\n"
                    bar_active = False

                try:
                    chunk, usage, status_code = await pending
                except StopAsyncIteration:
                    if ttft_holder["ms"] is None:
                        # Upstream ended without one byte and without a typed
                        # error. stream_post now raises for this shape, but
                        # never regress to a silent EOF — hand the client a
                        # structured, retryable frame (ACS-277).
                        status_holder["code"] = 502
                        logical_error_kind["value"] = "upstream_unreachable"
                        yield _sse_json_frame(
                            proxymod.upstream_unreachable_payload(
                                proxymod.UpstreamUnreachable(
                                    reason="upstream body ended before any bytes",
                                    attempts=1,
                                    ctx=backend_ctx,
                                )
                            )
                        )
                    break
                except proxymod.ColdBootError as exc:
                    status_holder["code"] = 503
                    logical_error_kind["value"] = "cold_boot"
                    cold_boot_holder["seen"] = True
                    payload = proxymod.cold_boot_error_payload(
                        exc.upstream_status,
                        retry_after_s=proxymod.COLD_BOOT_BACKOFF_S,
                    )
                    yield _sse_json_frame(payload)
                    break
                except proxymod.UpstreamUnreachable as exc:
                    status_holder["code"] = 502
                    logical_error_kind["value"] = "upstream_unreachable"
                    yield _sse_json_frame(proxymod.upstream_unreachable_payload(exc))
                    break
                except proxymod.UpstreamServerError as exc:
                    # Client-ish upstream errors (e.g. an out-of-range steering
                    # layer_index reflected as a 500) are the caller's fault: log
                    # 400 (not 502) so they stay out of the real_5xx alarm, and
                    # _finish then spares the breaker (code < 400/500). Mirrors
                    # the non-stream handler (ACS-322). ``logical_error_kind``
                    # keeps the precise kind for the recorded row.
                    status_holder["code"] = (
                        400 if exc.upstream_kind in proxymod.CLIENT_ISH_UPSTREAM_KINDS else 502
                    )
                    logical_error_kind["value"] = exc.upstream_kind or "upstream_5xx"
                    yield _sse_json_frame(proxymod.upstream_server_error_payload(exc))
                    break

                if ttft_holder["ms"] is None:
                    ttft_holder["ms"] = int((time.monotonic() - t0) * 1000)
                if status_code is not None:
                    status_holder["code"] = status_code
                    if status_code >= 400:
                        logical_error_kind["value"] = (
                            "upstream_5xx" if status_code >= 500 else "upstream_4xx"
                        )
                if usage:
                    captured.update(usage)
                if status_holder["code"] >= 400:
                    try:
                        parsed = json.loads(chunk)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        parsed = {
                            "error": {
                                "code": logical_error_kind["value"] or "upstream_error",
                                "message": chunk.decode("utf-8", "replace")[:512],
                                "type": "upstream_error",
                            }
                        }
                    yield _sse_json_frame(parsed)
                    break
                yield chunk

            await _finish()
        except asyncio.CancelledError:
            cancelled_response = True
            # Shutdown drain (a Railway deploy replacing the wrapper) severs
            # in-flight streams — the ACS-277 incident showed the client a
            # silent EOF mid-cold-boot. Best-effort: hand over one terminal,
            # retryable frame. If the transport is already gone the yield
            # never completes or re-cancels; either way, re-raise.
            try:
                if bar_active:
                    # A bar comment is dangling on the wire; without this
                    # handoff, naive newline-only parsers would glue the
                    # goodbye frame into the comment and swallow it —
                    # regressing the ACS-277 guarantee in exactly the
                    # deploy-drain-during-cold-boot scenario (review #284).
                    yield b"\n"
                yield _sse_json_frame(proxymod.server_restarting_payload())
            except Exception:  # noqa: BLE001 - transport mid-teardown
                # Exception (not BaseException): GeneratorExit/CancelledError
                # on the goodbye yield must propagate naturally rather than
                # be caught here and re-masked by the bare raise below.
                pass
            raise
        finally:
            try:
                if pending is not None and not pending.done():
                    pending.cancel()
                    try:
                        await pending
                    except asyncio.CancelledError:
                        pass
                try:
                    await upstream_iter.aclose()
                except Exception:
                    log.warning("upstream_stream_close_failed", request_id=request_id)
                if cancelled_response:
                    await session.rollback()
            finally:
                # Release the per-key in-flight slot even if cleanup fails.
                if key_semaphore is not None:
                    key_semaphore.release()

    async def _finish():
        n_prompt = captured.get("prompt_tokens", 0)
        n_completion = captured.get("completion_tokens", 0)
        if n_prompt + n_completion > 0:
            try:
                await authmod.commit_usage(session, caller.key_id, n_prompt, n_completion)
                # session_scope() commits on exit; no explicit commit needed here.
            except Exception:
                log.warning("usage_commit_failed", key_id=str(caller.key_id))
        if status_holder["code"] < 400 and request is not None:
            _mark_model_warm(request, bkey)
        # Split the streaming error taxonomy so /admin and logs can tell
        # client errors apart from upstream server errors. Mirrors the
        # non-streaming path's ``upstream_4xx`` / ``upstream_5xx`` split.
        code = status_holder["code"]
        if logical_error_kind["value"] is not None:
            stream_error_kind = logical_error_kind["value"]
        elif code < 400:
            stream_error_kind = None
        elif code >= 500:
            stream_error_kind = "upstream_5xx"
        else:
            stream_error_kind = "upstream_4xx"
        # Breaker bookkeeping: streaming success closes the breaker; mid-stream
        # 5xx counts as a failure (the upstream did answer something, but
        # something bad). 4xx is the caller's fault — don't trip the breaker
        # on bad client requests.
        if breakers is not None and bkey:
            if code < 400:
                await breakers.record_success(bkey)
            elif code >= 500 and stream_error_kind != "cold_boot":
                await breakers.record_failure(
                    bkey,
                    stream_error_kind or "upstream_5xx",
                )
        await _record_request(
            session,
            request_id=request_id,
            caller=caller,
            ip=ip,
            endpoint="/v1/completions",
            model=model_name,
            n_prompt=n_prompt,
            n_completion=n_completion,
            status_code=code,
            latency_ms=int((time.monotonic() - t0) * 1000),
            upstream_latency_ms=None,
            error_kind=stream_error_kind,
            telemetry=telemetry,
            ttft_ms=ttft_holder["ms"],
            cold_boot=(cold_boot_holder["seen"] or bool(backend_ctx.cold_hint and code < 400)),
        )

    # Close the auth tx before the long-lived stream begins. See docstring.
    await session.commit()
    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


def _sse_json_frame(payload: dict[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
    )


async def _error(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller | None,
    ip: str | None,
    endpoint: str,
    t0: float,
    code: int,
    error_kind: str,
    message: str,
) -> JSONResponse:
    await _record_request(
        session,
        request_id=request_id,
        caller=caller,
        ip=ip,
        endpoint=endpoint,
        model=None,
        n_prompt=None,
        n_completion=None,
        status_code=code,
        latency_ms=int((time.monotonic() - t0) * 1000),
        upstream_latency_ms=None,
        error_kind=error_kind,
    )
    return JSONResponse(
        status_code=code,
        content={
            "error": {"message": message, "type": "invalid_request_error", "code": error_kind},
            "request_id": request_id,
        },
    )
