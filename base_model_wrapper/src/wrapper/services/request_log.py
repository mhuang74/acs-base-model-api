"""API request logging and persistence."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import web_auth as webauth
from ..error_kinds import is_known
from ..logging import get_logger, log_request_meta
from ..models import ApiRequest, HarvestJob

_log = get_logger("request_log")

# ``api_requests.activation_layers`` value for a capture request. Capture is
# all-or-nothing (the engine returns every decoder layer), so we record -1 =
# "all layers" rather than a per-layer count. NULL = the request didn't capture.
ALL_LAYERS_SENTINEL = -1

if TYPE_CHECKING:
    from ..schemas import CompletionsRequest


@dataclass(slots=True)
class RequestTelemetry:
    """Request-shape metadata captured once per request and attached to its
    ``api_requests`` row(s).

    Pure metadata — no prompt / completion / logprobs content. ``stream`` and
    ``workload_type`` are known up-front; the sampling fields mirror the
    validated request. Built once (``from_completions_request``) and passed to
    every ``record_request`` call for that request, so all rows for a given
    logical request share the same shape, including late error rows.
    """

    stream: bool = False
    workload_type: str | None = None
    req_max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    logprobs_set: bool = False
    prompt_logprobs_set: bool = False
    seed_set: bool = False
    echo_set: bool = False
    # Activation harvesting / steering (ACS-199).
    activation: bool = False
    activation_layers: int | None = None
    activation_steering_vectors: int | None = None

    @classmethod
    def from_completions_request(
        cls,
        parsed: CompletionsRequest,
        *,
        stream: bool,
        workload_type: str | None,
    ) -> RequestTelemetry:
        return cls(
            stream=stream,
            workload_type=workload_type,
            req_max_tokens=parsed.max_tokens,
            temperature=parsed.temperature,
            top_p=parsed.top_p,
            logprobs_set=parsed.logprobs is not None,
            prompt_logprobs_set=parsed.prompt_logprobs is not None,
            seed_set=parsed.seed is not None,
            echo_set=bool(parsed.echo),
            activation=parsed.has_activation_params,
            # ``activation_layers``: ALL_LAYERS_SENTINEL (-1) when a request captures
            # every layer (``output_residual_stream: true``), the layer COUNT when it
            # captures a subset (``output_residual_stream: [..]``, ACS-266), else None.
            # The column is int|None (no per-layer list); the meaningful signals are
            # ``activation`` (did it capture/steer) and ``activation_steering_vectors``.
            activation_layers=(
                ALL_LAYERS_SENTINEL
                if parsed.output_residual_stream is True
                else (
                    len(parsed.output_residual_stream)
                    if isinstance(parsed.output_residual_stream, list)
                    else None
                )
            ),
            activation_steering_vectors=(
                len(parsed.apply_steering_vectors)
                if parsed.apply_steering_vectors is not None
                else None
            ),
        )


_EMPTY_TELEMETRY = RequestTelemetry()


async def record_request(
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
    telemetry: RequestTelemetry | None = None,
    cold_boot: bool = False,
    ttft_ms: int | None = None,
) -> None:
    """Persist one ``api_requests`` row and emit the privacy-safe stdout line.

    ``telemetry`` carries the request-shape fields (stream / workload /
    sampling); ``cold_boot`` and ``ttft_ms`` are per-outcome and supplied by the
    caller at record time. All are pure metadata — the privacy guarantee holds.
    """
    tel = telemetry or _EMPTY_TELEMETRY
    # Write-time guard: every producer should stamp a canonical ErrorKind. An
    # unknown value means a producer drifted from the enum (typically a typo);
    # warn loudly so it surfaces in logs instead of silently fragmenting a
    # dashboard grouping. We still persist the row — telemetry must never fail
    # an otherwise-valid request — but the warning is the signal to fix it.
    if not is_known(error_kind):
        _log.warning(
            "error_kind_not_in_enum",
            error_kind=error_kind,
            request_id=request_id,
            endpoint=endpoint,
        )
    log_request_meta(
        request_id=request_id,
        key_id=str(caller.key_id) if caller else None,
        key_prefix=caller.key_prefix if caller else None,
        user_email=caller.user_email if caller else None,
        ip=ip,
        endpoint=endpoint,
        model=model,
        n_prompt=n_prompt,
        n_completion=n_completion,
        status=status_code,
        latency_ms=latency_ms,
        upstream_latency_ms=upstream_latency_ms,
        error_kind=error_kind,
        stream=tel.stream,
        cold_boot=cold_boot,
        ttft_ms=ttft_ms,
        workload_type=tel.workload_type,
    )
    if caller is None:
        return
    session.add(
        ApiRequest(
            key_id=caller.key_id,
            # Postgres INET rejects non-IP strings (FastAPI TestClient sends
            # the literal 'testclient'; some reverse proxies in dev send
            # Unix-socket paths). Drop silently rather than crash on insert.
            ip=webauth._safe_ip(ip),
            endpoint=endpoint,
            model=model,
            n_prompt=n_prompt,
            n_completion=n_completion,
            status=status_code,
            latency_ms=latency_ms,
            error_kind=error_kind,
            upstream_latency_ms=upstream_latency_ms,
            ttft_ms=ttft_ms,
            stream=tel.stream,
            cold_boot=cold_boot,
            workload_type=tel.workload_type,
            req_max_tokens=tel.req_max_tokens,
            temperature=tel.temperature,
            top_p=tel.top_p,
            logprobs_set=tel.logprobs_set,
            prompt_logprobs_set=tel.prompt_logprobs_set,
            seed_set=tel.seed_set,
            echo_set=tel.echo_set,
            activation=tel.activation,
            activation_layers=tel.activation_layers,
            activation_steering_vectors=tel.activation_steering_vectors,
        )
    )


def _current_month_start() -> dt.datetime:
    """First instant of the current UTC month (budget windows are monthly)."""
    now = dt.datetime.now(dt.UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def count_activation_requests_this_month(
    session: AsyncSession, key_id: uuid.UUID
) -> int:
    """Count a key's activation requests since the start of the current UTC month.

    The tally the per-key monthly activation quota (ACS-199) checks against.
    Counts committed ``api_requests`` rows flagged ``activation`` — so it
    excludes the in-flight request being checked (an off-by-the-current-one that
    is the correct semantics for "how many have you already used").

    Scope of what counts: any request that *reached the activation engine* — its
    row is stamped ``activation=true`` before dispatch, so a request that woke the
    engine and then errored still counts (deliberate: the GPU cost was incurred).
    Only a request rejected *before* dispatch — the pre-flight quota 429, or the
    ``activations_unsupported`` 400 — is recorded ``activation=false`` and so does
    not count against the quota.
    """
    result = await session.execute(
        select(func.count())
        .select_from(ApiRequest)
        .where(
            ApiRequest.key_id == key_id,
            ApiRequest.activation.is_(True),
            ApiRequest.ts >= _current_month_start(),
        )
    )
    return int(result.scalar_one() or 0)


# A pending/running harvest_jobs row older than this is provably not running:
# the harvest function's own Modal timeout is 150 min (after which Modal
# hard-kills the container), so 180 min = timeout + generous slack. Such rows
# only exist when a terminal poll never happened (client stopped polling,
# wrapper died mid-spawn) and are lazily aged out to 'failed' so they can't
# deadlock the per-key concurrency cap forever.
HARVEST_STALE_AFTER = dt.timedelta(minutes=180)


async def count_harvest_jobs_this_month(session: AsyncSession, key_id: uuid.UUID) -> int:
    """Count a key's bulk-harvest jobs started since the start of the current UTC month.

    The tally the per-key monthly harvest quota (ACS-245,
    ``api_keys.monthly_harvest_budget``) checks against — the harvest sibling of
    ``count_activation_requests_this_month``, but counting ``harvest_jobs`` rows
    instead of ``api_requests`` rows. A started job counts regardless of how it
    ends (the GPU spawn happened — same "cost was incurred" semantics as the
    activation quota) with ONE exception: rows in a terminal state WITHOUT ever
    getting a Modal call id never engaged a GPU, so they don't burn quota. That
    covers a spawn error / harvest app not deployed (``failed``) and a
    cancellation of a still-``pending`` job before its spawn confirmed
    (``cancelled``, ACS-344); a cancellation AFTER the spawn keeps its call id
    and DOES count. The pre-flight 429 inserts no row at all.
    """
    result = await session.execute(
        select(func.count())
        .select_from(HarvestJob)
        .where(
            HarvestJob.key_id == key_id,
            HarvestJob.created_at >= _current_month_start(),
            ~(
                HarvestJob.status.in_(("failed", "cancelled"))
                & HarvestJob.modal_call_id.is_(None)
            ),
        )
    )
    return int(result.scalar_one() or 0)


async def count_running_harvest_jobs(
    session: AsyncSession,
    key_id: uuid.UUID,
    *,
    only_model_ids: set[str] | None = None,
    exclude_model_ids: set[str] | None = None,
) -> int:
    """Count a key's harvest jobs currently occupying a concurrency slot.

    Backs the per-key concurrency cap on POST /v1/harvest. Since ACS-321 that
    cap has two **lanes** — expensive multi-GPU models keep the strict cap while
    cheap ones allow several at once — so a caller passes either
    ``only_model_ids`` (count just these; the big-model lane) or
    ``exclude_model_ids`` (count everything else; the small-model lane). The
    small lane excludes rather than includes deliberately: a job whose model has
    since been retired from the registry is in neither id set, and it should
    consume a slot rather than silently become free capacity.

    With neither filter this counts every lane, which is the pre-ACS-321
    behaviour. ``pending`` counts too — a
    submit that has passed the gate but not yet confirmed its spawn holds a
    slot, which is what makes the gate race-free (the row is inserted inside
    the locked quota transaction). Reads the persisted status only — a job
    that finished on Modal but hasn't been polled yet still counts until a
    GET /v1/harvest/<id> reconciles it (deliberate: no Modal RPC on the submit
    path; clients are nudged to poll). Rows older than ``HARVEST_STALE_AFTER``
    are excluded — they are provably dead and get aged out by
    ``expire_stale_harvest_jobs``.
    """
    conditions = [
        HarvestJob.key_id == key_id,
        HarvestJob.status.in_(("pending", "running")),
        HarvestJob.created_at >= dt.datetime.now(dt.UTC) - HARVEST_STALE_AFTER,
    ]
    if only_model_ids is not None:
        # An empty set means "this lane has no models", which must count zero —
        # `IN ()` does exactly that, but be explicit rather than relying on it.
        if not only_model_ids:
            return 0
        conditions.append(HarvestJob.model_id.in_(only_model_ids))
    if exclude_model_ids:
        conditions.append(HarvestJob.model_id.notin_(exclude_model_ids))
    result = await session.execute(select(func.count()).select_from(HarvestJob).where(*conditions))
    return int(result.scalar_one() or 0)


async def count_running_big_model_harvest_jobs(
    session: AsyncSession, big_model_ids: set[str]
) -> int:
    """Count ALL keys' running/pending harvest jobs for the given big models.

    Backs the GLOBAL cap on POST /v1/harvest for multi-GPU models
    (``Settings.harvest_max_running_big_model``): the per-key cap can't stop N
    different keys stampeding 80×H200. Same ``pending``-counts + stale-window
    semantics as ``count_running_harvest_jobs``, but across every key and
    filtered to ``big_model_ids``. Best-effort: the submit path's FOR UPDATE
    lock is per-caller-key, so distinct keys can race past the gate (overshoot ≈
    the number of keys racing at once) — enough to bound a stampede, not a hard
    admission queue.
    """
    if not big_model_ids:
        return 0
    result = await session.execute(
        select(func.count())
        .select_from(HarvestJob)
        .where(
            HarvestJob.model_id.in_(big_model_ids),
            HarvestJob.status.in_(("pending", "running")),
            HarvestJob.created_at >= dt.datetime.now(dt.UTC) - HARVEST_STALE_AFTER,
        )
    )
    return int(result.scalar_one() or 0)


async def expire_stale_harvest_jobs(session: AsyncSession, key_id: uuid.UUID) -> int:
    """Mark a key's over-age ``pending``/``running`` harvest jobs ``failed``.

    Lazy janitor for the concurrency cap (no background worker): any
    pending/running row older than ``HARVEST_STALE_AFTER`` cannot still be
    executing (Modal hard-kills at the function timeout), so it's flipped to
    ``failed`` with a client-safe explanation. Runs inside the submit path's
    locked quota transaction and on GET for the affected job. Returns the
    number of rows aged out.
    """
    now = dt.datetime.now(dt.UTC)
    result = await session.execute(
        sa_update(HarvestJob)
        .where(
            HarvestJob.key_id == key_id,
            HarvestJob.status.in_(("pending", "running")),
            HarvestJob.created_at < now - HARVEST_STALE_AFTER,
        )
        .values(
            status="failed",
            error=(
                "job exceeded the maximum harvest runtime and was marked failed "
                "by the wrapper; if it completed on Modal the result was never "
                "collected — resubmit"
            ),
            completed_at=now,
            updated_at=now,
        )
    )
    return int(result.rowcount or 0)
