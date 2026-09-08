"""Per-model availability over time (ACS-51).

Computes uptime %, downtime windows, and MTTR for each served model from the
``api_requests`` table alone. There's no separate breaker-event log: every
breaker trip that mattered to a real caller already shows up as a row with
``error_kind = "circuit_open"``, and a trip that mattered to no one is by
definition not downtime worth reporting.

Formula (spelled out so a reviewer doesn't have to reverse-engineer it):

  denominator = rows where:
    cold_boot = false
    AND status NOT IN (401, 403, 429)   # caller errors + rate-limit aren't outages

  successes  = rows in denominator where:
    status < 400
    AND error_kind IS NULL

  uptime_pct = successes / denominator  (None if denominator == 0)

  downtime_windows = contiguous runs of rows where ``error_kind = 'circuit_open'``,
    grouped per-model, ordered by ts. A run is (start_ts, end_ts, duration_s)
    where end_ts is the timestamp of the first non-``circuit_open`` row after
    the run. A run still open at the end of the lookback window is reported
    with end_ts = None (its duration counts up to "now").

  MTTR = mean(end_ts - start_ts) across only the *closed* windows in the period.
    Open windows (no recovery observed yet) are excluded from the average so
    one stuck outage doesn't poison the mean — they're still reported in
    ``downtime_windows`` for visibility.

Why ``circuit_open`` not just ``status >= 500``? Because plain 5xx rows are
already counted as failures in the uptime ratio. Downtime *windows* are about
"the breaker was open and short-circuiting" — a distinct operational state
worth surfacing separately, because it tells the operator the wrapper made a
correct load-shedding decision (not that a one-off 500 slipped through).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ApiRequest

# Status codes that are caller-side or protective, not service outages.
_NON_OUTAGE_STATUSES = (401, 403, 429)


@dataclass(frozen=True)
class DowntimeWindow:
    """One contiguous run of ``circuit_open`` rows for a model."""

    start_ts: dt.datetime
    end_ts: dt.datetime | None  # None ⇒ still open at end of lookback period
    duration_s: float


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    window_start: dt.datetime
    window_end: dt.datetime
    total_requests: int  # raw row count, all statuses (for context)
    denominator: int  # excludes cold-boot + caller errors
    successes: int
    uptime_pct: float | None  # None when denominator == 0
    downtime_windows: list[DowntimeWindow]
    mttr_s: float | None  # mean of *closed* window durations; None if no closed windows


async def compute_availability(
    session: AsyncSession,
    *,
    model_id: str,
    window: dt.timedelta,
    now: dt.datetime | None = None,
) -> ModelAvailability:
    """Compute availability for one model over ``[now - window, now]``.

    ``now`` is injectable for tests; defaults to ``datetime.now(UTC)``.
    """
    end = now or dt.datetime.now(tz=dt.UTC)
    start = end - window

    # One aggregate query: total, denominator, successes. PostgreSQL handles
    # the FILTER clauses cheaply; this is one round-trip per model.
    is_in_denom = (
        (ApiRequest.cold_boot.is_(False))
        & (~ApiRequest.status.in_(_NON_OUTAGE_STATUSES))
    )
    is_success = is_in_denom & (ApiRequest.status < 400) & (ApiRequest.error_kind.is_(None))

    agg = (
        await session.execute(
            select(
                func.count().label("total"),
                func.count().filter(is_in_denom).label("denominator"),
                func.count().filter(is_success).label("successes"),
            ).where(
                ApiRequest.model == model_id,
                ApiRequest.ts >= start,
                ApiRequest.ts < end,
            )
        )
    ).one()

    total = int(agg.total or 0)
    denominator = int(agg.denominator or 0)
    successes = int(agg.successes or 0)
    uptime_pct = (successes / denominator) if denominator > 0 else None

    # Pull the ts + error_kind sequence for this model, ordered, to derive
    # contiguous ``circuit_open`` runs. We don't pull bodies or anything else.
    rows = list(
        (
            await session.execute(
                select(ApiRequest.ts, ApiRequest.error_kind)
                .where(
                    ApiRequest.model == model_id,
                    ApiRequest.ts >= start,
                    ApiRequest.ts < end,
                )
                .order_by(ApiRequest.ts.asc())
            )
        ).all()
    )

    downtime_windows = _derive_circuit_open_windows(rows, end_of_period=end)
    closed = [w for w in downtime_windows if w.end_ts is not None]
    mttr_s = (sum(w.duration_s for w in closed) / len(closed)) if closed else None

    return ModelAvailability(
        model_id=model_id,
        window_start=start,
        window_end=end,
        total_requests=total,
        denominator=denominator,
        successes=successes,
        uptime_pct=uptime_pct,
        downtime_windows=downtime_windows,
        mttr_s=mttr_s,
    )


def _derive_circuit_open_windows(
    rows: list[tuple[dt.datetime, str | None]],
    *,
    end_of_period: dt.datetime,
) -> list[DowntimeWindow]:
    """Walk the time-ordered (ts, error_kind) sequence and emit one window
    per contiguous run of ``circuit_open`` rows.

    A run ends at the first non-``circuit_open`` row after it. If the run
    extends to the end of the period (no recovery observed yet) the window
    is emitted with ``end_ts=None`` and duration measured up to
    ``end_of_period``.
    """
    windows: list[DowntimeWindow] = []
    run_start: dt.datetime | None = None
    for ts, kind in rows:
        if kind == "circuit_open":
            if run_start is None:
                run_start = ts
        else:
            if run_start is not None:
                windows.append(
                    DowntimeWindow(
                        start_ts=run_start,
                        end_ts=ts,
                        duration_s=(ts - run_start).total_seconds(),
                    )
                )
                run_start = None
    if run_start is not None:
        windows.append(
            DowntimeWindow(
                start_ts=run_start,
                end_ts=None,
                duration_s=(end_of_period - run_start).total_seconds(),
            )
        )
    return windows


async def list_models_with_recent_requests(
    session: AsyncSession,
    *,
    window: dt.timedelta,
    now: dt.datetime | None = None,
) -> list[str]:
    """Return distinct ``model`` values seen in the lookback window.

    Used to decide which rows to render on the admin page — a model with no
    traffic at all has no availability number worth showing.
    """
    end = now or dt.datetime.now(tz=dt.UTC)
    start = end - window
    rows = (
        await session.execute(
            select(ApiRequest.model)
            .where(
                ApiRequest.model.is_not(None),
                ApiRequest.ts >= start,
                ApiRequest.ts < end,
            )
            .distinct()
            .order_by(ApiRequest.model.asc())
        )
    ).all()
    return [r[0] for r in rows]
