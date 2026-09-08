"""Per-backend circuit breaker.

Tracks per-model upstream health in-memory on ``app.state.breakers``. When a
backend hits ``FAILURE_THRESHOLD`` consecutive failures, the breaker trips to
``OPEN`` and subsequent requests for that model short-circuit with a 503 +
``circuit_open`` error code for ``OPEN_DURATION_S`` seconds. After that the
state moves to ``HALF_OPEN``, where the next request is allowed through as a
probe — success closes the breaker, failure re-opens it.

Why this instead of just letting Modal absorb the load? Because in practice
upstream failures cascade: a hung Modal app drains httpx's connection pool,
which starves *other* models. The breaker is a load-shedder; it pays for itself
the first time one of three backends melts down and the others keep serving.

Cold-boot is **not** counted as a failure — Modal is doing what it's supposed
to. Only ``UpstreamUnreachable`` and ``UpstreamServerError`` increment the
counter.

In-memory (rather than Postgres) on purpose: the breaker reflects *current*
reachability, not historical state, and the Railway deploy is single-replica.
Persisted state would just delay recovery on restart (better to assume healthy
and learn fast). The capacity-probe data in Postgres is the long-term audit
trail; the breaker is the runtime decision.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from .logging import get_logger

log = get_logger()

State = Literal["closed", "open", "half_open"]

# Tunables. Module-level so tests can monkey-patch quickly. Order of magnitude
# rather than precisely-tuned values — adjust once we have prod incident data.
FAILURE_THRESHOLD = 5
OPEN_DURATION_S = 60.0


def _alert_breaker_open(model_id: str, kind: str, consecutive_failures: int) -> None:
    """Log + page when a backend breaker trips to ``open`` (model degraded).

    Why this exists: ``/health`` returns HTTP 200 for a ``degraded`` (breaker-
    open) state — by design, so one sick model doesn't restart the whole
    wrapper — which means an HTTP-status uptime monitor never fires for it. This
    is the alerting path for "a model is failing for users" (they get 503
    ``circuit_open``). Grouped by a stable per-model Sentry fingerprint so a long
    outage (and the 60s half-open re-opens) collapse into one issue → one Discord
    ping, not a storm. Sentry capture is a no-op when no DSN is configured, and
    the whole thing is wrapped so alerting can never break request handling.
    """
    log.warning(
        "backend_breaker_open",
        model_id=model_id,
        last_failure_kind=kind or "unknown",
        consecutive_failures=consecutive_failures,
    )
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.fingerprint = ["backend-breaker-open", model_id]
            scope.set_tag("model_id", model_id)
            scope.set_extra("last_failure_kind", kind or "unknown")
            scope.set_extra("consecutive_failures", consecutive_failures)
            sentry_sdk.capture_message(
                f"Backend breaker OPEN for {model_id!r} — upstream unhealthy "
                f"({consecutive_failures} consecutive failures, last: "
                f"{kind or 'unknown'}); users get 503 circuit_open for this model.",
                level="warning",
            )
    except Exception as exc:  # noqa: BLE001 — alerting must never break request handling
        log.warning("backend_breaker_alert_failed", model_id=model_id, error=f"{type(exc).__name__}: {exc}")


@dataclass
class _ModelState:
    """Per-model breaker state. Mutated under ``BackendBreakers._lock``."""

    consecutive_failures: int = 0
    state: State = "closed"
    opened_at: float = 0.0
    # Most recent reason a failure was recorded (for /admin/breakers).
    last_failure_kind: str = ""
    # Total counters since process start — admin-only, for triage.
    total_failures: int = 0
    total_successes: int = 0
    total_trips: int = 0
    # Wall-clock of the most recent state transition; admin display.
    state_changed_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BreakerStatus:
    """Read-only snapshot of one model's breaker state.

    Returned from ``snapshot()`` for the admin view. Frozen so a caller can't
    accidentally mutate the live state through this object.
    """

    model_id: str
    state: State
    consecutive_failures: int
    opens_until: float | None  # epoch seconds when half-open eligible, or None
    last_failure_kind: str
    total_failures: int
    total_successes: int
    total_trips: int
    state_changed_at: float


class BackendBreakers:
    """Per-model circuit breaker registry, lives on ``app.state.breakers``.

    Thread-safe (well, asyncio-safe) via a single per-instance lock. The lock
    is held only during state transitions — checking ``allow()`` and recording
    outcomes are quick mutations. No I/O happens under the lock.
    """

    def __init__(self) -> None:
        self._states: dict[str, _ModelState] = {}
        self._lock = asyncio.Lock()

    def _get(self, model_id: str) -> _ModelState:
        st = self._states.get(model_id)
        if st is None:
            st = _ModelState()
            self._states[model_id] = st
        return st

    async def allow(self, model_id: str) -> bool:
        """Return True if the breaker permits a request for ``model_id`` right now.

        Closed → True (normal traffic).
        Open → True if the open window has elapsed (transition to half-open
        and let one probe through), else False.
        Half-open → True for exactly one request at a time; concurrent requests
        get False until that probe resolves. (Implemented by moving back to
        ``open`` on a half-open probe failure and to ``closed`` on success.)
        """
        async with self._lock:
            st = self._get(model_id)
            now = time.monotonic()
            if st.state == "closed":
                return True
            if st.state == "open":
                if now - st.opened_at >= OPEN_DURATION_S:
                    st.state = "half_open"
                    st.state_changed_at = time.time()
                    return True
                return False
            # half_open: only one probe in flight. We can't track "in flight"
            # without explicit ack from caller, so we just allow — concurrent
            # callers will all probe. Acceptable: half-open is brief, the
            # cost of N probes during a degraded period is fine.
            return True

    async def record_success(self, model_id: str) -> None:
        async with self._lock:
            st = self._get(model_id)
            st.consecutive_failures = 0
            st.total_successes += 1
            if st.state != "closed":
                st.state = "closed"
                st.state_changed_at = time.time()

    async def record_failure(self, model_id: str, kind: str) -> None:
        """Increment failure counter; trip to ``open`` at the threshold.

        ``kind`` is a short label (e.g. "upstream_unreachable",
        "upstream_5xx") surfaced in /admin/breakers for triage. Cold-boot is
        NOT a kind we accept here — callers must not record_failure on
        cold-boot.
        """
        tripped = False
        failures = 0
        async with self._lock:
            st = self._get(model_id)
            st.consecutive_failures += 1
            st.total_failures += 1
            st.last_failure_kind = kind
            if st.state == "half_open":
                # Half-open probe failed → re-open.
                st.state = "open"
                st.opened_at = time.monotonic()
                st.state_changed_at = time.time()
                st.total_trips += 1
                tripped = True
                failures = st.consecutive_failures
            elif st.consecutive_failures >= FAILURE_THRESHOLD and st.state == "closed":
                st.state = "open"
                st.opened_at = time.monotonic()
                st.state_changed_at = time.time()
                st.total_trips += 1
                tripped = True
                failures = st.consecutive_failures
        # Alert OUTSIDE the lock (no I/O while held). Grouped per-model so a long
        # outage's repeated re-opens don't spam Discord (see _alert_breaker_open).
        if tripped:
            _alert_breaker_open(model_id, kind, failures)

    async def reset(self, model_id: str) -> None:
        """Force-close a breaker (admin action). Counters preserved."""
        async with self._lock:
            st = self._get(model_id)
            st.consecutive_failures = 0
            st.state = "closed"
            st.state_changed_at = time.time()

    def snapshot(self, model_id: str) -> BreakerStatus:
        """Read-only snapshot for /admin/breakers. No locking — reads are
        idempotent and a stale value is fine for display."""
        st = self._get(model_id)
        opens_until = None
        if st.state == "open":
            # Convert monotonic to wall-clock for display.
            opens_until = time.time() + max(0.0, OPEN_DURATION_S - (time.monotonic() - st.opened_at))
        return BreakerStatus(
            model_id=model_id,
            state=st.state,
            consecutive_failures=st.consecutive_failures,
            opens_until=opens_until,
            last_failure_kind=st.last_failure_kind,
            total_failures=st.total_failures,
            total_successes=st.total_successes,
            total_trips=st.total_trips,
            state_changed_at=st.state_changed_at,
        )

    def all_snapshots(self) -> list[BreakerStatus]:
        return [self.snapshot(mid) for mid in sorted(self._states.keys())]


def circuit_open_payload(status: BreakerStatus) -> dict:
    """Structured 503 body for a request rejected because the breaker is open.

    Includes the model id and an approximate retry window so clients can
    decide whether to wait (short outage) or batch later (long outage).
    """
    retry_after_s = None
    if status.opens_until is not None:
        retry_after_s = max(1, int(status.opens_until - time.time()))
    return {
        "error": {
            "code": "circuit_open",
            "message": (
                f"Upstream backend for {status.model_id!r} is unhealthy "
                f"({status.consecutive_failures} consecutive failures, last "
                f"kind: {status.last_failure_kind or 'unknown'}). Try again later."
            ),
            "type": "upstream_error",
            "model_id": status.model_id,
            "consecutive_failures": status.consecutive_failures,
            "last_failure_kind": status.last_failure_kind,
            "retry_after_seconds": retry_after_s,
            "retryable": True,
        }
    }
