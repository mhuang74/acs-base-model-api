"""Thin wrapper around the Modal SDK.

Everything in the rest of the wrapper that needs to talk to Modal goes
through this module — keeps SDK-version drift (Modal 1.x renames things
between minor releases) contained to one place.

Modal SDK pin: 1.5.x (migrated from 1.4.3 in ACS-125). Two things differ
from older docs/snippets:

- ``modal.Function.lookup`` is gone; use ``modal.Function.from_name``.
- ``modal.App`` no longer exposes ``.stop()`` or a ``.state`` attribute.
  We mirror the CLI's lower-level RPC path (``AppGetByDeploymentName`` +
  ``AppStop``) via ``modal.Client`` for those.

Every internal/undocumented symbol this module touches was verified
byte-identical across 1.4.3, 1.5.0 and 1.5.1 (ACS-125): the stub RPCs
(``AppGetByDeploymentName``, ``AppStop``, ``FunctionGet``,
``FunctionGetCurrentStats``), the response fields (``lifecycle.app_state``,
``num_total_tasks``), the ``api_pb2`` app-state enums, ``_Client.from_env``,
and ``Function.from_name(...).update_autoscaler(min_containers=...)``. The
1.5.x read path (``_resolve_app`` / ``get_app_state`` /
``get_active_runner_count``) was run against the live workspace and returns
real states + runner counts. So the ACS-123 "float to 1.5" incident was NOT
an internal-symbol rename (all symbols match 1.4.3); its trigger was
environmental (see ACS-126, credential source-of-truth). The pin floors at
``1.5.1`` — the version verified live — not ``1.5.0`` (the version prod was
on during the ACS-123 incident, which could not be re-verified).

Sync vs. async split (Modal 1.5.x, unchanged from 1.4.3):

- ``modal.Client.from_env()`` is a synchronicity-wrapped sync entry point,
  but the stub it returns (``client.stub``) exposes RPC methods whose
  ``__call__`` is ``async def``. Calling them from an async context returns
  a coroutine that was never awaited (production crash: "'coroutine'
  object has no attribute 'app_id'"). To use the stub safely from inside
  the wrapper's async request handlers we therefore go through the
  underlying ``modal.client._Client`` (true async) and await each RPC.
- ``modal.Function.from_name(name, "serve").update_autoscaler(...)`` is
  nominally synchronicity-wrapped, but under the 1.5.x pin, called from an
  ``asyncio.to_thread`` worker it issues its ``FunctionGet`` /
  ``FunctionUpdateSchedulingParams`` RPCs "outside of task context" and
  hangs for minutes (ACS-202 — the /admin Release/Keep-warm buttons spun
  forever and returned 499). ``set_min_containers`` therefore drives the
  same two RPCs itself over the true-async ``_Client`` stub (the
  ``AppStop`` path pattern) and is awaited inline — no ``to_thread``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any

import grpclib.exceptions
import modal
from modal.client import _Client as _ModalAsyncClient
from modal_proto import api_pb2

from .logging import get_logger

log = get_logger("modal_ops")

SERVE_FUNCTION_NAME = "serve"

MODAL_SOURCE_DIR = "/app/modal_source"
MODAL_REQUIRED_SOURCE_FILES = (
    "modal_app.py",
    os.path.join("acs_model_registry", "__init__.py"),
    os.path.join("serving", "__init__.py"),
    os.path.join("serving", "modal_config.py"),
    os.path.join("serving", "modal_lifecycle.py"),
    os.path.join("serving", "modal_resources.py"),
    os.path.join("serving", "staging.py"),
    os.path.join("serving", "vllm_runtime.py"),
)
DEPLOY_TIMEOUT_S = 120


class ModalOpsError(RuntimeError):
    """Raised when a Modal API call fails or auth isn't configured."""


def _auth() -> None:
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        raise ModalOpsError("MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set in env")


_STATE_BY_PROTO = {
    api_pb2.APP_STATE_DEPLOYED: "deployed",
    api_pb2.APP_STATE_EPHEMERAL: "deployed",
    api_pb2.APP_STATE_DETACHED: "deployed",
    api_pb2.APP_STATE_DETACHED_DISCONNECTED: "deployed",
    api_pb2.APP_STATE_STOPPED: "stopped",
    api_pb2.APP_STATE_STOPPING: "stopped",
    api_pb2.APP_STATE_DISABLED: "stopped",
    api_pb2.APP_STATE_INITIALIZING: "initializing",
}


async def _resolve_app(app_name: str) -> tuple[str | None, int]:
    """Return ``(app_id, app_state)`` for a deployed-or-recently-stopped name.

    Returns ``(None, APP_STATE_UNSPECIFIED)`` when no app with the given name
    exists. Mirrors what ``modal app stop`` / ``modal app history`` do.
    """
    client = await _ModalAsyncClient.from_env()
    try:
        resp = await client.stub.AppGetByDeploymentName(
            api_pb2.AppGetByDeploymentNameRequest(name=app_name, environment_name="")
        )
    except modal.exception.NotFoundError:
        return None, api_pb2.APP_STATE_UNSPECIFIED
    if resp.app_id:
        return resp.app_id, resp.lifecycle.app_state
    if resp.previous_app_id:
        return resp.previous_app_id, resp.lifecycle.app_state
    return None, api_pb2.APP_STATE_UNSPECIFIED


async def stop_app(app_name: str) -> None:
    """Stop a deployed Modal app by name. Idempotent."""
    _auth()
    app_id, app_state = await _resolve_app(app_name)
    if app_id is None:
        return
    if app_state == api_pb2.APP_STATE_STOPPED:
        return
    client = await _ModalAsyncClient.from_env()
    await client.stub.AppStop(
        api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
    )
    # Defined below; clears both state + runners caches and their refresh markers.
    invalidate_state(app_name)


# Cap a single autoscaler write (FunctionGet + FunctionUpdateSchedulingParams).
# The Release / Keep-warm handlers and the warm-window cron await this inline,
# so an unbounded call hangs the HTTP request (browser spins → 499) or wedges a
# scheduler job. Modal's control plane has documented slow-spells (this module
# cites AppGetByDeploymentName seen hanging >2 min), so a bound is mandatory,
# not belt-and-braces. Matches the 30 s runner-fetch cap above.
_AUTOSCALER_UPDATE_TIMEOUT_S = 30


async def set_min_containers(app_name: str, n: int) -> None:
    """Override ``min_containers`` on the model's serve Function (runtime, not deploy-time).

    Drives ``FunctionGet`` → ``FunctionUpdateSchedulingParams`` over the
    true-async ``_Client`` stub (mirroring ``stop_app``'s ``AppStop`` path),
    rather than the synchronicity-wrapped ``modal.Function.update_autoscaler``.
    The latter, called from an ``asyncio.to_thread`` worker under the 1.5.x
    pin, issued its RPCs "outside of task context" and hung for minutes
    (ACS-202). This coroutine is awaited directly by callers — no ``to_thread``.

    Does NOT boot a stopped app — it only writes autoscaler config; the app
    stays in ``APP_STATE_STOPPED``. Callers that need a guard against silently
    no-opping on a stopped app should ``await assert_app_running(app_name)``
    first.
    """
    _auth()
    try:
        async with asyncio.timeout(_AUTOSCALER_UPDATE_TIMEOUT_S):
            client = await _ModalAsyncClient.from_env()
            get_resp = await client.stub.FunctionGet(
                api_pb2.FunctionGetRequest(
                    app_name=app_name,
                    object_tag=SERVE_FUNCTION_NAME,
                    environment_name="",
                )
            )
            await client.stub.FunctionUpdateSchedulingParams(
                api_pb2.FunctionUpdateSchedulingParamsRequest(
                    function_id=get_resp.function_id,
                    settings=api_pb2.AutoscalerSettings(min_containers=n),
                )
            )
    except modal.exception.NotFoundError as exc:
        raise ModalOpsError(f"Modal function {app_name}/{SERVE_FUNCTION_NAME} not found") from exc
    except TimeoutError as exc:
        raise ModalOpsError(
            f"Timed out after {_AUTOSCALER_UPDATE_TIMEOUT_S}s setting "
            f"min_containers={n} on {app_name} (Modal control-plane slow-spell)"
        ) from exc


# States where update_autoscaler / a fresh HTTP request will NOT boot a new
# container. Calling set_min_containers against an app in one of these states
# silently no-ops — the Modal SDK happily writes autoscaler config but the app
# stays stopped, so HTTP requests keep 404ing. Callers should ``await
# assert_app_running`` and surface the failure (admin redirect, warm_window
# log) rather than logging fake success.
_NOT_RUNNING_STATES = frozenset(
    {
        api_pb2.APP_STATE_STOPPED,
        api_pb2.APP_STATE_STOPPING,
        api_pb2.APP_STATE_DISABLED,
        api_pb2.APP_STATE_UNSPECIFIED,
    }
)


async def assert_app_running(app_name: str) -> None:
    """Raise ``ModalOpsError`` if the app exists but isn't in a runnable state.

    Separate from ``set_min_containers`` so the sync caller surface stays sync
    (no signature ripple through tests). Pair them at every call site that
    expects a stopped app to surface as an error: ``warm_window`` cron,
    ``/admin/models/<id>/keep-warm``, ``/admin/models/<id>/release``.
    """
    _auth()
    _, proto_state = await _resolve_app(app_name)
    if proto_state in _NOT_RUNNING_STATES:
        raise ModalOpsError(
            f"Modal app {app_name!r} is not running "
            f"(state={_STATE_BY_PROTO.get(proto_state, 'unspecified')}); "
            "update_autoscaler would no-op. Redeploy the app first."
        )


# Stale-while-revalidate (SWR) caches for the two Modal control-plane lookups
# that back every workbench render.
#
# Why SWR: a single Modal RPC was measured at ~18s on a slow control-plane
# day. With a plain TTL cache, the first read after expiry blocks the user
# for the full RPC time. SWR returns the cached value immediately even when
# stale, and schedules a background refresh so the next read sees fresh data.
#
# Cache-miss (no entry at all) still blocks — that path only fires once per
# app per wrapper lifetime, and is typically pre-warmed at startup by
# ``prewarm_caches`` so user-visible reads never hit it.
_STATE_CACHE_TTL_S = 300  # 5 min — state changes rarely (admin actions)
_RUNNERS_CACHE_TTL_S = 300  # 5 min — runner count is more volatile, but
# 5 min stale "warm pill" is acceptable;
# admin actions force invalidation.

# Cap a single get_current_stats RPC. Modal's control plane has slow-spells (an
# AppGetByDeploymentName probe was seen hanging >2 min); without a bound a slow
# runner read would hang the admin render. On timeout the read surfaces as
# "unknown" (None) rather than blocking — see get_active_runner_count.
# 30s, not 10s: this module's own notes cite ~18s Modal control-plane
# slow-spells, so a 10s cap converted slow-but-successful fetches into
# "unavailable" for every model under load. Failures are now logged (see the
# runner_count_fetch_failed events) so the true cause is visible in prod.
_RUNNER_FETCH_TIMEOUT_S = 30

_STATE_CACHE: dict[str, tuple[float, str]] = {}
# Value is None when the live count couldn't be determined (a failed/timed-out
# fetch with no prior good value). Callers MUST NOT treat that as a real 0 —
# rendering an unknown count as "scaled to zero" misreported warm always-on
# models as cold (ACS-128).
_RUNNERS_CACHE: dict[str, tuple[float, int | None]] = {}

# In-flight background refresh dedupe — one entry per app per cache. Avoids
# stampedes where N concurrent stale reads each kick off their own refresh.
_STATE_REFRESHING: set[str] = set()
_RUNNERS_REFRESHING: set[str] = set()


async def _fetch_state(app_name: str) -> str:
    """Single Modal RPC → resolved state string. No caching, no dedupe."""
    try:
        _, proto_state = await _resolve_app(app_name)
    except Exception:
        return "unknown"
    if proto_state == api_pb2.APP_STATE_UNSPECIFIED:
        return "stopped"
    return _STATE_BY_PROTO.get(proto_state, "unknown")


async def _refresh_state(app_name: str) -> None:
    """Background refresh task body for the state cache.

    Removes itself from ``_STATE_REFRESHING`` on completion. Only updates
    the cache on success — a failed fetch leaves the prior stale value in
    place rather than clobbering it with a transient error.
    """
    try:
        state = await _fetch_state(app_name)
        _STATE_CACHE[app_name] = (time.monotonic(), state)
    except Exception:
        # Defensive: _fetch_state already swallows everything, but a refresh
        # task that escapes here would be an unhandled task exception.
        pass
    finally:
        _STATE_REFRESHING.discard(app_name)


async def get_app_state(app_name: str) -> str:
    """Return one of: 'deployed', 'stopped', 'initializing', 'unknown'.

    SWR semantics:
    - Fresh hit: return cached value, no RPC.
    - Stale hit: return cached value immediately, schedule a background
      refresh (deduped via ``_STATE_REFRESHING``).
    - Miss: block on a real Modal RPC, populate cache, return.
    """
    _auth()
    now = time.monotonic()
    cached = _STATE_CACHE.get(app_name)
    if cached is not None:
        age = now - cached[0]
        if age < _STATE_CACHE_TTL_S:
            return cached[1]
        # Stale — return immediately, kick off background refresh (deduped).
        if app_name not in _STATE_REFRESHING:
            _STATE_REFRESHING.add(app_name)
            asyncio.create_task(_refresh_state(app_name))
        return cached[1]
    # Cold miss — block on the RPC. Pre-warming at lifespan start keeps this
    # off the user-visible path in normal operation.
    state = await _fetch_state(app_name)
    _STATE_CACHE[app_name] = (now, state)
    return state


def _clear_state_cache() -> None:
    """Reset both caches and both refreshing sets. Used by tests."""
    _STATE_CACHE.clear()
    _RUNNERS_CACHE.clear()
    _STATE_REFRESHING.clear()
    _RUNNERS_REFRESHING.clear()


def invalidate_state(app_name: str) -> None:
    """Drop cached state + runners for ``app_name`` so the next render hits Modal.

    Called from admin Stop / Deploy / Keep-warm / Release handlers; an
    in-flight refresh's late-arriving value should not clobber the
    freshly-invalidated cache, hence clearing the refreshing set too.
    """
    _STATE_CACHE.pop(app_name, None)
    _RUNNERS_CACHE.pop(app_name, None)
    _STATE_REFRESHING.discard(app_name)
    _RUNNERS_REFRESHING.discard(app_name)


# Separate cache for "is there a live container right now?" Reads
# Function.get_current_stats() — one RPC per model per TTL. ``deployed``
# alone isn't enough: Modal apps stay "deployed" forever after the first
# deploy, even with zero containers. ``num_total_runners > 0`` is the
# right signal for the workbench warm/cold pill.


async def _fetch_runner_count(
    app_name: str, function_name: str = SERVE_FUNCTION_NAME
) -> int:
    """Live runner count via the internal async client. Raises on failure/timeout.

    Uses ``_ModalAsyncClient`` + ``client.stub`` — the same path that powers the
    app-state lookup (``_resolve_app``) and works reliably in the deployed
    container. The high-level ``modal.Function.from_name(...).get_current_stats``
    API relies on an ambient client that is NOT established in the wrapper's
    background context: it logged "RPC made outside of task context" and hung
    until the timeout for *every* model in prod (while a laptop probe returned in
    ~0.3s), which silently zeroed the cost sampler and showed "count unavailable"
    on the admin page. The stub path resolves the function id (``FunctionGet``)
    then reads stats (``FunctionGetCurrentStats``); the runner count is the
    response's ``num_total_tasks`` (what the Modal SDK surfaces as
    ``num_total_runners``).

    Bounded by ``_RUNNER_FETCH_TIMEOUT_S``; callers decide what to do on failure:
    ``get_active_runner_count`` caches ``None`` (unknown, never a misleading 0,
    ACS-128); ``fetch_runner_count_fresh`` re-raises so the cost sampler skips.
    """

    async def _go() -> int:
        client = await _ModalAsyncClient.from_env()
        fn_resp = await client.stub.FunctionGet(
            api_pb2.FunctionGetRequest(
                app_name=app_name,
                object_tag=function_name,
                environment_name="",
            )
        )
        if not fn_resp.function_id:
            raise ModalOpsError(
                f"Modal function {app_name}/{function_name} not found"
            )
        stats = await client.stub.FunctionGetCurrentStats(
            api_pb2.FunctionGetCurrentStatsRequest(function_id=fn_resp.function_id)
        )
        return int(stats.num_total_tasks)

    return await asyncio.wait_for(_go(), timeout=_RUNNER_FETCH_TIMEOUT_S)


async def fetch_runner_count_fresh(app_name: str) -> int:
    """Uncached live runner count — one RPC, no stale-while-revalidate.

    ``get_active_runner_count`` serves a cached value and, on a control-plane
    failure, returns (and caches) 0. That's right for fast page renders but wrong
    for the cost sampler: a boot-time / transient 0 would be billed as "nothing
    running" even for an always-on model. This bypasses the cache to read the
    true current count, then writes the fresh value through so other readers
    (e.g. the workbench warm pill) benefit. Raises on auth/RPC failure — the
    caller decides whether to skip (the cost sampler does, rather than record a
    misleading 0).
    """
    _auth()
    count = await _fetch_runner_count(app_name)
    _RUNNERS_CACHE[app_name] = (time.monotonic(), count)
    return count


# Suffix that turns a serving model id into its activation-engine key — used
# for the breaker / warm-state keys (routes.api, ACS-199) and the cost-sample
# series id (cost_monitor, ACS-221). One constant so the runtime keys and the
# persisted gpu_cost_sample series can't drift.
ACTIVATION_KEY_SUFFIX = "::activation"


def activation_app_name(model_id: str) -> str:
    """Default Modal app name of the per-model ACTIVATION engine.

    Matches ``modal_app_activation.py``'s ``acs-{MODEL_ID}-activation`` when
    ``model_id`` is the shared-registry key the app was deployed under. Derived
    from the model id, NOT ``modal_app_name``: Trinity's serving app kept the
    pre-rename name (``acs-trinity-base``) while its activation app is keyed on
    the current id (``acs-trinity-truebase-activation``). Callers should prefer
    ``resolve_activation_app`` so an explicit registry override wins.
    """
    return f"acs-{model_id}-activation"


def resolve_activation_app(entry: object, model_id: str) -> str:
    """Activation app name for a registry entry: explicit ``activation_app_name``
    wins, else derived from the model id (mirrors ``harvest_app_name``)."""
    return getattr(entry, "activation_app_name", None) or activation_app_name(model_id)


# The activation app's web function is NOT named "serve": the plain multi-GPU
# path exposes ``serve_activation`` (@app.function), the single-GPU snapshot
# path exposes the class service function ``ActivationSnap.*`` (@app.cls).
# Probed live 2026-07-22: acs-llama-405b-activation / acs-trinity-truebase-
# activation resolve "serve_activation"; acs-llama-8b-activation resolves
# "ActivationSnap.*".
ACTIVATION_FUNCTION_TAGS = ("serve_activation", "ActivationSnap.*")

# Which tag resolved last time, per app — a deployed app's tag never changes
# (short of a redeploy onto the other lifecycle path, which the fallback loop
# still covers), so remembering the winner avoids a guaranteed-miss FunctionGet
# every tick for apps on the snapshot path (llama-8b).
_ACTIVATION_TAG_CACHE: dict[str, str] = {}


async def fetch_activation_runner_count_fresh(app_name: str) -> int:
    """Uncached live runner count for an ACTIVATION app (cost sampler, ACS-221).

    Same fresh/write-through semantics as ``fetch_runner_count_fresh``, but the
    web function's tag depends on which lifecycle path the app was deployed
    with, so this tries the known tags (cached winner first) and remembers
    which one resolved. If none does, raises ``ModalOpsError`` naming every
    tag's failure — a single tag's error would blame the wrong lifecycle path
    (e.g. a transient RPC error on the real tag hidden behind a clean
    "not found" from the other). The cost sampler skips on failure rather than
    record a misleading 0.
    """
    _auth()
    cached = _ACTIVATION_TAG_CACHE.get(app_name)
    tags = [cached, *(t for t in ACTIVATION_FUNCTION_TAGS if t != cached)] if cached else [
        *ACTIVATION_FUNCTION_TAGS
    ]
    failures: list[str] = []
    for tag in tags:
        try:
            count = await _fetch_runner_count(app_name, function_name=tag)
        except Exception as exc:  # noqa: BLE001 — try the next lifecycle path's tag
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
            continue
        _ACTIVATION_TAG_CACHE[app_name] = tag
        _RUNNERS_CACHE[app_name] = (time.monotonic(), count)
        return count
    raise ModalOpsError(
        f"no activation function reachable on {app_name} ({'; '.join(failures)})"
    )


# Suffix for the bulk-harvest engine's cost-sample series id (cost_monitor,
# ACS-281) — sibling of ACTIVATION_KEY_SUFFIX. Harvest has no breaker/warm-state
# key (jobs are spawned, not proxied), so this only names gpu_cost_sample rows.
HARVEST_KEY_SUFFIX = "::harvest"


async def fetch_harvest_runner_count_fresh(app_name: str) -> int:
    """Uncached live runner count for a bulk-HARVEST app (cost sampler, ACS-281).

    Same fresh/write-through semantics as ``fetch_runner_count_fresh``. No tag
    probing needed: ``harvest`` is the app's only GPU function (see the
    ACS-245 section below; the sibling ``verify_shard`` is CPU-only and
    cost-irrelevant, so it is deliberately not counted). Raises on failure —
    the cost sampler skips the row rather than record a misleading 0.
    """
    _auth()
    count = await _fetch_runner_count(app_name, function_name=HARVEST_FUNCTION_NAME)
    _RUNNERS_CACHE[app_name] = (time.monotonic(), count)
    return count


async def _refresh_runner_count(app_name: str) -> None:
    """Background refresh task body for the runners cache.

    Mirrors ``_refresh_state``: pops from ``_RUNNERS_REFRESHING`` in
    ``finally``; leaves cache untouched if the fetch raised.
    """
    try:
        count = await _fetch_runner_count(app_name)
        _RUNNERS_CACHE[app_name] = (time.monotonic(), count)
    except Exception as exc:  # noqa: BLE001 — leave cache untouched, but log why
        log.warning(
            "runner_count_fetch_failed",
            app_name=app_name,
            where="background_refresh",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _RUNNERS_REFRESHING.discard(app_name)


async def get_active_runner_count(app_name: str) -> int | None:
    """Live container count for ``app_name``'s serve fn, or ``None`` if unknown.

    ``None`` means "we couldn't determine it" — the fetch failed/timed out and
    there's no prior good value. This is deliberately NOT a confident ``0``:
    callers must distinguish "genuinely scaled to zero" from "couldn't ask
    Modal", because rendering an errored count as "scaled to zero" misreported
    warm always-on models as cold (ACS-128). Raises ``ModalOpsError`` on auth.

    SWR semantics:
    - Fresh int hit: return it, no RPC.
    - Stale int hit: return it immediately, kick a background refresh.
    - Unknown (None) hit: return None but always kick a background retry —
      unknown is never "fresh", so it self-heals on the next read instead of
      poisoning the value for the full TTL.
    - Miss: block once on a timeout-bounded fetch; cache the int on success, or
      cache None on failure (so the page stays fast) but as *unknown*, not 0.

    Two mechanisms still correct a stale/unknown value out of band:
    ``mark_runner_warm`` writes the true count through the moment a real
    request succeeds, and the SWR background refresh retries when stale.
    """
    _auth()
    now = time.monotonic()
    cached = _RUNNERS_CACHE.get(app_name)
    if cached is not None:
        ts, value = cached
        if value is None:
            # Unknown: answer immediately, but always retry in the background
            # so a transient failure can't pin the count for the full TTL.
            if app_name not in _RUNNERS_REFRESHING:
                _RUNNERS_REFRESHING.add(app_name)
                asyncio.create_task(_refresh_runner_count(app_name))
            return None
        if now - ts < _RUNNERS_CACHE_TTL_S:
            return value
        if app_name not in _RUNNERS_REFRESHING:
            _RUNNERS_REFRESHING.add(app_name)
            asyncio.create_task(_refresh_runner_count(app_name))
        return value
    # Cache miss: block once on a timeout-bounded fetch. On failure cache None
    # (keeps subsequent reads fast) but as unknown — never a misleading 0.
    try:
        count: int | None = await _fetch_runner_count(app_name)
    except Exception as exc:  # noqa: BLE001 — cache as unknown (None), but log why
        log.warning(
            "runner_count_fetch_failed",
            app_name=app_name,
            where="blocking_miss",
            error=f"{type(exc).__name__}: {exc}",
        )
        count = None
    _RUNNERS_CACHE[app_name] = (now, count)
    return count


def mark_runner_warm(app_name: str, count: int = 1) -> None:
    """Force-populate the runner cache as if Modal just told us ``count``.

    Called from the completion-success path: when a request to
    ``app_name`` succeeds, we have ground-truth that at least one runner
    is alive — write it through to the cache so the workbench warm pill
    reflects reality immediately, even if Modal's control plane is too
    slow to answer ``get_current_stats``. Cheaper and more accurate than
    waiting for the next cache refresh.
    """
    _RUNNERS_CACHE[app_name] = (time.monotonic(), count)
    _RUNNERS_REFRESHING.discard(app_name)


def invalidate_runners(app_name: str) -> None:
    """Drop cached runner count for ``app_name`` so the next render hits Modal.

    Also clears any in-flight refresh marker for the same reason as
    ``invalidate_state``: late-arriving value mustn't clobber the
    invalidation.
    """
    _RUNNERS_CACHE.pop(app_name, None)
    _RUNNERS_REFRESHING.discard(app_name)


def prewarm_caches(app_names: list[str]) -> None:
    """Schedule background tasks to populate both caches for each app name.

    Called once from the FastAPI lifespan hook at wrapper startup so that
    user-visible reads never hit the cold-miss (blocking-RPC) path. Sync
    by design; returns immediately. Assumes a running asyncio loop —
    callers are expected to invoke from inside the lifespan async function.

    If any task fails (Modal unreachable, function missing), it fails
    silently — the cache just stays empty for that app, and the normal
    request path's cache-miss branch will fetch it on the next read.
    """
    for app_name in app_names:
        if app_name not in _STATE_REFRESHING:
            _STATE_REFRESHING.add(app_name)
            asyncio.create_task(_refresh_state(app_name))
        if app_name not in _RUNNERS_REFRESHING:
            _RUNNERS_REFRESHING.add(app_name)
            asyncio.create_task(_refresh_runner_count(app_name))


def _deploy_subprocess(model_id: str) -> tuple[int, str]:
    """Synchronous core of ``deploy_app``; kept separate so tests can patch it."""
    if not os.path.isdir(MODAL_SOURCE_DIR):
        raise ModalOpsError(
            f"modal source directory {MODAL_SOURCE_DIR!r} missing — "
            "Dockerfile did not stage Modal source files at build time"
        )
    missing = [
        rel_path
        for rel_path in MODAL_REQUIRED_SOURCE_FILES
        if not os.path.isfile(os.path.join(MODAL_SOURCE_DIR, rel_path))
    ]
    if missing:
        raise ModalOpsError(
            f"modal source file(s) missing from {MODAL_SOURCE_DIR!r}: {', '.join(missing)}"
        )
    env = {**os.environ, "MODEL_ID": model_id}
    result = subprocess.run(
        ["modal", "deploy", "modal_app.py"],
        cwd=MODAL_SOURCE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=DEPLOY_TIMEOUT_S,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


async def deploy_app(model_id: str) -> tuple[bool, str]:
    """Run ``modal deploy modal_app.py`` with ``MODEL_ID=<model_id>``.

    Returns ``(success, combined_output)``. ``model_id`` is the registry key
    (e.g. ``llama-1b``); ``modal_app.py`` reads ``MODEL_ID`` at module load to
    pick the matching entry from ``serving.modal_config.MODELS`` and derive the modal
    app name. Idempotent — re-deploying an already-deployed app is a fast
    code-upload no-op on Modal's side.
    """
    _auth()
    rc, output = await asyncio.to_thread(_deploy_subprocess, model_id)
    return rc == 0, output


# --- Self-serve bulk activation harvest (ACS-245) ----------------------------
#
# Spawn/poll the offline bulk harvester (serving/harvest_offline.py), deployed
# per-model as app ``acs-<model_id>-harvest``. Its GPU function is ``harvest``
# (plus a CPU-only ``verify_shard`` sibling, not touched here).
#
# Concurrency pattern — the OPPOSITE of ``set_min_containers``: there is no
# clean single-RPC spawn path over the true-async ``_Client`` stub (spawn is
# FunctionMap + serialized-args plumbing), so these calls use the high-level
# synchronicity-wrapped API (``Function.from_name(...).spawn`` /
# ``FunctionCall.from_id(...).get``) inside ``asyncio.to_thread``. Two traps,
# both hit while building this (2026-07-17, modal 1.5.1 + uvicorn/uvloop):
#
#   1. The high-level API called directly on the event loop hangs "outside of
#      task context" (the ACS-202 failure mode) → hence ``to_thread``.
#   2. ``to_thread`` alone is NOT enough in this process: the wrapper's async
#      paths (prewarm, admin controls) await ``_Client.from_env()`` on the
#      main loop, which CACHES a client singleton bound to that loop. The
#      high-level API's own ``from_env`` then reuses the poisoned singleton
#      from synchronicity's thread and hangs on its first RPC (reproduced:
#      "FunctionGet made outside of task context", spawn stuck >90 s; a fresh
#      process without the main-loop client use does NOT hang). Fix: a
#      DEDICATED sync client via ``modal.Client.from_credentials``, hydrated
#      explicitly onto every high-level object, bypassing the from_env cache.
#      (Verified: with the poisoned singleton present, fresh-client spawn/poll
#      both return in <1 s.)
#
# ``_HARVEST_CLIENT`` is created lazily inside the worker thread and reused —
# one extra grpc channel per process, none per request.

HARVEST_FUNCTION_NAME = "harvest"

# Bound a single spawn/poll RPC exchange. Modal's control plane has documented
# slow-spells (see the caches section above); an unbounded call would pin the
# HTTP request. On timeout the worker thread leaks until the RPC returns —
# accepted, same trade-off as the other bounded calls in this module.
_HARVEST_SPAWN_TIMEOUT_S = 60  # spawn also uploads the prompt payload (MBs)
_HARVEST_POLL_TIMEOUT_S = 30

_HARVEST_CLIENT: modal.Client | None = None


def harvest_app_name(model_id: str) -> str:
    """Default Modal app name of the per-model bulk harvester.

    Matches ``serving/harvest_offline.py``'s ``acs-{MODEL_ID}-harvest`` when
    ``model_id`` is the shared-registry key the app was deployed under. The
    route's ``_resolve_harvest_app`` uses this (explicit registry
    ``harvest_app_name`` wins) — keyed on the model id, NOT ``modal_app_name``,
    for the same reason as ``activation_app_name``: Trinity's serving app kept
    the pre-rename name while its sidecars are named from the current id
    (ACS-273).
    """
    return f"acs-{model_id}-harvest"


def resolve_harvest_app(entry: object, model_id: str) -> str | None:
    """Harvest app name for a registry entry, or None if harvest-unsupported.

    Explicit registry ``harvest_app_name`` wins, else derived from the MODEL ID
    (``harvest_app_name(model_id)``). ``modal_app_name`` presence gates
    SUPPORT — an entry with no Modal serving app (legacy / ad-hoc upstream) has
    no harvest sibling either (same gate as the /v1/harvest route, ACS-273).
    Shared by the route's ``_resolve_harvest_app`` and the cost sampler
    (ACS-281) so the two can't drift.
    """
    explicit = getattr(entry, "harvest_app_name", None)
    if explicit:
        return explicit
    if getattr(entry, "modal_app_name", None):
        return harvest_app_name(model_id)
    return None


class HarvestUnavailableError(ModalOpsError):
    """The harvest app for this model isn't deployed / reachable (→ 503)."""


def _harvest_client() -> modal.Client:
    """The dedicated sync client for harvest ops (see trap 2 above).

    Only ever touched from ``to_thread`` workers; benign if two first calls
    race (last write wins, both clients are valid).
    """
    global _HARVEST_CLIENT
    if _HARVEST_CLIENT is None:
        _HARVEST_CLIENT = modal.Client.from_credentials(
            os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"]
        )
    return _HARVEST_CLIENT


def _spawn_harvest_sync(
    app_name: str,
    model_id: str,
    prompts: list[str],
    layer_indices: list[int] | str | None,
    shard_size: int,
    batch_size: int | None,
    run_id: str,
    project_onto: dict | None = None,
    add_special_tokens: bool | None = None,
) -> str:
    try:
        fn = modal.Function.from_name(app_name, HARVEST_FUNCTION_NAME)
        fn.hydrate(client=_harvest_client())
        spawn_kwargs: dict[str, object] = {
            "prompts": prompts,
            "layer_indices": layer_indices,
            "shard_size": shard_size,
            "run_id": run_id,
            "batch_size": batch_size,
        }
        # Only send project_onto when the caller asked for it (ACS-320): a harvest
        # app deployed before that parameter existed rejects the extra kwarg, so
        # omitting it keeps ordinary jobs working against an older deployment. A
        # projection job against an old app fails loudly at Modal — the right
        # signal to redeploy (see the deploy order in the runbook).
        if project_onto is not None:
            spawn_kwargs["project_onto"] = project_onto
        # Same contract for add_special_tokens (ACS-319): omit it unless the caller
        # set it, so the spawn stays compatible with apps predating the parameter.
        if add_special_tokens is not None:
            spawn_kwargs["add_special_tokens"] = add_special_tokens
        call = fn.spawn(**spawn_kwargs)
    except modal.exception.NotFoundError as exc:
        raise HarvestUnavailableError(
            f"harvest app {app_name!r} is not deployed for model {model_id!r}"
        ) from exc
    return call.object_id


async def spawn_harvest(
    app_name: str,
    model_id: str,
    prompts: list[str],
    layer_indices: list[int] | str | None,
    shard_size: int,
    batch_size: int | None,
    run_id: str,
    project_onto: dict | None = None,
    add_special_tokens: bool | None = None,
) -> str:
    """Spawn one bulk-harvest job on ``app_name``; return the Modal
    FunctionCall id (the poll handle persisted on the ``harvest_jobs`` row).

    ``batch_size=None`` is passed through to the harvest function, whose own
    default applies (batched on single-GPU, sequential on multi-GPU) — the
    wrapper never invents a batch size.

    ``add_special_tokens=None`` (the default) means "do not send the kwarg" — the
    harvest function's own default (True) applies and the spawn stays compatible
    with harvest apps deployed before the parameter existed (ACS-319). A non-None
    value is only supplied when the API caller explicitly set it.

    Raises ``HarvestUnavailableError`` when the app isn't deployed,
    ``ModalOpsError`` when Modal credentials are missing or the exchange
    exceeds its time bound. Anything else (control-plane failure mid-spawn)
    propagates for the route to map to 503; the caller owns the already-
    inserted ``pending`` job row and marks it failed.
    """
    _auth()
    try:
        async with asyncio.timeout(_HARVEST_SPAWN_TIMEOUT_S):
            return await asyncio.to_thread(
                _spawn_harvest_sync,
                app_name,
                model_id,
                prompts,
                layer_indices,
                shard_size,
                batch_size,
                run_id,
                project_onto,
                add_special_tokens,
            )
    except TimeoutError as exc:
        raise ModalOpsError(
            f"Timed out after {_HARVEST_SPAWN_TIMEOUT_S}s spawning harvest for "
            f"{model_id!r} (Modal control-plane slow-spell)"
        ) from exc


# Infrastructure exceptions during a poll — say nothing about the JOB, only
# about the wrapper↔Modal exchange, so the poll reports "running" and the next
# poll retries. Includes the whole GRPCError family (UNAVAILABLE, auth,
# not-found on a control-plane blip), stream teardown, and socket errors.
# NB: a permanently-wrong call id therefore polls as "running" until the
# stale-job age-out (``request_log.HARVEST_STALE_AFTER``) fails it — bounded.
_TRANSIENT_POLL_EXCEPTIONS = (
    grpclib.exceptions.GRPCError,
    grpclib.exceptions.StreamTerminatedError,
    modal.exception.ClientClosed,
    modal.exception.ConnectionError,
    OSError,  # covers builtin ConnectionError / socket failures
)


def classify_poll_exception(exc: BaseException) -> tuple[str, str | None]:
    """Map an exception from ``FunctionCall.get(timeout=0)`` to a poll outcome.

    Returns ``(status, client_safe_error)``. Only exceptions that mean *the
    function itself failed* become terminal ``failed``; infrastructure errors
    (grpc UNAVAILABLE, auth, network, hydration) report ``running`` so the next
    poll retries — a transient control-plane blip must never permanently fail a
    job. The client-safe string carries at most the exception CLASS name; raw
    exception text stays in server logs (it can embed infra details).

    Modal exception taxonomy (1.5.x, verified against ``modal.exception`` MROs):
    ``modal.exception.TimeoutError`` is modal's own class (NOT the builtin) and
    is what ``get(timeout=0)`` raises while the call is still executing;
    ``FunctionTimeoutError``/``OutputExpiredError`` subclass it but are
    terminal (remote execution timeout / result expired), so they're checked
    first. A remote raise surfaces as the function's own (deserialized)
    exception class or ``RemoteError`` — anything not classified as
    still-running or transient-infra is treated as that terminal case.
    """
    if isinstance(exc, modal.exception.FunctionTimeoutError):
        return "failed", "job exceeded its Modal execution timeout"
    if isinstance(exc, modal.exception.OutputExpiredError):
        return "failed", (
            "job result expired on Modal before it was collected; resubmit "
            "(shards may still be on the harvest volume — ask the operators)"
        )
    if isinstance(exc, (TimeoutError, modal.exception.TimeoutError)):
        return "running", None  # not finished yet — the normal in-flight case
    if isinstance(exc, _TRANSIENT_POLL_EXCEPTIONS):
        return "running", None
    return "failed", f"job failed on Modal ({type(exc).__name__}); details in server logs"


def _poll_harvest_sync(call_id: str) -> tuple[str, Any]:
    # Client construction and hydration live INSIDE the try: a failure there is
    # infrastructure too and must classify as transient, not 500 the route or
    # fail the job.
    try:
        fc = modal.FunctionCall.from_id(call_id, client=_harvest_client())
        result = fc.get(timeout=0)
    except Exception as exc:
        status, safe_error = classify_poll_exception(exc)
        if status == "failed" or not isinstance(exc, (TimeoutError, modal.exception.TimeoutError)):
            # Full detail server-side only (skip the noisy still-running case).
            log.warning(
                "harvest_poll_exception",
                call_id=call_id,
                classified=status,
                error=f"{type(exc).__name__}: {exc}",
            )
        return status, safe_error
    return "done", result


async def poll_harvest(call_id: str) -> tuple[str, Any]:
    """Non-blocking status probe of a spawned harvest call.

    Returns one of:
      - ``("running", None)`` — the call hasn't finished, or the exchange hit a
        transient infrastructure error (retry on the next poll);
      - ``("done", result_dict)`` — finished; ``result_dict`` is the harvest
        function's return value (manifest/shard URLs, volume path, timings);
      - ``("failed", client_safe_error)`` — the remote function itself failed
        (remote raise / remote timeout / result expired). Never raw exception
        text — see ``classify_poll_exception``.

    A wrapper-side time bound on the exchange also reports ``running`` — a
    control-plane slow-spell says nothing about the job.
    """
    _auth()
    try:
        async with asyncio.timeout(_HARVEST_POLL_TIMEOUT_S):
            return await asyncio.to_thread(_poll_harvest_sync, call_id)
    except TimeoutError:
        log.warning("harvest_poll_timeout", call_id=call_id)
        return "running", None


def _cancel_harvest_sync(call_id: str) -> None:
    fc = modal.FunctionCall.from_id(call_id, client=_harvest_client())
    # terminate_containers=True actually kills the running container (and its
    # GPU); a bare cancel() only detaches the call handle and leaves a one-input
    # harvest container running to completion. Verified against modal 1.5.1
    # (FunctionCall.cancel(self, terminate_containers: bool = False)).
    fc.cancel(terminate_containers=True)


async def cancel_harvest(call_id: str) -> bool:
    """Best-effort request that Modal stop a spawned harvest call.

    Returns ``True`` when the cancel RPC was accepted, ``False`` on any error
    (bad/expired call id, control-plane blip, credentials, time bound). The
    caller has ALREADY marked the job ``cancelled`` and freed the key's
    concurrency slot in Postgres — this only frees the GPU sooner. A ``False``
    return therefore means "the DB is authoritative; the Modal container may
    keep running until it finishes or hits its function timeout", never that
    the cancellation failed for the client. Never raises — cancellation must
    not 500 the DELETE route once the job is already recorded cancelled.
    """
    # _auth() is INSIDE the try on purpose: it raises ModalOpsError when Modal
    # credentials are unset, and that must classify as a (best-effort) False, not
    # bubble a 500 out of the DELETE route after the job is already cancelled.
    try:
        _auth()
        async with asyncio.timeout(_HARVEST_POLL_TIMEOUT_S):
            await asyncio.to_thread(_cancel_harvest_sync, call_id)
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort; DB is authoritative
        log.warning(
            "harvest_cancel_failed",
            call_id=call_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return False


# ---------------------------------------------------------------------------
# Cold-boot stage Dict (ACS-272). Serving containers publish authored boot
# stages (container_started → weights_loading → … → serving, or failed) into
# a shared Modal Dict keyed by app name — see serving/boot_status.py. The
# wrapper reads them here to show users cold-boot progress.
#
# Read path deliberately mirrors _fetch_runner_count: raw RPCs over the true
# async client, NOT the high-level modal.Dict API — the synchronicity-wrapped
# path issues RPCs "outside of task context" in the deployed wrapper and
# hangs (same failure mode as ACS-202 / the get_current_stats incident above).
# RPC names + serialization helpers verified against the pinned modal 1.5.1
# (DictGetOrCreate / DictGet, modal._serialization.serialize/deserialize —
# the exact calls modal/dict.py itself makes).
# ---------------------------------------------------------------------------

BOOT_STATUS_DICT_NAME = "acs-boot-status"  # keep in sync with serving/boot_status.py
_BOOT_STATUS_TTL_S = 4.0  # keepalive/status ticks are ~5 s; one RPC per tick max
_BOOT_STATUS_TIMEOUT_S = 2.0
_BOOT_STATUS_CACHE: dict[str, tuple[float, dict | None]] = {}
_BOOT_STATUS_DICT_ID: str | None = None


async def _fetch_boot_status(app_name: str) -> dict | None:
    global _BOOT_STATUS_DICT_ID
    from modal._serialization import deserialize, serialize

    client = await _ModalAsyncClient.from_env()
    if _BOOT_STATUS_DICT_ID is None:
        resp = await client.stub.DictGetOrCreate(
            api_pb2.DictGetOrCreateRequest(
                deployment_name=BOOT_STATUS_DICT_NAME,
                environment_name="",
                object_creation_type=api_pb2.OBJECT_CREATION_TYPE_CREATE_IF_MISSING,
            )
        )
        _BOOT_STATUS_DICT_ID = resp.dict_id
    get_resp = await client.stub.DictGet(
        api_pb2.DictGetRequest(dict_id=_BOOT_STATUS_DICT_ID, key=serialize(app_name))
    )
    if not get_resp.found:
        return None
    value = deserialize(get_resp.value, client)
    return value if isinstance(value, dict) else None


async def get_boot_status(app_name: str) -> dict | None:
    """Latest boot-status entry for ``app_name``, or None. Never raises.

    Cached for _BOOT_STATUS_TTL_S so the 5 s keepalive/status loops cost at
    most one RPC per tick per model, and time-bounded so a slow control plane
    can never stall a keepalive. Failures cache None — callers treat that
    identically to "no entry" (the banner falls back to elapsed-time labels).
    """
    if not app_name:
        return None
    now = time.monotonic()
    cached = _BOOT_STATUS_CACHE.get(app_name)
    if cached is not None and now - cached[0] < _BOOT_STATUS_TTL_S:
        return cached[1]
    try:
        _auth()
        value = await asyncio.wait_for(
            _fetch_boot_status(app_name), timeout=_BOOT_STATUS_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001 - progress info is best-effort
        log.debug("boot_status_fetch_failed", app_name=app_name, error=repr(exc))
        value = None
    _BOOT_STATUS_CACHE[app_name] = (now, value)
    return value
