"""Workbench query and model-state helpers."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import modal_ops as modalops
from ..logging import get_logger
from ..models import ApiKey, ChatGeneration, ChatSession, ChatSnapshot, ModelWarmWindow

log = get_logger()


# Per-model startup-time estimate for the "needs startup" label. Coarse but
# data-informed: two prod 405B cold boots measured ~3 min and ~6 min (variance
# from volume-cache/Modal capacity), so the range starts at 3. The upper bound
# is set conservatively (10) to under-promise — better a pleasant surprise than
# a blown expectation. Edit when per-model cold-boot p50/p95s are available.
_STARTUP_LABEL_BIG = "Needs startup (~3-10 min first request)"
_STARTUP_LABEL_SMALL = "Needs startup (~30s first request)"
_STARTUP_TOOLTIP = (
    "This model scales to zero between requests. The first request after an "
    "idle period waits for the container to boot; later requests are fast."
)
_ALWAYS_ON_LABEL = "Always on"
_ALWAYS_ON_TOOLTIP = (
    "This model is kept warm, so requests return immediately without a "
    "cold-start wait."
)
# Soft "recently used" hint (ACS-98): a non-always-on model used within its
# scale-down window is *probably* still warm. Deliberately under-promises — a
# container can die inside the window — so the wording is "usually warm", not a
# guarantee. Derived from the in-memory last_completion_at map; no Modal RPC.
_WARM_HINT_LABEL = "Recently used — usually warm"
_WARM_HINT_TOOLTIP = (
    "This model was used recently, so it's probably still warm and the next "
    "request should be fast. It scales to zero when idle, so this isn't a "
    "guarantee — an unused model cold-starts again."
)
# Fallback when a model has no shared-registry spec (e.g. the dev-local gpt2
# single-model fallback). Mirrors acs_model_registry.DEFAULT_SCALEDOWN_WINDOW_S.
_DEFAULT_SCALEDOWN_WINDOW_S = 30 * 60


def _maybe_model_spec_fn():
    """Import ``acs_model_registry.maybe_model_spec`` with the same disk-path
    fallback ``settings._wrapper_defaults_for_model`` uses, so wrapper envs
    that ship the registry as a sibling directory (not installed as a package)
    still resolve specs. Returns None if the registry is unreachable.
    """
    try:
        from acs_model_registry import maybe_model_spec as _fn
        return _fn
    except ModuleNotFoundError:
        pass
    import sys
    from importlib import util as importlib_util
    from pathlib import Path
    registry_path = Path(__file__).resolve().parents[4] / "acs_model_registry" / "__init__.py"
    if not registry_path.is_file():
        return None
    spec = importlib_util.spec_from_file_location("acs_model_registry", registry_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib_util.module_from_spec(spec)
    sys.modules["acs_model_registry"] = module
    spec.loader.exec_module(module)
    return module.maybe_model_spec


def model_spec_for(model_id: str):
    """Return the shared-registry ModelSpec for ``model_id``, or None."""
    fn = _maybe_model_spec_fn()
    if fn is None:
        return None
    return fn(model_id)


def compute_model_startup_labels(
    registry: dict[str, Any],
    warm_window_rows: list[ModelWarmWindow],
    last_completion_at: dict[str, dt.datetime] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute the per-model startup/warmth label for the workbench picker.

    Priority:
      1. Enabled ``ModelWarmWindow`` row -> "Always on" (treated as 24/7 for
         the beta label even though the cron only flips warm during a window;
         refine here if users get confused).
      2. ``acs_model_registry.SPECS[model_id].min_containers >= 1`` -> "Always on".
      3. Otherwise, for an on-demand model: if it was used within its scale-down
         window (``last_completion_at[model_id]`` + the spec's
         ``scaledown_window_s``) -> a soft "recently used - usually warm" hint
         (ACS-98); else -> "Needs startup (~Xs)" from the GPU-shape heuristic.

    The warmth hint reads only the in-memory ``last_completion_at`` map (also
    used by ``/health``) — **no Modal RPC**, so it doesn't reintroduce the
    per-render latency that retired the old live pill. It under-promises by
    design ("usually warm"): a container can die inside the window.

    Returns ``{model_id: {"always_on": bool, "warm_hint": bool, "label": str,
    "tooltip": str}}``. Models not in the shared registry (e.g. the dev-local
    ``gpt2`` fallback) default to "Needs startup" with the small-model estimate
    — safer to under-promise than to falsely claim always-on.
    """
    enabled_warm_ids = {w.model_id for w in warm_window_rows if w.enabled}
    last_completion_at = last_completion_at or {}
    if now is None:
        now = dt.datetime.now(tz=dt.UTC)
    out: dict[str, dict[str, Any]] = {}
    for entry in registry.values():
        model_id = entry.model_id
        spec = model_spec_for(model_id)
        registry_always_on = bool(spec and spec.min_containers >= 1)
        always_on = registry_always_on or (model_id in enabled_warm_ids)
        if always_on:
            out[model_id] = {
                "always_on": True,
                "warm_hint": False,
                "label": _ALWAYS_ON_LABEL,
                "tooltip": _ALWAYS_ON_TOOLTIP,
            }
            continue
        # Soft "recently used - usually warm" hint from the RPC-free
        # last-completion timestamp, within the model's scale-down window.
        last = last_completion_at.get(model_id)
        window_s = spec.scaledown_window_s if spec else _DEFAULT_SCALEDOWN_WINDOW_S
        if last is not None and (now - last).total_seconds() < window_s:
            out[model_id] = {
                "always_on": False,
                "warm_hint": True,
                "label": _WARM_HINT_LABEL,
                "tooltip": _WARM_HINT_TOOLTIP,
            }
            continue
        # GPU-shape heuristic for the startup estimate. n_gpu * n_nodes is the
        # total GPU count; >=8 are the cold-boot-heavy 70B / 405B / Trinity
        # shapes. Single-GPU models (Llama-8B) cold-boot in tens of seconds.
        total_gpus = (spec.n_gpu * spec.n_nodes) if spec else 1
        label = _STARTUP_LABEL_BIG if total_gpus >= 8 else _STARTUP_LABEL_SMALL
        out[model_id] = {
            "always_on": False,
            "warm_hint": False,
            "label": label,
            "tooltip": _STARTUP_TOOLTIP,
        }
    return out


async def all_warm_window_rows(session: AsyncSession) -> list[ModelWarmWindow]:
    """All ``ModelWarmWindow`` rows; used by the workbench startup-label helper."""
    return list((await session.execute(select(ModelWarmWindow))).scalars().all())


async def workbench_keys(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """Active, completions-scoped keys the user can pick for a workbench chat."""
    rows = (
        (
            await session.execute(
                select(ApiKey)
                .where(ApiKey.user_id == user_id)
                .where(ApiKey.revoked_at.is_(None))
                .order_by(ApiKey.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [
        {"id": str(k.id), "name": k.name, "key_prefix": k.key_prefix}
        for k in rows
        if "completions" in (k.scopes or [])
    ]


async def user_chat_sessions(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    include_archived: bool = False,
    limit: int,
) -> list[ChatSession]:
    """Sidebar listing: pinned chats first (newest pin on top), then the rest
    by recency (ACS-257)."""
    q = select(ChatSession).where(ChatSession.user_id == user_id)
    if not include_archived:
        q = q.where(ChatSession.archived_at.is_(None))
    q = q.order_by(
        ChatSession.pinned_at.desc().nullslast(),
        ChatSession.updated_at.desc(),
    ).limit(limit)
    return list((await session.execute(q)).scalars().all())


async def get_owned_chat_session(
    session: AsyncSession,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ChatSession | None:
    """Fetch a chat session by id, only if it belongs to this user."""
    row = (
        await session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    return row


async def session_snapshots(
    session: AsyncSession,
    session_id: uuid.UUID,
    *,
    limit: int,
) -> list[ChatSnapshot]:
    q = (
        select(ChatSnapshot)
        .where(ChatSnapshot.session_id == session_id)
        .order_by(ChatSnapshot.ts.desc())
        .limit(limit)
    )
    return list((await session.execute(q)).scalars().all())


async def resolve_workbench_model_states(
    registry: dict[str, Any],
) -> dict[str, str]:
    """Fan out Modal state lookups for live models with a Modal app name."""
    targets = [m for m in registry.values() if m.status == "live" and m.modal_app_name]
    if not targets:
        return {}
    results = await asyncio.gather(
        *(modalops.get_app_state(m.modal_app_name) for m in targets),
        return_exceptions=True,
    )
    out: dict[str, str] = {}
    for model, result in zip(targets, results):
        if isinstance(result, BaseException):
            log.warning(
                "workbench_model_state_lookup_failed",
                model_id=model.model_id,
                modal_app_name=model.modal_app_name,
                err=repr(result),
            )
            continue
        out[model.model_id] = result
    return out


async def running_generation_for_session(
    session: AsyncSession,
    session_id: uuid.UUID,
) -> ChatGeneration | None:
    return (
        await session.execute(
            select(ChatGeneration)
            .where(ChatGeneration.session_id == session_id)
            .where(ChatGeneration.status == "running")
            .order_by(ChatGeneration.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
