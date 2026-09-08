"""Workbench routes and route-local generation compatibility."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import web_auth as webauth
from .. import workbench_generations as genmod
from ..db import get_session
from ..dependencies import get_settings
from ..logging import get_logger
from ..model_resolution import _resolve_model, _unknown_model_response
from ..models import ChatGeneration, ChatSession, ChatSnapshot, CompareSnapshot, User
from ..services import completions as completion_svc
from ..services import workbench as workbench_svc
from ..settings import Settings
from ..tokenizer import get_token_counter

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

log = get_logger()
router = APIRouter()

GenerationState = genmod.GenerationState
LoomGenerationState = genmod.LoomGenerationState
GENERATION_EVICT_S = genmod.GENERATION_EVICT_S
GENERATION_FLUSH_INTERVAL_S = genmod.GENERATION_FLUSH_INTERVAL_S
GENERATION_FLUSH_CHARS = genmod.GENERATION_FLUSH_CHARS
HEARTBEAT_INTERVAL_S = genmod.HEARTBEAT_INTERVAL_S
SSE_HEADERS = genmod.SSE_HEADERS


async def _workbench_keys(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """Active, completions-scoped keys the user can pick for a workbench chat.

    Newest first — matches ``primary_authed_caller``'s default ordering, so the
    top entry is the same key the workbench would have used before this picker
    existed. Returns ``id`` (str), ``name``, and ``key_prefix`` for the dropdown.
    """
    return await workbench_svc.workbench_keys(session, user_id)


# --- web chat workbench ------------------------------------------------------

_CHAT_DEFAULT_MAX_TOKENS = 200
# Hard cap on max_tokens for a single Continue / compare lane. Surfaced inline in
# the composer (ACS-346) so users see it before a request trips it, matching how
# Compare shows its lane cap. Single source for the server clamp (_build_lane_body),
# the number input's `max`, and the visible hint.
_CHAT_MAX_MAX_TOKENS = 4000
# Base models are faithfully sampled at temperature 1.0 (0.7 is a chat-tuned
# default that biases the distribution). Our research audience expects 1.0 as
# the neutral starting point — ACS-145 (Theia's feedback).
_CHAT_DEFAULT_TEMPERATURE = 1.0
_CHAT_SIDEBAR_LIMIT = 50
_CHAT_SNAPSHOT_LIMIT = 20
# Compare-mode saved snapshots (ACS-180): one row per Run-all, capped like the
# single-pane history — oldest pruned once the cap is exceeded.
_COMPARE_SNAPSHOT_LIMIT = 20

# Loom (ACS-148) clamps. ``n`` is the branching factor per generate; capped low
# so a single explore can't fan out an unbounded number of upstream sequences.
# ``logprobs`` is the per-token top-k for the heatmap popover; small because the
# UI only shows a handful of alternatives and the payload is stored as JSONB.
_LOOM_DEFAULT_N = 3
_LOOM_MAX_N = 8
_LOOM_DEFAULT_MAX_TOKENS = 60
_LOOM_MAX_MAX_TOKENS = 1000
_LOOM_DEFAULT_LOGPROBS = 5
_LOOM_MAX_LOGPROBS = 20
# Hard cap on nodes per loom so a runaway explore can't grow the tree without
# bound; the UI surfaces the cap when hit.
_LOOM_MAX_NODES = 2000

# Models hidden from the workbench dropdown even when they're still live in
# the registry. Keep empty for the beta surface: all live registry models are
# intentionally visible to testers.
_UI_HIDDEN_MODEL_IDS: frozenset[str] = frozenset()


def _model_total_gpus(model_id: str) -> int:
    """Total GPUs (``n_gpu × n_nodes``) for a model from the shared registry
    spec, or 0 if the spec is unavailable. Used to rank always-on models by
    size for the workbench default pre-selection (ACS-145)."""
    spec = workbench_svc.model_spec_for(model_id)
    return (spec.n_gpu * spec.n_nodes) if spec else 0


def _select_workbench_model_id(
    chat_model: str | None,
    available_models: list[dict[str, Any]],
    default_model_id: str,
) -> str:
    """Model pre-selected in the workbench picker (ACS-145).

    An existing chat keeps its saved model. A new chat defaults to the
    **largest always-on** model so a tester never lands on an on-demand model
    and triggers an expensive cold boot by accident (Theia's picker defaulted
    to 405B). Falls back to the configured default when nothing is always-on
    (e.g. the dev-local single-model registry).
    """
    if chat_model:
        return chat_model
    always_on_ids = [am["id"] for am in available_models if am.get("startup_always_on")]
    if not always_on_ids:
        return default_model_id
    return max(always_on_ids, key=_model_total_gpus)


_friendly_upstream_error = genmod._friendly_upstream_error
_derive_title = genmod._derive_title


async def _user_chat_sessions(
    session: AsyncSession, user_id: uuid.UUID, *, include_archived: bool = False
) -> list[ChatSession]:
    """Sidebar listing: this user's non-archived sessions, newest activity first."""
    return await workbench_svc.user_chat_sessions(
        session,
        user_id,
        include_archived=include_archived,
        limit=_CHAT_SIDEBAR_LIMIT,
    )


async def _get_owned_chat_session(
    session: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> ChatSession | None:
    """Fetch a chat session by id, but only if it belongs to this user.

    Ownership enforcement: every /workbench/<id>/* route uses this; mismatched
    user_id returns None (caller raises 404) so we don't leak the existence
    of other users' session ids.
    """
    return await workbench_svc.get_owned_chat_session(session, session_id, user_id)


async def _session_snapshots(session: AsyncSession, session_id: uuid.UUID) -> list[ChatSnapshot]:
    return await workbench_svc.session_snapshots(
        session,
        session_id,
        limit=_CHAT_SNAPSHOT_LIMIT,
    )


async def _resolve_workbench_model_states(
    registry: dict[str, Any],
) -> dict[str, str]:
    """Fan out Modal state lookups in parallel for every live model with a
    ``modal_app_name``. Returns ``{model_id: state}``.

    State values mirror ``modal_ops.get_app_state``: ``"deployed"``,
    ``"stopped"``, ``"initializing"``, ``"unknown"``. Models without
    ``modal_app_name`` and models where the RPC raised are omitted from the
    map; callers should treat their absence as "no veto" (show them) so a
    Modal control-plane hiccup can't empty the workbench dropdown.

    Currently dead code — the workbench warm/cold pill UX it fed is disabled
    (see commit 72da33a and the commented block in workbench.html). Helper
    preserved here so a future revival of the pill can plug it back in
    without rewriting the parallel-lookup logic.
    """
    return await workbench_svc.resolve_workbench_model_states(registry)


async def _running_generation_for_session(
    session: AsyncSession, session_id: uuid.UUID
) -> ChatGeneration | None:
    """Most-recent running generation for a chat, or None."""
    return await workbench_svc.running_generation_for_session(session, session_id)


def _build_model_options(
    registry: dict[str, Any], startup_labels: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Shared workbench model dropdown data.

    Returns ``(available_models, model_labels)``. ``available_models`` is the
    live, non-hidden subset shown in pickers (single-pane workbench and the
    compare lanes); ``model_labels`` covers *every* registered model so old
    snapshots/generations can still resolve a friendly name even after a model
    is retired.
    """
    available_models: list[dict[str, Any]] = []
    model_labels: dict[str, str] = {}
    for m in registry.values():
        label = (m.served_model_name or m.model_id).rsplit("/", 1)[-1]
        model_labels[m.model_id] = label
        if m.status != "live":
            continue
        if m.model_id in _UI_HIDDEN_MODEL_IDS:
            continue
        status_info = startup_labels.get(
            m.model_id,
            {"always_on": False, "warm_hint": False, "label": "", "tooltip": ""},
        )
        available_models.append(
            {
                "id": m.model_id,
                # Display the upstream's served_model_name (last path segment) so
                # users see e.g. "Llama-3.1-405B" instead of the short id
                # "llama-405b". Falls back to the short id if no served name.
                "label": label,
                "gpu_shape": m.gpu_shape_label,
                "status": m.status,
                "startup_always_on": status_info["always_on"],
                "startup_warm_hint": status_info.get("warm_hint", False),
                "startup_label": status_info["label"],
                "startup_tooltip": status_info["tooltip"],
            }
        )
    return available_models, model_labels


async def _render_chat(
    request: Request,
    *,
    user: User,
    chat: ChatSession | None,
    sidebar: list[ChatSession],
    snapshots: list[ChatSnapshot],
    key_prefix: str | None,
    keys: list[dict[str, Any]] | None = None,
    selected_key_id: str | None = None,
    no_key: bool = False,
    error: str | None = None,
    status_code: int = 200,
    running_generation: ChatGeneration | None = None,
    session: AsyncSession | None = None,
    mode: str = "single",
):
    # Compare-mode prompt prefill (issue #11): the session prompt is a rolling
    # continuation buffer by design, so it holds prompt + completion after a
    # single-pane run. Compare must start from the prompt baseline instead.
    # The boundary comes from the latest roll-forward snapshot (written in the
    # same transaction as the roll-forward — a failed/cancelled-with-partial
    # run also writes one), NOT the latest generation: a failed run has no
    # snapshot, and a generation row restored-over would hold the wrong
    # boundary. Substitute the baseline only when the session prompt is
    # EXACTLY baseline + completion — the unmodified roll-forward. Every other
    # state (post-run edits, fresh typing, a Compare run's write-back, snapshot
    # restore) fails the equality check and carries over untouched.
    compare_prompt = chat.prompt_text if chat is not None else ""
    compare_roll_forward = None
    if chat is not None and snapshots:
        last = snapshots[0]  # session_snapshots returns newest first
        rolled = last.prompt_before + last.completion_text
        if chat.prompt_text == rolled:
            compare_roll_forward = {
                "baseline": last.prompt_before,
                "completion": last.completion_text,
            }
            compare_prompt = last.prompt_before
    # Per-model startup/warmth label for the picker. Always-on vs needs-startup
    # is registry-config-derived (ACS-80, incl. enabled ModelWarmWindow rows);
    # on-demand models additionally get a soft "recently used - usually warm"
    # hint from the in-memory last_completion_at map (ACS-98). Both are RPC-free
    # — no per-render Modal call, so TTFB stays fast (the reason the old live
    # pill was retired). If we have a DB session, look up warm-window overrides.
    registry = request.app.state.models
    if session is not None:
        warm_windows = await workbench_svc.all_warm_window_rows(session)
    else:
        warm_windows = []
    startup_labels = workbench_svc.compute_model_startup_labels(
        registry,
        warm_windows,
        last_completion_at=getattr(request.app.state, "last_completion_at", {}),
        now=dt.datetime.now(tz=dt.UTC),
    )
    # One builder shared with the compare page (_build_model_options), so the two
    # pickers can't drift. ``startup_labels`` here carries the richer warm-hint
    # context; the helper just shapes it.
    available_models, model_labels = _build_model_options(registry, startup_labels)
    # Pre-select the largest always-on model for a new chat (ACS-145) — see
    # _select_workbench_model_id.
    selected_model_id = _select_workbench_model_id(
        chat.model if chat else None,
        available_models,
        request.app.state.default_model_id,
    )
    resp = templates.TemplateResponse(
        request,
        "workbench.html",
        {
            "user": user,
            "chat": chat,
            "sidebar": sidebar,
            "snapshots": snapshots,
            "key_prefix": key_prefix,
            "keys": keys or [],
            "selected_key_id": selected_key_id,
            "no_key": no_key,
            "error": error,
            "available_models": available_models,
            "selected_model_id": selected_model_id,
            "model_labels": model_labels,
            # Which composer the unified workbench opens in (ACS-163). "compare"
            # renders the multi-lane pane; anything else falls back to single.
            # Compare-pane prefill (issue #11): the baseline-substituted shared
            # prompt (see top of this function) plus the exact boundary data the
            # client toggle guard needs to recognise the unmodified roll-forward
            # in the single textarea. ``compare_roll_forward`` is None whenever
            # the session prompt is NOT an unmodified roll-forward (no snapshot,
            # edits, restore, fresh typing), and the toggle then copies verbatim.
            "mode": "compare" if mode == "compare" else "single",
            "compare_prompt": compare_prompt,
            "compare_roll_forward": compare_roll_forward,
            # Compare-pane config (folded into the same page as single-pane).
            "default_max_tokens": _CHAT_DEFAULT_MAX_TOKENS,
            "max_max_tokens": _CHAT_MAX_MAX_TOKENS,
            "default_temperature": _CHAT_DEFAULT_TEMPERATURE,
            "max_lanes": _COMPARE_MAX_LANES,
            "running_generation": (
                {
                    "id": str(running_generation.id),
                    "completion_text": running_generation.completion_text,
                    "prompt_before": running_generation.prompt_before,
                    # Epoch ms of the real generation start, so a resuming tab
                    # anchors its cold-boot timer to the true elapsed time
                    # (ticks smoothly from the right value) rather than from 0.
                    "started_at_ms": int(running_generation.started_at.timestamp() * 1000),
                }
                if running_generation is not None
                else None
            ),
        },
        status_code=status_code,
    )
    # Chromium/Safari restore textarea form state when navigating to the same
    # URL (e.g. after a POST→303 redirect that lands back on /workbench/<id>).
    # Disable bfcache + caching so a fresh GET always reflects the DB.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@router.get("/workbench")
async def chat_index(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Land on the most-recent session; create one if the user has none yet."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    sidebar = await _user_chat_sessions(session, user.id)
    if not sidebar:
        # First-time landing: mint an empty session so the URL has an id.
        chat = ChatSession(user_id=user.id, title="Untitled", prompt_text="")
        session.add(chat)
        await session.flush()
        return RedirectResponse(url=f"/workbench/{chat.id}", status_code=303)
    return RedirectResponse(url=f"/workbench/{sidebar[0].id}", status_code=303)


@router.get("/workbench/models/status")
async def workbench_models_status(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
):
    """JSON snapshot of the workbench-visible models. No Modal RPCs.

    Historically returned per-model warm/cold pills via Modal control-plane
    lookups. That coupled every workbench render (and every 10 s JS poll)
    to Modal's latency — a slow Modal day produced 18 s page TTFBs. The
    pill UX is gone; this endpoint stays for the JS poll's no-op call and
    for any external clients introspecting the dropdown contents.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthenticated"}})
    registry = request.app.state.models
    models = []
    for m in registry.values():
        if m.status != "live":
            continue
        if m.model_id in _UI_HIDDEN_MODEL_IDS:
            continue
        models.append({"id": m.model_id})
    return JSONResponse(status_code=200, content={"models": models})


@router.post("/workbench/new")
async def chat_new(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    chat = ChatSession(user_id=user.id, title="Untitled", prompt_text="")
    session.add(chat)
    await session.flush()
    return RedirectResponse(url=f"/workbench/{chat.id}", status_code=303)


@router.get("/workbench/export-all.jsonl")
async def chat_export_all(
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Bulk export every saved session for the current user as JSONL.

    Declared BEFORE ``/workbench/{session_id}`` so the literal path matches
    first; otherwise FastAPI tries to UUID-parse ``export-all.jsonl`` and 422s.
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    sessions = list(
        (
            await session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user.id)
                .order_by(ChatSession.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    import json as _json

    lines: list[str] = []
    for chat in sessions:
        snaps = list(
            (
                await session.execute(
                    select(ChatSnapshot)
                    .where(ChatSnapshot.session_id == chat.id)
                    .order_by(ChatSnapshot.ts.asc())
                )
            )
            .scalars()
            .all()
        )
        lines.append(_json.dumps(_session_to_jsonl_record(chat, snaps)))
    return Response(
        content="\n".join(lines) + ("\n" if lines else ""),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="workbench-all.jsonl"'},
    )


@router.get("/workbench/{session_id}")
async def chat_open(
    request: Request,
    session_id: uuid.UUID,
    mode: str = "single",
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    sidebar = await _user_chat_sessions(session, user.id)
    snapshots = await _session_snapshots(session, chat.id)
    keys = await _workbench_keys(session, user.id)
    # Which key the picker pre-selects: the chat's saved choice if it's still
    # selectable, otherwise the newest active key (keys[0]) — same default as
    # primary_authed_caller, so behaviour is unchanged for chats that never
    # picked one. A revoked/deleted saved key silently falls back here.
    saved_key_id = str(chat.api_key_id) if chat.api_key_id else None
    selectable_ids = {k["id"] for k in keys}
    if saved_key_id in selectable_ids:
        selected_key_id = saved_key_id
    elif keys:
        selected_key_id = keys[0]["id"]
    else:
        selected_key_id = None
    selected_prefix = next((k["key_prefix"] for k in keys if k["id"] == selected_key_id), None)
    running_gen = await _running_generation_for_session(session, chat.id)
    return await _render_chat(
        request,
        user=user,
        chat=chat,
        sidebar=sidebar,
        snapshots=snapshots,
        key_prefix=selected_prefix,
        keys=keys,
        selected_key_id=selected_key_id,
        no_key=not keys,
        running_generation=running_gen,
        session=session,
        mode=mode,
    )


@router.post("/workbench/{session_id}/rename")
async def chat_rename(
    request: Request,
    session_id: uuid.UUID,
    title: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    # The workbench renames via fetch and stays on the page (ACS-254: the old
    # full-reload redirect threw away unsaved textarea state, wiping any
    # compare-mode prompt). The redirect remains as the no-JS fallback.
    wants_json = "application/json" in request.headers.get("accept", "")
    if user is None:
        if wants_json:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    chat.title = (title.strip() or "Untitled")[:120]
    if wants_json:
        return JSONResponse({"ok": True, "title": chat.title})
    return RedirectResponse(url=f"/workbench/{chat.id}", status_code=303)


@router.post("/workbench/{session_id}/pin")
async def chat_pin_toggle(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Toggle sidebar pinning (ACS-257). Fetch-only surface → JSON always."""
    if user is None:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    chat.pinned_at = None if chat.pinned_at else dt.datetime.now(tz=dt.UTC)
    # Echo the server timestamp so the client's in-place re-sort uses the same
    # value the next full render will ORDER BY (client clocks can skew).
    return JSONResponse(
        {
            "ok": True,
            "pinned": chat.pinned_at is not None,
            "pinned_at": chat.pinned_at.isoformat() if chat.pinned_at else None,
        }
    )


@router.post("/workbench/{session_id}/delete")
async def chat_delete(
    request: Request,
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    # JSON branch for the sidebar-menu fetch (ACS-257): without it, an expired
    # session's 303 → /login is followed by fetch to a 200, which reads as a
    # successful delete client-side while nothing was archived.
    wants_json = "application/json" in request.headers.get("accept", "")
    if user is None:
        if wants_json:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    import datetime as _dt

    chat.archived_at = _dt.datetime.now(_dt.UTC)
    if wants_json:
        return JSONResponse({"ok": True})
    return RedirectResponse(url="/workbench", status_code=303)


@router.post("/workbench/{session_id}/revert/{snapshot_id}")
async def chat_revert(
    session_id: uuid.UUID,
    snapshot_id: int,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    snap = (
        await session.execute(
            select(ChatSnapshot).where(
                ChatSnapshot.id == snapshot_id,
                ChatSnapshot.session_id == chat.id,
            )
        )
    ).scalar_one_or_none()
    if snap is None:
        raise HTTPException(status_code=404, detail="snapshot not found")
    import datetime as _dt

    chat.prompt_text = snap.prompt_before
    chat.last_max_tokens = snap.max_tokens
    chat.last_temperature = snap.temperature
    chat.updated_at = _dt.datetime.now(_dt.UTC)
    return RedirectResponse(url=f"/workbench/{chat.id}", status_code=303)


@router.post("/workbench/{session_id}/stream")
async def chat_stream_gone(session_id: uuid.UUID):
    """410 Gone shim for the pre-tab-independent endpoint.

    Old browser tabs from before the deploy still POST here; rather than
    dangling silently, fail loud with a structured body the workbench JS
    can surface. Removable once the feature has been live one release.
    """
    return JSONResponse(
        status_code=410,
        content={
            "error": {
                "message": (
                    "This endpoint was replaced by POST /workbench/{id}/generations + "
                    "GET /workbench/{id}/generations/{gen_id}/events. Please refresh."
                ),
                "code": "endpoint_gone",
                "type": "deprecated",
            }
        },
    )


def _sync_generation_config() -> None:
    genmod.GENERATION_EVICT_S = GENERATION_EVICT_S
    genmod.GENERATION_FLUSH_INTERVAL_S = GENERATION_FLUSH_INTERVAL_S
    genmod.GENERATION_FLUSH_CHARS = GENERATION_FLUSH_CHARS
    genmod.HEARTBEAT_INTERVAL_S = HEARTBEAT_INTERVAL_S


_sse_status_frame = genmod._sse_status_frame
_sse_error_frame = genmod._sse_error_frame
_sse_replay_frame = genmod._sse_replay_frame
_sse_done_frame = genmod._sse_done_frame
_absorb_chunk_text = genmod._absorb_chunk_text
_parse_upstream_error_message = genmod._parse_upstream_error_message
# Loom (ACS-148) compatibility exports — same monkeypatch-friendly pattern as
# the single-stream helpers above so tests can import them from this module.
_absorb_loom_chunk = genmod._absorb_loom_chunk
_normalise_logprobs = genmod._normalise_logprobs
_sse_loom_chunk_frame = genmod._sse_loom_chunk_frame


async def _gen_live(state: GenerationState, since: int | None, gen_id: uuid.UUID | None = None):
    _sync_generation_config()
    async for frame in genmod._gen_live(state, since, gen_id=gen_id):
        yield frame


async def _run_generation_task(**kwargs) -> None:
    _sync_generation_config()
    await genmod._run_generation_task(
        **kwargs,
        mark_model_warm=genmod.mark_model_warm_from_app,
    )


async def _loom_gen_live(state: LoomGenerationState, gen_id: uuid.UUID | None = None):
    _sync_generation_config()
    async for frame in genmod._loom_gen_live(state, gen_id=gen_id):
        yield frame


async def _run_loom_generation_task(**kwargs) -> None:
    _sync_generation_config()
    await genmod._run_loom_generation_task(
        **kwargs,
        mark_model_warm=genmod.mark_model_warm_from_app,
    )


@router.post("/workbench/{session_id}/generations")
async def chat_generation_start(
    request: Request,
    session_id: uuid.UUID,
    prompt: str = Form(""),
    max_tokens: int = Form(_CHAT_DEFAULT_MAX_TOKENS),
    temperature: float = Form(_CHAT_DEFAULT_TEMPERATURE),
    top_p: float = Form(1.0),
    top_k: int = Form(-1),
    min_p: float = Form(0.0),
    presence_penalty: float = Form(0.0),
    frequency_penalty: float = Form(0.0),
    repetition_penalty: float = Form(1.0),
    seed: str = Form(""),
    stop: str = Form(""),
    logprobs: str = Form(""),
    model: str = Form(""),
    api_key_id: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Start a tab-independent generation.

    Inserts a ``ChatGeneration`` row (status=``running``), spawns the streaming
    task, and returns the generation id. The client then opens an SSE connection
    to ``/generations/{gen_id}/events`` to consume the stream. Closing that
    connection no longer aborts the generation — only an explicit POST to
    ``/cancel`` does.
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {"message": "not logged in", "code": "unauthorized"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {"message": "chat session not found", "code": "not_found"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    # Per-chat key selection. Resolve the key to charge against, preferring the
    # form's pick (the workbench picker), then the chat's saved choice, then the
    # user's newest active key. We only *persist* a valid, explicitly-chosen key
    # so a malformed/foreign form value can't clobber a good saved choice — the
    # picker only ever offers the user's own keys, so garbage means tampering.
    chosen_key_id: uuid.UUID | None = None
    raw_key_id = (api_key_id or "").strip()
    if raw_key_id:
        try:
            chosen_key_id = uuid.UUID(raw_key_id)
        except ValueError:
            chosen_key_id = None

    caller = None
    if chosen_key_id is not None:
        caller = await authmod.authed_caller_for_key(session, user.id, chosen_key_id)
        if caller is not None and chat.api_key_id != caller.key_id:
            chat.api_key_id = caller.key_id
    if caller is None and chat.api_key_id is not None:
        caller = await authmod.authed_caller_for_key(session, user.id, chat.api_key_id)
    if caller is None:
        caller = await authmod.primary_authed_caller(session, user.id)
    if caller is None:
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": "no active API key — visit /dashboard to create one",
                    "code": "no_key",
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    requested_model = (model or chat.model or "").strip() or None
    resolved = _resolve_model(request, requested_model)
    if resolved is None:
        return _unknown_model_response(request, requested_model)
    model_id, entry = resolved
    if chat.model != model_id:
        chat.model = model_id

    # Build the upstream body via the same _build_lane_body the multi-run
    # compare lanes use, so the single-pane workbench now honours the full
    # sampling panel (top_p / top_k / min_p / penalties / seed / stop) and
    # requests logprobs when the heatmap toggle is on. One clamp/shape path for
    # single-pane + lanes. clamped_max/temp are read back from the body so the
    # ChatGeneration row + snapshot record exactly what went on the wire.
    upstream_body: dict[str, Any] = _build_lane_body(
        {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "stop": stop,
            "logprobs": logprobs,
        },
        served_model_name=entry.served_model_name,
        prompt=prompt,
    )
    clamped_max = upstream_body["max_tokens"]
    clamped_temp = upstream_body["temperature"]

    # Pre-flight context-window check (ACS-341): reject a prompt whose
    # (BOS-included) token count + max_tokens exceeds this model's context
    # window locally, with the same "context_length_exceeded" 400 the
    # /v1/completions route returns — before we cancel any in-flight run,
    # insert a row, or dispatch to the GPU. Placed before the cancel loop so a
    # rejected request never kills the user's current generation.
    overflow = _context_overflow_message(entry, prompt, clamped_max, settings)
    if overflow is not None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {"message": overflow, "code": "context_length_exceeded"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Concurrency rule: at most one running generation per chat session.
    # If one exists already, cancel it (flip DB + signal the task) then
    # continue with the new one. Matches the "Stop means Stop" semantics
    # decided in the plan's open-questions.
    existing_running = list(
        (
            await session.execute(
                select(ChatGeneration).where(
                    ChatGeneration.session_id == chat.id,
                    ChatGeneration.status == "running",
                )
            )
        )
        .scalars()
        .all()
    )
    for old in existing_running:
        old.status = "cancelled"
        old.ended_at = dt.datetime.now(tz=dt.UTC)
        st = request.app.state.generations.get(old.id)
        if st is not None:
            st.cancel_requested = True
            st.cancel_event.set()

    new_gen = ChatGeneration(
        session_id=chat.id,
        prompt_before=prompt,
        completion_text="",
        model=model_id,
        max_tokens=clamped_max,
        temperature=clamped_temp,
        status="running",
        started_at=dt.datetime.now(tz=dt.UTC),
    )
    session.add(new_gen)
    await session.flush()
    gen_id = new_gen.id

    # Commit the session-scoped writes (cancel of old + insert of new) before
    # we spawn the task. The task uses its own session, so it'd otherwise race
    # the row insert.
    await session.commit()

    state = GenerationState()
    request.app.state.generations[gen_id] = state

    upstream_url = f"{entry.upstream_url.rstrip('/')}/v1/completions"
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None

    asyncio.create_task(
        _run_generation_task(
            app=request.app,
            gen_id=gen_id,
            chat_id=chat.id,
            state=state,
            body=upstream_body,
            upstream_url=upstream_url,
            vllm_api_key=settings.vllm_api_key,
            upstream_timeout_s=settings.upstream_timeout_s,
            model_id=model_id,
            tokenizer_repo=entry.tokenizer_repo,
            hf_token=settings.hf_token,
            caller_key_id=caller.key_id,
            clamped_max=clamped_max,
            clamped_temp=clamped_temp,
            prompt=prompt,
            request_id=request_id,
            ip=ip,
            t0=time.monotonic(),
        )
    )
    return JSONResponse(status_code=200, content={"generation_id": str(gen_id)})


@router.get("/workbench/{session_id}/generations/{gen_id}/events")
async def chat_generation_events(
    request: Request,
    session_id: uuid.UUID,
    gen_id: uuid.UUID,
    since: int | None = None,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """SSE tail for a tab-independent generation.

    Replay-then-tail: first frame is ``event: replay`` carrying whatever text
    has accumulated so far (and a ``cursor`` for resume offset); subsequent
    frames are the raw upstream chunks + status + error frames the streaming
    task fans out. Closes after ``event: done`` (terminal status).
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {"message": "not logged in", "code": "unauthorized"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {"message": "chat session not found", "code": "not_found"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Look up the in-process state, or fall back to the DB row.
    state: GenerationState | None = request.app.state.generations.get(gen_id)
    db_row = (
        await session.execute(
            select(ChatGeneration).where(
                ChatGeneration.id == gen_id,
                ChatGeneration.session_id == chat.id,
            )
        )
    ).scalar_one_or_none()
    if db_row is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "generation not found", "code": "not_found"}},
        )

    # Memory-evicted (or wrapper-restart-orphaned) generation: dump the DB row
    # as a single replay + done sequence, no live tail possible.
    if state is None:

        async def gen_static():
            text = db_row.completion_text or ""
            offset = since or 0
            if offset and offset <= len(text):
                text = text[offset:]
            yield _sse_replay_frame({"text": text, "cursor": len(db_row.completion_text or "")})
            yield _sse_done_frame(
                {
                    "status": db_row.status,
                    "usage": {
                        "prompt_tokens": db_row.n_prompt_tokens or 0,
                        "completion_tokens": db_row.n_completion_tokens or 0,
                    },
                    "error_message": db_row.error_message,
                }
            )

        return StreamingResponse(
            gen_static(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    return StreamingResponse(
        _gen_live(state, since, gen_id=gen_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/workbench/{session_id}/generations/{gen_id}/cancel")
async def chat_generation_cancel(
    request: Request,
    session_id: uuid.UUID,
    gen_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Cancel an in-flight generation.

    Flips the DB row to ``cancelled`` and signals the streaming task; the task
    observes ``state.cancel_requested`` between upstream events and exits
    cleanly (no more chunks broadcast; ``event: done`` carries status
    ``cancelled``).
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {"message": "not logged in", "code": "unauthorized"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {"message": "chat session not found", "code": "not_found"},
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    row = (
        await session.execute(
            select(ChatGeneration).where(
                ChatGeneration.id == gen_id,
                ChatGeneration.session_id == chat.id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "generation not found", "code": "not_found"}},
        )

    if row.status == "running":
        row.status = "cancelled"
        row.ended_at = dt.datetime.now(tz=dt.UTC)
    state = request.app.state.generations.get(gen_id)
    if state is not None:
        state.cancel_requested = True
        # Wake the proxy cold-boot sleep loop so cancellation propagates
        # within milliseconds instead of at the next slice boundary
        # (ACS-132). ``done_event`` wakes subscriber loops; ``cancel_event``
        # wakes the producer.
        state.cancel_event.set()
        state.done_event.set()
    return Response(status_code=204)


# --- compare mode ------------------------------------------------------------
#
# Compare runs one shared prompt against several model/parameter "lanes" at
# once. It reuses the per-generation machinery wholesale — each lane is an
# ordinary ``ChatGeneration`` + ``GenerationState`` + background task, consumed
# via the same ``GET .../generations/{gen_id}/events`` SSE tail and cancelled
# via the same ``POST .../{gen_id}/cancel``. Two deliberate differences from the
# single-pane "Continue" path: (1) lanes spawn with ``persist_session_state=
# False`` so they don't write snapshots or clobber the session's rolling prompt,
# and (2) launching a compare batch does NOT cancel sibling running generations
# (the whole point is concurrency).

_COMPARE_MAX_LANES = 6


def _context_overflow_message(
    entry: Any, prompt: str, max_tokens: int, settings: Settings
) -> str | None:
    """Return a context-window-exceeded message, or ``None`` when the request fits.

    Pre-flight guard for the workbench generate + compare paths (ACS-341). The
    OpenAI-compatible ``/v1/completions`` route runs this check via
    ``services.completions.check_sequence_length`` before dispatching upstream,
    but the workbench paths did not — so an over-long prompt streamed straight to
    vLLM and bounced off the GPU with an opaque ``vllm_context_length`` error
    instead of a clean local rejection. This mirrors that service check: reject
    when ``prompt_tokens + max_tokens`` would exceed the model's context window.

    The prompt-token count already includes BOS (``TokenCounter.count`` uses
    ``add_special_tokens=True`` since ACS-317), so the boundary is exact:
    ``prompt_tokens + max_tokens == max_model_len`` fits and is accepted.

    No-op (returns ``None``) when the registry entry declares no
    ``max_model_len`` (legacy fallback) or the prompt is empty.
    """
    if entry.max_model_len is None or not prompt:
        return None
    counter = get_token_counter(entry.tokenizer_repo, settings.hf_token)
    prompt_tokens = completion_svc.count_prompt_tokens(prompt, counter)
    needed = prompt_tokens + (max_tokens or 1)
    if needed <= entry.max_model_len:
        return None
    return (
        f"Request would exceed model context window: prompt={prompt_tokens} tokens"
        f" + max_tokens={max_tokens} > max_model_len={entry.max_model_len}."
        " Reduce the prompt or max_tokens."
    )


def _build_lane_body(
    entry: dict[str, Any], *, served_model_name: str, prompt: str
) -> dict[str, Any]:
    """Clamp one lane's sampling params into a vLLM ``/v1/completions`` body.

    Defaults mirror the documented API defaults (schemas.CompletionsRequest);
    optional knobs (seed/stop/logprobs) are only included when the lane set
    them, so an unset field stays at the upstream default rather than being
    pinned here.
    """

    def _num(key: str, default: float, lo: float, hi: float) -> float:
        try:
            v = float(entry.get(key, default))
        except (TypeError, ValueError):
            v = default
        return max(lo, min(v, hi))

    try:
        max_tokens = int(entry.get("max_tokens", _CHAT_DEFAULT_MAX_TOKENS))
    except (TypeError, ValueError):
        max_tokens = _CHAT_DEFAULT_MAX_TOKENS
    max_tokens = max(1, min(max_tokens, _CHAT_MAX_MAX_TOKENS))

    body: dict[str, Any] = {
        "model": served_model_name,
        "prompt": prompt,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": _num("temperature", _CHAT_DEFAULT_TEMPERATURE, 0.0, 100.0),
        "top_p": _num("top_p", 1.0, 0.0, 1.0),
        "min_p": _num("min_p", 0.0, 0.0, 1.0),
        "presence_penalty": _num("presence_penalty", 0.0, -2.0, 2.0),
        "frequency_penalty": _num("frequency_penalty", 0.0, -2.0, 2.0),
        "repetition_penalty": _num("repetition_penalty", 1.0, 0.0, 2.0),
    }

    try:
        top_k = int(entry.get("top_k", -1))
    except (TypeError, ValueError):
        top_k = -1
    body["top_k"] = top_k if top_k >= 1 else -1

    seed = entry.get("seed")
    if seed not in (None, ""):
        try:
            body["seed"] = int(seed)
        except (TypeError, ValueError):
            pass

    stop = entry.get("stop")
    if isinstance(stop, str) and stop.strip():
        body["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        body["stop"] = [str(s) for s in stop if str(s)][:4]

    logprobs = entry.get("logprobs")
    if logprobs not in (None, "", False):
        try:
            body["logprobs"] = max(0, min(int(logprobs), 20))
        except (TypeError, ValueError):
            pass

    return body


@router.get("/workbench/{session_id}/compare")
async def compare_open(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
):
    """Legacy compare-surface URL — now a mode of the unified workbench.

    Compare was folded into ``/workbench/{id}`` as a mode/tab (ACS-163), so the
    standalone page is retired. We keep the URL working (tutorial links,
    bookmarks) by 303-redirecting into the session in compare mode. Ownership /
    404 are enforced by ``chat_open`` on the target URL; we mirror its
    unauthenticated → /login redirect here so the auth contract is unchanged.
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url=f"/workbench/{session_id}?mode=compare", status_code=303)


@router.post("/workbench/{session_id}/compare")
async def compare_start(
    request: Request,
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Launch one generation per lane against a shared prompt.

    Body (JSON)::

        {"prompt": "...", "api_key_id": "<uuid>"|"",
         "lanes": [{"model": "...", "max_tokens": 200, "temperature": 0.7,
                    "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                    "presence_penalty": 0, "frequency_penalty": 0,
                    "repetition_penalty": 1.0, "seed": null, "stop": null,
                    "logprobs": null}, ...]}

    Returns ``{"lanes": [{"index", "model", "generation_id"} | {"index","error"}]}``.
    All lanes are billed to the one resolved API key. Unlike "Continue", this
    does not cancel other running generations.
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "not logged in", "code": "unauthorized"}},
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "chat session not found", "code": "not_found"}},
        )

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    prompt = str(payload.get("prompt") or "")
    raw_lanes = payload.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no lanes provided", "code": "bad_request"}},
        )
    lanes = [lane for lane in raw_lanes if isinstance(lane, dict)][:_COMPARE_MAX_LANES]
    if not lanes:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no valid lanes", "code": "bad_request"}},
        )

    # Resolve the single API key all lanes bill against. Mirrors
    # chat_generation_start's resolution (form pick → chat's saved → newest
    # active); intentionally duplicated rather than refactoring the working
    # single-pane endpoint.
    chosen_key_id: uuid.UUID | None = None
    raw_key_id = str(payload.get("api_key_id") or "").strip()
    if raw_key_id:
        try:
            chosen_key_id = uuid.UUID(raw_key_id)
        except ValueError:
            chosen_key_id = None
    caller = None
    if chosen_key_id is not None:
        caller = await authmod.authed_caller_for_key(session, user.id, chosen_key_id)
        if caller is not None and chat.api_key_id != caller.key_id:
            chat.api_key_id = caller.key_id
    if caller is None and chat.api_key_id is not None:
        caller = await authmod.authed_caller_for_key(session, user.id, chat.api_key_id)
    if caller is None:
        caller = await authmod.primary_authed_caller(session, user.id)
    if caller is None:
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": "no active API key — visit /dashboard to create one",
                    "code": "no_key",
                }
            },
        )

    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None

    if prompt:
        chat.prompt_text = prompt

    # Server-side batch identity for this Run-all (ACS-186): the N lane rows
    # share one compare_run_id so the last lane to reach a terminal state can
    # assemble + persist the CompareSnapshot server-side (the barrier in
    # _run_generation_task), surviving a tab close mid-run.
    compare_run_id = uuid.uuid4()

    # First pass: validate models + insert all the ChatGeneration rows, then
    # commit once before spawning any task (the tasks use their own sessions and
    # would otherwise race the inserts).
    pending: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for idx, lane in enumerate(lanes):
        requested_model = str(lane.get("model") or chat.model or "").strip() or None
        resolved = _resolve_model(request, requested_model)
        if resolved is None:
            results.append(
                {
                    "index": idx,
                    "error": {
                        "message": f"unknown model: {requested_model}",
                        "code": "unknown_model",
                    },
                }
            )
            continue
        model_id, entry = resolved
        body = _build_lane_body(lane, served_model_name=entry.served_model_name, prompt=prompt)
        # Pre-flight context-window check per lane (ACS-341): a lane whose
        # prompt + max_tokens exceeds *that lane model's* context window is
        # marked with a clean "context_length_exceeded" error and skipped,
        # rather than dispatched to the GPU. Mirrors the unknown-model per-lane
        # error above; sibling lanes still launch.
        overflow = _context_overflow_message(entry, prompt, int(body["max_tokens"]), settings)
        if overflow is not None:
            results.append(
                {
                    "index": idx,
                    "error": {"message": overflow, "code": "context_length_exceeded"},
                }
            )
            continue
        # Persist the clamped per-lane sampling config (ACS-186) so the
        # server-side barrier can rebuild the snapshot lane faithfully — the base
        # ChatGeneration columns only carry model/max_tokens/temperature, but a
        # snapshot lane also needs top_p/top_k/min_p/penalties/seed/stop. Also
        # stash the launch ``index`` so the barrier can order lanes correctly
        # (rows have no ordinal column and started_at can tie in this loop).
        compare_config = {
            "index": idx,
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "min_p": body.get("min_p"),
            "presence_penalty": body.get("presence_penalty"),
            "frequency_penalty": body.get("frequency_penalty"),
            "repetition_penalty": body.get("repetition_penalty"),
            "seed": body.get("seed"),
            "stop": body.get("stop"),
        }
        new_gen = ChatGeneration(
            session_id=chat.id,
            prompt_before=prompt,
            completion_text="",
            model=model_id,
            max_tokens=int(body["max_tokens"]),
            temperature=float(body["temperature"]),
            status="running",
            started_at=dt.datetime.now(tz=dt.UTC),
            compare_run_id=compare_run_id,
            compare_config=compare_config,
        )
        session.add(new_gen)
        await session.flush()
        pending.append(
            {
                "index": idx,
                "gen": new_gen,
                "model_id": model_id,
                "entry": entry,
                "body": body,
            }
        )

    if not pending:
        # Every lane failed model resolution — nothing to run.
        return JSONResponse(status_code=400, content={"lanes": results})

    await session.commit()

    for item in pending:
        gen = item["gen"]
        entry = item["entry"]
        gen_id = gen.id
        state = GenerationState()
        request.app.state.generations[gen_id] = state
        upstream_url = f"{entry.upstream_url.rstrip('/')}/v1/completions"
        asyncio.create_task(
            _run_generation_task(
                app=request.app,
                gen_id=gen_id,
                chat_id=chat.id,
                state=state,
                body=item["body"],
                upstream_url=upstream_url,
                vllm_api_key=settings.vllm_api_key,
                upstream_timeout_s=settings.upstream_timeout_s,
                model_id=item["model_id"],
                tokenizer_repo=entry.tokenizer_repo,
                hf_token=settings.hf_token,
                caller_key_id=caller.key_id,
                clamped_max=int(item["body"]["max_tokens"]),
                clamped_temp=float(item["body"]["temperature"]),
                prompt=prompt,
                request_id=request_id,
                ip=ip,
                t0=time.monotonic(),
                persist_session_state=False,
                compare_run_id=compare_run_id,
            )
        )
        results.append(
            {"index": item["index"], "model": item["model_id"], "generation_id": str(gen_id)}
        )

    results.sort(key=lambda r: r["index"])
    return JSONResponse(status_code=200, content={"lanes": results})


# --- compare snapshots (ACS-180, server-side barrier ACS-186) ---------------
#
# One saved snapshot per Compare "Run all". As of ACS-186 the primary writer is
# the *server-side barrier* in ``_run_generation_task``: each Run-all stamps a
# shared ``compare_run_id`` on its lane rows + persists the per-lane sampling
# config, and the last lane to reach a terminal state assembles + writes the
# snapshot. That survives the user closing the tab mid-run (the #158 limitation).
#
# The client POST below is now a FALLBACK / idempotent path: the browser still
# POSTs its assembled snapshot when Run-all settles, but the handler dedupes
# against the server barrier by resolving this session's most-recent batch's
# ``compare_run_id`` (from ChatGeneration) and no-op'ing if a snapshot for that
# run already exists. The client HTML is frozen (no run id in its payload), so
# the dedup is resolved entirely server-side. The fallback still matters for the
# race where the client settles before the barrier commits, or a legacy client.

# Fields we accept per lane; anything else in the client payload is dropped.
_COMPARE_LANE_NUM_FIELDS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
)


def _sanitize_compare_lane(lane: Any) -> dict[str, Any]:
    """Whitelist one lane dict from the client into the shape we persist.

    Numbers are coerced (bad values → None); text fields truncated so a rogue
    client can't bloat the JSONB row. Only the documented keys survive.
    """
    if not isinstance(lane, dict):
        lane = {}
    out: dict[str, Any] = {}
    out["model"] = (str(lane.get("model") or "").strip() or None) if lane.get("model") else None
    for key in _COMPARE_LANE_NUM_FIELDS:
        val = lane.get(key)
        if val is None or val == "":
            out[key] = None
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            out[key] = None
            continue
        # max_tokens/top_k/seed are integers; the rest are floats.
        out[key] = int(num) if key in ("max_tokens", "top_k", "seed") else num
    stop = lane.get("stop")
    out["stop"] = str(stop)[:200] if isinstance(stop, str) and stop != "" else None
    ct = lane.get("completion_text")
    out["completion_text"] = str(ct)[:100_000] if ct else ""
    out["cancelled"] = bool(lane.get("cancelled"))
    return out


async def _compare_snapshots(
    session: AsyncSession, session_id: uuid.UUID
) -> list[CompareSnapshot]:
    """This session's compare snapshots, newest first, capped."""
    return list(
        (
            await session.execute(
                select(CompareSnapshot)
                .where(CompareSnapshot.session_id == session_id)
                # id.desc() tiebreaker so list order matches the prune query's
                # keep-set ordering when two rows share a ts.
                .order_by(CompareSnapshot.ts.desc(), CompareSnapshot.id.desc())
                .limit(_COMPARE_SNAPSHOT_LIMIT)
            )
        )
        .scalars()
        .all()
    )


def _compare_snapshot_json(snap: CompareSnapshot) -> dict[str, Any]:
    return {
        "id": snap.id,
        "ts": snap.ts.isoformat(),
        "prompt": snap.prompt,
        "n_lanes": snap.n_lanes,
        "lanes": snap.lanes,
        # Batch identity (ACS-186); NULL on legacy #158 rows. Exposed so a client
        # can tell a server-barrier-written snapshot from a NULL-run legacy one.
        "compare_run_id": str(snap.compare_run_id) if snap.compare_run_id else None,
    }


@router.get("/workbench/{session_id}/compare/snapshots")
async def compare_snapshots_list(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """List this session's compare-run snapshots (newest first), scoped to the
    owning user — same ownership gate as every other /workbench/{id}/* route."""
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "not logged in", "code": "unauthorized"}},
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "chat session not found", "code": "not_found"}},
        )
    snaps = await _compare_snapshots(session, chat.id)
    return JSONResponse(
        status_code=200,
        content={"snapshots": [_compare_snapshot_json(s) for s in snaps]},
    )


@router.post("/workbench/{session_id}/compare/snapshots")
async def compare_snapshot_create(
    request: Request,
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Persist ONE compare-run snapshot posted by the client after Run-all.

    Body::

        {"prompt": "...", "lanes": [{"model", "max_tokens", ..., "stop",
                                     "completion_text", "cancelled"}, ...]}

    Ownership-scoped (only the session's owner can write to it). Prunes the
    oldest rows past ``_COMPARE_SNAPSHOT_LIMIT`` so history stays bounded, like
    single-pane. Returns the created snapshot's JSON.
    """
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "not logged in", "code": "unauthorized"}},
        )
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "chat session not found", "code": "not_found"}},
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    raw_lanes = payload.get("lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no lanes provided", "code": "bad_request"}},
        )
    lanes = [_sanitize_compare_lane(lane) for lane in raw_lanes[:_COMPARE_MAX_LANES]]
    prompt = str(payload.get("prompt") or "")

    # Dedup against the server-side barrier (ACS-186). Resolve this session's
    # most-recent compare batch's run id from ChatGeneration; if the barrier (or
    # a prior client POST) already wrote a snapshot for that run, no-op and
    # return the existing row instead of double-writing. The client HTML can't
    # send a run id, so the whole dedup is resolved here server-side.
    latest_run_id = (
        await session.execute(
            select(ChatGeneration.compare_run_id)
            .where(
                ChatGeneration.session_id == chat.id,
                ChatGeneration.compare_run_id.is_not(None),
            )
            .order_by(ChatGeneration.started_at.desc(), ChatGeneration.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if latest_run_id is not None:
        existing = (
            await session.execute(
                select(CompareSnapshot).where(
                    CompareSnapshot.compare_run_id == latest_run_id
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            # Server barrier (or an earlier POST) already persisted this batch —
            # idempotent no-op.
            return JSONResponse(status_code=200, content=_compare_snapshot_json(existing))

    snap = CompareSnapshot(
        session_id=chat.id,
        prompt=prompt,
        lanes=lanes,
        n_lanes=len(lanes),
        # Stamp the resolved run id so the UNIQUE constraint dedupes a racing
        # barrier write (and vice-versa). NULL only if this session has no
        # compare batch rows at all (a legacy/degenerate client POST).
        compare_run_id=latest_run_id,
    )
    session.add(snap)
    try:
        await session.flush()
    except IntegrityError:
        # A barrier write for the same run id landed between our SELECT and
        # flush — roll back and return the winner. This is the last dedup guard.
        await session.rollback()
        existing = (
            await session.execute(
                select(CompareSnapshot).where(
                    CompareSnapshot.compare_run_id == latest_run_id
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return JSONResponse(status_code=200, content=_compare_snapshot_json(existing))
        raise

    # Prune oldest beyond the cap (ownership already enforced by session_id).
    keep_ids = (
        select(CompareSnapshot.id)
        .where(CompareSnapshot.session_id == chat.id)
        .order_by(CompareSnapshot.ts.desc(), CompareSnapshot.id.desc())
        .limit(_COMPARE_SNAPSHOT_LIMIT)
    )
    await session.execute(
        delete(CompareSnapshot).where(
            CompareSnapshot.session_id == chat.id,
            CompareSnapshot.id.not_in(keep_ids),
        )
    )
    await session.commit()
    await session.refresh(snap)
    return JSONResponse(status_code=201, content=_compare_snapshot_json(snap))


# --- chat exports ------------------------------------------------------------


@router.get("/workbench/{session_id}/export.txt")
async def chat_export_txt(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat.title)[:60]
    return Response(
        content=chat.prompt_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_title or "chat"}.txt"'},
    )


def _session_to_jsonl_record(chat: ChatSession, snapshots: list[ChatSnapshot]) -> dict[str, Any]:
    return {
        "id": str(chat.id),
        "title": chat.title,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
        "last_max_tokens": chat.last_max_tokens,
        "last_temperature": chat.last_temperature,
        "prompt_text": chat.prompt_text,
        "snapshots": [
            {
                "ts": s.ts.isoformat(),
                "prompt_before": s.prompt_before,
                "completion_text": s.completion_text,
                "n_completion": s.n_completion,
                "max_tokens": s.max_tokens,
                "temperature": s.temperature,
                "cancelled": s.cancelled,
                "model": s.model,
            }
            for s in snapshots
        ],
    }


@router.get("/workbench/{session_id}/export.jsonl")
async def chat_export_jsonl(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    chat = await _get_owned_chat_session(session, session_id, user.id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    snaps = list(
        (
            await session.execute(
                select(ChatSnapshot)
                .where(ChatSnapshot.session_id == chat.id)
                .order_by(ChatSnapshot.ts.asc())
            )
        )
        .scalars()
        .all()
    )
    import json as _json

    body = _json.dumps(_session_to_jsonl_record(chat, snaps)) + "\n"
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat.title)[:60]
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{safe_title or "chat"}.jsonl"'},
    )
