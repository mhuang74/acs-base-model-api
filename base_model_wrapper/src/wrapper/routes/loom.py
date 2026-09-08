"""Loom routes — tree/branching exploration as a first-class saved object.

ACS-148 follow-up (on review): a loom is now its own document, like a
workbench chat — it has its own ``/loom`` list, its own ``/loom/<id>`` URL, and
is created / named / renamed / listed / deleted on its own, rather than being a
1:1 sub-view of a ``ChatSession``.

This module owns every ``/loom*`` route. It deliberately reuses the workbench
module's model/key helpers and loom clamps (imported below) so the workbench
diff stays near-zero and there's a single source of truth for the sampling
knobs. The additive ``LoomGenerationState`` / ``loom_chunk`` streaming path in
``workbench_generations`` is unchanged — only its owner key moved from
``session_id`` to ``loom_id``.

Old URL: ``/workbench/{session_id}/loom`` now 307-redirects to
``/loom/{session_id}`` — backfilled looms reuse the origin session id, and
``_get_or_create_loom_for_session`` lazily mints one (with the same id) for a
session that had the loom page open but never persisted a node, so the redirect
is idempotent (no duplicate looms on repeat hits).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import web_auth as webauth
from .. import workbench_generations as genmod
from ..db import get_session
from ..dependencies import get_settings
from ..model_resolution import _resolve_model, _unknown_model_response
from ..models import ChatSession, Loom, LoomNode, User
from ..settings import Settings
from . import workbench as workbench_routes
from .workbench import (
    _CHAT_DEFAULT_TEMPERATURE,
    _LOOM_DEFAULT_LOGPROBS,
    _LOOM_DEFAULT_MAX_TOKENS,
    _LOOM_DEFAULT_N,
    _LOOM_MAX_LOGPROBS,
    _LOOM_MAX_MAX_TOKENS,
    _LOOM_MAX_N,
    _LOOM_MAX_NODES,
    SSE_HEADERS,
    LoomGenerationState,
    _loom_gen_live,  # workbench wrapper: syncs gen config + marks model warm
    _run_loom_generation_task,  # workbench wrapper: same
    _select_workbench_model_id,
    _workbench_keys,
    templates,
)

# NB: read ``_UI_HIDDEN_MODEL_IDS`` off the workbench module at call time, not as
# a by-value import. ``main._sync_workbench_compat()`` rebinds
# ``workbench_routes._UI_HIDDEN_MODEL_IDS`` at runtime to propagate a monkeypatched
# override; a by-value import here would keep the stale empty frozenset and let a
# hidden model leak into the loom dropdown while the workbench correctly hides it.

router = APIRouter()

_LOOM_SIDEBAR_LIMIT = 50

_sse_done_frame = genmod._sse_done_frame


# --- ownership + listing helpers ---------------------------------------------


async def _user_looms(session: AsyncSession, user_id: uuid.UUID) -> list[Loom]:
    """Sidebar listing: this user's non-archived looms, newest activity first."""
    q = (
        select(Loom)
        .where(Loom.user_id == user_id, Loom.archived_at.is_(None))
        .order_by(Loom.updated_at.desc())
        .limit(_LOOM_SIDEBAR_LIMIT)
    )
    return list((await session.execute(q)).scalars().all())


async def _get_owned_loom(
    session: AsyncSession, loom_id: uuid.UUID, user_id: uuid.UUID
) -> Loom | None:
    """Fetch a loom by id, only if it belongs to this user.

    Mismatched user_id returns None (caller raises 404) so we don't leak the
    existence of other users' loom ids — mirrors ``_get_owned_chat_session``.
    Archived (deleted) looms are treated as gone.
    """
    return (
        await session.execute(
            select(Loom).where(
                Loom.id == loom_id,
                Loom.user_id == user_id,
                Loom.archived_at.is_(None),
            )
        )
    ).scalar_one_or_none()


async def _get_or_create_loom_for_session(
    session: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID
) -> Loom | None:
    """Resolve the loom for a legacy ``/workbench/{session_id}/loom`` link.

    Backfilled looms reuse their origin session's id, so we look up a loom whose
    id == session_id first. If none exists (the user opened the loom page but
    never persisted a node), lazily mint one **with that same id** so the old
    URL stays a stable, idempotent redirect target. Returns None if the session
    isn't owned by this user (caller raises 404).
    """
    # Look up by id ignoring archived_at: an archived loom still occupies the
    # PK, so we must return it (not mint a duplicate that would PK-collide). The
    # loom page itself will 404 on the archived row via _get_owned_loom.
    existing = (
        await session.execute(select(Loom).where(Loom.id == session_id, Loom.user_id == user_id))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    chat = (
        await session.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
    ).scalar_one_or_none()
    if chat is None:
        return None
    loom = Loom(
        id=chat.id,
        user_id=user_id,
        title=(chat.title or "").strip() or "Untitled loom",
        model=chat.model,
        api_key_id=chat.api_key_id,
    )
    session.add(loom)
    await session.flush()
    return loom


def _normalize_newlines(text: str) -> str:
    """Collapse CRLF / lone CR to LF (ACS-339).

    A browser textarea round-trips ``\\n`` as ``\\r\\n`` on submit, which grows
    the node and — because CRLF tokenises differently from LF — silently changes
    the token sequence of every branch generated below it. We normalise on the
    server for every form-sourced node write so stored text stays LF, and so the
    ACS-338 "did the text actually change?" comparison is done on normalised text
    (a raw CRLF-vs-stored-LF compare would always read as changed).
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _next_sibling_position(
    session: AsyncSession, loom_id: uuid.UUID, parent_id: uuid.UUID | None
) -> int:
    """Next 0-based ``position`` ordinal among a parent's existing children (ACS-340).

    ``max(position) + 1`` over the siblings sharing ``parent_id`` (roots share the
    NULL parent). Empty → 0. Not a unique constraint: two concurrent generations
    under one parent can read the same max and collide; that rare case degrades to
    ``created_at`` order rather than failing an insert.
    """
    cond = LoomNode.parent_id.is_(None) if parent_id is None else (LoomNode.parent_id == parent_id)
    max_pos = (
        await session.execute(
            select(func.max(LoomNode.position)).where(LoomNode.loom_id == loom_id, cond)
        )
    ).scalar_one_or_none()
    return 0 if max_pos is None else int(max_pos) + 1


async def _loom_nodes(session: AsyncSession, loom_id: uuid.UUID) -> list[LoomNode]:
    # Order by (position, created_at): position gives stable sibling order even
    # when a generation batch shares a created_at to the microsecond (ACS-340);
    # created_at is the tiebreaker for the rare concurrent-batch position clash.
    rows = await session.execute(
        select(LoomNode)
        .where(LoomNode.loom_id == loom_id)
        .order_by(LoomNode.position.asc(), LoomNode.created_at.asc())
    )
    return list(rows.scalars().all())


def _loom_node_dict(node: LoomNode) -> dict[str, Any]:
    """JSON shape the loom front-end consumes for one node."""
    return {
        "id": str(node.id),
        "parent_id": str(node.parent_id) if node.parent_id else None,
        "text": node.text,
        "model": node.model,
        "seed": node.seed,
        "position": node.position,
        "logprobs": node.logprobs,
        "created_at": node.created_at.isoformat() if node.created_at else None,
    }


def _loom_prefix_for_node(nodes_by_id: dict[uuid.UUID, LoomNode], node_id: uuid.UUID) -> str:
    """Reconstruct the full text context up to and including ``node_id`` by
    walking parent links (root → … → node) and concatenating their text."""
    chain: list[str] = []
    cur: uuid.UUID | None = node_id
    seen: set[uuid.UUID] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        node = nodes_by_id.get(cur)
        if node is None:
            break
        chain.append(node.text)
        cur = node.parent_id
    return "".join(reversed(chain))


def _touch(loom: Loom) -> None:
    loom.updated_at = dt.datetime.now(dt.UTC)


def _subtree_ids(nodes: list[LoomNode], root_id: uuid.UUID) -> set[uuid.UUID]:
    """The set of node ids in the subtree rooted at ``root_id`` (inclusive).

    Walks the parent→children map so callers can reason about which nodes a
    topology mutation touches (delete/split/edit all change the reconstructed
    context of everything in the affected node's subtree).
    """
    children: dict[uuid.UUID, list[uuid.UUID]] = {}
    for nd in nodes:
        if nd.parent_id is not None:
            children.setdefault(nd.parent_id, []).append(nd.id)
    subtree: set[uuid.UUID] = set()
    stack = [root_id]
    while stack:
        nid = stack.pop()
        if nid in subtree:
            continue
        subtree.add(nid)
        stack.extend(children.get(nid, []))
    return subtree


def _running_gen_in_subtree(request: Request, loom_id: uuid.UUID, subtree: set[uuid.UUID]) -> bool:
    """True if a running loom generation targets a node inside ``subtree``.

    Mirrors ``loom_delete_node``'s guard: a generate task writes children under
    its ``parent_id`` on completion. If that parent (or an ancestor of it) is
    about to be reparented / mutated / deleted, the child insert can hit a
    swallowed FK violation or land under a node whose text has changed —
    silently corrupting the branch's reconstructed context.
    """
    loom_gens = getattr(request.app.state, "loom_generations", {})
    for st in list(loom_gens.values()):
        if (
            getattr(st, "loom_id", None) == loom_id
            and getattr(st, "status", None) == "running"
            and getattr(st, "parent_id", None) in subtree
        ):
            return True
    return False


# --- list / create / rename / delete -----------------------------------------


@router.get("/loom")
async def loom_index(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """The loom list page: New loom + Recent looms (mirrors /workbench)."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    looms = await _user_looms(session, user.id)
    resp = templates.TemplateResponse(
        request,
        "loom_list.html",
        {"user": user, "looms": looms},
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@router.post("/loom/new")
async def loom_new(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Create a fresh empty loom and redirect to it."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    loom = Loom(user_id=user.id, title="Untitled loom")
    session.add(loom)
    await session.flush()
    return RedirectResponse(url=f"/loom/{loom.id}", status_code=303)


@router.post("/loom/{loom_id}/rename")
async def loom_rename(
    request: Request,
    loom_id: uuid.UUID,
    title: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    # Renames in place via fetch, same as the workbench rename (ACS-254): the
    # old full-reload redirect killed a live branch stream and any unsaved
    # node-editor/seed text. Redirect kept as the no-JS fallback.
    wants_json = "application/json" in request.headers.get("accept", "")
    if user is None:
        if wants_json:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        return RedirectResponse(url="/login", status_code=303)
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        raise HTTPException(status_code=404, detail="loom not found")
    loom.title = (title.strip() or "Untitled loom")[:120]
    _touch(loom)
    if wants_json:
        return JSONResponse({"ok": True, "title": loom.title})
    return RedirectResponse(url=f"/loom/{loom.id}", status_code=303)


@router.post("/loom/{loom_id}/delete")
async def loom_delete(
    loom_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        raise HTTPException(status_code=404, detail="loom not found")
    loom.archived_at = dt.datetime.now(dt.UTC)
    return RedirectResponse(url="/loom", status_code=303)


# --- old-URL redirect --------------------------------------------------------


@router.get("/workbench/{session_id}/loom")
async def loom_legacy_redirect(
    session_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Redirect the retired per-session loom URL to the standalone loom.

    Keeps old links / bookmarks / tutorial deep-links working. Idempotent: a
    backfilled loom reuses the session id, and a missing one is lazily created
    with that same id (see ``_get_or_create_loom_for_session``).
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    loom = await _get_or_create_loom_for_session(session, session_id, user.id)
    if loom is None:
        raise HTTPException(status_code=404, detail="loom not found")
    return RedirectResponse(url=f"/loom/{loom.id}", status_code=307)


# --- the loom page + tree endpoints ------------------------------------------


@router.get("/loom/{loom_id}")
async def loom_page(
    request: Request,
    loom_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Render the loom (tree exploration) page for an owned loom."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        raise HTTPException(status_code=404, detail="loom not found")
    keys = await _workbench_keys(session, user.id)
    saved_key_id = str(loom.api_key_id) if loom.api_key_id else None
    selectable_ids = {k["id"] for k in keys}
    if saved_key_id in selectable_ids:
        selected_key_id = saved_key_id
    elif keys:
        selected_key_id = keys[0]["id"]
    else:
        selected_key_id = None

    registry = request.app.state.models
    available_models: list[dict[str, Any]] = []
    model_labels: dict[str, str] = {}
    for m in registry.values():
        label = (m.served_model_name or m.model_id).rsplit("/", 1)[-1]
        model_labels[m.model_id] = label
        if m.status != "live" or m.model_id in workbench_routes._UI_HIDDEN_MODEL_IDS:
            continue
        available_models.append({"id": m.model_id, "label": label})
    selected_model_id = _select_workbench_model_id(
        loom.model, available_models, request.app.state.default_model_id
    )
    nodes = await _loom_nodes(session, loom.id)
    resp = templates.TemplateResponse(
        request,
        "loom.html",
        {
            "user": user,
            "loom": loom,
            "keys": keys,
            "selected_key_id": selected_key_id,
            "no_key": not keys,
            "available_models": available_models,
            "selected_model_id": selected_model_id,
            "model_labels": model_labels,
            "nodes": [_loom_node_dict(n) for n in nodes],
            "loom_defaults": {
                "n": _LOOM_DEFAULT_N,
                "max_n": _LOOM_MAX_N,
                "max_tokens": _LOOM_DEFAULT_MAX_TOKENS,
                "logprobs": _LOOM_DEFAULT_LOGPROBS,
            },
        },
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@router.get("/loom/{loom_id}/tree")
async def loom_tree(
    loom_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """JSON snapshot of the whole loom tree (for reload / poll)."""
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    nodes = await _loom_nodes(session, loom.id)
    return JSONResponse(status_code=200, content={"nodes": [_loom_node_dict(n) for n in nodes]})


@router.post("/loom/{loom_id}/root")
async def loom_create_root(
    request: Request,
    loom_id: uuid.UUID,
    text: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Create (or overwrite) the seed/root node for a loom from typed text.

    A loom needs a root before it can branch. The root is a hand-typed prompt,
    not a generation, so it's a plain insert (no upstream call). We keep at most
    one root per loom for the MVP — re-posting replaces the previous root and
    its descendants (CASCADE) so the user can restart cleanly.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    # Guard: replacing the root deletes the whole tree via CASCADE. If any
    # generation is still running in this loom, its completion would insert
    # children under a now-deleted parent (FK violation, swallowed → branches
    # silently lost). Refuse while a generate is in flight, mirroring
    # loom_delete_node's node_busy guard. (Any running gen blocks — create_root
    # nukes the entire tree, so no per-subtree check is needed.)
    loom_gens = getattr(request.app.state, "loom_generations", {})
    for st in list(loom_gens.values()):
        if (
            getattr(st, "loom_id", None) == loom.id
            and getattr(st, "status", None) == "running"
        ):
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": "a generation is still running in this loom; wait for it or cancel it before replacing the root",
                        "code": "loom_busy",
                    }
                },
            )
    existing_roots = (
        (
            await session.execute(
                select(LoomNode).where(
                    LoomNode.loom_id == loom.id,
                    LoomNode.parent_id.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    for r in existing_roots:
        await session.delete(r)
    await session.flush()
    # One root per loom (existing roots just deleted) → ordinal 0. Normalise
    # newlines so a hand-typed seed with CRLF stores as LF (ACS-339).
    root = LoomNode(loom_id=loom.id, parent_id=None, text=_normalize_newlines(text), position=0)
    session.add(root)
    _touch(loom)
    await session.flush()
    return JSONResponse(status_code=200, content={"node": _loom_node_dict(root)})


@router.post("/loom/{loom_id}/nodes/{node_id}/delete")
async def loom_delete_node(
    request: Request,
    loom_id: uuid.UUID,
    node_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Delete a node and (via CASCADE) its whole subtree."""
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    node = (
        await session.execute(
            select(LoomNode).where(
                LoomNode.id == node_id,
                LoomNode.loom_id == loom.id,
            )
        )
    ).scalar_one_or_none()
    if node is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    # Guard: refuse to delete a node that an in-flight generate is about to write
    # children under. The generate task writes LoomNode children with
    # parent_id=<target> on completion; if the target (or an ancestor of it) is
    # deleted first, the CASCADE removes the parent and the child insert hits an
    # FK violation — branches silently lost.
    nodes = await _loom_nodes(session, loom.id)
    subtree = _subtree_ids(nodes, node_id)
    if _running_gen_in_subtree(request, loom.id, subtree):
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "message": "a generation is still running under this node",
                    "code": "node_busy",
                }
            },
        )
    await session.delete(node)
    _touch(loom)
    return Response(status_code=204)


# --- Tier-2 power-user editing surface (ACS-166) -----------------------------
#
# Text convention (documented in docs/workbench/loom.md): a node's ``text`` is a
# *delta*, and ``_loom_prefix_for_node`` reconstructs context by concatenating
# ancestor deltas root→node. So any op that mutates a node's text invalidates
# that node's stored ``logprobs`` (they were aligned to the pre-edit text) — we
# clear ``logprobs`` on every mutated node rather than attempt a fiddly (and
# error-prone) char-cursor→token-boundary re-alignment. The heatmap JS renders
# a null/empty payload as plain text, so this degrades cleanly.


@router.post("/loom/{loom_id}/nodes/{node_id}/edit")
async def loom_edit_node(
    request: Request,
    loom_id: uuid.UUID,
    node_id: uuid.UUID,
    text: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Edit a node's text in place (reference-loom "edit node" convention).

    Mutates the node's own delta text; its descendants keep their own deltas, so
    their reconstructed context shifts by exactly the edit — which is the point.
    The node's stored ``logprobs`` are cleared (they were aligned to the old
    text). Guarded against in-flight generation the same way as delete: editing a
    node whose subtree a running generate targets would change the prefix under
    which its branches are about to be written.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    nodes = await _loom_nodes(session, loom.id)
    nodes_by_id = {nd.id: nd for nd in nodes}
    node = nodes_by_id.get(node_id)
    if node is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    subtree = _subtree_ids(nodes, node_id)
    if _running_gen_in_subtree(request, loom.id, subtree):
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "message": "a generation is still running under this node",
                    "code": "node_busy",
                }
            },
        )
    # ACS-339: normalise CRLF→LF first (the textarea round-trips \n as \r\n on
    # submit). ACS-338: only clear logprobs when the token sequence actually
    # changed — compare the *normalised* incoming text against the stored text.
    # A no-op save (open editor, change nothing) is byte-identical after
    # normalisation, so its logprobs are preserved; comparing raw CRLF input
    # against stored LF would falsely read as changed and drop them anyway.
    new_text = _normalize_newlines(text)
    if new_text != node.text:
        node.text = new_text
        node.logprobs = None  # stale: aligned to the pre-edit text
    _touch(loom)
    await session.flush()
    return JSONResponse(status_code=200, content={"node": _loom_node_dict(node)})


@router.post("/loom/{loom_id}/nodes/{node_id}/split")
async def loom_split_node(
    request: Request,
    loom_id: uuid.UUID,
    node_id: uuid.UUID,
    offset: int = Form(...),
    text: str | None = Form(None),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Split a node's text at a character ``offset`` into parent + child.

    The signature loom op. Given node N = ``"ABCD"`` split at offset 2:

      * N keeps ``"AB"``;
      * a new child M is inserted with ``"CD"`` (``parent_id = N``);
      * **N's existing children are reparented onto M** — critical: they
        continued from ``"ABCD"``, so leaving them under N would silently shorten
        their reconstructed prefix to ``"AB…"``. Moving them under M preserves
        every descendant's full context byte-for-byte.

    ``text`` (optional) lets the client commit an *edited* node body and the
    split point in one atomic request — so a "split while editing" can't leave a
    partial write (text saved, split failed). When present, ``text`` replaces the
    node body and ``offset`` indexes into it.

    Both N and M get ``logprobs`` cleared (N's text changed; M's slice has no
    aligned payload) and M's ``seed`` is cleared — the slice was not produced by
    that seed, so keeping it would misrepresent reproducibility. The split point
    selects M so branching continues from the tail. Offsets clamp to
    ``[0, len(text)]``.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    nodes = await _loom_nodes(session, loom.id)
    nodes_by_id = {nd.id: nd for nd in nodes}
    node = nodes_by_id.get(node_id)
    if node is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    if len(nodes) >= _LOOM_MAX_NODES:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": f"loom node cap reached ({_LOOM_MAX_NODES})", "code": "node_cap"}},
        )
    subtree = _subtree_ids(nodes, node_id)
    if _running_gen_in_subtree(request, loom.id, subtree):
        return JSONResponse(
            status_code=409,
            content={
                "error": {"message": "a generation is still running under this node", "code": "node_busy"}
            },
        )
    # If the client sent an edited body, split within that (atomic edit+split).
    # Normalise CRLF→LF on the (possibly edited) body for the same reason as
    # loom_edit_node (ACS-339): keep stored node text LF so downstream branches
    # tokenise identically.
    full = _normalize_newlines(text) if text is not None else (node.text or "")
    cut = max(0, min(int(offset), len(full)))
    head, tail = full[:cut], full[cut:]

    # M is the tail child under N; after reparenting it becomes N's sole direct
    # child, so its ordinal among N's children is next-after-existing (ACS-340).
    child_position = await _next_sibling_position(session, loom.id, node.id)
    # New child M carries the tail. Keep the model label (same model's output),
    # but DON'T inherit the seed: the tail is a text slice, not a generation
    # reproducible from that seed — a seed badge on M would mislead.
    child = LoomNode(
        loom_id=loom.id,
        parent_id=node.id,
        text=tail,
        model=node.model,
        seed=None,
        position=child_position,
    )
    session.add(child)
    await session.flush()  # need child.id before reparenting

    # Reparent N's *existing* children onto M (exclude M itself). They keep their
    # own ``position`` values, so their sibling order is preserved under M.
    for nd in nodes:
        if nd.parent_id == node.id and nd.id != child.id:
            nd.parent_id = child.id

    node.text = head
    node.logprobs = None  # text changed
    _touch(loom)
    await session.flush()
    return JSONResponse(
        status_code=200,
        content={"node": _loom_node_dict(node), "child": _loom_node_dict(child)},
    )


@router.post("/loom/{loom_id}/nodes/{node_id}/sibling")
async def loom_new_sibling(
    request: Request,
    loom_id: uuid.UUID,
    node_id: uuid.UUID,
    text: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Insert a new empty (or typed) sibling of ``node_id`` under the same parent.

    Lets a user hand-write an alternative continuation at a branch point instead
    of generating one. Restricted to **non-root** nodes: a root has no parent, and
    ``loom_create_root`` treats "one root per loom" as an invariant, so we don't
    mint a second root here (create a fresh loom for a different seed instead).
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    nodes = await _loom_nodes(session, loom.id)
    nodes_by_id = {nd.id: nd for nd in nodes}
    node = nodes_by_id.get(node_id)
    if node is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    if node.parent_id is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "cannot add a sibling to the root; create a new loom for a different seed",
                    "code": "root_no_sibling",
                }
            },
        )
    if len(nodes) >= _LOOM_MAX_NODES:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": f"loom node cap reached ({_LOOM_MAX_NODES})", "code": "node_cap"}},
        )
    # Race guard, for parity with edit/split/delete (the "ownership-scoped AND
    # race-guarded" rule): a sibling lands under this node's parent, which is
    # exactly where a running generate writes its own children — refuse while a
    # generation targets the shared parent's subtree so the two inserts don't
    # interleave against a tree the user is also reshaping.
    subtree = _subtree_ids(nodes, node.parent_id)
    if _running_gen_in_subtree(request, loom.id, subtree):
        return JSONResponse(
            status_code=409,
            content={
                "error": {"message": "a generation is still running under this node", "code": "node_busy"}
            },
        )
    # Hand-written text, not a generation → model=None (matches a hand-typed
    # root), so the UI doesn't mislabel it as model-produced. Normalise newlines
    # (ACS-339) and give it the next sibling ordinal under the shared parent
    # (ACS-340) so it sorts stably after existing siblings.
    sib_position = await _next_sibling_position(session, loom.id, node.parent_id)
    sib = LoomNode(
        loom_id=loom.id,
        parent_id=node.parent_id,
        text=_normalize_newlines(text),
        model=None,
        position=sib_position,
    )
    session.add(sib)
    _touch(loom)
    await session.flush()
    return JSONResponse(status_code=200, content={"node": _loom_node_dict(sib)})


@router.get("/loom/{loom_id}/export")
async def loom_export(
    loom_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Export the whole loom tree as a downloadable JSON document.

    Shape: ``{"loom": {"title", "model"}, "nodes": [<node dicts>]}`` — the same
    per-node shape the front-end already consumes, so import is a mirror image.
    Served as an attachment so the browser saves a file.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    nodes = await _loom_nodes(session, loom.id)
    doc = {
        "version": 1,
        "loom": {"title": loom.title, "model": loom.model},
        "nodes": [_loom_node_dict(n) for n in nodes],
    }
    body = json.dumps(doc, indent=2)
    fname = f"loom-{loom.id}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/loom/import")
async def loom_import(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Import a JSON loom tree into a brand-new loom owned by the current user.

    Security: incoming node IDs are **never trusted** — they'd collide with, or
    cross-reference, other rows. We regenerate a fresh UUID per node (old→new
    map) and rewire ``parent_id`` in a second pass, dropping any parent link that
    doesn't resolve within the imported set (defends against a doctored file
    pointing at another loom's node). Only whitelisted fields are read. Node
    count is capped at ``_LOOM_MAX_NODES``. A fresh loom is created, so there's no
    in-flight-generation race to guard.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"code": "bad_json"}})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": {"code": "bad_shape"}})
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "'nodes' must be a list", "code": "bad_shape"}},
        )
    if len(raw_nodes) > _LOOM_MAX_NODES:
        return JSONResponse(
            status_code=413,
            content={
                "error": {"message": f"too many nodes (max {_LOOM_MAX_NODES})", "code": "too_large"}
            },
        )

    loom_meta = payload.get("loom") if isinstance(payload.get("loom"), dict) else {}
    title = (str(loom_meta.get("title") or "Imported loom").strip() or "Imported loom")[:120]
    model = loom_meta.get("model")
    model = str(model)[:200] if isinstance(model, str) and model.strip() else None

    loom = Loom(user_id=user.id, title=title, model=model)
    session.add(loom)
    await session.flush()

    # Pass 1: mint new ids, insert nodes with parent_id=None.
    id_map: dict[str, uuid.UUID] = {}
    created: list[tuple[LoomNode, Any]] = []  # (node, raw old parent_id)
    for idx, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            continue
        old_id = raw.get("id")
        text = raw.get("text")
        if not isinstance(text, str):
            text = ""
        node_model = raw.get("model")
        node_model = str(node_model)[:200] if isinstance(node_model, str) else None
        seed = raw.get("seed")
        # bool is a subclass of int — exclude it so `"seed": true` doesn't become
        # a fabricated seed of 1.
        seed = int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None
        # Sibling ordinal (ACS-340): honour an exported ``position`` (same bool
        # guard as seed). A legacy export predates the field, but its nodes were
        # written in (position, created_at) order, so the array index is a
        # faithful fallback that preserves sibling order on round-trip.
        raw_position = raw.get("position")
        position = (
            int(raw_position)
            if isinstance(raw_position, int) and not isinstance(raw_position, bool)
            else idx
        )
        logprobs = raw.get("logprobs")
        if not isinstance(logprobs, list):
            logprobs = None
        nd = LoomNode(
            loom_id=loom.id,
            parent_id=None,
            text=text,
            model=node_model,
            seed=seed,
            position=position,
            logprobs=logprobs,
        )
        session.add(nd)
        await session.flush()
        if isinstance(old_id, str):
            id_map[old_id] = nd.id
        created.append((nd, raw.get("parent_id")))

    # Identify the *intended* roots (original parent was null) vs. dangling
    # orphans (original parent was a non-null ref that doesn't resolve inside the
    # set — e.g. a doctored file, or a partial export). We keep the one-root
    # invariant the rest of the loom code assumes (loom_create_root /
    # root_no_sibling): when exactly one intended root exists, re-home dangling
    # orphans under it so the import stays a single tree; otherwise leave them
    # parentless (a genuine multi-root/forest export round-trips as-is).
    intended_roots = [nd for nd, old_parent in created if old_parent is None]
    fallback_root = intended_roots[0] if len(intended_roots) == 1 else None
    # Pass 2: rewire parents; resolve in-set refs, re-home dangling orphans.
    for nd, old_parent in created:
        if old_parent is None:
            continue  # intended root
        if isinstance(old_parent, str) and old_parent in id_map:
            nd.parent_id = id_map[old_parent]
        elif fallback_root is not None and nd is not fallback_root:
            nd.parent_id = fallback_root.id  # dangling → attach under the sole root

    # Pass 3: break cycles. A crafted file can wire n1→n2→n1 (or a self-parent):
    # both refs resolve in-set, so pass 2 happily builds a cycle with no root. The
    # UI's ancestry walk has a `seen` guard so it won't hang, but a rootless loop
    # is unusable — null the parent of any node that can't reach a root, promoting
    # it to a root so the tree stays walkable.
    by_id = {nd.id: nd for nd, _ in created}
    for nd, _ in created:
        seen: set[uuid.UUID] = set()
        cur = nd
        while cur is not None and cur.parent_id is not None:
            if cur.id in seen:  # looped back without hitting a root
                nd.parent_id = None
                break
            seen.add(cur.id)
            cur = by_id.get(cur.parent_id)
    await session.flush()
    return JSONResponse(status_code=200, content={"loom_id": str(loom.id)})


@router.post("/loom/{loom_id}/generate")
async def loom_generate(
    request: Request,
    loom_id: uuid.UUID,
    parent_id: str = Form(""),
    n: int = Form(_LOOM_DEFAULT_N),
    max_tokens: int = Form(_LOOM_DEFAULT_MAX_TOKENS),
    temperature: float = Form(_CHAT_DEFAULT_TEMPERATURE),
    logprobs: int = Form(_LOOM_DEFAULT_LOGPROBS),
    seed: str = Form(""),
    model: str = Form(""),
    api_key_id: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Branch from ``parent_id``: one upstream request with ``n`` completions.

    The single request fans out N choices; on completion the streaming task
    writes one ``LoomNode`` child per non-empty branch (shared ``parent_id``).
    Returns a ``generation_id``; the client opens the SSE tail to watch the N
    branches stream in parallel.
    """
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})

    # Resolve parent node (must belong to this loom).
    parent_uuid: uuid.UUID | None = None
    raw_parent = (parent_id or "").strip()
    if raw_parent:
        try:
            parent_uuid = uuid.UUID(raw_parent)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": {"code": "bad_parent"}})
    if parent_uuid is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "parent_id required", "code": "bad_parent"}},
        )

    nodes = await _loom_nodes(session, loom.id)
    if len(nodes) >= _LOOM_MAX_NODES:
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": f"loom node cap reached ({_LOOM_MAX_NODES})",
                    "code": "node_cap",
                }
            },
        )
    nodes_by_id = {nd.id: nd for nd in nodes}
    if parent_uuid not in nodes_by_id:
        return JSONResponse(status_code=404, content={"error": {"code": "parent_not_found"}})
    prompt = _loom_prefix_for_node(nodes_by_id, parent_uuid)

    # Resolve the key to charge (same precedence as the single-stream path).
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
        if caller is not None and loom.api_key_id != caller.key_id:
            loom.api_key_id = caller.key_id
    if caller is None and loom.api_key_id is not None:
        caller = await authmod.authed_caller_for_key(session, user.id, loom.api_key_id)
    if caller is None:
        caller = await authmod.primary_authed_caller(session, user.id)
    if caller is None:
        return JSONResponse(
            status_code=403,
            content={"error": {"message": "no active API key", "code": "no_key"}},
        )

    clamped_n = max(1, min(int(n), _LOOM_MAX_N))
    clamped_max = max(1, min(int(max_tokens), _LOOM_MAX_MAX_TOKENS))
    clamped_temp = max(0.0, min(float(temperature), 100.0))
    clamped_logprobs = max(0, min(int(logprobs), _LOOM_MAX_LOGPROBS))
    seed_val: int | None = None
    raw_seed = (seed or "").strip()
    if raw_seed:
        try:
            seed_val = max(0, int(raw_seed))
        except ValueError:
            seed_val = None

    requested_model = (model or loom.model or "").strip() or None
    resolved = _resolve_model(request, requested_model)
    if resolved is None:
        return _unknown_model_response(request, requested_model)
    model_id, entry = resolved
    if loom.model != model_id:
        loom.model = model_id
    _touch(loom)

    await session.commit()

    gen_id = uuid.uuid4()
    state = LoomGenerationState(clamped_n, loom.id, parent_uuid)
    request.app.state.loom_generations[gen_id] = state

    upstream_body: dict[str, Any] = {
        "model": entry.served_model_name,
        "prompt": prompt,
        "max_tokens": clamped_max,
        "temperature": clamped_temp,
        "n": clamped_n,
        "stream": True,
    }
    if clamped_logprobs > 0:
        upstream_body["logprobs"] = clamped_logprobs
    if seed_val is not None:
        upstream_body["seed"] = seed_val

    upstream_url = f"{entry.upstream_url.rstrip('/')}/v1/completions"
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    ip = request.client.host if (request.client and settings.log_ip) else None

    asyncio.create_task(
        _run_loom_generation_task(
            app=request.app,
            gen_id=gen_id,
            loom_id=loom.id,
            parent_id=parent_uuid,
            state=state,
            body=upstream_body,
            upstream_url=upstream_url,
            vllm_api_key=settings.vllm_api_key,
            upstream_timeout_s=settings.upstream_timeout_s,
            model_id=model_id,
            caller_key_id=caller.key_id,
            seed=seed_val,
            request_id=request_id,
            ip=ip,
            t0=time.monotonic(),
        )
    )
    return JSONResponse(
        status_code=200,
        content={"generation_id": str(gen_id), "n": clamped_n, "parent_id": str(parent_uuid)},
    )


@router.get("/loom/{loom_id}/generations/{gen_id}/events")
async def loom_generation_events(
    request: Request,
    loom_id: uuid.UUID,
    gen_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """SSE tail for a loom generate: per-branch ``loom_chunk`` frames + done."""
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    state: LoomGenerationState | None = request.app.state.loom_generations.get(gen_id)
    if state is not None and state.loom_id != loom.id:
        # The gen_id exists but belongs to a different loom — owner-gating the
        # path loom is not enough (the map is global). Refuse rather than tail
        # another loom's stream.
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    if state is None:
        # Evicted / unknown: the durable record is the LoomNode rows; the client
        # falls back to GET /loom/{id}/tree. Emit a terminal done so EventSource
        # doesn't reconnect-storm.
        async def _gone():
            yield _sse_done_frame({"status": "gone", "usage": {}, "error_message": None})

        return StreamingResponse(_gone(), media_type="text/event-stream", headers=SSE_HEADERS)
    return StreamingResponse(
        _loom_gen_live(state, gen_id=gen_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/loom/{loom_id}/generations/{gen_id}/cancel")
async def loom_generation_cancel(
    request: Request,
    loom_id: uuid.UUID,
    gen_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Cancel an in-flight loom generate; partial branches are still persisted."""
    if user is None:
        return JSONResponse(status_code=401, content={"error": {"code": "unauthorized"}})
    loom = await _get_owned_loom(session, loom_id, user.id)
    if loom is None:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    state = request.app.state.loom_generations.get(gen_id)
    if state is not None and state.loom_id != loom.id:
        return JSONResponse(status_code=404, content={"error": {"code": "not_found"}})
    if state is not None:
        state.cancel_requested = True
        state.cancel_event.set()
        state.done_event.set()
    return Response(status_code=204)
