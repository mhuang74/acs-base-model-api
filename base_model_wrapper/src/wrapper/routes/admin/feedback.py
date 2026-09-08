"""Admin feedback triage routes."""

from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import web_auth as webauth
from ...db import get_session
from ...models import Feedback, FeedbackScreenshot, User
from .common import _FEEDBACK_CATEGORIES, _feedback_redirect, log, templates

router = APIRouter()


@router.get("/admin/feedback")
async def admin_feedback(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Feedback triage list. ``status`` (all/open/resolved) and ``category``
    (all/bug/feature/general) query params filter the list; stats are computed
    across all rows regardless of the active filter."""
    status_filter = (request.query_params.get("status") or "all").strip().lower()
    if status_filter not in {"all", "open", "resolved"}:
        status_filter = "all"
    category_filter = (request.query_params.get("category") or "all").strip().lower()
    if category_filter not in ("all", *_FEEDBACK_CATEGORIES):
        category_filter = "all"

    all_rows = list(
        (await session.execute(select(Feedback).order_by(Feedback.created_at.desc())))
        .scalars()
        .all()
    )
    stats = {
        "total": len(all_rows),
        "open": sum(1 for r in all_rows if r.status == "open"),
        "resolved": sum(1 for r in all_rows if r.status == "resolved"),
        "bugs": sum(1 for r in all_rows if r.category == "bug"),
        "features": sum(1 for r in all_rows if r.category == "feature"),
    }

    rows = [
        r
        for r in all_rows
        if (status_filter == "all" or r.status == status_filter)
        and (category_filter == "all" or r.category == category_filter)
    ]

    # Resolve submitter emails for non-anonymous rows in one query, plus the
    # resolving admin's email, so the template doesn't trigger lazy loads.
    user_ids = {r.user_id for r in rows if r.user_id is not None and not r.is_anonymous}
    user_ids |= {r.resolved_by_user_id for r in rows if r.resolved_by_user_id is not None}
    emails_by_id: dict[uuid.UUID, str] = {}
    if user_ids:
        for u in (await session.execute(select(User).where(User.id.in_(user_ids)))).scalars().all():
            emails_by_id[u.id] = u.email

    items = []
    for r in rows:
        shots = list(
            (
                await session.execute(
                    select(FeedbackScreenshot)
                    .where(FeedbackScreenshot.feedback_id == r.id)
                    .order_by(FeedbackScreenshot.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        submitter = None if r.is_anonymous or r.user_id is None else emails_by_id.get(r.user_id)
        items.append(
            {
                "fb": r,
                "screenshot_ids": [s.id for s in shots],
                "submitter_email": submitter,
                "resolved_by_email": emails_by_id.get(r.resolved_by_user_id)
                if r.resolved_by_user_id
                else None,
            }
        )

    return templates.TemplateResponse(
        request,
        "admin_feedback.html",
        {
            "user": admin,
            "items": items,
            "stats": stats,
            "status_filter": status_filter,
            "category_filter": category_filter,
            "categories": _FEEDBACK_CATEGORIES,
            "flash_message": request.query_params.get("msg") or None,
            "flash_error": request.query_params.get("err") or None,
        },
    )


async def _load_feedback_or_404(session: AsyncSession, feedback_id: uuid.UUID) -> Feedback:
    fb = (
        await session.execute(select(Feedback).where(Feedback.id == feedback_id))
    ).scalar_one_or_none()
    if fb is None:
        raise HTTPException(status_code=404, detail="feedback not found")
    return fb


@router.post("/admin/feedback/{feedback_id}/resolve")
async def admin_feedback_resolve(
    feedback_id: uuid.UUID,
    admin_notes: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    fb = await _load_feedback_or_404(session, feedback_id)
    fb.status = "resolved"
    fb.resolved_at = dt.datetime.now(tz=dt.UTC)
    fb.resolved_by_user_id = admin.id
    fb.admin_notes = (admin_notes or "").strip() or None
    log.info(
        "feedback_resolved",
        feedback_id=str(fb.id),
        admin_id=str(admin.id),
        has_notes=fb.admin_notes is not None,
    )
    return _feedback_redirect(msg="Feedback marked resolved.")


@router.post("/admin/feedback/{feedback_id}/reopen")
async def admin_feedback_reopen(
    feedback_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    fb = await _load_feedback_or_404(session, feedback_id)
    fb.status = "open"
    fb.resolved_at = None
    fb.resolved_by_user_id = None
    log.info("feedback_reopened", feedback_id=str(fb.id), admin_id=str(admin.id))
    return _feedback_redirect(msg="Feedback reopened.")


@router.get("/admin/feedback/screenshots/{screenshot_id}")
async def admin_feedback_screenshot(
    screenshot_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Serve a feedback screenshot's raw bytes (admin only)."""
    shot = (
        await session.execute(
            select(FeedbackScreenshot).where(FeedbackScreenshot.id == screenshot_id)
        )
    ).scalar_one_or_none()
    if shot is None:
        raise HTTPException(status_code=404, detail="screenshot not found")
    # Defense in depth: even though uploads are allowlisted to raster types, never
    # let stored bytes render as an active document in the admin origin. Force a
    # download and forbid content-type sniffing so a mislabelled/SVG blob can't XSS.
    return Response(
        content=shot.image_bytes,
        media_type=shot.content_type,
        headers={
            "Content-Disposition": "attachment",
            "X-Content-Type-Options": "nosniff",
        },
    )


# --- email audit log + Resend delivery webhook -------------------------------

# Cap the admin list so a large history never renders an unbounded page; the
