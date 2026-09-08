"""Transactional email audit and webhook routes."""

from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import web_auth as webauth
from ...db import get_session
from ...dependencies import get_settings
from ...models import EmailEvent, EmailLog, User
from ...settings import Settings
from .common import log, templates

router = APIRouter()

_EMAIL_LIST_LIMIT = 500


def _parse_iso(ts: object) -> dt.datetime | None:
    """Best-effort ISO-8601 parse of Resend's ``created_at`` value."""
    if not isinstance(ts, str):
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.post("/webhooks/resend")
async def resend_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Ingest Resend delivery webhooks (delivered/bounced/complained/opened/…).

    Public endpoint — secured by the Svix signature Resend sends, not admin
    auth. On a verified event we record one EmailEvent (full raw payload) and
    update the matching EmailLog's ``last_event``. When no ``resend_webhook_secret``
    is configured we accept (200) but store nothing: unverified payloads are
    never persisted, and a 200 stops Resend retrying forever.
    """
    raw = await request.body()
    secret = settings.resend_webhook_secret
    if not secret:
        log.warning("resend_webhook_unconfigured")
        return Response(status_code=200)

    from svix.webhooks import Webhook, WebhookVerificationError

    try:
        event = Webhook(secret).verify(raw, dict(request.headers))
    except WebhookVerificationError:
        log.warning("resend_webhook_bad_signature")
        raise HTTPException(status_code=400, detail="invalid signature")

    event_type = str(event.get("type") or "unknown")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    message_id = data.get("email_id") or data.get("id")
    message_id = message_id if isinstance(message_id, str) else None
    occurred_at = _parse_iso(event.get("created_at"))

    matched: EmailLog | None = None
    if message_id:
        matched = (
            await session.execute(select(EmailLog).where(EmailLog.resend_message_id == message_id))
        ).scalar_one_or_none()

    session.add(
        EmailEvent(
            email_log_id=matched.id if matched else None,
            resend_message_id=message_id,
            event_type=event_type,
            raw_payload=event,
            occurred_at=occurred_at,
        )
    )
    if matched is not None:
        matched.last_event = event_type
        matched.last_event_at = occurred_at or dt.datetime.now(tz=dt.UTC)

    log.info(
        "resend_webhook_received",
        event_type=event_type,
        matched=matched is not None,
        message_id=message_id,
    )
    return Response(status_code=200)


@router.get("/admin/emails")
async def admin_emails(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Audit list of transactional email the app sent or attempted.

    ``status`` (all/sent/skipped/failed) and ``kind`` (all/<email kind>) query
    params filter the list; stats are computed across the (capped) window
    regardless of the active filter.
    """
    rows = list(
        (
            await session.execute(
                select(EmailLog).order_by(EmailLog.created_at.desc()).limit(_EMAIL_LIST_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    stats = {
        "total": len(rows),
        "sent": sum(1 for r in rows if r.send_status == "sent"),
        "skipped": sum(1 for r in rows if r.send_status == "skipped"),
        "failed": sum(1 for r in rows if r.send_status == "failed"),
        "delivered": sum(1 for r in rows if r.last_event == "email.delivered"),
        "bounced": sum(1 for r in rows if r.last_event == "email.bounced"),
    }
    kinds = sorted({r.kind for r in rows})

    status_filter = (request.query_params.get("status") or "all").strip().lower()
    if status_filter not in {"all", "sent", "skipped", "failed"}:
        status_filter = "all"
    kind_filter = (request.query_params.get("kind") or "all").strip().lower()
    if kind_filter != "all" and kind_filter not in kinds:
        kind_filter = "all"

    filtered = [
        r
        for r in rows
        if (status_filter == "all" or r.send_status == status_filter)
        and (kind_filter == "all" or r.kind == kind_filter)
    ]

    # Recipient emails in one query so the template doesn't lazy-load per row.
    user_ids = {r.user_id for r in filtered if r.user_id is not None}
    emails_by_id: dict[uuid.UUID, str] = {}
    if user_ids:
        for uid, em in (
            await session.execute(select(User.id, User.email).where(User.id.in_(user_ids)))
        ).all():
            emails_by_id[uid] = em

    items = [{"row": r, "recipient_email": emails_by_id.get(r.user_id)} for r in filtered]

    return templates.TemplateResponse(
        request,
        "admin_emails.html",
        {
            "user": admin,
            "items": items,
            "stats": stats,
            "kinds": kinds,
            "kind_filter": kind_filter,
            "status_filter": status_filter,
            "truncated": len(rows) >= _EMAIL_LIST_LIMIT,
            "list_limit": _EMAIL_LIST_LIMIT,
        },
    )


@router.get("/admin/emails/{email_id}")
async def admin_email_detail(
    request: Request,
    email_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Single email: metadata, full HTML (sandboxed preview) + text, and the
    timeline of delivery events with their raw payloads."""
    import json as _json

    row = (
        await session.execute(select(EmailLog).where(EmailLog.id == email_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="email not found")

    events = list(
        (
            await session.execute(
                select(EmailEvent)
                .where(EmailEvent.email_log_id == row.id)
                .order_by(EmailEvent.received_at.asc())
            )
        )
        .scalars()
        .all()
    )
    events_view = [
        {
            "ev": e,
            "pretty": _json.dumps(e.raw_payload, indent=2, sort_keys=True, default=str),
        }
        for e in events
    ]

    recipient_email = None
    if row.user_id is not None:
        recipient_email = (
            await session.execute(select(User.email).where(User.id == row.user_id))
        ).scalar_one_or_none()

    return templates.TemplateResponse(
        request,
        "admin_email_detail.html",
        {
            "user": admin,
            "row": row,
            "events": events_view,
            "recipient_email": recipient_email,
        },
    )


# --- admin: model lifecycle + capacity probe ---------------------------------
