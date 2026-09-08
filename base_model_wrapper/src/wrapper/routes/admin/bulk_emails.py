"""Admin bulk email: CSV upload → preview/edit → send now or scheduled (ACS-228).

Flow: an admin uploads a CSV of fully-addressed emails (one row per
recipient — the CSV arrives pre-personalized, there is no templating here).
That creates a *draft* batch whose items can be previewed and edited in the
browser. Sending is explicit: "send now" processes the batch in-request
(fine at beta scale — uploads are capped), or "schedule" stamps a UTC time
and the APScheduler job in ``lifespan.py`` picks the batch up within a
minute and sends it outside any request.

Every attempted send records an ``email_logs`` row (kind='bulk'), so bulk
mail shows up in the existing /admin/emails audit with webhook delivery
tracking. Every bulk email gets a one-click unsubscribe footer; opted-out
addresses (``email_optouts``) are suppressed at upload AND again at send
time. Transactional mail is unaffected.
"""

from __future__ import annotations

import csv
import datetime as dt
import html as html_mod
import io
import re
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy import func, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from ... import mailer
from ... import web_auth as webauth
from ...db import get_session
from ...dependencies import get_http, get_settings
from ...models import BulkEmailBatch, BulkEmailItem, EmailLog, EmailOptOut, UpdateSubscriber, User
from ...settings import Settings
from .common import _redirect, log, templates

router = APIRouter()

_MAX_BULK_CSV_BYTES = 2 * 1024 * 1024
# Bounds an accidental blast and keeps "send now" fast enough for a request;
# bigger campaigns should be scheduled (the scheduler job sends off-request).
_MAX_BULK_ROWS = 500
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_EMAIL_RE = EMAIL_RE  # internal alias; EMAIL_RE is shared with routes/web.py
# Batch states that still accept mutation (edit/send/schedule/cancel).
_EDITABLE_STATUSES = ("draft", "scheduled")
# Cap the opt-out / subscriber panels on the admin page, like the email audit.
_LIST_LIMIT = 500

OPTOUT_SALT = "email-optout"


# --- small pure helpers -------------------------------------------------------


def _optout_serializer(session_secret: str) -> URLSafeSerializer:
    return URLSafeSerializer(session_secret, salt=OPTOUT_SALT)


def optout_token(email: str, session_secret: str) -> str:
    return _optout_serializer(session_secret).dumps(email.lower())


def email_from_optout_token(token: str, session_secret: str) -> str | None:
    try:
        value = _optout_serializer(session_secret).loads(token)
    except BadSignature:
        return None
    return value if isinstance(value, str) and _EMAIL_RE.match(value) else None


def _split_addresses(raw: str) -> list[str]:
    """Split a cc/bcc cell on commas/semicolons; lowercase, drop empties."""
    return [a.strip().lower() for a in re.split(r"[,;]", raw or "") if a.strip()]


def _bare_address(value: str) -> str:
    """'Ivar <ivar@acsresearch.org>' → 'ivar@acsresearch.org' (lowercased)."""
    m = re.search(r"<([^<>]+)>", value)
    return (m.group(1) if m else value).strip().lower()


def _domain(value: str) -> str:
    addr = _bare_address(value)
    return addr.rsplit("@", 1)[-1] if "@" in addr else ""


def _html_from_text(text: str) -> str:
    """Minimal paragraph HTML for text-only rows (blank line = new paragraph)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n".join(
        "<p>" + html_mod.escape(p).replace("\n", "<br>") + "</p>" for p in paragraphs
    )


def _text_from_html(html: str) -> str:
    """Naive tag-strip fallback when a row supplies only html_body."""
    text = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text).strip()


_HEADER_ALIASES = {
    "from": "from",
    "from_email": "from",
    "to": "to",
    "to_email": "to",
    "email": "to",
    "cc": "cc",
    "bcc": "bcc",
    "reply_to": "reply_to",
    "replyto": "reply_to",
    "subject": "subject",
    "html_body": "html_body",
    "html": "html_body",
    "body_html": "html_body",
    "text_body": "text_body",
    "text": "text_body",
    "body": "text_body",
    "body_text": "text_body",
}


def _normalize_header(name: str) -> str | None:
    # strip/lower/'-'→'_'/' '→'_', mirroring users.py's invite-CSV parser, so
    # spreadsheet-style headers ("Reply To", "HTML Body") match their aliases.
    return _HEADER_ALIASES.get((name or "").strip().lower().replace("-", "_").replace(" ", "_"))


def _valid_from(value: str, allowed_domain: str) -> bool:
    """One address (bare or 'Name <addr>') on the Resend-verified domain.

    Rejects list-shaped values outright — a comma/semicolon would smuggle a
    second, unvalidated sender to Resend verbatim.
    """
    if re.search(r"[,;]", value):
        return False
    bare = _bare_address(value)
    return bool(_EMAIL_RE.match(bare)) and bare.rsplit("@", 1)[-1] == allowed_domain


def parse_bulk_csv(
    csv_text: str, *, default_from: str, allowed_domain: str
) -> tuple[list[dict[str, str]], list[str]]:
    """Parse the bulk CSV into per-item dicts + a list of row errors.

    Accepted columns (aliases in ``_HEADER_ALIASES``): from, to, cc, bcc,
    reply_to, subject, html_body, text_body. Required per row: ``to`` (one
    address), ``subject``, and at least one body column — the missing body
    variant is derived. ``from``/``reply_to`` must be on ``allowed_domain``
    (the Resend-verified domain) or Resend would reject the send anyway.
    """
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    if not reader.fieldnames:
        return [], ["CSV is empty or missing a header row."]

    fieldmap = {name: _normalize_header(name) for name in reader.fieldnames}
    if "to" not in fieldmap.values():
        return [], ["CSV must include a 'to' (or 'email') column."]
    if "subject" not in fieldmap.values():
        return [], ["CSV must include a 'subject' column."]
    if "html_body" not in fieldmap.values() and "text_body" not in fieldmap.values():
        return [], ["CSV must include an 'html_body' or 'text_body' column."]

    items: list[dict[str, str]] = []
    errors: list[str] = []
    seen_to: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        if len(items) >= _MAX_BULK_ROWS:
            errors.append(f"CSV has more than {_MAX_BULK_ROWS} rows; extra rows ignored.")
            break
        norm: dict[str, str] = {}
        for raw_name, canon in fieldmap.items():
            if canon:
                norm[canon] = (row.get(raw_name) or "").strip()

        to = norm.get("to", "").lower()
        if not to and not any(norm.values()):
            continue  # fully blank line
        if not _EMAIL_RE.match(to):
            errors.append(f"Row {line_number}: invalid 'to' address {to!r}.")
            continue
        if re.search(r"[,;]", norm.get("to", "")):
            errors.append(f"Row {line_number}: one recipient per row (got several in 'to').")
            continue
        if to in seen_to:
            errors.append(f"Row {line_number}: duplicate recipient {to} — row skipped.")
            continue

        subject = norm.get("subject", "")
        if not subject:
            errors.append(f"Row {line_number}: missing subject.")
            continue

        html_body = norm.get("html_body", "")
        text_body = norm.get("text_body", "")
        if not html_body and not text_body:
            errors.append(f"Row {line_number}: missing body.")
            continue
        if not html_body:
            html_body = _html_from_text(text_body)
        if not text_body:
            text_body = _text_from_html(html_body)

        from_email = norm.get("from") or default_from
        if not _valid_from(from_email, allowed_domain):
            errors.append(
                f"Row {line_number}: 'from' must be one address @{allowed_domain} (got {from_email!r})."
            )
            continue
        reply_to = norm.get("reply_to", "")
        if reply_to and not _EMAIL_RE.match(_bare_address(reply_to)):
            errors.append(f"Row {line_number}: invalid reply_to {reply_to!r}.")
            continue

        bad_copy = [
            a
            for a in _split_addresses(norm.get("cc", "")) + _split_addresses(norm.get("bcc", ""))
            if not _EMAIL_RE.match(a)
        ]
        if bad_copy:
            errors.append(f"Row {line_number}: invalid cc/bcc address(es) {bad_copy}.")
            continue

        seen_to.add(to)
        items.append(
            {
                "from_email": from_email,
                "to_email": to,
                "cc": ", ".join(_split_addresses(norm.get("cc", ""))),
                "bcc": ", ".join(_split_addresses(norm.get("bcc", ""))),
                "reply_to": reply_to,
                "subject": subject,
                "body_html": html_body,
                "body_text": text_body,
            }
        )
    return items, errors


def _with_unsubscribe_footer(
    *, html: str, text: str, unsubscribe_url: str | None
) -> tuple[str, str]:
    """Append the one-click unsubscribe footer to both bodies.

    When no ``session_secret`` is configured we can't sign tokens; the footer
    then asks the recipient to reply instead (still an out).
    """
    if unsubscribe_url:
        html_footer = (
            '<p style="color:#888888;font-size:12px;margin-top:24px">'
            "You're receiving this because you're part of the ACS Infra beta. "
            f'<a href="{unsubscribe_url}">Unsubscribe</a> — one click, no login.</p>'
        )
        text_footer = f"\n\n--\nUnsubscribe (one click): {unsubscribe_url}"
    else:
        html_footer = (
            '<p style="color:#888888;font-size:12px;margin-top:24px">'
            "You're receiving this because you're part of the ACS Infra beta. "
            "Reply to this email to unsubscribe.</p>"
        )
        text_footer = "\n\n--\nReply to this email to unsubscribe."
    return html + "\n" + html_footer, text + text_footer


# --- send machinery (shared by "send now" and the scheduler job) --------------


async def claim_batch(session: AsyncSession, batch_id: uuid.UUID) -> bool:
    """Atomically claim ``batch_id`` for sending: sendable → 'sending', COMMITTED.

    Compare-and-set + immediate commit so a concurrent "send now" request, an
    overlapping scheduler tick, or a second replica can never claim the same
    batch — whoever loses sees rowcount 0. Without the committed flip, both
    paths read 'scheduled' in their own transactions and every recipient gets
    the campaign twice.
    """
    result = await session.execute(
        sa_update(BulkEmailBatch)
        .where(BulkEmailBatch.id == batch_id, BulkEmailBatch.status.in_(("draft", "scheduled")))
        .values(status="sending")
    )
    await session.commit()
    return result.rowcount == 1


async def send_batch(
    *,
    batch: BulkEmailBatch,
    session: AsyncSession,
    settings: Settings,
    http: httpx.AsyncClient,
) -> dict[str, int]:
    """Send every pending item of the (already claimed) ``batch``.

    Caller must hold the claim from :func:`claim_batch`. Never raises.

    Each item commits individually, so an accepted Resend send is durably
    'sent' even if a later item or the process fails — an interrupted batch
    reverts to 'draft' (see except below) and a re-send only touches the
    still-'pending' items; nobody gets the campaign twice. Re-checks
    ``email_optouts`` at send time (a recipient may have opted out between
    upload and send). Each attempt records an ``email_logs`` row (kind='bulk').
    """
    counts = {"sent": 0, "failed": 0, "skipped": 0, "suppressed": 0}
    try:
        recipient_list = [
            row[0]
            for row in (
                await session.execute(
                    select(BulkEmailItem.to_email).where(BulkEmailItem.batch_id == batch.id)
                )
            ).all()
        ]
        optouts = {
            row[0]
            for row in (
                await session.execute(
                    select(EmailOptOut.email).where(EmailOptOut.email.in_(recipient_list))
                )
            ).all()
        }
        items = list(
            (
                await session.execute(
                    select(BulkEmailItem)
                    .where(
                        BulkEmailItem.batch_id == batch.id, BulkEmailItem.status == "pending"
                    )
                    .order_by(BulkEmailItem.position.asc())
                )
            )
            .scalars()
            .all()
        )
        base_url = settings.public_base_url.rstrip("/")
        for item in items:
            if item.to_email in optouts:
                item.status = "suppressed"
                counts["suppressed"] += 1
                await session.commit()
                continue
            unsubscribe_url = None
            headers = None
            if settings.session_secret:
                token = optout_token(item.to_email, settings.session_secret)
                unsubscribe_url = f"{base_url}/unsubscribe/{token}"
                # RFC 8058 one-click headers: providers POST to the URL, which
                # our POST /unsubscribe/{token} route accepts.
                headers = {
                    "List-Unsubscribe": f"<{unsubscribe_url}>",
                    "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                }
            html, text = _with_unsubscribe_footer(
                html=item.body_html, text=item.body_text, unsubscribe_url=unsubscribe_url
            )
            outcome = await mailer.send(
                settings=settings,
                http=http,
                subject=item.subject,
                to=item.to_email,
                html=html,
                text=text,
                from_email=item.from_email,
                cc=_split_addresses(item.cc or "") or None,
                bcc=_split_addresses(item.bcc or "") or None,
                reply_to=item.reply_to or None,
                headers=headers,
            )
            log_row = EmailLog(
                user_id=None,
                kind="bulk",
                to_email=item.to_email,
                from_email=item.from_email,
                subject=item.subject,
                body_html=html,
                body_text=text,
                send_status=outcome.status,
                skip_reason=outcome.skip_reason,
                error=outcome.error,
                http_status=outcome.http_status,
                resend_message_id=outcome.resend_message_id,
            )
            session.add(log_row)
            await session.flush()
            item.email_log_id = log_row.id
            item.status = (
                outcome.status if outcome.status in ("sent", "failed", "skipped") else "failed"
            )
            item.error = outcome.error
            item.sent_at = dt.datetime.now(tz=dt.UTC) if outcome.status == "sent" else None
            counts[item.status] += 1
            # Durable per item: an accepted send can never be re-sent later.
            await session.commit()

        batch.status = "sent"
        batch.sent_at = dt.datetime.now(tz=dt.UTC)
        await session.commit()
        log.info("bulk_email_batch_sent", batch_id=str(batch.id), **counts)
    except Exception as exc:  # noqa: BLE001 — a mid-batch failure must leave the batch resumable
        log.warning(
            "bulk_email_batch_interrupted",
            batch_id=str(batch.id),
            error=f"{type(exc).__name__}: {exc}",
            **counts,
        )
        try:
            await session.rollback()
            batch.status = "draft"  # re-sendable; only still-'pending' items go out
            batch.scheduled_at = None
            await session.commit()
        except Exception as exc2:  # noqa: BLE001
            log.warning(
                "bulk_email_batch_revert_failed",
                batch_id=str(batch.id),
                error=f"{type(exc2).__name__}: {exc2}",
            )
    return counts


def _email_configured(settings: Settings) -> bool:
    """Can we actually deliver? Refusing to claim a batch when the mailer would
    soft-skip every item keeps the batch sendable after the config is fixed
    (instead of burning it as 'sent' with zero deliveries)."""
    return bool(settings.email_enabled and settings.resend_api_key)


async def send_due_batches(app) -> None:
    """Scheduler entrypoint: send every scheduled batch whose time has come.

    Runs on a 1-minute interval (see ``lifespan.py``). Opens its own session
    per batch and its own HTTP client — no request context here. Each batch is
    atomically claimed first, so an overlapping tick / admin "send now" can't
    double-send; ``send_batch`` never raises and commits per item.
    """
    sessions = app.state.sessions
    settings = app.state.settings
    now = dt.datetime.now(tz=dt.UTC)
    async with sessions() as session:
        due_ids = [
            row[0]
            for row in (
                await session.execute(
                    select(BulkEmailBatch.id).where(
                        BulkEmailBatch.status == "scheduled",
                        BulkEmailBatch.scheduled_at <= now,
                    )
                )
            ).all()
        ]
    if not due_ids:
        return
    if not _email_configured(settings):
        # Leave the batches 'scheduled': they send on the first tick after the
        # config is fixed. Warn each tick so the operator sees it.
        log.warning("bulk_email_due_but_email_disabled", batch_ids=[str(b) for b in due_ids])
        return
    async with httpx.AsyncClient() as http:
        for batch_id in due_ids:
            async with sessions() as session:
                if not await claim_batch(session, batch_id):
                    continue  # someone else got it between the select and now
                batch = (
                    await session.execute(
                        select(BulkEmailBatch).where(BulkEmailBatch.id == batch_id)
                    )
                ).scalar_one()
                await send_batch(batch=batch, session=session, settings=settings, http=http)


# --- routes -------------------------------------------------------------------


async def _batch_or_404(session: AsyncSession, batch_id: uuid.UUID) -> BulkEmailBatch:
    batch = (
        await session.execute(select(BulkEmailBatch).where(BulkEmailBatch.id == batch_id))
    ).scalar_one_or_none()
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch


@router.get("/admin/emails/bulk")
async def admin_bulk_emails(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Bulk-email home: CSV upload form, batch list, opt-outs, subscribers."""
    batches = list(
        (
            await session.execute(
                select(BulkEmailBatch).order_by(BulkEmailBatch.created_at.desc()).limit(100)
            )
        )
        .scalars()
        .all()
    )
    item_counts: dict[uuid.UUID, dict[str, int]] = {}
    if batches:
        rows = (
            await session.execute(
                select(BulkEmailItem.batch_id, BulkEmailItem.status, func.count())
                .where(BulkEmailItem.batch_id.in_([b.id for b in batches]))
                .group_by(BulkEmailItem.batch_id, BulkEmailItem.status)
            )
        ).all()
        for bid, status, n in rows:
            item_counts.setdefault(bid, {})[status] = n
    optouts = list(
        (
            await session.execute(
                select(EmailOptOut).order_by(EmailOptOut.created_at.desc()).limit(_LIST_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    subscribers = list(
        (
            await session.execute(
                select(UpdateSubscriber)
                .order_by(UpdateSubscriber.created_at.desc())
                .limit(_LIST_LIMIT)
            )
        )
        .scalars()
        .all()
    )
    return templates.TemplateResponse(
        request,
        "admin_emails_bulk.html",
        {
            "user": admin,
            "batches": [
                {"batch": b, "counts": item_counts.get(b.id, {})} for b in batches
            ],
            "optouts": optouts,
            "subscribers": subscribers,
            "default_from": settings.email_from,
            "flash_message": request.query_params.get("msg"),
            "flash_error": request.query_params.get("err"),
        },
    )


@router.post("/admin/emails/bulk/upload")
async def admin_bulk_emails_upload(
    request: Request,
    csv_file: UploadFile = File(...),
    name: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Parse the uploaded CSV into a draft batch; redirect to its review page."""
    raw = await csv_file.read()
    if len(raw) > _MAX_BULK_CSV_BYTES:
        return _redirect(
            "/admin/emails/bulk", err=f"CSV is larger than {_MAX_BULK_CSV_BYTES // 1024} KB."
        )
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return _redirect("/admin/emails/bulk", err="CSV must be UTF-8 encoded.")

    parsed, errors = parse_bulk_csv(
        csv_text,
        default_from=settings.email_from,
        allowed_domain=_domain(settings.email_from),
    )
    if not parsed:
        return _redirect(
            "/admin/emails/bulk", err=errors[0] if errors else "No valid rows found."
        )

    recipient_list = [entry["to_email"] for entry in parsed]
    optouts = {
        row[0]
        for row in (
            await session.execute(
                select(EmailOptOut.email).where(EmailOptOut.email.in_(recipient_list))
            )
        ).all()
    }
    batch = BulkEmailBatch(
        created_by_user_id=admin.id,
        name=name.strip() or (csv_file.filename or "bulk batch"),
        status="draft",
    )
    session.add(batch)
    await session.flush()
    for position, entry in enumerate(parsed):
        session.add(
            BulkEmailItem(
                batch_id=batch.id,
                # Explicit CSV position: created_at can't order rows inserted in
                # one transaction (Postgres now() is fixed per transaction).
                position=position,
                status="suppressed" if entry["to_email"] in optouts else "pending",
                from_email=entry["from_email"],
                to_email=entry["to_email"],
                cc=entry["cc"] or None,
                bcc=entry["bcc"] or None,
                reply_to=entry["reply_to"] or None,
                subject=entry["subject"],
                body_html=entry["body_html"],
                body_text=entry["body_text"],
            )
        )
    # The generic request-scoped session dependency commits during its ``yield``
    # cleanup.  FastAPI may run that cleanup after the redirect response has
    # already been sent, so a browser can follow the Location header before the
    # new batch is visible to the next request.  Commit the batch and its items
    # before publishing the review URL.
    await session.commit()
    log.info(
        "bulk_email_batch_uploaded",
        admin_id=str(admin.id),
        batch_id=str(batch.id),
        rows=len(parsed),
        parse_errors=len(errors),
    )
    shown = "; ".join(errors[:5]) + (f" (+{len(errors) - 5} more)" if len(errors) > 5 else "")
    return _redirect(
        f"/admin/emails/bulk/{batch.id}",
        msg=f"Uploaded {len(parsed)} emails.",
        err=shown or None,
    )


@router.get("/admin/emails/bulk/{batch_id}")
async def admin_bulk_batch(
    request: Request,
    batch_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Batch review page: item list with statuses + send/schedule/cancel."""
    batch = await _batch_or_404(session, batch_id)
    items = list(
        (
            await session.execute(
                select(BulkEmailItem)
                .where(BulkEmailItem.batch_id == batch.id)
                .order_by(BulkEmailItem.position.asc())
            )
        )
        .scalars()
        .all()
    )
    return templates.TemplateResponse(
        request,
        "admin_email_bulk_batch.html",
        {
            "user": admin,
            "batch": batch,
            "items": items,
            "editable": batch.status in _EDITABLE_STATUSES,
            "flash_message": request.query_params.get("msg"),
            "flash_error": request.query_params.get("err"),
        },
    )


@router.get("/admin/emails/bulk/{batch_id}/items/{item_id}")
async def admin_bulk_item(
    request: Request,
    batch_id: uuid.UUID,
    item_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Single-email preview (sandboxed, like the email-audit detail) + editor."""
    batch = await _batch_or_404(session, batch_id)
    item = (
        await session.execute(
            select(BulkEmailItem).where(
                BulkEmailItem.id == item_id, BulkEmailItem.batch_id == batch.id
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return templates.TemplateResponse(
        request,
        "admin_email_bulk_item.html",
        {
            "user": admin,
            "batch": batch,
            "item": item,
            "editable": batch.status in _EDITABLE_STATUSES,
            "flash_message": request.query_params.get("msg"),
            "flash_error": request.query_params.get("err"),
        },
    )


@router.post("/admin/emails/bulk/{batch_id}/items/{item_id}")
async def admin_bulk_item_save(
    batch_id: uuid.UUID,
    item_id: uuid.UUID,
    from_email: str = Form(...),
    subject: str = Form(...),
    body_text: str = Form(...),
    body_html: str = Form(""),
    cc: str = Form(""),
    bcc: str = Form(""),
    reply_to: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Save in-browser edits to one drafted email (draft/scheduled batches only).

    Re-runs the same validation as the CSV parser — an edit must not be able to
    store an unsendable item that only fails mid-send.
    """
    batch = await _batch_or_404(session, batch_id)
    if batch.status not in _EDITABLE_STATUSES:
        raise HTTPException(status_code=409, detail="batch is no longer editable")
    item = (
        await session.execute(
            select(BulkEmailItem).where(
                BulkEmailItem.id == item_id, BulkEmailItem.batch_id == batch.id
            )
        )
    ).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")

    item_url = f"/admin/emails/bulk/{batch.id}/items/{item.id}"
    allowed = _domain(settings.email_from)
    if not _valid_from(from_email.strip(), allowed):
        return _redirect(item_url, err=f"'From' must be one address @{allowed}. Not saved.")
    if not subject.strip():
        return _redirect(item_url, err="Subject can't be empty. Not saved.")
    reply_to = reply_to.strip()
    if reply_to and not _EMAIL_RE.match(_bare_address(reply_to)):
        return _redirect(item_url, err=f"Invalid reply-to {reply_to!r}. Not saved.")
    bad_copy = [
        a
        for a in _split_addresses(cc) + _split_addresses(bcc)
        if not _EMAIL_RE.match(a)
    ]
    if bad_copy:
        return _redirect(item_url, err=f"Invalid cc/bcc address(es) {bad_copy}. Not saved.")

    item.from_email = from_email.strip()
    item.subject = subject.strip()
    item.body_text = body_text
    item.body_html = body_html.strip() or _html_from_text(body_text)
    item.cc = ", ".join(_split_addresses(cc)) or None
    item.bcc = ", ".join(_split_addresses(bcc)) or None
    item.reply_to = reply_to or None
    log.info("bulk_email_item_edited", admin_id=str(admin.id), item_id=str(item.id))
    return _redirect(item_url, msg="Saved.")


@router.post("/admin/emails/bulk/{batch_id}/send")
async def admin_bulk_batch_send(
    batch_id: uuid.UUID,
    send_mode: str = Form("now"),  # 'now' | 'schedule'
    scheduled_at: str = Form(""),  # datetime-local value, interpreted as UTC
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    """Send the batch now, or stamp a UTC schedule for the background job."""
    batch = await _batch_or_404(session, batch_id)
    batch_url = f"/admin/emails/bulk/{batch.id}"
    if batch.status not in _EDITABLE_STATUSES:
        raise HTTPException(status_code=409, detail=f"batch is {batch.status}, not sendable")

    if send_mode == "schedule":
        try:
            when = dt.datetime.fromisoformat(scheduled_at).replace(tzinfo=dt.UTC)
        except ValueError:
            return _redirect(batch_url, err="Invalid schedule time.")
        if when <= dt.datetime.now(tz=dt.UTC):
            return _redirect(batch_url, err="Schedule time is in the past.")
        batch.status = "scheduled"
        batch.scheduled_at = when
        log.info(
            "bulk_email_batch_scheduled",
            admin_id=str(admin.id),
            batch_id=str(batch.id),
            scheduled_at=when.isoformat(),
        )
        return _redirect(batch_url, msg=f"Scheduled for {when:%Y-%m-%d %H:%M} UTC.")

    if not _email_configured(settings):
        # Refuse rather than burn the batch: the mailer would soft-skip every
        # item and the batch would end 'sent' with zero deliveries.
        return _redirect(
            batch_url,
            err="Email sending is disabled (EMAIL_ENABLED / RESEND_API_KEY). Batch left as is.",
        )
    if not await claim_batch(session, batch.id):
        return _redirect(batch_url, err="Batch was already picked up (scheduler or another admin).")
    counts = await send_batch(batch=batch, session=session, settings=settings, http=http)
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    return _redirect(batch_url, msg=f"Done: {summary or 'nothing to send'}.")


@router.post("/admin/emails/bulk/{batch_id}/cancel")
async def admin_bulk_batch_cancel(
    batch_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Cancel a draft or scheduled batch (kept for audit, never deleted)."""
    batch = await _batch_or_404(session, batch_id)
    if batch.status not in _EDITABLE_STATUSES:
        raise HTTPException(status_code=409, detail=f"batch is {batch.status}, not cancelable")
    batch.status = "canceled"
    batch.scheduled_at = None
    log.info("bulk_email_batch_canceled", admin_id=str(admin.id), batch_id=str(batch.id))
    return _redirect(f"/admin/emails/bulk/{batch.id}", msg="Canceled.")
