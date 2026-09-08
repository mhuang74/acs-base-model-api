"""Admin user, key-budget, invite, and public invite routes."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
import io
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ... import auth as authmod
from ... import mailer as mailermod
from ... import web_auth as webauth
from ...db import get_session
from ...dependencies import get_http, get_settings
from ...keys import generate as generate_key
from ...models import (
    USER_STATUSES,
    ApiKey,
    EmailLog,
    Feedback,
    SignupInvite,
    SignupInviteRedemption,
    UsageMonthly,
    User,
)
from ...rate_limit import limiter
from ...services import customer360, user_tags
from ...settings import Settings
from .common import _pending_key_serializer, log, templates

router = APIRouter()

# The Discord invite prefill comes from settings.beta_discord_url (ACS-269 —
# the old checked-in constant pointed at an expired invite).
_DEFAULT_BETA_SURVEY_URL = "https://forms.example/survey"
_MAX_INVITE_CSV_BYTES = 256 * 1024
# Hard cap on rows per upload. The byte cap alone allows ~30k tiny rows, and each
# row creates an invite + does one sequential, awaited email send in a single
# request/transaction — a huge CSV would block for minutes, time out, and roll
# back all invites *after* emails already went out (live links with no token).
# A generous cap keeps the request fast and bounds an accidental bulk blast;
# split larger lists into batches. (Beta cohort is ~30.)
_MAX_INVITE_CSV_ROWS = 200


@dataclass(frozen=True)
class PersonalizedInviteCsvRow:
    line_number: int
    email: str
    name: str
    survey_respondent: bool


def _csv_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value.strip()
    return ""


def _looks_like_survey_row(row: dict[str, str]) -> bool:
    explicit = _csv_value(row, "survey_respondent", "survey", "from_survey").lower()
    if explicit in {"1", "true", "yes", "y"}:
        return True
    if explicit in {"0", "false", "no", "n"}:
        return False
    source = " ".join(_csv_value(row, key).lower() for key in ("source", "cohort", "survey_status"))
    return "survey" in source


def _parse_personalized_invite_csv(
    csv_text: str,
) -> tuple[list[PersonalizedInviteCsvRow], list[str]]:
    """Parse beta-invite CSV text.

    Required column: ``email``. Optional columns: ``name`` plus one of
    ``survey_respondent``, ``survey``, ``from_survey``, ``source``, ``cohort``,
    or ``survey_status`` to switch the template to the survey-respondent intro.
    """
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    if not reader.fieldnames:
        return [], ["CSV is empty or missing a header row."]

    normalized_headers = {
        (header or "").strip().lower().replace("-", "_").replace(" ", "_").lstrip("\ufeff"): header
        for header in reader.fieldnames
    }
    if "email" not in normalized_headers:
        return [], ["CSV must include an email column."]

    rows: list[PersonalizedInviteCsvRow] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(reader, start=2):
        # Reject the whole upload if it's too big \u2014 don't process a partial batch
        # (which would send some emails before erroring). Split into batches.
        if line_number - 1 > _MAX_INVITE_CSV_ROWS:
            return [], [
                f"CSV has more than {_MAX_INVITE_CSV_ROWS} rows. "
                "Split it into smaller batches and upload each separately."
            ]
        normalized = {
            (key or "").strip().lower().replace("-", "_").replace(" ", "_").lstrip("\ufeff"): (
                value or ""
            )
            for key, value in raw.items()
        }
        email = _csv_value(normalized, "email").lower()
        if not email or "@" not in email:
            errors.append(f"line {line_number}: missing or invalid email")
            continue
        if email in seen:
            errors.append(f"line {line_number}: duplicate email {email}")
            continue
        seen.add(email)
        rows.append(
            PersonalizedInviteCsvRow(
                line_number=line_number,
                email=email,
                name=_csv_value(normalized, "name", "full_name"),
                survey_respondent=_looks_like_survey_row(normalized),
            )
        )
    return rows, errors


# Admin users-list ordering: regular accounts at the top, admins grouped at
# the bottom. The admin pool is tiny (≤5) and rarely changes; keeping it out
# of the day-to-day-user scroll makes the list scannable for the actual job
# (approving signups, reviewing usage, deleting test accounts). Newest-first
# within each group preserves the existing chronology cue.
_ALL_USERS_ORDER = (
    case((User.role == "admin", 1), else_=0).asc(),
    User.created_at.desc(),
)


async def _render_admin_user_detail(
    request: Request,
    admin: User,
    target: User,
    session: AsyncSession,
    *,
    message: str | None = None,
    error: str | None = None,
    status_code: int = 200,
    set_password_link: str | None = None,
):
    keys = list(
        (
            await session.execute(
                select(ApiKey).where(ApiKey.user_id == target.id).order_by(ApiKey.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    period_start = authmod._current_period_start()
    key_rows = []
    for k in keys:
        usage = (
            await session.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == k.id,
                    UsageMonthly.period_start == period_start,
                )
            )
        ).scalar_one_or_none()
        used = (usage.tokens_prompt + usage.tokens_completion) if usage else 0
        key_rows.append({"key": k, "this_month": used})
    aggregate_used = sum(r["this_month"] for r in key_rows)
    # Latest approval email so the admin can see whether the onboarding/
    # set-password link was actually delivered (ACS-100).
    latest_approval_email = (
        await session.execute(
            select(EmailLog)
            .where(EmailLog.user_id == target.id, EmailLog.kind == "approval")
            # id tie-break so same-timestamp rows pick a deterministic "latest".
            .order_by(EmailLog.created_at.desc(), EmailLog.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    # Person-page extras (ACS-300): engagement + lifetime usage, full email
    # history (also matches pre-account sends by address), their feedback.
    engagement = await customer360.load_user_engagement(session, target.id)
    email_history = list(
        (
            await session.execute(
                select(EmailLog)
                .where(
                    (EmailLog.user_id == target.id)
                    | (func.lower(EmailLog.to_email) == target.email.lower())
                )
                .order_by(EmailLog.created_at.desc(), EmailLog.id.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    feedback_items = list(
        (
            await session.execute(
                select(Feedback)
                .where(Feedback.user_id == target.id)
                .order_by(Feedback.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    tags = (await user_tags.load_tags_for_users(session, [target.id])).get(target.id, [])
    known_tags = await user_tags.all_tags(session)
    # Who suspended them (ACS-353) — display only; the FK is SET NULL, so an
    # admin who has since been deleted just renders as an unattributed suspend.
    suspended_by = None
    if target.suspended_by_user_id is not None:
        suspended_by = (
            await session.execute(
                select(User.email).where(User.id == target.suspended_by_user_id)
            )
        ).scalar_one_or_none()
    return templates.TemplateResponse(
        request,
        "admin_user_detail.html",
        {
            "user": admin,
            "target": target,
            "keys": key_rows,
            "aggregate_used": aggregate_used,
            "message": message,
            "error": error,
            "needs_password": target.password_hash is None,
            "latest_approval_email": latest_approval_email,
            "set_password_link": set_password_link,
            "engagement": engagement,
            "email_history": email_history,
            "feedback_items": feedback_items,
            "suspended_by": suspended_by,
            "tags": tags,
            "known_tags": known_tags,
        },
        status_code=status_code,
    )


async def _load_target_user(session: AsyncSession, user_id: uuid.UUID) -> User:
    target = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    return target


def _approval_available_models(request: Request) -> list[dict[str, object]]:
    """Live models to include in the approval email handoff."""
    registry = getattr(request.app.state, "models", {}) or {}
    rows = []
    for m in registry.values():
        if getattr(m, "status", "live") != "live":
            continue
        rows.append(
            {
                "id": m.model_id,
                "served_model_name": m.served_model_name,
                "gpu_shape": m.gpu_shape_label,
                "max_model_len": m.max_model_len,
            }
        )
    return rows


@router.get("/admin/users/{user_id}")
async def admin_user_detail(
    request: Request,
    user_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    target = await _load_target_user(session, user_id)
    return await _render_admin_user_detail(request, admin, target, session)


@router.post("/admin/users/{user_id}/approve")
async def admin_approve_user(
    request: Request,
    user_id: uuid.UUID,
    approval_reason: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    import datetime as _dt

    target = await _load_target_user(session, user_id)
    if target.status == "approved":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="User is already approved.",
            status_code=400,
        )
    if not settings.session_secret:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="SESSION_SECRET not configured — cannot encrypt one-shot key.",
            status_code=503,
        )

    target.status = "approved"
    target.approved_at = _dt.datetime.now(tz=_dt.UTC)
    target.approved_by_user_id = admin.id
    target.rejected_at = None
    # Approving a suspended account is a legitimate way back (ACS-353) — clear
    # its stamps too, or the detail page would keep claiming it's suspended.
    target.suspended_at = None
    target.suspended_by_user_id = None
    # Approval reason lands in the free-text notes as a dated entry, newest on
    # top (ACS-300) — the approval moment is when "why did we let them in" is
    # actually known.
    reason = (approval_reason or "").strip()[:2000]
    if reason:
        entry = f"{target.approved_at.strftime('%Y-%m-%d')} (approved): {reason}"
        target.notes = entry if not target.notes else f"{entry}\n\n{target.notes}"
    # Never overwrite a manually-set aggregate budget.
    if target.monthly_token_budget_total is None:
        target.monthly_token_budget_total = settings.default_monthly_token_budget_total

    # Mint the initial key + stash the one-shot plaintext (encrypted at rest with
    # the session_secret-derived pending-key cipher; the first /dashboard render
    # decrypts, shows once, clears). Shared with invite acceptance.
    api_key = await _provision_approved_user(session, user=target, settings=settings)

    # Password-less accounts (approved without ever going through signup or
    # invite-accept) have no way to sign in — the login page only offers
    # "forgot password", which is confusing for someone who never set one
    # (ACS-99). Mint a single-use, long-lived set-password link and surface it
    # in the approval email so they land directly on the set-password page.
    set_password_url: str | None = None
    set_password_expiry_minutes: int | None = None
    if target.password_hash is None:
        set_password_expiry_minutes = settings.approval_set_password_expiry_minutes
        reset_token = await webauth.create_password_reset(
            session, target, expiry_minutes=set_password_expiry_minutes
        )
        set_password_url = settings.public_base_url.rstrip("/") + f"/reset-password/{reset_token}"

    dashboard_url = settings.public_base_url.rstrip("/") + "/dashboard"
    await mailermod.send_approval_email(
        settings=settings,
        http=http,
        session=session,
        user=target,
        dashboard_url=dashboard_url,
        tutorial_url=settings.public_base_url.rstrip("/") + "/tutorial",
        available_models=_approval_available_models(request),
        set_password_url=set_password_url,
        set_password_expiry_minutes=set_password_expiry_minutes,
        discord_url=settings.beta_discord_url,
        discord_connect_url=(
            dashboard_url + "#community" if settings.discord_oauth_enabled else None
        ),
    )
    log.info(
        "admin_user_approved",
        admin_id=str(admin.id),
        user_id=str(target.id),
        key_id=str(api_key.id),
        set_password_link_sent=set_password_url is not None,
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/resend-set-password")
async def admin_resend_set_password(
    request: Request,
    user_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    """Mint a fresh single-use set-password link for a password-less approved
    user and re-send the approval email — and show the link to the admin so it
    can be delivered out-of-band if email delivery is down (ACS-100). The
    previous outstanding link is invalidated (single live link)."""
    target = await _load_target_user(session, user_id)
    if target.status != "approved":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="Only approved users can be sent a set-password link.",
            status_code=400,
        )
    if target.password_hash is not None:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="This user already has a password — they can use “Forgot password?” to reset it.",
            status_code=400,
        )

    expiry = settings.approval_set_password_expiry_minutes
    reset_token = await webauth.create_password_reset(session, target, expiry_minutes=expiry)
    base = settings.public_base_url.rstrip("/")
    set_password_url = f"{base}/reset-password/{reset_token}"
    await mailermod.send_approval_email(
        settings=settings,
        http=http,
        session=session,
        user=target,
        dashboard_url=f"{base}/dashboard",
        tutorial_url=f"{base}/tutorial",
        available_models=_approval_available_models(request),
        set_password_url=set_password_url,
        set_password_expiry_minutes=expiry,
        discord_url=settings.beta_discord_url,
        discord_connect_url=(
            f"{base}/dashboard#community" if settings.discord_oauth_enabled else None
        ),
    )
    log.info(
        "admin_resend_set_password",
        admin_id=str(admin.id),
        user_id=str(target.id),
    )
    return await _render_admin_user_detail(
        request,
        admin,
        target,
        session,
        message=(
            "Generated a fresh single-use set-password link and re-sent the "
            "approval email (see status above). The previous link no longer "
            "works. Copy the link below to deliver it to the user directly."
        ),
        set_password_link=set_password_url,
    )


@router.post("/admin/users/{user_id}/reject")
async def admin_reject_user(
    request: Request,
    user_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    _same_origin: None = webauth.SameOriginDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    import datetime as _dt

    target = await _load_target_user(session, user_id)
    if target.status == "rejected":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="User is already rejected.",
            status_code=400,
        )
    target.status = "rejected"
    target.rejected_at = _dt.datetime.now(tz=_dt.UTC)
    # Rejecting supersedes a suspension (ACS-353) — clear its stamps, or the
    # Profile panel would show "Rejected …" and "Suspended … by …" side by side,
    # reading as two live states at once. Same reasoning as approve.
    target.suspended_at = None
    target.suspended_by_user_id = None
    # Belt-and-braces with the auth-layer status check (ACS-212): revoke any
    # keys the user already minted (relevant when rejecting a previously
    # approved account), so rejection reads as a full access cut in the key
    # tables and dashboards too — not just an auth-time refusal.
    revoked = (
        await session.execute(
            update(ApiKey)
            .where(ApiKey.user_id == target.id, ApiKey.revoked_at.is_(None))
            .values(revoked_at=_dt.datetime.now(tz=_dt.UTC))
        )
    ).rowcount
    # ...and their web sessions (up to 30 days long): without this, a user
    # rejected while logged in keeps a cookie that can browse the workbench
    # and mint fresh keys via /me/keys (#211 review finding).
    await webauth.revoke_all_user_sessions(session, target.id)
    await mailermod.send_rejection_email(settings=settings, http=http, session=session, user=target)
    log.info(
        "admin_user_rejected",
        admin_id=str(admin.id),
        user_id=str(target.id),
        keys_revoked=revoked,
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/delete")
async def admin_delete_user(
    request: Request,
    user_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    _same_origin: None = webauth.SameOriginDep,
    session: AsyncSession = Depends(get_session),
):
    """Permanently delete a user and the data that cascades from them.

    Goes away with the user: api_keys + their api_requests / usage_monthly /
    usage_daily rows (CASCADE), user_sessions, password_resets, chat_sessions
    (+ snapshots/generations). Preserved with NULL attribution: feedback,
    email_logs (audit trail), signup_invites + redemptions, probe_schedules,
    model_warm_windows.

    Guarded so an admin can't lock themselves (or the whole admin role) out:
    deleting yourself is rejected, and deleting the last remaining admin is
    rejected. The action is irreversible — the confirm dialog in the UI
    surfaces that to the operator.
    """
    target = await _load_target_user(session, user_id)
    if target.id == admin.id:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="You can't delete your own account from the admin UI.",
            status_code=400,
        )
    if target.role == "admin":
        # Race-safe last-admin guard. Without locking, two concurrent
        # "admin A deletes admin B" / "admin B deletes admin A" requests both
        # observe count(other-admins) == 1, both pass the check, both commit —
        # and the admin role is gone. SELECT … FOR UPDATE in a stable id order
        # serializes concurrent deletes touching the admin role: the second
        # transaction blocks on the lock until the first commits, then re-reads
        # and sees the new (decremented) count.
        admins_locked = list(
            (
                await session.execute(
                    select(User).where(User.role == "admin").order_by(User.id).with_for_update()
                )
            )
            .scalars()
            .all()
        )
        if len([a for a in admins_locked if a.id != target.id]) == 0:
            return await _render_admin_user_detail(
                request,
                admin,
                target,
                session,
                error="Can't delete the last remaining admin.",
                status_code=400,
            )

    target_email = target.email
    log.info(
        "admin_user_deleted",
        admin_id=str(admin.id),
        user_id=str(target.id),
        target_email=target_email[:3] + "***",
        target_role=target.role,
    )
    await session.delete(target)

    from urllib.parse import urlencode

    return RedirectResponse(
        url="/admin/users?" + urlencode({"msg": f"Deleted {target_email}."}),
        status_code=303,
    )


@router.post("/admin/users/{user_id}/budget")
async def admin_set_user_budget(
    request: Request,
    user_id: uuid.UUID,
    monthly_token_budget_total: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    target = await _load_target_user(session, user_id)
    raw = (monthly_token_budget_total or "").strip()
    if raw == "":
        target.monthly_token_budget_total = None
    else:
        try:
            n = int(raw)
            if n < 0:
                raise ValueError
        except ValueError:
            return await _render_admin_user_detail(
                request,
                admin,
                target,
                session,
                error="Budget must be a non-negative integer or blank.",
                status_code=400,
            )
        target.monthly_token_budget_total = n
    log.info(
        "admin_user_budget_set",
        admin_id=str(admin.id),
        user_id=str(target.id),
        budget=target.monthly_token_budget_total,
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/notes")
async def admin_save_user_notes(
    request: Request,
    user_id: uuid.UUID,
    notes: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    target = await _load_target_user(session, user_id)
    # Browsers submit textarea content with CRLF; normalize so notes don't mix
    # line endings with the \n-joined approval entries and the 20k cap can't
    # disagree with the client-side maxlength.
    cleaned = (notes or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(cleaned) > 20_000:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="Notes are limited to 20,000 characters.",
            status_code=400,
        )
    target.notes = cleaned or None
    log.info(
        "admin_user_notes_saved",
        admin_id=str(admin.id),
        user_id=str(target.id),
        length=len(cleaned),
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/suspend")
async def admin_suspend_user(
    request: Request,
    user_id: uuid.UUID,
    reason: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    _same_origin: None = webauth.SameOriginDep,
    session: AsyncSession = Depends(get_session),
):
    """Park an account's access, reversibly (ACS-353).

    Unlike ``reject`` this is NOT a decision about the person — it's for access
    that was always time-boxed (a hiring cohort, event tutorial accounts) or an
    account we want to idle before deleting. So, deliberately, it does **not**
    sweep the user's API keys the way ``admin_reject_user`` does: that comment
    justifies its key sweep as making the cut "read as a full access cut in the
    key tables and dashboards too", which is right for a terminal state and
    wrong for a reversible one. Revoking would make unsuspend unable to restore
    what was taken, and touching ``disabled_at`` would clobber keys the user
    paused themselves from their dashboard. The auth-layer status check refuses
    every request on status alone (no auth cache), so the keys need no help.

    Web sessions DO get revoked. ``current_user`` re-reads status per request
    since ACS-353, so the cookie is already inert while the account is
    suspended; revoking matters for what happens *after* an unsuspend — without
    it, every stale cookie from before the suspension would spring back to life
    the moment the account is restored.
    """
    target = await _load_target_user(session, user_id)
    if target.id == admin.id:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="You can't suspend your own account.",
            status_code=400,
        )
    if target.status == "suspended":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="This account is already suspended.",
            status_code=400,
        )
    # Only an APPROVED account can be suspended. Without this, suspending a
    # pending applicant and then unsuspending would promote them to approved
    # behind ``admin_approve_user``'s back — no key minted, no approval stamps,
    # no default budget, no email — and suspend→unsuspend on a rejected row
    # would silently un-reject it while leaving ``rejected_at`` set and every
    # key still revoked from the reject sweep. ``admin_unsuspend_user`` can
    # hardcode ``status = "approved"`` precisely because this is the only door in.
    if target.status != "approved":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error=(
                f"Only an approved account can be suspended (this one is {target.status}). "
                "Use reject to decline a pending application."
            ),
            status_code=400,
        )
    target.status = "suspended"
    target.suspended_at = dt.datetime.now(tz=dt.UTC)
    target.suspended_by_user_id = admin.id
    # Same dated-note convention as approve (ACS-300): newest on top.
    cleaned = (reason or "").strip()[:2000]
    if cleaned:
        entry = f"{target.suspended_at.strftime('%Y-%m-%d')} (suspended): {cleaned}"
        target.notes = entry if not target.notes else f"{entry}\n\n{target.notes}"
    await webauth.revoke_all_user_sessions(session, target.id)
    log.info(
        "admin_user_suspended",
        admin_id=str(admin.id),
        user_id=str(target.id),
        has_reason=bool(cleaned),
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/unsuspend")
async def admin_unsuspend_user(
    request: Request,
    user_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Restore a suspended account (ACS-353).

    Deliberately NOT routed through ``/approve``: that mints a fresh API key via
    ``_provision_approved_user``, overwrites ``approved_at`` /
    ``approved_by_user_id``, and sends the approval email — none of which is
    wanted when handing back access the user already had. Since suspend left the
    keys alone, flipping the status back is the whole job.
    """
    target = await _load_target_user(session, user_id)
    if target.status != "suspended":
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="Only a suspended account can be unsuspended.",
            status_code=400,
        )
    target.status = "approved"
    target.suspended_at = None
    target.suspended_by_user_id = None
    log.info("admin_user_unsuspended", admin_id=str(admin.id), user_id=str(target.id))
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/tags/add")
async def admin_add_user_tag(
    request: Request,
    user_id: uuid.UUID,
    tag: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Apply one tag to this account (ACS-371)."""
    target = await _load_target_user(session, user_id)
    normalized = user_tags.normalize_tag(tag)
    if normalized is None:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error=(
                "A tag needs at least one letter or digit; it is lowercased and "
                "may contain only a-z, 0-9 and dashes."
            ),
            status_code=400,
        )
    added = await user_tags.add_tag(
        session, [target.id], normalized, created_by_user_id=admin.id
    )
    log.info(
        "admin_user_tag_added",
        admin_id=str(admin.id),
        user_id=str(target.id),
        tag=normalized,
        # 0 = already had it; the redirect still reads as success, which is the
        # right UX for an idempotent action.
        added=added,
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/users/{user_id}/tags/remove")
async def admin_remove_user_tag(
    request: Request,
    user_id: uuid.UUID,
    tag: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Remove one tag from this account (ACS-371)."""
    target = await _load_target_user(session, user_id)
    # Normalize on the way in too: the chip posts the stored value, but a tag
    # applied before a normalizer change should still be removable.
    normalized = user_tags.normalize_tag(tag)
    if normalized is not None:
        removed = await user_tags.remove_tag(session, [target.id], normalized)
        log.info(
            "admin_user_tag_removed",
            admin_id=str(admin.id),
            user_id=str(target.id),
            tag=normalized,
            removed=removed,
        )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


def _parse_optional_budget(raw: str) -> int | None:
    raw = (raw or "").strip()
    if raw == "":
        return None
    n = int(raw)
    if n < 0:
        raise ValueError("negative")
    return n


@router.post("/admin/keys/{key_id}/budgets")
async def admin_edit_key_budgets(
    request: Request,
    key_id: uuid.UUID,
    monthly_token_budget: str = Form("0"),
    daily_token_budget: str = Form(""),
    monthly_input_token_budget: str = Form(""),
    monthly_output_token_budget: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    ).scalar_one_or_none()
    if api_key is None:
        raise HTTPException(status_code=404, detail="key not found")
    target = await _load_target_user(session, api_key.user_id)
    try:
        # monthly_token_budget is non-nullable in the schema; default 0 = unlimited.
        monthly = int((monthly_token_budget or "0").strip() or 0)
        if monthly < 0:
            raise ValueError
        daily = _parse_optional_budget(daily_token_budget)
        inp = _parse_optional_budget(monthly_input_token_budget)
        out = _parse_optional_budget(monthly_output_token_budget)
    except ValueError:
        return await _render_admin_user_detail(
            request,
            admin,
            target,
            session,
            error="Budgets must be non-negative integers (or blank).",
            status_code=400,
        )
    api_key.monthly_token_budget = monthly
    api_key.daily_token_budget = daily
    api_key.monthly_input_token_budget = inp
    api_key.monthly_output_token_budget = out
    log.info(
        "admin_key_budgets_set",
        admin_id=str(admin.id),
        key_id=str(api_key.id),
        monthly=monthly,
        daily=daily,
        input=inp,
        output=out,
    )
    return RedirectResponse(url=f"/admin/users/{target.id}", status_code=303)


@router.post("/admin/keys/{key_id}/revoke-web")
async def admin_revoke_key_web(
    request: Request,
    key_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    """Cookie-auth'd revoke from the admin user-detail page. The existing
    header-auth ``POST /admin/keys/{id}/revoke`` remains for the CLI."""
    import datetime as _dt

    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    ).scalar_one_or_none()
    if api_key is None:
        raise HTTPException(status_code=404, detail="key not found")
    if api_key.revoked_at is None:
        api_key.revoked_at = _dt.datetime.now(tz=_dt.UTC)
    log.info(
        "admin_key_revoked_web",
        admin_id=str(admin.id),
        key_id=str(api_key.id),
    )
    return RedirectResponse(url=f"/admin/users/{api_key.user_id}", status_code=303)


# --- admin: users page (user list + invite management) ----------------------


def _invite_status(invite: SignupInvite, redemption_count: int) -> str:
    """Derive display status from invite timestamps + how many accounts it created.

    ``max_uses is None`` means unlimited. A capped link is 'used' once its
    redemptions reach the cap; revoke/expire take precedence over usage state.
    """
    if invite.revoked_at is not None:
        return "revoked"
    capped = invite.max_uses is not None and redemption_count >= invite.max_uses
    if capped:
        # Exhausted links read as 'used' regardless of expiry (clearer for admin).
        return "used"
    if dt.datetime.now(tz=dt.UTC) > invite.expires_at:
        return "expired"
    if redemption_count > 0:
        return "accepted"
    return "pending"


async def _build_invite_rows(
    session: AsyncSession, invites_raw: list[SignupInvite], base_url: str
) -> list[dict]:
    """Assemble admin-table rows for a list of invites.

    Batch-loads redemption rows (count + created-account emails) in two queries
    total — one for the redemption rows, one for the user emails — to avoid an
    N+1 across invites. Shared by the GET render and the POST-create render.
    """
    invite_ids = [inv.id for inv in invites_raw]
    redemptions: list[SignupInviteRedemption] = []
    if invite_ids:
        redemptions = list(
            (
                await session.execute(
                    select(SignupInviteRedemption)
                    .where(SignupInviteRedemption.invite_id.in_(invite_ids))
                    .order_by(SignupInviteRedemption.redeemed_at.asc())
                )
            )
            .scalars()
            .all()
        )

    # Resolve every redeemer email in one query.
    user_ids = {r.user_id for r in redemptions if r.user_id is not None}
    email_by_id: dict[uuid.UUID, str] = {}
    if user_ids:
        for uid, em in (
            await session.execute(select(User.id, User.email).where(User.id.in_(user_ids)))
        ).all():
            email_by_id[uid] = em

    redemptions_by_invite: dict[uuid.UUID, list[SignupInviteRedemption]] = {}
    for r in redemptions:
        redemptions_by_invite.setdefault(r.invite_id, []).append(r)

    rows = []
    for inv in invites_raw:
        inv_redemptions = redemptions_by_invite.get(inv.id, [])
        count = len(inv_redemptions)
        # Created-account emails, oldest first; drop deleted users (user_id NULL).
        created_emails = [
            email_by_id[r.user_id]
            for r in inv_redemptions
            if r.user_id is not None and r.user_id in email_by_id
        ]
        rows.append(
            {
                "invite": inv,
                "status": _invite_status(inv, count),
                "max_uses": inv.max_uses,  # None = unlimited
                "redemption_count": count,
                "created_emails": created_emails,
                "invite_url": f"{base_url}/invite/{inv.token}",
            }
        )
    return rows


def _parse_max_uses(raw: str) -> int | None:
    """Parse the usage-cap picker value into ``max_uses``.

    'unlimited' (or empty) → None; otherwise a positive int. Invalid input
    falls back to single-use (1) — the safe, least-surprising default rather
    than accidentally minting an unlimited link.
    """
    raw = (raw or "").strip().lower()
    if raw in ("", "unlimited", "0"):
        return None if raw == "unlimited" else 1
    try:
        n = int(raw)
    except ValueError:
        return 1
    return n if n >= 1 else 1


async def _invite_capacity_reached(session: AsyncSession, invite: SignupInvite) -> bool:
    """Non-locking check: has a capped invite already created max_uses accounts?

    Used on the GET path (informational) and as a fast pre-check on POST. The
    authoritative, race-safe check happens inside the locked transaction in
    ``invite_accept``. Unlimited (``max_uses is None``) never reaches capacity.
    """
    if invite.max_uses is None:
        return False
    from sqlalchemy import func as _func

    count = (
        await session.execute(
            select(_func.count())
            .select_from(SignupInviteRedemption)
            .where(SignupInviteRedemption.invite_id == invite.id)
        )
    ).scalar_one()
    return count >= invite.max_uses


def _parse_emails(raw: str) -> list[str]:
    """Parse a multi-email input (newline/comma/whitespace-separated)."""
    import re as _re

    parts = _re.split(r"[\s,]+", raw.strip())
    return [p.strip().lower() for p in parts if p.strip()]


async def _provision_approved_user(
    session: AsyncSession,
    *,
    user: User,
    settings: Settings,
) -> ApiKey:
    """Create the initial API key + set pending_key_plaintext on an approved user.

    Extracted to be shared by admin_approve_user and invite acceptance.
    The user row must already have status='approved' set by the caller.
    Caller is responsible for committing the session. Returns the created key
    (its id is also stored on ``user.pending_key_id``).
    """
    gk = generate_key()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=gk.hash_,
        key_prefix=gk.prefix,
        name="initial",
        monthly_token_budget=settings.default_per_key_budget,
    )
    session.add(api_key)
    await session.flush()
    user.pending_key_plaintext = _pending_key_serializer(settings.session_secret).dumps(
        gk.plaintext
    )
    user.pending_key_id = api_key.id
    return api_key


_BUCKET_RANK = {"active": 0, "cooling": 1, "churned": 2, "never_activated": 3}


def _sort_user_rows(user_rows: list[dict], sort: str) -> None:
    """Apply the optional roster sort in place (ACS-300).

    Anything outside the four known keys leaves ``_ALL_USERS_ORDER`` intact.
    """
    if sort == "bucket":
        user_rows.sort(
            key=lambda r: (
                _BUCKET_RANK.get(r["eng"].engagement if r["eng"] else "", 4),
                r["eng"].days_since_last
                if r["eng"] and r["eng"].days_since_last is not None
                else 10**6,
            )
        )
    elif sort == "last_seen":
        user_rows.sort(
            key=lambda r: (
                r["eng"].days_since_last
                if r["eng"] and r["eng"].days_since_last is not None
                else 10**6
            )
        )
    elif sort == "tokens":
        user_rows.sort(key=lambda r: -(r["eng"].tokens_30d if r["eng"] else 0))
    elif sort == "weeks":
        user_rows.sort(key=lambda r: -(r["eng"].active_weeks if r["eng"] else 0))


async def _users_page_context(
    request: Request,
    admin: User,
    session: AsyncSession,
    settings: Settings,
    *,
    flash_message: str | None = None,
    flash_error: str | None = None,
    new_invite_url: str | None = None,
    invite_results: list | None = None,
    csv_parse_errors: list[str] | None = None,
    sort_override: str | None = None,
) -> dict:
    """Build the whole ``admin_users.html`` context — the ONE place that does.

    Both render paths (``GET /admin/users`` and the in-place re-render after
    ``POST /admin/users/invite``) used to assemble this independently, and the
    copies had already drifted: the invite path silently dropped the ``?sort=``
    parameter, so creating an invite reset the roster's ordering. Adding the tag
    column and the status/tag filters (ACS-371) to two copies would have tripled
    that debt, so they are unified here first.
    """
    pending = list(
        (
            await session.execute(
                select(User).where(User.status == "pending").order_by(User.created_at.asc())
            )
        )
        .scalars()
        .all()
    )

    # Filters (ACS-371). Applied in SQL for status, post-load for tag — the tag
    # map is needed for rendering anyway, so filtering on it costs no extra query.
    status_filter = (request.query_params.get("status") or "").strip()
    tag_filter = user_tags.normalize_tag(request.query_params.get("tag") or "")

    stmt = select(User)
    if status_filter in USER_STATUSES:
        stmt = stmt.where(User.status == status_filter)
    all_users = list((await session.execute(stmt.order_by(*_ALL_USERS_ORDER))).scalars().all())

    roster = await customer360.load_roster_engagement(session)
    tags_by_user = await user_tags.load_tags_for_users(session, [u.id for u in all_users])
    user_rows = [
        {"user": u, "eng": roster.get(u.id), "tags": tags_by_user.get(u.id, [])} for u in all_users
    ]
    if tag_filter:
        user_rows = [r for r in user_rows if tag_filter in r["tags"]]

    # The invite POST carries the sort as a form field (its form action has no
    # query string), so it is passed in rather than read off the URL.
    sort = sort_override if sort_override is not None else request.query_params.get("sort", "")
    _sort_user_rows(user_rows, sort)

    invites_raw = list(
        (await session.execute(select(SignupInvite).order_by(SignupInvite.created_at.desc())))
        .scalars()
        .all()
    )
    base_url = settings.public_base_url.rstrip("/")
    invite_rows = await _build_invite_rows(session, invites_raw, base_url)

    # Roster-wide counts, deliberately NOT len(all_users) — that is the filtered
    # list, so the stat card would silently start reporting the filter's size.
    total_users = (await session.execute(select(func.count()).select_from(User))).scalar_one()
    stats = {
        "total_users": total_users,
        "pending_signups": len(pending),
        # "Open" = still claimable (pending or partly-used multi-use links).
        "pending_invites": sum(1 for r in invite_rows if r["status"] in ("pending", "accepted")),
        # Total accounts created across all invite links.
        "accepted_invites": sum(r["redemption_count"] for r in invite_rows),
    }

    return {
        "user": admin,
        "pending": pending,
        "users": user_rows,
        "invites": invite_rows,
        "stats": stats,
        "flash_message": flash_message,
        "flash_error": flash_error,
        "new_invite_url": new_invite_url,
        "invite_results": invite_results,
        "csv_parse_errors": csv_parse_errors,
        "beta_discord_url": settings.beta_discord_url,
        # Filter UI state (ACS-371).
        "all_tags": await user_tags.all_tags(session),
        "active_tag": tag_filter or "",
        "active_status": status_filter if status_filter in USER_STATUSES else "",
        # Resolved sort (URL on GET, form field on the invite POST) — the
        # template must read THIS, not request.query_params, or the invite
        # re-render silently drops it again.
        "active_sort": sort,
        # Where the bulk-action POST should send the admin back to, so a
        # sweep over a filtered cohort returns to that same filtered view
        # instead of an unfiltered roster (ACS-372).
        "return_to": str(request.url.path)
        + (f"?{request.url.query}" if request.url.query else ""),
        "user_statuses": sorted(USER_STATUSES),
        "showing_filtered": bool(tag_filter or status_filter in USER_STATUSES),
        "shown_count": len(user_rows),
    }


@router.get("/admin/users")
async def admin_users(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """User management: all users, pending signups, invite management."""
    ctx = await _users_page_context(
        request,
        admin,
        session,
        settings,
        flash_message=request.query_params.get("msg") or None,
        flash_error=request.query_params.get("err") or None,
    )
    return templates.TemplateResponse(request, "admin_users.html", ctx)


@router.post("/admin/users/invite")
async def admin_create_invite(
    request: Request,
    invite_type: str = Form(...),  # 'link' or 'email'
    emails: str = Form(""),
    max_uses: str = Form("1"),  # picker value: positive int or 'unlimited'
    csv_file: UploadFile | None = File(None),
    discord_url: str = Form(""),
    survey_url: str = Form(_DEFAULT_BETA_SURVEY_URL),
    tag: str = Form(""),  # ACS-371: auto-applied to every account this invite creates
    sort: str = Form(""),  # roster sort carried across the POST (form action has no query string)
    admin: User = webauth.AdminRequiredDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    """Create one or more invite tokens from the admin users panel.

    'link' → single link-only invite with the chosen usage cap, shown inline.
    'email' → one invite per parsed email address; sends invite emails. Email
    invites are always single-use (one account per addressed recipient) — see
    ``max_uses_link`` handling below.
    """
    import secrets as _secrets

    expires_at = dt.datetime.now(tz=dt.UTC) + dt.timedelta(days=settings.invite_expiry_days)
    base_url = settings.public_base_url.rstrip("/")
    # Cap applies to the link flow; email invites stay single-use (1).
    link_max_uses = _parse_max_uses(max_uses)
    # Normalize once here so all three invite branches store the same form.
    invite_tag = user_tags.normalize_tag(tag)

    # ---- build the common response context (we'll augment below) ----
    async def _render_users_page(
        *,
        new_invite_url: str | None = None,
        invite_results: list | None = None,
        csv_parse_errors: list[str] | None = None,
        flash_error: str | None = None,
    ):
        ctx = await _users_page_context(
            request,
            admin,
            session,
            settings,
            flash_error=flash_error,
            new_invite_url=new_invite_url,
            invite_results=invite_results,
            csv_parse_errors=csv_parse_errors,
            sort_override=sort,
        )
        return templates.TemplateResponse(request, "admin_users.html", ctx)

    if invite_type == "link":
        token = _secrets.token_urlsafe(32)
        invite = SignupInvite(
            token=token,
            email=None,
            max_uses=link_max_uses,
            created_by_user_id=admin.id,
            expires_at=expires_at,
            tag=invite_tag,
        )
        session.add(invite)
        await session.flush()
        log.info(
            "invite_link_created",
            admin_id=str(admin.id),
            invite_id=str(invite.id),
            max_uses=link_max_uses,
        )
        invite_url = f"{base_url}/invite/{token}"
        return await _render_users_page(new_invite_url=invite_url)

    elif invite_type == "email":
        raw_emails = _parse_emails(emails)
        if not raw_emails:
            return await _render_users_page(flash_error="Please enter at least one email address.")

        # Look up existing users in one query to skip them gracefully.
        existing_users = set(
            row[0]
            for row in (
                await session.execute(select(User.email).where(User.email.in_(raw_emails)))
            ).all()
        )

        results = []
        for email_addr in raw_emails:
            if email_addr in existing_users:
                results.append(
                    {"email": email_addr, "ok": False, "error": "already has an account"}
                )
                continue
            token = _secrets.token_urlsafe(32)
            invite = SignupInvite(
                token=token,
                email=email_addr,
                max_uses=1,  # one account per addressed recipient
                created_by_user_id=admin.id,
                expires_at=expires_at,
                tag=invite_tag,
            )
            session.add(invite)
            await session.flush()
            invite_url = f"{base_url}/invite/{token}"
            log.info(
                "invite_email_created",
                admin_id=str(admin.id),
                invite_id=str(invite.id),
                to_email=email_addr[:3] + "***",
            )
            # Soft-fail: send invite email but never let a failure break the flow.
            await mailermod.send_invite_email(
                settings=settings,
                http=http,
                session=session,
                to_email=email_addr,
                invite_url=invite_url,
                expires_at=expires_at,
            )
            results.append({"email": email_addr, "ok": True, "error": None})

        return await _render_users_page(invite_results=results)

    elif invite_type == "csv_personalized":
        if csv_file is None or not csv_file.filename:
            return await _render_users_page(flash_error="Please choose a CSV file.")

        raw = await csv_file.read(_MAX_INVITE_CSV_BYTES + 1)
        if len(raw) > _MAX_INVITE_CSV_BYTES:
            return await _render_users_page(
                flash_error="CSV is too large. Keep beta invite CSVs under 256 KB."
            )
        try:
            csv_text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return await _render_users_page(flash_error="CSV must be UTF-8 encoded.")

        parsed_rows, parse_errors = _parse_personalized_invite_csv(csv_text)
        if not parsed_rows:
            return await _render_users_page(
                flash_error="No valid invite rows found.",
                csv_parse_errors=parse_errors,
            )

        existing_users = set(
            row[0]
            for row in (
                await session.execute(
                    select(User.email).where(User.email.in_([r.email for r in parsed_rows]))
                )
            ).all()
        )

        # The form marks this required, but a scripted POST could omit it and
        # mail an empty Discord link (the personalized template renders it
        # unconditionally). Fall back to the configured invite, else refuse.
        clean_discord_url = discord_url.strip() or (settings.beta_discord_url or "")
        if not clean_discord_url:
            return await _render_users_page(
                flash_error=(
                    "A Discord invite URL is required for personalized invites "
                    "(set BETA_DISCORD_URL or fill the field)."
                )
            )
        clean_survey_url = survey_url.strip()
        results = []
        for row in parsed_rows:
            if row.email in existing_users:
                results.append(
                    {
                        "email": row.email,
                        "ok": False,
                        "error": "already has an account",
                    }
                )
                continue

            token = _secrets.token_urlsafe(32)
            invite = SignupInvite(
                token=token,
                email=row.email,
                max_uses=1,
                created_by_user_id=admin.id,
                expires_at=expires_at,
                tag=invite_tag,
            )
            session.add(invite)
            await session.flush()
            invite_url = f"{base_url}/invite/{token}"
            log.info(
                "personalized_beta_invite_created",
                admin_id=str(admin.id),
                invite_id=str(invite.id),
                to_email=row.email[:3] + "***",
                line_number=row.line_number,
                survey_respondent=row.survey_respondent,
            )
            await mailermod.send_personalized_beta_invite_email(
                settings=settings,
                http=http,
                session=session,
                to_email=row.email,
                recipient_name=row.name,
                invite_url=invite_url,
                expires_at=expires_at,
                survey_respondent=row.survey_respondent,
                discord_url=clean_discord_url,
                survey_url=clean_survey_url,
            )
            results.append({"email": row.email, "ok": True, "error": None})

        return await _render_users_page(
            invite_results=results,
            csv_parse_errors=parse_errors,
        )

    else:
        return await _render_users_page(flash_error=f"Unknown invite_type: {invite_type!r}")


@router.post("/admin/invites/{invite_id}/revoke")
async def admin_revoke_invite(
    invite_id: uuid.UUID,
    admin: User = webauth.AdminRequiredDep,
    session: AsyncSession = Depends(get_session),
):
    invite = (
        await session.execute(select(SignupInvite).where(SignupInvite.id == invite_id))
    ).scalar_one_or_none()
    if invite is None:
        raise HTTPException(status_code=404, detail="invite not found")
    # Revoking is idempotent; a partly-used multi-use link can still be revoked
    # to stop further accounts being created.
    if invite.revoked_at is None:
        invite.revoked_at = dt.datetime.now(tz=dt.UTC)
        log.info("invite_revoked", admin_id=str(admin.id), invite_id=str(invite.id))
    from urllib.parse import urlencode

    return RedirectResponse(
        url="/admin/users?" + urlencode({"msg": "Invite revoked."}), status_code=303
    )


# --- admin: models page (model lifecycle + warm windows) --------------------


@router.get("/invite/{token}")
@limiter.limit("20/minute")
async def invite_form(
    request: Request,
    token: str,
    session: AsyncSession = Depends(get_session),
):
    """Show the invite signup form. Validates the token but does not consume it."""
    invite = (
        await session.execute(select(SignupInvite).where(SignupInvite.token == token))
    ).scalar_one_or_none()

    def _invalid(reason: str):
        return templates.TemplateResponse(
            request,
            "invite_invalid.html",
            {"user": None, "reason": reason},
            status_code=410,
        )

    if invite is None:
        return _invalid("This invite link is invalid or doesn't exist.")
    if invite.revoked_at is not None:
        return _invalid("This invite has been revoked by an admin.")
    if dt.datetime.now(tz=dt.UTC) > invite.expires_at:
        return _invalid(
            f"This invite expired on {invite.expires_at.strftime('%Y-%m-%d %H:%M UTC')}."
        )
    if await _invite_capacity_reached(session, invite):
        return _invalid("This invite link is no longer available — it has reached its usage limit.")

    return templates.TemplateResponse(
        request,
        "invite.html",
        {
            "user": None,
            "token": token,
            "suggested_email": invite.email,
            "email": invite.email or "",
            "name": "",
            "org": "",
            "expires_at": invite.expires_at,
            "max_uses": invite.max_uses,  # None = unlimited
            "error": None,
        },
    )


@router.post("/invite/{token}")
@limiter.limit("5/minute")
async def invite_accept(
    request: Request,
    token: str,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    name: str = Form(""),
    org: str = Form(""),
    agree: str = Form(""),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Accept an invite: validate token, create approved user, mint key, auto-login."""
    if not settings.session_secret:
        raise HTTPException(
            status_code=503, detail="SESSION_SECRET not configured — cannot process invite."
        )

    def _invalid(reason: str):
        return templates.TemplateResponse(
            request,
            "invite_invalid.html",
            {"user": None, "reason": reason},
            status_code=410,
        )

    invite = (
        await session.execute(select(SignupInvite).where(SignupInvite.token == token))
    ).scalar_one_or_none()
    if invite is None:
        return _invalid("This invite link is invalid or doesn't exist.")
    if invite.revoked_at is not None:
        return _invalid("This invite has been revoked by an admin.")
    if dt.datetime.now(tz=dt.UTC) > invite.expires_at:
        return _invalid(
            f"This invite expired on {invite.expires_at.strftime('%Y-%m-%d %H:%M UTC')}."
        )
    # Fast pre-check (non-locking) — the authoritative check is under FOR UPDATE.
    if await _invite_capacity_reached(session, invite):
        return _invalid("This invite link is no longer available — it has reached its usage limit.")

    email_n = email.strip().lower()
    name_n = name.strip() or None
    org_n = org.strip() or None

    def _form_error(msg: str):
        return templates.TemplateResponse(
            request,
            "invite.html",
            {
                "user": None,
                "token": token,
                "suggested_email": invite.email,
                "email": email_n,
                "name": name_n or "",
                "org": org_n or "",
                "expires_at": invite.expires_at,
                "max_uses": invite.max_uses,  # None = unlimited
                "error": msg,
            },
            status_code=400,
        )

    if password != confirm_password:
        return _form_error("Passwords don't match.")
    if len(password.encode("utf-8")) < webauth.PASSWORD_MIN_LEN:
        return _form_error(f"Password must be at least {webauth.PASSWORD_MIN_LEN} characters.")
    if len(password.encode("utf-8")) > webauth.PASSWORD_MAX_LEN:
        return _form_error(f"Password must be no more than {webauth.PASSWORD_MAX_LEN} characters.")
    # Mandatory usage-rules agreement — same consent gate as /signup (ACS-209);
    # the shared _ground_rules.html partial renders the checkbox on both forms.
    if not agree:
        return _form_error("Please agree to the usage rules to continue.")

    # Check email uniqueness.
    existing = (
        await session.execute(select(User).where(User.email == email_n))
    ).scalar_one_or_none()
    if existing is not None:
        return _form_error(
            "That email address is already registered. Use a different email or sign in instead."
        )

    # Create the approved user.
    new_user = User(
        email=email_n,
        name=name_n,
        org=org_n,
        status="approved",
        approved_at=dt.datetime.now(tz=dt.UTC),
        agreed_terms_at=dt.datetime.now(tz=dt.UTC),
        monthly_token_budget_total=settings.default_monthly_token_budget_total,
    )
    try:
        await webauth.set_password(session, new_user, password)
    except ValueError as exc:
        return _form_error(str(exc))
    session.add(new_user)
    await session.flush()

    # Provision the initial key (one-shot delivery via /dashboard).
    await _provision_approved_user(session, user=new_user, settings=settings)

    # Auto-tag from the invite (ACS-371): a cohort labels itself at claim time,
    # so nobody has to remember to tag people afterwards. Read from the
    # unlocked row — the tag is set at creation and never mutated, so it can't
    # race with the FOR UPDATE cap check below.
    if invite.tag:
        await user_tags.add_tag(
            session, [new_user.id], invite.tag, created_by_user_id=invite.created_by_user_id
        )

    # Race-safe cap enforcement. Lock the invite row (SELECT ... FOR UPDATE) so
    # concurrent accepts on the same link serialize here: each waits for the
    # other's transaction to commit before it can count redemptions, so the
    # count-then-insert can't exceed max_uses. (NULL max_uses = unlimited.)
    from sqlalchemy import func as _func

    locked = (
        await session.execute(
            select(SignupInvite).where(SignupInvite.id == invite.id).with_for_update()
        )
    ).scalar_one()
    now = dt.datetime.now(tz=dt.UTC)
    if locked.max_uses is not None:
        used = (
            await session.execute(
                select(_func.count())
                .select_from(SignupInviteRedemption)
                .where(SignupInviteRedemption.invite_id == locked.id)
            )
        ).scalar_one()
        if used >= locked.max_uses:
            # Cap reached while we held the form open (or a concurrent accept
            # won the last slot). We created the user row above; roll it back.
            await session.rollback()
            return _invalid(
                "This invite link is no longer available — it has reached its usage limit."
            )

    # Record the redemption; stamp the legacy accepted_* columns on the FIRST use.
    session.add(SignupInviteRedemption(invite_id=locked.id, user_id=new_user.id, redeemed_at=now))
    if locked.accepted_at is None:
        locked.accepted_at = now
        locked.accepted_by_user_id = new_user.id

    # Auto-login: create a session and set the cookie.
    us = await webauth.create_user_session(
        session,
        new_user.id,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if (request.client and settings.log_ip) else None,
    )
    cookie_value = webauth.sign_session_cookie(us.id, settings.session_secret)

    log.info(
        "invite_accepted",
        invite_id=str(invite.id),
        user_id=str(new_user.id),
        email_prefix=email_n[:3] + "***",
    )

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        webauth.COOKIE_NAME,
        cookie_value,
        max_age=settings.session_max_age_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    return response
