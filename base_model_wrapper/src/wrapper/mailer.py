"""Transactional email — used by /admin approve+reject and friends.

Named ``mailer`` (not ``email``) to avoid shadowing stdlib ``email``. Provider
is Resend; SMTP fallback intentionally not implemented.

Soft-fail by design: when ``settings.email_enabled`` is False or
``settings.resend_api_key`` is missing, ``send`` returns a 'skipped' outcome
without making an HTTP call. Network/HTTP errors during a real send are caught
and returned as a 'failed' outcome but never raised — admin approval flow must
always succeed even if the email provider is down. Re-notifying a user is
cheap; losing their approval is not.

Every attempt is also persisted: the high-level ``send_approval_email`` /
``send_rejection_email`` helpers write an ``EmailLog`` row (full HTML + text
body, send outcome, Resend message id) into the request session, so the admin
Emails page is a complete audit of what the app tried to send. The low-level
``send`` stays DB-free and returns its outcome, which keeps it unit-testable
and makes "record" a single guaranteed step in the helpers callers actually use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .logging import get_logger
from .models import EmailLog

if TYPE_CHECKING:
    import datetime as dt

    from sqlalchemy.ext.asyncio import AsyncSession

    from .models import User
    from .settings import Settings

log = get_logger()

_EMAIL_TEMPLATES_DIR = Path(__file__).parent / "templates" / "email"
_email_env = Environment(
    loader=FileSystemLoader(str(_EMAIL_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _render(name: str, **ctx: object) -> str:
    return _email_env.get_template(name).render(**ctx)


@dataclass
class SendOutcome:
    """Result of one ``send`` attempt — what gets recorded on the EmailLog row.

    ``status`` is our send outcome ('sent' | 'skipped' | 'failed'), distinct
    from later *delivery* status which arrives via webhooks.
    """

    status: str
    skip_reason: str | None = None
    error: str | None = None
    http_status: int | None = None
    resend_message_id: str | None = None


async def send(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    subject: str,
    to: str,
    html: str,
    text: str,
    from_email: str | None = None,
    cc: Sequence[str] | None = None,
    bcc: Sequence[str] | None = None,
    reply_to: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> SendOutcome:
    """Send one email via Resend. Soft-fail: never raises.

    Returns a :class:`SendOutcome` describing what happened. Returns early
    without an HTTP call (status 'skipped') when ``email_enabled`` is False or
    no ``resend_api_key`` is configured — logged at WARN so the operator sees a
    notification was suppressed.

    ``from_email`` defaults to ``settings.email_from``; the bulk-email flow
    (ACS-228) overrides it per row (any address on the Resend-verified domain,
    e.g. ``ivar@acsresearch.org``) and may add cc/bcc/reply-to.
    """
    if not settings.email_enabled:
        log.warning(
            "email_skipped", reason="email_disabled", to_prefix=to[:3] + "***", subject=subject
        )
        return SendOutcome(status="skipped", skip_reason="email_disabled")
    if not settings.resend_api_key:
        log.warning("email_skipped", reason="no_api_key", to_prefix=to[:3] + "***", subject=subject)
        return SendOutcome(status="skipped", skip_reason="no_api_key")
    payload: dict[str, object] = {
        "from": from_email or settings.email_from,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": text,
    }
    if cc:
        payload["cc"] = list(cc)
    if bcc:
        payload["bcc"] = list(bcc)
    if reply_to:
        payload["reply_to"] = reply_to
    if headers:
        # e.g. RFC 8058 List-Unsubscribe / List-Unsubscribe-Post on bulk mail.
        payload["headers"] = dict(headers)
    try:
        resp = await http.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {settings.resend_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15.0,
        )
        if resp.status_code >= 400:
            log.warning(
                "email_send_failed",
                status=resp.status_code,
                body=resp.text[:300],
                to_prefix=to[:3] + "***",
                subject=subject,
            )
            return SendOutcome(
                status="failed", http_status=resp.status_code, error=resp.text[:1000]
            )
        message_id: str | None = None
        try:
            parsed = resp.json()
            mid = parsed.get("id") if isinstance(parsed, dict) else None
            message_id = mid if isinstance(mid, str) else None
        except Exception:  # malformed body — still a success, just no id to join on
            message_id = None
        log.info("email_sent", to_prefix=to[:3] + "***", subject=subject, message_id=message_id)
        return SendOutcome(
            status="sent", http_status=resp.status_code, resend_message_id=message_id
        )
    except Exception as exc:  # network, timeout, unexpected — never propagate
        log.warning("email_send_failed", error=repr(exc), to_prefix=to[:3] + "***", subject=subject)
        return SendOutcome(status="failed", error=repr(exc))


def _record_email(
    session: "AsyncSession | None",
    *,
    kind: str,
    user: "User | None",
    to: str,
    from_email: str,
    subject: str,
    html: str,
    text: str,
    outcome: SendOutcome,
) -> None:
    """Add an EmailLog row for this attempt to the request session.

    The row commits with the surrounding request transaction (e.g. the approval
    that triggered it), keeping send-record and approval atomic. ``session`` is
    optional only so unit tests can exercise rendering without a DB.
    """
    if session is None:
        return
    session.add(
        EmailLog(
            user_id=getattr(user, "id", None),
            kind=kind,
            to_email=to,
            from_email=from_email,
            subject=subject,
            body_html=html,
            body_text=text,
            send_status=outcome.status,
            skip_reason=outcome.skip_reason,
            error=outcome.error,
            http_status=outcome.http_status,
            resend_message_id=outcome.resend_message_id,
        )
    )


# Placeholder substituted for the live set-password URL before the approval
# email body is persisted — same single-use-secret rationale as the reset link
# above (an over-shared email_logs row must not hand out a working set-password
# link). Only present for password-less approvals.
_SET_PASSWORD_LINK_REDACTED = "[set-password link redacted from log]"


def _format_minutes_human(minutes: int | None) -> str:
    """Render a token lifetime as a short human string ("7 days", "60 minutes")."""
    if not minutes:
        return ""
    if minutes % (24 * 60) == 0:
        days = minutes // (24 * 60)
        return f"{days} day" if days == 1 else f"{days} days"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


async def send_approval_email(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    user: "User",
    dashboard_url: str,
    tutorial_url: str | None = None,
    available_models: Sequence[Mapping[str, object]] | None = None,
    set_password_url: str | None = None,
    set_password_expiry_minutes: int | None = None,
    discord_url: str | None = None,
    discord_connect_url: str | None = None,
) -> None:
    subject = "Your ACS Infra account has been approved"
    base_url = getattr(settings, "public_base_url", "http://localhost:8000").rstrip("/")
    ctx = {
        "user_name": user.name or user.email,
        "dashboard_url": dashboard_url,
        "tutorial_url": tutorial_url or f"{base_url}/tutorial",
        "discord_url": discord_url,
        # Preferred over the raw invite: sends them to the dashboard's
        # Connect-Discord button so the account gets mapped (ACS-310).
        "discord_connect_url": discord_connect_url,
        "available_models": list(available_models or []),
        # Only set for password-less accounts (admin-approve of a user who never
        # set a password) — the template shows a "set your password" CTA so they
        # aren't forced into the "forgot password" dance (ACS-99).
        "set_password_url": set_password_url,
        "set_password_expiry": _format_minutes_human(set_password_expiry_minutes),
    }
    html = _render("approval.html", **ctx)
    text = _render("approval.txt", **ctx)
    outcome = await send(
        settings=settings, http=http, subject=subject, to=user.email, html=html, text=text
    )
    # Redact the single-use set-password link from the persisted EmailLog.
    if set_password_url:
        redacted_ctx = {**ctx, "set_password_url": _SET_PASSWORD_LINK_REDACTED}
        rec_html = _render("approval.html", **redacted_ctx)
        rec_text = _render("approval.txt", **redacted_ctx)
    else:
        rec_html, rec_text = html, text
    _record_email(
        session,
        kind="approval",
        user=user,
        to=user.email,
        from_email=settings.email_from,
        subject=subject,
        html=rec_html,
        text=rec_text,
        outcome=outcome,
    )


async def send_signup_notification(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    user: "User",
    admin_url: str,
    reapplication: bool = False,
) -> None:
    """Notify the team that a public /signup application is pending review (ACS-24).

    Sent to ``settings.signup_notify_email`` so admins don't have to poll
    /admin/users. Soft-fails like every other send (never raises) and records an
    EmailLog row. The body carries no secret link, so nothing is redacted.
    """
    to = settings.signup_notify_email
    subject = (
        f"Re-application pending review: {user.email}"
        if reapplication
        else f"New ACS Infra signup pending review: {user.email}"
    )
    ctx = {
        "email": user.email,
        "applicant_name": user.name,
        "org": user.org,
        "profile_link": user.signup_profile_link,
        "use_case": user.signup_use_case,
        "outcome": user.signup_outcome,
        "prior_work": user.signup_prior_work,
        "referral": user.signup_referral,
        "source": user.signup_source,
        "admin_url": admin_url,
        # Re-application after a rejection (ACS-312) — previous answers are in
        # the user's admin notes.
        "reapplication": reapplication,
    }
    html = _render("signup_notification.html", **ctx)
    text = _render("signup_notification.txt", **ctx)
    outcome = await send(settings=settings, http=http, subject=subject, to=to, html=html, text=text)
    _record_email(
        session,
        kind="signup_notification",
        user=user,
        to=to,
        from_email=settings.email_from,
        subject=subject,
        html=html,
        text=text,
        outcome=outcome,
    )


# Placeholder substituted for the live reset URL before the email body is
# persisted. The reset link is a single-use secret (see ``EmailLog`` docstring's
# "do NOT log bodies that carry secrets" warning): the REAL body — with the live
# URL — goes to Resend, but the EmailLog row stores this redacted copy so a DB
# leak can't hand an attacker a working reset link.
_RESET_LINK_REDACTED = "[reset link redacted from log]"


async def send_password_reset_email(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    user: "User",
    reset_url: str,
    first_time: bool = False,
) -> None:
    """Email a single-use password-reset link, recording a REDACTED EmailLog.

    ``first_time`` swaps the copy for an account that has never had a password
    (invite-onboarded, or approved after a password-less signup) — "set" rather
    than "reset" (ACS-313).

    Soft-fail like the rest (never raises) but always records the attempt for
    audit. The recorded body has ``reset_url`` swapped for ``_RESET_LINK_REDACTED``
    so the secret link never lands in the database — only the live send carries
    the real URL.
    """
    subject = "Set your ACS Infra password" if first_time else "Reset your ACS Infra password"
    ctx = {
        "user_name": user.name or user.email,
        "reset_url": reset_url,
        "first_time": first_time,
        "expiry_minutes": settings.password_reset_expiry_minutes,
    }
    # Live body — the real link goes out over the wire to Resend.
    html = _render("password_reset.html", **ctx)
    text = _render("password_reset.txt", **ctx)
    outcome = await send(
        settings=settings, http=http, subject=subject, to=user.email, html=html, text=text
    )
    # Redacted body — render the same templates with the URL replaced, so the
    # persisted EmailLog never contains the token. (We re-render rather than
    # string-replace so a template that splits the URL across attributes can't
    # leak a fragment.)
    redacted_ctx = {**ctx, "reset_url": _RESET_LINK_REDACTED}
    redacted_html = _render("password_reset.html", **redacted_ctx)
    redacted_text = _render("password_reset.txt", **redacted_ctx)
    _record_email(
        session,
        kind="password_reset",
        user=user,
        to=user.email,
        from_email=settings.email_from,
        subject=subject,
        html=redacted_html,
        text=redacted_text,
        outcome=outcome,
    )


async def send_invite_email(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    to_email: str,
    invite_url: str,
    expires_at: "dt.datetime",
) -> None:
    """Send an invite email containing the magic signup link.

    ``user`` is None here because the recipient doesn't have an account yet —
    the EmailLog row is still recorded for audit, with ``user_id=None``.
    Soft-fail: a send failure never raises and never breaks invite creation.

    NOTE: the invite URL contains a secret token; we deliberately do NOT log
    the body HTML/text in a way that exposes the link beyond the EmailLog row,
    which is admin-only. The EmailLog body DOES include the URL (same as
    approval emails include the dashboard URL); treat the email_logs table as
    sensitive accordingly.
    """
    subject = "You're invited to ACS Infra"
    ctx = {
        "invite_url": invite_url,
        "expires_at": expires_at.strftime("%Y-%m-%d %H:%M UTC"),
    }
    html = _render("invite.html", **ctx)
    text = _render("invite.txt", **ctx)
    outcome = await send(
        settings=settings, http=http, subject=subject, to=to_email, html=html, text=text
    )
    _record_email(
        session,
        kind="invite",
        user=None,
        to=to_email,
        from_email=settings.email_from,
        subject=subject,
        html=html,
        text=text,
        outcome=outcome,
    )


async def send_personalized_beta_invite_email(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    to_email: str,
    recipient_name: str | None,
    invite_url: str,
    expires_at: "dt.datetime",
    survey_respondent: bool,
    discord_url: str,
    survey_url: str,
) -> None:
    """Send one combined beta outreach email with a personal invite link.

    This is distinct from ``send_invite_email``: the generic helper is the
    short transactional platform invite, while this helper is the human-facing
    beta invitation sent from CSV campaigns. It still records an EmailLog row
    with ``user_id=None`` because the account does not exist yet.
    """
    subject = "Invitation to ACS Infra beta"
    base_url = getattr(settings, "public_base_url", "http://localhost:8000").rstrip("/")
    ctx = {
        "recipient_name": recipient_name or to_email,
        "invite_url": invite_url,
        "expires_at": expires_at.strftime("%Y-%m-%d %H:%M UTC"),
        "survey_respondent": survey_respondent,
        "discord_url": discord_url,
        "survey_url": survey_url,
        "tutorial_url": f"{base_url}/tutorial",
        "workbench_url": f"{base_url}/workbench",
    }
    html = _render("personalized_beta_invite.html", **ctx)
    text = _render("personalized_beta_invite.txt", **ctx)
    outcome = await send(
        settings=settings, http=http, subject=subject, to=to_email, html=html, text=text
    )
    _record_email(
        session,
        kind="invite_personalized",
        user=None,
        to=to_email,
        from_email=settings.email_from,
        subject=subject,
        html=html,
        text=text,
        outcome=outcome,
    )


async def send_rejection_email(
    *,
    settings: "Settings",
    http: httpx.AsyncClient,
    session: "AsyncSession | None",
    user: "User",
) -> None:
    subject = "Update on your ACS Infra application"
    ctx = {"user_name": user.name or user.email}
    html = _render("rejection.html", **ctx)
    text = _render("rejection.txt", **ctx)
    outcome = await send(
        settings=settings, http=http, subject=subject, to=user.email, html=html, text=text
    )
    _record_email(
        session,
        kind="rejection",
        user=user,
        to=user.email,
        from_email=settings.email_from,
        subject=subject,
        html=html,
        text=text,
        outcome=outcome,
    )
