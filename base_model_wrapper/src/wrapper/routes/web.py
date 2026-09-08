"""Public web, account, self-service key, and feedback routes."""

from __future__ import annotations

import datetime as dt
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from itsdangerous import BadData, BadSignature, URLSafeSerializer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import mailer as mailermod
from .. import web_auth as webauth
from ..db import get_session
from ..dependencies import get_http, get_settings
from ..keys import generate as generate_key
from ..logging import get_logger
from ..models import (
    ApiKey,
    EmailOptOut,
    Feedback,
    FeedbackScreenshot,
    UpdateSubscriber,
    UsageMonthly,
    User,
)
from ..rate_limit import limiter
from ..routes.admin import _pending_key_serializer
from ..settings import Settings

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_FAVICON_SVG = _STATIC_DIR / "favicon.svg"

log = get_logger()
router = APIRouter()

# In-app feedback limits. Screenshots are stored inline as Postgres bytea (no
# object storage; single-replica), so we keep them small and few. Domain
# constants, not config — these are product decisions, not deployment knobs.
_FEEDBACK_CATEGORIES = ("bug", "feature", "general")
_FEEDBACK_MAX_SCREENSHOTS = 5
_FEEDBACK_MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MB per image
_FEEDBACK_MAX_DESCRIPTION_LEN = 5000
# Allowlist of safe raster image types. Deliberately excludes image/svg+xml:
# an SVG can carry <script>, and screenshots are served back to admins, so a
# permissive "image/*" check let a stored SVG run in the admin origin (XSS).
_FEEDBACK_ALLOWED_IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)

# --- public pages: /, /tutorial ----------------------------------------------


@router.get("/")
async def landing(request: Request, user: User | None = webauth.CurrentUserDep):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"user": user, "subscribed": request.query_params.get("subscribed") == "1"},
    )


@router.post("/subscribe")
@limiter.limit("5/minute")
async def subscribe_updates(
    request: Request,
    email: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    """Landing-page "subscribe to updates" field (ACS-228).

    Stores the lowercased address with its opt-in timestamp. Idempotent — the
    INSERT is ON CONFLICT DO NOTHING (no check-then-act race on the unique
    index), and the response never says whether the address was known (no
    enumeration surface).
    """
    from .admin.bulk_emails import EMAIL_RE

    addr = email.strip().lower()
    if not EMAIL_RE.match(addr):
        return RedirectResponse(url="/?subscribed=0#subscribe", status_code=303)
    await session.execute(
        pg_insert(UpdateSubscriber)
        .values(email=addr, source="landing")
        .on_conflict_do_nothing(index_elements=["email"])
    )
    log.info("update_subscriber_added", email_prefix=addr[:3] + "***")
    return RedirectResponse(url="/?subscribed=1#subscribe", status_code=303)


def _optout_email_or_none(token: str, settings: Settings) -> str | None:
    if not settings.session_secret:
        return None
    from .admin.bulk_emails import email_from_optout_token

    return email_from_optout_token(token, settings.session_secret)


@router.get("/unsubscribe/{token}")
async def unsubscribe_confirm(
    request: Request,
    token: str,
    settings: Settings = Depends(get_settings),
):
    """Unsubscribe confirmation page (ACS-228). No login required; NO mutation.

    The GET only renders a confirm button — mail-provider link scanners
    (SafeLinks and friends) prefetch every URL in a delivered email, and a
    state-mutating GET would silently opt recipients out. The actual opt-out
    happens in the POST below. The token is the recipient's address signed
    with the app secret (salt 'email-optout') — unguessable, no expiry: the
    link keeps working for as long as the email sits in an inbox.
    """
    addr = _optout_email_or_none(token, settings)
    if addr is None:
        return templates.TemplateResponse(
            request,
            "unsubscribe.html",
            {"user": None, "email": None, "confirmed": False},
            status_code=404,
        )
    return templates.TemplateResponse(
        request, "unsubscribe.html", {"user": None, "email": addr, "confirmed": False}
    )


@router.post("/unsubscribe/{token}")
async def unsubscribe_submit(
    request: Request,
    token: str,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Record the opt-out. Target of both the confirm button and RFC 8058
    one-click POSTs (mail providers POST here via the List-Unsubscribe-Post
    header, body ignored). ON CONFLICT DO NOTHING — double clicks are fine."""
    addr = _optout_email_or_none(token, settings)
    if addr is None:
        return templates.TemplateResponse(
            request,
            "unsubscribe.html",
            {"user": None, "email": None, "confirmed": False},
            status_code=404,
        )
    await session.execute(
        pg_insert(EmailOptOut)
        .values(email=addr, source="link")
        .on_conflict_do_nothing(index_elements=["email"])
    )
    log.info("email_optout_added", email_prefix=addr[:3] + "***")
    return templates.TemplateResponse(
        request, "unsubscribe.html", {"user": None, "email": addr, "confirmed": True}
    )


# --- /tutorial multi-page docs site (ACS-89) ---------------------------------
#
# The /tutorial surface was a single 480-line template until ACS-89; it's now
# a Modal-style docs site built from markdown files under wrapper/docs/. The
# routes below dispatch into the pre-rendered RenderedDoc store hung off
# app.state.tutorial_pages by lifespan.py.
#
# URL contract:
#   GET /tutorial                  → Overview + legacy-fragment redirect script
#   GET /tutorial/quick-start      → 301 → /tutorial/overview#quick-start
#                                     (preserves the one inbound link Modal-
#                                     style sites care about most)
#   GET /tutorial/{slug}           → top-level page (models, api, account,
#                                     examples [as an index], or "overview"
#                                     as an explicit alias of /tutorial)
#   GET /tutorial/{section}/{slug} → a sub-folder page (examples/…, workbench/…)
#
# Public route — ACS-68 mandated the docs be readable without an account; the
# template branches on ``user`` for the private-beta banner / authed CTAs.

# Legacy fragment-redirect map: the single-page /tutorial used in-page anchors
# (#logprobs, #prompt-logprobs, …). Those URLs exist in wikis / Slack / browser
# bookmarks, and the new layout has dedicated pages for each. This map is the
# source of truth — rendered into the Overview page's <head> as a JS object so
# the redirect happens before first paint (avoids a flash of Overview content
# when the user follows a legacy link).
LEGACY_FRAGMENT_MAP: dict[str, str] = {
    "logprobs": "/tutorial/examples/logprobs",
    "prompt-logprobs": "/tutorial/examples/prompt-logprobs",
    "echo": "/tutorial/examples/echo",
    "stream": "/tutorial/examples/stream",
    "batch-rollouts": "/tutorial/examples/batch-rollouts",
    "cold-boot": "/tutorial/examples/cold-boot",
    "budget-cap": "/tutorial/examples/budget-cap",
    "quick-start": "/tutorial/overview#quick-start",
}


def _render_tutorial_page(
    request: Request,
    user: User | None,
    page_slug: str,
    *,
    include_legacy_map: bool = False,
) -> Response:
    """Look up ``page_slug`` in app.state.tutorial_pages and render it.

    Raises ``HTTPException(404)`` if the slug isn't in the pre-rendered store —
    pages are validated at boot, so this only fires on URL typos.
    """
    pages: dict = request.app.state.tutorial_pages
    nav = request.app.state.tutorial_nav
    page = pages.get(page_slug)
    if page is None:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        request,
        "tutorial_page.html",
        {
            "user": user,
            "page": page,
            "nav": nav,
            "current_slug": page_slug,
            "legacy_fragment_map": LEGACY_FRAGMENT_MAP if include_legacy_map else None,
        },
    )


@router.get("/tutorial")
async def tutorial(request: Request, user: User | None = webauth.CurrentUserDep):
    """Canonical home for the docs site. Renders Overview + redirect script.

    Function name preserved (``tutorial``, not ``tutorial_root``) so the
    ``main.py`` compatibility export ``tutorial = web_routes.tutorial``
    keeps working — older tests reference that symbol directly.
    """
    return _render_tutorial_page(request, user, "overview", include_legacy_map=True)


@router.get("/tutorial/quick-start")
async def tutorial_quick_start_alias():
    """301 alias for the old single-page anchor.

    The legacy ``/tutorial#quick-start`` URL still works (handled JS-side by
    the redirect map on /tutorial); this is the server-side equivalent for
    ``/tutorial/quick-start`` as a literal path — e.g. someone pasting the
    fragment as a path component, or a curl probe.
    """
    return RedirectResponse(url="/tutorial/overview#quick-start", status_code=301)


@router.get("/tutorial/{section}/{slug}")
async def tutorial_subpage(
    section: str,
    slug: str,
    request: Request,
    user: User | None = webauth.CurrentUserDep,
):
    """A sub-folder page — e.g. ``examples/logprobs`` or ``workbench/loom``.

    Generic over the section so a new docs sub-folder routes without a new
    handler (mirrors ``docs_build._doc_paths``). ``_render_tutorial_page`` 404s
    on an unknown ``section/slug``.
    """
    return _render_tutorial_page(request, user, f"{section}/{slug}")


@router.get("/llms.txt", include_in_schema=False)
async def llms_txt(request: Request):
    """Whole tutorial as one plain-markdown page, for LLMs.

    Served at the conventional ``/llms.txt`` path (and linked from every
    tutorial page via a visible link + a ``<link rel="alternate">``) so an
    assistant handed a tutorial URL by a human can grab the full docs in one
    fetch. Generated from the same markdown as the rendered pages (see
    ``docs_build.build_combined_markdown``) — no second copy to maintain.
    """
    return PlainTextResponse(
        request.app.state.tutorial_combined,
        media_type="text/markdown; charset=utf-8",
    )


@router.get("/favicon.svg", include_in_schema=False)
@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the site favicon.

    Browsers auto-request ``/favicon.ico`` on every page; without this route
    that was the one 404 in the app's console. We ship a single SVG and serve
    it at both ``/favicon.svg`` and ``/favicon.ico`` (the ``.ico`` path returns
    SVG bytes with an SVG media type — every current browser accepts that).
    """
    return FileResponse(_FAVICON_SVG, media_type="image/svg+xml")


# --- shareable example artifacts (ACS-206) ------------------------------------
# Worked-example notebooks/scripts the tutorial links to. Served from the app
# because the GitHub repo is private — users can't follow repo links. Public,
# like /tutorial (nothing sensitive; the artifacts are documentation).
_EXAMPLES_DIR = _STATIC_DIR / "examples"
_EXAMPLE_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    # .py served as text so browsers display instead of download-prompting.
    ".py": "text/plain; charset=utf-8",
}


@router.get("/examples/{filename}", include_in_schema=False)
async def example_artifact(filename: str):
    """Serve one example artifact by filename (flat directory, no nesting)."""
    path = (_EXAMPLES_DIR / filename).resolve()
    media = _EXAMPLE_MEDIA_TYPES.get(path.suffix)
    if (
        media is None
        or not str(path).startswith(str(_EXAMPLES_DIR.resolve()) + "/")
        or not path.is_file()
    ):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=300"})


@router.get("/tutorial/{slug}")
async def tutorial_top(
    slug: str,
    request: Request,
    user: User | None = webauth.CurrentUserDep,
):
    """Top-level docs page — overview / models / api / account / examples (index)."""
    return _render_tutorial_page(request, user, slug)


# --- web auth: login / logout / dashboard ------------------------------------


def _session_misconfigured() -> JSONResponse:
    """Returned by /login etc. when SESSION_SECRET isn't set in env."""
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "Web auth is not configured. Set SESSION_SECRET env var.",
                "code": "session_misconfigured",
            }
        },
    )


@router.get("/login")
async def login_form(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
):
    if user is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "user": None,
            "error": None,
            "email": "",
            # Success flash carried via ?msg= (e.g. after a password reset).
            "flash_message": request.query_params.get("msg") or None,
        },
    )


# --- password reset (forgot password) ----------------------------------------

# Neutral, enumeration-safe response shown for ANY email on POST /forgot-password
# — identical whether or not the address maps to a real account.
_FORGOT_PW_NEUTRAL = "If an account exists for that email, we've sent a reset link."


@router.get("/forgot-password")
async def forgot_password_form(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
):
    if user is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        request,
        "forgot_password.html",
        {"user": None, "submitted": False, "email": ""},
    )


@router.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password_submit(
    request: Request,
    email: str = Form(...),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    """Always render the same neutral confirmation (no account enumeration).

    Only when the email maps to a real **approved** user do we actually mint a
    token + send mail; otherwise we render the identical page and do nothing.
    """
    email_n = email.strip().lower()
    user = (await session.execute(select(User).where(User.email == email_n))).scalar_one_or_none()
    # A password-less *approved* account must be recoverable too (ACS-313):
    # admin-created rows, and after ACS-314 every public applicant (invite
    # acceptance sets a password on the spot, so those aren't affected). Their
    # approval-email set-password link expires after a week, and without this
    # the only way back is asking an admin to resend it.
    #
    # Recoverability keys off status ALONE (ACS-353). It used to be
    # ``password_hash is not None or status == 'approved'``, whose first clause
    # short-circuited the status test — so a previously-approved-then-rejected
    # user who still had a password kept getting reset emails, contradicting
    # the rule stated right here. Only approved accounts have access to recover;
    # pending / rejected / suspended rows get nothing.
    first_time = user is not None and user.password_hash is None
    recoverable = user is not None and user.status == "approved"
    if recoverable:
        token = await webauth.create_password_reset(
            session, user, expiry_minutes=settings.password_reset_expiry_minutes
        )
        reset_url = settings.public_base_url.rstrip("/") + f"/reset-password/{token}"
        # Soft-fail send; the EmailLog row records a REDACTED body (the real
        # link only goes to Resend), per the secret-bearing-email rule.
        await mailermod.send_password_reset_email(
            settings=settings,
            http=http,
            session=session,
            user=user,
            reset_url=reset_url,
            first_time=first_time,
        )
        log.info("password_reset_requested", user_id=str(user.id), first_time=first_time)
    else:
        # Unknown / passwordless account: do nothing, but render the same page.
        log.info("password_reset_no_account", email_prefix=email_n[:3] + "***")
    return templates.TemplateResponse(
        request,
        "forgot_password.html",
        {"user": None, "submitted": True, "email": ""},
    )


def _reset_invalid_response(request: Request, *, first_time: bool = False) -> Response:
    return templates.TemplateResponse(
        request,
        "reset_password.html",
        {"user": None, "invalid": True, "token": "", "error": None, "first_time": first_time},
        status_code=400,
    )


@router.get("/reset-password/{token}")
async def reset_password_form(
    request: Request,
    token: str,
    session: AsyncSession = Depends(get_session),
):
    # A token whose user has no password yet is an initial set-password
    # (admin-approve / ACS-99), not a reset — drives friendlier copy.
    target = await webauth.peek_password_reset(session, token)
    if target is None:
        return _reset_invalid_response(request)
    first_time = target.password_hash is None
    return templates.TemplateResponse(
        request,
        "reset_password.html",
        {"user": None, "invalid": False, "token": token, "error": None, "first_time": first_time},
    )


@router.post("/reset-password/{token}")
@limiter.limit("5/minute")
async def reset_password_submit(
    request: Request,
    token: str,
    password: str = Form(...),
    confirm_password: str = Form(...),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    # First-time set-password (password-less user, ACS-99) vs true reset, for copy.
    peeked = await webauth.peek_password_reset(session, token)
    first_time = peeked is not None and peeked.password_hash is None

    def _err(msg: str):
        return templates.TemplateResponse(
            request,
            "reset_password.html",
            {
                "user": None,
                "invalid": False,
                "token": token,
                "error": msg,
                "first_time": first_time,
            },
            status_code=400,
        )

    if password != confirm_password:
        return _err("Passwords don't match.")
    if len(password.encode("utf-8")) < webauth.PASSWORD_MIN_LEN:
        return _err(f"Password must be at least {webauth.PASSWORD_MIN_LEN} characters.")
    if len(password.encode("utf-8")) > webauth.PASSWORD_MAX_LEN:
        return _err(f"Password must be no more than {webauth.PASSWORD_MAX_LEN} characters.")

    # Atomic single-use consume; returns the user only if the token was valid
    # AND this call won the race to mark it used.
    user = await webauth.consume_password_reset(session, token)
    if user is None:
        return _reset_invalid_response(request, first_time=first_time)

    try:
        await webauth.set_password(session, user, password)
    except ValueError as exc:
        return _err(str(exc))
    # Revoke existing web sessions so a leaked cookie can't outlive the reset.
    await webauth.revoke_all_user_sessions(session, user.id)
    log.info("password_reset_completed", user_id=str(user.id), first_time=first_time)
    msg = (
        "Password+set.+Sign+in+with+your+new+password."
        if first_time
        else "Password+reset.+Sign+in+with+your+new+password."
    )
    return RedirectResponse(url=f"/login?msg={msg}", status_code=303)


# --- public signup -----------------------------------------------------------


def _signup_disabled_response() -> Response:
    return Response(status_code=404)


# Channel-attribution cap (ACS-210): the ?src= tag is a short slug we chose
# ourselves ("constellation", "cyborgism"); anything longer is someone playing
# with the URL — keep whatever fits, drop the rest.
_SIGNUP_SOURCE_MAX_LEN = 64

# Server-side mirror of the use-case textarea's maxlength in signup.html —
# keep the two in sync. The HTML attribute only binds honest browsers; a
# crafted POST could otherwise store an unbounded blob in the Text column.
_USE_CASE_MAX_LEN = 2000


def _normalize_signup_source(raw: str) -> str | None:
    # Second strip: truncation can leave interior whitespace at the new tail.
    return raw.strip()[:_SIGNUP_SOURCE_MAX_LEN].strip() or None


# Bot guards (ACS-260): a visually hidden honeypot input ("website") plus a
# signed render-timestamp — a human can't fill seven required fields in under
# 3 s, and naive form-fillers stuff every input they see. Either trip ends in
# the same 303 a real submission gets, with no row and no notification, so the
# bot can't tell what caught it.
_SIGNUP_TS_SALT = "acs-signup-ts-v1"
_SIGNUP_MIN_FORM_SECONDS = 3.0


def _sign_signup_ts(settings: Settings) -> str | None:
    """Signed epoch-seconds token embedded in the rendered signup form, or
    ``None`` when no ``session_secret`` is configured (nothing to sign with;
    the time-trap is skipped on submit too)."""
    if not settings.session_secret:
        return None
    return URLSafeSerializer(settings.session_secret, salt=_SIGNUP_TS_SALT).dumps(int(time.time()))


def _signup_form_age(settings: Settings, token: str) -> float | None:
    """Seconds elapsed since the form carrying ``token`` was rendered, or
    ``None`` when the token is missing or forged."""
    try:
        rendered = URLSafeSerializer(settings.session_secret or "", salt=_SIGNUP_TS_SALT).loads(
            token
        )
        return time.time() - float(rendered)
    # BadData covers BadSignature AND its sibling BadPayload (valid signature,
    # corrupt payload) — unreachable today, but belt-and-braces (review #266).
    except (BadData, ValueError, TypeError):
        return None


@router.get("/signup")
async def signup_form(
    request: Request,
    settings: Settings = Depends(get_settings),
    user: User | None = webauth.CurrentUserDep,
):
    if not settings.signup_enabled:
        # Friendly explanation instead of a bare 404: we're in private beta, so
        # tell visitors how to ask for access rather than dead-ending them. The
        # form itself is suppressed (signup_disabled) so no account can be made.
        return templates.TemplateResponse(
            request,
            "signup.html",
            {
                "user": None,
                "submitted": False,
                "signup_disabled": True,
                "error": None,
                "email": "",
                "name": "",
                "org": "",
            },
        )
    if user is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    submitted = request.query_params.get("ok") == "1"
    # A re-application (ACS-312) deliberately ignores the submitted password.
    resubmitted = submitted and request.query_params.get("again") == "1"
    return templates.TemplateResponse(
        request,
        "signup.html",
        {
            "user": None,
            "submitted": submitted,
            "resubmitted": resubmitted,
            "error": None,
            "email": "",
            "name": "",
            "org": "",
            "use_case": "",
            "profile_link": "",
            "outcome": "",
            "prior_work": "",
            "referral": "",
            # Channel attribution (ACS-210): the ?src= tag on the link we post
            # per community rides into the form as a hidden field.
            "src": _normalize_signup_source(request.query_params.get("src", "")) or "",
            # Time-trap token (ACS-260): when it was rendered, signed.
            "ts": _sign_signup_ts(settings),
        },
    )


# One copied answer can be 2,000 chars and each reopen appends five of them,
# so three honest cycles would blow the 20,000-char cap in the admin notes
# editor and lock admins out of saving that row (review finding).
_NOTES_ANSWER_EXCERPT = 300


def _notes_excerpt(value: str | None) -> str:
    """One copied answer, trimmed and indented for the notes history block.

    Every line is indented: a flush-left continuation line could otherwise
    forge an entry indistinguishable from a real one written by the approve
    flow, e.g. "2026-01-04 (approved): vetted by …" (review finding).
    """
    if not value:
        return "    — not provided"
    text = value.strip()
    if len(text) > _NOTES_ANSWER_EXCERPT:
        text = text[:_NOTES_ANSWER_EXCERPT].rstrip() + "… (full text in the application above)"
    return "\n".join("    " + line for line in text.splitlines())


def _reopen_rejected_application(
    user: User,
    *,
    name: str | None,
    org: str | None,
    use_case: str,
    profile_link: str | None,
    outcome: str | None,
    prior_work: str | None,
    referral: str | None,
    src: str | None,
) -> User:
    """Turn a rejected account back into a pending application (ACS-312).

    The previous answers and the rejection date are preserved as a dated
    ``notes`` entry (the ACS-300 convention) so the record survives and the
    reviewer can compare both versions.

    Credentials are deliberately untouched: accepting the submitted password
    here would let anyone who knows a rejected address re-apply with their own
    password and, once approved, sign in as that identity. The caller also
    refuses to reopen anything but a plain ``role='user'`` row.
    """
    now = dt.datetime.now(tz=dt.UTC)
    header = f"{now.strftime('%Y-%m-%d')} — re-applied" + (
        f" (previously rejected {user.rejected_at.strftime('%Y-%m-%d')})"
        if user.rejected_at
        else " (previously rejected)"
    )
    if user.approved_at:
        header += f"; had been approved {user.approved_at.strftime('%Y-%m-%d')}"
    previous = [
        header,
        "  Previous answers:",
        "    Planned usage:",
        _notes_excerpt(user.signup_use_case),
        "    Results:",
        _notes_excerpt(user.signup_outcome),
        "    Prior work:",
        _notes_excerpt(user.signup_prior_work),
        "    Heard about us:",
        _notes_excerpt(user.signup_referral),
        "    Profile link:",
        _notes_excerpt(user.signup_profile_link),
    ]
    entry = "\n".join(previous)
    user.notes = entry if not user.notes else f"{entry}\n\n{user.notes}"

    user.name = name
    user.org = org
    user.signup_use_case = use_case
    user.signup_profile_link = profile_link
    user.signup_outcome = outcome
    user.signup_prior_work = prior_work
    user.signup_referral = referral
    user.signup_source = src
    user.agreed_terms_at = now
    user.status = "pending"
    user.rejected_at = None
    # A stale approval stamp would render "Approved <date>" beside a pending
    # badge on the very screen where the decision is made; the history now
    # lives in the notes entry above.
    user.approved_at = None
    user.approved_by_user_id = None
    return user


@router.post("/signup")
# 5/minute bounds a burst; the stacked 20/day bounds a slow drip (5/minute
# alone allows 7,200 signups/day from one IP). Same in-memory per-IP bucket.
@limiter.limit("5/minute;20/day")
async def signup_submit(
    request: Request,
    email: str = Form(...),
    name: str = Form(""),
    org: str = Form(""),
    profile_link: str = Form(""),
    use_case: str = Form(""),
    outcome: str = Form(""),
    prior_work: str = Form(""),
    referral: str = Form(""),
    agree: str = Form(""),
    src: str = Form(""),
    website: str = Form(""),  # honeypot (ACS-260) — humans never see it
    ts: str = Form(""),  # signed render-timestamp (ACS-260)
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    if not settings.signup_enabled:
        return _signup_disabled_response()

    email_n = email.strip().lower()
    name_n = name.strip() or None
    org_n = org.strip() or None
    profile_link_n = profile_link.strip() or None
    use_case_n = use_case.strip()
    outcome_n = outcome.strip() or None
    prior_work_n = prior_work.strip() or None
    referral_n = referral.strip() or None
    src_n = _normalize_signup_source(src)

    # Bot guards (ACS-260), checked before any validation so a tripped submit
    # is indistinguishable from a real one: same 303, no row, no notification.
    # The rare human false-positive (a sub-3 s submit) sees "submitted" and
    # nothing arrives — acceptable; a genuine applicant will follow up.
    trip: str | None = None
    if website.strip():
        trip = "honeypot"
    elif settings.session_secret:
        age = _signup_form_age(settings, ts)
        if age is None:
            trip = "form_token"
        elif age < _SIGNUP_MIN_FORM_SECONDS:
            trip = "too_fast"
    if trip is not None:
        log.info(
            "signup_bot_dropped",
            reason=trip,
            email_prefix=email_n[:3] + "***",
            signup_source=src_n,
        )
        return RedirectResponse(url="/signup?ok=1", status_code=303)

    def _err(msg: str, code: int = 400):
        return templates.TemplateResponse(
            request,
            "signup.html",
            {
                "user": None,
                "submitted": False,
                "error": msg,
                "email": email_n,
                "name": name_n or "",
                "org": org_n or "",
                "use_case": use_case_n,
                "profile_link": profile_link_n or "",
                "outcome": outcome_n or "",
                "prior_work": prior_work_n or "",
                "referral": referral_n or "",
                "src": src_n or "",
                # Pass the original render-token through so a corrected
                # resubmit still measures from the first render (ACS-260).
                "ts": ts,
            },
            status_code=code,
        )

    # Mandatory usage-rules agreement (ACS-170) + identity + a use-case to
    # evaluate (ACS-24). Name/org are required so a reviewer always has a
    # who-is-this signal — pseudonyms and "Independent" are explicitly fine.
    # Checked before the DB lookup so an incomplete form fails cheaply.
    if not agree:
        return _err("Please agree to the usage rules to continue.")
    if not name_n:
        return _err("Please enter your name — a publicly used pseudonym is fine.")
    if not org_n:
        return _err("Please enter your organization — “Independent” is fine.")
    if not use_case_n:
        return _err("Please tell us what you'd run — the Planned usage question.")
    for label, answer, cap in (
        ("Planned usage", use_case_n, _USE_CASE_MAX_LEN),
        ("Results", outcome_n, _USE_CASE_MAX_LEN),
        ("Prior work", prior_work_n, _USE_CASE_MAX_LEN),
        ("How did you hear about us?", referral_n, _USE_CASE_MAX_LEN),
        ("Profile link", profile_link_n, 500),
    ):
        if answer and len(answer) > cap:
            return _err(
                f"Please keep “{label}” to {cap} characters or fewer "
                f"(yours is {len(answer)})."
            )

    # Generic response on duplicate email — don't leak whether the address is
    # already registered. The admin sees the dupe at INFO level in logs.
    # Exception: a *rejected* application can be re-submitted (ACS-312) — we
    # tell declined applicants they may come back with a stronger case, so the
    # product has to honour that without an admin deleting their row.
    existing = (
        await session.execute(select(User).where(User.email == email_n))
    ).scalar_one_or_none()
    # Only a plain rejected *user* row reopens. Roles survive rejection, so
    # without this an anonymous signup could push a rejected admin row back
    # into the approval queue for an unsuspecting one-click approve (review
    # finding); those fall through to the generic duplicate response.
    reopenable = (
        existing is not None and existing.status == "rejected" and existing.role == "user"
    )
    if existing is not None and not reopenable:
        log.info("signup_duplicate_email", email_prefix=email_n[:3] + "***")
        return _err(
            "Could not create account. If you already have one, sign in instead.",
        )

    if reopenable:
        user = _reopen_rejected_application(
            existing,
            name=name_n,
            org=org_n,
            use_case=use_case_n,
            profile_link=profile_link_n,
            outcome=outcome_n,
            prior_work=prior_work_n,
            referral=referral_n,
            src=src_n,
        )
        await session.flush()
        log.info(
            "signup_reapplied",
            user_id=str(user.id),
            email_prefix=email_n[:3] + "***",
            signup_source=src_n,
        )
        admin_url = settings.public_base_url.rstrip("/") + f"/admin/users/{user.id}"
        await mailermod.send_signup_notification(
            settings=settings,
            http=http,
            session=session,
            user=user,
            admin_url=admin_url,
            reapplication=True,
        )
        return RedirectResponse(url="/signup?ok=1&again=1", status_code=303)

    user = User(
        email=email_n,
        name=name_n,
        org=org_n,
        status="pending",
        signup_use_case=use_case_n,
        signup_profile_link=profile_link_n,
        signup_outcome=outcome_n,
        signup_prior_work=prior_work_n,
        signup_referral=referral_n,
        agreed_terms_at=dt.datetime.now(tz=dt.UTC),
        signup_source=src_n,
    )
    # No password at application time (ACS-314): approval mints a single-use
    # set-password link to the account address, so access requires that mailbox.
    session.add(user)
    await session.flush()
    log.info(
        "signup_submitted",
        user_id=str(user.id),
        email_prefix=email_n[:3] + "***",
        signup_source=src_n,
    )
    # Notify the team so they don't have to poll /admin/users (ACS-24). Soft-fails
    # internally — a mailer hiccup must never block the applicant's submission.
    admin_url = settings.public_base_url.rstrip("/") + f"/admin/users/{user.id}"
    await mailermod.send_signup_notification(
        settings=settings, http=http, session=session, user=user, admin_url=admin_url
    )
    return RedirectResponse(url="/signup?ok=1", status_code=303)


@router.post("/login")
@limiter.limit("5/minute")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    if not settings.session_secret:
        return _session_misconfigured()
    email = email.strip().lower()
    result = await webauth.authenticate_user(session, email, password)
    if result is None:
        log.info("login_failed", email_prefix=email[:3] + "***")
        return templates.TemplateResponse(
            request,
            "login.html",
            {"user": None, "error": "Invalid email or password.", "email": email},
            status_code=401,
        )
    if isinstance(result, webauth.LoginBlocked):
        if result.reason == "pending":
            msg = "Your account is pending admin approval. You'll get an email when it's reviewed."
        elif result.reason == "rejected":
            msg = (
                "Your account application was not approved. "
                "Email infra@acsresearch.org if this is a mistake."
            )
        elif result.reason == "suspended":
            # Suspension is reversible and often scheduled (a time-boxed cohort),
            # so the copy points at a way back rather than reading as a verdict.
            msg = (
                "Your account access is currently suspended. "
                "Email infra@acsresearch.org if you'd like it restored."
            )
        else:
            msg = "Your account is not currently active."
        log.info("login_blocked", email_prefix=email[:3] + "***", reason=result.reason)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"user": None, "error": msg, "email": email},
            status_code=403,
        )
    user = result
    us = await webauth.create_user_session(
        session,
        user.id,
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if (request.client and settings.log_ip) else None,
    )
    cookie_value = webauth.sign_session_cookie(us.id, settings.session_secret)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        webauth.COOKIE_NAME,
        cookie_value,
        max_age=settings.session_max_age_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )
    log.info("login_success", user_id=str(user.id), email=user.email)
    return response


@router.post("/logout")
async def logout(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    cookie = request.cookies.get(webauth.COOKIE_NAME)
    if cookie and settings.session_secret:
        sid = webauth.verify_session_cookie(cookie, settings.session_secret)
        if sid is not None:
            await webauth.revoke_user_session(session, sid)
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(webauth.COOKIE_NAME)
    return response


async def _dashboard_keys(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, Any]]:
    """Fetch the user's API keys + this-month's usage. One row per key, newest first.

    Used by /dashboard (item 4) and the /me/keys management surface (item 5).
    """
    rows = (
        (
            await session.execute(
                select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    period_start = authmod._current_period_start()
    out: list[dict[str, Any]] = []
    for k in rows:
        usage = (
            await session.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == k.id,
                    UsageMonthly.period_start == period_start,
                )
            )
        ).scalar_one_or_none()
        used = (usage.tokens_prompt + usage.tokens_completion) if usage else 0
        out.append(
            {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "name": k.name,
                "created_at": k.created_at,
                "last_used_at": k.last_used_at,
                "monthly_token_budget": k.monthly_token_budget,
                "tokens_used_this_month": used,
                "revoked_at": k.revoked_at,
                "disabled_at": k.disabled_at,
            }
        )
    return out


async def _render_dashboard(
    request: Request,
    user: User,
    session: AsyncSession,
    *,
    pw_error: str | None = None,
    pw_success: str | None = None,
    key_error: str | None = None,
    status_code: int = 200,
):
    settings = request.app.state.settings
    keys = await _dashboard_keys(session, user.id)
    aggregate_used = sum(k["tokens_used_this_month"] for k in keys)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "keys": keys,
            "aggregate_used": aggregate_used,
            "pw_error": pw_error,
            "pw_success": pw_success,
            "key_error": key_error,
            "default_budget": settings.default_per_key_budget,
            # Account-level aggregate cap (None = unlimited). The create-key form
            # surfaces this so users know per-key "Unbounded" is still governed.
            "account_limit": user.monthly_token_budget_total,
            # Community card (ACS-269): Connect-Discord button when the OAuth
            # flow is configured; ?discord=<status> carries callback outcomes.
            "discord_oauth_enabled": settings.discord_oauth_enabled,
            "beta_discord_url": settings.beta_discord_url,
            # Deep link to the server for members who already joined — the
            # discord.gg invite is for *joining*, not opening (ACS-306).
            "discord_server_url": (
                f"https://discord.com/channels/{settings.discord_guild_id}"
                if settings.discord_guild_id
                else None
            ),
            "discord_status": request.query_params.get("discord"),
        },
        status_code=status_code,
    )


@router.get("/dashboard")
async def dashboard(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    # One-shot post-approval key delivery: if the admin approved this user and
    # an encrypted plaintext is sitting on the row, decrypt it, render the
    # key_created partial once, then clear both columns in the same transaction.
    if user.pending_key_plaintext and settings.session_secret:
        try:
            plaintext = _pending_key_serializer(settings.session_secret).loads(
                user.pending_key_plaintext
            )
        except BadSignature:
            # Session_secret rotated since approval — key unrecoverable. Clear
            # the columns and log; admin can revoke + reissue.
            log.warning(
                "pending_key_decrypt_failed",
                user_id=str(user.id),
                key_id=str(user.pending_key_id) if user.pending_key_id else None,
            )
            user.pending_key_plaintext = None
            user.pending_key_id = None
            return await _render_dashboard(
                request,
                user,
                session,
                key_error="Initial API key could not be recovered. Create a new one below or contact an admin.",
            )
        pending_key_id = user.pending_key_id
        api_key = (
            await session.execute(select(ApiKey).where(ApiKey.id == pending_key_id))
        ).scalar_one_or_none()
        user.pending_key_plaintext = None
        user.pending_key_id = None
        log.info(
            "pending_key_revealed",
            user_id=str(user.id),
            key_id=str(api_key.id) if api_key else None,
        )
        return templates.TemplateResponse(
            request,
            "key_created.html",
            {
                "user": user,
                "plaintext_key": plaintext,
                "key_prefix": api_key.key_prefix if api_key else "—",
                "name": api_key.name if api_key else "initial",
                "key_id": api_key.id if api_key else None,
                "monthly_token_budget": api_key.monthly_token_budget if api_key else 0,
                "via_approval": True,
                # Discord ask on the key-reveal screen (ACS-310).
                "discord_oauth_enabled": settings.discord_oauth_enabled,
            },
        )
    return await _render_dashboard(request, user, session)


# --- self-service key management ---------------------------------------------


@router.post("/me/keys")
async def me_create_key(
    request: Request,
    name: str = Form(""),
    # "unbounded" (checkbox/toggle) means the key imposes no per-key sub-cap —
    # represented as monthly_token_budget == 0, the legacy "unlimited" sentinel.
    # The account aggregate (users.monthly_token_budget_total) still governs
    # such a key, enforced independently in AuthedCaller.effective_remaining().
    unbounded: str = Form(""),
    budget: int | None = Form(None),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    # Default UX is now "Unbounded" — the key has no own sub-cap and is governed
    # only by the account aggregate. An explicit number is honoured but clamped
    # to the account limit (a per-key cap above the account total is meaningless,
    # and must never let a key escape the aggregate).
    is_unbounded = (unbounded or "").strip().lower() in {"on", "true", "1", "yes"}
    if is_unbounded or budget is None:
        budget = 0  # 0 = unlimited per-key (account aggregate still applies)
    else:
        budget = max(0, min(int(budget), 1_000_000_000))
        account_cap = user.monthly_token_budget_total
        if account_cap is not None and budget > account_cap:
            budget = account_cap
    gk = generate_key()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=gk.hash_,
        key_prefix=gk.prefix,
        name=name.strip() or None,
        monthly_token_budget=budget,
    )
    session.add(api_key)
    await session.flush()
    log.info(
        "self_service_key_created", user_id=str(user.id), key_id=str(api_key.id), prefix=gk.prefix
    )
    return templates.TemplateResponse(
        request,
        "key_created.html",
        {
            "user": user,
            "plaintext_key": gk.plaintext,
            "key_prefix": gk.prefix,
            "name": api_key.name,
            "key_id": api_key.id,
            "monthly_token_budget": budget,
        },
    )


@router.post("/me/keys/{key_id}/rename")
async def me_rename_key(
    request: Request,
    key_id: uuid.UUID,
    name: str = Form(""),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Rename an existing key the logged-in user owns. Owner-only; 404 (not 403)
    for keys the user doesn't own so other users' ids aren't enumerable."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    ).scalar_one_or_none()
    if api_key is None:
        return await _render_dashboard(
            request, user, session, key_error="Key not found.", status_code=404
        )
    api_key.name = (name or "").strip()[:64] or None
    log.info("self_service_key_renamed", user_id=str(user.id), key_id=str(api_key.id))
    return RedirectResponse(url="/dashboard", status_code=303)


@router.post("/me/keys/{key_id}/pause")
async def me_pause_key(
    request: Request,
    key_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Reversibly pause (disable) a key. Distinct from revoke — see resume."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    ).scalar_one_or_none()
    if api_key is None:
        return await _render_dashboard(
            request, user, session, key_error="Key not found.", status_code=404
        )
    # Don't pause an already-revoked key; pausing is idempotent otherwise.
    if api_key.revoked_at is None and api_key.disabled_at is None:
        api_key.disabled_at = dt.datetime.now(tz=dt.UTC)
        log.info("self_service_key_paused", user_id=str(user.id), key_id=str(api_key.id))
    return RedirectResponse(url="/dashboard", status_code=303)


@router.post("/me/keys/{key_id}/resume")
async def me_resume_key(
    request: Request,
    key_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Resume a paused key. No-op for a revoked key (permanent)."""
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    ).scalar_one_or_none()
    if api_key is None:
        return await _render_dashboard(
            request, user, session, key_error="Key not found.", status_code=404
        )
    if api_key.revoked_at is None and api_key.disabled_at is not None:
        api_key.disabled_at = None
        log.info("self_service_key_resumed", user_id=str(user.id), key_id=str(api_key.id))
    return RedirectResponse(url="/dashboard", status_code=303)


@router.post("/me/keys/{key_id}/revoke")
async def me_revoke_key(
    request: Request,
    key_id: uuid.UUID,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    import datetime as dt

    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    ).scalar_one_or_none()
    # 404 (not 403) so other users' key IDs aren't enumerable.
    if api_key is None:
        return await _render_dashboard(
            request,
            user,
            session,
            key_error="Key not found.",
            status_code=404,
        )
    if api_key.revoked_at is None:
        api_key.revoked_at = dt.datetime.now(tz=dt.UTC)
        log.info("self_service_key_revoked", user_id=str(user.id), key_id=str(api_key.id))
    return RedirectResponse(url="/dashboard", status_code=303)


@router.post("/me/password")
async def me_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    if not webauth.verify_password(current_password, user.password_hash):
        return await _render_dashboard(
            request,
            user,
            session,
            pw_error="Current password is incorrect.",
            status_code=401,
        )
    if new_password != confirm_password:
        return await _render_dashboard(
            request,
            user,
            session,
            pw_error="New password and confirmation don't match.",
            status_code=400,
        )
    try:
        await webauth.set_password(session, user, new_password)
    except ValueError as e:
        return await _render_dashboard(
            request,
            user,
            session,
            pw_error=str(e),
            status_code=400,
        )
    log.info("self_service_password_changed", user_id=str(user.id))
    return await _render_dashboard(request, user, session, pw_success="Password updated.")


# --- in-app feedback (user submit) -------------------------------------------


def _feedback_truthy(value: str | None) -> bool:
    """Checkbox-style form value → bool. HTML checkboxes send 'on'; the modal's
    fetch builds a FormData that may send 'true'/'1'. Anything else is False."""
    return (value or "").strip().lower() in {"on", "true", "1", "yes"}


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    category: str = Form(...),
    description: str = Form(...),
    is_anonymous: str = Form(""),
    page_path: str = Form(""),
    screenshots: list[UploadFile] = File(default=[]),
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Accept an in-app feedback submission from the floating modal.

    Called via ``fetch`` from the browser, so failures return JSON (not a
    redirect) — a 401 when not logged in, a 400 with a clear message on
    validation failure, ``{"ok": true}`` on success. Screenshots are stored
    inline as bytea (capped per ``_FEEDBACK_MAX_*``).
    """
    if user is None:
        return JSONResponse({"error": "You must be logged in to send feedback."}, status_code=401)

    category = (category or "").strip().lower()
    if category not in _FEEDBACK_CATEGORIES:
        return JSONResponse(
            {"error": f"Category must be one of {', '.join(_FEEDBACK_CATEGORIES)}."},
            status_code=400,
        )

    description = (description or "").strip()
    if not description:
        return JSONResponse({"error": "Description is required."}, status_code=400)
    if len(description) > _FEEDBACK_MAX_DESCRIPTION_LEN:
        return JSONResponse(
            {"error": f"Description exceeds {_FEEDBACK_MAX_DESCRIPTION_LEN} characters."},
            status_code=400,
        )

    # FastAPI sends a single empty UploadFile for an absent file field in some
    # multipart bodies; drop entries without a filename before counting.
    files = [f for f in (screenshots or []) if f is not None and f.filename]
    if len(files) > _FEEDBACK_MAX_SCREENSHOTS:
        return JSONResponse(
            {"error": f"At most {_FEEDBACK_MAX_SCREENSHOTS} screenshots allowed."},
            status_code=400,
        )

    image_blobs: list[tuple[str, bytes]] = []
    for f in files:
        ctype = (f.content_type or "").split(";", 1)[0].strip().lower()
        if ctype not in _FEEDBACK_ALLOWED_IMAGE_TYPES:
            return JSONResponse(
                {"error": "Screenshots must be PNG, JPEG, GIF, or WebP images."},
                status_code=400,
            )
        data = await f.read()
        if len(data) > _FEEDBACK_MAX_IMAGE_BYTES:
            return JSONResponse(
                {"error": "Each screenshot must be 2 MB or smaller."},
                status_code=400,
            )
        image_blobs.append((ctype, data))

    anonymous = _feedback_truthy(is_anonymous)
    fb = Feedback(
        user_id=None if anonymous else user.id,
        category=category,
        description=description,
        is_anonymous=anonymous,
        user_agent=request.headers.get("user-agent"),
        page_path=(page_path or "").strip() or None,
    )
    session.add(fb)
    await session.flush()
    for content_type, data in image_blobs:
        session.add(
            FeedbackScreenshot(feedback_id=fb.id, content_type=content_type, image_bytes=data)
        )
    log.info(
        "feedback_submitted",
        feedback_id=str(fb.id),
        category=category,
        anonymous=anonymous,
        n_screenshots=len(image_blobs),
        page_path=fb.page_path,
    )
    return JSONResponse({"ok": True})
