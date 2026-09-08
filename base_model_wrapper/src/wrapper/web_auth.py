"""Email + password web authentication + server-side sessions.

Two distinct auth surfaces in this codebase:

- **API auth** (``auth.py``): ``Authorization: Bearer acs-bm-...`` for the
  ``/v1/*`` endpoints. Used by programmatic clients.
- **Web auth** (this module): email + password at ``/login`` → signed cookie
  carrying a ``user_sessions.id`` → protected dashboard, chat, and admin routes.

The same ``users`` row backs both. A user logs into the dashboard, then
manages the API keys that authorise their CLI / scripts.

Password storage: bcrypt via passlib, ``users.password_hash`` (LargeBinary,
nullable for legacy rows). Cookies signed with itsdangerous; sessions
server-side so admins (and ``/logout``) can revoke without rotating the cookie
key.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import secrets
import uuid
from dataclasses import dataclass

import bcrypt
from fastapi import Depends, Request
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .logging import get_logger

from .db import get_session
from .models import PasswordReset, User, UserSession

COOKIE_NAME = "acs_session"
COOKIE_SALT = "acs-session-v1"
PENDING_KEY_SALT = "acs-pending-key-v1"


log = get_logger()


@dataclass(frozen=True)
class LoginBlocked:
    """Sentinel returned by ``authenticate_user`` when the password was correct
    but the account isn't currently usable (signup not yet approved, or rejected).

    Distinct from a None return — None means "wrong credentials" and gets a
    generic 'Invalid email or password' message; LoginBlocked carries a reason
    so the caller can surface a discriminating message.
    """

    reason: str  # 'pending' | 'rejected' | 'suspended' (any non-'approved' status)


# Bcrypt has a hard 72-byte input limit. Cap at the form layer too (PASSWORD_MAX_LEN).
PASSWORD_MAX_LEN = 72
PASSWORD_MIN_LEN = 8


# --- password hashing -------------------------------------------------------


def hash_password(plaintext: str) -> bytes:
    """Bcrypt-hash a plaintext password; result is the bytes stored in users.password_hash.

    Rounds default to bcrypt's recommendation (12), ~100 ms / hash — slow enough
    to throttle brute force, fast enough to not bother a logged-in user.

    Raises ValueError if the password is shorter than ``PASSWORD_MIN_LEN`` or
    longer than ``PASSWORD_MAX_LEN`` (bcrypt's hard 72-byte cap). The /login
    POST handler catches the same lengths before reaching this function so
    the user sees a form error rather than a 500.
    """
    n = len(plaintext.encode("utf-8"))
    if n < PASSWORD_MIN_LEN:
        raise ValueError(f"Password shorter than {PASSWORD_MIN_LEN} chars")
    if n > PASSWORD_MAX_LEN:
        raise ValueError(f"Password exceeds bcrypt's {PASSWORD_MAX_LEN}-byte limit")
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt())


def verify_password(plaintext: str, password_hash: bytes | None) -> bool:
    """Constant-time-ish bcrypt verify. ``None`` hash always returns False —
    legacy users without a set password can't log in.
    """
    if password_hash is None:
        return False
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), password_hash)
    except (ValueError, TypeError):
        return False


# Pre-computed dummy hash used to burn the same ~100 ms as a real verify when
# the email is unknown. Stops timing-based account enumeration.
_DUMMY_HASH = bcrypt.hashpw(b"never-matches-anything-real", bcrypt.gensalt())


def _dummy_verify() -> None:
    bcrypt.checkpw(b"x", _DUMMY_HASH)


def generate_temp_password(length: int = 16) -> str:
    """Generate a URL-safe temp password for admin-issued accounts.

    16 chars of urlsafe is ~96 bits — fine for a one-time login that the user
    rotates immediately. The admin emails this; the user changes it on first
    login (or doesn't, that's their problem).
    """
    return secrets.token_urlsafe(length)


# --- cookie sign / verify ---------------------------------------------------


def sign_session_cookie(session_id: uuid.UUID, secret: str) -> str:
    return URLSafeSerializer(secret, salt=COOKIE_SALT).dumps(str(session_id))


def verify_session_cookie(token: str, secret: str) -> uuid.UUID | None:
    try:
        raw = URLSafeSerializer(secret, salt=COOKIE_SALT).loads(token)
        return uuid.UUID(raw)
    except (BadSignature, ValueError, TypeError):
        return None


# --- user + session DB helpers ----------------------------------------------


async def authenticate_user(
    session: AsyncSession, email: str, plaintext_password: str
) -> User | LoginBlocked | None:
    """Look up a user by email and verify the password.

    Returns:
        - ``User`` on success (status == 'approved').
        - ``LoginBlocked(reason)`` when the password was correct but status is
          'pending' or 'rejected' — the caller surfaces a discriminating message.
        - ``None`` on credential failure (missing email, wrong password, unset
          hash). All credential failures collapse to one "Invalid email or
          password" message to avoid leaking which emails are registered.
    """
    user = (await session.execute(select(User).where(User.email == email))).scalar_one_or_none()
    if user is None:
        # Burn ~100 ms hashing a fake password so login latency doesn't leak
        # account existence via a timing side-channel. Cheap insurance.
        _dummy_verify()
        return None
    if not verify_password(plaintext_password, user.password_hash):
        return None
    if user.status != "approved":
        return LoginBlocked(reason=user.status)
    return user


def _safe_ip(value: str | None) -> str | None:
    """Return ``value`` if it parses as a valid IPv4/IPv6 address, else None.

    Postgres' INET column rejects non-IP strings (e.g. FastAPI's TestClient
    sends the literal host ``testclient``; some reverse proxies in dev send
    Unix-socket paths). Silently drop those rather than crashing on insert.
    """
    if not value:
        return None
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        return None


async def create_user_session(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    user_agent: str | None,
    ip: str | None,
) -> UserSession:
    us = UserSession(user_id=user_id, user_agent=user_agent, ip=_safe_ip(ip))
    session.add(us)
    await session.flush()
    return us


async def revoke_user_session(session: AsyncSession, session_id: uuid.UUID) -> None:
    us = (
        await session.execute(select(UserSession).where(UserSession.id == session_id))
    ).scalar_one_or_none()
    if us is not None and us.revoked_at is None:
        us.revoked_at = dt.datetime.now(tz=dt.UTC)


async def set_password(session: AsyncSession, user: User, plaintext: str) -> None:
    """Hash + persist a new password for ``user``. Caller is responsible for
    committing (typically `session_scope` handles that on exit).
    """
    user.password_hash = hash_password(plaintext)
    session.add(user)


# --- password-reset tokens --------------------------------------------------
#
# The token the user receives in their email is a high-entropy
# ``secrets.token_urlsafe(32)``. We persist only its SHA-256 hash (hex) — same
# rationale as API keys in ``keys.py``: 256 bits of entropy makes SHA-256 safe,
# and bcrypt would only burn CPU. Lookup hashes the presented token and matches
# the stored hash.


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_password_reset(session: AsyncSession, user: User, *, expiry_minutes: int) -> str:
    """Mint a single-use reset token for ``user`` and return the plaintext.

    Invalidates any of the user's still-outstanding tokens first (marks them
    used) so only the newest link works — limits the blast radius if an earlier
    email was intercepted. Returns the plaintext token; only its hash is stored.
    """
    now = dt.datetime.now(tz=dt.UTC)
    # Burn any prior outstanding tokens for this user (single live link).
    await session.execute(
        update(PasswordReset)
        .where(PasswordReset.user_id == user.id, PasswordReset.used_at.is_(None))
        .values(used_at=now)
    )
    token = secrets.token_urlsafe(32)
    session.add(
        PasswordReset(
            user_id=user.id,
            token_hash=hash_reset_token(token),
            expires_at=now + dt.timedelta(minutes=expiry_minutes),
        )
    )
    return token


async def consume_password_reset(session: AsyncSession, token: str) -> User | None:
    """Validate + atomically consume a reset token, returning its ``User``.

    Returns None for unknown / expired / already-used tokens. Consumption is a
    single guarded UPDATE (``WHERE id=? AND used_at IS NULL``) so two concurrent
    submits can't both succeed — the DB picks exactly one winner. Caller must
    then set the new password and commit in the same transaction.
    """
    token_hash = hash_reset_token(token)
    pr = (
        await session.execute(select(PasswordReset).where(PasswordReset.token_hash == token_hash))
    ).scalar_one_or_none()
    if pr is None or pr.used_at is not None:
        return None
    now = dt.datetime.now(tz=dt.UTC)
    if pr.expires_at <= now:
        return None
    # Atomic single-use: only the row that is still unused gets stamped, and we
    # require exactly one affected row to proceed.
    result = await session.execute(
        update(PasswordReset)
        .where(PasswordReset.id == pr.id, PasswordReset.used_at.is_(None))
        .values(used_at=now)
    )
    if result.rowcount != 1:
        return None
    return (await session.execute(select(User).where(User.id == pr.user_id))).scalar_one_or_none()


async def peek_password_reset(session: AsyncSession, token: str) -> User | None:
    """Non-consuming lookup: return the ``User`` for a still-valid token, else
    ``None`` (unknown / used / expired). Used by ``GET /reset-password/{token}``
    to decide whether to show the form and to tell an initial set-password
    (user has no password yet) apart from a true reset.
    """
    token_hash = hash_reset_token(token)
    pr = (
        await session.execute(select(PasswordReset).where(PasswordReset.token_hash == token_hash))
    ).scalar_one_or_none()
    if pr is None or pr.used_at is not None:
        return None
    if pr.expires_at <= dt.datetime.now(tz=dt.UTC):
        return None
    return (await session.execute(select(User).where(User.id == pr.user_id))).scalar_one_or_none()


async def revoke_all_user_sessions(session: AsyncSession, user_id: uuid.UUID) -> None:
    """Revoke every active web session for a user (used after a password reset
    so a leaked cookie can't outlive the reset). Guarded UPDATE over the user's
    not-yet-revoked sessions.
    """
    await session.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=dt.datetime.now(tz=dt.UTC))
    )


# --- FastAPI dependency -----------------------------------------------------


async def current_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> User | None:
    """Return the User row for the requester's cookie session, or None.

    Resolves to None for: missing cookie, missing/invalid signature, missing
    DB row, revoked session, expired session. Routes that require login
    should check ``user is None`` and 303 to ``/login``.

    Shares the route's DB session via ``Depends(get_session)`` — the returned
    User stays attached, so lazy-loaded relationships (e.g. ``user.keys`` in
    the dashboard) work without a re-fetch.
    """
    settings = request.app.state.settings
    if not settings.session_secret:
        return None
    cookie = request.cookies.get(COOKIE_NAME)
    if not cookie:
        return None
    sid = verify_session_cookie(cookie, settings.session_secret)
    if sid is None:
        return None

    row = (
        await session.execute(
            select(UserSession, User)
            .join(User, UserSession.user_id == User.id)
            .where(UserSession.id == sid)
        )
    ).one_or_none()
    if row is None:
        return None
    us, user = row
    if us.revoked_at is not None:
        return None
    max_age = dt.timedelta(days=settings.session_max_age_days)
    if dt.datetime.now(tz=dt.UTC) - us.created_at > max_age:
        return None
    # Status is re-read on EVERY request, not just at login (ACS-353). Without
    # this a suspended/rejected user keeps a valid cookie for up to
    # ``session_max_age_days`` (30) — enough to browse the workbench and mint
    # fresh keys via /me/keys. Rejection only looked safe because
    # ``admin_reject_user`` explicitly revokes sessions; that stays as
    # belt-and-braces, but the gate belongs here so no future status-changing
    # path can forget it. The User row is already joined, so this costs nothing.
    if user.status != "approved":
        return None
    return user


async def require_admin(
    user: User | None = Depends(current_user),
) -> User:
    """Use as a dependency on /admin* routes — raises 403 if not admin.

    Web-side admin attribution. CLI keeps using ``X-Admin-Token`` for break
    glass + automation; this is only for the web routes that need to know
    *which* admin took an action.
    """
    from fastapi import HTTPException, status

    if user is None:
        raise HTTPException(status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
    if user.role != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail={"error": {"message": "Admin role required.", "code": "admin_required"}},
        )
    return user


CurrentUserDep = Depends(current_user)


async def require_same_origin(request: Request) -> None:
    """Reject a cross-site state-changing POST (partial ACS-102).

    This repo has no CSRF token anywhere; the only thing standing between an
    admin's cookie and a forged POST is ``samesite="lax"``, which browsers
    honour but which is not a guarantee we control. Bulk actions change the
    arithmetic — one forged request used to cost one victim, and now it could
    cost the whole roster — so the destructive admin routes get a same-origin
    check now rather than waiting for full token CSRF (ACS-102).

    Checks ``Origin`` first (present on every cross-site POST in every browser
    that matters), falling back to ``Referer`` for same-origin requests where
    Origin may be omitted. A request carrying neither is allowed through: curl
    and the test client send neither, and blocking them would break the CLI
    smoke paths without stopping any browser-driven attack — the header is
    attacker-uncontrollable precisely *because* browsers set it.
    """
    from fastapi import HTTPException, status

    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    candidate = origin or referer
    if not candidate:
        return

    from urllib.parse import urlsplit

    settings = request.app.state.settings

    def _host_of(value: str) -> str:
        """Lowercased host[:port] of an absolute URL, or "" if it isn't one."""
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc:
            return ""
        return parts.netloc.lower()

    # Compare HOSTS, not scheme://host. Behind Railway's TLS-terminating proxy
    # ``request.url.scheme`` is only https when uvicorn trusts the forwarding
    # proxy, and FORWARDED_ALLOW_IPS is not configured in this repo — so the
    # scheme can read as http while the browser sends an https Origin. Matching
    # on scheme would then 403 a legitimate admin on the *secondary* hostname,
    # which is exactly the case the self-host entry below exists to allow.
    # Scheme downgrade isn't the threat here anyway: this guard exists to stop a
    # POST from a *different site*, and the host alone answers that.
    allowed = {_host_of(settings.public_base_url)}
    # The request's own host is always legitimate — the app is reachable on more
    # than one hostname and public_base_url names only one of them.
    allowed.add(request.url.netloc.lower())
    allowed.discard("")

    if _host_of(candidate) in allowed:
        return
    log.warning(
        "cross_origin_post_rejected",
        path=request.url.path,
        origin=origin,
        referer=referer,
    )
    raise HTTPException(
        status.HTTP_403_FORBIDDEN,
        detail={
            "error": {
                "message": "Cross-origin form submission rejected.",
                "code": "cross_origin_rejected",
            }
        },
    )


SameOriginDep = Depends(require_same_origin)

AdminRequiredDep = Depends(require_admin)
