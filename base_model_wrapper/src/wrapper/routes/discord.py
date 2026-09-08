"""Discord ↔ platform account linking via OAuth2 ``identify guilds.join`` (ACS-269).

"Connect Discord" on the dashboard → Discord consent → callback exchanges the
code, reads the exact ``discord_user_id``, adds the user to the community
guild via the bot (201 added / 204 already a member), and persists the mapping
on ``users``. OAuth tokens are used within the single callback request and
never stored or logged.

Feature-gated: every route 404s unless all four ``discord_*`` settings are
present (``settings.discord_oauth_enabled``), so deploys are safe before the
Developer-Portal setup.
"""

from __future__ import annotations

import datetime as dt
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadData, URLSafeSerializer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .. import web_auth as webauth
from ..db import get_session
from ..dependencies import get_http, get_settings
from ..logging import get_logger
from ..models import User
from ..settings import Settings

log = get_logger()
router = APIRouter()

_DISCORD_API = "https://discord.com/api"
_STATE_SALT = "acs-discord-oauth-v1"
# URLSafeSerializer doesn't expire on its own — we sign an epoch and check age
# at the callback, same pattern as the ACS-260 signup time-trap.
_STATE_MAX_AGE_S = 600


def _state_serializer(settings: Settings) -> URLSafeSerializer:
    return URLSafeSerializer(settings.session_secret or "", salt=_STATE_SALT)


def _redirect_uri(settings: Settings) -> str:
    # Must byte-match the redirect URI registered in the Developer Portal.
    return settings.public_base_url.rstrip("/") + "/discord/callback"


def _require_enabled(settings: Settings, user: User | None) -> User:
    if not settings.discord_oauth_enabled or not settings.session_secret:
        raise HTTPException(status_code=404)
    if user is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def _back(status: str) -> RedirectResponse:
    return RedirectResponse(url=f"/dashboard?discord={status}", status_code=303)


@router.get("/discord/connect")
async def discord_connect(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
):
    user = _require_enabled(settings, user)
    state = _state_serializer(settings).dumps({"user_id": str(user.id), "ts": int(time.time())})
    params = urlencode(
        {
            "client_id": settings.discord_client_id,
            "response_type": "code",
            "redirect_uri": _redirect_uri(settings),
            "scope": "identify guilds.join",
            "state": state,
        }
    )
    return RedirectResponse(url=f"https://discord.com/oauth2/authorize?{params}", status_code=303)


@router.get("/discord/callback")
async def discord_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    http: httpx.AsyncClient = Depends(get_http),
):
    user = _require_enabled(settings, user)

    if error == "access_denied":
        # User clicked "Cancel" on the consent screen — not an error condition.
        return _back("declined")

    # State: signature + age + the same logged-in user who started the flow.
    try:
        payload = _state_serializer(settings).loads(state)
        state_user_id = str(payload["user_id"])
        state_age = time.time() - float(payload["ts"])
    except (BadData, KeyError, TypeError, ValueError):
        log.warning("discord_link_bad_state", user_id=str(user.id))
        return _back("error")
    if state_age > _STATE_MAX_AGE_S or state_user_id != str(user.id):
        log.warning(
            "discord_link_state_rejected",
            user_id=str(user.id),
            expired=state_age > _STATE_MAX_AGE_S,
        )
        return _back("error")
    if not code:
        return _back("error")

    # Code → access token. Secrets and tokens are never logged.
    token_resp = await http.post(
        f"{_DISCORD_API}/oauth2/token",
        data={
            "client_id": settings.discord_client_id,
            "client_secret": settings.discord_client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _redirect_uri(settings),
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if token_resp.status_code != 200:
        log.warning("discord_link_token_exchange_failed", status=token_resp.status_code)
        return _back("error")
    access_token = token_resp.json().get("access_token")
    if not access_token:
        log.warning("discord_link_token_missing")
        return _back("error")

    me_resp = await http.get(
        f"{_DISCORD_API}/users/@me", headers={"Authorization": f"Bearer {access_token}"}
    )
    if me_resp.status_code != 200:
        log.warning("discord_link_identify_failed", status=me_resp.status_code)
        return _back("error")
    me = me_resp.json()
    discord_id = str(me["id"])
    discord_username = me.get("global_name") or me.get("username") or discord_id

    # One Discord account ↔ one platform account (unique column). Friendly
    # pre-check; the DB constraint is the real guarantee.
    other = (
        await session.execute(
            select(User).where(User.discord_user_id == discord_id, User.id != user.id)
        )
    ).scalar_one_or_none()
    if other is not None:
        log.warning(
            "discord_link_conflict", user_id=str(user.id), discord_user_id=discord_id
        )
        return _back("conflict")

    # Add to the guild (bot token). 201 = added, 204 = already a member — both
    # fine. 403 = bot lacks Create Invite (or similar): capture the identity
    # anyway and tell the user to use the invite link.
    joined = False
    put_resp = await http.put(
        f"{_DISCORD_API}/guilds/{settings.discord_guild_id}/members/{discord_id}",
        json={"access_token": access_token},
        headers={"Authorization": f"Bot {settings.discord_bot_token}"},
    )
    if put_resp.status_code in (201, 204):
        joined = True
    else:
        log.error(
            "discord_guild_join_failed",
            status=put_resp.status_code,
            user_id=str(user.id),
            discord_user_id=discord_id,
        )

    # Read the id up front: a rollback below expires the ORM object, and
    # touching an expired attribute would trigger a lazy load mid-teardown.
    user_id_str = str(user.id)
    user.discord_user_id = discord_id
    user.discord_username = discord_username
    user.discord_connected_at = dt.datetime.now(tz=dt.UTC)
    # Persisted so the invite fallback survives a reload (ACS-307) — ?discord=
    # only carries the outcome for one render.
    user.discord_guild_joined = joined
    # Flush here rather than leaving it to the session dependency: that commits
    # *after* the response is sent, so a lost unique-constraint race would tell
    # the user "connected" while the row was actually rolled back.
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        log.warning(
            "discord_link_conflict_race", user_id=user_id_str, discord_user_id=discord_id
        )
        return _back("conflict")
    log.info(
        "discord_account_linked",
        user_id=user_id_str,
        discord_user_id=discord_id,
        discord_username=discord_username,
        guild_joined=joined,
    )
    return _back("connected" if joined else "noadd")


@router.post("/discord/disconnect")
async def discord_disconnect(
    request: Request,
    user: User | None = webauth.CurrentUserDep,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    """Clear the mapping (does not remove the user from the server)."""
    user = _require_enabled(settings, user)
    user.discord_user_id = None
    user.discord_username = None
    user.discord_connected_at = None
    user.discord_guild_joined = None
    log.info("discord_account_unlinked", user_id=str(user.id))
    return _back("disconnected")
