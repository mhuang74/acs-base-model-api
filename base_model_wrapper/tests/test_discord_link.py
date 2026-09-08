"""DB-gated tests for the Discord OAuth2 account-linking flow (ACS-269).

The three Discord endpoints (token exchange, /users/@me, guild member PUT)
are mocked with respx; no real Discord traffic. Feature gating, state
signature/age/user checks, duplicate mapping, and the 403-guild-join path
are all exercised through the real ASGI app.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from itsdangerous import URLSafeSerializer
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import User
from wrapper.routes.discord import _STATE_SALT
from wrapper.web_auth import hash_password

try:
    import respx
    from httpx import Response
except ImportError:  # pragma: no cover
    respx = None  # type: ignore[assignment]

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run discord-link tests",
)
respx_required = pytest.mark.skipif(respx is None, reason="respx not installed")

_DISCORD_VARS = {
    "DISCORD_CLIENT_ID": "test-client-id",
    "DISCORD_CLIENT_SECRET": "test-client-secret",
    "DISCORD_BOT_TOKEN": "test-bot-token",
    "DISCORD_GUILD_ID": "123456789012345678",
    # Join invite — distinct from the server deep link built from the guild id.
    "BETA_DISCORD_URL": "https://discord.gg/testinvite",
}


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-discord-link-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.pop("HF_TOKEN", None)
    for k, v in _DISCORD_VARS.items():
        os.environ[k] = v

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    for k in _DISCORD_VARS:
        os.environ.pop(k, None)


@pytest.fixture
def disabled_client():
    """App booted WITHOUT the discord settings → feature must 404."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-discord-link-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("EMAIL_ENABLED", "false")
    for k in _DISCORD_VARS:
        os.environ.pop(k, None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


async def _make_user(password: str = "test-pw-12345") -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = f"discord-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password), status="approved")
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _get_user(user_id: uuid.UUID) -> User:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            return (await s.execute(select(User).where(User.id == user_id))).scalar_one()
    finally:
        await engine.dispose()


async def _set_discord(user_id: uuid.UUID, discord_id: str, *, joined: bool = True) -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            u.discord_user_id = discord_id
            u.discord_username = "someone-else"
            u.discord_guild_joined = joined
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str = "test-pw-12345") -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, r.text[:200]


def _state(client: TestClient, user_id: uuid.UUID, *, age_s: int = 0) -> str:
    secret = client.app.state.settings.session_secret
    return URLSafeSerializer(secret, salt=_STATE_SALT).dumps(
        {"user_id": str(user_id), "ts": int(time.time()) - age_s}
    )


def _fresh_discord_id() -> str:
    return str(uuid.uuid4().int % 10**17 + 10**17)


def _mock_discord(
    router: "respx.MockRouter", *, put_status: int = 201, discord_id: str = "111222333"
) -> None:
    router.post("https://discord.com/api/oauth2/token").mock(
        return_value=Response(200, json={"access_token": "user-access-token"})
    )
    router.get("https://discord.com/api/users/@me").mock(
        return_value=Response(
            200, json={"id": discord_id, "username": "tester", "global_name": "Tester"}
        )
    )
    router.put(
        f"https://discord.com/api/guilds/{_DISCORD_VARS['DISCORD_GUILD_ID']}/members/{discord_id}"
    ).mock(return_value=Response(put_status))


# ---- gating ------------------------------------------------------------------


@dbtest
async def test_disabled_feature_404s(disabled_client):
    _, email = await _make_user()
    _login(disabled_client, email)
    assert disabled_client.get("/discord/connect", follow_redirects=False).status_code == 404
    assert disabled_client.get("/discord/callback", follow_redirects=False).status_code == 404
    assert disabled_client.post("/discord/disconnect", follow_redirects=False).status_code == 404


@dbtest
def test_connect_requires_login(client):
    r = client.get("/discord/connect", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_connect_redirects_to_discord_authorize(client):
    uid, email = await _make_user()
    _login(client, email)
    r = client.get("/discord/connect", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("https://discord.com/oauth2/authorize?")
    assert "client_id=test-client-id" in loc
    assert "scope=identify+guilds.join" in loc
    assert "state=" in loc
    assert "%2Fdiscord%2Fcallback" in loc


# ---- callback ----------------------------------------------------------------


@dbtest
@respx_required
@pytest.mark.parametrize("put_status", [201, 204])
async def test_callback_happy_path_links_and_joins(client, put_status):
    uid, email = await _make_user()
    _login(client, email)
    did = _fresh_discord_id()
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        _mock_discord(router, put_status=put_status, discord_id=did)
        r = client.get(
            f"/discord/callback?code=authcode&state={_state(client, uid)}",
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=connected"
    u = await _get_user(uid)
    assert u.discord_user_id == did
    assert u.discord_username == "Tester"
    assert u.discord_connected_at is not None
    assert u.discord_guild_joined is True


@dbtest
async def test_callback_declined_consent(client):
    uid, email = await _make_user()
    _login(client, email)
    r = client.get("/discord/callback?error=access_denied", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=declined"
    assert (await _get_user(uid)).discord_user_id is None


@dbtest
async def test_callback_rejects_forged_expired_or_foreign_state(client):
    uid, email = await _make_user()
    other_uid, _ = await _make_user()
    _login(client, email)

    for bad_state in (
        "forged-garbage",
        _state(client, uid, age_s=601),  # expired
        _state(client, other_uid),  # signed for a different user
    ):
        r = client.get(f"/discord/callback?code=authcode&state={bad_state}", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/dashboard?discord=error"
    assert (await _get_user(uid)).discord_user_id is None


@dbtest
@respx_required
async def test_callback_duplicate_discord_account_conflicts(client):
    uid, email = await _make_user()
    other_uid, _ = await _make_user()
    did = _fresh_discord_id()
    await _set_discord(other_uid, did)
    _login(client, email)
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        _mock_discord(router, discord_id=did)
        r = client.get(
            f"/discord/callback?code=authcode&state={_state(client, uid)}",
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=conflict"
    assert (await _get_user(uid)).discord_user_id is None


@dbtest
@respx_required
async def test_callback_unique_race_reports_conflict(client, monkeypatch):
    """If the friendly pre-check misses but the DB unique constraint rejects the
    row, the user must see the conflict banner — not a "connected" lie that was
    silently rolled back after the response (review finding on #298).

    The race is reproduced faithfully: a real conflicting row exists and the
    real constraint fires; only the pre-check SELECT is blinded.
    """
    from wrapper.routes import discord as discord_routes

    uid, email = await _make_user()
    other_uid, _ = await _make_user()
    did = _fresh_discord_id()
    await _set_discord(other_uid, did)  # the row that will lose us the race
    _login(client, email)

    real_select = discord_routes.select
    monkeypatch.setattr(
        discord_routes,
        "select",
        lambda *a, **kw: real_select(*a, **kw).where(User.id == uuid.uuid4()),
    )

    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        _mock_discord(router, discord_id=did)
        r = client.get(
            f"/discord/callback?code=authcode&state={_state(client, uid)}",
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=conflict"
    assert (await _get_user(uid)).discord_user_id is None


@dbtest
@respx_required
async def test_callback_guild_join_403_still_links(client):
    uid, email = await _make_user()
    _login(client, email)
    did = _fresh_discord_id()
    with respx.mock(assert_all_mocked=False, assert_all_called=False) as router:
        _mock_discord(router, put_status=403, discord_id=did)
        r = client.get(
            f"/discord/callback?code=authcode&state={_state(client, uid)}",
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=noadd"
    u = await _get_user(uid)
    assert u.discord_user_id == did  # identity captured despite no-join
    assert u.discord_guild_joined is False  # persisted, so the invite survives a reload (ACS-307)


# ---- disconnect + dashboard rendering ------------------------------------------


@dbtest
async def test_disconnect_clears_mapping(client):
    uid, email = await _make_user()
    await _set_discord(uid, _fresh_discord_id())
    _login(client, email)
    r = client.post("/discord/disconnect", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard?discord=disconnected"
    u = await _get_user(uid)
    assert u.discord_user_id is None
    assert u.discord_username is None
    assert u.discord_connected_at is None
    assert u.discord_guild_joined is None


@dbtest
async def test_dashboard_shows_connect_button_and_connected_state(client):
    uid, email = await _make_user()
    _login(client, email)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert 'href="/discord/connect"' in r.text

    await _set_discord(uid, _fresh_discord_id())
    r = client.get("/dashboard")
    assert "someone-else" in r.text
    assert 'action="/discord/disconnect"' in r.text
    # ACS-306: a linked member gets the server deep link, not a join invite.
    guild = _DISCORD_VARS["DISCORD_GUILD_ID"]
    assert f"https://discord.com/channels/{guild}" in r.text


@dbtest
async def test_dashboard_invite_link_only_when_autojoin_failed(client):
    """ACS-306: the discord.gg invite is for *joining* — it belongs only in the
    'we couldn't add you' state, not next to the Connect button or the
    connected-and-in-the-server state."""
    uid, email = await _make_user()
    _login(client, email)

    # Not connected: Connect button, no invite escape hatch.
    r = client.get("/dashboard")
    assert 'href="/discord/connect"' in r.text
    assert "discord.gg" not in r.text

    # Linked and joined: server deep link, still no invite.
    await _set_discord(uid, _fresh_discord_id())
    r = client.get("/dashboard")
    assert "discord.gg" not in r.text

    # Linked but auto-join failed: the invite is the way in, and the server
    # deep link is suppressed — it wouldn't work for a non-member yet. The
    # persisted flag is the source of truth (ACS-307); ?discord= only adds the
    # banner copy, so set the state the callback would have written.
    await _set_discord(uid, _fresh_discord_id(), joined=False)
    r = client.get("/dashboard?discord=noadd")
    assert "couldn't add you to the server" in r.text
    assert "discord.gg" in r.text
    assert f"https://discord.com/channels/{_DISCORD_VARS['DISCORD_GUILD_ID']}" not in r.text


@dbtest
async def test_invite_fallback_survives_reload_when_not_joined(client):
    """ACS-307: the invite must still be offered on a later visit, driven by
    the persisted flag rather than the one-render ?discord= status."""
    uid, email = await _make_user()
    await _set_discord(uid, _fresh_discord_id(), joined=False)
    _login(client, email)

    r = client.get("/dashboard")  # plain reload, no query string
    assert r.status_code == 200
    assert "discord.gg" in r.text
    assert "couldn't add you to the server" in r.text
    guild = _DISCORD_VARS["DISCORD_GUILD_ID"]
    assert f"https://discord.com/channels/{guild}" not in r.text


@dbtest
async def test_dashboard_nudges_unconnected_users(client):
    """ACS-310: approved-but-unlinked users get a banner pointing at the card;
    it disappears once linked, and stays quiet right after a callback."""
    uid, email = await _make_user()
    _login(client, email)

    r = client.get("/dashboard")
    assert "Join the community" in r.text
    assert 'href="#community"' in r.text

    # Not while the card is showing its own callback status message.
    r = client.get("/dashboard?discord=declined")
    assert "Join the community." not in r.text

    await _set_discord(uid, _fresh_discord_id())
    r = client.get("/dashboard")
    assert "Join the community." not in r.text
