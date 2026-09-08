"""DB-gated tests for the cookie-auth /admin/* web UI.

Header-token /admin endpoints (used by acs-keys CLI) keep their existing
behaviour and are not exercised here.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import (
    ApiKey,
    ApiRequest,
    EmailLog,
    Feedback,
    PasswordReset,
    SignupInvite,
    UsageDaily,
    UsageMonthly,
    User,
    UserTag,
)
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run admin UI tests",
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-admin-ui-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    # Email disabled — mailer should no-op so approve/reject succeed without network.
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"admin-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(
    *, role: str = "user", status: str = "approved", password: str = "test-pw-12345"
) -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password), role=role, status=status)
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


# ---- admin gating -----------------------------------------------------------


@dbtest
async def test_admin_routes_block_non_admin(client):
    _, email = await _make_user(role="user")
    _login(client, email, "test-pw-12345")
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 403


@dbtest
def test_admin_routes_require_login(client):
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ---- approve flow -----------------------------------------------------------


@dbtest
async def test_approve_creates_key_and_pending_columns(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    assert r.status_code == 303, r.text[:500]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "approved"
            assert u.approved_at is not None
            assert u.monthly_token_budget_total == 500000
            assert u.pending_key_plaintext is not None
            assert u.pending_key_id is not None
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert len(keys) == 1
            assert keys[0].monthly_token_budget == 500000
    finally:
        await engine.dispose()


# ---- customer360: engagement, email history, feedback (ACS-300 §2–3) --------


async def _seed_activity(
    user_id: uuid.UUID, *, days_ago: int = 2, n_prompt: int = 1000, n_completion: int = 500
) -> None:
    import datetime as dt

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            key = ApiKey(
                user_id=user_id,
                key_hash=os.urandom(32),
                key_prefix="acs-bm-test",
                name="seed",
                monthly_token_budget=0,
            )
            s.add(key)
            await s.flush()
            s.add(
                ApiRequest(
                    key_id=key.id,
                    ts=dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=days_ago),
                    endpoint="/v1/completions",
                    model="gpt2",
                    status=200,
                    n_prompt=n_prompt,
                    n_completion=n_completion,
                )
            )
    finally:
        await engine.dispose()


@dbtest
async def test_roster_engagement_columns(client):
    _, admin_email = await _make_user(role="admin")
    active_id, _ = await _make_user()
    await _make_user()  # second user with no activity → never_activated
    await _seed_activity(active_id, days_ago=2)
    # An internal account is still identifiable on the roster — but via the
    # `internal` TAG now, not an @acsresearch.org guess (ACS-374). Tag it
    # explicitly: asserting on a bare ">internal<" while merely *creating* an
    # @acsresearch.org user used to pass only because leftover users from
    # earlier runs had been tagged by migration 0045 and their chips rendered
    # the same string — it failed on any fresh database.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            team = User(
                email=f"team-{uuid.uuid4().hex[:8]}@acsresearch.org",
                password_hash=hash_password("test-pw-12345"),
            )
            s.add(team)
            await s.flush()
            s.add(UserTag(user_id=team.id, tag="internal"))
    finally:
        await engine.dispose()
    _login(client, admin_email, "test-pw-12345")

    r = client.get("/admin/users")
    assert r.status_code == 200
    assert "Engagement" in r.text and "Tokens (30d)" in r.text
    assert "eng-active" in r.text
    assert "never activated" in r.text
    assert "1,500" in r.text  # 1000 + 500 tokens within 30d
    # The chip for the tagged account. Matched on the opening tag + the label
    # rather than the exact element, so restoring/removing the explanatory
    # title attribute doesn't silently break this — while still being specific
    # enough that the tag-filter <option> ("internal (N)") can't satisfy it.
    assert re.search(r'<span class="tag-chip"[^>]*>internal</span>', r.text)
    assert "Aggregate cap" not in r.text
    for sort in ("bucket", "last_seen", "tokens", "weeks"):
        assert client.get(f"/admin/users?sort={sort}").status_code == 200


@dbtest
async def test_invite_rerender_keeps_engagement_columns(client):
    """The invite-POST re-render of admin_users.html must carry engagement rows
    too (review finding on #295 — it kept the old this-month rows)."""
    _, admin_email = await _make_user(role="admin")
    active_id, _ = await _make_user()
    await _seed_activity(active_id, days_ago=2)
    _login(client, admin_email, "test-pw-12345")

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "1"},
        follow_redirects=False,
    )
    assert r.status_code == 200, r.text[:300]
    assert "pill eng-active" in r.text
    assert "1,500" in r.text


@dbtest
async def test_invite_table_collapses_inactive(client):
    """ACS-303: only actionable invites render in the main table; expired /
    used / revoked ones sit behind a disclosure."""
    import datetime as dt

    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")
    for _ in range(2):
        r = client.post(
            "/admin/users/invite",
            data={"invite_type": "link", "max_uses": "1"},
            follow_redirects=False,
        )
        assert r.status_code == 200

    # Expire the newest one directly. (The shared test DB accumulates invites
    # across tests/runs, so assert structure rather than exact counts.)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(
                    select(SignupInvite).order_by(SignupInvite.created_at.desc()).limit(1)
                )
            ).scalar_one()
            inv.expires_at = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=1)
    finally:
        await engine.dispose()

    r = client.get("/admin/users")
    assert r.status_code == 200
    assert " active)" in r.text  # heading counts active invites only
    assert "inactive (expired / used / revoked)" in r.text  # disclosure present


@dbtest
async def test_detail_engagement_email_history_feedback(client):
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user()
    await _seed_activity(target_id, days_ago=2)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            # user_id NULL + differently-cased address → matched by email.
            s.add(
                EmailLog(
                    user_id=None,
                    kind="bulk",
                    to_email=target_email.upper(),
                    from_email="infra@acsresearch.org",
                    subject="Beta re-engagement nudge",
                    body_html="<p>hi</p>",
                    body_text="hi",
                    send_status="sent",
                )
            )
            s.add(
                Feedback(
                    user_id=target_id,
                    category="bug",
                    description="Logprobs panel is broken on Firefox",
                )
            )
    finally:
        await engine.dispose()
    _login(client, admin_email, "test-pw-12345")

    r = client.get(f"/admin/users/{target_id}")
    assert r.status_code == 200
    assert "eng-active" in r.text
    assert "1,500" in r.text  # tokens 30d and all-time
    assert "Beta re-engagement nudge" in r.text
    assert "Logprobs panel is broken on Firefox" in r.text
    assert "not connected" in r.text  # Discord row pre-ACS-269


@dbtest
async def test_admin_shows_discord_linked_but_not_in_server(client):
    """ACS-307: `is false` must distinguish 'linked but the bot couldn't add
    them' (⚠ / "not in server") from both 'joined' (✓) and 'never linked' (—),
    on the roster and the person page."""
    _, admin_email = await _make_user(role="admin")
    joined_id, _ = await _make_user()
    stranded_id, _ = await _make_user()
    unlinked_id, _ = await _make_user()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            for uid, username, joined in (
                (joined_id, "in-server", True),
                (stranded_id, "stranded", False),
            ):
                u = (await s.execute(select(User).where(User.id == uid))).scalar_one()
                u.discord_user_id = uuid.uuid4().hex
                u.discord_username = username
                u.discord_guild_joined = joined
    finally:
        await engine.dispose()
    _login(client, admin_email, "test-pw-12345")

    r = client.get("/admin/users")
    assert r.status_code == 200
    assert 'title="stranded — linked, but not added to the server">⚠' in r.text
    assert 'title="in-server">✓' in r.text

    r = client.get(f"/admin/users/{stranded_id}")
    assert "not in server" in r.text
    r = client.get(f"/admin/users/{joined_id}")
    assert "not in server" not in r.text
    r = client.get(f"/admin/users/{unlinked_id}")
    assert "not connected" in r.text
    assert "not in server" not in r.text


# ---- notes + approval reason (ACS-300) --------------------------------------


async def _set_notes(user_id: uuid.UUID, notes: str | None) -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            u.notes = notes
    finally:
        await engine.dispose()


async def _get_notes(user_id: uuid.UUID) -> str | None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            return u.notes
    finally:
        await engine.dispose()


@dbtest
async def test_approve_with_reason_prepends_dated_note(client):
    import datetime as dt

    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    await _set_notes(target_id, "2026-01-01: earlier note")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(
        f"/admin/users/{target_id}/approve",
        data={"approval_reason": "  SAE steering researcher, met at HAAISS  "},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text[:500]

    notes = await _get_notes(target_id)
    today = dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%d")
    assert notes is not None
    assert notes.startswith(f"{today} (approved): SAE steering researcher, met at HAAISS")
    assert notes.endswith("2026-01-01: earlier note")


@dbtest
async def test_approve_without_reason_leaves_notes_untouched(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    assert r.status_code == 303, r.text[:500]
    assert await _get_notes(target_id) is None


@dbtest
async def test_notes_save_show_and_clear(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email, "test-pw-12345")

    r = client.post(
        f"/admin/users/{target_id}/notes",
        data={"notes": "2026-07-31: pinged on Discord re: logprobs docs\r\n\r\nsecond line"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text[:500]
    assert (
        await _get_notes(target_id)
        == "2026-07-31: pinged on Discord re: logprobs docs\n\nsecond line"
    )

    r = client.get(f"/admin/users/{target_id}")
    assert r.status_code == 200
    assert f'action="/admin/users/{target_id}/notes"' in r.text
    assert "pinged on Discord re: logprobs docs" in r.text

    r = client.post(
        f"/admin/users/{target_id}/notes", data={"notes": "   "}, follow_redirects=False
    )
    assert r.status_code == 303
    assert await _get_notes(target_id) is None


@dbtest
async def test_notes_reject_over_length_cap(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    await _set_notes(target_id, "keep me")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(
        f"/admin/users/{target_id}/notes",
        data={"notes": "x" * 20_001},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert await _get_notes(target_id) == "keep me"


@dbtest
async def test_admin_pages_render_split_signup_answers(client):
    """ACS-302: the split-question answers show on the pending list details and
    the user detail page; URLs in prior work are linkified, escaped, _blank."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            u.signup_use_case = "Steering experiments."
            u.signup_outcome = "An atlas of directions."
            u.signup_prior_work = "See https://example.com/paper <b>bold</b>"
            u.signup_referral = "Twitter, via @someone"
    finally:
        await engine.dispose()
    _login(client, admin_email, "test-pw-12345")

    for url in ("/admin/users", f"/admin/users/{target_id}"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert "Planned usage" in r.text, url
        assert "An atlas of directions." in r.text, url
        assert "Twitter, via @someone" in r.text, url
        # urlize: real anchor, opens in new tab, and the raw HTML is escaped.
        assert '<a href="https://example.com/paper"' in r.text, url
        assert 'rel="noopener" target="_blank"' in r.text, url
        assert "<b>bold</b>" not in r.text, url


def _signup_block(html: str, email: str) -> str:
    """The <details> block for one applicant on /admin/users.

    Page-wide substring assertions are useless here: several users render on
    the same page, so another user's row can satisfy them (this scoping is
    what catches an ungated fallback leaking onto an invitee).
    """
    idx = html.index(email)
    start = html.index('<details class="signup-details"', idx)
    return html[start : html.index("</details>", start)]


@dbtest
async def test_unanswered_optional_signup_fields_are_shown_as_not_provided(client):
    """A skipped optional question is signal for the reviewer, so it must render
    as "not provided" rather than vanish — but only for public-signup accounts;
    invitees were never asked (real case reviewing an application 2026-07-31)."""
    _, admin_email = await _make_user(role="admin")
    applicant_id, applicant_email = await _make_user(status="pending")
    invitee_id, invitee_email = await _make_user(status="pending")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == applicant_id))).scalar_one()
            # Answered the required question + profile link; skipped the rest.
            u.signup_use_case = "Evals, data pipelines"
            u.signup_outcome = "Make the tracker better."
            u.signup_profile_link = "https://example.dev"
    finally:
        await engine.dispose()
    _login(client, admin_email, "test-pw-12345")

    roster = client.get("/admin/users")
    assert roster.status_code == 200

    # The applicant's own block pairs each skipped label with the fallback.
    block = _signup_block(roster.text, applicant_email)
    assert "Make the tracker better." in block
    assert re.search(r"Prior work.*?not provided", block, re.S)
    assert re.search(r"hear about us.*?not provided", block, re.S)
    assert "https://example.dev" in block

    # The invitee was never asked — no labels, no fallback, neutral dash only.
    invitee_block = _signup_block(roster.text, invitee_email)
    assert "Prior work" not in invitee_block
    assert "not provided" not in invitee_block
    assert "invited / admin-created" in invitee_block

    # Same rules on the person page.
    detail = client.get(f"/admin/users/{applicant_id}").text
    assert re.search(r"Prior work.*?not provided", detail, re.S)
    assert "Make the tracker better." in detail

    invitee_detail = client.get(f"/admin/users/{invitee_id}").text
    assert "Prior work" not in invitee_detail
    assert "not provided" not in invitee_detail


@dbtest
async def test_pending_detail_page_offers_reason_input(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")

    r = client.get(f"/admin/users/{target_id}")
    assert r.status_code == 200
    assert 'name="approval_reason"' in r.text

    r = client.get("/admin/users")
    assert r.status_code == 200
    assert 'name="approval_reason"' in r.text


@dbtest
async def test_approve_passwordless_user_sends_redacted_set_password_link(client):
    """A user approved without ever setting a password (admin-created, never
    signed up) gets a single-use set-password link in the approval email, and
    that link is redacted from the persisted EmailLog (ACS-99)."""
    _, admin_email = await _make_user(role="admin")
    # Password-less pending account.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=None, status="pending")
            s.add(u)
            await s.flush()
            target_id = u.id
    finally:
        await engine.dispose()

    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    assert r.status_code == 303, r.text[:500]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            # A single-use, unexpired set-password token was minted.
            prs = (
                (await s.execute(select(PasswordReset).where(PasswordReset.user_id == target_id)))
                .scalars()
                .all()
            )
            assert len(prs) == 1
            assert prs[0].used_at is None
            # The approval email is recorded with the link REDACTED.
            elog = (
                (
                    await s.execute(
                        select(EmailLog).where(
                            EmailLog.user_id == target_id, EmailLog.kind == "approval"
                        )
                    )
                )
                .scalars()
                .one()
            )
            assert "/reset-password/" not in elog.body_html
            assert "/reset-password/" not in elog.body_text
            assert "redacted" in elog.body_html.lower()
            assert "set your password" in elog.body_html.lower()
    finally:
        await engine.dispose()


@dbtest
async def test_approve_user_with_password_sends_no_set_password_link(client):
    """A user who already has a password (e.g. signed up) is approved WITHOUT a
    set-password token — they can already sign in (ACS-99)."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")  # has a password
    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            prs = (
                (await s.execute(select(PasswordReset).where(PasswordReset.user_id == target_id)))
                .scalars()
                .all()
            )
            assert prs == []
            elog = (
                (
                    await s.execute(
                        select(EmailLog).where(
                            EmailLog.user_id == target_id, EmailLog.kind == "approval"
                        )
                    )
                )
                .scalars()
                .one()
            )
            assert "set your password" not in elog.body_html.lower()
    finally:
        await engine.dispose()


@dbtest
async def test_dashboard_reveals_pending_key_once(client):
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")
    client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    client.cookies.clear()

    # Log in as the just-approved user and hit /dashboard.
    _login(client, target_email, "test-pw-12345")
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 200
    assert "acs-bm-" in r.text  # plaintext key shown
    assert "Welcome — your account is ready" in r.text
    # Brand-new user: start CTAs, not a "back to dashboard" they've never seen.
    assert "Open the workbench" in r.text
    assert 'href="/dashboard">Back to dashboard' not in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.pending_key_plaintext is None
            assert u.pending_key_id is None
    finally:
        await engine.dispose()

    # Second dashboard load: normal view, key not present again.
    r2 = client.get("/dashboard", follow_redirects=False)
    assert r2.status_code == 200
    assert "acs-bm-" not in r2.text


# ---- reject flow ------------------------------------------------------------


@dbtest
async def test_reject_sets_status_and_does_not_create_key(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/reject", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "rejected"
            assert u.rejected_at is not None
            assert u.pending_key_plaintext is None
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert keys == []
    finally:
        await engine.dispose()


@dbtest
async def test_reject_revokes_existing_keys(client):
    """Rejecting a previously-approved user revokes their live keys (ACS-212) —
    belt-and-braces with the auth-layer status gate."""
    from wrapper.keys import generate as generate_key

    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            s.add(
                ApiKey(
                    user_id=target_id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    monthly_token_budget=0,
                )
            )
    finally:
        await engine.dispose()

    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/reject", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "rejected"
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert keys and all(k.revoked_at is not None for k in keys)
    finally:
        await engine.dispose()


@dbtest
async def test_reject_revokes_web_sessions(client):
    """Rejecting a user kills their live web session (#211 review finding) —
    otherwise a rejected-while-logged-in user keeps a cookie for up to 30 days
    that can browse the workbench and mint fresh keys."""
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="approved")

    # Target logs in and can reach their dashboard.
    _login(client, target_email, "test-pw-12345")
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 200
    target_cookies = dict(client.cookies)

    # Admin (separate cookie jar) rejects them.
    client.cookies.clear()
    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/reject", follow_redirects=False)
    assert r.status_code == 303

    # The target's old session no longer works.
    client.cookies.clear()
    client.cookies.update(target_cookies)
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


# ---- detail-page decision block (ACS-211) -----------------------------------


@dbtest
async def test_user_detail_shows_decision_actions_by_status(client):
    """The signup-notification email deep-links to /admin/users/{id}, so the
    approve/reject decision must be makeable there: pending → both buttons,
    rejected → approve-after-all, approved → no decision forms (ACS-211)."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    pending_id, _ = await _make_user(status="pending")
    r = client.get(f"/admin/users/{pending_id}")
    assert f'action="/admin/users/{pending_id}/approve"' in r.text
    assert f'action="/admin/users/{pending_id}/reject"' in r.text

    rejected_id, _ = await _make_user(status="rejected")
    r = client.get(f"/admin/users/{rejected_id}")
    assert f'action="/admin/users/{rejected_id}/approve"' in r.text
    assert f'action="/admin/users/{rejected_id}/reject"' not in r.text

    approved_id, _ = await _make_user(status="approved")
    r = client.get(f"/admin/users/{approved_id}")
    assert f'action="/admin/users/{approved_id}/approve"' not in r.text
    assert f'action="/admin/users/{approved_id}/reject"' not in r.text

    # And the pending buttons actually work from that page: approve round-trips
    # back to the detail page with the flipped status. Assert on the rendered
    # pill markup — bare "status-approved" always matches the <style> block.
    r = client.post(f"/admin/users/{pending_id}/approve", follow_redirects=True)
    assert r.status_code == 200
    assert 'pill status-approved">approved' in r.text


# ---- aggregate-budget edit --------------------------------------------------


@dbtest
async def test_admin_can_set_user_aggregate_budget(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(
        f"/admin/users/{target_id}/budget",
        data={"monthly_token_budget_total": "1500000"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.monthly_token_budget_total == 1500000
    finally:
        await engine.dispose()

    # Blank → NULL (unlimited).
    r2 = client.post(
        f"/admin/users/{target_id}/budget",
        data={"monthly_token_budget_total": ""},
        follow_redirects=False,
    )
    assert r2.status_code == 303
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.monthly_token_budget_total is None
    finally:
        await engine.dispose()


# ---- per-key budget edit ----------------------------------------------------


@dbtest
async def test_admin_can_set_per_key_budgets(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    # Approve creates one key — borrow an existing key by creating one manually.
    from wrapper.keys import generate as generate_key

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            k = ApiKey(
                user_id=target_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                monthly_token_budget=100,
            )
            s.add(k)
            await s.flush()
            key_id = k.id
    finally:
        await engine.dispose()

    r = client.post(
        f"/admin/keys/{key_id}/budgets",
        data={
            "monthly_token_budget": "200000",
            "daily_token_budget": "10000",
            "monthly_input_token_budget": "100000",
            "monthly_output_token_budget": "100000",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.id == key_id))).scalar_one()
            assert k.monthly_token_budget == 200000
            assert k.daily_token_budget == 10000
            assert k.monthly_input_token_budget == 100000
            assert k.monthly_output_token_budget == 100000
    finally:
        await engine.dispose()


# ---- ACS-100: resend set-password link + surface failed approval emails -----


@dbtest
async def test_resend_set_password_for_passwordless_user(client):
    """Resend mints a fresh single-use link for a password-less approved user,
    shows it to the admin to copy, and the link works (ACS-100)."""
    import re

    _, admin_email = await _make_user(role="admin")
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(
                email=email,
                password_hash=None,
                status="approved",
                monthly_token_budget_total=500000,
            )
            s.add(u)
            await s.flush()
            target_id = u.id
    finally:
        await engine.dispose()

    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/resend-set-password", follow_redirects=False)
    assert r.status_code == 200, r.text[:400]
    # The single-use link is shown to the admin (out-of-band delivery).
    m = re.search(r"/reset-password/([A-Za-z0-9_-]+)", r.text)
    assert m, "set-password link not shown on the page"
    token = m.group(1)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            prs = (
                (await s.execute(select(PasswordReset).where(PasswordReset.user_id == target_id)))
                .scalars()
                .all()
            )
            assert len(prs) >= 1
            assert any(p.used_at is None for p in prs)
    finally:
        await engine.dispose()

    # The shown link actually works → first-time set-password page.
    r2 = client.get(f"/reset-password/{token}")
    assert r2.status_code == 200
    assert "Set your password" in r2.text


@dbtest
async def test_resend_set_password_rejected_for_password_having_user(client):
    """A user who already has a password can't be sent a set-password link —
    they self-serve via Forgot password (ACS-100)."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")  # has a password
    _login(client, admin_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/resend-set-password", follow_redirects=False)
    assert r.status_code == 400
    assert "already has a password" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            prs = (
                (await s.execute(select(PasswordReset).where(PasswordReset.user_id == target_id)))
                .scalars()
                .all()
            )
            assert prs == []
    finally:
        await engine.dispose()


@dbtest
async def test_user_detail_warns_when_approval_email_not_delivered(client):
    """After approving a password-less user with email disabled (send=skipped),
    the detail page surfaces the failure and offers the resend action (ACS-100)."""
    _, admin_email = await _make_user(role="admin")
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=None, status="pending")
            s.add(u)
            await s.flush()
            target_id = u.id
    finally:
        await engine.dispose()

    _login(client, admin_email, "test-pw-12345")
    client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    r = client.get(f"/admin/users/{target_id}")
    assert r.status_code == 200
    assert "skipped" in r.text  # email disabled in tests → skipped send
    assert "not set" in r.text  # password state pill
    assert "Resend set-password link" in r.text


# ---- ACS-59: admin can delete a user ----------------------------------------


async def _seed_user_with_history(target_id: uuid.UUID) -> tuple[uuid.UUID, str]:
    """Give a target user one key with request + usage rows so the CASCADE
    path actually gets exercised. Also add an email_log row (SET NULL FK) so
    we can verify it's preserved with NULL attribution. Returns
    (key_id, email_log_to_email) — the to_email is per-call unique so multiple
    test runs against the same DB don't trip over each other."""
    import datetime as _dt

    from wrapper.keys import generate as generate_key

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            k = ApiKey(
                user_id=target_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                monthly_token_budget=1000,
            )
            s.add(k)
            await s.flush()
            s.add(
                ApiRequest(
                    key_id=k.id,
                    endpoint="/v1/completions",
                    model="gpt2",
                    n_prompt=10,
                    n_completion=20,
                    status=200,
                )
            )
            today = _dt.date.today()
            s.add(
                UsageMonthly(
                    key_id=k.id,
                    period_start=today.replace(day=1),
                    tokens_prompt=10,
                    tokens_completion=20,
                    request_count=1,
                )
            )
            s.add(
                UsageDaily(
                    key_id=k.id,
                    period_start=today,
                    tokens_prompt=10,
                    tokens_completion=20,
                    request_count=1,
                )
            )
            log_email = f"emaillog-{uuid.uuid4().hex[:8]}@example.local"
            s.add(
                EmailLog(
                    user_id=target_id,
                    kind="approval",
                    to_email=log_email,
                    from_email="noreply@example.local",
                    subject="welcome",
                    body_html="<p>welcome</p>",
                    body_text="welcome",
                    send_status="sent",
                )
            )
            return k.id, log_email
    finally:
        await engine.dispose()


@dbtest
async def test_admin_delete_user_removes_user_and_cascades_history(client):
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="approved")
    key_id, log_email = await _seed_user_with_history(target_id)
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/delete", follow_redirects=False)
    assert r.status_code == 303, r.text[:400]
    assert r.headers["location"].startswith("/admin/users")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert (
                await s.execute(select(User).where(User.id == target_id))
            ).scalar_one_or_none() is None
            assert (
                await s.execute(select(ApiKey).where(ApiKey.id == key_id))
            ).scalar_one_or_none() is None
            assert (
                await s.execute(select(ApiRequest).where(ApiRequest.key_id == key_id))
            ).scalars().all() == []
            assert (
                await s.execute(select(UsageMonthly).where(UsageMonthly.key_id == key_id))
            ).scalars().all() == []
            assert (
                await s.execute(select(UsageDaily).where(UsageDaily.key_id == key_id))
            ).scalars().all() == []
            # email_logs is SET NULL, not CASCADE — the row survives, sans attribution.
            email_logs = (
                (await s.execute(select(EmailLog).where(EmailLog.to_email == log_email)))
                .scalars()
                .all()
            )
            assert len(email_logs) == 1
            assert email_logs[0].user_id is None
    finally:
        await engine.dispose()


@dbtest
async def test_admin_cannot_delete_themselves(client):
    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{admin_id}/delete", follow_redirects=False)
    assert r.status_code == 400
    assert "can&#39;t delete your own" in r.text or "can't delete your own" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert (
                await s.execute(select(User).where(User.id == admin_id))
            ).scalar_one() is not None
    finally:
        await engine.dispose()


@dbtest
async def test_admin_delete_admin_when_others_remain_succeeds(client):
    """Admin A deleting admin B is allowed as long as at least one admin remains
    after the delete. Belt-and-suspenders for the (race-only) last-admin guard."""
    _, admin_a_email = await _make_user(role="admin")
    admin_b_id, _ = await _make_user(role="admin")
    # A third admin so deleting B clearly isn't a last-admin attempt.
    await _make_user(role="admin")
    _login(client, admin_a_email, "test-pw-12345")

    r = client.post(f"/admin/users/{admin_b_id}/delete", follow_redirects=False)
    assert r.status_code == 303, r.text[:400]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert (
                await s.execute(select(User).where(User.id == admin_b_id))
            ).scalar_one_or_none() is None
    finally:
        await engine.dispose()


@dbtest
async def test_admin_delete_user_blocks_non_admin(client):
    _, user_email = await _make_user(role="user")
    target_id, _ = await _make_user(status="approved")
    _login(client, user_email, "test-pw-12345")
    r = client.post(f"/admin/users/{target_id}/delete", follow_redirects=False)
    assert r.status_code == 403

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert (
                await s.execute(select(User).where(User.id == target_id))
            ).scalar_one() is not None
    finally:
        await engine.dispose()


# ---- ACS-59 follow-up: inline delete on the users list ----------------------


@dbtest
async def test_admin_users_list_row_actions_contract(client):
    """The roster's destructive-action contract, restated for bulk (ACS-372).

    ACS-303 removed per-row delete from this table because it was too easy to
    fat-finger, and the original version of this test pinned that with a bare
    ``assert "/delete" not in r.text``. Bulk delete deliberately walks part of
    that back — but the old assertion would have kept passing *by accident*,
    since the bulk form posts to ``/admin/users/bulk``, and a test that passes
    for the wrong reason is worse than one that fails. So the contract is now
    asserted explicitly:

      - no per-row delete form (the ACS-303 guarantee, still held);
      - a select checkbox per row, but never for the admin's own row;
      - delete reachable only through the bulk bar, gated on a typed phrase;
      - the email cell still links to the person page.
    """
    admin_id, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    r = client.get("/admin/users")
    assert r.status_code == 200

    # No per-row delete form anywhere in the roster.
    assert f"/admin/users/{target_id}/delete" not in r.text
    assert f"/admin/users/{admin_id}/delete" not in r.text

    # Selectable rows — but not the admin's own.
    assert f'class="bulk-pick" name="user_ids" value="{target_id}"' in r.text
    assert f'value="{admin_id}"' not in r.text

    # Delete lives in the bulk bar, disabled until the phrase is typed.
    assert 'action="/admin/users/bulk"' in r.text
    assert 'name="confirm_phrase"' in r.text
    assert 'id="bulk-delete" disabled' in r.text
    # ...and it is a TWO-step action (ACS-379): the confirmation block ships
    # collapsed behind a reveal button. Both halves are asserted because the
    # two assertions above are blind in both directions once the block is
    # hidden — they pass whether the reveal broke (delete unreachable) or the
    # `hidden` attribute was dropped (delete reachable without the reveal).
    assert 'id="bulk-danger" hidden' in r.text, "the delete block must ship collapsed"
    assert 'id="bulk-delete-open"' in r.text, "the reveal button must exist"

    # The email cell still links to the person page.
    assert f'<a class="user-link" href="/admin/users/{target_id}">' in r.text

    # Danger zone still exists on the detail page for other users…
    r = client.get(f"/admin/users/{target_id}")
    assert f"/admin/users/{target_id}/delete" in r.text
    # …but never for the admin's own page (self-delete is blocked).
    r = client.get(f"/admin/users/{admin_id}")
    assert f"/admin/users/{admin_id}/delete" not in r.text


# ---- ACS-59 follow-up: admins grouped at the bottom of the list -------------


@dbtest
async def test_admin_users_list_orders_admins_at_the_bottom(client):
    """Admins always appear after every non-admin in /admin/users, regardless
    of which was created first. Newest-first ordering is preserved within each
    group.

    Crucially, the test orders create-time so an admin is created LAST. Under
    the previous ``created_at DESC`` sort that admin would render at the top
    of the list; under the role-aware sort it must drop below the user rows.
    """
    _, admin_email = await _make_user(role="admin")
    user_first_id, _ = await _make_user(role="user")
    user_second_id, _ = await _make_user(role="user")
    # Most recently created — under the OLD ordering this admin row would
    # render at the very top, so its position vs the user rows is what proves
    # the role-aware sort is wired up.
    admin_late_id, _ = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    r = client.get("/admin/users")
    assert r.status_code == 200
    body = r.text

    # Position lookup uses the inline detail-link URL — exactly one per row,
    # so the substring is unambiguous (the inline delete form URL contains
    # an extra ``/delete`` suffix, so the bare ``/admin/users/{uid}`` match
    # always hits the detail link first).
    def pos(uid: uuid.UUID) -> int:
        return body.index(f'href="/admin/users/{uid}"')

    # Every non-admin row must come before every admin row.
    assert pos(user_first_id) < pos(admin_late_id)
    assert pos(user_second_id) < pos(admin_late_id)
    # Within the user group, newest-first is preserved.
    assert pos(user_second_id) < pos(user_first_id)


# ---- ACS-353: suspend / unsuspend -------------------------------------------


@dbtest
async def test_suspend_blocks_access_without_touching_keys(client):
    """Suspension is a full access cut but a reversible one.

    The point of the feature is that unsuspending hands back exactly what was
    taken — so unlike reject (which sweeps ``revoked_at`` over every key), this
    must leave ``api_keys`` untouched and lean on the auth-layer status check.
    """
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="approved")

    # Give them a key so we can assert it survives.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            from wrapper.keys import generate as generate_key

            gk = generate_key()
            s.add(
                ApiKey(
                    user_id=target_id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    monthly_token_budget=0,
                )
            )
    finally:
        await engine.dispose()

    _login(client, admin_email, "test-pw-12345")
    r = client.post(
        f"/admin/users/{target_id}/suspend",
        data={"reason": "hiring cohort window closed"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "suspended"
            assert u.suspended_at is not None
            assert u.suspended_by_user_id is not None
            # The reason landed in notes as a dated entry.
            assert "hiring cohort window closed" in (u.notes or "")
            assert "(suspended)" in (u.notes or "")
            # Keys are deliberately left alone — this is the reversibility contract.
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert len(list(keys)) == 1
            assert all(k.revoked_at is None and k.disabled_at is None for k in keys)
    finally:
        await engine.dispose()

    # And they can no longer log in.
    client.cookies.clear()
    r = client.post(
        "/login",
        data={"email": target_email, "password": "test-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert "suspended" in r.text.lower()


@dbtest
async def test_suspend_kills_live_web_session(client):
    """A user suspended mid-session loses access immediately, not in 30 days.

    Belt-and-braces: ``current_user`` also re-reads status per request now, so
    this holds even if the session row somehow survived.
    """
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="approved")

    _login(client, target_email, "test-pw-12345")
    assert client.get("/dashboard", follow_redirects=False).status_code == 200
    target_cookies = dict(client.cookies)

    client.cookies.clear()
    _login(client, admin_email, "test-pw-12345")
    assert (
        client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False).status_code == 303
    )

    client.cookies.clear()
    client.cookies.update(target_cookies)
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


@dbtest
async def test_unsuspend_restores_without_minting_a_key_or_emailing(client):
    """Unsuspend must not route through /approve — that mints a fresh key and
    sends the approval email, neither of which is wanted when handing back
    access the user already had."""
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False)
    r = client.post(f"/admin/users/{target_id}/unsuspend", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "approved"
            assert u.suspended_at is None
            assert u.suspended_by_user_id is None
            # No key minted by the round-trip…
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert list(keys) == []
            assert u.pending_key_plaintext is None
            # …and no approval email queued.
            mails = (
                (
                    await s.execute(
                        select(EmailLog).where(
                            EmailLog.user_id == target_id, EmailLog.kind == "approval"
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert list(mails) == []
    finally:
        await engine.dispose()

    # Access is genuinely back.
    client.cookies.clear()
    _login(client, target_email, "test-pw-12345")
    assert client.get("/dashboard", follow_redirects=False).status_code == 200


@dbtest
async def test_cannot_suspend_self(client):
    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{admin_id}/suspend", follow_redirects=False)
    assert r.status_code == 400
    assert "your own account" in r.text.lower()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == admin_id))).scalar_one()
            assert u.status == "approved"
    finally:
        await engine.dispose()


@dbtest
async def test_unsuspend_rejects_non_suspended_account(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/unsuspend", follow_redirects=False)
    assert r.status_code == 400
    assert "only a suspended account" in r.text.lower()


@dbtest
async def test_approving_a_suspended_account_clears_the_suspend_stamps(client):
    """Approve is a legitimate second way back; it must not leave the detail
    page claiming the account is still suspended."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False)
    r = client.post(f"/admin/users/{target_id}/approve", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "approved"
            assert u.suspended_at is None
            assert u.suspended_by_user_id is None
    finally:
        await engine.dispose()


@dbtest
async def test_detail_page_offers_the_right_control_per_status(client):
    """An approved account offers Suspend; a suspended one offers Unsuspend.

    Before ACS-353 the decision block was ``{% if pending %}{% elif rejected %}``
    with no else, so a suspended user's page would have rendered no actions at all.
    """
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    r = client.get(f"/admin/users/{target_id}")
    assert f"/admin/users/{target_id}/suspend" in r.text
    assert f"/admin/users/{target_id}/unsuspend" not in r.text

    client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False)
    r = client.get(f"/admin/users/{target_id}")
    assert f"/admin/users/{target_id}/unsuspend" in r.text
    assert 'class="pill status-suspended"' in r.text


@dbtest
async def test_admin_own_detail_page_offers_no_suspend(client):
    """Self-suspend is refused server-side; don't dangle the button either."""
    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    r = client.get(f"/admin/users/{admin_id}")
    assert r.status_code == 200
    assert f"/admin/users/{admin_id}/suspend" not in r.text


async def _set_status_direct(user_id: uuid.UUID, status: str) -> None:
    """Flip status in the DB *without* revoking sessions — the whole point.

    Going through /suspend would also call revoke_all_user_sessions, which is
    what makes the session dead. These tests must isolate the current_user gate
    itself, or deleting it from web_auth.py would leave every test green.
    """
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            u.status = status
    finally:
        await engine.dispose()


@dbtest
@pytest.mark.parametrize("status", ["suspended", "rejected", "pending"])
async def test_current_user_regates_on_status_without_session_revocation(client, status):
    """A live cookie must stop working the moment the account stops being
    approved — even if nobody revoked the session row (ACS-353).

    This is the latent hole the PR closes: ``current_user`` used to resolve a
    session without ever re-reading ``status``, so a rejected-while-logged-in
    user kept /dashboard and /me/keys (i.e. could mint fresh keys) for up to
    ``session_max_age_days`` = 30. It only *looked* safe because
    ``admin_reject_user`` happens to revoke sessions explicitly.
    """
    target_id, target_email = await _make_user(status="approved")
    _login(client, target_email, "test-pw-12345")
    assert client.get("/dashboard", follow_redirects=False).status_code == 200

    await _set_status_direct(target_id, status)

    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303, f"{status} kept dashboard access"
    assert r.headers["location"].startswith("/login")
    # …and specifically cannot mint a new API key.
    r = client.post("/me/keys", data={"name": "sneaky"}, follow_redirects=False)
    assert r.status_code in (303, 403), r.status_code
    assert "/login" in r.headers.get("location", ""), "non-approved user reached /me/keys"


@dbtest
async def test_suspended_admin_loses_the_admin_surface(client):
    """require_admin resolves through current_user, so a suspended admin is
    bounced exactly like a logged-out one."""
    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")
    assert client.get("/admin/users", follow_redirects=False).status_code == 200

    await _set_status_direct(admin_id, "suspended")

    r = client.get("/admin/users", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


@dbtest
@pytest.mark.parametrize("status", ["pending", "rejected"])
async def test_cannot_suspend_a_non_approved_account(client, status):
    """Suspend is only a door out of 'approved'.

    Otherwise suspend→unsuspend would promote a pending applicant straight to
    approved behind admin_approve_user's back (no key, no approval stamps, no
    budget, no email), or silently un-reject a rejected row while leaving
    rejected_at set and its keys revoked.
    """
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status=status)
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False)
    assert r.status_code == 400
    assert "only an approved account" in r.text.lower()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == status, "status changed despite the guard"
            assert u.suspended_at is None
    finally:
        await engine.dispose()


@dbtest
async def test_rejecting_a_suspended_account_clears_the_suspend_stamps(client):
    """Reject supersedes suspend; the Profile panel must not read as both."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user(status="approved")
    _login(client, admin_email, "test-pw-12345")

    client.post(f"/admin/users/{target_id}/suspend", follow_redirects=False)
    assert (
        client.post(f"/admin/users/{target_id}/reject", follow_redirects=False).status_code == 303
    )

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == target_id))).scalar_one()
            assert u.status == "rejected"
            assert u.rejected_at is not None
            assert u.suspended_at is None
            assert u.suspended_by_user_id is None
    finally:
        await engine.dispose()
