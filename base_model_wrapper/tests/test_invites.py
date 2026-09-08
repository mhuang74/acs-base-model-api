"""DB-gated tests for the invite-token feature (signup_invites table).

Mirrors ``test_admin_emails.py``: skip unless TEST_DATABASE_URL points at a
migrated Postgres. Covers:
  - create link-only invite via POST /admin/users/invite
  - create email-targeted invite (EmailLog with kind='invite' written)
  - accept invite → approved user with role 'user' + an API key exists + token consumed
  - expired token rejected
  - already-used token rejected (single-use)
  - revoked token rejected
  - email uniqueness enforced at acceptance
  - /admin/users gates on admin + renders
  - /admin/models gates on admin + renders
"""

from __future__ import annotations

import datetime as dt
import os
import secrets
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import (
    ApiKey,
    EmailLog,
    SignupInvite,
    SignupInviteRedemption,
    User,
)
from wrapper.routes.admin.users import _parse_personalized_invite_csv
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run invite tests",
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-invites-secret-xyz")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.setdefault("PUBLIC_BASE_URL", "http://localhost:5173")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"inv-{uuid.uuid4().hex[:8]}@example.local"


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


def _login(client: TestClient, email: str, password: str = "test-pw-12345") -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


async def _insert_invite(
    *,
    email: str | None = None,
    expires_delta: dt.timedelta = dt.timedelta(days=4),
    accepted_at: dt.datetime | None = None,
    revoked_at: dt.datetime | None = None,
    max_uses: int | None = 1,
) -> tuple[uuid.UUID, str]:
    """Insert a SignupInvite directly via the DB. Returns (id, token).

    ``max_uses`` defaults to 1 (single-use); pass None for an unlimited link.
    """
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        token = secrets.token_urlsafe(32)
        async with session_scope(factory) as s:
            inv = SignupInvite(
                token=token,
                email=email,
                expires_at=dt.datetime.now(tz=dt.UTC) + expires_delta,
                accepted_at=accepted_at,
                revoked_at=revoked_at,
                max_uses=max_uses,
            )
            s.add(inv)
            await s.flush()
            return inv.id, token
    finally:
        await engine.dispose()


async def _redemption_count(invite_id: uuid.UUID) -> int:
    """Count redemption rows for an invite."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = (
                (
                    await s.execute(
                        select(SignupInviteRedemption).where(
                            SignupInviteRedemption.invite_id == invite_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            return len(rows)
    finally:
        await engine.dispose()


def _accept(client: TestClient, token: str, email: str):
    """POST the acceptance form. Returns the response (no redirect follow)."""
    return client.post(
        f"/invite/{token}",
        data={
            "email": email,
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            "agree": "yes",
        },
        follow_redirects=False,
    )


# ---- CSV personalized invite parsing ---------------------------------------


def test_parse_personalized_invite_csv_accepts_survey_markers_and_reports_bad_rows():
    rows, errors = _parse_personalized_invite_csv(
        "email,name,source\n"
        "Alice@Example.Local,Alice,survey\n"
        "bad-email,Bad,manual\n"
        "alice@example.local,Duplicate,survey\n"
        "bob@example.local,Bob,direct\n"
    )

    assert [r.email for r in rows] == ["alice@example.local", "bob@example.local"]
    assert rows[0].name == "Alice"
    assert rows[0].survey_respondent is True
    assert rows[1].survey_respondent is False
    assert "line 3: missing or invalid email" in errors
    assert "line 4: duplicate email alice@example.local" in errors


def test_parse_personalized_invite_csv_rejects_oversized_upload():
    """Over the row cap → whole upload rejected (no partial batch sent)."""
    from wrapper.routes.admin.users import _MAX_INVITE_CSV_ROWS

    body = "email\n" + "".join(
        f"user{i}@example.local\n" for i in range(_MAX_INVITE_CSV_ROWS + 5)
    )
    rows, errors = _parse_personalized_invite_csv(body)
    assert rows == []  # nothing processed → no partial email blast
    assert errors and "more than" in errors[0].lower()

    # At the cap it still works.
    ok_body = "email\n" + "".join(
        f"user{i}@example.local\n" for i in range(_MAX_INVITE_CSV_ROWS)
    )
    ok_rows, ok_errors = _parse_personalized_invite_csv(ok_body)
    assert len(ok_rows) == _MAX_INVITE_CSV_ROWS
    assert ok_errors == []


# ---- /admin/users gating + render ------------------------------------------


@dbtest
def test_admin_users_requires_login(client):
    r = client.get("/admin/users", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_admin_users_blocks_non_admin(client):
    _, email = await _make_user(role="user")
    _login(client, email)
    r = client.get("/admin/users", follow_redirects=False)
    assert r.status_code == 403


@dbtest
async def test_admin_users_renders(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    r = client.get("/admin/users")
    assert r.status_code == 200
    assert "Invite someone" in r.text


# ---- /admin/models gating + render -----------------------------------------


@dbtest
def test_admin_models_requires_login(client):
    r = client.get("/admin/models", follow_redirects=False)
    assert r.status_code == 303


@dbtest
async def test_admin_models_blocks_non_admin(client):
    _, email = await _make_user(role="user")
    _login(client, email)
    r = client.get("/admin/models", follow_redirects=False)
    assert r.status_code == 403


@dbtest
async def test_admin_models_renders(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    r = client.get("/admin/models")
    assert r.status_code == 200
    assert "Models" in r.text


# ---- create link-only invite ------------------------------------------------


@dbtest
async def test_create_link_only_invite(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link"},
        follow_redirects=False,
    )
    # Returns the users page (200) with the invite URL shown.
    assert r.status_code == 200
    assert "/invite/" in r.text

    # Verify the row was created in the DB.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            invites = (
                (await s.execute(select(SignupInvite).where(SignupInvite.email.is_(None))))
                .scalars()
                .all()
            )
            assert len(invites) >= 1
            newest = max(invites, key=lambda i: i.created_at)
            assert newest.accepted_at is None
            assert newest.revoked_at is None
    finally:
        await engine.dispose()


# ---- create email invite + EmailLog written --------------------------------


@dbtest
async def test_create_email_invite_records_email_log(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    target = _ue()
    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "email", "emails": target},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "invite sent" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            # Invite row created.
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.email == target))
            ).scalar_one()
            assert inv.accepted_at is None

            # EmailLog row with kind='invite' written.
            log_row = (
                await s.execute(
                    select(EmailLog).where(EmailLog.to_email == target, EmailLog.kind == "invite")
                )
            ).scalar_one()
            assert log_row.send_status == "skipped"  # EMAIL_ENABLED=false
            assert log_row.skip_reason == "email_disabled"
    finally:
        await engine.dispose()


# ---- email invite skips existing user --------------------------------------


@dbtest
async def test_email_invite_skips_existing_user(client):
    _, admin_email = await _make_user(role="admin")
    _, existing_email = await _make_user(role="user")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "email", "emails": existing_email},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "already has an account" in r.text


# ---- accept invite → approved user + API key + token consumed --------------


@dbtest
async def test_accept_invite_creates_approved_user(client):
    _, token = await _insert_invite()
    new_email = _ue()

    r = client.post(
        f"/invite/{token}",
        data={
            "email": new_email,
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    # Should auto-login and redirect to /dashboard.
    assert r.status_code == 303, r.text[:300]
    assert r.headers["location"] == "/dashboard"

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            # User was created and is approved.
            user = (await s.execute(select(User).where(User.email == new_email))).scalar_one()
            assert user.status == "approved"
            assert user.role == "user"
            assert user.approved_at is not None
            # ACS-209: the invite-accept form now carries the same consent
            # gate as /signup — agreement is stamped at account creation.
            assert user.agreed_terms_at is not None

            # Has an initial API key.
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == user.id))).scalars().all()
            )
            assert len(keys) == 1

            # Invite is consumed.
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.token == token))
            ).scalar_one()
            assert inv.accepted_at is not None
            assert inv.accepted_by_user_id == user.id
    finally:
        await engine.dispose()


@dbtest
async def test_accept_invite_requires_agreement(client):
    """No account is created via an invite unless the usage-rules checkbox is
    ticked (ACS-209) — same consent gate as /signup."""
    _, token = await _insert_invite()
    new_email = _ue()

    r = client.post(
        f"/invite/{token}",
        data={
            "email": new_email,
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            # no "agree"
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "agree to the usage rules" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (
                await s.execute(select(User).where(User.email == new_email))
            ).scalar_one_or_none()
            assert u is None
    finally:
        await engine.dispose()


# ---- expired token rejected ------------------------------------------------


@dbtest
async def test_expired_invite_rejected(client):
    _, token = await _insert_invite(expires_delta=dt.timedelta(seconds=-1))

    r = client.get(f"/invite/{token}")
    assert r.status_code == 410
    assert "expired" in r.text.lower()

    r2 = client.post(
        f"/invite/{token}",
        data={
            "email": _ue(),
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    assert r2.status_code == 410


# ---- already-used token rejected (single-use) --------------------------------


@dbtest
async def test_used_invite_rejected(client):
    # A single-use invite that already produced one account (redemption row).
    inv_id, token = await _insert_invite(max_uses=1)
    new_email = _ue()
    r0 = _accept(client, token, new_email)
    assert r0.status_code == 303, r0.text[:300]

    # The link is now exhausted: GET shows the friendly 410.
    r = client.get(f"/invite/{token}")
    assert r.status_code == 410
    assert "no longer available" in r.text.lower()
    assert await _redemption_count(inv_id) == 1


# ---- revoked token rejected ------------------------------------------------


@dbtest
async def test_revoked_invite_rejected(client):
    _, token = await _insert_invite(revoked_at=dt.datetime.now(tz=dt.UTC))

    r = client.get(f"/invite/{token}")
    assert r.status_code == 410
    assert "revoked" in r.text


# ---- email uniqueness enforced at acceptance --------------------------------


@dbtest
async def test_accept_invite_rejects_duplicate_email(client):
    _, existing_email = await _make_user(role="user")
    _, token = await _insert_invite()

    r = client.post(
        f"/invite/{token}",
        data={
            "email": existing_email,
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    # Should return the form with an error, not create the user.
    assert r.status_code == 400
    assert "already registered" in r.text

    # Invite NOT consumed.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.token == token))
            ).scalar_one()
            assert inv.accepted_at is None
    finally:
        await engine.dispose()


# ---- revoke endpoint -------------------------------------------------------


@dbtest
async def test_revoke_invite(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    inv_id, _ = await _insert_invite()
    r = client.post(f"/admin/invites/{inv_id}/revoke", follow_redirects=False)
    assert r.status_code == 303
    assert "/admin/users" in r.headers["location"]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.id == inv_id))
            ).scalar_one()
            assert inv.revoked_at is not None
    finally:
        await engine.dispose()


# ---- email-targeted invite: email field is still editable ------------------


@dbtest
async def test_email_targeted_invite_allows_different_email(client):
    """The invite's target email does NOT lock the signup — any email is accepted."""
    target_email = _ue()
    _, token = await _insert_invite(email=target_email)
    different_email = _ue()

    r = client.get(f"/invite/{token}")
    assert r.status_code == 200
    # suggested_email shown as hint but input is not readonly.
    assert target_email in r.text

    # Sign up with a DIFFERENT email — should succeed.
    r2 = client.post(
        f"/invite/{token}",
        data={
            "email": different_email,
            "password": "goodpassword123",
            "confirm_password": "goodpassword123",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    assert r2.status_code == 303, r2.text[:300]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.email == different_email))).scalar_one()
            assert user.status == "approved"
    finally:
        await engine.dispose()


# ---- multi-use: finite cap creates N users + N redemptions, blocks N+1 ------


@dbtest
async def test_finite_cap_allows_n_then_blocks(client):
    cap = 3
    inv_id, token = await _insert_invite(max_uses=cap)

    emails = [_ue() for _ in range(cap)]
    for em in emails:
        r = _accept(client, token, em)
        assert r.status_code == 303, r.text[:300]

    # N users created, N redemption rows recorded.
    assert await _redemption_count(inv_id) == cap
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            for em in emails:
                u = (await s.execute(select(User).where(User.email == em))).scalar_one()
                assert u.status == "approved"
    finally:
        await engine.dispose()

    # The (N+1)th accept is rejected — cap reached.
    r_over = _accept(client, token, _ue())
    assert r_over.status_code == 410
    assert "no longer available" in r_over.text.lower()
    # No extra user, no extra redemption row.
    assert await _redemption_count(inv_id) == cap


# ---- multi-use: unlimited invite accepts repeatedly -------------------------


@dbtest
async def test_unlimited_invite_accepts_multiple(client):
    inv_id, token = await _insert_invite(max_uses=None)

    emails = [_ue() for _ in range(4)]
    for em in emails:
        r = _accept(client, token, em)
        assert r.status_code == 303, r.text[:300]

    assert await _redemption_count(inv_id) == 4
    # Still claimable (no cap) — GET returns the form, not a 410.
    r_form = client.get(f"/invite/{token}")
    assert r_form.status_code == 200


# ---- admin: create link invite honours the usage-cap picker -----------------


@dbtest
async def test_create_link_invite_with_cap(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "10"},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "/invite/" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            invites = (
                (await s.execute(select(SignupInvite).where(SignupInvite.email.is_(None))))
                .scalars()
                .all()
            )
            newest = max(invites, key=lambda i: i.created_at)
            assert newest.max_uses == 10
    finally:
        await engine.dispose()


@dbtest
async def test_create_link_invite_unlimited(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "unlimited"},
        follow_redirects=False,
    )
    assert r.status_code == 200

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            invites = (
                (await s.execute(select(SignupInvite).where(SignupInvite.email.is_(None))))
                .scalars()
                .all()
            )
            newest = max(invites, key=lambda i: i.created_at)
            assert newest.max_uses is None  # unlimited
    finally:
        await engine.dispose()


@dbtest
async def test_email_invite_is_single_use(client):
    """Email-targeted invites are always single-use regardless of any picker."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    target = _ue()
    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "email", "emails": target, "max_uses": "25"},
        follow_redirects=False,
    )
    assert r.status_code == 200

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.email == target))
            ).scalar_one()
            assert inv.max_uses == 1
    finally:
        await engine.dispose()


@dbtest
async def test_csv_personalized_invite_creates_single_use_invite_and_email_log(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    target = _ue()
    csv_text = f"email,name,source\n{target},Ada Survey,survey\n"
    r = client.post(
        "/admin/users/invite",
        data={
            "invite_type": "csv_personalized",
            "discord_url": "https://discord.example/invite",
            "survey_url": "https://forms.example/survey",
        },
        files={"csv_file": ("beta.csv", csv_text, "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 200
    assert "invite sent" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.email == target))
            ).scalar_one()
            assert inv.max_uses == 1

            log_row = (
                await s.execute(
                    select(EmailLog).where(
                        EmailLog.to_email == target,
                        EmailLog.kind == "invite_personalized",
                    )
                )
            ).scalar_one()
            assert log_row.send_status == "skipped"
            assert log_row.skip_reason == "email_disabled"
            assert "Dear Ada Survey" in log_row.body_text
            assert "You recently filled in our Base Models survey" in log_row.body_text
            assert "https://discord.example/invite" in log_row.body_text
            assert "/invite/" in log_row.body_text
    finally:
        await engine.dispose()


# ---- admin table shows count + created-account emails -----------------------


@dbtest
async def test_admin_table_shows_redemptions(client):
    _, admin_email = await _make_user(role="admin")

    # Create a multi-use link and redeem it twice (before logging in as admin).
    inv_id, token = await _insert_invite(max_uses=5)
    em1, em2 = _ue(), _ue()
    assert _accept(client, token, em1).status_code == 303
    assert _accept(client, token, em2).status_code == 303

    _login(client, admin_email)
    r = client.get("/admin/users")
    assert r.status_code == 200
    # Count + cap and both created-account emails appear on the page.
    assert "2 / 5" in r.text
    assert em1 in r.text
    assert em2 in r.text


# ---- expired / revoked multi-use links still rejected -----------------------


@dbtest
async def test_expired_multiuse_rejected(client):
    _, token = await _insert_invite(max_uses=10, expires_delta=dt.timedelta(seconds=-1))
    r = client.get(f"/invite/{token}")
    assert r.status_code == 410
    assert "expired" in r.text.lower()
    r2 = _accept(client, token, _ue())
    assert r2.status_code == 410


@dbtest
async def test_revoked_multiuse_rejected(client):
    inv_id, token = await _insert_invite(max_uses=10)
    # One successful accept, then revoke.
    assert _accept(client, token, _ue()).status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.id == inv_id))
            ).scalar_one()
            inv.revoked_at = dt.datetime.now(tz=dt.UTC)
    finally:
        await engine.dispose()

    r = client.get(f"/invite/{token}")
    assert r.status_code == 410
    assert "revoked" in r.text
    r2 = _accept(client, token, _ue())
    assert r2.status_code == 410
    assert await _redemption_count(inv_id) == 1  # no new redemption
