"""DB-gated tests for the email audit log, admin Emails page, and Resend webhook.

Mirrors ``test_feedback.py``: skip unless TEST_DATABASE_URL points at a migrated
Postgres. Covers that approve/reject record an EmailLog row, the admin list +
detail pages render and gate on admin, and the /webhooks/resend endpoint
(no-secret no-op, bad-signature reject, verified event recorded + matched).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import EmailEvent, EmailLog, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run email-log tests",
)

# A valid Svix signing secret: 'whsec_' + base64. The webhook tests sign with
# this and the endpoint verifies against it.
_WEBHOOK_SECRET = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-emails-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    # Email disabled — approve/reject still record a 'skipped' EmailLog row.
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.setdefault("RESEND_WEBHOOK_SECRET", _WEBHOOK_SECRET)
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"em-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(
    *, role: str = "user", status: str = "approved", password: str = "test-pw-12345"
) -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(
                email=email, password_hash=hash_password(password), role=role, status=status
            )
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


async def _insert_email_log(**overrides) -> uuid.UUID:
    base = dict(
        kind="approval",
        to_email="recipient@example.local",
        from_email="Test <test@example.local>",
        subject="Test subject",
        body_html="<p>Hello world</p>",
        body_text="Hello world",
        send_status="sent",
        resend_message_id=f"re_{uuid.uuid4().hex}",
    )
    base.update(overrides)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = EmailLog(**base)
            s.add(row)
            await s.flush()
            return row.id
    finally:
        await engine.dispose()


# ---- approve/reject record a log row ----------------------------------------

@dbtest
async def test_approve_records_email_log(client):
    _, admin_email = await _make_user(role="admin")
    pend_id, _ = await _make_user(status="pending")
    _login(client, admin_email, "test-pw-12345")

    r = client.post(f"/admin/users/{pend_id}/approve", follow_redirects=False)
    assert r.status_code == 303, r.text[:300]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (
                await s.execute(
                    select(EmailLog).where(EmailLog.user_id == pend_id)
                )
            ).scalars().one()
            assert row.kind == "approval"
            # EMAIL_ENABLED=false → suppressed, but still logged for audit.
            assert row.send_status == "skipped"
            assert row.skip_reason == "email_disabled"
            assert "/dashboard" in row.body_html
    finally:
        await engine.dispose()


# ---- admin list + detail gating + render ------------------------------------

@dbtest
def test_admin_emails_requires_login(client):
    r = client.get("/admin/emails", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


@dbtest
async def test_admin_emails_blocks_non_admin(client):
    _, email = await _make_user(role="user")
    _login(client, email, "test-pw-12345")
    r = client.get("/admin/emails", follow_redirects=False)
    assert r.status_code == 403


@dbtest
async def test_admin_emails_list_and_detail_render(client):
    log_id = await _insert_email_log(subject="Unique subject ABC123")
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email, "test-pw-12345")

    r_list = client.get("/admin/emails")
    assert r_list.status_code == 200
    assert "Unique subject ABC123" in r_list.text

    r_detail = client.get(f"/admin/emails/{log_id}")
    assert r_detail.status_code == 200
    # Sandboxed preview + the rendered body present.
    assert "sandbox" in r_detail.text
    assert "Hello world" in r_detail.text

    r_404 = client.get(f"/admin/emails/{uuid.uuid4()}")
    assert r_404.status_code == 404


# ---- webhook ----------------------------------------------------------------

@dbtest
def test_webhook_rejects_bad_signature(client):
    r = client.post(
        "/webhooks/resend",
        content=b'{"type":"email.delivered"}',
        headers={
            "svix-id": "msg_1",
            "svix-timestamp": "1700000000",
            "svix-signature": "v1,deadbeef",
        },
    )
    assert r.status_code == 400


@dbtest
async def test_webhook_records_and_matches_event(client):
    msg_id = f"re_{uuid.uuid4().hex}"
    log_id = await _insert_email_log(resend_message_id=msg_id)

    payload = {
        "type": "email.delivered",
        "created_at": "2026-06-01T12:00:00.000Z",
        "data": {"email_id": msg_id, "to": ["recipient@example.local"]},
    }
    body = json.dumps(payload).encode()

    from svix.webhooks import Webhook

    svix_id = "msg_test_evt"
    now = dt.datetime.now(tz=dt.UTC)
    signature = Webhook(_WEBHOOK_SECRET).sign(svix_id, now, body.decode())
    headers = {
        "svix-id": svix_id,
        "svix-timestamp": str(int(now.timestamp())),
        "svix-signature": signature,
    }

    r = client.post("/webhooks/resend", content=body, headers=headers)
    assert r.status_code == 200, r.text[:300]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            ev = (
                await s.execute(
                    select(EmailEvent).where(EmailEvent.resend_message_id == msg_id)
                )
            ).scalars().one()
            assert ev.event_type == "email.delivered"
            assert ev.email_log_id == log_id
            assert ev.raw_payload["data"]["email_id"] == msg_id

            log_row = (
                await s.execute(select(EmailLog).where(EmailLog.id == log_id))
            ).scalar_one()
            assert log_row.last_event == "email.delivered"
            assert log_row.last_event_at is not None
    finally:
        await engine.dispose()
