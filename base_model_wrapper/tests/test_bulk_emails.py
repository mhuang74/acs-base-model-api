"""Tests for the admin bulk-email feature (ACS-228).

Two layers, mirroring ``test_admin_emails.py``:

- Pure-function tests (always run): CSV parsing/validation, the unsubscribe
  token roundtrip, footer injection, and the html/text body derivations.
- DB-gated flow tests (skip unless TEST_DATABASE_URL points at a migrated
  Postgres): upload → draft batch → edit → send (email disabled → items
  'skipped' but EmailLog rows written), opt-out suppression, the public
  /unsubscribe and /subscribe endpoints, and admin gating.
"""

from __future__ import annotations

import io
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import (
    BulkEmailBatch,
    BulkEmailItem,
    EmailLog,
    EmailOptOut,
    UpdateSubscriber,
    User,
)
from wrapper.routes.admin.bulk_emails import (
    _html_from_text,
    _text_from_html,
    _with_unsubscribe_footer,
    admin_bulk_emails_upload,
    email_from_optout_token,
    optout_token,
    parse_bulk_csv,
)
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run bulk-email tests",
)

_FROM_DOMAIN = "acsresearch.org"
_DEFAULT_FROM = "ACS Infra <infra@acsresearch.org>"


# ---- pure: CSV parsing -------------------------------------------------------


def _parse(text: str):
    return parse_bulk_csv(text, default_from=_DEFAULT_FROM, allowed_domain=_FROM_DOMAIN)


def test_parse_minimal_csv_defaults_from():
    items, errors = _parse("to,subject,text_body\na@example.org,Hi there,Hello\n")
    assert errors == []
    assert len(items) == 1
    assert items[0]["from_email"] == _DEFAULT_FROM
    assert items[0]["to_email"] == "a@example.org"
    assert "<p>Hello</p>" in items[0]["body_html"]


def test_parse_full_columns_and_aliases():
    # Spreadsheet-style headers ("Reply To", "HTML Body") must match too.
    csv_text = (
        "From,To,CC,BCC,Reply To,Subject,HTML Body\n"
        'ivar@acsresearch.org,a@example.org,"x@example.org; y@example.org",'
        "infra@acsresearch.org,ivar@acsresearch.org,Hey,<p>Hi <b>there</b></p>\n"
    )
    items, errors = _parse(csv_text)
    assert errors == []
    item = items[0]
    assert item["from_email"] == "ivar@acsresearch.org"
    assert item["cc"] == "x@example.org, y@example.org"
    assert item["bcc"] == "infra@acsresearch.org"
    assert item["reply_to"] == "ivar@acsresearch.org"
    assert item["body_text"] == "Hi there"  # derived from html


def test_parse_rejects_foreign_or_malformed_from():
    items, errors = _parse("from,to,subject,text_body\nevil@gmail.com,a@example.org,Hi,Yo\n")
    assert items == []
    assert any("must be one address @acsresearch.org" in e for e in errors)
    # Multi-address 'from' would smuggle a second sender to Resend verbatim.
    items, errors = _parse(
        'from,to,subject,text_body\n"Ivar <ivar@acsresearch.org>, evil@e.com",a@example.org,Hi,Yo\n'
    )
    assert items == []
    # Empty local part.
    items, errors = _parse("from,to,subject,text_body\n@acsresearch.org,a@example.org,Hi,Yo\n")
    assert items == []


def test_parse_rejects_missing_required_columns():
    _, errors = _parse("to,text_body\na@example.org,Hello\n")
    assert any("subject" in e for e in errors)
    _, errors = _parse("subject,text_body\nHi,Hello\n")
    assert any("'to'" in e for e in errors)
    _, errors = _parse("to,subject\na@example.org,Hi\n")
    assert any("body" in e for e in errors)


def test_parse_flags_bad_rows_keeps_good_ones():
    csv_text = (
        "to,subject,text_body\n"
        "good@example.org,Hi,Hello\n"
        "not-an-email,Hi,Hello\n"
        "good@example.org,Dup,Hello\n"
        "second@example.org,,Hello\n"
    )
    items, errors = _parse(csv_text)
    assert [i["to_email"] for i in items] == ["good@example.org"]
    assert len(errors) == 3  # bad address, duplicate, missing subject


# ---- pure: bodies, footer, token --------------------------------------------


def test_html_text_derivations():
    assert _html_from_text("Line one\n\nLine <two>") == "<p>Line one</p>\n<p>Line &lt;two&gt;</p>"
    assert _text_from_html("<p>Hello <b>world</b></p><p>Bye</p>").startswith("Hello world")


def test_footer_appended_with_and_without_url():
    html, text = _with_unsubscribe_footer(
        html="<p>Hi</p>", text="Hi", unsubscribe_url="https://x/unsubscribe/tok"
    )
    assert 'href="https://x/unsubscribe/tok"' in html
    assert "https://x/unsubscribe/tok" in text
    html, text = _with_unsubscribe_footer(html="<p>Hi</p>", text="Hi", unsubscribe_url=None)
    assert "Reply to this email" in html


@pytest.mark.asyncio
async def test_upload_commits_before_returning_redirect():
    """The review redirect must never outrun persistence of its batch id."""

    class _Result:
        def all(self):
            return []

    class _Session:
        def __init__(self):
            self.added = []
            self.commits = 0

        async def execute(self, _query):
            return _Result()

        def add(self, value):
            if isinstance(value, BulkEmailBatch) and value.id is None:
                value.id = uuid.uuid4()
            self.added.append(value)

        async def flush(self):
            return None

        async def commit(self):
            self.commits += 1

    session = _Session()
    upload = UploadFile(
        filename="batch.csv",
        file=io.BytesIO(b"to,subject,text_body\na@example.org,Hello,Hi\n"),
    )

    response = await admin_bulk_emails_upload(
        request=None,
        csv_file=upload,
        name="race regression",
        admin=SimpleNamespace(id=uuid.uuid4()),
        settings=SimpleNamespace(email_from=_DEFAULT_FROM),
        session=session,
    )

    assert response.status_code == 303
    assert "/admin/emails/bulk/" in response.headers["location"]
    assert session.commits == 1


def test_optout_token_roundtrip_and_tamper():
    secret = "test-secret"
    token = optout_token("Person@Example.ORG", secret)
    assert email_from_optout_token(token, secret) == "person@example.org"
    assert email_from_optout_token(token + "x", secret) is None
    assert email_from_optout_token(token, "other-secret") is None


# ---- DB-gated flow tests ------------------------------------------------------


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-bulk-email-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.setdefault("DEFAULT_MONTHLY_TOKEN_BUDGET_TOTAL", "500000")
    os.environ.setdefault("DEFAULT_PER_KEY_BUDGET", "500000")
    # Email disabled: sends soft-skip, which still exercises the full flow
    # (item statuses, EmailLog rows) without any network call.
    os.environ.setdefault("EMAIL_ENABLED", "false")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _ue() -> str:
    return f"bulk-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(*, role: str = "user", password: str = "test-pw-12345"):
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password), role=role, status="approved")
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str = "test-pw-12345") -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


def _upload(client: TestClient, csv_text: str, name: str = "test batch"):
    return client.post(
        "/admin/emails/bulk/upload",
        files={"csv_file": ("batch.csv", io.BytesIO(csv_text.encode()), "text/csv")},
        data={"name": name},
        follow_redirects=False,
    )


async def _db():
    engine = make_engine(TEST_DATABASE_URL)
    factory = make_session_factory(engine)
    return engine, factory


@dbtest
async def test_bulk_page_requires_admin(client):
    _, user_email = await _make_user(role="user")
    _login(client, user_email)
    r = client.get("/admin/emails/bulk", follow_redirects=False)
    assert r.status_code in (302, 303, 403)


@dbtest
async def test_bulk_page_not_shadowed_by_email_detail_route(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    # /admin/emails/{email_id} parses a UUID; "bulk" must route to the bulk page.
    r = client.get("/admin/emails/bulk")
    assert r.status_code == 200
    assert "Bulk email" in r.text


@dbtest
async def test_send_refused_while_email_disabled(client):
    """Email disabled (test env) → send-now refuses and the batch stays draft,
    instead of burning it as 'sent' with zero deliveries."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = _upload(client, f"to,subject,text_body\n{_ue()},Hello,Hi\n")
    batch_id = r.headers["location"].split("/admin/emails/bulk/")[1].split("?")[0]
    r = client.post(
        f"/admin/emails/bulk/{batch_id}/send", data={"send_mode": "now"}, follow_redirects=False
    )
    assert r.status_code == 303 and "err=" in r.headers["location"]

    engine, factory = await _db()
    try:
        async with session_scope(factory) as s:
            batch = (
                await s.execute(
                    select(BulkEmailBatch).where(BulkEmailBatch.id == uuid.UUID(batch_id))
                )
            ).scalar_one()
            assert batch.status == "draft"
    finally:
        await engine.dispose()


@dbtest
async def test_upload_edit_send_flow(client, monkeypatch):
    from wrapper import mailer as mailer_mod
    from wrapper.mailer import SendOutcome
    from wrapper.routes.admin import bulk_emails as bulk_mod

    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    to_addr = _ue()
    r = _upload(client, f"to,subject,text_body\n{to_addr},Hello,Hi there\n\nHow are you?\n")
    assert r.status_code == 303
    batch_id = r.headers["location"].split("/admin/emails/bulk/")[1].split("?")[0]

    # Batch page renders with the pending item.
    r = client.get(f"/admin/emails/bulk/{batch_id}")
    assert r.status_code == 200 and to_addr in r.text and "Send now" in r.text

    engine, factory = await _db()
    try:
        async with session_scope(factory) as s:
            item = (
                await s.execute(
                    select(BulkEmailItem)
                    .join(BulkEmailBatch)
                    .where(BulkEmailBatch.id == uuid.UUID(batch_id))
                )
            ).scalar_one()
            item_id, orig_html = item.id, item.body_html
        assert "<p>" in orig_html

        # Rejected edit (foreign from-domain) → err flash, nothing stored.
        r = client.post(
            f"/admin/emails/bulk/{batch_id}/items/{item_id}",
            data={
                "from_email": "someone@gmail.com",
                "subject": "X",
                "body_text": "Y",
                "body_html": "",
                "cc": "",
                "bcc": "",
                "reply_to": "",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303 and "err=" in r.headers["location"]

        # Valid edit of subject + body before sending.
        r = client.post(
            f"/admin/emails/bulk/{batch_id}/items/{item_id}",
            data={
                "from_email": "ivar@acsresearch.org",
                "subject": "Hello v2",
                "body_text": "Hi there — edited",
                "body_html": "",
                "cc": "infra@acsresearch.org",
                "bcc": "",
                "reply_to": "ivar@acsresearch.org",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303 and "err=" not in r.headers["location"]

        # Send now with the Resend call faked out (email is disabled in the
        # test env, so bypass the config guard and stub the mailer).
        sent_calls = []

        async def fake_send(**kwargs):
            sent_calls.append(kwargs)
            return SendOutcome(status="sent", http_status=200, resend_message_id="re_test")

        monkeypatch.setattr(bulk_mod, "_email_configured", lambda s: True)
        monkeypatch.setattr(mailer_mod, "send", fake_send)
        r = client.post(
            f"/admin/emails/bulk/{batch_id}/send",
            data={"send_mode": "now"},
            follow_redirects=False,
        )
        assert r.status_code == 303

        assert len(sent_calls) == 1
        assert sent_calls[0]["from_email"] == "ivar@acsresearch.org"
        assert sent_calls[0]["cc"] == ["infra@acsresearch.org"]
        # RFC 8058 one-click headers ride along.
        assert "List-Unsubscribe" in (sent_calls[0]["headers"] or {})

        async with session_scope(factory) as s:
            batch = (
                await s.execute(
                    select(BulkEmailBatch).where(BulkEmailBatch.id == uuid.UUID(batch_id))
                )
            ).scalar_one()
            assert batch.status == "sent"
            item = (
                await s.execute(select(BulkEmailItem).where(BulkEmailItem.id == item_id))
            ).scalar_one()
            assert item.status == "sent"
            assert item.subject == "Hello v2"
            assert item.from_email == "ivar@acsresearch.org"
            assert item.email_log_id is not None
            log_row = (
                await s.execute(select(EmailLog).where(EmailLog.id == item.email_log_id))
            ).scalar_one()
            assert log_row.kind == "bulk"
            assert log_row.send_status == "sent"
            assert "Unsubscribe" in log_row.body_html

        # The batch is consumed: a second send-now can't re-claim it.
        r = client.post(
            f"/admin/emails/bulk/{batch_id}/send",
            data={"send_mode": "now"},
            follow_redirects=False,
        )
        assert r.status_code == 409
    finally:
        await engine.dispose()


@dbtest
async def test_optout_suppression_at_upload_and_send(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    opted_out, fresh = _ue(), _ue()
    engine, factory = await _db()
    try:
        async with session_scope(factory) as s:
            s.add(EmailOptOut(email=opted_out, source="manual"))

        r = _upload(
            client,
            f"to,subject,text_body\n{opted_out},Hi,Hello\n{fresh},Hi,Hello\n",
        )
        batch_id = r.headers["location"].split("/admin/emails/bulk/")[1].split("?")[0]

        async with session_scope(factory) as s:
            statuses = {
                row.to_email: row.status
                for row in (
                    await s.execute(
                        select(BulkEmailItem).where(
                            BulkEmailItem.batch_id == uuid.UUID(batch_id)
                        )
                    )
                ).scalars()
            }
        assert statuses[opted_out] == "suppressed"
        assert statuses[fresh] == "pending"
    finally:
        await engine.dispose()


@dbtest
async def test_unsubscribe_link_and_subscribe_field(client):
    secret = os.environ["SESSION_SECRET"]
    addr = _ue()
    token = optout_token(addr, secret)

    engine, factory = await _db()
    try:
        # GET only renders the confirm page — link-scanner prefetch must NOT
        # opt anyone out. The POST (confirm button / RFC 8058 one-click)
        # records it; a second POST is a no-op (ON CONFLICT DO NOTHING).
        r = client.get(f"/unsubscribe/{token}")
        assert r.status_code == 200 and "Unsubscribe?" in r.text
        async with session_scope(factory) as s:
            assert (
                await s.execute(select(EmailOptOut).where(EmailOptOut.email == addr))
            ).scalar_one_or_none() is None

        for _ in range(2):
            r = client.post(f"/unsubscribe/{token}")
            assert r.status_code == 200 and "unsubscribed" in r.text.lower()
        r = client.get("/unsubscribe/not-a-real-token")
        assert r.status_code == 404
        r = client.post("/unsubscribe/not-a-real-token")
        assert r.status_code == 404

        sub_addr = _ue()
        r = client.post("/subscribe", data={"email": sub_addr}, follow_redirects=False)
        assert r.status_code == 303 and "subscribed=1" in r.headers["location"]
        # Idempotent: same address again is still a friendly redirect.
        r = client.post("/subscribe", data={"email": sub_addr}, follow_redirects=False)
        assert r.status_code == 303 and "subscribed=1" in r.headers["location"]

        async with session_scope(factory) as s:
            assert (
                await s.execute(select(EmailOptOut).where(EmailOptOut.email == addr))
            ).scalar_one_or_none() is not None
            assert (
                await s.execute(
                    select(UpdateSubscriber).where(UpdateSubscriber.email == sub_addr)
                )
            ).scalar_one_or_none() is not None
    finally:
        await engine.dispose()


@dbtest
async def test_schedule_and_cancel(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = _upload(client, f"to,subject,text_body\n{_ue()},Hi,Hello\n")
    batch_id = r.headers["location"].split("/admin/emails/bulk/")[1].split("?")[0]

    r = client.post(
        f"/admin/emails/bulk/{batch_id}/send",
        data={"send_mode": "schedule", "scheduled_at": "2030-01-01T09:00"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and "Scheduled" in r.headers["location"]

    r = client.post(f"/admin/emails/bulk/{batch_id}/cancel", follow_redirects=False)
    assert r.status_code == 303

    engine, factory = await _db()
    try:
        async with session_scope(factory) as s:
            batch = (
                await s.execute(
                    select(BulkEmailBatch).where(BulkEmailBatch.id == uuid.UUID(batch_id))
                )
            ).scalar_one()
            assert batch.status == "canceled"
            assert batch.scheduled_at is None
        # A canceled batch can't be sent.
        r = client.post(
            f"/admin/emails/bulk/{batch_id}/send",
            data={"send_mode": "now"},
            follow_redirects=False,
        )
        assert r.status_code == 409
    finally:
        await engine.dispose()
