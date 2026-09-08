"""Unit tests for mailer.send — exercises the soft-fail paths without a DB.

mailer.send must NEVER raise: dev environments with email_enabled=False just
log and return; production with a bad API key logs a warning and returns; HTTP
errors / network failures are caught and never propagated up to the approve
flow.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from wrapper import mailer


def _settings(**overrides):
    base = dict(
        email_enabled=False,
        resend_api_key=None,
        email_from="Test <test@example.local>",
        public_base_url="http://localhost:8000",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_send_skips_when_email_disabled():
    http = MagicMock()
    http.post = AsyncMock()
    await mailer.send(
        settings=_settings(email_enabled=False, resend_api_key="re_xxx"),
        http=http,
        subject="hi",
        to="user@example.local",
        html="<p>hi</p>",
        text="hi",
    )
    http.post.assert_not_called()


@pytest.mark.asyncio
async def test_send_skips_when_api_key_missing():
    http = MagicMock()
    http.post = AsyncMock()
    await mailer.send(
        settings=_settings(email_enabled=True, resend_api_key=None),
        http=http,
        subject="hi",
        to="user@example.local",
        html="<p>hi</p>",
        text="hi",
    )
    http.post.assert_not_called()


@pytest.mark.asyncio
async def test_send_posts_to_resend_on_success():
    http = MagicMock()
    response = MagicMock(spec=httpx.Response, status_code=200, text="{}")
    http.post = AsyncMock(return_value=response)

    await mailer.send(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        subject="Test subject",
        to="user@example.local",
        html="<p>hello</p>",
        text="hello",
    )
    http.post.assert_awaited_once()
    args, kwargs = http.post.call_args
    assert args[0] == "https://api.resend.com/emails"
    assert kwargs["headers"]["Authorization"] == "Bearer re_test"
    body = kwargs["json"]
    assert body["to"] == ["user@example.local"]
    assert body["subject"] == "Test subject"
    assert body["html"] == "<p>hello</p>"
    assert body["text"] == "hello"


@pytest.mark.asyncio
async def test_send_swallows_http_500():
    http = MagicMock()
    response = MagicMock(spec=httpx.Response, status_code=500, text="upstream down")
    http.post = AsyncMock(return_value=response)
    # Must not raise.
    await mailer.send(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        subject="x",
        to="user@example.local",
        html="x",
        text="x",
    )


@pytest.mark.asyncio
async def test_send_swallows_network_error():
    http = MagicMock()
    http.post = AsyncMock(side_effect=httpx.ConnectError("network down"))
    # Must not raise.
    await mailer.send(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        subject="x",
        to="user@example.local",
        html="x",
        text="x",
    )


@pytest.mark.asyncio
async def test_send_approval_email_renders_dashboard_url():
    """Renders the approval template with the user + dashboard_url and posts it."""
    http = MagicMock()
    response = MagicMock(spec=httpx.Response, status_code=200, text="{}")
    http.post = AsyncMock(return_value=response)
    user = SimpleNamespace(email="alice@example.local", name="Alice")
    # A MagicMock stands in for the DB session: _record_email only calls
    # session.add(EmailLog(...)), so this captures the row without a real DB.
    session = MagicMock()
    await mailer.send_approval_email(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        session=session,
        user=user,
        dashboard_url="https://api.example/dashboard",
        tutorial_url="https://api.example/tutorial",
        available_models=[
            {
                "id": "llama-405b",
                "served_model_name": "meta-llama/Llama-3.1-405B",
                "gpu_shape": "8xH200",
                "max_model_len": 32768,
            }
        ],
    )
    body = http.post.call_args.kwargs["json"]
    assert body["to"] == ["alice@example.local"]
    assert "Alice" in body["text"]
    assert "https://api.example/dashboard" in body["text"]
    assert "https://api.example/tutorial" in body["text"]
    assert "llama-405b" in body["text"]
    assert "API keys by email" in body["text"]
    assert "approved" in body["subject"].lower()
    # An EmailLog row is recorded with the full rendered body + 'approval' kind.
    logged = session.add.call_args.args[0]
    assert logged.kind == "approval"
    assert logged.send_status == "sent"
    assert "Alice" in logged.body_text
    assert "https://api.example/dashboard" in logged.body_html
    assert "llama-405b" in logged.body_html


@pytest.mark.asyncio
async def test_send_rejection_email_neutral_wording():
    http = MagicMock()
    response = MagicMock(spec=httpx.Response, status_code=200, text="{}")
    http.post = AsyncMock(return_value=response)
    user = SimpleNamespace(email="bob@example.local", name=None)
    session = MagicMock()
    await mailer.send_rejection_email(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        session=session,
        user=user,
    )
    body = http.post.call_args.kwargs["json"]
    assert body["to"] == ["bob@example.local"]
    # When user.name is None, the template falls back to email.
    assert "bob@example.local" in body["text"]
    assert session.add.call_args.args[0].kind == "rejection"


@pytest.mark.asyncio
async def test_skipped_send_still_records_log_row():
    """Even when email is disabled, the attempt is logged (status 'skipped')."""
    http = MagicMock()
    http.post = AsyncMock()
    session = MagicMock()
    user = SimpleNamespace(email="carol@example.local", name="Carol")
    await mailer.send_approval_email(
        settings=_settings(email_enabled=False, resend_api_key="re_test"),
        http=http,
        session=session,
        user=user,
        dashboard_url="https://api.example/dashboard",
    )
    http.post.assert_not_called()
    logged = session.add.call_args.args[0]
    assert logged.send_status == "skipped"
    assert logged.skip_reason == "email_disabled"
    # Body is still captured for audit even though nothing was sent.
    assert "Carol" in logged.body_text


@pytest.mark.asyncio
async def test_send_personalized_beta_invite_renders_single_combined_message():
    http = MagicMock()
    response = MagicMock(spec=httpx.Response, status_code=200, text="{}")
    http.post = AsyncMock(return_value=response)
    session = MagicMock()

    await mailer.send_personalized_beta_invite_email(
        settings=_settings(email_enabled=True, resend_api_key="re_test"),
        http=http,
        session=session,
        to_email="dana@example.local",
        recipient_name="Dana",
        invite_url="https://base.example/invite/token123",
        expires_at=dt.datetime(2026, 6, 30, 12, tzinfo=dt.UTC),
        survey_respondent=True,
        discord_url="https://discord.example/invite",
        survey_url="https://forms.example/survey",
    )

    body = http.post.call_args.kwargs["json"]
    assert body["to"] == ["dana@example.local"]
    assert body["subject"] == "Invitation to ACS Infra beta"
    assert "Dear Dana" in body["text"]
    assert "You recently filled in our Base Models survey" in body["text"]
    assert "https://base.example/invite/token123" in body["text"]
    assert "https://discord.example/invite" in body["text"]
    assert "https://forms.example/survey" not in body["text"]

    logged = session.add.call_args.args[0]
    assert logged.kind == "invite_personalized"
    assert logged.send_status == "sent"
    assert "https://base.example/invite/token123" in logged.body_text


# ---- signup notification: unanswered questions stay visible (ACS-311) -------


def _notification_ctx(**overrides):
    ctx = {
        "email": "applicant@example.local",
        "applicant_name": "Test Applicant",
        "org": "Independent",
        "profile_link": None,
        "use_case": "Evals, data pipelines",
        "outcome": None,
        "prior_work": None,
        "referral": None,
        "source": None,
        "admin_url": "https://infra.example/admin/users/1",
    }
    ctx.update(overrides)
    return ctx


def test_signup_notification_marks_skipped_questions_not_provided():
    """The reviewer decides from this email, so a skipped question must be
    visible as skipped — not silently absent (ACS-311)."""
    from wrapper.mailer import _render

    ctx = _notification_ctx()
    for tmpl in ("signup_notification.txt", "signup_notification.html"):
        body = _render(tmpl, **ctx)
        assert "Evals, data pipelines" in body, tmpl
        for label in ("Results", "Prior work", "Profile"):
            assert label in body, f"{tmpl}: {label} row missing"
        # One "not provided" per skipped question: results, prior work,
        # referral, profile link.
        assert body.count("not provided") == 4, tmpl


def test_signup_notification_renders_answers_when_present():
    from wrapper.mailer import _render

    ctx = _notification_ctx(
        profile_link="https://example.dev",
        outcome="An atlas of steering directions.",
        prior_work="https://example.com/paper",
        referral="Twitter, via @someone",
    )
    for tmpl in ("signup_notification.txt", "signup_notification.html"):
        body = _render(tmpl, **ctx)
        assert "not provided" not in body, tmpl
        for value in (
            "https://example.dev",
            "An atlas of steering directions.",
            "https://example.com/paper",
            "Twitter, via @someone",
        ):
            assert value in body, f"{tmpl}: {value} missing"


def test_signup_notification_html_escapes_answers():
    """Autoescaping must hold for the fields an applicant controls."""
    from wrapper.mailer import _render

    body = _render(
        "signup_notification.html",
        **_notification_ctx(prior_work="<script>alert(1)</script>"),
    )
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


# ---- password-reset copy: set vs reset (ACS-313) ----------------------------


def test_password_reset_email_copy_switches_for_first_timers():
    """A password-less account is *setting* a password, not resetting one —
    the wrong verb here is confusing precisely when someone is already stuck."""
    import re

    from wrapper.mailer import _render

    def flat(tmpl, first_time):
        body = _render(
            tmpl, user_name="A", reset_url="https://x/y", expiry_minutes=60, first_time=first_time
        )
        return re.sub(r"\s+", " ", body)  # both parts hard-wrap

    for tmpl in ("password_reset.txt", "password_reset.html"):
        first = flat(tmpl, True)
        assert "set the password" in first, tmpl
        assert "no password will be set" in first, tmpl
        assert "reset the password" not in first, tmpl

        again = flat(tmpl, False)
        assert "reset the password" in again, tmpl
        assert "password won't change" in again, tmpl
