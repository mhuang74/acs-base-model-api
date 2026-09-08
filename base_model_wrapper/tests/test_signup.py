"""DB-gated tests for /signup + login gating on pending/rejected accounts."""

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
from wrapper.routes.web import _SIGNUP_TS_SALT
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run signup tests",
)


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-signup-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ["SIGNUP_ENABLED"] = "true"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    # Don't leak the flag into later test files (ACS-308).
    os.environ.pop("SIGNUP_ENABLED", None)


def _unique_email() -> str:
    return f"signup-{uuid.uuid4().hex[:8]}@example.local"


def _guard(client, age_seconds: int = 10) -> dict:
    """Bot-guard fields (ACS-260) a real browser submit carries: a signed
    render-timestamp old enough to clear the time-trap. The honeypot field is
    simply absent, exactly as it is for a human submit."""
    secret = client.app.state.settings.session_secret
    ts = URLSafeSerializer(secret, salt=_SIGNUP_TS_SALT).dumps(int(time.time()) - age_seconds)
    return {"ts": ts}


async def _user_row(email: str) -> User | None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            return (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
    finally:
        await engine.dispose()


# ---- signup -----------------------------------------------------------------


@dbtest
async def test_signup_creates_pending_user(client):
    email = _unique_email()
    r = client.post(
        "/signup",
        data={
            "email": email,
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "ACS",
            "use_case": "Probing base-model personas for alignment research.",
            "agree": "yes",
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/signup?ok=1"

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one()
            assert u.status == "pending"
            assert u.name == "Test Applicant"
            assert u.org == "ACS"
            # No password at application time (ACS-314) — approval mints a
            # set-password link to the account address.
            assert u.password_hash is None
            # ACS-24 / ACS-170: use-case captured, agreement timestamped.
            assert u.signup_use_case == "Probing base-model personas for alignment research."
            assert u.agreed_terms_at is not None
    finally:
        await engine.dispose()


@dbtest
async def test_signup_split_questions_persist(client):
    """ACS-302: the three optional split questions persist trimmed; blank → NULL."""
    email = _unique_email()
    r = client.post(
        "/signup",
        data={
            "email": email,
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "Steering-vector experiments on llama-405b.",
            "profile_link": "  https://scholar.google.com/citations?user=xyz  ",
            "outcome": "  An atlas of steering directions; publish on LessWrong.  ",
            "prior_work": "See https://example.com/paper and my LW posts.",
            "referral": "   ",
            "agree": "yes",
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    u = await _user_row(email)
    assert u is not None
    assert u.signup_profile_link == "https://scholar.google.com/citations?user=xyz"
    assert u.signup_outcome == "An atlas of steering directions; publish on LessWrong."
    assert u.signup_prior_work == "See https://example.com/paper and my LW posts."
    assert u.signup_referral is None


@dbtest
async def test_signup_optional_question_length_cap(client):
    email = _unique_email()
    r = client.post(
        "/signup",
        data={
            "email": email,
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "Some research.",
            "outcome": "x" * 2001,
            "agree": "yes",
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "Results" in r.text
    assert await _user_row(email) is None


@dbtest
def test_signup_error_rerender_preserves_split_answers(client):
    """A validation error must not eat the applicant's optional answers."""
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "Some research.",
            "profile_link": "https://scholar.google.com/citations?user=xyz",
            "outcome": "An atlas of steering directions.",
            "prior_work": "https://example.com/paper",
            "referral": "Twitter, via @someone",
            **_guard(client),  # no "agree" → validation error re-renders the form
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "agree to the usage rules" in r.text
    assert "An atlas of steering directions." in r.text
    assert "https://example.com/paper" in r.text
    assert "https://scholar.google.com/citations?user=xyz" in r.text
    assert "Twitter, via @someone" in r.text


@dbtest
def test_signup_form_shows_split_questions(client):
    r = client.get("/signup")
    assert r.status_code == 200
    for label in (
        "Planned usage",
        "Results",
        "Prior work",
        "How did you hear about us?",
        "Profile link",
    ):
        assert label in r.text
    assert "err on the side of rejecting" in r.text
    # Required fields marked with the conventional asterisk (ACS-303): email,
    # name, org, planned usage — password fields are gone (ACS-314).
    assert r.text.count('class="req"') >= 4
    assert "full name is preferred" in r.text
    # ACS-303 follow-up: asterisk lives inline in the caption span (the base
    # label style is a flex column — a bare span would get its own row), and
    # the "(optional…)" markers are gone (the asterisk carries the meaning).
    assert "<span>Email <span class=\"req\"" in r.text
    assert "(optional" not in r.text
    # ACS-314: no credentials on the application form at all.
    assert 'name="password"' not in r.text


@dbtest
async def test_signup_source_attribution(client):
    """?src= on the form link is persisted per application; bare form → NULL;
    the value survives a validation-error re-render (ACS-210)."""
    # GET with ?src= renders the hidden field…
    r = client.get("/signup?src=constellation")
    assert 'name="src" value="constellation"' in r.text
    # …bare GET renders no src field at all.
    r = client.get("/signup")
    assert 'name="src"' not in r.text
    # Hostile src is autoescaped — locks the select_autoescape assumption so a
    # future template-env change can't silently reopen quote-breaking XSS.
    r = client.get('/signup?src="><script>alert(1)</script>')
    assert "<script>alert(1)" not in r.text
    assert "&#34;&gt;" in r.text
    # Over-long src truncates to the 64-char cap.
    r = client.get("/signup?src=" + "x" * 200)
    assert f'value="{"x" * 64}"' in r.text
    assert "x" * 65 not in r.text

    # An error re-render keeps carrying it.
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "Some research.",
            "src": "constellation",
            # no "agree" → 400
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert 'name="src" value="constellation"' in r.text

    # A tagged submit persists the source; an untagged one stays NULL.
    tagged, untagged = _unique_email(), _unique_email()
    for email, extra in ((tagged, {"src": "constellation"}), (untagged, {})):
        r = client.post(
            "/signup",
            data={
                "email": email,
                "password": "applicant-pw-12345",
                "confirm_password": "applicant-pw-12345",
                "name": "Test Applicant",
                "org": "Independent",
                "use_case": "Some research.",
                "agree": "yes",
                **_guard(client),
                **extra,
            },
            follow_redirects=False,
        )
        assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u1 = (await s.execute(select(User).where(User.email == tagged))).scalar_one()
            u2 = (await s.execute(select(User).where(User.email == untagged))).scalar_one()
            assert u1.signup_source == "constellation"
            assert u2.signup_source is None
    finally:
        await engine.dispose()


@dbtest
def test_signup_use_case_server_side_cap(client):
    """The 2000-char textarea maxlength is mirrored server-side (a crafted POST
    bypassing the form must not store an unbounded blob)."""
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "x" * 2001,
            "agree": "yes",
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "to 2000 characters or fewer" in r.text


@dbtest
def test_signup_requires_agreement(client):
    """No account is created unless the usage-rules checkbox is ticked (ACS-170)."""
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "use_case": "Some research.",
            # no "agree"
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "agree to the usage rules" in r.text


@dbtest
def test_signup_requires_use_case(client):
    """No account is created without a use-case to evaluate (ACS-24)."""
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "agree": "yes",
            # no "use_case"
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "Planned usage" in r.text


@dbtest
def test_signup_requires_name_and_org(client):
    """Name and org are mandatory (pseudonym / "Independent" are fine) so the
    reviewer always has a who-is-this signal (ACS-24 refinement)."""
    base = {
        "email": _unique_email(),
        "password": "applicant-pw-12345",
        "confirm_password": "applicant-pw-12345",
        "use_case": "Some research.",
        "agree": "yes",
        **_guard(client),
    }
    # Assert on the error sentences, not "pseudonym"/"Independent" — those words
    # also appear in the field labels of every re-render, so they can't
    # distinguish which check fired.
    r = client.post("/signup", data={**base, "org": "Independent"}, follow_redirects=False)
    assert r.status_code == 400
    assert "Please enter your name" in r.text
    r = client.post("/signup", data={**base, "name": "Test Applicant"}, follow_redirects=False)
    assert r.status_code == 400
    assert "Please enter your organization" in r.text


@dbtest
async def test_signup_duplicate_email_is_generic_400(client):
    email = _unique_email()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(User(email=email, password_hash=hash_password("existing-pw")))
    finally:
        await engine.dispose()

    r = client.post(
        "/signup",
        data={
            "email": email,
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            "name": "Test Applicant",
            "org": "Independent",
            "use_case": "Some research.",
            "agree": "yes",
            **_guard(client),
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    # Generic wording — must not reveal that the email is already registered.
    assert "Could not create account" in r.text


@dbtest
def test_signup_takes_no_password(client):
    """ACS-314: the form asks for no credentials, and a scripted POST that
    sends them anyway must not set one — access requires the approval email."""
    r = client.get("/signup")
    assert 'name="password"' not in r.text
    assert 'name="confirm_password"' not in r.text


@dbtest
def test_signup_disabled_shows_friendly_page_and_refuses_post(client, monkeypatch):
    """When signups are off (private beta), GET /signup shows a friendly
    "email us" page (not a bare 404) with no signup form, and POST still
    refuses to create an account (ACS-78)."""
    client.app.state.settings.signup_enabled = False
    try:
        r = client.get("/signup", follow_redirects=False)
        assert r.status_code == 200
        body = r.text
        assert "private beta" in body.lower()
        assert "infra@acsresearch.org" in body
        # The actual signup form must be suppressed.
        assert 'action="/signup"' not in body
        # POST must still hard-refuse (no account creation when disabled).
        r2 = client.post(
            "/signup",
            data={
                "email": _unique_email(),
                "password": "x" * 10,
                "confirm_password": "x" * 10,
            },
            follow_redirects=False,
        )
        assert r2.status_code == 404
    finally:
        client.app.state.settings.signup_enabled = True


# ---- bot guards (ACS-260) -----------------------------------------------------


def _valid_form(client, email: str, **overrides) -> dict:
    """A fully valid application, guard fields included — overrides let each
    test trip exactly one guard."""
    return {
        "email": email,
        "password": "applicant-pw-12345",
        "confirm_password": "applicant-pw-12345",
        "name": "Test Applicant",
        "org": "Independent",
        "use_case": "Some research.",
        "agree": "yes",
        **_guard(client),
        **overrides,
    }


@dbtest
async def test_signup_honeypot_dropped_silently(client, monkeypatch):
    """A filled honeypot gets the exact 303 a real submit gets — and no row
    and no admin notification, so the bot can't tell it was caught. The mailer
    spy pins "no notification" directly, not just via no-row (review #266)."""
    import wrapper.routes.web as webroutes

    notified = []

    async def _spy(**kwargs):
        notified.append(kwargs)

    monkeypatch.setattr(webroutes.mailermod, "send_signup_notification", _spy)
    email = _unique_email()
    r = client.post(
        "/signup",
        data=_valid_form(client, email, website="https://spam.example"),
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/signup?ok=1"
    assert await _user_row(email) is None
    assert notified == []


@dbtest
async def test_signup_too_fast_dropped_silently(client):
    """A submit under the 3 s human floor (here: rendered this instant) is
    dropped with the same fake success."""
    email = _unique_email()
    r = client.post(
        "/signup",
        data=_valid_form(client, email, **_guard(client, age_seconds=0)),
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/signup?ok=1"
    assert await _user_row(email) is None


@dbtest
async def test_signup_missing_or_forged_ts_dropped_silently(client):
    """No render-token or a forged one → dropped; bots that skip hidden fields
    or replay a token signed with the wrong key never create rows."""
    for ts in ("", "garbage-token"):
        email = _unique_email()
        r = client.post(
            "/signup",
            data=_valid_form(client, email, ts=ts),
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/signup?ok=1"
        assert await _user_row(email) is None


@dbtest
def test_signup_form_renders_guard_fields(client):
    """The rendered form carries both guard fields, and an error re-render
    passes the ORIGINAL token through — a corrected resubmit measures from the
    first render, not the re-render."""
    r = client.get("/signup")
    assert 'name="ts" value="' in r.text
    assert 'name="website"' in r.text
    guard = _guard(client)
    r = client.post(
        "/signup",
        data={
            "email": _unique_email(),
            "password": "applicant-pw-12345",
            "confirm_password": "applicant-pw-12345",
            # no "agree" → 400 re-render
            **guard,
        },
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert f'name="ts" value="{guard["ts"]}"' in r.text


@dbtest
def test_signup_burst_rate_limited(client):
    """The 6th rapid POST from one IP trips the 5/minute layer of the stacked
    5/minute;20/day limit. (Proving the /day layer needs 21 posts — the parse
    of the combined string is exercised at import time either way.)"""
    client.app.state.limiter.enabled = True
    try:
        statuses = []
        for _ in range(6):
            r = client.post(
                "/signup",
                data={
                    "email": _unique_email(),
                    "password": "x" * 10,
                    "confirm_password": "x" * 10,
                },
                follow_redirects=False,
            )
            statuses.append(r.status_code)
        assert statuses[-1] == 429, statuses
    finally:
        client.app.state.limiter.enabled = False


# ---- login gating on non-approved accounts ----------------------------------


@dbtest
async def test_login_blocked_when_pending(client):
    email = _unique_email()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(
                User(
                    email=email,
                    password_hash=hash_password("pending-pw-12345"),
                    status="pending",
                )
            )
    finally:
        await engine.dispose()

    r = client.post(
        "/login",
        data={"email": email, "password": "pending-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert "pending admin approval" in r.text


@dbtest
async def test_login_blocked_when_rejected(client):
    email = _unique_email()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(
                User(
                    email=email,
                    password_hash=hash_password("rejected-pw-12345"),
                    status="rejected",
                )
            )
    finally:
        await engine.dispose()

    r = client.post(
        "/login",
        data={"email": email, "password": "rejected-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert "not approved" in r.text


@dbtest
async def test_login_works_for_approved(client):
    email = _unique_email()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(
                User(
                    email=email,
                    password_hash=hash_password("approved-pw-12345"),
                    status="approved",
                )
            )
    finally:
        await engine.dispose()

    r = client.post(
        "/login",
        data={"email": email, "password": "approved-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 303


# ---- re-application after rejection (ACS-312) -------------------------------


async def _reject(user_id) -> None:
    import datetime as dt

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            u.status = "rejected"
            u.rejected_at = dt.datetime.now(tz=dt.UTC)
    finally:
        await engine.dispose()


def _apply(client, email, *, use_case, password="applicant-pw-12345", **extra):
    data = {
        "email": email,
        "password": password,
        "confirm_password": password,
        "name": extra.pop("name", "Test Applicant"),
        "org": "Independent",
        "use_case": use_case,
        "agree": "yes",
        **extra,
        **_guard(client),
    }
    return client.post("/signup", data=data, follow_redirects=False)


@dbtest
async def test_rejected_applicant_can_reapply_and_history_is_kept(client):
    """A declined applicant is told they may come back — so a new application
    reopens their row instead of dead-ending, and the old answers survive."""
    email = _unique_email()
    r = _apply(client, email, use_case="Curious about the platform.", outcome="Have a look.")
    assert r.status_code == 303
    first = await _user_row(email)
    await _reject(first.id)

    r = _apply(client, email, use_case="Steering-vector experiments on llama-405b.")
    assert r.status_code == 303
    # &again=1 drives the "we've updated your earlier application" note.
    assert r.headers["location"] == "/signup?ok=1&again=1"

    u = await _user_row(email)
    assert u.id == first.id  # same row — the record, not a new account
    assert u.status == "pending"
    assert u.rejected_at is None
    assert u.signup_use_case == "Steering-vector experiments on llama-405b."
    # Previous answers preserved for the reviewer.
    assert "re-applied" in u.notes
    assert "Curious about the platform." in u.notes
    assert "Have a look." in u.notes


@dbtest
async def test_signup_never_sets_credentials(client):
    """Neither a fresh application nor a re-application may set a password —
    scripted POSTs can still send the fields. Access must always come from the
    approval email's set-password link, which only reaches the account address
    (ACS-312 credential guard, preserved under ACS-314)."""
    email = _unique_email()
    _apply(client, email, use_case="First try.", password="original-pw-12345")
    first = await _user_row(email)
    assert first.password_hash is None
    await _reject(first.id)

    r = _apply(client, email, use_case="Second try.", password="attacker-pw-12345")
    assert r.status_code == 303

    u = await _user_row(email)
    assert u.password_hash is None
    r = client.post(
        "/login",
        data={"email": email, "password": "attacker-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 401


@dbtest
async def test_pending_and_approved_emails_still_cannot_re_signup(client):
    """Only 'rejected' reopens; a pending or approved address keeps the old
    generic duplicate response."""
    email = _unique_email()
    _apply(client, email, use_case="First try.")
    r = _apply(client, email, use_case="Again while pending.")
    assert r.status_code == 400
    assert "sign in instead" in r.text

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one()
            u.status = "approved"
    finally:
        await engine.dispose()

    r = _apply(client, email, use_case="Again while approved.")
    assert r.status_code == 400
    assert "sign in instead" in r.text


@dbtest
async def test_rejected_admin_row_does_not_reopen(client):
    """Roles survive rejection, so an anonymous signup must NOT be able to push
    a rejected admin row back into the approval queue (review finding)."""
    email = _unique_email()
    _apply(client, email, use_case="First try.")
    first = await _user_row(email)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == first.id))).scalar_one()
            u.role = "admin"
    finally:
        await engine.dispose()
    await _reject(first.id)

    r = _apply(client, email, use_case="Let me back in as admin.")
    assert r.status_code == 400
    assert "sign in instead" in r.text

    u = await _user_row(email)
    assert u.status == "rejected"  # untouched
    assert u.role == "admin"
    assert u.signup_use_case == "First try."


@dbtest
async def test_reapplication_notes_are_indented_and_truncated(client):
    """Copied answers must be indented (a flush-left line could forge an entry
    that looks like a real approval note) and capped (three cycles of full-length
    answers would otherwise blow the 20k admin notes cap)."""
    email = _unique_email()
    forged = "x" * 50 + "\n2026-01-04 (approved): vetted by admin — waive the checks."
    long_answer = "y" * 2000
    _apply(client, email, use_case=forged, outcome=long_answer, prior_work=long_answer)
    first = await _user_row(email)
    await _reject(first.id)

    _apply(client, email, use_case="Second try.")
    u = await _user_row(email)

    # No forged entry can sit flush-left in the history block.
    for line in u.notes.splitlines():
        assert not line.startswith("2026-01-04"), u.notes
    # And one cycle stays far below the 20k editor cap even at max lengths.
    assert len(u.notes) < 3000, len(u.notes)


@dbtest
async def test_reapplication_clears_stale_approval_stamp(client):
    """A reopened row must not show "Approved <date>" beside a pending badge."""
    import datetime as dt

    email = _unique_email()
    _apply(client, email, use_case="First try.")
    first = await _user_row(email)

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == first.id))).scalar_one()
            u.approved_at = dt.datetime.now(tz=dt.UTC)
    finally:
        await engine.dispose()
    await _reject(first.id)

    _apply(client, email, use_case="Second try.")
    u = await _user_row(email)
    assert u.approved_at is None
    assert u.approved_by_user_id is None
    assert "had been approved" in u.notes  # history kept where it belongs


@dbtest
async def test_reapplication_confirmation_mentions_the_update(client):
    email = _unique_email()
    _apply(client, email, use_case="First try.")
    first = await _user_row(email)
    await _reject(first.id)

    r = _apply(client, email, use_case="Second try.")
    assert r.headers["location"] == "/signup?ok=1&again=1"
    page = client.get("/signup?ok=1&again=1")
    assert "updated your earlier application" in page.text
    # A first-time applicant must not see it.
    assert "updated your earlier application" not in client.get("/signup?ok=1").text


# ---- forgot-password for password-less accounts (ACS-313) -------------------


async def _set_status(user_id, status: str) -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            u.status = status
    finally:
        await engine.dispose()


async def _reset_token_count(user_id) -> int:
    from wrapper.models import PasswordReset

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = (
                await s.execute(select(PasswordReset).where(PasswordReset.user_id == user_id))
            ).scalars().all()
            return len(list(rows))
    finally:
        await engine.dispose()


@dbtest
async def test_forgot_password_recovers_approved_account_without_a_password(client):
    """After ACS-314 every approved user starts password-less, and the approval
    email's set-password link expires — so self-service recovery must work for
    them, or the only way back is asking an admin (ACS-313)."""
    email = _unique_email()
    _apply(client, email, use_case="Research.")
    user = await _user_row(email)
    assert user.password_hash is None
    await _set_status(user.id, "approved")

    r = client.post("/forgot-password", data={"email": email}, follow_redirects=False)
    assert r.status_code == 200
    assert await _reset_token_count(user.id) == 1
    # …and the mail says "set", not "reset" (they never had one).
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            from wrapper.models import EmailLog

            log_row = (
                await s.execute(
                    select(EmailLog)
                    .where(EmailLog.user_id == user.id, EmailLog.kind == "password_reset")
                    .order_by(EmailLog.created_at.desc())
                )
            ).scalars().first()
            assert log_row is not None
            assert log_row.subject == "Set your ACS Infra password"
    finally:
        await engine.dispose()


@dbtest
async def test_forgot_password_ignores_pending_and_rejected_accounts(client):
    """No access to recover — and the response stays neutral either way."""
    for status in ("pending", "rejected"):
        email = _unique_email()
        _apply(client, email, use_case="Research.")
        user = await _user_row(email)
        await _set_status(user.id, status)

        r = client.post("/forgot-password", data={"email": email}, follow_redirects=False)
        assert r.status_code == 200, status
        assert await _reset_token_count(user.id) == 0, status


# ---- ACS-353: suspended accounts --------------------------------------------


@dbtest
async def test_login_blocked_when_suspended(client):
    """Suspension is reversible, so the copy points at a way back rather than
    reading as a verdict."""
    email = _unique_email()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            s.add(
                User(
                    email=email,
                    password_hash=hash_password("suspended-pw-12345"),
                    status="suspended",
                )
            )
    finally:
        await engine.dispose()

    r = client.post(
        "/login",
        data={"email": email, "password": "suspended-pw-12345"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert "suspended" in r.text.lower()
    # Not the generic fallback, and not the rejection copy.
    assert "was not approved" not in r.text


@dbtest
async def test_forgot_password_ignores_suspended_accounts(client):
    email = _unique_email()
    _apply(client, email, use_case="Research.")
    user = await _user_row(email)
    await _set_status(user.id, "suspended")

    r = client.post("/forgot-password", data={"email": email}, follow_redirects=False)
    assert r.status_code == 200
    assert await _reset_token_count(user.id) == 0


@dbtest
async def test_forgot_password_ignores_unapproved_accounts_that_have_a_password(client):
    """Regression (ACS-353): recoverability used to be
    ``password_hash is not None or status == 'approved'``, whose first clause
    short-circuited the status test — so a previously-approved-then-rejected
    user who still had a password kept receiving reset emails, contradicting the
    rule stated in that very comment. The older
    ``test_forgot_password_ignores_pending_and_rejected_accounts`` never caught
    it because its users are password-less.
    """
    for status in ("rejected", "suspended", "pending"):
        email = _unique_email()
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                s.add(
                    User(
                        email=email,
                        password_hash=hash_password("had-access-pw-12345"),
                        status=status,
                    )
                )
        finally:
            await engine.dispose()

        user = await _user_row(email)
        r = client.post("/forgot-password", data={"email": email}, follow_redirects=False)
        assert r.status_code == 200, status
        assert await _reset_token_count(user.id) == 0, status
