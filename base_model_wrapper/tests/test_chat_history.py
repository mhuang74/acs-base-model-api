"""Tests for chat history, streaming workbench, exports, and the /usage tab.

Unit-level tests cover helpers (title derivation, jsonl record shape).
DB-gated tests (TEST_DATABASE_URL) exercise the routes via TestClient so the
ownership check + snapshot persistence + usage view actually hit Postgres.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, ChatSession, ChatSnapshot, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


# --- unit tests (no DB) ------------------------------------------------------


def test_derive_title_single_line():
    from wrapper.main import _derive_title
    assert _derive_title("Once upon a time in a far away land") == "Once upon a time in a far away land"


def test_derive_title_truncates_long():
    from wrapper.main import _derive_title
    long_prompt = "x" * 100
    assert _derive_title(long_prompt) == "x" * 40


def test_derive_title_strips_and_takes_first_line():
    from wrapper.main import _derive_title
    assert _derive_title("  hello world  \nignored\n") == "hello world"


def test_derive_title_falls_back_to_untitled():
    from wrapper.main import _derive_title
    assert _derive_title("") == "Untitled"
    assert _derive_title("   ") == "Untitled"


def test_session_to_jsonl_record_shape():
    """Export records carry session metadata + all snapshots, with iso-formatted ts."""
    import datetime as _dt
    from wrapper.main import _session_to_jsonl_record

    chat = ChatSession(
        user_id=uuid.uuid4(),
        title="My chat",
        prompt_text="hello world",
        last_max_tokens=200,
        last_temperature=0.7,
    )
    chat.id = uuid.uuid4()
    chat.created_at = _dt.datetime(2026, 5, 1, tzinfo=_dt.timezone.utc)
    chat.updated_at = _dt.datetime(2026, 5, 2, tzinfo=_dt.timezone.utc)

    snap = ChatSnapshot(
        session_id=chat.id,
        prompt_before="hello",
        completion_text=" world",
        n_completion=2,
        max_tokens=200,
        temperature=0.7,
        cancelled=False,
    )
    snap.ts = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)

    rec = _session_to_jsonl_record(chat, [snap])
    assert rec["id"] == str(chat.id)
    assert rec["title"] == "My chat"
    assert rec["prompt_text"] == "hello world"
    assert rec["last_max_tokens"] == 200
    assert rec["created_at"] == "2026-05-01T00:00:00+00:00"
    assert len(rec["snapshots"]) == 1
    assert rec["snapshots"][0]["prompt_before"] == "hello"
    assert rec["snapshots"][0]["completion_text"] == " world"
    assert rec["snapshots"][0]["cancelled"] is False
    # Round-trips through json without raising.
    json.dumps(rec)


# --- DB-gated route tests ----------------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan started + the same envvars as test_self_service."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-chat-history-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("RATE_LIMIT_LOGIN_PER_IP", "1000/minute")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


async def _make_user(password: str = "test-pw-12345") -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = f"chat-history-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password))
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _make_chat_session(user_id: uuid.UUID, *, title: str = "fixture") -> uuid.UUID:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = ChatSession(user_id=user_id, title=title, prompt_text="seed")
            s.add(chat)
            await s.flush()
            return chat.id
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


# Ownership: user A cannot open user B's session.
@dbtest
async def test_chat_open_other_users_session_404(client):
    user_a_id, email_a = await _make_user()
    user_b_id, _ = await _make_user()

    # Session belongs to B.
    chat_b_id = await _make_chat_session(user_b_id, title="b-owned")

    _login(client, email_a, "test-pw-12345")
    r = client.get(f"/workbench/{chat_b_id}", follow_redirects=False)
    assert r.status_code == 404


@dbtest
async def test_chat_new_creates_and_lists_session(client):
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")

    r = client.post("/workbench/new", follow_redirects=False)
    assert r.status_code == 303
    new_url = r.headers["location"]
    assert new_url.startswith("/workbench/")
    new_id = new_url.rsplit("/", 1)[-1]

    # Fetch the chat page; the sidebar should list it.
    r = client.get(new_url)
    assert r.status_code == 200
    assert new_id in r.text
    assert "Untitled" in r.text


@dbtest
async def test_chat_rename_persists(client):
    user_id, email = await _make_user()
    chat_id = await _make_chat_session(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(
        f"/workbench/{chat_id}/rename", data={"title": "renamed-via-test"}, follow_redirects=False
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.title == "renamed-via-test"
    finally:
        await engine.dispose()


@dbtest
async def test_chat_delete_soft_deletes(client):
    user_id, email = await _make_user()
    chat_id = await _make_chat_session(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(f"/workbench/{chat_id}/delete", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.archived_at is not None
    finally:
        await engine.dispose()


@dbtest
async def test_chat_revert_restores_prompt_text(client):
    """A revert sets chat.prompt_text back to the snapshot's prompt_before."""
    user_id, email = await _make_user()
    chat_id = await _make_chat_session(user_id)
    _login(client, email, "test-pw-12345")

    # Insert a snapshot directly to avoid spinning up an upstream mock.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            chat.prompt_text = "current rolling prompt"
            snap = ChatSnapshot(
                session_id=chat.id,
                prompt_before="earlier state",
                completion_text=" later",
                n_completion=1,
                max_tokens=200,
                temperature=0.7,
                cancelled=False,
            )
            s.add(snap)
            await s.flush()
            snap_id = snap.id
    finally:
        await engine.dispose()

    r = client.post(f"/workbench/{chat_id}/revert/{snap_id}", follow_redirects=False)
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.prompt_text == "earlier state"
    finally:
        await engine.dispose()


@dbtest
async def test_chat_export_txt_returns_prompt(client):
    user_id, email = await _make_user()
    chat_id = await _make_chat_session(user_id, title="export-target")

    # Set prompt_text to something recognisable.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            chat.prompt_text = "the rolling prompt content"
    finally:
        await engine.dispose()

    _login(client, email, "test-pw-12345")
    r = client.get(f"/workbench/{chat_id}/export.txt")
    assert r.status_code == 200
    assert r.text == "the rolling prompt content"
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"]
    assert "export-target" in r.headers["content-disposition"]


@dbtest
async def test_chat_export_jsonl_includes_snapshots(client):
    user_id, email = await _make_user()
    chat_id = await _make_chat_session(user_id, title="jsonl-target")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            chat.prompt_text = "p"
            s.add(
                ChatSnapshot(
                    session_id=chat.id,
                    prompt_before="before",
                    completion_text="after",
                    n_completion=1,
                    max_tokens=200,
                    temperature=0.7,
                    cancelled=True,
                )
            )
    finally:
        await engine.dispose()

    _login(client, email, "test-pw-12345")
    r = client.get(f"/workbench/{chat_id}/export.jsonl")
    assert r.status_code == 200
    rec = json.loads(r.text.strip())
    assert rec["title"] == "jsonl-target"
    assert rec["prompt_text"] == "p"
    assert len(rec["snapshots"]) == 1
    assert rec["snapshots"][0]["cancelled"] is True


@dbtest
async def test_chat_export_all_returns_only_own_sessions(client):
    user_a_id, email_a = await _make_user()
    user_b_id, _ = await _make_user()
    await _make_chat_session(user_a_id, title="mine-1")
    await _make_chat_session(user_a_id, title="mine-2")
    await _make_chat_session(user_b_id, title="not-mine")

    _login(client, email_a, "test-pw-12345")
    r = client.get("/workbench/export-all.jsonl")
    assert r.status_code == 200
    lines = [json.loads(line) for line in r.text.strip().split("\n") if line.strip()]
    titles = sorted(rec["title"] for rec in lines)
    assert "not-mine" not in titles
    assert "mine-1" in titles and "mine-2" in titles


@dbtest
async def test_usage_tab_no_keys_renders_empty(client):
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.get("/usage")
    assert r.status_code == 200
    assert "don't have any API keys yet" in r.text or "No usage" in r.text


@dbtest
async def test_usage_tab_with_key_shows_zero_total(client):
    """A user with a key but no requests should render with month_total = 0."""
    user_id, email = await _make_user()

    # Give them an active key with no usage rows.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            s.add(
                ApiKey(
                    user_id=user_id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    name="usage-test",
                    monthly_token_budget=1_000_000,
                )
            )
    finally:
        await engine.dispose()

    _login(client, email, "test-pw-12345")
    r = client.get("/usage")
    assert r.status_code == 200
    # The headline number for "this month" should be 0.
    assert "This month" in r.text
    assert "0</span>" in r.text or ">0<" in r.text


@dbtest
async def test_usage_context_per_key_filter():
    """build_usage_context scopes totals + budgets to ?key=, falling back safely.

    Two keys with distinct this-month usage and per-key budgets. The all-keys
    view sums both and surfaces the user-level monthly-total cap; selecting one
    key scopes the totals to that key and swaps the budget bars to that key's
    own caps; an unknown / malformed key id degrades to the all-keys view.
    """
    from wrapper.auth import _current_period_start
    from wrapper.models import UsageMonthly
    from wrapper.services.usage_reports import build_usage_context

    user_id, email = await _make_user()
    period = _current_period_start()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.id == user_id))).scalar_one()
            user.monthly_token_budget_total = 1_000_000

            gk_a, gk_b = generate_key(), generate_key()
            key_a = ApiKey(
                user_id=user_id, key_hash=gk_a.hash_, key_prefix=gk_a.prefix,
                name="key-a", monthly_token_budget=400_000,
                monthly_input_token_budget=300_000, monthly_output_token_budget=120_000,
            )
            key_b = ApiKey(
                user_id=user_id, key_hash=gk_b.hash_, key_prefix=gk_b.prefix,
                name="key-b", monthly_token_budget=600_000,
            )
            s.add_all([key_a, key_b])
            await s.flush()
            key_a_id, key_b_id = key_a.id, key_b.id
            # key-a: 100k in / 40k out; key-b: 10k in / 5k out (this month).
            s.add(UsageMonthly(key_id=key_a_id, period_start=period,
                               tokens_prompt=100_000, tokens_completion=40_000, request_count=20))
            s.add(UsageMonthly(key_id=key_b_id, period_start=period,
                               tokens_prompt=10_000, tokens_completion=5_000, request_count=3))

        async with session_scope(factory) as s:
            user = (await s.execute(select(User).where(User.id == user_id))).scalar_one()

            # All keys: totals summed, user-level monthly-total cap shown.
            allv = await build_usage_context(s, user, selected_key=None)
            assert allv["selected_key_id"] is None
            assert allv["month_total"] == 155_000  # 140k + 15k
            assert allv["month_input"] == 110_000 and allv["month_output"] == 45_000
            assert allv["month_requests"] == 23
            assert {o["id"] for o in allv["key_options"]} == {str(key_a_id), str(key_b_id)}
            total_bar = next(b for b in allv["budgets"] if b["label"] == "Monthly total (all keys)")
            assert total_bar["budget"] == 1_000_000  # user-level cap

            # Filter to key-a: totals scope to it; budgets become key-a's caps.
            av = await build_usage_context(s, user, selected_key=str(key_a_id))
            assert av["selected_key_id"] == str(key_a_id)
            assert av["selected_key_name"] == "key-a"
            assert av["month_total"] == 140_000
            assert av["month_input"] == 100_000 and av["month_output"] == 40_000
            assert av["month_requests"] == 20
            labels = {b["label"]: b["budget"] for b in av["budgets"]}
            assert labels["Monthly total (this key)"] == 400_000  # key-a's own cap, not the user's
            assert labels["Monthly input"] == 300_000
            assert labels["Monthly output"] == 120_000
            # By-key table still lists every key (all-keys overview).
            assert len(av["per_key"]) == 2

            # Unknown + malformed key ids fall back to the all-keys view.
            for bogus in (str(uuid.uuid4()), "not-a-uuid"):
                fb = await build_usage_context(s, user, selected_key=bogus)
                assert fb["selected_key_id"] is None
                assert fb["month_total"] == 155_000
    finally:
        await engine.dispose()
