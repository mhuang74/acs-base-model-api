"""Compare-mode page render: the shared prompt must be the prompt baseline.

The session prompt is a rolling continuation buffer (the Continue workflow), so
after a single-pane run ``chat_sessions.prompt_text`` holds prompt + completion.
The compare pane must run lanes on the prompt baseline instead — recovered from
the latest roll-forward snapshot only when the session prompt is EXACTLY
baseline + completion (issue #11, resolving #5). Spec's one seam: DB-gated e2e
HTTP render of the Compare-mode page (?mode=compare), asserting external
behavior only — the rendered shared prompt and the embedded boundary data.

Mirrors ``tests/test_workbench_generations.py`` for fixtures/seams. Whole files
are run, never single node ids.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import json
import os
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper import proxy as proxymod
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, ChatGeneration, ChatSession, ChatSnapshot, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run compare-render tests",
)

# Generous poll bound so a genuinely stuck task fails fast instead of hanging
# the suite (same rationale as test_workbench_generations._TEST_WAIT_S).
_POLL_STEPS = 120
_POLL_INTERVAL_S = 0.05


@pytest.fixture
def client():
    """TestClient with lifespan started + the same envvars as test_chat_history."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-compare-render-secret")
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
        email = f"compare-render-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password))
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _make_chat(user_id: uuid.UUID, prompt_text: str) -> uuid.UUID:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = ChatSession(user_id=user_id, title="compare-render", prompt_text=prompt_text)
            s.add(chat)
            await s.flush()
            return chat.id
    finally:
        await engine.dispose()


async def _add_key(user_id: uuid.UUID) -> None:
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
                    name="render-key",
                    monthly_token_budget=1_000_000,
                )
            )
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


class _FakeStreamResp:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return None

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aiter_raw(self):
        for c in self._chunks:
            yield c


def _patch_upstream_chunks(monkeypatch, chunks: list[bytes], status_code: int = 200) -> None:
    """Make ``proxymod.stream_post_with_status`` deterministic for one call."""

    async def _fake(client, upstream_url, api_key, body, timeout_s, **kwargs):
        if status_code >= 400:
            yield {"kind": "error", "data": b"".join(chunks), "status": status_code}
            return
        first = True
        for c in chunks:
            yield {
                "kind": "chunk",
                "data": c,
                "usage": proxymod._extract_usage(c),
                "status": status_code if first else None,
            }
            first = False

    monkeypatch.setattr(proxymod, "stream_post_with_status", _fake)


async def _run_single_completion(
    client: TestClient,
    monkeypatch,
    chat_id: uuid.UUID,
    *,
    prompt: str,
    completion: str,
) -> None:
    """Drive one completed single-pane Continue through the patched upstream."""
    _patch_upstream_chunks(
        monkeypatch,
        [
            f'data: {{"choices":[{{"text":"{completion}"}}]}}\n\n'.encode(),
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": prompt, "max_tokens": 8, "temperature": 0.5},
    )
    assert r.status_code == 200, r.text
    gen_id = uuid.UUID(r.json()["generation_id"])
    await _wait_generation_terminal(gen_id)
    await _wait_snapshots(chat_id, n=1)


async def _wait_generation_terminal(gen_id: uuid.UUID) -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        for _ in range(_POLL_STEPS):
            async with session_scope(factory) as s:
                row = (
                    await s.execute(select(ChatGeneration).where(ChatGeneration.id == gen_id))
                ).scalar_one()
                if row.status != "running":
                    return
            await asyncio.sleep(_POLL_INTERVAL_S)
        raise AssertionError("generation never left running")
    finally:
        await engine.dispose()


async def _wait_snapshots(chat_id: uuid.UUID, *, n: int) -> None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        for _ in range(_POLL_STEPS):
            async with session_scope(factory) as s:
                rows = list(
                    (
                        await s.execute(
                            select(ChatSnapshot).where(ChatSnapshot.session_id == chat_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                if len(rows) >= n:
                    return
            await asyncio.sleep(_POLL_INTERVAL_S)
        raise AssertionError(f"expected {n} snapshot(s), saw fewer")
    finally:
        await engine.dispose()


def _textarea_content(page: str, dom_id: str) -> str:
    m = re.search(
        r'<textarea id="' + re.escape(dom_id) + r'"[^>]*>(.*?)</textarea>',
        page,
        re.DOTALL,
    )
    assert m is not None, f"textarea #{dom_id} not rendered"
    return html_mod.unescape(m.group(1))


def _roll_forward_data(page: str) -> dict | None:
    m = re.search(
        r'<script id="cmp-roll-forward" type="application/json">(.*?)</script>',
        page,
        re.DOTALL,
    )
    assert m is not None, "cmp-roll-forward embed not rendered"
    return json.loads(html_mod.unescape(m.group(1)))


def _compare_page(client: TestClient, chat_id: uuid.UUID) -> str:
    r = client.get(f"/workbench/{chat_id}", params={"mode": "compare"})
    assert r.status_code == 200, r.text[:300]
    return r.text


# --- the seam (issue #11) -----------------------------------------------------


@dbtest
async def test_compare_page_renders_baseline_after_completed_run(client, monkeypatch):
    """After a completed single-pane run the Compare page's shared prompt is the
    prompt baseline, NOT baseline + completion — and the server exposes the
    boundary as separate values for the client toggle guard. Fails pre-fix,
    where the rolled-forward session prompt leaked into the lanes verbatim."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id, "seed")
    await _add_key(user_id)
    _login(client, email, "test-pw-12345")

    await _run_single_completion(
        client, monkeypatch, chat_id, prompt="Once upon a time", completion="hi"
    )

    page = _compare_page(client, chat_id)
    assert _textarea_content(page, "cmp-prompt") == "Once upon a time"
    # The single pane keeps the rolling buffer (Continue workflow untouched).
    assert _textarea_content(page, "wb-prompt") == "Once upon a timehi"
    # Boundary data for the toggle guard: baseline + completion as separate values.
    assert _roll_forward_data(page) == {
        "baseline": "Once upon a time",
        "completion": "hi",
    }


@dbtest
async def test_compare_page_renders_session_prompt_without_generations(client):
    """A chat with no generations has no roll-forward snapshot to recover, so
    the Compare page renders the session prompt unchanged."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id, "fresh typed text")
    await _add_key(user_id)
    _login(client, email, "test-pw-12345")

    page = _compare_page(client, chat_id)
    assert _textarea_content(page, "cmp-prompt") == "fresh typed text"
    assert _roll_forward_data(page) is None


@dbtest
async def test_compare_page_renders_baseline_after_cancel_with_partial(client, monkeypatch):
    """A run cancelled mid-completion rolls the PARTIAL completion into the
    session prompt (and writes a cancelled snapshot with the exact boundary);
    the Compare page still renders the prompt baseline (issue #11, story 13)."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id, "seed")
    await _add_key(user_id)
    _login(client, email, "test-pw-12345")

    release = asyncio.Event()

    async def _partial_then_hang(client_, url, api_key, body, timeout, **kwargs):
        yield {
            "kind": "chunk",
            "data": b'data: {"choices":[{"text":"par"}]}\n\n',
            "usage": None,
            "status": 200,
        }
        await release.wait()
        yield {
            "kind": "chunk",
            "data": b"data: [DONE]\n\n",
            "usage": None,
            "status": None,
        }

    monkeypatch.setattr(proxymod, "stream_post_with_status", _partial_then_hang)
    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "Once upon a time", "max_tokens": 8},
    )
    assert r.status_code == 200, r.text
    gen_id = uuid.UUID(r.json()["generation_id"])
    await asyncio.sleep(0.1)  # let the task absorb the first chunk

    cr = client.post(f"/workbench/{chat_id}/generations/{gen_id}/cancel")
    assert cr.status_code == 204
    release.set()
    await _wait_generation_terminal(gen_id)
    await _wait_snapshots(chat_id, n=1)

    # The roll-forward landed with the partial text.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.prompt_text == "Once upon a timepar"
    finally:
        await engine.dispose()

    page = _compare_page(client, chat_id)
    assert _textarea_content(page, "cmp-prompt") == "Once upon a time"
    assert _roll_forward_data(page) == {
        "baseline": "Once upon a time",
        "completion": "par",
    }


@dbtest
async def test_compare_page_renders_restored_baseline_not_latest(client, monkeypatch):
    """After restoring an older snapshot, the restored baseline renders as-is —
    NOT the latest generation's baseline. A restore point must never be
    discarded by a switch to Compare (issue #11, story 16); this also pins the
    design choice of the equality check over latest-generation anchoring."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id, "seed")
    await _add_key(user_id)
    _login(client, email, "test-pw-12345")

    # Two completed runs: snapshot 2 (first promptone, two) is the latest.
    await _run_single_completion(
        client, monkeypatch, chat_id, prompt="first prompt", completion="one"
    )
    await _run_single_completion(
        client, monkeypatch, chat_id, prompt="first promptone", completion="two"
    )

    # Restore snapshot 1: the session prompt becomes its baseline.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            snaps = list(
                (
                    await s.execute(
                        select(ChatSnapshot)
                        .where(ChatSnapshot.session_id == chat_id)
                        .order_by(ChatSnapshot.ts.asc())
                    )
                )
                .scalars()
                .all()
            )
            assert len(snaps) == 2
            restored_id = snaps[0].id
            assert snaps[0].prompt_before == "first prompt"
    finally:
        await engine.dispose()
    r = client.post(f"/workbench/{chat_id}/revert/{restored_id}", follow_redirects=False)
    assert r.status_code == 303

    page = _compare_page(client, chat_id)
    # Restored baseline as-is — not the latest snapshot's baseline ("first
    # promptone") and not its roll-forward.
    assert _textarea_content(page, "cmp-prompt") == "first prompt"
    assert _roll_forward_data(page) is None


@dbtest
async def test_compare_page_renders_session_prompt_after_edit(client, monkeypatch):
    """Any state that isn't exactly an unmodified roll-forward is intentional
    (issue #11, story 18): a post-run prompt edit fails the equality check, so
    the Compare page renders the edited text verbatim."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id, "seed")
    await _add_key(user_id)
    _login(client, email, "test-pw-12345")

    await _run_single_completion(
        client, monkeypatch, chat_id, prompt="Once upon a time", completion="hi"
    )

    # Simulate a post-run edit the server saw (e.g. a compare write-back).
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            chat.prompt_text = "Once upon a timehi, edited"
    finally:
        await engine.dispose()

    page = _compare_page(client, chat_id)
    assert _textarea_content(page, "cmp-prompt") == "Once upon a timehi, edited"
    assert _roll_forward_data(page) is None
