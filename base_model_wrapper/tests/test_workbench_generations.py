"""Tab-independent workbench streaming: ChatGeneration lifecycle + SSE replay.

Mirrors ``tests/test_chat_history.py`` for the DB fixtures and the
``TEST_DATABASE_URL``-gating, plus a couple of pure-Python unit tests for the
SSE frame helpers and the in-memory ``GenerationState`` fan-out.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper import proxy as proxymod
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import (
    ApiKey,
    ApiRequest,
    ChatGeneration,
    ChatSession,
    ChatSnapshot,
    CompareSnapshot,
    UsageMonthly,
    User,
)
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)

# Generous hang-guard for the SSE-stream ``wait_for`` calls below. These bounds
# only exist so a genuinely stuck generator fails fast instead of hanging the
# suite — the awaits they wrap complete in milliseconds. A tight 1.0s tripped
# spuriously on cold CI runners (ACS-349), so keep it well above any plausible
# scheduling stall while still bounded. Env-overridable for very slow hosts.
_TEST_WAIT_S = float(os.environ.get("TEST_STREAM_WAIT_S", "10.0"))


# --- unit tests (no DB) ------------------------------------------------------


def test_sse_status_frame_shape():
    from wrapper.main import _sse_status_frame
    frame = _sse_status_frame({"phase": "sending", "attempt": 0})
    assert frame.startswith(b"event: status\ndata: ")
    assert frame.endswith(b"\n\n")
    assert b'"phase": "sending"' in frame


def test_sse_replay_frame_shape():
    from wrapper.main import _sse_replay_frame
    frame = _sse_replay_frame({"text": "abc", "cursor": 3})
    assert frame.startswith(b"event: replay\ndata: ")
    assert frame.endswith(b"\n\n")


def test_sse_done_frame_shape():
    from wrapper.main import _sse_done_frame
    frame = _sse_done_frame({"status": "completed", "usage": {}, "error_message": None})
    assert frame.startswith(b"event: done\ndata: ")
    assert b'"status": "completed"' in frame


def test_absorb_chunk_text_extracts_delta_and_done():
    from wrapper.main import _absorb_chunk_text
    chunk = (
        b'data: {"choices":[{"text":"hel"}]}\n\n'
        b'data: {"choices":[{"text":"lo"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    delta, saw_done = _absorb_chunk_text(chunk)
    assert delta == "hello"
    assert saw_done is True


def test_absorb_chunk_text_ignores_keepalive_and_junk():
    from wrapper.main import _absorb_chunk_text
    delta, saw_done = _absorb_chunk_text(b": keep-alive\n\n")
    assert delta == ""
    assert saw_done is False


def test_absorb_chunk_logprobs_flattens_normalised_entries():
    """The single-pane logprobs accumulator (ACS-189) parses a raw SSE chunk's
    ``choices[0].logprobs`` into the shared normalised heatmap shape."""
    from wrapper.workbench_generations import _absorb_chunk_logprobs

    chunk = (
        b'data: {"choices":[{"text":"he","logprobs":'
        b'{"tokens":["he"],"token_logprobs":[-0.1],'
        b'"top_logprobs":[{"he":-0.1," hi":-1.2}]}}]}\n\n'
        b'data: {"choices":[{"text":"llo","logprobs":'
        b'{"tokens":["llo"],"token_logprobs":[-2.3],"top_logprobs":[{"llo":-2.3}]}}]}\n\n'
    )
    out = _absorb_chunk_logprobs(chunk)
    assert [e["token"] for e in out] == ["he", "llo"]
    assert out[0]["logprob"] == pytest.approx(-0.1)
    # top-k sorted highest-probability-first (logprob descending).
    assert out[0]["top"][0]["token"] == "he"


def test_absorb_chunk_logprobs_empty_when_no_logprobs():
    """A plain (logprobs-off) chunk yields no entries — the snapshot then stores
    NULL and the UI shows plain text."""
    from wrapper.workbench_generations import _absorb_chunk_logprobs

    assert _absorb_chunk_logprobs(b'data: {"choices":[{"text":"hi"}]}\n\n') == []
    assert _absorb_chunk_logprobs(b": keep-alive\n\n") == []


def test_normalise_logprobs_tolerates_none_logprob():
    """A ``None`` in token_logprobs / top_logprobs must not TypeError the sort.

    ACS-165 flagged malformed upstream logprobs data. vLLM can emit a ``null``
    logprob (e.g. the very first token of a completion), and a top_logprobs map
    can carry a ``None`` value; the descending sort must not blow up on either.
    """
    from wrapper.workbench_generations import _normalise_logprobs

    out = _normalise_logprobs(
        {
            "tokens": ["a", "b"],
            "token_logprobs": [None, -1.2],
            "top_logprobs": [{"a": None, " x": -0.5}, {"b": -1.2}],
        }
    )
    assert len(out) == 2
    assert out[0]["token"] == "a"
    assert out[0]["logprob"] is None
    # None sorts to the bottom (treated as -inf), the real logprob first.
    assert out[0]["top"][0]["token"] == " x"
    # Non-dict / empty input degrades to [] rather than raising.
    assert _normalise_logprobs(None) == []
    assert _normalise_logprobs("garbage") == []


async def test_generation_state_subscribe_and_broadcast():
    from wrapper.main import GenerationState
    s = GenerationState()
    q1 = s.subscribe()
    q2 = s.subscribe()
    s.broadcast(b"hello")
    assert q1.get_nowait() == b"hello"
    assert q2.get_nowait() == b"hello"
    s.unsubscribe(q1)
    s.broadcast(b"world")
    assert q2.get_nowait() == b"world"
    assert q1.qsize() == 0


async def test_generation_state_mark_done_sets_event():
    from wrapper.main import GenerationState
    s = GenerationState()
    assert not s.done_event.is_set()
    s.mark_done("completed")
    assert s.status == "completed"
    assert s.done_event.is_set()


async def test_gen_live_emits_keepalive_during_silence(monkeypatch):
    """Subscriber stream must keep bytes flowing while the producer is silent.

    Regression for "Network error: Error in input stream" — a corporate
    middlebox RSTed the idle TCP connection during the cold-boot header-wait
    because nothing was yielded to the browser between the initial replay
    frame and the first upstream chunk.
    """
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "HEARTBEAT_INTERVAL_S", 0.05)
    state = mainmod.GenerationState()

    gen = mainmod._gen_live(state, None)
    try:
        replay = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert replay.startswith(b"event: replay")

        # Queue stays silent — within a couple of intervals we must see a
        # comment-line keepalive, not just timeouts ignored.
        heartbeat = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert heartbeat == b": keepalive\n\n"

        # Real broadcast frames reset the keepalive timer and are forwarded
        # verbatim.
        state.broadcast(b"event: status\ndata: {\"phase\":\"sending\"}\n\n")
        forwarded = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert forwarded.startswith(b"event: status")
    finally:
        await gen.aclose()


async def test_gen_live_stops_after_done_frame():
    """Done frame ends the stream — no extra keepalive after terminal status."""
    from wrapper import main as mainmod

    state = mainmod.GenerationState()
    gen = mainmod._gen_live(state, None)
    try:
        _ = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)  # replay
        state.broadcast(b"event: done\ndata: {\"status\":\"completed\"}\n\n")
        done = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert done.startswith(b"event: done")
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
    finally:
        await gen.aclose()


async def test_gen_live_replays_current_status_on_subscribe():
    """A tab joining mid cold-boot must get the current phase immediately.

    Without this, the resumed banner stays invisible until the next broadcast
    status frame — up to ~a minute between Modal 303 backoff windows.
    """
    from wrapper import main as mainmod

    state = mainmod.GenerationState()
    state.broadcast_status(b"event: status\ndata: {\"phase\":\"cold_boot\"}\n\n")
    gen = mainmod._gen_live(state, None)
    try:
        replay = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert replay.startswith(b"event: replay")
        # Second frame, before any further broadcast, is the remembered status.
        status = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert status.startswith(b"event: status")
        assert b"cold_boot" in status
    finally:
        await gen.aclose()


async def test_gen_live_emits_done_when_subscribing_to_finished_state():
    """Late subscribers to a terminal state must still see a `done` frame.

    Without it, the browser's EventSource sees a clean connection close with
    no terminal event and auto-reconnects every ~3 s, producing the reconnect
    storms visible in Railway HTTP logs (many GETs with upstreamDuration=1ms)
    and the put_data_out 504 cleanup noise in Modal logs at shutdown.
    """
    from wrapper import main as mainmod

    state = mainmod.GenerationState()
    state.completion_text = "hello world"
    state.usage = {"prompt_tokens": 4, "completion_tokens": 2}
    state.mark_done("completed")

    gen = mainmod._gen_live(state, None)
    try:
        replay = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert replay.startswith(b"event: replay")
        done = await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
        assert done.startswith(b"event: done")
        assert b'"status": "completed"' in done
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(gen.__anext__(), timeout=_TEST_WAIT_S)
    finally:
        await gen.aclose()


async def test_generation_state_broadcast_status_remembers_last_frame():
    from wrapper.main import GenerationState
    s = GenerationState()
    assert s.last_status_frame is None
    s.broadcast_status(b"event: status\ndata: {\"phase\":\"sending\"}\n\n")
    s.broadcast_status(b"event: status\ndata: {\"phase\":\"cold_boot\"}\n\n")
    # last_status_frame tracks the most recent status (so a late subscriber
    # gets the current phase, not a stale "sending"). Plain broadcast() does
    # not touch it.
    assert s.last_status_frame == b"event: status\ndata: {\"phase\":\"cold_boot\"}\n\n"
    s.broadcast(b"data: {\"choices\":[]}\n\n")
    assert b"cold_boot" in s.last_status_frame


def test_parse_upstream_error_message_picks_openai_shape():
    from wrapper.main import _parse_upstream_error_message
    msg = _parse_upstream_error_message(
        b'{"error":{"message":"vLLM unhappy","type":"x"}}', 500
    )
    assert msg == "vLLM unhappy"


def test_parse_upstream_error_message_falls_back_to_status():
    from wrapper.main import _parse_upstream_error_message
    msg = _parse_upstream_error_message(b"not-json", 503)
    assert "503" in msg



# --- SSE error envelope threads upstream_kind / status -----------------------


def test_friendly_upstream_error_for_network_layer_zero():
    """status==0 (network-layer failure surfaced by stream_post_with_status)
    must not render as ``(HTTP 0)`` to the user."""
    from wrapper.main import _friendly_upstream_error
    msg = _friendly_upstream_error(0)
    assert "0" not in msg
    assert "upstream" in msg.lower() or "reach" in msg.lower()


def test_friendly_upstream_error_keeps_real_status_in_message():
    from wrapper.main import _friendly_upstream_error
    msg = _friendly_upstream_error(503)
    assert "503" in msg


def test_sse_done_frame_carries_code_for_late_subscribers():
    """The done frame must also carry ``code`` (= upstream_kind) so a
    subscriber that attaches after the real-time error frame fired still
    learns the structured failure mode. Late subscribers only see
    ``replay`` + ``done`` — losing ``code`` here is the regression we just
    fixed for the live-error path applied to the late-subscriber path."""
    from wrapper.main import _sse_done_frame
    frame = _sse_done_frame(
        {
            "status": "failed",
            "usage": {},
            "error_message": "boom",
            "code": "vllm_oom",
        }
    )
    assert frame.startswith(b"event: done\ndata: ")
    assert b'"code": "vllm_oom"' in frame
    assert b'"status": "failed"' in frame


async def test_generation_state_mark_done_records_error_kind():
    """``mark_done`` must persist the error_kind so late subscribers reading
    the live state can include it in the synthesised done frame."""
    from wrapper.main import GenerationState
    s = GenerationState()
    assert s.error_kind is None
    s.mark_done("failed", "boom", error_kind="upstream_unreachable")
    assert s.status == "failed"
    assert s.error == "boom"
    assert s.error_kind == "upstream_unreachable"


def test_sse_error_frame_carries_code_and_status():
    """The SSE error envelope must include ``status`` + ``code`` (= upstream_kind)
    + ``message``. The UI keys off ``code`` to render a kind-aware message;
    the structured fields must not lie about the real failure mode."""
    from wrapper.main import _sse_error_frame
    frame = _sse_error_frame(
        {"status": 502, "code": "upstream_unreachable", "message": "boom"}
    )
    assert frame.startswith(b"event: error\ndata: ")
    assert frame.endswith(b"\n\n")
    assert b'"status": 502' in frame
    assert b'"code": "upstream_unreachable"' in frame
    assert b'"message": "boom"' in frame


# --- DB-gated integration tests ----------------------------------------------


@pytest.fixture
def client():
    """TestClient with lifespan started + the same envvars as test_chat_history."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-workbench-gens-secret")
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
        email = f"workbench-gens-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password))
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _make_chat_and_key(
    user_id: uuid.UUID,
) -> tuple[uuid.UUID, uuid.UUID]:
    """Returns (chat_id, key_id)."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = ChatSession(user_id=user_id, title="gen-fixture", prompt_text="seed")
            s.add(chat)
            gk = generate_key()
            key = ApiKey(
                user_id=user_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                name="gen-key",
                monthly_token_budget=1_000_000,
            )
            s.add(key)
            await s.flush()
            return chat.id, key.id
    finally:
        await engine.dispose()


async def _add_key(user_id: uuid.UUID, name: str) -> uuid.UUID:
    """Add a second active completions-scoped key for the user; return its id."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gk = generate_key()
            key = ApiKey(
                user_id=user_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                name=name,
                monthly_token_budget=1_000_000,
            )
            s.add(key)
            await s.flush()
            return key.id
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": password}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


class _FakeStreamResp:
    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code = status_code
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


@dbtest
async def test_start_generation_creates_row_and_returns_id(client, monkeypatch):
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"hi"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "Once upon a time", "max_tokens": 8, "temperature": 0.5},
    )
    assert r.status_code == 200, r.text
    gen_id = uuid.UUID(r.json()["generation_id"])

    # Wait for the background task to finish.
    for _ in range(50):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                row = (
                    await s.execute(
                        select(ChatGeneration).where(ChatGeneration.id == gen_id)
                    )
                ).scalar_one()
                if row.status != "running":
                    break
        finally:
            await engine.dispose()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (
                await s.execute(
                    select(ChatGeneration).where(ChatGeneration.id == gen_id)
                )
            ).scalar_one()
            assert row.status == "completed"
            assert row.completion_text == "hi"
            assert row.n_completion_tokens == 1
            # Snapshot mirror was created.
            snaps = list(
                (
                    await s.execute(
                        select(ChatSnapshot).where(ChatSnapshot.session_id == chat_id)
                    )
                ).scalars().all()
            )
            assert len(snaps) == 1
            assert snaps[0].completion_text == "hi"
            # Chat prompt_text rolled forward.
            chat = (
                await s.execute(
                    select(ChatSession).where(ChatSession.id == chat_id)
                )
            ).scalar_one()
            assert chat.prompt_text == "Once upon a timehi"
    finally:
        await engine.dispose()


@dbtest
async def test_start_generation_forwards_sampling_params(client, monkeypatch):
    """The single-pane composer's sampling panel reaches the upstream body.

    ACS-165: ``chat_generation_start`` now shapes the upstream ``/v1/completions``
    body through the shared ``_build_lane_body`` instead of an inline dict, so the
    new ``top_p`` / ``min_p`` / ``seed`` / ``logprobs`` Form params must land on the
    wire. We capture the ``body`` kwarg passed to ``stream_post_with_status`` and
    assert the posted sampling params come through.
    """
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    captured: dict[str, object] = {}

    async def _fake(client, upstream_url, api_key, body, timeout_s, **kwargs):
        captured["body"] = body
        yield {
            "kind": "chunk",
            "data": b'data: {"choices":[{"text":"x"}]}\n\n',
            "usage": None,
            "status": 200,
        }
        yield {"kind": "chunk", "data": b"data: [DONE]\n\n", "usage": None, "status": None}

    monkeypatch.setattr(proxymod, "stream_post_with_status", _fake)

    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0.5,
            "top_p": 0.5,
            "min_p": 0.1,
            "seed": 1234,
            "logprobs": 5,
        },
    )
    assert r.status_code == 200, r.text

    # Wait for the background task to invoke the (patched) upstream call.
    for _ in range(50):
        await asyncio.sleep(0.05)
        if "body" in captured:
            break
    assert "body" in captured, "upstream stream was never called"
    body = captured["body"]
    assert body["top_p"] == 0.5
    assert body["min_p"] == 0.1
    assert body["seed"] == 1234
    assert body["logprobs"] == 5
    # Untouched knobs stay at the documented API defaults.
    assert body["top_k"] == -1
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 8


@dbtest
async def test_generation_uses_and_persists_selected_key(client, monkeypatch):
    """A per-chat key choice is charged for usage AND saved on the chat.

    The chat starts with the user's newest key as the implicit default; we POST
    a generation explicitly picking the *other* key and assert (a) usage lands
    on the chosen key, not the default, and (b) ``chat.api_key_id`` now points
    at the chosen key so reopening the workbench pre-selects it.
    """
    user_id, email = await _make_user()
    # _make_chat_and_key creates the first (newest-at-the-time) key; then we add
    # a second, which becomes the newest / implicit default. We deliberately
    # pick the *older* one to prove the form value overrides the default.
    chat_id, older_key_id = await _make_chat_and_key(user_id)
    newer_key_id = await _add_key(user_id, "newer-key")
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"hi"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "temperature": 0.5,
            "api_key_id": str(older_key_id),
        },
    )
    assert r.status_code == 200, r.text
    gen_id = uuid.UUID(r.json()["generation_id"])

    for _ in range(50):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                row = (
                    await s.execute(
                        select(ChatGeneration).where(ChatGeneration.id == gen_id)
                    )
                ).scalar_one()
                if row.status != "running":
                    break
        finally:
            await engine.dispose()

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            # (b) The choice is persisted on the chat (older key, not the default).
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.api_key_id == older_key_id

            # (a) Usage was charged to the chosen key, and none to the default.
            chosen_usage = (
                await s.execute(
                    select(UsageMonthly).where(UsageMonthly.key_id == older_key_id)
                )
            ).scalar_one_or_none()
            assert chosen_usage is not None
            assert chosen_usage.tokens_completion == 1
            default_usage = (
                await s.execute(
                    select(UsageMonthly).where(UsageMonthly.key_id == newer_key_id)
                )
            ).scalar_one_or_none()
            assert default_usage is None

            # (c) The api_requests row is tagged interactive + streaming, so the
            # usage dashboard's workload mix attributes workbench traffic
            # correctly instead of "undeclared".
            api_req = (
                await s.execute(
                    select(ApiRequest).where(ApiRequest.key_id == older_key_id)
                )
            ).scalar_one()
            assert api_req.endpoint == "/workbench"
            assert api_req.workload_type == "interactive"
            assert api_req.stream is True
    finally:
        await engine.dispose()


@dbtest
async def test_events_replay_then_done(client, monkeypatch):
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"ab"}]}\n\n',
            b'data: {"choices":[{"text":"cd"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "p", "max_tokens": 10},
    )
    assert r.status_code == 200
    gen_id = r.json()["generation_id"]

    # Drain the event stream synchronously (TestClient supports SSE via stream).
    with client.stream("GET", f"/workbench/{chat_id}/generations/{gen_id}/events") as resp:
        assert resp.status_code == 200
        body = b"".join(chunk for chunk in resp.iter_raw())
    assert b"event: replay" in body
    assert b"event: done" in body
    # Some "ab" or "cd" delta should have made it through, either via replay or chunk.
    assert b"ab" in body or b"cd" in body


@dbtest
async def test_start_generation_cancels_existing_running(client, monkeypatch):
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # First generation: hold it open with a slow upstream so it stays running
    # long enough for the second POST to cancel it.
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow(client_, url, api_key, body, timeout, **kwargs):
        started.set()
        await release.wait()
        yield {
            "kind": "chunk",
            "data": b'data: {"choices":[{"text":"x"}]}\n\n',
            "usage": None,
            "status": 200,
        }
        yield {
            "kind": "chunk",
            "data": b"data: [DONE]\n\n",
            "usage": None,
            "status": None,
        }

    monkeypatch.setattr(proxymod, "stream_post_with_status", _slow)
    r1 = client.post(
        f"/workbench/{chat_id}/generations", data={"prompt": "p1", "max_tokens": 4}
    )
    assert r1.status_code == 200
    gen1 = uuid.UUID(r1.json()["generation_id"])

    # Give the task time to start.
    for _ in range(50):
        if started.is_set():
            break
        await asyncio.sleep(0.02)

    # Second POST — should cancel gen1 immediately.
    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"y"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
    )
    r2 = client.post(
        f"/workbench/{chat_id}/generations", data={"prompt": "p2", "max_tokens": 4}
    )
    assert r2.status_code == 200
    gen2 = uuid.UUID(r2.json()["generation_id"])
    assert gen1 != gen2

    release.set()
    # Wait for both tasks to settle.
    for _ in range(100):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                rows = {
                    r.id: r.status
                    for r in (
                        await s.execute(
                            select(ChatGeneration).where(
                                ChatGeneration.session_id == chat_id
                            )
                        )
                    ).scalars().all()
                }
        finally:
            await engine.dispose()
        if rows.get(gen1) != "running" and rows.get(gen2) != "running":
            break

    assert rows[gen1] == "cancelled"
    assert rows[gen2] in ("completed", "cancelled")  # cancelled is acceptable timing


@dbtest
async def test_cancel_endpoint_flips_running_row(client, monkeypatch):
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    release = asyncio.Event()

    async def _hang(client_, url, api_key, body, timeout, **kwargs):
        await release.wait()
        yield {
            "kind": "chunk",
            "data": b"data: [DONE]\n\n",
            "usage": None,
            "status": 200,
        }

    monkeypatch.setattr(proxymod, "stream_post_with_status", _hang)
    r = client.post(
        f"/workbench/{chat_id}/generations", data={"prompt": "p", "max_tokens": 4}
    )
    assert r.status_code == 200
    gen_id = r.json()["generation_id"]

    # Allow the task to enter the upstream iterator.
    await asyncio.sleep(0.1)

    cr = client.post(f"/workbench/{chat_id}/generations/{gen_id}/cancel")
    assert cr.status_code == 204

    # Row should already be cancelled in DB.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (
                await s.execute(
                    select(ChatGeneration).where(
                        ChatGeneration.id == uuid.UUID(gen_id)
                    )
                )
            ).scalar_one()
            assert row.status == "cancelled"
    finally:
        await engine.dispose()

    # Release the upstream so the task finishes cleanly.
    release.set()
    await asyncio.sleep(0.2)


@dbtest
async def test_events_replay_from_db_when_state_evicted(client):
    """If the in-memory state is gone, /events should serve the DB row."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Seed a terminal ChatGeneration row directly, then ensure the in-memory
    # state dict has no entry for it. That's exactly the post-eviction shape.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = ChatGeneration(
                session_id=chat_id,
                prompt_before="hello",
                completion_text=" world",
                model="default",
                max_tokens=10,
                temperature=0.7,
                status="completed",
                n_prompt_tokens=1,
                n_completion_tokens=2,
            )
            s.add(row)
            await s.flush()
            gen_id = row.id
    finally:
        await engine.dispose()

    from wrapper.main import app
    app.state.generations.pop(gen_id, None)

    with client.stream("GET", f"/workbench/{chat_id}/generations/{gen_id}/events") as resp:
        assert resp.status_code == 200
        body = b"".join(chunk for chunk in resp.iter_raw())
    assert b"event: replay" in body
    assert b" world" in body
    assert b"event: done" in body
    assert b'"status": "completed"' in body


@dbtest
async def test_wrapper_restart_marks_running_as_failed(client):
    """The lifespan startup hook flips orphaned running rows to failed."""
    user_id, _email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)

    # Prime a stuck-running row.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = ChatGeneration(
                session_id=chat_id,
                prompt_before="x",
                completion_text="part",
                model="default",
                max_tokens=5,
                temperature=0.7,
                status="running",
            )
            s.add(row)
            await s.flush()
            gen_id = row.id
    finally:
        await engine.dispose()

    # Re-enter the lifespan: the existing TestClient fixture has already
    # started one, so simulate the boot hook by opening a fresh client.
    from wrapper.main import app
    with TestClient(app) as _c2:
        pass

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            row = (
                await s.execute(
                    select(ChatGeneration).where(ChatGeneration.id == gen_id)
                )
            ).scalar_one()
            assert row.status == "failed"
            assert row.error_message and "wrapper restarted" in row.error_message
            assert row.ended_at is not None
    finally:
        await engine.dispose()


@dbtest
def test_legacy_stream_endpoint_returns_410(client):
    """The pre-refactor POST endpoint surfaces a structured 410 instead of dangling."""
    chat_id = uuid.uuid4()
    r = client.post(f"/workbench/{chat_id}/stream", data={"prompt": "x"})
    assert r.status_code == 410
    body = r.json()
    assert body["error"]["code"] == "endpoint_gone"


# --- compare mode ------------------------------------------------------------


def test_context_overflow_message_boundary(monkeypatch):
    """Pure-Python guard for the pre-flight context check (ACS-341).

    ``prompt_tokens`` is pinned to 100 via a stub counter so the boundary is
    deterministic (no real tokenizer needed). The message fires only when
    ``prompt_tokens + max_tokens > max_model_len``; the at-limit case (``==``)
    is accepted — that's the boundary the BOS-inclusive count (ACS-317)
    protects. Mirrors ``services.completions.check_sequence_length``.
    """
    from unittest.mock import MagicMock

    from wrapper.routes import workbench as wb
    from wrapper.settings import ModelEntry

    monkeypatch.setattr(
        wb, "get_token_counter", lambda *a, **k: MagicMock(count=lambda t: 100)
    )
    settings = MagicMock(hf_token=None)

    def entry(max_len):
        return ModelEntry(
            model_id="m",
            upstream_url="https://u.example",
            served_model_name="s",
            tokenizer_repo="gpt2",
            gpu_shape_label="x",
            status="live",
            max_model_len=max_len,
        )

    # 100 + 60 = 160 > 150 → rejected with a legible message.
    msg = wb._context_overflow_message(entry(150), "some prompt", 60, settings)
    assert msg is not None
    assert "context window" in msg
    assert "max_model_len=150" in msg
    # 100 + 50 == 150 → accepted (the at-limit boundary).
    assert wb._context_overflow_message(entry(150), "some prompt", 50, settings) is None
    # Comfortably under → accepted.
    assert wb._context_overflow_message(entry(150), "some prompt", 10, settings) is None
    # No max_model_len declared (legacy entry) → check is skipped.
    assert wb._context_overflow_message(entry(None), "some prompt", 9999, settings) is None
    # Empty prompt → skipped (nothing to enforce).
    assert wb._context_overflow_message(entry(10), "", 9999, settings) is None


def test_build_lane_body_clamps_and_omits_unset_params():
    """Pure-Python clamp test for the per-lane sampling body builder.

    Verifies: hard clamps (max_tokens, temperature), top_k's -1-disables
    convention, that optional knobs (seed/stop/logprobs) are only present when
    the lane set them, and that ``stream`` is forced on.
    """
    from wrapper.routes.workbench import _build_lane_body

    full = _build_lane_body(
        {
            "max_tokens": 99999,
            "temperature": 150.0,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "seed": 7,
            "stop": "END",
            "logprobs": 50,
        },
        served_model_name="served-x",
        prompt="hello",
    )
    assert full["model"] == "served-x"
    assert full["prompt"] == "hello"
    assert full["stream"] is True
    assert full["max_tokens"] == 4000  # clamped down
    assert full["temperature"] == 100.0  # clamped to [0,100]
    assert full["top_p"] == 0.9
    assert full["top_k"] == 50
    assert full["min_p"] == 0.1
    assert full["seed"] == 7
    assert full["stop"] == ["END"]
    assert full["logprobs"] == 20  # clamped to [0,20]

    bare = _build_lane_body({}, served_model_name="m", prompt="p")
    assert bare["max_tokens"] == 200  # _CHAT_DEFAULT_MAX_TOKENS
    assert bare["temperature"] == 1.0  # _CHAT_DEFAULT_TEMPERATURE (ACS-145)
    assert bare["top_k"] == -1  # unset/disabled
    # Optional knobs absent so the upstream default applies, not a pinned value.
    assert "seed" not in bare
    assert "stop" not in bare
    assert "logprobs" not in bare


@dbtest
async def test_compare_fans_out_without_snapshots_or_prompt_mutation(client, monkeypatch):
    """A compare batch spawns one generation per lane, all completing, but —
    unlike single-pane "Continue" — must NOT write snapshots and must NOT roll
    the completion into the session's prompt_text (persist_session_state=False).

    Since ACS-254 (workbench.py compare_start) the compare handler DOES persist
    the *submitted* prompt — the prompt is one shared single/compare value, so a
    compare-only session would otherwise lose it on reload. So prompt_text becomes
    the submitted prompt, never the prompt+completion roll-forward."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)  # fixture seeds prompt_text="seed"
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"hi"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={
            "prompt": "Once upon a time",
            "lanes": [{"max_tokens": 8}, {"max_tokens": 8, "temperature": 1.2}],
        },
    )
    assert r.status_code == 200, r.text
    lanes = r.json()["lanes"]
    assert len(lanes) == 2
    gen_ids = [uuid.UUID(lane["generation_id"]) for lane in lanes]
    assert lanes[0]["index"] == 0 and lanes[1]["index"] == 1

    # Wait for both background tasks to settle.
    for _ in range(100):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                rows = list(
                    (
                        await s.execute(
                            select(ChatGeneration).where(
                                ChatGeneration.session_id == chat_id
                            )
                        )
                    ).scalars().all()
                )
        finally:
            await engine.dispose()
        if rows and all(row.status != "running" for row in rows):
            break

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(ChatGeneration).where(
                            ChatGeneration.session_id == chat_id
                        )
                    )
                ).scalars().all()
            )
            assert {row.id for row in rows} == set(gen_ids)
            assert all(row.status == "completed" for row in rows)
            assert all(row.completion_text == "hi" for row in rows)

            # Invariant 1: compare lanes do NOT create snapshots.
            snaps = list(
                (
                    await s.execute(
                        select(ChatSnapshot).where(ChatSnapshot.session_id == chat_id)
                    )
                ).scalars().all()
            )
            assert snaps == []

            # Invariant 2 (ACS-254): compare persists the *submitted* prompt (the
            # shared single/compare value) but never rolls the completion into it.
            # So prompt_text is the submitted prompt — not "seed" (the pre-run
            # value) and not the single-pane roll-forward "Once upon a timehi".
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            assert chat.prompt_text == "Once upon a time"
    finally:
        await engine.dispose()


@dbtest
async def test_compare_unknown_model_lane_isolated(client, monkeypatch):
    """One lane with an unknown model fails only that lane (per-lane error
    entry); sibling lanes still launch and complete."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"],
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={
            "prompt": "p",
            "lanes": [
                {"model": "definitely-not-a-real-model"},
                {"max_tokens": 4},
            ],
        },
    )
    assert r.status_code == 200, r.text
    lanes = {lane["index"]: lane for lane in r.json()["lanes"]}
    assert "error" in lanes[0]
    assert lanes[0]["error"]["code"] == "unknown_model"
    assert "generation_id" in lanes[1]


@dbtest
async def test_generation_rejects_over_context_and_accepts_at_limit(client, monkeypatch):
    """Pre-flight context guard on the single-pane generate path (ACS-341).

    An over-long prompt is rejected locally with a clean 400
    ``context_length_exceeded`` — before any DB row is inserted and before the
    upstream stream is ever started — instead of streaming to the GPU and
    bouncing off ``vllm_context_length``. A prompt exactly at the limit
    (prompt_tokens + max_tokens == max_model_len) is accepted: the boundary the
    BOS-inclusive token count (ACS-317) protects.
    """
    import dataclasses
    from unittest.mock import MagicMock

    from wrapper.main import app
    from wrapper.routes import workbench as wb

    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Deterministic 100-token prompt regardless of tokenizer availability.
    monkeypatch.setattr(
        wb, "get_token_counter", lambda *a, **k: MagicMock(count=lambda t: 100)
    )
    # Give the default model a small context window (setitem auto-restores).
    default_id = app.state.default_model_id
    entry = app.state.models[default_id]
    monkeypatch.setitem(
        app.state.models, default_id, dataclasses.replace(entry, max_model_len=150)
    )

    # Track whether the upstream stream is ever started.
    called = {"n": 0}

    async def _fake(client_, url, api_key, body, timeout_s, **kwargs):
        called["n"] += 1
        yield {
            "kind": "chunk",
            "data": b'data: {"choices":[{"text":"x"}]}\n\n',
            "usage": None,
            "status": 200,
        }
        yield {"kind": "chunk", "data": b"data: [DONE]\n\n", "usage": None, "status": None}

    monkeypatch.setattr(proxymod, "stream_post_with_status", _fake)

    # 100 (prompt) + 60 (max_tokens) = 160 > 150 → clean local 400, no dispatch.
    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "over the limit", "max_tokens": 60},
    )
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "context_length_exceeded"

    # No ChatGeneration row inserted, and the upstream stream was never started.
    await asyncio.sleep(0.1)
    assert called["n"] == 0
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(ChatGeneration).where(
                            ChatGeneration.session_id == chat_id
                        )
                    )
                ).scalars().all()
            )
            assert rows == []
    finally:
        await engine.dispose()

    # 100 + 50 == 150 → accepted (the at-limit boundary), stream starts once.
    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "over the limit", "max_tokens": 50},
    )
    assert r.status_code == 200, r.text
    for _ in range(50):
        await asyncio.sleep(0.05)
        if called["n"] >= 1:
            break
    assert called["n"] == 1


@dbtest
async def test_compare_lane_over_context_isolated(client, monkeypatch):
    """One compare lane over the context window fails only that lane (ACS-341).

    The over-limit lane gets a per-lane ``context_length_exceeded`` error and is
    never dispatched; the sibling lane that fits still launches — mirroring the
    unknown-model per-lane isolation.
    """
    import dataclasses
    from unittest.mock import MagicMock

    from wrapper.main import app
    from wrapper.routes import workbench as wb

    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    monkeypatch.setattr(
        wb, "get_token_counter", lambda *a, **k: MagicMock(count=lambda t: 100)
    )
    default_id = app.state.default_model_id
    entry = app.state.models[default_id]
    monkeypatch.setitem(
        app.state.models, default_id, dataclasses.replace(entry, max_model_len=150)
    )

    _patch_upstream_chunks(
        monkeypatch,
        [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"],
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={"prompt": "p", "lanes": [{"max_tokens": 60}, {"max_tokens": 40}]},
    )
    assert r.status_code == 200, r.text
    lanes = {lane["index"]: lane for lane in r.json()["lanes"]}
    # Lane 0: 100 + 60 = 160 > 150 → clean per-lane error, not dispatched.
    assert "error" in lanes[0]
    assert lanes[0]["error"]["code"] == "context_length_exceeded"
    # Lane 1: 100 + 40 = 140 <= 150 → launches.
    assert "generation_id" in lanes[1]

    # The over-limit lane inserted NO ChatGeneration row: exactly one row exists
    # (lane 1's), and it is the one whose id was returned. Proves the rejected
    # lane never reached the row insert / dispatch, not just that the response
    # shape looked right.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(ChatGeneration).where(
                            ChatGeneration.session_id == chat_id
                        )
                    )
                ).scalars().all()
            )
            assert len(rows) == 1
            assert str(rows[0].id) == lanes[1]["generation_id"]
    finally:
        await engine.dispose()


# --- Integration tests for SSE error-envelope threading ----------------------
#
# These exercise the full ``_run_generation_task`` path with a patched upstream
# and a *pre-subscribed* ``GenerationState`` so we deterministically observe
# the SSE ``error`` frame the workbench JS would receive in production (no
# race between the background task firing the error and the subscriber
# attaching to the queue). Each test also asserts the resulting
# ``api_requests.error_kind`` column so the dashboard view of the same
# failure stays in sync with the SSE envelope.


async def _start_generation_and_subscribe(client, monkeypatch, *, fake_stream):
    """Helper: log in, POST a generation against ``fake_stream``, subscribe to
    the in-memory state, return (gen_uuid, drain_task, captured)."""
    from wrapper.main import app

    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    monkeypatch.setattr(proxymod, "stream_post_with_status", fake_stream)

    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "Hi", "max_tokens": 4},
    )
    assert r.status_code == 200, r.text
    gen_uuid = uuid.UUID(r.json()["generation_id"])

    # Wait for the background task to register a GenerationState we can
    # subscribe to.
    for _ in range(100):
        if gen_uuid in app.state.generations:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("GenerationState never appeared in app.state")

    state = app.state.generations[gen_uuid]
    q = state.subscribe()
    captured: list[bytes] = []

    async def _drain() -> None:
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=3.0)
                except TimeoutError:
                    if state.done_event.is_set():
                        return
                    continue
                captured.append(frame)
                if frame.startswith(b"event: done") or state.done_event.is_set():
                    return
        finally:
            state.unsubscribe(q)

    return gen_uuid, asyncio.create_task(_drain()), captured


async def _wait_for_terminal_row(gen_uuid):
    """Poll the ChatGeneration row + the matching ApiRequest row."""
    from wrapper.models import ApiRequest

    for _ in range(100):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                row = (
                    await s.execute(
                        select(ChatGeneration).where(ChatGeneration.id == gen_uuid)
                    )
                ).scalar_one()
                if row.status == "running":
                    continue
                api_row = (
                    await s.execute(
                        select(ApiRequest)
                        .where(ApiRequest.endpoint == "/workbench")
                        .order_by(ApiRequest.ts.desc())
                    )
                ).scalars().first()
                return row, api_row
        finally:
            await engine.dispose()
    raise AssertionError(f"row never left running state: {gen_uuid}")


def _make_gated_error_upstream(*, status, upstream_kind, body, release):
    """Build a patched ``stream_post_with_status`` that waits on ``release``
    before yielding the error event — so the test can attach a subscriber
    first."""
    async def _fake(client, upstream_url, api_key, body_, timeout_s, **kwargs):
        await release.wait()
        ev = {"kind": "error", "data": body, "status": status}
        if upstream_kind is not None:
            ev["upstream_kind"] = upstream_kind
        yield ev
    return _fake


@dbtest
async def test_upstream_unreachable_threads_through_sse_and_api_requests(
    client, monkeypatch
):
    """status=0 + upstream_kind=upstream_unreachable from the proxy must surface
    as ``code=upstream_unreachable`` in the SSE error frame, with the audit
    log status translated to 502 (matching the non-streaming path)."""
    release = asyncio.Event()
    fake = _make_gated_error_upstream(
        status=0,
        upstream_kind="upstream_unreachable",
        body=b"ConnectError: DNS failure",
        release=release,
    )
    gen_uuid, drain_task, captured = await _start_generation_and_subscribe(
        client, monkeypatch, fake_stream=fake
    )
    release.set()
    await asyncio.wait_for(drain_task, timeout=5.0)

    error_frames = [f for f in captured if f.startswith(b"event: error")]
    assert error_frames, f"no error frame in captured broadcasts: {captured!r}"
    body = error_frames[0]
    # SSE error frame carries the real code; status 0 is translated to 502
    # so the UI does not display "(HTTP 0)".
    assert b'"code": "upstream_unreachable"' in body
    assert b'"status": 502' in body
    assert b'"status": 0' not in body

    row, api_row = await _wait_for_terminal_row(gen_uuid)
    assert row.status == "failed"
    # ApiRequest row preserves the kind, not the generic "upstream_error".
    assert api_row is not None, "no ApiRequest row written for workbench failure"
    assert api_row.error_kind == "upstream_unreachable"
    assert api_row.status == 502


@dbtest
async def test_vllm_oom_threads_through_sse_and_api_requests(client, monkeypatch):
    """A 500 with upstream_kind=vllm_oom must surface as ``code=vllm_oom`` in
    the SSE frame and as ``error_kind=vllm_oom`` in the api_requests row."""
    release = asyncio.Event()
    fake = _make_gated_error_upstream(
        status=500,
        upstream_kind="vllm_oom",
        body=b'{"error":{"message":"CUDA out of memory","type":"engine"}}',
        release=release,
    )
    gen_uuid, drain_task, captured = await _start_generation_and_subscribe(
        client, monkeypatch, fake_stream=fake
    )
    release.set()
    await asyncio.wait_for(drain_task, timeout=5.0)

    error_frames = [f for f in captured if f.startswith(b"event: error")]
    assert error_frames, f"no error frame in captured broadcasts: {captured!r}"
    body = error_frames[0]
    assert b'"code": "vllm_oom"' in body
    assert b'"status": 500' in body
    # Upstream message preserved verbatim so the UI can show what vLLM said.
    assert b"CUDA out of memory" in body

    # The done frame produced for late subscribers must also carry code so
    # the workbench JS late-subscribe path (page reload after error fired)
    # can render the kind-aware prefix.
    done_frames = [f for f in captured if f.startswith(b"event: done")]
    assert done_frames, f"no done frame in captured broadcasts: {captured!r}"
    assert b'"code": "vllm_oom"' in done_frames[0]

    row, api_row = await _wait_for_terminal_row(gen_uuid)
    assert row.status == "failed"
    assert api_row is not None
    assert api_row.error_kind == "vllm_oom"
    assert api_row.status == 500


@dbtest
async def test_upstream_4xx_without_kind_falls_back_to_generic_label(
    client, monkeypatch
):
    """When the proxy doesn't classify a body (no upstream_kind), the SSE frame
    still gets a stable fallback code (``upstream_4xx``/``upstream_5xx``) so
    the UI is never left with a kind-less error envelope."""
    release = asyncio.Event()
    fake = _make_gated_error_upstream(
        status=404,
        upstream_kind=None,
        body=b'{"error":{"message":"model not found"}}',
        release=release,
    )
    gen_uuid, drain_task, captured = await _start_generation_and_subscribe(
        client, monkeypatch, fake_stream=fake
    )
    release.set()
    await asyncio.wait_for(drain_task, timeout=5.0)

    error_frames = [f for f in captured if f.startswith(b"event: error")]
    assert error_frames, f"no error frame in captured broadcasts: {captured!r}"
    body = error_frames[0]
    assert b'"code": "upstream_4xx"' in body
    assert b'"status": 404' in body

    row, api_row = await _wait_for_terminal_row(gen_uuid)
    assert row.status == "failed"
    assert api_row is not None
    assert api_row.error_kind == "upstream_4xx"
    assert api_row.status == 404


@dbtest
async def test_internal_exception_threads_internal_error_code(client, monkeypatch):
    """An unexpected exception while iterating the upstream stream must surface
    as ``code=internal_error`` in the SSE frame (the catch-all in
    ``_run_generation_task`` previously dropped any kind signal and reported
    a generic HTTP 500)."""
    release = asyncio.Event()

    async def _boom(client_, url, api_key, body, timeout, **kwargs):
        await release.wait()
        raise RuntimeError("boom: synthetic test failure")
        yield  # pragma: no cover — makes this an async generator

    gen_uuid, drain_task, captured = await _start_generation_and_subscribe(
        client, monkeypatch, fake_stream=_boom
    )
    release.set()
    await asyncio.wait_for(drain_task, timeout=5.0)

    error_frames = [f for f in captured if f.startswith(b"event: error")]
    assert error_frames, f"no error frame in captured broadcasts: {captured!r}"
    body = error_frames[0]
    assert b'"code": "internal_error"' in body
    assert b'"status": 500' in body

    row, api_row = await _wait_for_terminal_row(gen_uuid)
    assert row.status == "failed"
    assert api_row is not None
    assert api_row.error_kind == "internal_error"
    assert api_row.status == 500


# --- compare-mode saved snapshots (ACS-180) ---------------------------------


@dbtest
async def test_compare_snapshot_create_list_and_scope(client):
    """POST /compare/snapshots persists one combined row; GET lists it newest
    first; the shared prompt + per-lane config/completion round-trip; and both
    endpoints are ownership-scoped (another user gets 404, no IDOR)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Empty list initially.
    r = client.get(f"/workbench/{chat_id}/compare/snapshots")
    assert r.status_code == 200, r.text
    assert r.json()["snapshots"] == []

    # Write one snapshot (mirrors the client's post after Run-all completes).
    payload = {
        "prompt": "Once upon a time",
        "lanes": [
            {
                "model": "gpt2",
                "max_tokens": 8,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "seed": None,
                "stop": None,
                "completion_text": " there was a lane",
                "cancelled": False,
            },
            {
                "model": "gpt2",
                "max_tokens": 8,
                "temperature": 1.2,
                "completion_text": " partial",
                "cancelled": True,
            },
        ],
    }
    r = client.post(f"/workbench/{chat_id}/compare/snapshots", json=payload)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["prompt"] == "Once upon a time"
    assert created["n_lanes"] == 2
    assert created["lanes"][0]["model"] == "gpt2"
    assert created["lanes"][0]["completion_text"] == " there was a lane"
    assert created["lanes"][1]["cancelled"] is True

    # It appears in the list.
    r = client.get(f"/workbench/{chat_id}/compare/snapshots")
    assert r.status_code == 200, r.text
    snaps = r.json()["snapshots"]
    assert len(snaps) == 1
    assert snaps[0]["prompt"] == "Once upon a time"
    assert snaps[0]["n_lanes"] == 2
    assert snaps[0]["lanes"][1]["cancelled"] is True

    # DB row is scoped to this session.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(CompareSnapshot).where(
                            CompareSnapshot.session_id == chat_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 1
            assert rows[0].n_lanes == 2
    finally:
        await engine.dispose()

    # IDOR: a second user cannot read or write this session's snapshots.
    # Reuse the same running client (its lifespan is owned by the fixture) but
    # swap the session cookie by logging in as the other user.
    other_id, other_email = await _make_user()
    await _make_chat_and_key(other_id)
    client.cookies.clear()
    _login(client, other_email, "test-pw-12345")
    assert client.get(f"/workbench/{chat_id}/compare/snapshots").status_code == 404
    assert (
        client.post(
            f"/workbench/{chat_id}/compare/snapshots",
            json={"prompt": "x", "lanes": [{"model": "gpt2"}]},
        ).status_code
        == 404
    )


@dbtest
async def test_compare_snapshot_prunes_to_limit(client):
    """History is capped at _COMPARE_SNAPSHOT_LIMIT; oldest rows are pruned."""
    from wrapper.routes.workbench import _COMPARE_SNAPSHOT_LIMIT

    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    for i in range(_COMPARE_SNAPSHOT_LIMIT + 5):
        r = client.post(
            f"/workbench/{chat_id}/compare/snapshots",
            json={"prompt": f"run-{i}", "lanes": [{"model": "gpt2"}]},
        )
        assert r.status_code == 201, r.text

    r = client.get(f"/workbench/{chat_id}/compare/snapshots")
    snaps = r.json()["snapshots"]
    assert len(snaps) == _COMPARE_SNAPSHOT_LIMIT
    # Newest first — the most recent run survived; the oldest were pruned.
    prompts = [s["prompt"] for s in snaps]
    assert prompts[0] == f"run-{_COMPARE_SNAPSHOT_LIMIT + 4}"
    assert "run-0" not in prompts


@dbtest
async def test_compare_snapshot_rejects_empty_lanes(client):
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")
    r = client.post(
        f"/workbench/{chat_id}/compare/snapshots", json={"prompt": "x", "lanes": []}
    )
    assert r.status_code == 400, r.text


@dbtest
async def test_compare_snapshots_deleted_with_session(client):
    """CASCADE FK: deleting the chat session removes its compare snapshots."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")
    client.post(
        f"/workbench/{chat_id}/compare/snapshots",
        json={"prompt": "x", "lanes": [{"model": "gpt2"}]},
    )

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = (
                await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
            ).scalar_one()
            await s.delete(chat)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(CompareSnapshot).where(
                            CompareSnapshot.session_id == chat_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert rows == []
    finally:
        await engine.dispose()


# --- ACS-186: server-side compare-snapshot barrier ---------------------------
#
# The barrier lives in ``_run_generation_task``: the last lane of a Compare
# "Run all" to reach a terminal state assembles + writes ONE CompareSnapshot
# from the persisted lane rows, so the snapshot survives closing the tab mid-run
# (no client POST needed). Dedup is enforced by a UNIQUE constraint on
# ``compare_snapshots.compare_run_id``.


async def _wait_for_compare_snapshot(chat_id, *, timeout_polls: int = 120):
    """Poll for exactly the compare snapshot(s) of a session, returning the list."""
    for _ in range(timeout_polls):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                rows = list(
                    (
                        await s.execute(
                            select(CompareSnapshot).where(
                                CompareSnapshot.session_id == chat_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                # Also confirm every lane row has left running, so we know the
                # batch actually finished (not just a slow first lane).
                gens = list(
                    (
                        await s.execute(
                            select(ChatGeneration).where(
                                ChatGeneration.session_id == chat_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
        finally:
            await engine.dispose()
        if gens and all(g.status != "running" for g in gens) and rows:
            return rows
    return rows


@dbtest
async def test_compare_barrier_writes_one_snapshot_without_client_post(client, monkeypatch):
    """A compare Run-all whose lanes all complete writes EXACTLY ONE
    CompareSnapshot server-side — no client POST — capturing the shared prompt,
    per-lane sampling config, and each lane's final completion (ACS-186)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"hi"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={
            "prompt": "Once upon a time",
            "lanes": [
                {"max_tokens": 8, "temperature": 0.7, "top_p": 0.9},
                {"max_tokens": 8, "temperature": 1.2, "top_k": 5},
            ],
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["lanes"]) == 2

    snaps = await _wait_for_compare_snapshot(chat_id)
    # Exactly ONE snapshot for the whole batch — not one per lane, not zero.
    assert len(snaps) == 1, f"expected 1 barrier snapshot, got {len(snaps)}"
    snap = snaps[0]
    assert snap.prompt == "Once upon a time"
    assert snap.n_lanes == 2
    assert snap.compare_run_id is not None
    # Lanes are in launch-index order and carry the persisted sampling config +
    # final completion the server saw (faithful reconstruction, no client input).
    assert [lane_["completion_text"] for lane_ in snap.lanes] == ["hi", "hi"]
    assert snap.lanes[0]["temperature"] == 0.7
    assert snap.lanes[0]["top_p"] == 0.9
    assert snap.lanes[1]["temperature"] == 1.2
    assert snap.lanes[1]["top_k"] == 5
    assert snap.lanes[0]["cancelled"] is False

    # All lane rows share the one batch id.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            gens = list(
                (
                    await s.execute(
                        select(ChatGeneration).where(ChatGeneration.session_id == chat_id)
                    )
                )
                .scalars()
                .all()
            )
            run_ids = {g.compare_run_id for g in gens}
            assert len(run_ids) == 1 and snap.compare_run_id in run_ids
    finally:
        await engine.dispose()


@dbtest
async def test_compare_client_post_dedupes_against_barrier(client, monkeypatch):
    """After the server barrier writes a batch's snapshot, the fallback client
    POST for that same run is an idempotent no-op — it returns the existing row
    and does NOT create a second snapshot (ACS-186 dedup on compare_run_id)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [b'data: {"choices":[{"text":"ok"}]}\n\n', b"data: [DONE]\n\n"],
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={"prompt": "p", "lanes": [{"max_tokens": 4}]},
    )
    assert r.status_code == 200, r.text

    snaps = await _wait_for_compare_snapshot(chat_id)
    assert len(snaps) == 1
    barrier_id = snaps[0].id
    barrier_run_id = snaps[0].compare_run_id

    # Now the client POSTs its assembled snapshot for the SAME run (as the real
    # browser would). It must dedupe: 200 (not 201), same id, still one row.
    r2 = client.post(
        f"/workbench/{chat_id}/compare/snapshots",
        json={
            "prompt": "p",
            "lanes": [{"model": "gpt2", "completion_text": "ok", "cancelled": False}],
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["id"] == barrier_id
    assert r2.json().get("compare_run_id", barrier_run_id) is not None

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (
                    await s.execute(
                        select(CompareSnapshot).where(
                            CompareSnapshot.session_id == chat_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 1, f"dedup failed: {len(rows)} snapshots"
            assert rows[0].id == barrier_id
    finally:
        await engine.dispose()


@dbtest
async def test_compare_barrier_snapshots_batch_with_failed_lanes(client, monkeypatch):
    """A compare batch whose lanes reach a terminal state via FAILURE (not just
    completion) still yields exactly ONE server-side snapshot — the barrier
    counts "not running", so failed/cancelled lanes count as terminal. The
    failed lanes are marked cancelled=True with their (empty) text preserved,
    and the sampling config still round-trips (ACS-186)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Every lane's upstream errors → both lanes reach terminal status "failed".
    _patch_upstream_chunks(
        monkeypatch,
        [b'{"error":{"message":"boom"}}'],
        status_code=502,
    )

    r = client.post(
        f"/workbench/{chat_id}/compare",
        json={
            "prompt": "will fail",
            "lanes": [
                {"max_tokens": 8, "top_p": 0.5, "seed": 7},
                {"max_tokens": 8},
            ],
        },
    )
    assert r.status_code == 200, r.text

    snaps = await _wait_for_compare_snapshot(chat_id)
    assert len(snaps) == 1, f"failed batch must still snapshot once, got {len(snaps)}"
    snap = snaps[0]
    assert snap.n_lanes == 2
    assert snap.compare_run_id is not None
    # Failed lanes are surfaced with cancelled=True (same flag the client uses).
    assert all(lane_["cancelled"] is True for lane_ in snap.lanes)
    # Persisted sampling config round-trips even on failure.
    assert snap.lanes[0]["top_p"] == 0.5
    assert snap.lanes[0]["seed"] == 7


@dbtest
async def test_compare_barrier_one_snapshot_per_batch_across_two_runs(client, monkeypatch):
    """Two successive Run-alls in one session produce two distinct snapshots
    (one per batch, distinct compare_run_id) — the barrier keys on the batch id,
    so a second run doesn't dedupe against the first (ACS-186)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [b'data: {"choices":[{"text":"x"}]}\n\n', b"data: [DONE]\n\n"],
    )

    for run in ("first", "second"):
        r = client.post(
            f"/workbench/{chat_id}/compare",
            json={"prompt": run, "lanes": [{"max_tokens": 4}]},
        )
        assert r.status_code == 200, r.text
        # Wait for THIS batch's snapshot before firing the next run.
        for _ in range(120):
            await asyncio.sleep(0.05)
            r2 = client.get(f"/workbench/{chat_id}/compare/snapshots")
            prompts = [s["prompt"] for s in r2.json()["snapshots"]]
            if run in prompts:
                break

    r = client.get(f"/workbench/{chat_id}/compare/snapshots")
    snaps = r.json()["snapshots"]
    assert len(snaps) == 2, f"expected 2 snapshots (one per batch), got {len(snaps)}"
    run_ids = {s["compare_run_id"] for s in snaps}
    assert len(run_ids) == 2 and None not in run_ids
    assert {s["prompt"] for s in snaps} == {"first", "second"}


async def _await_generation_done(gen_id: uuid.UUID) -> None:
    """Poll until the background generation task leaves ``running``."""
    for _ in range(120):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                row = (
                    await s.execute(
                        select(ChatGeneration).where(ChatGeneration.id == gen_id)
                    )
                ).scalar_one()
                if row.status != "running":
                    return
        finally:
            await engine.dispose()


async def _only_snapshot(chat_id: uuid.UUID) -> ChatSnapshot:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            snaps = list(
                (
                    await s.execute(
                        select(ChatSnapshot).where(ChatSnapshot.session_id == chat_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(snaps) == 1, f"expected exactly one snapshot, got {len(snaps)}"
            return snaps[0]
    finally:
        await engine.dispose()


@dbtest
async def test_snapshot_persists_and_roundtrips_logprobs(client, monkeypatch):
    """ACS-189: when a single-pane Continue ran with logprobs on, the created
    ``ChatSnapshot`` stores the per-token logprobs in the shared normalised
    ``[{token, logprob, top:[...]}]`` shape so the saved snapshot can be
    re-coloured with the heatmap."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Upstream streams two tokens, each with a logprobs block (as vLLM does when
    # ``logprobs`` is requested).
    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"he","logprobs":'
            b'{"tokens":["he"],"token_logprobs":[-0.1],'
            b'"top_logprobs":[{"he":-0.1," hi":-1.2}]}}]}\n\n',
            b'data: {"choices":[{"text":"llo","logprobs":'
            b'{"tokens":["llo"],"token_logprobs":[-2.3],'
            b'"top_logprobs":[{"llo":-2.3}]}}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/generations",
        # logprobs=5 is what flips the run into logprobs mode (FormData omits the
        # field entirely when the toggle is off).
        data={"prompt": "say", "max_tokens": 8, "temperature": 0.0, "logprobs": 5},
    )
    assert r.status_code == 200, r.text
    await _await_generation_done(uuid.UUID(r.json()["generation_id"]))

    snap = await _only_snapshot(chat_id)
    assert snap.completion_text == "hello"
    assert isinstance(snap.logprobs, list)
    assert [e["token"] for e in snap.logprobs] == ["he", "llo"]
    # Round-tripped through JSONB: shape + values survive intact.
    assert snap.logprobs[0]["logprob"] == pytest.approx(-0.1)
    assert snap.logprobs[0]["top"][0]["token"] == "he"


@dbtest
async def test_snapshot_logprobs_null_when_logprobs_off(client, monkeypatch):
    """ACS-189: a run WITHOUT logprobs stores NULL, so the snapshot UI falls back
    to plain text (no heatmap, no error)."""
    user_id, email = await _make_user()
    chat_id, _ = await _make_chat_and_key(user_id)
    _login(client, email, "test-pw-12345")

    _patch_upstream_chunks(
        monkeypatch,
        [
            b'data: {"choices":[{"text":"hi"}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ],
    )

    r = client.post(
        f"/workbench/{chat_id}/generations",
        data={"prompt": "say", "max_tokens": 8, "temperature": 0.0},
    )
    assert r.status_code == 200, r.text
    await _await_generation_done(uuid.UUID(r.json()["generation_id"]))

    snap = await _only_snapshot(chat_id)
    assert snap.completion_text == "hi"
    assert snap.logprobs is None
