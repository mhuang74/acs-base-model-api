"""Loom (tree/branching exploration) — ACS-148.

Unit tests for the per-index absorber, logprob normaliser, frame helper, and
``LoomGenerationState`` fan-out; plus DB-gated integration tests for root
creation, n>1 branch generation (asserting the upstream body carries
``n``/``logprobs``/``seed``), tree persistence across reload, and node deletion.

Mirrors ``tests/test_workbench_generations.py`` for the DB fixtures + the
``TEST_DATABASE_URL`` gating.
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
from wrapper.models import ApiKey, ChatSession, Loom, LoomNode, User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run lifecycle tests",
)


# --- unit tests (no DB) ------------------------------------------------------


def test_normalise_logprobs_flattens_vllm_shape():
    from wrapper.main import _normalise_logprobs

    lp = {
        "tokens": ["he", "llo"],
        "token_logprobs": [-0.1, -2.3],
        "top_logprobs": [
            {"he": -0.1, " hi": -1.2},
            {"llo": -2.3, "y": -2.9},
        ],
    }
    out = _normalise_logprobs(lp)
    assert len(out) == 2
    assert out[0]["token"] == "he"
    assert out[0]["logprob"] == -0.1
    # Highest-probability alternative first.
    assert out[0]["top"][0]["token"] == "he"
    assert out[0]["top"][1]["token"] == " hi"
    assert out[1]["token"] == "llo"


def test_normalise_logprobs_tolerates_missing_fields():
    from wrapper.main import _normalise_logprobs

    assert _normalise_logprobs(None) == []
    assert _normalise_logprobs({}) == []
    # tokens present but no top_logprobs → top is empty, logprob None.
    out = _normalise_logprobs({"tokens": ["a"]})
    assert out == [{"token": "a", "logprob": None, "top": []}]


def test_absorb_loom_chunk_splits_by_index():
    from wrapper.main import _absorb_loom_chunk

    chunk = (
        b'data: {"choices":[{"index":0,"text":"foo"},{"index":1,"text":"bar"}]}\n\n'
        b'data: {"choices":[{"index":0,"text":"!"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    by_index, saw_done = _absorb_loom_chunk(chunk)
    assert saw_done is True
    assert by_index[0]["text"] == "foo!"
    assert by_index[1]["text"] == "bar"


def test_absorb_loom_chunk_carries_logprobs():
    from wrapper.main import _absorb_loom_chunk

    chunk = (
        b'data: {"choices":[{"index":0,"text":"hi",'
        b'"logprobs":{"tokens":["hi"],"token_logprobs":[-0.5],'
        b'"top_logprobs":[{"hi":-0.5,"yo":-1.0}]}}]}\n\n'
    )
    by_index, saw_done = _absorb_loom_chunk(chunk)
    assert saw_done is False
    assert by_index[0]["text"] == "hi"
    assert by_index[0]["logprobs"][0]["token"] == "hi"
    assert by_index[0]["logprobs"][0]["top"][0]["token"] == "hi"


def test_absorb_loom_chunk_ignores_keepalive_and_junk():
    from wrapper.main import _absorb_loom_chunk

    by_index, saw_done = _absorb_loom_chunk(b": keep-alive\n\n")
    assert by_index == {}
    assert saw_done is False


def test_sse_loom_chunk_frame_shape():
    from wrapper.main import _sse_loom_chunk_frame

    frame = _sse_loom_chunk_frame({"index": 1, "text": "x", "logprobs": []})
    assert frame.startswith(b"event: loom_chunk\ndata: ")
    assert frame.endswith(b"\n\n")
    assert b'"index": 1' in frame


def test_loom_generation_state_branches_and_broadcast():
    from wrapper.main import LoomGenerationState

    s = LoomGenerationState(3, uuid.uuid4())
    assert len(s.branches) == 3
    q1 = s.subscribe()
    q2 = s.subscribe()
    s.broadcast(b"hi")
    assert q1.get_nowait() == b"hi"
    assert q2.get_nowait() == b"hi"
    s.unsubscribe(q1)
    s.broadcast(b"bye")
    assert q2.get_nowait() == b"bye"
    assert q1.qsize() == 0


async def test_loom_gen_live_replays_and_done():
    """A subscriber must get a replay of accumulated branch text + a done."""
    from wrapper import main as mainmod

    state = mainmod.LoomGenerationState(2, uuid.uuid4())
    state.branches[0]["text"] = "alpha"
    state.branches[1]["text"] = "beta"
    state.mark_done("completed")

    gen = mainmod.workbench_routes._loom_gen_live(state)
    frames = []
    try:
        for _ in range(5):
            try:
                frames.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))
            except StopAsyncIteration:
                break
    finally:
        await gen.aclose()
    blob = b"".join(frames)
    assert b"event: loom_chunk" in blob
    assert b"alpha" in blob and b"beta" in blob
    assert b"event: done" in blob


# --- DB-gated integration tests ----------------------------------------------


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-loom-secret-000000000000000")
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
        email = f"loom-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password))
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _make_loom_and_key(user_id: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            loom = Loom(user_id=user_id, title="loom-fixture")
            s.add(loom)
            gk = generate_key()
            key = ApiKey(
                user_id=user_id,
                key_hash=gk.hash_,
                key_prefix=gk.prefix,
                name="loom-key",
                monthly_token_budget=1_000_000,
            )
            s.add(key)
            await s.flush()
            return loom.id, key.id
    finally:
        await engine.dispose()


async def _make_chat(user_id: uuid.UUID) -> uuid.UUID:
    """A legacy chat session with no loom yet — for the old-URL redirect test."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            chat = ChatSession(user_id=user_id, title="legacy-chat", prompt_text="")
            s.add(chat)
            await s.flush()
            return chat.id
    finally:
        await engine.dispose()


def _login(client: TestClient, email: str, password: str) -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


def _patch_per_index_upstream(monkeypatch, *, captured_body: dict) -> None:
    """Fake ``stream_post_with_status`` that emits two indexed branches.

    Records the upstream body into ``captured_body`` so the test can assert
    n/logprobs/seed were plumbed through. Accepts ``**kwargs`` so the
    ``cancel_event=`` the loom task passes doesn't break the signature.
    """

    async def _fake(client, upstream_url, api_key, body, timeout_s, **kwargs):
        captured_body.update(body)
        yield {
            "kind": "chunk",
            "data": (
                b'data: {"choices":['
                b'{"index":0,"text":"aaa","logprobs":{"tokens":["aaa"],'
                b'"token_logprobs":[-0.2],"top_logprobs":[{"aaa":-0.2}]}},'
                b'{"index":1,"text":"bbb","logprobs":{"tokens":["bbb"],'
                b'"token_logprobs":[-0.9],"top_logprobs":[{"bbb":-0.9}]}}'
                b"]}\n\n"
            ),
            "usage": None,
            "status": 200,
        }
        yield {
            "kind": "chunk",
            "data": b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "status": None,
        }
        yield {"kind": "chunk", "data": b"data: [DONE]\n\n", "usage": None, "status": None}

    monkeypatch.setattr(proxymod, "stream_post_with_status", _fake)


@dbtest
async def test_create_root_then_branch_persists_nodes(client, monkeypatch):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # 1. Create the root node from typed text.
    r = client.post(f"/loom/{loom_id}/root", data={"text": "Once upon a time"})
    assert r.status_code == 200, r.text
    root_id = r.json()["node"]["id"]

    # 2. Branch from the root with n=2 + logprobs + seed.
    captured_body: dict = {}
    _patch_per_index_upstream(monkeypatch, captured_body=captured_body)
    r = client.post(
        f"/loom/{loom_id}/generate",
        data={
            "parent_id": root_id,
            "n": 2,
            "max_tokens": 8,
            "temperature": 0.7,
            "logprobs": 5,
            "seed": 1234,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["generation_id"]
    assert r.json()["n"] == 2

    # Wait for the background task to write the child nodes.
    children = []
    for _ in range(60):
        await asyncio.sleep(0.05)
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                children = list(
                    (
                        await s.execute(
                            select(LoomNode).where(
                                LoomNode.loom_id == loom_id,
                                LoomNode.parent_id == uuid.UUID(root_id),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
        finally:
            await engine.dispose()
        if len(children) >= 2:
            break

    # The upstream body carried n/logprobs/seed.
    assert captured_body.get("n") == 2
    assert captured_body.get("logprobs") == 5
    assert captured_body.get("seed") == 1234
    # The prompt was the reconstructed prefix (just the root text here).
    assert captured_body.get("prompt") == "Once upon a time"

    assert len(children) == 2
    texts = sorted(c.text for c in children)
    assert texts == ["aaa", "bbb"]
    # Logprobs + seed persisted on the nodes.
    for c in children:
        assert c.seed == 1234
        assert c.logprobs and c.logprobs[0]["token"] in ("aaa", "bbb")

    # 3. The tree JSON endpoint returns root + 2 children.
    tr = client.get(f"/loom/{loom_id}/tree")
    assert tr.status_code == 200
    nodes = tr.json()["nodes"]
    assert len(nodes) == 3
    assert sum(1 for n in nodes if n["parent_id"] is None) == 1


@dbtest
async def test_loom_events_stream_loom_chunks_and_done(client, monkeypatch):
    """The /loom/generations/{id}/events SSE tail emits per-branch loom_chunk
    frames (for both indices, via replay or live) + a terminal done frame."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(f"/loom/{loom_id}/root", data={"text": "seed"})
    assert r.status_code == 200
    root_id = r.json()["node"]["id"]

    captured_body: dict = {}
    _patch_per_index_upstream(monkeypatch, captured_body=captured_body)
    r = client.post(
        f"/loom/{loom_id}/generate",
        data={"parent_id": root_id, "n": 2, "max_tokens": 8, "logprobs": 5},
    )
    assert r.status_code == 200, r.text
    gen_id = r.json()["generation_id"]

    with client.stream("GET", f"/loom/{loom_id}/generations/{gen_id}/events") as resp:
        assert resp.status_code == 200
        body = b"".join(chunk for chunk in resp.iter_raw())
    assert b"event: loom_chunk" in body
    assert b"event: done" in body
    # Both branch texts should appear (via replay or live chunk).
    assert b"aaa" in body and b"bbb" in body


@dbtest
async def test_generate_requires_parent(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")
    r = client.post(f"/loom/{loom_id}/generate", data={"n": 2})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_parent"


@dbtest
async def test_delete_node_cascades_subtree(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Seed a root + child + grandchild directly.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root")
            s.add(root)
            await s.flush()
            child = LoomNode(loom_id=loom_id, parent_id=root.id, text="child")
            s.add(child)
            await s.flush()
            grand = LoomNode(loom_id=loom_id, parent_id=child.id, text="grand")
            s.add(grand)
            await s.flush()
            child_id = child.id
    finally:
        await engine.dispose()

    # Delete the middle node — child + grandchild should both vanish.
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/delete")
    assert r.status_code == 204

    tr = client.get(f"/loom/{loom_id}/tree")
    nodes = tr.json()["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["text"] == "root"


@dbtest
async def test_loom_page_renders(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")
    r = client.get(f"/loom/{loom_id}")
    assert r.status_code == 200
    assert b"Loom" in r.content
    assert b"loom-bootstrap" in r.content


@dbtest
async def test_loom_ownership_enforced(client):
    """A second user cannot touch the first user's loom."""
    owner_id, owner_email = await _make_user()
    loom_id, _ = await _make_loom_and_key(owner_id)
    _other_id, other_email = await _make_user()
    _login(client, other_email, "test-pw-12345")
    r = client.get(f"/loom/{loom_id}/tree")
    assert r.status_code == 404


@dbtest
async def test_loom_generation_events_and_cancel_scoped_to_session(client):
    """A loom ``gen_id`` is only accessible under its OWN session, even for a user
    who owns a *different* session — the generations map is global, so owner-gating
    the path session isn't enough on its own. Regression for the cross-session
    IDOR on /events + /cancel.
    """
    from wrapper.main import LoomGenerationState

    owner_id, _ = await _make_user()
    owner_loom, _ = await _make_loom_and_key(owner_id)
    attacker_id, attacker_email = await _make_user()
    attacker_loom, _ = await _make_loom_and_key(attacker_id)
    _login(client, attacker_email, "test-pw-12345")

    gens = client.app.state.loom_generations
    victim_gen = uuid.uuid4()
    gens[victim_gen] = LoomGenerationState(2, owner_loom)  # belongs to the owner
    own_gen = uuid.uuid4()
    gens[own_gen] = LoomGenerationState(2, attacker_loom)  # belongs to the attacker
    try:
        # Attacker owns attacker_chat (path passes ownership) but victim_gen
        # belongs to owner_chat → 404; no tailing or cancelling another session's
        # generation.
        r = client.get(f"/loom/{attacker_loom}/generations/{victim_gen}/events")
        assert r.status_code == 404
        r = client.post(f"/loom/{attacker_loom}/generations/{victim_gen}/cancel")
        assert r.status_code == 404
        # Control: the attacker's own generation under their own session is fine.
        r = client.post(f"/loom/{attacker_loom}/generations/{own_gen}/cancel")
        assert r.status_code == 204
    finally:
        gens.pop(victim_gen, None)
        gens.pop(own_gen, None)


@dbtest
async def test_delete_node_blocked_while_generation_running(client):
    """Deleting a node that an in-flight generate targets as its parent (or an
    ancestor of it) is refused with 409, so the generate's child insert can't hit
    a swallowed FK violation that silently drops the branches.
    """
    from wrapper.main import LoomGenerationState

    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root")
            s.add(root)
            await s.flush()
            child = LoomNode(loom_id=loom_id, parent_id=root.id, text="child")
            s.add(child)
            await s.flush()
            root_id, child_id = root.id, child.id
    finally:
        await engine.dispose()

    gens = client.app.state.loom_generations
    gen = uuid.uuid4()
    st = LoomGenerationState(2, loom_id, child_id)  # running, parent = child
    gens[gen] = st
    try:
        # Deleting the active parent is blocked …
        r = client.post(f"/loom/{loom_id}/nodes/{child_id}/delete")
        assert r.status_code == 409
        # … and so is deleting an ancestor of it (CASCADE would still orphan it).
        r = client.post(f"/loom/{loom_id}/nodes/{root_id}/delete")
        assert r.status_code == 409
        # Once the generation finishes, the delete is allowed again.
        st.status = "completed"
        r = client.post(f"/loom/{loom_id}/nodes/{child_id}/delete")
        assert r.status_code == 204
    finally:
        gens.pop(gen, None)


@dbtest
async def test_create_root_blocked_while_generation_running(client):
    """Replacing the root deletes the whole tree via CASCADE, so an in-flight
    generate's child insert would hit a swallowed FK violation. Any running
    generation in the loom must block the root re-post with 409.
    """
    from wrapper.main import LoomGenerationState

    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root")
            s.add(root)
            await s.flush()
            root_id = root.id
    finally:
        await engine.dispose()

    gens = client.app.state.loom_generations
    gen = uuid.uuid4()
    st = LoomGenerationState(2, loom_id, root_id)  # running under the current root
    gens[gen] = st
    try:
        r = client.post(f"/loom/{loom_id}/root", data={"text": "new root"})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "loom_busy"
        # Once the generation finishes, replacing the root is allowed again.
        st.status = "completed"
        r = client.post(f"/loom/{loom_id}/root", data={"text": "new root"})
        assert r.status_code == 200
    finally:
        gens.pop(gen, None)


# --- Tier-2 editing surface: edit / split / sibling / export / import (ACS-166) --


async def _seed_chain(loom_id: uuid.UUID, texts: list[str]) -> list[uuid.UUID]:
    """Insert a linear root→child→… chain with the given deltas; return ids."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            ids: list[uuid.UUID] = []
            parent: uuid.UUID | None = None
            for txt in texts:
                nd = LoomNode(loom_id=loom_id, parent_id=parent, text=txt)
                s.add(nd)
                await s.flush()
                ids.append(nd.id)
                parent = nd.id
            return ids
    finally:
        await engine.dispose()


async def _fetch_nodes(loom_id: uuid.UUID) -> dict[uuid.UUID, LoomNode]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = list(
                (await s.execute(select(LoomNode).where(LoomNode.loom_id == loom_id))).scalars().all()
            )
            # Detach so attributes stay readable after the session closes.
            return {r.id: r for r in rows}
    finally:
        await engine.dispose()


def _prefix(nodes: dict[uuid.UUID, LoomNode], node_id: uuid.UUID) -> str:
    """Reconstruct a node's full context by walking parent links (mirror of the
    server's ``_loom_prefix_for_node``) — used to assert split preserves context."""
    chain: list[str] = []
    cur: uuid.UUID | None = node_id
    seen: set[uuid.UUID] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        nd = nodes.get(cur)
        if nd is None:
            break
        chain.append(nd.text)
        cur = nd.parent_id
    return "".join(reversed(chain))


@dbtest
async def test_edit_node_mutates_text_and_clears_logprobs(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # Seed a node WITH a logprobs payload so we can assert it gets cleared.
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root")
            s.add(root)
            await s.flush()
            child = LoomNode(
                loom_id=loom_id,
                parent_id=root.id,
                text="old text",
                logprobs=[{"token": "old", "logprob": -0.1, "top": []}],
            )
            s.add(child)
            await s.flush()
            child_id = child.id
    finally:
        await engine.dispose()

    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/edit", data={"text": "new text"})
    assert r.status_code == 200, r.text
    assert r.json()["node"]["text"] == "new text"

    nodes = await _fetch_nodes(loom_id)
    assert nodes[child_id].text == "new text"
    assert nodes[child_id].logprobs is None  # stale payload cleared


@dbtest
async def test_split_reparents_children_and_preserves_context(client):
    """The load-bearing invariant: splitting a node MUST reparent its existing
    children onto the new tail child, so every descendant's reconstructed context
    is byte-identical before and after the split."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    # root("R") -> mid("ABCD") -> leaf("Z"); leaf's full context is "RABCDZ".
    root_id, mid_id, leaf_id = await _seed_chain(loom_id, ["R", "ABCD", "Z"])
    before = await _fetch_nodes(loom_id)
    ctx_before = _prefix(before, leaf_id)
    assert ctx_before == "RABCDZ"

    # Split "ABCD" at offset 2 → mid keeps "AB", new child gets "CD".
    r = client.post(f"/loom/{loom_id}/nodes/{mid_id}/split", data={"offset": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    new_child_id = uuid.UUID(body["child"]["id"])
    assert body["node"]["text"] == "AB"
    assert body["child"]["text"] == "CD"

    after = await _fetch_nodes(loom_id)
    # mid now has exactly one child: the new "CD" node.
    assert after[mid_id].text == "AB"
    assert after[mid_id].parent_id == root_id
    # The original leaf was reparented onto the new "CD" node, NOT left under mid.
    assert after[leaf_id].parent_id == new_child_id
    assert after[new_child_id].parent_id == mid_id
    # Byte-identical reconstructed context — the whole point of the split.
    assert _prefix(after, leaf_id) == "RABCDZ"
    # Split clears logprobs on both mutated/created nodes.
    assert after[mid_id].logprobs is None
    assert after[new_child_id].logprobs is None


@dbtest
async def test_new_sibling_shares_parent(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    root_id, child_id = await _seed_chain(loom_id, ["root", "child"])
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/sibling", data={"text": "alt"})
    assert r.status_code == 200, r.text
    sib_id = uuid.UUID(r.json()["node"]["id"])

    nodes = await _fetch_nodes(loom_id)
    assert nodes[sib_id].parent_id == root_id
    assert nodes[sib_id].text == "alt"
    # Hand-written → not mislabelled with a model.
    assert nodes[sib_id].model is None


@dbtest
async def test_split_with_edited_text_is_atomic(client):
    """Passing `text` to /split commits the edited body and the split in one
    request (no partial-write window)."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")
    root_id, mid_id = await _seed_chain(loom_id, ["R", "ABCD"])
    # Edit the body to "WXYZ" and split at offset 2 in ONE call.
    r = client.post(f"/loom/{loom_id}/nodes/{mid_id}/split", data={"offset": 2, "text": "WXYZ"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node"]["text"] == "WX"
    assert body["child"]["text"] == "YZ"
    # The tail child does NOT inherit a seed (a slice isn't seed-reproducible).
    after = await _fetch_nodes(loom_id)
    child_id = uuid.UUID(body["child"]["id"])
    assert after[child_id].seed is None


@dbtest
async def test_split_offset_is_python_indexed(client):
    """``offset`` indexes the node text the way Python slices it, and the two
    halves always concatenate back to the original.
    """
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")
    root_id, mid_id = await _seed_chain(loom_id, ["X", "ab\U0001F600cd"])

    r = client.post(f"/loom/{loom_id}/nodes/{mid_id}/split", data={"offset": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node"]["text"] == "ab\U0001F600"
    assert body["child"]["text"] == "cd"
    # And the split is lossless regardless of where it lands: head + tail == full.
    assert body["node"]["text"] + body["child"]["text"] == "ab\U0001F600cd"


@dbtest
async def test_new_sibling_refused_on_root(client):
    """A root has no parent and loom_create_root assumes one root — refuse."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")
    (root_id,) = tuple(await _seed_chain(loom_id, ["root"]))
    r = client.post(f"/loom/{loom_id}/nodes/{root_id}/sibling", data={"text": "x"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "root_no_sibling"


@dbtest
async def test_export_then_import_roundtrips_tree(client):
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    root_id, a_id, b_id = await _seed_chain(loom_id, ["root", "aa", "bb"])

    # Export → JSON attachment with version/loom/nodes.
    r = client.get(f"/loom/{loom_id}/export")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    doc = r.json()
    assert doc["version"] == 1
    assert len(doc["nodes"]) == 3

    # Import into a NEW loom; IDs are regenerated, topology preserved.
    r = client.post("/loom/import", json=doc)
    assert r.status_code == 200, r.text
    new_loom_id = uuid.UUID(r.json()["loom_id"])
    assert new_loom_id != loom_id

    imported = await _fetch_nodes(new_loom_id)
    assert len(imported) == 3
    # IDs must be freshly minted — none of the original ids survive.
    assert loom_id not in imported
    assert all(nid not in (root_id, a_id, b_id) for nid in imported)
    # Exactly one root; the linear chain of texts is preserved.
    roots = [n for n in imported.values() if n.parent_id is None]
    assert len(roots) == 1 and roots[0].text == "root"
    leaf = [n for n in imported.values() if n.text == "bb"][0]
    assert _prefix(imported, leaf.id) == "rootaabb"


@dbtest
async def test_import_rehomes_dangling_parent_ref_under_sole_root(client):
    """A doctored file whose parent_id points outside the imported set must not
    cross-reference another loom's node. The link is not resolvable, so with a
    single intended root present the orphan is re-homed under that root — keeping
    the one-root invariant the rest of the loom code assumes (never a foreign ref).
    """
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    foreign = str(uuid.uuid4())
    doc = {
        "version": 1,
        "loom": {"title": "evil", "model": None},
        "nodes": [
            {"id": "n1", "parent_id": None, "text": "root"},
            {"id": "n2", "parent_id": foreign, "text": "orphan"},  # dangling ref
        ],
    }
    r = client.post("/loom/import", json=doc)
    assert r.status_code == 200, r.text
    new_loom_id = uuid.UUID(r.json()["loom_id"])
    imported = await _fetch_nodes(new_loom_id)
    roots = [n for n in imported.values() if n.parent_id is None]
    # Exactly one root ("root"); the orphan was re-homed under it, not left as a
    # second root and not pointing at the foreign UUID.
    assert len(roots) == 1 and roots[0].text == "root"
    orphan = [n for n in imported.values() if n.text == "orphan"][0]
    assert orphan.parent_id == roots[0].id


@dbtest
async def test_import_preserves_genuine_forest(client):
    """A genuine multi-root export (two intended roots) round-trips as a forest —
    we only re-home dangling orphans when there's a single unambiguous root."""
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    doc = {
        "version": 1,
        "loom": {"title": "forest", "model": None},
        "nodes": [
            {"id": "r1", "parent_id": None, "text": "root one"},
            {"id": "r2", "parent_id": None, "text": "root two"},
        ],
    }
    r = client.post("/loom/import", json=doc)
    assert r.status_code == 200, r.text
    imported = await _fetch_nodes(uuid.UUID(r.json()["loom_id"]))
    roots = [n for n in imported.values() if n.parent_id is None]
    assert len(roots) == 2


@dbtest
async def test_import_breaks_cycles_and_ignores_bool_seed(client):
    """A crafted file can wire a parent cycle (n1→n2→n1) and pass a bool `seed`.
    Import must break the cycle (leave the loom walkable, no rootless loop) and
    not coerce `seed: true` into a fabricated seed of 1."""
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")
    doc = {
        "version": 1,
        "loom": {"title": "cyclic", "model": None},
        "nodes": [
            {"id": "n1", "parent_id": "n2", "text": "a", "seed": True},
            {"id": "n2", "parent_id": "n1", "text": "b"},
        ],
    }
    r = client.post("/loom/import", json=doc)
    assert r.status_code == 200, r.text
    imported = await _fetch_nodes(uuid.UUID(r.json()["loom_id"]))
    # The cycle is broken → at least one node reaches a root (no rootless loop).
    roots = [n for n in imported.values() if n.parent_id is None]
    assert len(roots) >= 1
    # `seed: true` was NOT coerced to 1.
    assert all(n.seed is None for n in imported.values())


@dbtest
async def test_edit_and_split_blocked_while_generation_running(client):
    """Editing/splitting a node whose subtree a running generate targets is 409 —
    same race guard as delete (the generate's child insert would land under a
    node whose text/parent just changed)."""
    from wrapper.main import LoomGenerationState

    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    root_id, child_id = await _seed_chain(loom_id, ["root", "child"])
    gens = client.app.state.loom_generations
    gen = uuid.uuid4()
    st = LoomGenerationState(2, loom_id, child_id)  # running, parent = child
    gens[gen] = st
    try:
        # Editing the child (in the running subtree) is blocked …
        r = client.post(f"/loom/{loom_id}/nodes/{child_id}/edit", data={"text": "x"})
        assert r.status_code == 409
        # … and so is splitting an ancestor of it.
        r = client.post(f"/loom/{loom_id}/nodes/{root_id}/split", data={"offset": 1})
        assert r.status_code == 409
        st.status = "completed"
        r = client.post(f"/loom/{loom_id}/nodes/{child_id}/edit", data={"text": "ok"})
        assert r.status_code == 200
    finally:
        gens.pop(gen, None)


@dbtest
async def test_tier2_endpoints_ownership_enforced(client):
    """A second user cannot edit/split/sibling/export another user's loom."""
    owner_id, _ = await _make_user()
    loom_id, _ = await _make_loom_and_key(owner_id)
    (node_id,) = tuple(await _seed_chain(loom_id, ["root"]))
    _other_id, other_email = await _make_user()
    _login(client, other_email, "test-pw-12345")

    assert client.post(f"/loom/{loom_id}/nodes/{node_id}/edit", data={"text": "x"}).status_code == 404
    assert client.post(f"/loom/{loom_id}/nodes/{node_id}/split", data={"offset": 0}).status_code == 404
    assert client.post(f"/loom/{loom_id}/nodes/{node_id}/sibling", data={"text": "x"}).status_code == 404
    assert client.get(f"/loom/{loom_id}/export").status_code == 404


# --- first-class object lifecycle (list / new / rename / delete / redirect) --


@dbtest
async def test_loom_list_new_rename_delete(client):
    """A loom is a first-class saved object: listed, created, renamed, archived."""
    user_id, email = await _make_user()
    _login(client, email, "test-pw-12345")

    # New loom → redirect to /loom/<id>.
    r = client.post("/loom/new", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/loom/")
    loom_id = loc.rsplit("/", 1)[-1]

    # It appears on the list page.
    r = client.get("/loom")
    assert r.status_code == 200
    assert b"Untitled loom" in r.content

    # Rename it → new title shows on the loom page.
    r = client.post(
        f"/loom/{loom_id}/rename", data={"title": "My named loom"}, follow_redirects=False
    )
    assert r.status_code == 303
    r = client.get(f"/loom/{loom_id}")
    assert r.status_code == 200
    assert b"My named loom" in r.content

    # Delete (archive) it → gone from the list, and the page 404s.
    r = client.post(f"/loom/{loom_id}/delete", follow_redirects=False)
    assert r.status_code == 303
    r = client.get(f"/loom/{loom_id}")
    assert r.status_code == 404


@dbtest
async def test_legacy_workbench_loom_url_redirects_idempotently(client):
    """The retired /workbench/{session_id}/loom URL redirects to /loom/{id},
    lazily minting a loom with the SAME id — repeat hits reuse it (no dupes)."""
    user_id, email = await _make_user()
    chat_id = await _make_chat(user_id)
    _login(client, email, "test-pw-12345")

    r = client.get(f"/workbench/{chat_id}/loom", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == f"/loom/{chat_id}"

    # Second hit is idempotent — same target, and only one loom exists.
    r2 = client.get(f"/workbench/{chat_id}/loom", follow_redirects=False)
    assert r2.headers["location"] == f"/loom/{chat_id}"

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            looms = list(
                (await s.execute(select(Loom).where(Loom.user_id == user_id))).scalars().all()
            )
    finally:
        await engine.dispose()
    assert len(looms) == 1
    assert looms[0].id == chat_id


# --- save fidelity: logprobs preservation + newline normalisation (ACS-338/339) --


@dbtest
async def test_noop_edit_preserves_logprobs_and_normalises_crlf(client):
    """The field-report bug (Josh, 2026-08-06 §1/§2): opening a node and saving
    without changing the text destroyed its logprobs, and the textarea round-trip
    rewrote LF→CRLF. Faithful repro: seed LF text + logprobs, POST the *same*
    text with every ``\\n`` turned into ``\\r\\n`` (what the browser submits).
    After normalisation the token sequence is unchanged, so logprobs MUST survive
    and the stored text MUST stay LF (ACS-338 + ACS-339 in one)."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    stored_lf = "line one\n\nline two"
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="seed", position=0)
            s.add(root)
            await s.flush()
            child = LoomNode(
                loom_id=loom_id,
                parent_id=root.id,
                text=stored_lf,
                position=0,
                logprobs=[{"token": "line", "logprob": -0.3, "top": []}],
            )
            s.add(child)
            await s.flush()
            child_id = child.id
    finally:
        await engine.dispose()

    # What the textarea actually POSTs: the same content, CRLF-encoded.
    crlf_roundtrip = stored_lf.replace("\n", "\r\n")
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/edit", data={"text": crlf_roundtrip})
    assert r.status_code == 200, r.text

    nodes = await _fetch_nodes(loom_id)
    # Logprobs preserved — the token sequence did not actually change.
    assert nodes[child_id].logprobs == [{"token": "line", "logprob": -0.3, "top": []}]
    # Stored text is LF, byte-identical to what we seeded (no CRLF growth).
    assert nodes[child_id].text == stored_lf
    assert "\r" not in nodes[child_id].text


@dbtest
async def test_edit_that_changes_text_still_clears_logprobs(client):
    """The guard is narrow: a save that DOES change the token sequence must still
    drop the stale (now-misaligned) logprobs, exactly as before."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="seed", position=0)
            s.add(root)
            await s.flush()
            child = LoomNode(
                loom_id=loom_id,
                parent_id=root.id,
                text="original",
                position=0,
                logprobs=[{"token": "orig", "logprob": -0.1, "top": []}],
            )
            s.add(child)
            await s.flush()
            child_id = child.id
    finally:
        await engine.dispose()

    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/edit", data={"text": "genuinely different"})
    assert r.status_code == 200, r.text
    nodes = await _fetch_nodes(loom_id)
    assert nodes[child_id].text == "genuinely different"
    assert nodes[child_id].logprobs is None  # stale payload cleared on real change


@dbtest
async def test_crlf_normalised_to_lf_on_root_and_sibling(client):
    """CRLF (and lone CR) collapse to LF wherever node text is written from a
    form — root create and hand-written sibling — so downstream branches tokenise
    identically regardless of whether the editor touched the node (ACS-339)."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(f"/loom/{loom_id}/root", data={"text": "a\r\nb\rc"})
    assert r.status_code == 200, r.text
    root_id = uuid.UUID(r.json()["node"]["id"])
    assert r.json()["node"]["text"] == "a\nb\nc"

    (child_id,) = tuple(await _seed_chain_under(loom_id, root_id, ["child"]))
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/sibling", data={"text": "x\r\ny"})
    assert r.status_code == 200, r.text
    assert r.json()["node"]["text"] == "x\ny"

    nodes = await _fetch_nodes(loom_id)
    assert nodes[root_id].text == "a\nb\nc"
    assert all("\r" not in n.text for n in nodes.values())


# --- stable sibling order via `position` (ACS-340) ----------------------------


async def _seed_chain_under(
    loom_id: uuid.UUID, parent_id: uuid.UUID, texts: list[str]
) -> list[uuid.UUID]:
    """Insert a linear chain rooted under an existing ``parent_id``; return ids."""
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            ids: list[uuid.UUID] = []
            parent = parent_id
            for txt in texts:
                nd = LoomNode(loom_id=loom_id, parent_id=parent, text=txt)
                s.add(nd)
                await s.flush()
                ids.append(nd.id)
                parent = nd.id
            return ids
    finally:
        await engine.dispose()


@dbtest
async def test_generated_and_sibling_positions_are_assigned_and_ordered(client, monkeypatch):
    """Siblings get a distinct 0-based ``position`` at creation so their order is
    stable even when a generate batch shares one ``created_at`` (ACS-340)."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(f"/loom/{loom_id}/root", data={"text": "seed"})
    assert r.status_code == 200, r.text
    root_id = r.json()["node"]["id"]

    # Two generated branches under the root → positions 0 and 1.
    _patch_per_index_upstream(monkeypatch, captured_body={})
    r = client.post(
        f"/loom/{loom_id}/generate",
        data={"parent_id": root_id, "n": 2, "max_tokens": 8, "logprobs": 5, "seed": 1234},
    )
    assert r.status_code == 200, r.text
    gen_children = []
    for _ in range(60):
        await asyncio.sleep(0.05)
        nodes = await _fetch_nodes(loom_id)
        gen_children = [n for n in nodes.values() if str(n.parent_id) == root_id]
        if len(gen_children) >= 2:
            break
    assert len(gen_children) == 2
    assert sorted(n.position for n in gen_children) == [0, 1]

    # A hand-written sibling of one branch lands under the SAME parent (root) and
    # takes the next ordinal after the batch → position 2.
    any_child_id = gen_children[0].id
    r = client.post(f"/loom/{loom_id}/nodes/{any_child_id}/sibling", data={"text": "hand"})
    assert r.status_code == 200, r.text
    assert r.json()["node"]["position"] == 2

    # The tree endpoint returns the root's children in (position, created_at)
    # order — 0,1,2 with no gaps.
    nodes = await _fetch_nodes(loom_id)
    kids = sorted(
        (n for n in nodes.values() if str(n.parent_id) == root_id),
        key=lambda n: (n.position, n.created_at),
    )
    assert [n.position for n in kids] == [0, 1, 2]


@dbtest
async def test_loom_nodes_orders_by_position_when_created_at_ties(client):
    """Direct proof that ``_loom_nodes`` sorts siblings by ``position`` first:
    three rows sharing one ``created_at`` but inserted with positions [2,0,1]
    come back ordered [0,1,2] (ACS-340). Without ``position`` the tie on
    ``created_at`` left order undefined — the export→import reordering bug."""
    import datetime as _dt

    from wrapper.routes.loom import _loom_nodes

    user_id, _email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)

    shared = _dt.datetime(2026, 8, 6, 12, 0, 0, tzinfo=_dt.UTC)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root", position=0)
            s.add(root)
            await s.flush()
            # Insert in a scrambled position order, all with the SAME created_at.
            for txt, pos in [("c", 2), ("a", 0), ("b", 1)]:
                s.add(
                    LoomNode(
                        loom_id=loom_id,
                        parent_id=root.id,
                        text=txt,
                        position=pos,
                        created_at=shared,
                    )
                )
            await s.flush()
            root_id = root.id
            ordered = await _loom_nodes(s, loom_id)
            # Read inside the session (attrs expire on commit): (text, position).
            siblings = [
                (n.text, n.position) for n in ordered if n.parent_id == root_id
            ]
    finally:
        await engine.dispose()

    assert [t for t, _ in siblings] == ["a", "b", "c"]
    assert [p for _, p in siblings] == [0, 1, 2]


@dbtest
async def test_migration_backfills_position_for_preexisting_rows():
    """The 0041 backfill numbers each parent's children 0..N-1 by their existing
    ``(created_at, id)`` order. The alembic-on-empty-DB harness never runs the
    backfill over real data, so we execute the migration's *exact* statement
    against simulated pre-add-column rows (all position 0, one shared
    ``created_at``) and assert it produces a dense, id-ordered ordinal."""
    import datetime as _dt
    import importlib.util
    from pathlib import Path

    from sqlalchemy import text as _sql_text

    # Load the migration's BACKFILL_POSITION_SQL constant by file path (the
    # versions dir isn't an importable package) so the test can never drift from
    # the SQL the migration actually runs.
    mig_path = (
        Path(__file__).resolve().parent.parent
        / "alembic"
        / "versions"
        / "0041_loom_node_position.py"
    )
    spec = importlib.util.spec_from_file_location("mig_0041", mig_path)
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)
    backfill_sql = mig.BACKFILL_POSITION_SQL

    user_id, _email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)

    shared = _dt.datetime(2026, 8, 6, 12, 0, 0, tzinfo=_dt.UTC)
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            root = LoomNode(loom_id=loom_id, parent_id=None, text="root", position=0)
            s.add(root)
            await s.flush()
            # Five siblings, ALL position 0 (the post-add-column / pre-backfill
            # state) sharing one created_at → order must fall back to id.
            child_ids: list[uuid.UUID] = []
            for txt in ["s0", "s1", "s2", "s3", "s4"]:
                nd = LoomNode(
                    loom_id=loom_id,
                    parent_id=root.id,
                    text=txt,
                    position=0,
                    created_at=shared,
                )
                s.add(nd)
                await s.flush()
                child_ids.append(nd.id)

            # Run the migration's backfill (raw SQL UPDATE).
            await s.execute(_sql_text(backfill_sql))
            # Read positions back with a plain SQL SELECT (not the ORM, whose
            # identity-map rows the raw UPDATE didn't touch) so we see the
            # freshly-backfilled values, not cached 0s.
            res = await s.execute(
                _sql_text(
                    "SELECT id, position FROM loom_nodes "
                    "WHERE loom_id = :lid AND parent_id IS NOT NULL"
                ),
                {"lid": str(loom_id)},
            )
            id_pos = [(row[0], row[1]) for row in res.all()]
    finally:
        await engine.dispose()

    by_pos = sorted(id_pos, key=lambda ip: ip[1])
    # Dense 0..N-1 ordinal, no duplicates, no gaps.
    assert [pos for _, pos in by_pos] == [0, 1, 2, 3, 4]
    # Tie on created_at → ordered by id (Postgres uuid order == Python uuid
    # order), so the position order equals ascending-id order.
    assert [i for i, _ in by_pos] == sorted(child_ids)


@dbtest
async def test_export_import_preserves_sibling_order(client):
    """The reported ACS-340 symptom end-to-end: a loom with several siblings under
    one parent must round-trip export→import with the children in the SAME order.
    ``position`` is carried in the export and honoured on import."""
    user_id, email = await _make_user()
    loom_id, _ = await _make_loom_and_key(user_id)
    _login(client, email, "test-pw-12345")

    r = client.post(f"/loom/{loom_id}/root", data={"text": "root"})
    assert r.status_code == 200, r.text
    root_id = uuid.UUID(r.json()["node"]["id"])

    # One real child (position 0), then two hand siblings of it (positions 1, 2).
    (child_id,) = tuple(await _seed_chain_under(loom_id, root_id, ["first"]))
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/sibling", data={"text": "second"})
    assert r.json()["node"]["position"] == 1
    r = client.post(f"/loom/{loom_id}/nodes/{child_id}/sibling", data={"text": "third"})
    assert r.json()["node"]["position"] == 2

    # Export carries position on every node.
    doc = client.get(f"/loom/{loom_id}/export").json()
    assert all("position" in n for n in doc["nodes"])

    # Import into a fresh loom and read its children back in (position, created_at)
    # order — must be the original [first, second, third].
    new_loom_id = uuid.UUID(client.post("/loom/import", json=doc).json()["loom_id"])
    imported = await _fetch_nodes(new_loom_id)
    new_root = next(n for n in imported.values() if n.parent_id is None)
    kids = sorted(
        (n for n in imported.values() if n.parent_id == new_root.id),
        key=lambda n: (n.position, n.created_at),
    )
    assert [n.text for n in kids] == ["first", "second", "third"]
    assert [n.position for n in kids] == [0, 1, 2]
