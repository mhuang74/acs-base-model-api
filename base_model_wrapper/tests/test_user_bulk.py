"""Bulk admin actions over a multi-selected user set (ACS-372)."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, User, UserTag
from wrapper.services import user_tags
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run bulk-action tests",
)


def _ue() -> str:
    return f"bulk-{uuid.uuid4().hex[:8]}@example.local"


async def _make_user(*, role: str = "user", status: str = "approved") -> tuple[uuid.UUID, str]:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        email = _ue()
        async with session_scope(factory) as s:
            u = User(
                email=email,
                password_hash=hash_password("test-pw-12345"),
                role=role,
                status=status,
            )
            s.add(u)
            await s.flush()
            return u.id, email
    finally:
        await engine.dispose()


async def _fetch(user_id: uuid.UUID) -> User | None:
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            return (await s.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    finally:
        await engine.dispose()


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-bulk-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    os.environ["EMAIL_ENABLED"] = "false"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True


def _login(client: TestClient, email: str) -> None:
    r = client.post(
        "/login", data={"email": email, "password": "test-pw-12345"}, follow_redirects=False
    )
    assert r.status_code == 303, f"login failed: {r.status_code}"


def _bulk(client: TestClient, **data):
    return client.post("/admin/users/bulk", data=data, follow_redirects=False)


# ---- routing ----------------------------------------------------------------


@dbtest
async def test_bulk_path_resolves_to_the_bulk_endpoint(client):
    """POST /admin/users/bulk reaches this endpoint, not the UUID route.

    Note on what this does *not* prove: registering ``user_bulk`` before
    ``users`` is currently inert for this route. Starlette only short-circuits
    on ``Match.FULL``; a path match with the wrong method yields
    ``Match.PARTIAL``, which is remembered but not returned early — and
    ``/admin/users/{user_id}`` is GET-only. So this passes with either
    registration order. The ordering is kept as cheap insurance against someone
    adding a ``GET /admin/users/bulk`` page later, which is the GET-vs-GET
    collision the bulk_emails/emails precedent actually describes.
    """
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    r = _bulk(client, action="suspend")
    assert r.status_code == 303, f"did not reach the bulk endpoint: {r.status_code}"
    # A GET on the same path *does* hit the UUID route — the asymmetry above.
    assert client.get("/admin/users/bulk", follow_redirects=False).status_code == 422


# ---- guards -----------------------------------------------------------------


@dbtest
async def test_empty_selection_is_rejected(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    r = _bulk(client, action="suspend")
    assert r.status_code == 303
    assert "err=" in r.headers["location"]


@dbtest
async def test_unknown_action_is_rejected_not_ignored(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)
    r = _bulk(client, action="explode", user_ids=[str(target_id)])
    assert r.status_code == 303
    assert "err=" in r.headers["location"]
    assert (await _fetch(target_id)).status == "approved"


@dbtest
async def test_own_account_is_always_dropped_from_the_selection(client):
    """Selecting yourself must never suspend you out of the admin surface."""
    admin_id, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(client, action="suspend", user_ids=[str(admin_id), str(target_id)])
    assert r.status_code == 303
    assert (await _fetch(admin_id)).status == "approved"
    assert (await _fetch(target_id)).status == "suspended"


# ---- suspend / unsuspend ----------------------------------------------------


@dbtest
async def test_bulk_suspend_and_unsuspend_round_trip(client):
    _, admin_email = await _make_user(role="admin")
    a_id, _ = await _make_user()
    b_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(client, action="suspend", user_ids=[str(a_id), str(b_id)])
    assert r.status_code == 303
    assert "Suspended+2" in r.headers["location"] or "Suspended%202" in r.headers["location"]
    for uid in (a_id, b_id):
        u = await _fetch(uid)
        assert u.status == "suspended"
        assert u.suspended_at is not None and u.suspended_by_user_id is not None

    r = _bulk(client, action="unsuspend", user_ids=[str(a_id), str(b_id)])
    assert r.status_code == 303
    for uid in (a_id, b_id):
        u = await _fetch(uid)
        assert u.status == "approved"
        assert u.suspended_at is None and u.suspended_by_user_id is None


@dbtest
async def test_bulk_suspend_skips_non_approved_and_reports_it(client):
    """A mixed selection suspends the eligible ones rather than erroring — and
    the flash reports the real number, not the selection size."""
    _, admin_email = await _make_user(role="admin")
    ok_id, _ = await _make_user(status="approved")
    pending_id, _ = await _make_user(status="pending")
    _login(client, admin_email)

    r = _bulk(client, action="suspend", user_ids=[str(ok_id), str(pending_id)])
    assert r.status_code == 303
    loc = r.headers["location"]
    assert "Suspended+1" in loc or "Suspended%201" in loc
    assert "skipped" in loc
    assert (await _fetch(ok_id)).status == "suspended"
    assert (await _fetch(pending_id)).status == "pending"


@dbtest
async def test_bulk_suspend_revokes_live_sessions(client):
    _, admin_email = await _make_user(role="admin")
    target_id, target_email = await _make_user()

    _login(client, target_email)
    assert client.get("/dashboard", follow_redirects=False).status_code == 200
    victim_cookies = dict(client.cookies)

    client.cookies.clear()
    _login(client, admin_email)
    assert _bulk(client, action="suspend", user_ids=[str(target_id)]).status_code == 303

    client.cookies.clear()
    client.cookies.update(victim_cookies)
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/login")


# ---- tags -------------------------------------------------------------------


@dbtest
async def test_bulk_tag_add_and_remove(client):
    _, admin_email = await _make_user(role="admin")
    a_id, _ = await _make_user()
    b_id, _ = await _make_user()
    _login(client, admin_email)
    tag = f"bulk-{uuid.uuid4().hex[:6]}"

    r = _bulk(client, action="tag_add", tag=tag.upper(), user_ids=[str(a_id), str(b_id)])
    assert r.status_code == 303
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            m = await user_tags.load_tags_for_users(s, [a_id, b_id])
            assert m[a_id] == [tag] and m[b_id] == [tag]  # normalised
    finally:
        await engine.dispose()

    # Re-applying is idempotent and says so.
    r = _bulk(client, action="tag_add", tag=tag, user_ids=[str(a_id), str(b_id)])
    assert "already" in r.headers["location"]

    r = _bulk(client, action="tag_remove", tag=tag, user_ids=[str(a_id), str(b_id)])
    assert r.status_code == 303
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert await user_tags.load_tags_for_users(s, [a_id, b_id]) == {}
    finally:
        await engine.dispose()


@dbtest
async def test_bulk_tag_rejects_an_unusable_tag(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)
    r = _bulk(client, action="tag_add", tag="!!!", user_ids=[str(target_id)])
    assert r.status_code == 303
    assert "err=" in r.headers["location"]

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            rows = (
                (await s.execute(select(UserTag).where(UserTag.user_id == target_id)))
                .scalars()
                .all()
            )
            assert list(rows) == []
    finally:
        await engine.dispose()


# ---- budgets ----------------------------------------------------------------


@dbtest
async def test_bulk_set_budget_with_and_without_keys(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
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
                    monthly_token_budget=5_000_000,
                )
            )
    finally:
        await engine.dispose()
    _login(client, admin_email)

    # Aggregate only — the key keeps its own budget.
    assert (
        _bulk(client, action="set_budget", budget="300000", user_ids=[str(target_id)]).status_code
        == 303
    )
    assert (await _fetch(target_id)).monthly_token_budget_total == 300000
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalar_one()
            assert k.monthly_token_budget == 5_000_000
    finally:
        await engine.dispose()

    # Now cascade to the keys too — this is the lever that actually caps spend.
    assert (
        _bulk(
            client,
            action="set_budget",
            budget="300000",
            also_per_key="1",
            user_ids=[str(target_id)],
        ).status_code
        == 303
    )
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalar_one()
            assert k.monthly_token_budget == 300000
    finally:
        await engine.dispose()


@dbtest
async def test_bulk_set_budget_rejects_garbage(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)
    for bad in ("abc", "-5"):
        r = _bulk(client, action="set_budget", budget=bad, user_ids=[str(target_id)])
        assert r.status_code == 303, bad
        assert "err=" in r.headers["location"], bad
        assert (await _fetch(target_id)).monthly_token_budget_total is None, bad


# ---- delete -----------------------------------------------------------------


@dbtest
async def test_bulk_delete_requires_the_exact_typed_phrase(client):
    _, admin_email = await _make_user(role="admin")
    a_id, _ = await _make_user()
    b_id, _ = await _make_user()
    _login(client, admin_email)
    ids = [str(a_id), str(b_id)]

    # No phrase, wrong phrase, and a phrase naming the wrong count all refuse.
    for phrase in ("", "delete", "Permanently delete 1 user", "permanently delete 2 users"):
        r = _bulk(client, action="delete", confirm_phrase=phrase, user_ids=ids)
        assert r.status_code == 303, phrase
        assert "err=" in r.headers["location"], phrase
        assert await _fetch(a_id) is not None, phrase
        assert await _fetch(b_id) is not None, phrase

    r = _bulk(client, action="delete", confirm_phrase="Permanently delete 2 users", user_ids=ids)
    assert r.status_code == 303
    assert "err=" not in r.headers["location"]
    assert await _fetch(a_id) is None
    assert await _fetch(b_id) is None


@dbtest
async def test_bulk_delete_phrase_is_singular_for_one(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(
        client,
        action="delete",
        confirm_phrase="Permanently delete 1 users",
        user_ids=[str(target_id)],
    )
    assert "err=" in r.headers["location"]
    assert await _fetch(target_id) is not None

    r = _bulk(
        client,
        action="delete",
        confirm_phrase="Permanently delete 1 user",
        user_ids=[str(target_id)],
    )
    assert "err=" not in r.headers["location"]
    assert await _fetch(target_id) is None


@dbtest
async def test_bulk_delete_refuses_the_whole_batch_if_it_contains_an_admin(client):
    """The single-user last-admin guard uses SELECT … FOR UPDATE; inside a bulk
    loop it would evaluate against not-yet-committed state and happily delete
    N-1 admins. Admins are simply not bulk-deletable."""
    _, admin_email = await _make_user(role="admin")
    other_admin_id, other_admin_email = await _make_user(role="admin")
    plain_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(
        client,
        action="delete",
        confirm_phrase="Permanently delete 2 users",
        user_ids=[str(other_admin_id), str(plain_id)],
    )
    assert r.status_code == 303
    assert "err=" in r.headers["location"]
    # Nothing deleted — the whole batch is refused, not just the admin.
    assert await _fetch(other_admin_id) is not None
    assert await _fetch(plain_id) is not None


@dbtest
async def test_bulk_delete_cascades_keys(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
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
    _login(client, admin_email)

    r = _bulk(
        client,
        action="delete",
        confirm_phrase="Permanently delete 1 user",
        user_ids=[str(target_id)],
    )
    assert r.status_code == 303
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            keys = (
                (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalars().all()
            )
            assert list(keys) == []
    finally:
        await engine.dispose()


# ---- redirect target --------------------------------------------------------


@dbtest
async def test_return_to_preserves_the_filtered_view(client):
    """The redirect must be a *well-formed* URL carrying both the filter and the
    flash.

    Substring-matching the Location header is not enough: the first version of
    this test asserted ``"status=approved" in loc`` and passed happily against
    ``?status=approved&sort=tokens?msg=...`` — a second "?" that swallowed the
    flash into the sort value and, with a tag filter, sent the admin to an empty
    roster with no confirmation right after a bulk delete. Parse it.
    """
    from urllib.parse import parse_qs, urlsplit

    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(
        client,
        action="suspend",
        user_ids=[str(target_id)],
        return_to="/admin/users?status=approved&sort=tokens",
    )
    assert r.status_code == 303
    parts = urlsplit(r.headers["location"])
    assert parts.path == "/admin/users"
    q = parse_qs(parts.query)
    assert q["status"] == ["approved"]
    assert q["sort"] == ["tokens"], f"sort corrupted: {q.get('sort')}"
    assert q["msg"][0].startswith("Suspended"), "flash message lost"


@dbtest
async def test_return_to_drops_a_stale_flash(client):
    """return_to is captured from the current URL, so last action's msg/err
    would otherwise round-trip into this one's redirect."""
    from urllib.parse import parse_qs, urlsplit

    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(
        client,
        action="suspend",
        user_ids=[str(target_id)],
        return_to="/admin/users?tag=x&msg=OLD+MESSAGE&err=OLD+ERROR",
    )
    q = parse_qs(urlsplit(r.headers["location"]).query)
    assert q["tag"] == ["x"]
    assert "OLD MESSAGE" not in q.get("msg", [""])[0]
    assert "err" not in q


@dbtest
async def test_return_to_cannot_be_used_as_an_open_redirect(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    for evil in (
        "https://evil.example/x",
        "//evil.example",
        "/admin/emails",
        "javascript:alert(1)",
    ):
        r = _bulk(client, action="suspend", user_ids=[str(target_id)], return_to=evil)
        assert r.status_code == 303, evil
        assert r.headers["location"].startswith("/admin/users"), evil
        assert "evil.example" not in r.headers["location"], evil


# ---- same-origin guard (partial ACS-102) ------------------------------------


@dbtest
async def test_cross_origin_bulk_post_is_rejected(client):
    """Bulk actions turn one forged POST from one victim into N (ACS-372).

    The repo has no CSRF token; ``samesite=lax`` is the only other mitigation
    and it is the browser's promise, not ours. A cross-site Origin is refused.
    """
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        "/admin/users/bulk",
        data={
            "action": "delete",
            "confirm_phrase": "Permanently delete 1 user",
            "user_ids": [str(target_id)],
        },
        headers={"Origin": "https://evil.example"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert await _fetch(target_id) is not None, "cross-origin POST deleted a user"


@dbtest
async def test_same_origin_bulk_post_is_allowed(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        "/admin/users/bulk",
        data={"action": "suspend", "user_ids": [str(target_id)]},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert (await _fetch(target_id)).status == "suspended"


@dbtest
async def test_cross_origin_single_user_delete_is_rejected(client):
    """The same guard covers the pre-existing destructive single-user routes."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        f"/admin/users/{target_id}/delete",
        headers={"Referer": "https://evil.example/attack"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert await _fetch(target_id) is not None


# ---- gaps the review named ---------------------------------------------------


@dbtest
async def test_budget_zero_with_per_key_is_refused(client):
    """0 means *blocked* on the user aggregate but *unlimited* on a key (the
    legacy sentinel in auth.py). Applying it to both from one number would do
    the exact opposite of what the operator asked, in one submit."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
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
                    monthly_token_budget=1000,
                )
            )
    finally:
        await engine.dispose()
    _login(client, admin_email)

    r = _bulk(client, action="set_budget", budget="0", also_per_key="1", user_ids=[str(target_id)])
    assert r.status_code == 303
    assert "err=" in r.headers["location"]
    assert "UNLIMITED" in r.headers["location"]
    # Nothing changed on either side.
    assert (await _fetch(target_id)).monthly_token_budget_total is None
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalar_one()
            assert k.monthly_token_budget == 1000
    finally:
        await engine.dispose()

    # Aggregate-only 0 is still allowed — that one genuinely means "blocked".
    r = _bulk(client, action="set_budget", budget="0", user_ids=[str(target_id)])
    assert "err=" not in r.headers["location"]
    assert (await _fetch(target_id)).monthly_token_budget_total == 0


@dbtest
async def test_blank_budget_clears_the_aggregate_and_unlimits_keys(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
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
                    monthly_token_budget=1000,
                )
            )
    finally:
        await engine.dispose()
    _login(client, admin_email)

    r = _bulk(client, action="set_budget", budget="", also_per_key="1", user_ids=[str(target_id)])
    assert "err=" not in r.headers["location"]
    assert (await _fetch(target_id)).monthly_token_budget_total is None
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            k = (await s.execute(select(ApiKey).where(ApiKey.user_id == target_id))).scalar_one()
            assert k.monthly_token_budget == 0  # 0 == unlimited for a key
    finally:
        await engine.dispose()


@dbtest
async def test_tag_add_tolerates_a_stale_id(client):
    """A roster tab left open while an account was deleted elsewhere must not
    500 the whole request on a FK violation."""
    _, admin_email = await _make_user(role="admin")
    live_id, _ = await _make_user()
    ghost = str(uuid.uuid4())
    _login(client, admin_email)
    tag = f"ghost-{uuid.uuid4().hex[:6]}"

    r = _bulk(client, action="tag_add", tag=tag, user_ids=[str(live_id), ghost])
    assert r.status_code == 303, f"stale id crashed the request: {r.status_code}"
    assert "err=" not in r.headers["location"]
    # And the count is honest — 1 tagged, the ghost isn't reported as "already had it".
    assert "Tagged+1" in r.headers["location"]
    assert "already" not in r.headers["location"]


@dbtest
async def test_bulk_suspend_skips_admins(client):
    """Suspending every other admin in one click would soft-lock them all out
    (a suspended admin fails current_user). They're skipped and reported."""
    _, admin_email = await _make_user(role="admin")
    other_admin_id, _ = await _make_user(role="admin")
    plain_id, _ = await _make_user()
    _login(client, admin_email)

    r = _bulk(client, action="suspend", user_ids=[str(other_admin_id), str(plain_id)])
    assert r.status_code == 303
    assert (await _fetch(other_admin_id)).status == "approved", "an admin was bulk-suspended"
    assert (await _fetch(plain_id)).status == "suspended"
    assert "skipped" in r.headers["location"]


@dbtest
async def test_batch_over_the_cap_is_refused(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    ids = [str(uuid.uuid4()) for _ in range(501)]
    r = _bulk(client, action="suspend", user_ids=ids)
    assert r.status_code == 303
    assert "err=" in r.headers["location"]
    assert "Too+many" in r.headers["location"]


@dbtest
async def test_header_less_request_is_allowed(client):
    """Pins the documented CSRF decision: curl and the CLI smoke paths send
    neither Origin nor Referer, and blocking them would cost tooling without
    stopping a browser-driven attack (browsers always send Origin cross-site)."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        "/admin/users/bulk",
        data={"action": "suspend", "user_ids": [str(target_id)]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert (await _fetch(target_id)).status == "suspended"


@dbtest
async def test_same_origin_referer_is_accepted(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        "/admin/users/bulk",
        data={"action": "suspend", "user_ids": [str(target_id)]},
        headers={"Referer": "http://testserver/admin/users?status=approved"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert (await _fetch(target_id)).status == "suspended"


@dbtest
async def test_origin_suffix_lookalike_is_rejected(client):
    """https://<real-host>.evil.com must not pass a startswith-style check."""
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    for bad in ("http://testserver.evil.com", "null", "http://evil.com"):
        r = client.post(
            "/admin/users/bulk",
            data={"action": "suspend", "user_ids": [str(target_id)]},
            headers={"Origin": bad},
            follow_redirects=False,
        )
        assert r.status_code == 403, bad
        assert (await _fetch(target_id)).status == "approved", bad
