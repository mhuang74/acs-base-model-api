"""User tags: normalisation, the service helpers, and the admin surface (ACS-371)."""

from __future__ import annotations

import os
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import SignupInvite, User, UserTag
from wrapper.services import user_tags
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run user-tag tests",
)


# ---- normalize_tag (pure, always runs) --------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hiring-2026-08", "hiring-2026-08"),
        ("Hiring 2026 08", "hiring-2026-08"),  # spaces + case
        ("  internal  ", "internal"),
        ("HAAISS_summer", "haaiss-summer"),  # underscores are not in the alphabet
        ("a--b", "a-b"),  # dash runs collapse
        ("-lead-trail-", "lead-trail"),
        ("béta", "b-ta"),  # non-ascii becomes a separator, not silently dropped
        ("", None),
        ("   ", None),
        ("---", None),  # nothing but separators
        ("!!!", None),
        (None, None),
    ],
)
def test_normalize_tag(raw, expected):
    assert user_tags.normalize_tag(raw) == expected


def test_normalize_tag_truncates_without_a_trailing_dash():
    """A cut landing on a dash must not leave one dangling."""
    raw = "a" * 31 + "-" + "b" * 10
    out = user_tags.normalize_tag(raw)
    assert out is not None
    assert len(out) <= user_tags.MAX_TAG_LEN
    assert not out.endswith("-")


# ---- DB helpers -------------------------------------------------------------


def _ue() -> str:
    return f"tag-{uuid.uuid4().hex[:8]}@example.local"


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


@pytest.fixture
def client():
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-user-tags-secret")
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


def _login(client: TestClient, email: str, password: str = "test-pw-12345") -> None:
    r = client.post("/login", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, f"login failed: {r.status_code} {r.text[:200]}"


# ---- service helpers --------------------------------------------------------


@dbtest
async def test_add_tag_is_idempotent():
    """Re-tagging an already-tagged user is a no-op, not an IntegrityError.

    The bulk path's normal case is a selection that partly overlaps the tag
    already, so ON CONFLICT DO NOTHING is load-bearing, not a nicety.
    """
    uid, _ = await _make_user()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert await user_tags.add_tag(s, [uid], "cohort-a") == 1
        async with session_scope(factory) as s:
            assert await user_tags.add_tag(s, [uid], "cohort-a") == 0
        async with session_scope(factory) as s:
            rows = (await s.execute(select(UserTag).where(UserTag.user_id == uid))).scalars().all()
            assert len(list(rows)) == 1
    finally:
        await engine.dispose()


@dbtest
async def test_load_tags_and_remove():
    uid1, _ = await _make_user()
    uid2, _ = await _make_user()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            await user_tags.add_tag(s, [uid1, uid2], "shared")
            await user_tags.add_tag(s, [uid1], "only-one")
        async with session_scope(factory) as s:
            m = await user_tags.load_tags_for_users(s, [uid1, uid2])
            assert m[uid1] == ["only-one", "shared"]  # sorted
            assert m[uid2] == ["shared"]
            # Empty input short-circuits rather than emitting WHERE IN ()
            assert await user_tags.load_tags_for_users(s, []) == {}
        async with session_scope(factory) as s:
            assert await user_tags.remove_tag(s, [uid1, uid2], "shared") == 2
        async with session_scope(factory) as s:
            m = await user_tags.load_tags_for_users(s, [uid1, uid2])
            assert m[uid1] == ["only-one"]
            assert uid2 not in m  # untagged users are absent, not empty-listed
    finally:
        await engine.dispose()


@dbtest
async def test_deleting_a_user_cascades_their_tags():
    uid, _ = await _make_user()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            await user_tags.add_tag(s, [uid], "doomed")
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.id == uid))).scalar_one()
            await s.delete(u)
        async with session_scope(factory) as s:
            rows = (await s.execute(select(UserTag).where(UserTag.user_id == uid))).scalars().all()
            assert list(rows) == []
    finally:
        await engine.dispose()


@dbtest
async def test_removing_the_tagging_admin_keeps_the_tag():
    """created_by_user_id is SET NULL — a tag describes the tagged user, not
    the admin who applied it."""
    tagger_id, _ = await _make_user(role="admin")
    uid, _ = await _make_user()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            await user_tags.add_tag(s, [uid], "survives", created_by_user_id=tagger_id)
        async with session_scope(factory) as s:
            a = (await s.execute(select(User).where(User.id == tagger_id))).scalar_one()
            await s.delete(a)
        async with session_scope(factory) as s:
            row = (await s.execute(select(UserTag).where(UserTag.user_id == uid))).scalar_one()
            assert row.tag == "survives"
            assert row.created_by_user_id is None
    finally:
        await engine.dispose()


# ---- admin surface ----------------------------------------------------------


@dbtest
async def test_admin_can_add_and_remove_a_tag(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        f"/admin/users/{target_id}/tags/add",
        data={"tag": "Hiring 2026 08"},  # normalised on the way in
        follow_redirects=False,
    )
    assert r.status_code == 303

    # Assert on the chip markup, not a bare substring: the section's help text
    # uses `hiring-2026-08` as its worked example, so a substring check would
    # pass whether or not the tag was ever applied — in BOTH directions.
    chip = '<span class="tag-chip">hiring-2026-08'
    assert chip in client.get(f"/admin/users/{target_id}").text

    r = client.post(
        f"/admin/users/{target_id}/tags/remove",
        data={"tag": "hiring-2026-08"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert chip not in client.get(f"/admin/users/{target_id}").text


@dbtest
async def test_unusable_tag_is_rejected_with_a_message(client):
    _, admin_email = await _make_user(role="admin")
    target_id, _ = await _make_user()
    _login(client, admin_email)

    r = client.post(
        f"/admin/users/{target_id}/tags/add", data={"tag": "!!!"}, follow_redirects=False
    )
    assert r.status_code == 400
    assert "at least one letter or digit" in r.text


@dbtest
async def test_roster_filters_by_tag_and_status(client):
    _, admin_email = await _make_user(role="admin")
    tagged_id, tagged_email = await _make_user()
    _, plain_email = await _make_user()
    suspended_id, suspended_email = await _make_user()
    _login(client, admin_email)

    client.post(f"/admin/users/{tagged_id}/tags/add", data={"tag": "cohort-x"})
    client.post(f"/admin/users/{suspended_id}/suspend", data={"reason": "t"})

    r = client.get("/admin/users?tag=cohort-x")
    assert tagged_email in r.text
    assert plain_email not in r.text

    r = client.get("/admin/users?status=suspended")
    assert suspended_email in r.text
    assert tagged_email not in r.text

    # An unknown status is ignored rather than returning an empty roster.
    r = client.get("/admin/users?status=bogus")
    assert tagged_email in r.text


@dbtest
async def test_total_users_stat_is_not_the_filtered_count(client):
    """stats.total_users was len(all_users); with a filter that would silently
    start reporting the filter's size instead of the roster's."""
    _, admin_email = await _make_user(role="admin")
    tagged_id, _ = await _make_user()
    await _make_user()
    _login(client, admin_email)
    # Unique per run: the test DB is reused across runs, so a fixed tag would
    # accumulate users and the "1 of N" would climb with every invocation.
    tag = f"solo-{uuid.uuid4().hex[:8]}"
    client.post(f"/admin/users/{tagged_id}/tags/add", data={"tag": tag})

    # Assert the invariant, not an exact total: the roster count drifts as other
    # tests add users. What matters is that the header reports
    # "<filtered> of <roster>" with the roster strictly larger — i.e.
    # total_users is a roster-wide count, not len(the filtered list).
    r = client.get(f"/admin/users?tag={tag}")
    m = re.search(r"\((\d+) of (\d+)\)", r.text)
    assert m, "filtered header not rendered"
    shown, total = int(m.group(1)), int(m.group(2))
    assert shown == 1, f"expected exactly the one tagged user, got {shown}"
    assert total > shown, "total_users is reporting the filtered count"


@dbtest
async def test_invite_tag_autotags_the_claiming_account(client):
    """A cohort labels itself at claim time — the whole point of invite tags."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "5", "tag": "auto-cohort"},
        follow_redirects=False,
    )
    assert r.status_code == 200

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(
                    select(SignupInvite)
                    .where(SignupInvite.tag == "auto-cohort")
                    .order_by(SignupInvite.created_at.desc())
                    .limit(1)
                )
            ).scalar_one()
            token = inv.token
    finally:
        await engine.dispose()

    client.cookies.clear()
    email = _ue()
    r = client.post(
        f"/invite/{token}",
        data={
            "email": email,
            "name": "Claimer",
            "org": "Independent",
            "password": "claim-pw-12345",
            "confirm_password": "claim-pw-12345",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one()
            tags = (await user_tags.load_tags_for_users(s, [u.id])).get(u.id, [])
            assert tags == ["auto-cohort"]
    finally:
        await engine.dispose()


@dbtest
async def test_untagged_invite_leaves_the_account_untagged(client):
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "1", "tag": ""},
        follow_redirects=False,
    )
    assert r.status_code == 200
    token = r.text.split("/invite/")[-1].split("<")[0].strip()

    client.cookies.clear()
    email = _ue()
    r = client.post(
        f"/invite/{token}",
        data={
            "email": email,
            "name": "Plain",
            "org": "Independent",
            "password": "claim-pw-12345",
            "confirm_password": "claim-pw-12345",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one()
            assert (await user_tags.load_tags_for_users(s, [u.id])) == {}
    finally:
        await engine.dispose()


@dbtest
async def test_invite_forms_carry_the_active_sort(client):
    """Creating an invite must not reset the roster ordering (ACS-371).

    The invite forms post to a bare ``/admin/users/invite`` with no query
    string, so ``request.query_params`` is empty on that POST — the sort has to
    ride along as a form field or it is simply lost. (An earlier version of this
    test asserted that "Tokens (30d)" appeared in the response, which is a
    static <th> present on every render and therefore proved nothing.)

    Deliberately only ``sort``: carrying ``status``/``tag`` too could re-render
    a filtered roster that hides the invite just created.
    """
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.get("/admin/users?sort=tokens&status=approved")
    assert r.status_code == 200
    assert '<input type="hidden" name="sort" value="tokens">' in r.text
    # One per invite form (link / email / csv).
    assert r.text.count('<input type="hidden" name="sort" value="tokens">') == 3
    # ...and the filters are NOT carried into the POST.
    assert 'name="status" value="approved"' not in r.text

    # Unsorted roster emits no hidden field at all.
    r = client.get("/admin/users")
    assert 'type="hidden" name="sort"' not in r.text


@dbtest
async def test_invite_post_honours_the_submitted_sort(client):
    """The handler reads the sort off the form, not the URL."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "1", "sort": "tokens"},
        follow_redirects=False,
    )
    assert r.status_code == 200
    # The re-render round-trips the sort into its own forms, proving the value
    # reached the shared context builder rather than being dropped.
    assert '<input type="hidden" name="sort" value="tokens">' in r.text


@dbtest
async def test_email_and_csv_invites_also_carry_the_tag(client):
    """The link branch is the one the tag test exercises, but the hiring flow
    actually uses the email/CSV branches — all three must thread the tag."""
    _, admin_email = await _make_user(role="admin")
    _login(client, admin_email)

    target = f"invitee-{uuid.uuid4().hex[:8]}@example.local"
    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "email", "emails": target, "tag": "Email Cohort"},
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
            assert inv.tag == "email-cohort"
    finally:
        await engine.dispose()


@dbtest
async def test_autotag_is_attributed_to_the_invite_creator(client):
    """The tag records the admin who created the invite, not the claimer."""
    admin_id, admin_email = await _make_user(role="admin")
    _login(client, admin_email)
    tag = f"attrib-{uuid.uuid4().hex[:6]}"
    r = client.post(
        "/admin/users/invite",
        data={"invite_type": "link", "max_uses": "1", "tag": tag},
        follow_redirects=False,
    )
    assert r.status_code == 200

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            inv = (
                await s.execute(select(SignupInvite).where(SignupInvite.tag == tag))
            ).scalar_one()
            token = inv.token
    finally:
        await engine.dispose()

    client.cookies.clear()
    email = _ue()
    r = client.post(
        f"/invite/{token}",
        data={
            "email": email,
            "name": "Claimer",
            "org": "Independent",
            "password": "claim-pw-12345",
            "confirm_password": "claim-pw-12345",
            "agree": "yes",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            u = (await s.execute(select(User).where(User.email == email))).scalar_one()
            row = (await s.execute(select(UserTag).where(UserTag.user_id == u.id))).scalar_one()
            assert row.tag == tag
            assert row.created_by_user_id == admin_id, "attributed to the claimer, not the inviter"
    finally:
        await engine.dispose()


@dbtest
async def test_add_tag_tolerates_duplicate_ids_and_empty_input():
    """PR3's bulk path will pass a selection that may repeat ids; ON CONFLICT
    DO NOTHING handles intra-statement duplicates (DO UPDATE would not)."""
    uid, _ = await _make_user()
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert await user_tags.add_tag(s, [], "nobody") == 0
            assert await user_tags.remove_tag(s, [], "nobody") == 0
            assert await user_tags.add_tag(s, [uid, uid, uid], "dupes") == 1
        async with session_scope(factory) as s:
            rows = (await s.execute(select(UserTag).where(UserTag.user_id == uid))).scalars().all()
            assert len(list(rows)) == 1
    finally:
        await engine.dispose()


# ---- the `internal` tag replaces the domain heuristic (ACS-374) --------------


def test_metabase_sql_excludes_internal_by_tag_not_by_domain():
    """The dashboard SQL is applied by hand into hosted Metabase, so an error
    here ships silently as wrong numbers. Pin the exclusion mechanism.

    Two traps this guard has to avoid falling into:

    - ``sql.count("t.tag = 'internal'") >= 5`` would pass with every active
      site commented out — which is exactly the edit that silently ships wrong
      numbers. Match on line shape and count active vs commented separately.
    - Substring-matching ``"@acsresearch.org'"`` only catches the old rule
      written with that exact trailing quote; ``ILIKE``, ``LIKE '%…%'`` or a
      regex would sail past. Assert no *code* line mentions the domain at all.
    """
    import re
    from pathlib import Path as _P

    sql = (
        _P(__file__).resolve().parents[2] / "docs" / "runbooks" / "metabase-dashboard-queries.sql"
    ).read_text()

    active = re.findall(
        r"^\s*AND NOT EXISTS \(SELECT 1 FROM user_tags t .*t\.tag = 'internal'", sql, re.M
    )
    commented = re.findall(
        r"^\s*-- AND NOT EXISTS \(SELECT 1 FROM user_tags t .*t\.tag = 'internal'", sql, re.M
    )
    # 3 active (B8, C1, C2) + 2 opt-in (B2, B7) — the ratio is the contract.
    assert (len(active), len(commented)) == (3, 2), (
        f"expected 3 active / 2 commented tag exclusions, got {len(active)}/{len(commented)}"
    )

    # No *code* line may mention the team domain. (The header prose still refers
    # to the old rule by name, which is fine and deliberate.)
    offenders = [
        ln
        for ln in sql.splitlines()
        if "acsresearch.org" in ln and not ln.lstrip().startswith("--")
    ]
    assert not offenders, f"an email-domain exclusion crept back in: {offenders}"
    assert "personal-test@example.com" not in sql, "the uncommittable escape hatch is back"


def test_roster_template_has_no_domain_check():
    """The roster's `internal` pill came from a Jinja endswith; it is now an
    ordinary tag chip, so the heuristic must be gone from the template too."""
    from pathlib import Path

    tpl = (
        Path(__file__).resolve().parents[1] / "src" / "wrapper" / "templates" / "admin_users.html"
    ).read_text()
    assert "acsresearch.org" not in tpl


@dbtest
async def test_internal_tag_is_editable_like_any_other(client):
    """The point of the change: membership is now curated, so an account
    outside the domain can be marked internal and one inside it can be
    un-marked — neither was expressible before."""
    _, admin_email = await _make_user(role="admin")
    outsider_id, _ = await _make_user()  # @example.local, not the team domain
    _login(client, admin_email)

    r = client.post(
        f"/admin/users/{outsider_id}/tags/add", data={"tag": "internal"}, follow_redirects=False
    )
    assert r.status_code == 303

    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert (await user_tags.load_tags_for_users(s, [outsider_id]))[outsider_id] == [
                "internal"
            ]
    finally:
        await engine.dispose()

    r = client.post(
        f"/admin/users/{outsider_id}/tags/remove",
        data={"tag": "internal"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    engine = make_engine(TEST_DATABASE_URL)
    try:
        factory = make_session_factory(engine)
        async with session_scope(factory) as s:
            assert await user_tags.load_tags_for_users(s, [outsider_id]) == {}
    finally:
        await engine.dispose()
