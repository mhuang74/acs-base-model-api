"""ACS-89: multi-page /tutorial docs site.

The /tutorial surface used to be a single 480-line template. After ACS-89 it's
a Modal-style three-column docs site (left nav / content / right TOC) built
from markdown files under ``wrapper/docs/`` and pre-rendered at boot. These
tests pin the URL contract + the no-authed-link-leak invariant from ACS-68.

DB plumbing: the wrapper's lifespan touches Postgres at startup, so the tests
need ``TEST_DATABASE_URL`` pointed at a migrated database. The route handlers
themselves are DB-free, so the ``_make_user`` helper only kicks in when a test
needs an authed view.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.models import User
from wrapper.web_auth import hash_password

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run /tutorial tests",
)

# Substring assertions over the rendered HTML — keep them anchored to copy
# that is unique to one branch of the {% if user %} so flipping the wrong
# way fails the test rather than passing trivially.
BETA_BANNER_SUBSTR = "Private beta"
# NB: the plain string "Go to Dashboard" also appears in a CSS comment in
# tutorial_layout.html, so the *absence* assertion needs the markup itself.
AUTHED_CTA_SUBSTR = '<p class="tutorial-authed-cta">'

# The 7 worked-example slugs the site must serve. Anchor IDs on the OLD
# single-page tutorial used these too; the legacy-fragment redirect map
# depends on the contract holding.
EXAMPLE_SLUGS = (
    "logprobs",
    "prompt-logprobs",
    "echo",
    "stream",
    "batch-rollouts",
    "cold-boot",
    "budget-cap",
)

# Top-level pages that must render. Overview lives at /tutorial AND at
# /tutorial/overview (the canonical aliased slug); the latter is checked
# separately so we know both URLs work.
TOP_LEVEL_PATHS = (
    "/tutorial",
    "/tutorial/overview",
    "/tutorial/models",
    "/tutorial/api",
    "/tutorial/account",
    "/tutorial/examples",
)


@pytest.fixture
def client():
    """FastAPI TestClient with lifespan started + minimum env vars set."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-tutorial-public-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("RATE_LIMIT_LOGIN_PER_IP", "1000/minute")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    # Pin explicitly: these tests assert the private-beta banner, and
    # test_signup.py leaks SIGNUP_ENABLED=true into the process (ACS-308).
    os.environ["SIGNUP_ENABLED"] = "false"
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
        email = f"tut-{uuid.uuid4().hex[:8]}@example.local"
        async with session_scope(factory) as s:
            u = User(email=email, password_hash=hash_password(password), status="approved")
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


# ---- 1. /tutorial root serves Overview -------------------------------------

@dbtest
def test_tutorial_root_serves_overview(client):
    """GET /tutorial → 200, Overview content + the legacy redirect script."""
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    assert "Quick start" in body, "Overview must include the Quick start section"
    # The fragment-redirect script is only emitted on the Overview route —
    # its presence here doubles as a load-bearing check that Overview rendered.
    assert "location.replace" in body, "fragment-redirect script missing from /tutorial"


# ---- 2 + 3. Beta banner shown/hidden based on auth -------------------------

@dbtest
def test_tutorial_root_shows_beta_banner_unauthed(client):
    """Anonymous viewer sees the 'private beta' banner."""
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    assert BETA_BANNER_SUBSTR in r.text
    # And the authed-only CTA must NOT leak.
    assert AUTHED_CTA_SUBSTR not in r.text


@dbtest
async def test_tutorial_root_hides_banner_authed(client):
    """Logged-in viewer: no banner, dashboard CTA visible instead."""
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    assert BETA_BANNER_SUBSTR not in r.text
    assert AUTHED_CTA_SUBSTR in r.text


# ---- 4. ACS-68 regression pin: /tutorial does NOT 303 unauthed -------------

@dbtest
def test_tutorial_root_does_not_redirect_unauthed(client):
    """Regression: the pre-ACS-68 code 303'd unauthed users to /login."""
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code != 303, "tutorial must remain public for prospective collaborators"
    assert r.status_code == 200


# ---- 5. Examples index lists all 7 sub-pages -------------------------------

@dbtest
def test_examples_index_lists_all_seven(client):
    """GET /tutorial/examples → 200, with an href for each sub-page."""
    r = client.get("/tutorial/examples", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    for slug in EXAMPLE_SLUGS:
        href = f'/tutorial/examples/{slug}'
        assert href in body, f"Examples index is missing a link to {href}"


# ---- 6. Each example page renders ------------------------------------------

@dbtest
@pytest.mark.parametrize("slug", EXAMPLE_SLUGS)
def test_each_example_page_renders(client, slug):
    """Each /tutorial/examples/{slug} returns 200 with content + a code block."""
    r = client.get(f"/tutorial/examples/{slug}", follow_redirects=False)
    assert r.status_code == 200, f"/tutorial/examples/{slug} should return 200"
    body = r.text
    # Body must contain at least one fenced code block (every example does).
    assert "<pre><code" in body, f"{slug}: expected a <pre><code …> code block in the page body"


# ---- 7. Heading anchor IDs survive on each example page --------------------

@dbtest
@pytest.mark.parametrize("slug", EXAMPLE_SLUGS)
def test_example_pages_have_heading_anchor_ids(client, slug):
    """Each example page must surface ``id="<slug>"`` for cross-page deep links.

    Each example markdown carries a `## <slug-as-heading>` so that the
    anchors_plugin emits ``id="<slug>"``. The TOC + the right rail's
    IntersectionObserver enhancer both depend on this contract.
    """
    r = client.get(f"/tutorial/examples/{slug}", follow_redirects=False)
    assert r.status_code == 200
    # H2/H3 anchor IDs are slug-derived; at least ONE id on the page must
    # match a known anchor name. For these pages the slug doesn't always
    # appear as an H2 (e.g. "logprobs" page has H1 "logprobs" with H2 "curl"
    # / "Python" / "Gotcha"). So we check for a stable set of section anchors.
    body = r.text
    # Derive the expected anchors from the page's own H2s rather than
    # hardcoding names: the contract under test is "every H2 gets a slugified
    # id", and example pages legitimately differ (cold-boot has no "Gotcha").
    from wrapper.docs_build import _slugify, docs_root_default

    md = (docs_root_default() / "examples" / f"{slug}.md").read_text(encoding="utf-8")
    # Track fences: a "## " line inside a code block is sample text, not a
    # heading, and would become a phantom expected anchor.
    h2s, in_fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("## "):
            h2s.append(line[3:].strip())
    assert h2s, f"{slug}: fixture problem — no H2 headings in the source markdown"
    derived = {_slugify(h) for h in h2s}
    assert {"curl", "python"} <= derived, (
        f"{slug}: expected curl + Python sections (kept from the previous "
        f"hardcoded assertion); got {sorted(derived)}"
    )
    for heading in h2s:
        anchor = _slugify(heading)
        assert f'id="{anchor}"' in body, (
            f'{slug}: missing id="{anchor}" for H2 "{heading}" — anchors_plugin '
            f"should slugify every H2 heading into a stable id"
        )


# ---- 8. Legacy fragment map present on the root page -----------------------

@dbtest
def test_legacy_fragment_map_present_on_root(client):
    """The fragment-redirect map must list every legacy in-page anchor.

    Old wiki / Slack links point at ``/tutorial#logprobs`` etc.; the redirect
    script forwards them to the new per-page URL before first paint.
    """
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    for fragment in (*EXAMPLE_SLUGS, "quick-start"):
        # The map renders as a JSON object literal; the keys are bare
        # double-quoted strings (json-encoded), and the values point at the
        # new path. We only check key membership — a future schema change
        # could re-shape the values, but the legacy keys are the contract.
        assert f'"{fragment}"' in body, (
            f"legacy fragment {fragment!r} missing from the /tutorial "
            "redirect map — wiki/Slack links to /tutorial#"
            f"{fragment} would break silently"
        )


# ---- 9. Left nav links every top-level page --------------------------------

@dbtest
def test_left_nav_links_all_top_level_pages(client):
    """Sidebar HTML must contain an href for each top-level nav entry."""
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    # Overview's canonical URL is /tutorial (no slug); everything else uses
    # the slug as a path. Match against the sidebar's exact href format.
    expected_hrefs = (
        'href="/tutorial"',          # Overview
        'href="/tutorial/models"',
        'href="/tutorial/api"',
        'href="/tutorial/account"',
        'href="/tutorial/examples"',
    )
    for href in expected_hrefs:
        assert href in body, f"left nav is missing {href}"


# ---- 10. Right TOC contains H2+H3 but excludes H1 and H4 -------------------

@dbtest
def test_right_toc_contains_h2_h3_only(client):
    """The right rail surfaces H2/H3 anchors; the H1 (page title) is excluded.

    The /tutorial/api page has both H2 ("Errors") and H2 ("Limits") sections
    plus the implicit H1 ("API reference"). The TOC must list the H2s and
    NOT the H1.
    """
    r = client.get("/tutorial/api", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    # "On this page" only appears in the right TOC partial.
    assert "On this page" in body, "right-rail TOC missing from /tutorial/api"
    # H2 anchor present → it's in the TOC.
    assert 'href="#errors"' in body, "Errors H2 should appear in the right TOC"
    assert 'href="#limits"' in body, "Limits H2 should appear in the right TOC"
    # H1 ("API reference") must NOT be in the TOC. We can't grep for the text
    # alone (it appears in <title> and the H1 itself), so we check that no
    # TOC link has the slugified H1's anchor — anchors_plugin only attaches
    # ids to H2/H3 (max_level=3) so #api-reference shouldn't even exist.
    assert 'href="#api-reference"' not in body, "H1 leaked into the right TOC"


# ---- 11. Unauthed view does NOT leak authed-only links ---------------------
#
# ACS-68 / PR-36 finding 5: the unauthed /tutorial must not link to routes
# that 303→/login for anonymous viewers (the link is dead, and worse, telling
# the user "go to your dashboard" without an account is just confusing).
# We parametrise across every public docs path to catch a regression in any
# markdown file, not just the Overview page.

@dbtest
@pytest.mark.parametrize(
    "path",
    [
        "/tutorial",
        "/tutorial/overview",
        "/tutorial/models",
        "/tutorial/api",
        "/tutorial/account",
        "/tutorial/examples",
        *(f"/tutorial/examples/{slug}" for slug in EXAMPLE_SLUGS),
    ],
)
def test_unauthed_does_not_leak_authed_only_links(client, path):
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 200, f"{path} should return 200 for unauthed viewers"
    body = r.text
    for href in ('href="/dashboard"', 'href="/workbench"'):
        assert href not in body, (
            f"{path}: authed-only link {href!r} leaked into the unauthed "
            f"view. The base.html nav guards these with {{% if user %}}, "
            f"so the leak is coming from inside the docs markdown — drop "
            f"the link or rephrase as prose."
        )


@dbtest
async def test_authed_view_keeps_authed_only_links(client):
    """Companion to #11: signed-in users SEE the in-app links (in the nav)."""
    _, email = await _make_user()
    _login(client, email, "test-pw-12345")
    r = client.get("/tutorial", follow_redirects=False)
    assert r.status_code == 200
    body = r.text
    # base.html's signed-in nav contains both links; the docs body's
    # "Go to Dashboard →" CTA also surfaces /dashboard.
    assert 'href="/dashboard"' in body
    assert 'href="/workbench"' in body


# ---- 12. Unknown slugs 404 -------------------------------------------------

@dbtest
def test_unknown_slug_404s(client):
    """Mistyped top-level + example slugs both 404 (not 500, not redirect)."""
    r = client.get("/tutorial/does-not-exist", follow_redirects=False)
    assert r.status_code == 404, f"unknown top-level slug should 404, got {r.status_code}"
    r2 = client.get("/tutorial/examples/does-not-exist", follow_redirects=False)
    assert r2.status_code == 404, f"unknown example slug should 404, got {r2.status_code}"


# ---- Bonus: /tutorial/quick-start 301 redirect alias ----------------------

@dbtest
def test_quick_start_alias_301s(client):
    """``/tutorial/quick-start`` (literal path) → 301 → ``/tutorial/overview#quick-start``.

    Matters because some inbound links use it as a path component rather than
    a fragment. ``follow_redirects=False`` is critical here — TestClient
    auto-follows by default and would make us assert 200 instead of 301.
    """
    r = client.get("/tutorial/quick-start", follow_redirects=False)
    assert r.status_code == 301, f"expected 301, got {r.status_code}"
    assert r.headers["location"] == "/tutorial/overview#quick-start"


# ---- /llms.txt single-page combined docs (for LLMs) -----------------------

@dbtest
def test_llms_txt_serves_combined_markdown(client):
    """``/llms.txt`` returns the whole tutorial as one markdown doc, public,
    so a human can hand their assistant a single URL."""
    r = client.get("/llms.txt", follow_redirects=False)
    assert r.status_code == 200, f"expected 200, got {r.status_code}"
    assert r.headers["content-type"].startswith("text/markdown")
    body = r.text
    assert body.startswith("# ACS Infra — base-model API")
    # Pulls from multiple source pages (overview + api reference).
    assert "Quick start" in body
    assert "What's not supported" in body
    assert "{{API_BASE}}" not in body  # placeholder resolved at boot
