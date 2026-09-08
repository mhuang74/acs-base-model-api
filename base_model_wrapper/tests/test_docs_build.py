"""Unit tests for the boot-time docs renderer (ACS-89), focused on the
configurable API base: the ``{{API_BASE}}`` placeholder in the markdown must be
substituted with the deployed ``PUBLIC_BASE_URL`` + ``/v1`` at build time, so a
domain move is a single env-var change with no hardcoded host in the docs.

DB-free: ``build_docs`` is pure (markdown in → rendered HTML out), so these run
without a Postgres.
"""

from __future__ import annotations

from wrapper.docs_build import (
    build_combined_markdown,
    build_docs,
    docs_root_default,
)


def test_build_docs_substitutes_api_base():
    """The Quick-start export line follows the provided api_base."""
    pages, _nav = build_docs(docs_root_default(), api_base="https://example.test/v1")

    # (markdown escapes the quotes in code blocks to &quot;, so assert on the URL)
    overview = pages["overview"].body_html
    assert "https://example.test/v1" in overview, (
        "overview Quick start should render the configured api_base"
    )
    # The placeholder must be fully resolved...
    assert "{{API_BASE}}" not in overview
    # ...and the previously-hardcoded host must not reappear anywhere in the docs.
    for page in pages.values():
        assert "base-models.acsresearch.org/v1" not in page.body_html


def test_build_docs_api_base_tracks_a_different_domain():
    """A different PUBLIC_BASE_URL (e.g. the upcoming infra.acsresearch.org)
    flows straight through — proves it's a one-knob change."""
    pages, _nav = build_docs(
        docs_root_default(), api_base="https://infra.acsresearch.org/v1"
    )
    assert "https://infra.acsresearch.org/v1" in pages["overview"].body_html


def test_build_docs_site_base_is_api_base_without_v1():
    """``{{SITE_BASE}}`` resolves to the site root (api_base minus ``/v1``), so
    non-API links like the /llms.txt pointer render as absolute, paste-able URLs
    on the configured domain — not under ``/v1`` and not a stray placeholder."""
    pages, _nav = build_docs(
        docs_root_default(), api_base="https://infra.acsresearch.org/v1"
    )
    overview = pages["overview"].body_html
    assert "https://infra.acsresearch.org/llms.txt" in overview
    # The site-root link must NOT inherit the API's ``/v1`` suffix...
    assert "/v1/llms.txt" not in overview
    # ...and the placeholder must be fully resolved on every page.
    for page in pages.values():
        assert "{{SITE_BASE}}" not in page.body_html


def test_build_combined_markdown_covers_all_pages_and_substitutes():
    """The /llms.txt combined doc includes every page (in nav order) and
    substitutes the api_base from the same source — no second copy to sync."""
    md = build_combined_markdown(docs_root_default(), api_base="https://example.test/v1")

    assert md.startswith("# ACS Infra — base-model API")
    for slug in (
        "overview",
        "models",
        "api",
        "account",
        "examples",
        "examples/logprobs",
        "examples/cold-boot",
    ):
        assert f"docs/{slug}.md" in md, f"combined doc missing section for {slug}"
    # api_base substituted, placeholder gone, old host absent.
    assert "https://example.test/v1" in md
    assert "{{API_BASE}}" not in md
    assert "base-models.acsresearch.org/v1" not in md
