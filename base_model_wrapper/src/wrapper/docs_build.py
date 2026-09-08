"""Boot-time markdown → HTML renderer for the multi-page /tutorial site.

The wrapper renders the docs site from a tree of markdown files shipped inside
the wheel (``wrapper/docs/*.md`` and ``wrapper/docs/examples/*.md``) once at
startup. Each rendered page exposes:

* ``slug``     — the URL slug (``"overview"``, ``"examples/logprobs"``)
* ``title``    — extracted from the first H1 in the source
* ``body_html``— the rendered HTML (with ``id=`` anchors on H2/H3)
* ``toc``      — a list of ``TocNode``s (H2/H3 only) for the right-rail "On this page"

The left-rail nav is a fixed ``_NAV_ORDER`` — we don't infer the page tree from
the filesystem because Overview / Models / API reference / Account / Examples
have an editorial order and a parent-child relationship (Examples → 7 sub-pages)
that an alphabetic walk wouldn't capture.

Slug parity: the previous single-page /tutorial used hand-written
``id="prompt-logprobs"`` anchors (hyphenated, even though the API parameter is
``prompt_logprobs``). The hashes ``/tutorial#prompt-logprobs`` etc. are part
of the public contract — the legacy-fragment redirect script depends on them.
We pin that with an import-time assertion on ``_slugify`` so a future tweak to
the slug function can't silently break the redirect map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin


@dataclass(slots=True)
class TocNode:
    """One entry in the right-rail "On this page" table of contents."""

    level: int  # 2 or 3
    id: str
    text: str
    children: list["TocNode"] = field(default_factory=list)


@dataclass(slots=True)
class RenderedDoc:
    """One rendered markdown page."""

    slug: str
    title: str
    body_html: str
    toc: list[TocNode]


@dataclass(slots=True)
class NavEntry:
    """One entry in the left-rail nav tree."""

    slug: str
    title: str
    children: list["NavEntry"] = field(default_factory=list)


# Hard-coded left-nav order. Each tuple is (slug, default_title, children).
# Titles get overridden by the H1 of the corresponding markdown file when it
# renders successfully — the default here is the fallback when a file is
# missing (in which case we'd still want a sensible label, though build_docs
# raises rather than silently skipping).
_NAV_ORDER: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("overview", "Overview", ()),
    ("models", "Models", ()),
    ("api", "API reference", ()),
    ("activation-harvesting", "Activation harvesting", ()),
    ("bulk-harvest", "Bulk harvest", ()),
    ("activation-steering", "Activation steering", ()),
    ("account", "Account", ()),
    (
        "workbench",
        "Workbench",
        (
            ("workbench/loom", "Loom"),
            ("workbench/compare", "Compare mode"),
        ),
    ),
    (
        "examples",
        "Examples",
        (
            ("examples/logprobs", "logprobs"),
            ("examples/prompt-logprobs", "prompt_logprobs"),
            ("examples/echo", "echo"),
            ("examples/stream", "stream"),
            ("examples/batch-rollouts", "Batch rollouts"),
            ("examples/evaluate-with-inspect", "Evaluate with Inspect"),
            ("examples/cold-boot", "Cold-boot waiting"),
            ("examples/budget-cap", "Budget-cap recovery"),
        ),
    ),
)


def _slugify(text: str) -> str:
    """Lowercase → hyphenated slug, used as both heading id and URL fragment.

    The anchors_plugin uses this for every H2/H3 ``id=``. Keep behaviour
    stable: ``prompt_logprobs`` (the API param name) and ``Prompt logprobs``
    (the section heading text in the markdown) must both produce
    ``prompt-logprobs`` so legacy ``/tutorial#prompt-logprobs`` URLs keep
    pointing at the right anchor.
    """
    s = text.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


# Slug-parity assertion (see module docstring). If this ever fails the legacy
# fragment-redirect map in routes/web.py silently breaks.
assert _slugify("prompt_logprobs") == "prompt-logprobs", (
    "slug parity broken: prompt_logprobs must slugify to prompt-logprobs "
    "(legacy /tutorial#prompt-logprobs URLs depend on it)"
)
assert _slugify("Quick start") == "quick-start"
assert _slugify("Cold-boot waiting") == "cold-boot-waiting"


def _build_md() -> MarkdownIt:
    """Construct the shared MarkdownIt instance.

    * ``commonmark`` base (predictable, no quirks of GFM by default)
    * ``html: true`` so the markdown can drop in raw ``<span>``/``<sup>`` etc.
    * ``linkify`` auto-links bare URLs
    * ``table`` extension for the Models + Errors tables
    * ``anchors_plugin`` attaches ``id=`` to every H2/H3 (matches the
       hand-written anchors in the legacy single-page tutorial)
    """
    md = MarkdownIt("commonmark", {"html": True, "linkify": True})
    md.enable("table")
    md.use(anchors_plugin, max_level=3, slug_func=_slugify)
    return md


def _extract_title_and_toc(
    md: MarkdownIt, source: str
) -> tuple[str, list[TocNode]]:
    """Walk the token stream once: extract H1 title + H2/H3 TOC tree.

    H1s are the page title (one per file by convention; only the first counts
    if the author breaks the convention). H2/H3 become TocNodes. H4+ are
    ignored — the right rail is for top-level + one-deep navigation.
    """
    tokens = md.parse(source)
    title = ""
    toc: list[TocNode] = []
    last_h2: TocNode | None = None

    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        # The next token is the inline content; its .content is the heading text.
        inline = tokens[i + 1] if i + 1 < len(tokens) else None
        text = (inline.content if inline is not None else "").strip()
        if not text:
            continue
        level = int(tok.tag[1:])  # "h2" → 2
        if level == 1 and not title:
            title = text
            continue
        if level not in (2, 3):
            continue
        # anchors_plugin attaches the id via attrSet; fall back to the slug
        # function if for some reason the attr is missing.
        anchor_id = tok.attrGet("id") or _slugify(text)
        node = TocNode(level=level, id=anchor_id, text=text)
        if level == 2:
            toc.append(node)
            last_h2 = node
        else:  # level == 3
            (last_h2.children if last_h2 is not None else toc).append(node)

    return title, toc


def _render_one(md: MarkdownIt, source: str, slug: str) -> RenderedDoc:
    title, toc = _extract_title_and_toc(md, source)
    body_html = md.render(source)
    return RenderedDoc(
        slug=slug,
        title=title or slug,
        body_html=body_html,
        toc=toc,
    )


def _doc_paths(docs_root: Path) -> dict[str, Path]:
    """Map ``slug → absolute path`` for every markdown file under docs_root.

    Top-level: ``docs/foo.md`` → ``"foo"``.
    Sub-folder (one level): ``docs/examples/bar.md`` → ``"examples/bar"``,
    ``docs/workbench/loom.md`` → ``"workbench/loom"``. Any sub-folder is picked
    up, so adding a new section is just a folder + a ``_NAV_ORDER`` entry.
    """
    out: dict[str, Path] = {}
    for p in sorted(docs_root.glob("*.md")):
        out[p.stem] = p
    for sub in sorted(docs_root.iterdir()):
        if sub.is_dir():
            for p in sorted(sub.glob("*.md")):
                out[f"{sub.name}/{p.stem}"] = p
    return out


# Placeholder token substituted with the deployed API base (PUBLIC_BASE_URL +
# "/v1") at boot, so the documented endpoint follows the configured domain
# instead of being hardcoded. Lives in the markdown (e.g. overview.md's Quick
# start export line) so a domain move is a single env-var change.
_API_BASE_PLACEHOLDER = "{{API_BASE}}"

# Placeholder for the site *root* URL (PUBLIC_BASE_URL, i.e. ``api_base`` minus
# the trailing ``/v1``). Used for non-API links like ``{{SITE_BASE}}/llms.txt``
# that need an absolute, copy-pasteable URL — same single-knob domain tracking
# as ``{{API_BASE}}``, just without the ``/v1`` suffix.
_SITE_BASE_PLACEHOLDER = "{{SITE_BASE}}"


def _site_base(api_base: str) -> str:
    """Derive the site root from ``api_base`` by stripping the ``/v1`` suffix.

    ``api_base`` is always ``PUBLIC_BASE_URL.rstrip("/") + "/v1"`` (see
    ``lifespan.py``), so the root is just that minus the trailing ``/v1``.
    Fail loudly if the invariant ever breaks rather than emit a half-baked URL.
    """
    suffix = "/v1"
    if not api_base.endswith(suffix):
        raise ValueError(
            f"api_base {api_base!r} does not end with {suffix!r}; "
            "cannot derive the site-root URL for {{SITE_BASE}}"
        )
    return api_base[: -len(suffix)]


def _apply_placeholders(source: str, api_base: str) -> str:
    """Substitute every doc placeholder (``{{API_BASE}}``, ``{{SITE_BASE}}``)."""
    return source.replace(_API_BASE_PLACEHOLDER, api_base).replace(
        _SITE_BASE_PLACEHOLDER, _site_base(api_base)
    )


def build_docs(
    docs_root: Path,
    *,
    api_base: str,
) -> tuple[dict[str, RenderedDoc], list[NavEntry]]:
    """Render every markdown file under ``docs_root``; return pages + nav tree.

    Called once at startup from ``lifespan.py``. Raises if a file declared in
    ``_NAV_ORDER`` is missing — better to fail fast at boot than serve a 404
    on a "shipped" doc page in production.

    ``api_base`` (the deployed ``PUBLIC_BASE_URL`` + ``/v1``) replaces the
    ``{{API_BASE}}`` placeholder in the markdown source before rendering, so the
    documented endpoint tracks the configured domain.
    """
    md = _build_md()
    files = _doc_paths(docs_root)

    pages: dict[str, RenderedDoc] = {}
    for slug, path in files.items():
        source = _apply_placeholders(path.read_text(encoding="utf-8"), api_base)
        pages[slug] = _render_one(md, source, slug)

    # Build the nav tree from _NAV_ORDER. The editorial default_title wins
    # over the page's H1 here — H1s tend to be descriptive ("Using the
    # base-model API") while the left rail wants something short ("Overview").
    # The H1 still appears as the page title inside the docs-content area, so
    # nothing is lost — they just serve different roles.
    nav: list[NavEntry] = []
    for slug, default_title, children in _NAV_ORDER:
        if slug not in pages:
            raise FileNotFoundError(
                f"docs_build: required page {slug!r} not found under {docs_root}"
            )
        entry = NavEntry(slug=slug, title=default_title, children=[])
        for child_slug, child_default_title in children:
            if child_slug not in pages:
                raise FileNotFoundError(
                    f"docs_build: required example page {child_slug!r} "
                    f"not found under {docs_root}"
                )
            entry.children.append(
                NavEntry(slug=child_slug, title=child_default_title, children=[])
            )
        nav.append(entry)

    return pages, nav


def build_combined_markdown(docs_root: Path, *, api_base: str) -> str:
    """Concatenate the whole docs tree into one plain-markdown document.

    Served at ``/llms.txt`` for LLM consumption — a human can hand their
    assistant a single URL instead of crawling the multi-page site. Generated
    from the *same* markdown sources as the rendered pages (in ``_NAV_ORDER``),
    so there is no second copy to keep in sync. ``{{API_BASE}}`` is substituted
    exactly as in ``build_docs``.
    """
    files = _doc_paths(docs_root)
    # Flatten _NAV_ORDER (top-level then its children) into render order.
    order: list[str] = []
    for slug, _title, children in _NAV_ORDER:
        order.append(slug)
        order.extend(child_slug for child_slug, _ in children)

    parts: list[str] = [
        "# ACS Infra — base-model API (full documentation)",
        (
            "ACS Infra serves large language **base models** (raw next-token "
            "predictors — no chat template, no instruction tuning) over an "
            "OpenAI-compatible API, free for researchers. It exposes research-grade "
            "controls most hosted APIs hide: full `logprobs` and `prompt_logprobs`, "
            "a respected `seed`, arbitrary prefill/continuation, and `echo`. There "
            "are a few base models available (a small one for quick tests plus "
            "larger ones); some are kept warm, others cold-start on first use."
        ),
        (
            f"Endpoint: `POST {api_base}/completions` — there is no chat endpoint "
            "(these are base models, so they *continue* your text rather than "
            "answer like a chatbot). Authenticate with `Authorization: Bearer "
            "<API key>`, created from the dashboard after an invite. A browser "
            "Workbench is also available for no-code use."
        ),
        (
            "**If you are an AI assistant:** this single page is the complete "
            "tutorial + API reference (generated from the same source as the human "
            "docs at /tutorial; sections follow the site's navigation order). It "
            "contains everything you need to write a client, run completions, read "
            "logprobs/prompt_logprobs, and handle cold-boot and budget errors on "
            "the user's behalf."
        ),
    ]
    for slug in order:
        path = files.get(slug)
        if path is None:  # pragma: no cover — _NAV_ORDER validated in build_docs
            continue
        source = _apply_placeholders(path.read_text(encoding="utf-8"), api_base)
        parts.append(f"<!-- source: docs/{slug}.md (/tutorial/{slug}) -->")
        parts.append(source.strip())
    return "\n\n".join(parts) + "\n"


def docs_root_default() -> Path:
    """Default docs root: the ``docs/`` folder inside the wrapper package."""
    return Path(__file__).resolve().parent / "docs"
