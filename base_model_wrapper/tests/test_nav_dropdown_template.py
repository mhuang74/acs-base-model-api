"""Static checks for shared top-nav dropdown behavior."""

from __future__ import annotations

from pathlib import Path


BASE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "wrapper"
    / "templates"
    / "base.html"
)


def test_admin_dropdown_hover_bridge_covers_desktop_gap():
    """The floating admin menu has a visible gap; hover needs an invisible bridge."""
    html = BASE_TEMPLATE.read_text(encoding="utf-8")

    assert "top: calc(100% + 4px);" in html
    assert ".nav-dropdown::after" in html
    assert "height: 4px;" in html
    assert "width: max(100%, 150px);" in html
    assert ".nav-dropdown:hover::after { display: block; }" in html
