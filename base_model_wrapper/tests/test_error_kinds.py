"""Drift guard for the ``error_kind`` taxonomy (ACS-94).

``error_kind`` strings used to be free-form literals scattered across the proxy,
the API route, and the workbench. A typo (``vllm_oom`` → ``vlllm_oom``) would
silently fragment dashboards and break the runbook lookup. ``wrapper.error_kinds``
is now the single source of truth; these tests pin it to:

  1. the documented taxonomy table in ``docs/runbooks/error-troubleshooting.md``
     (the human-facing contract), and
  2. the ``error_kind`` literals the Metabase dashboard SQL groups on,

so the enum, the docs, and the dashboard can't drift apart without a red test.
"""

from __future__ import annotations

import re
from pathlib import Path

from wrapper.error_kinds import KNOWN_ERROR_KINDS, ErrorKind, is_known
from wrapper.proxy import _VLLM_ERROR_PATTERNS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "error-troubleshooting.md"
_DASHBOARD_SQL = _REPO_ROOT / "docs" / "runbooks" / "metabase-dashboard-queries.sql"


def _documented_kinds() -> set[str]:
    """Pull the ``error_kind`` codes from the runbook taxonomy table.

    The table rows look like ``| `cold_boot` | 200* | ... |`` — we take the
    backtick-wrapped token in the first column of every row that has one.
    """
    kinds: set[str] = set()
    row_re = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|")
    in_table = False
    for line in _RUNBOOK.read_text().splitlines():
        if line.startswith("## The taxonomy"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break  # next section — stop scanning
        m = row_re.match(line)
        if m and m.group(1) != "error_kind":  # skip the column-header cell
            kinds.add(m.group(1))
    return kinds


def test_enum_matches_documented_taxonomy() -> None:
    """The enum and the runbook taxonomy table must list exactly the same kinds."""
    documented = _documented_kinds()
    enum_values = {k.value for k in ErrorKind}
    assert documented, "failed to parse the taxonomy table — check the runbook format"
    missing_from_docs = enum_values - documented
    missing_from_enum = documented - enum_values
    assert not missing_from_docs, f"ErrorKind values absent from the runbook table: {missing_from_docs}"
    assert not missing_from_enum, f"runbook table kinds absent from ErrorKind: {missing_from_enum}"


def test_dashboard_sql_kinds_are_known() -> None:
    """Every ``error_kind`` literal the dashboard SQL groups on must be a real kind.

    Catches a dashboard query referencing a value the code never produces (or
    a value renamed in the enum but not in the SQL).
    """
    sql = _DASHBOARD_SQL.read_text()
    # Match string literals compared against error_kind, e.g. error_kind = 'cold_boot'
    # or NOT IN ('cold_boot','circuit_open'). The SQL file contains non-error_kind
    # literals too (gpu shapes, etc.), so only inspect lines that mention
    # error_kind and pull the string literals from those.
    error_kind_lines = [ln for ln in sql.splitlines() if "error_kind" in ln]
    used_near_error_kind = set(re.findall(r"'([a-z0-9_]+)'", "\n".join(error_kind_lines)))
    unknown = used_near_error_kind - KNOWN_ERROR_KINDS
    assert not unknown, f"dashboard SQL groups on unknown error_kind(s): {unknown}"


def test_classify_patterns_return_enum_members() -> None:
    """``proxy.classify_upstream_error_body`` must only ever produce real kinds."""
    for label, _pattern in _VLLM_ERROR_PATTERNS:
        assert isinstance(label, ErrorKind)
        assert is_known(label)


def test_is_known_contract() -> None:
    assert is_known(None) is True  # no error
    assert is_known(ErrorKind.vllm_oom) is True
    assert is_known("vllm_oom") is True
    assert is_known("vlllm_oom") is False  # the canonical typo
    assert is_known("totally_made_up") is False
