"""Drift guard for the ``users.status`` vocabulary (ACS-353).

``users.status`` is plain ``Text`` with no CHECK constraint (migration 0005),
deliberately: every gate in the app is deny-by-default (``status != 'approved'``),
so adding a state costs no migration — the ``EmailLog.kind`` open-vocabulary
precedent rather than the ``harvest_jobs`` CHECK one.

The thing a CHECK *would* have bought us is catching a typo'd status literal
(``'suspened'``) that silently locks someone out with no matching admin control.
These tests buy that back without the downgrade burden of folding live rows:
every status literal the code actually assigns must be a member of
``USER_STATUSES``.

No DB needed — this is a source-level scan, so it runs everywhere.

**What this does NOT catch** — do not over-trust it:

- Only ``src/wrapper/**.py``. Templates are not scanned, so a
  ``{% elif target.status == 'suspened' %}`` typo — the same bug class — slips
  through, as would one in ``alembic/``, ``cli/`` or ``scripts/``.
- Only receivers in ``_USERISH``. ``user_row.status`` or ``target_user.status``
  would be missed; add the name here if such a variable appears.
- **Unknown literals, not missing coverage.** It cannot tell you that some
  ``== "pending"`` allow-list check forgot to handle ``suspended``, which is the
  likelier bug. It would not have caught the Metabase ``WHERE u.status =
  'approved'`` filter that silently dropped suspended users from the C1 tile.
  Nothing validates a write against ``USER_STATUSES`` at runtime either — that
  is the deliberate cost of an open vocabulary.
"""

from __future__ import annotations

import re
from pathlib import Path

from wrapper.models import USER_STATUSES

_SRC = Path(__file__).resolve().parents[1] / "src" / "wrapper"

#: Receivers that denote a user row. A bare ``\.status`` scan is useless here:
#: model entries, harvest jobs, feedback and bulk-email batches all have their
#: own ``status`` and live in the same files, so it would drag in 'live',
#: 'running', 'sent' and friends. Anchoring on the receiver name is what makes
#: this precise. Keep in sync if a new variable name for a user row appears.
_USERISH = r"(?:user|target|u|new_user|existing|admin|caller|row\.user|User)"

#: ``target.status = "x"`` — assignment INTO the column.
_ASSIGN_RE = re.compile(rf"""\b{_USERISH}\.status\s*=\s*["']([a-z_]+)["']""")
#: ``user.status == "x"`` / ``!= "x"`` — comparisons that READ it.
_COMPARE_RE = re.compile(rf"""\b{_USERISH}\.status\s*[!=]=\s*["']([a-z_]+)["']""")
#: ``status="x"`` inside a ``User(...)`` construction.
_KWARG_RE = re.compile(r"""\bstatus\s*=\s*["']([a-z_]+)["']""")


def _status_literals() -> dict[str, set[str]]:
    """Map each literal -> the files it appears in, across the app source."""
    found: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text()
        for regex in (_ASSIGN_RE, _COMPARE_RE):
            for m in regex.finditer(text):
                found.setdefault(m.group(1), set()).add(path.name)
        # Keyword form only counts inside a User(...) construction, which is
        # the only place a bare ``status=`` sets an account status.
        for m in re.finditer(r"\bUser\((?:[^()]|\([^()]*\))*\)", text, re.S):
            for k in _KWARG_RE.finditer(m.group(0)):
                found.setdefault(k.group(1), set()).add(path.name)
    return found


def test_every_account_status_literal_is_known() -> None:
    """A typo'd status would lock a user out with no admin control to undo it."""
    found = _status_literals()
    assert found, "scanned no status literals — check the regexes/paths"
    unknown = {lit: sorted(files) for lit, files in found.items() if lit not in USER_STATUSES}
    assert not unknown, f"status literals absent from USER_STATUSES: {unknown}"


def test_suspended_is_a_known_status() -> None:
    """Pins the ACS-353 addition itself, so a revert has to be deliberate."""
    assert "suspended" in USER_STATUSES


def test_approved_is_the_only_status_granting_access() -> None:
    """The deny-by-default contract every gate relies on: exactly one status
    means "has access". If a second ever joins it, every ``!= 'approved'``
    check in the codebase becomes wrong and must be revisited."""
    assert USER_STATUSES - {"approved"} == {"pending", "rejected", "suspended"}
