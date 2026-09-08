"""Bulk actions over a multi-selected set of users (ACS-372).

One endpoint, ``POST /admin/users/bulk``, driven by the checkbox column in the
all-users table. Exists because the hiring-cohort workflow is otherwise ~30
individual forms: back-tag the candidates who already applied, cap their
budgets, and suspend the lot at the two-week mark.

**Route ordering matters.** ``/admin/users/{user_id}`` is UUID-typed, so a
literal ``/admin/users/bulk`` only matches if this module is registered *before*
``users`` — see the tuple in ``routes/admin/__init__.py``.

Everything here is set-based (one statement per action, not a loop) and runs in
the single request transaction owned by ``get_session``: either the whole batch
lands or none of it does.
"""

from __future__ import annotations

import datetime as dt
import uuid

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import delete as sql_delete
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ... import web_auth as webauth
from ...db import get_session
from ...models import ApiKey, User, UserSession
from ...services import user_tags
from .common import log

router = APIRouter()

#: Actions the bar can dispatch. Anything else is rejected rather than ignored —
#: a typo'd action silently doing nothing is worse than an error.
_ACTIONS = frozenset({"suspend", "unsuspend", "tag_add", "tag_remove", "set_budget", "delete"})

#: Upper bound on one batch. The roster is ~70 accounts today and unpaginated,
#: so this is a runaway guard (a crafted POST with 10k ids), not a UX limit.
_MAX_BATCH = 500


def _users_redirect(return_to: str, *, msg: str | None = None, err: str | None = None):
    """Redirect back to the roster, preserving the caller's filter/sort.

    ``return_to`` comes from a hidden field the roster renders, so an admin who
    bulk-suspends a filtered cohort lands back on that same filtered view rather
    than on an unfiltered roster where their selection is lost among everyone
    else. Only our own users path is honoured — an attacker who can post here
    could otherwise turn this into an open redirect.

    The query is **merged**, not concatenated. ``common._redirect`` appends
    ``"?" + urlencode(...)`` unconditionally, so handing it a path that already
    carries a query produced a second ``?``: the flash was swallowed into the
    previous parameter's value (``sort=tokens?msg=Suspended+1+account.``) and
    was never rendered. With a tag filter it was worse — the mangled value
    matched no tag, so a bulk delete landed the admin on an *empty* roster with
    no confirmation, which reads as "it deleted everyone".

    Inbound ``msg``/``err`` are dropped: ``return_to`` is captured from the
    current URL, so the previous action's flash would otherwise round-trip into
    the next one.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit

    query: list[tuple[str, str]] = []
    if return_to.startswith("/admin/users?"):
        parsed = urlsplit(return_to)
        query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k not in ("msg", "err")
        ]
    if msg:
        query.append(("msg", msg))
    if err:
        query.append(("err", err))
    qs = ("?" + urlencode(query)) if query else ""
    return RedirectResponse(url=f"/admin/users{qs}", status_code=303)


def _delete_phrase(n: int) -> str:
    """The sentence an admin must type to confirm a bulk delete.

    Naming the count is the point: it makes "I meant to delete 2, not 40"
    impossible to fat-finger past, because the phrase changes with the
    selection. Kept here so the server and the template can't drift.
    """
    return f"Permanently delete {n} user{'' if n == 1 else 's'}"


@router.post("/admin/users/bulk")
async def admin_users_bulk(
    request: Request,
    user_ids: list[uuid.UUID] = Form(default=[]),
    action: str = Form(...),
    tag: str = Form(""),
    budget: str = Form(""),
    also_per_key: str = Form(""),
    confirm_phrase: str = Form(""),
    return_to: str = Form(""),
    admin: User = webauth.AdminRequiredDep,
    _same_origin: None = webauth.SameOriginDep,
    session: AsyncSession = Depends(get_session),
):
    """Apply one action to every selected user."""
    if action not in _ACTIONS:
        return _users_redirect(return_to, err=f"Unknown bulk action {action!r}.")

    # De-dupe, and never act on yourself: every action here is something an
    # admin would be locked out by (suspend, delete) or would not mean to do to
    # their own account in a sweep.
    ids = [uid for uid in dict.fromkeys(user_ids) if uid != admin.id]
    if not ids:
        return _users_redirect(
            return_to, err="Select at least one account (your own is never included)."
        )
    if len(ids) > _MAX_BATCH:
        return _users_redirect(return_to, err=f"Too many accounts in one batch (max {_MAX_BATCH}).")

    now = dt.datetime.now(tz=dt.UTC)

    if action == "suspend":
        # Only approved accounts can be suspended — same rule as the single-user
        # handler, enforced in the WHERE so a mixed selection suspends the
        # eligible ones and reports the real number rather than erroring.
        # Admins are excluded, not refused: a suspended admin fails
        # current_user's status check, so one click on "select all" would soft-
        # lock every admin but the one doing it. Delete refuses the whole batch
        # instead because it is irreversible; suspend is recoverable, so
        # skipping and *saying so* is the less obstructive shape.
        affected = (
            await session.execute(
                update(User)
                .where(User.id.in_(ids), User.status == "approved", User.role != "admin")
                .values(status="suspended", suspended_at=now, suspended_by_user_id=admin.id)
            )
        ).rowcount or 0
        # Same bulk UPDATE revoke_all_user_sessions runs, lifted to the whole set.
        await session.execute(
            update(UserSession)
            .where(UserSession.user_id.in_(ids), UserSession.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        log.info("admin_bulk_suspend", admin_id=str(admin.id), count=affected)
        skipped = len(ids) - affected
        msg = f"Suspended {affected} account{'' if affected == 1 else 's'}."
        if skipped:
            msg += f" {skipped} skipped (not approved, or an admin — suspend those individually)."
        return _users_redirect(return_to, msg=msg)

    if action == "unsuspend":
        affected = (
            await session.execute(
                update(User)
                .where(User.id.in_(ids), User.status == "suspended")
                .values(status="approved", suspended_at=None, suspended_by_user_id=None)
            )
        ).rowcount or 0
        log.info("admin_bulk_unsuspend", admin_id=str(admin.id), count=affected)
        skipped = len(ids) - affected
        msg = f"Restored {affected} account{'' if affected == 1 else 's'}."
        if skipped:
            msg += f" {skipped} skipped (not suspended)."
        return _users_redirect(return_to, msg=msg)

    if action in ("tag_add", "tag_remove"):
        normalized = user_tags.normalize_tag(tag)
        if normalized is None:
            return _users_redirect(
                return_to,
                err="A tag needs at least one letter or digit (a-z, 0-9 and dashes).",
            )
        if action == "tag_add":
            # Narrow to ids that still exist. user_tags.user_id is NOT NULL
            # REFERENCES users(id), so a stale id — a roster tab left open while
            # another admin deleted an account — would raise ForeignKeyViolation
            # and 500 the whole request. Every other action here degrades
            # gracefully; this one shouldn't be the exception. It also makes the
            # "already had it" arithmetic honest instead of counting
            # non-existent accounts as already-tagged.
            live = set(
                (await session.execute(select(User.id).where(User.id.in_(ids)))).scalars().all()
            )
            ids = [uid for uid in ids if uid in live]
            if not ids:
                return _users_redirect(return_to, err="None of the selected accounts still exist.")
            added = await user_tags.add_tag(session, ids, normalized, created_by_user_id=admin.id)
            log.info("admin_bulk_tag_add", admin_id=str(admin.id), tag=normalized, count=added)
            already = len(ids) - added
            msg = f"Tagged {added} account{'' if added == 1 else 's'} “{normalized}”."
            if already:
                msg += f" {already} already had it."
            return _users_redirect(return_to, msg=msg)
        removed = await user_tags.remove_tag(session, ids, normalized)
        log.info("admin_bulk_tag_remove", admin_id=str(admin.id), tag=normalized, count=removed)
        return _users_redirect(
            return_to,
            msg=f"Removed “{normalized}” from {removed} account{'' if removed == 1 else 's'}.",
        )

    if action == "set_budget":
        raw = (budget or "").strip()
        if raw == "":
            value: int | None = None
        else:
            try:
                value = int(raw)
            except ValueError:
                return _users_redirect(
                    return_to, err=f"Budget must be a whole number (got {raw!r})."
                )
            if value < 0:
                return _users_redirect(return_to, err="Budget can't be negative.")
        # Validate before writing anything: this handler returns a redirect
        # rather than raising, so the request transaction still commits — a
        # bail-out placed after the aggregate UPDATE would leave it half-applied.
        if also_per_key and value == 0:
            # api_keys.monthly_token_budget == 0 is the legacy *unlimited*
            # sentinel (auth.py), while users.monthly_token_budget_total = 0
            # means hard-blocked. Applying 0 to both would block the aggregate
            # and simultaneously make every key unlimited — the exact opposite
            # of what the operator asked for, from one number, in one submit.
            return _users_redirect(
                return_to,
                err=(
                    "A per-key budget of 0 means UNLIMITED (legacy sentinel), not blocked. "
                    "Use 1 to effectively block a key, or clear the per-key checkbox."
                ),
            )
        affected = (
            await session.execute(
                update(User).where(User.id.in_(ids)).values(monthly_token_budget_total=value)
            )
        ).rowcount or 0
        keys_affected = 0
        if also_per_key:
            # Per-key budgets are NOT NULL, so a blank aggregate means
            # "unlimited" there but has no per-key equivalent — 0 is the
            # existing convention for an unbounded key.
            keys_affected = (
                await session.execute(
                    update(ApiKey)
                    .where(ApiKey.user_id.in_(ids), ApiKey.revoked_at.is_(None))
                    .values(monthly_token_budget=value if value is not None else 0)
                )
            ).rowcount or 0
        log.info(
            "admin_bulk_set_budget",
            admin_id=str(admin.id),
            budget=value,
            count=affected,
            keys=keys_affected,
        )
        shown = "unlimited" if value is None else f"{value:,}"
        msg = f"Set the monthly budget to {shown} on {affected} account{'' if affected == 1 else 's'}."
        if also_per_key:
            msg += f" Also updated {keys_affected} live key{'' if keys_affected == 1 else 's'}."
        return _users_redirect(return_to, msg=msg)

    # ---- delete ------------------------------------------------------------
    # Irreversible, and it takes the usage history with it (api_requests,
    # usage_* and harvest_jobs all CASCADE), so the Metabase C1/C2 tiles lose
    # those cohorts. Two guards stand in front of it.
    # FOR UPDATE so a concurrent role change can't land between this read and
    # the DELETE — otherwise an account promoted to admin in the meantime could
    # still be deleted by a batch that read it as a plain user.
    targets = list(
        (
            await session.execute(
                select(User).where(User.id.in_(ids)).order_by(User.id).with_for_update()
            )
        )
        .scalars()
        .all()
    )
    if not targets:
        return _users_redirect(return_to, err="None of the selected accounts still exist.")

    # Guard 1: refuse the whole batch if any target is an admin. The single-user
    # handler's last-admin check uses SELECT … FOR UPDATE, but inside a bulk
    # loop it would evaluate against not-yet-committed state and happily delete
    # N-1 admins. Rather than reimplement that correctly for a case nobody
    # needs, admins are simply not bulk-deletable — do it one at a time, where
    # the race-safe guard applies.
    admin_targets = sorted(t.email for t in targets if t.role == "admin")
    if admin_targets:
        return _users_redirect(
            return_to,
            err=(
                "Refusing the whole batch: it contains admin account(s) "
                f"{', '.join(admin_targets)}. Delete admins one at a time from their detail page."
            ),
        )

    # Guard 2: the typed confirmation, naming the exact count. The count is in
    # the phrase precisely so a stale selection can't slip through — if the
    # admin thought they had 2 selected and actually had 40, the phrase they
    # typed won't match.
    expected = _delete_phrase(len(targets))
    if confirm_phrase.strip() != expected:
        return _users_redirect(
            return_to,
            err=f"Type exactly “{expected}” to confirm. Nothing was deleted.",
        )

    emails = [t.email for t in targets]
    log.info(
        "admin_bulk_delete",
        admin_id=str(admin.id),
        count=len(targets),
        target_emails=[e[:3] + "***" for e in emails],
    )
    # Core DELETE rather than per-row session.delete(): the DB-level ON DELETE
    # CASCADE on api_keys covers what the ORM's delete-orphan would have done
    # (models.py), there are no before_delete hooks, and migration 0021 removed
    # the last RESTRICT blockers on the user path.
    await session.execute(sql_delete(User).where(User.id.in_([t.id for t in targets])))
    return _users_redirect(
        return_to,
        msg=f"Permanently deleted {len(targets)} account{'' if len(targets) == 1 else 's'}.",
    )
