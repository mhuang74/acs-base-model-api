"""Admin CLI — `acs-keys` — talks to the DB directly. Source of truth for ops.

The HTTP `/admin/*` endpoints exist for a future minimal UI; this CLI is what
the on-call team uses.

Usage:
    DATABASE_URL=... acs-keys user create alice@lab.org --org "LabX"
    DATABASE_URL=... acs-keys create alice@lab.org --budget 100000000 --name "rollout"
    DATABASE_URL=... acs-keys list
    DATABASE_URL=... acs-keys revoke <key_id>
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid

import typer
from sqlalchemy import select

from wrapper.auth import _current_period_start
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.keys import generate as generate_key
from wrapper.models import ApiKey, UsageMonthly, User
from wrapper.web_auth import generate_temp_password, hash_password

app = typer.Typer(help="ACS base-model API key management")
user_app = typer.Typer(help="User management")
app.add_typer(user_app, name="user")


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        typer.echo("DATABASE_URL is required", err=True)
        raise typer.Exit(2)
    return url


def _run(coro):
    return asyncio.run(coro)


@user_app.command("create")
def user_create(
    email: str,
    name: str | None = typer.Option(None),
    org: str | None = typer.Option(None),
    notes: str | None = typer.Option(None),
) -> None:
    async def _go():
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                user = User(email=email, name=name, org=org, notes=notes)
                s.add(user)
                await s.flush()
                typer.echo(f"created user {user.id} {email}")
        finally:
            await engine.dispose()

    _run(_go())


@app.command("create")
def create(
    email: str = typer.Argument(..., help="User email — must already exist (or use --create-user)."),
    name: str | None = typer.Option(None, help="Free-form label for this key."),
    budget: int = typer.Option(0, help="Monthly token budget (total). 0 = unlimited."),
    daily_budget: int | None = typer.Option(
        None,
        "--daily-budget",
        help=(
            "Per-day token ceiling for this key. Default NULL (unlimited / no "
            "daily cap), so existing behaviour is unchanged. Plan recommends "
            "setting this to roughly monthly_budget / 10 to smooth burst spend, "
            "but defaults to NULL for backwards compatibility — explicit opt-in."
        ),
    ),
    input_budget: int | None = typer.Option(
        None,
        "--input-budget",
        help=(
            "Per-month input (prompt) token ceiling. Default NULL = no "
            "per-direction cap; the --budget total still applies."
        ),
    ),
    output_budget: int | None = typer.Option(
        None,
        "--output-budget",
        help=(
            "Per-month output (completion) token ceiling. Default NULL = no "
            "per-direction cap; the --budget total still applies."
        ),
    ),
    create_user: bool = typer.Option(False, "--create-user", help="Create the user if missing."),
    password: str | None = typer.Option(
        None,
        help="Set the user's web-login password. If omitted with --create-user, an auto-generated temp password is printed.",
    ),
) -> None:
    pw: str | None = password  # alias to avoid free-variable shadowing inside _go

    async def _go():
        nonlocal pw
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                user = (
                    await s.execute(select(User).where(User.email == email))
                ).scalar_one_or_none()
                printed_password: str | None = None
                if user is None:
                    if not create_user:
                        typer.echo(
                            f"no user with email {email}; pass --create-user to add one",
                            err=True,
                        )
                        raise typer.Exit(2)
                    user = User(email=email)
                    if pw is None:
                        pw = generate_temp_password()
                        printed_password = pw
                    user.password_hash = hash_password(pw)
                    s.add(user)
                    await s.flush()
                elif pw is not None:
                    # Re-set password on an existing user when --password is given.
                    user.password_hash = hash_password(pw)
                gk = generate_key()
                api_key = ApiKey(
                    user_id=user.id,
                    key_hash=gk.hash_,
                    key_prefix=gk.prefix,
                    name=name,
                    monthly_token_budget=budget,
                    daily_token_budget=daily_budget,
                    monthly_input_token_budget=input_budget,
                    monthly_output_token_budget=output_budget,
                )
                s.add(api_key)
                await s.flush()
                typer.secho("KEY CREATED — shown ONCE, store now:", fg="yellow", bold=True)
                typer.echo(gk.plaintext)
                # Print every configured limit; omit the NULL ones to keep the
                # line short for the common single-budget case.
                limit_bits = [f"budget={budget}"]
                if daily_budget is not None:
                    limit_bits.append(f"daily={daily_budget}")
                if input_budget is not None:
                    limit_bits.append(f"input={input_budget}")
                if output_budget is not None:
                    limit_bits.append(f"output={output_budget}")
                typer.echo(
                    f"key_id={api_key.id}  prefix={gk.prefix}  " + "  ".join(limit_bits)
                )
                if printed_password is not None:
                    typer.secho(
                        "WEB-LOGIN PASSWORD (auto-generated, shown ONCE):",
                        fg="yellow",
                        bold=True,
                    )
                    typer.echo(printed_password)
        finally:
            await engine.dispose()

    _run(_go())


@app.command("set-password")
def set_password_cmd(
    email: str = typer.Argument(..., help="User email — must already exist."),
    password: str | None = typer.Option(
        None,
        help="Password to set. If omitted, an auto-generated temp password is printed.",
    ),
) -> None:
    """Reset a user's web-login password. Use for legacy users without one set
    yet, or when a user has forgotten theirs and asks admin to reset."""

    async def _go():
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                user = (
                    await s.execute(select(User).where(User.email == email))
                ).scalar_one_or_none()
                if user is None:
                    typer.echo(f"no user with email {email}", err=True)
                    raise typer.Exit(1)
                pw = password or generate_temp_password()
                user.password_hash = hash_password(pw)
                typer.echo(f"password set for {email}")
                if password is None:
                    typer.secho(
                        "AUTO-GENERATED PASSWORD (shown ONCE):", fg="yellow", bold=True
                    )
                    typer.echo(pw)
        finally:
            await engine.dispose()

    _run(_go())


@app.command("list")
def list_keys() -> None:
    async def _go():
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                rows = (
                    await s.execute(
                        select(ApiKey, User)
                        .join(User, ApiKey.user_id == User.id)
                        .order_by(ApiKey.created_at.desc())
                    )
                ).all()
                period_start = _current_period_start()
                hdr = f"{'id':36} {'prefix':8} {'email':32} {'budget':>10} {'used':>10} {'revoked'}"
                typer.echo(hdr)
                for api_key, user in rows:
                    usage = (
                        await s.execute(
                            select(UsageMonthly).where(
                                UsageMonthly.key_id == api_key.id,
                                UsageMonthly.period_start == period_start,
                            )
                        )
                    ).scalar_one_or_none()
                    used = (usage.tokens_prompt + usage.tokens_completion) if usage else 0
                    rev = api_key.revoked_at.isoformat() if api_key.revoked_at else "-"
                    typer.echo(
                        f"{str(api_key.id):36} {api_key.key_prefix:8} {user.email:32} "
                        f"{api_key.monthly_token_budget:>10} {used:>10} {rev}"
                    )
        finally:
            await engine.dispose()

    _run(_go())


@app.command("promote")
def promote(
    email: str = typer.Argument(..., help="User email — must already exist."),
) -> None:
    """Promote a user to admin role.

    Used to bootstrap the first admin so they can use the /admin web UI. After
    that, admins manage each other (and approve signups) through the dashboard.
    """

    async def _go():
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                user = (
                    await s.execute(select(User).where(User.email == email))
                ).scalar_one_or_none()
                if user is None:
                    typer.echo(f"no user with email {email}", err=True)
                    raise typer.Exit(1)
                if user.role == "admin":
                    typer.echo(f"{email} is already admin")
                    return
                user.role = "admin"
                typer.secho(f"promoted {email} to admin", fg="green")
        finally:
            await engine.dispose()

    _run(_go())


@app.command("revoke")
def revoke(key_id: str) -> None:
    async def _go():
        engine = make_engine(_db_url())
        try:
            sessions = make_session_factory(engine)
            async with session_scope(sessions) as s:
                try:
                    kid = uuid.UUID(key_id)
                except ValueError:
                    typer.echo(f"not a valid UUID: {key_id}", err=True)
                    raise typer.Exit(2)
                api_key = (
                    await s.execute(select(ApiKey).where(ApiKey.id == kid))
                ).scalar_one_or_none()
                if api_key is None:
                    typer.echo(f"no key {key_id}", err=True)
                    raise typer.Exit(1)
                if api_key.revoked_at is None:
                    api_key.revoked_at = dt.datetime.now(tz=dt.UTC)
                typer.echo(f"revoked {api_key.id} at {api_key.revoked_at.isoformat()}")
        finally:
            await engine.dispose()

    _run(_go())


if __name__ == "__main__":
    app()
