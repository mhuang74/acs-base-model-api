"""Token-admin JSON APIs."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ... import auth as authmod
from ...db import get_session
from ...dependencies import get_settings
from ...keys import generate as generate_key
from ...models import ApiKey, UsageMonthly, User
from ...schemas import CreateKeyBody, CreateKeyResponse, CreateUserBody, KeySummary
from ...settings import Settings

router = APIRouter()


@router.post("/admin/users", status_code=201)
async def admin_create_user(
    request: Request,
    body: CreateUserBody,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    authmod.assert_admin(request, settings.admin_token)
    user = User(email=str(body.email), name=body.name, org=body.org, notes=body.notes)
    session.add(user)
    await session.flush()
    return {"id": str(user.id), "email": user.email}


@router.post("/admin/keys", status_code=201, response_model=CreateKeyResponse)
async def admin_create_key(
    request: Request,
    body: CreateKeyBody,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    authmod.assert_admin(request, settings.admin_token)
    user = (
        await session.execute(select(User).where(User.email == str(body.user_email)))
    ).scalar_one_or_none()
    if user is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"No user with email {body.user_email}",
                    "code": "user_not_found",
                }
            },
        )
    gk = generate_key()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=gk.hash_,
        key_prefix=gk.prefix,
        name=body.name,
        monthly_token_budget=body.monthly_token_budget,
    )
    session.add(api_key)
    await session.flush()
    return CreateKeyResponse(
        key=gk.plaintext,
        key_id=api_key.id,
        key_prefix=gk.prefix,
        user_email=body.user_email,
        monthly_token_budget=body.monthly_token_budget,
    )


@router.get("/admin/keys", response_model=list[KeySummary])
async def admin_list_keys(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    authmod.assert_admin(request, settings.admin_token)
    rows = (
        await session.execute(
            select(ApiKey, User)
            .join(User, ApiKey.user_id == User.id)
            .order_by(ApiKey.created_at.desc())
        )
    ).all()
    period_start = authmod._current_period_start()
    out: list[KeySummary] = []
    for api_key, user in rows:
        usage = (
            await session.execute(
                select(UsageMonthly).where(
                    UsageMonthly.key_id == api_key.id,
                    UsageMonthly.period_start == period_start,
                )
            )
        ).scalar_one_or_none()
        out.append(
            KeySummary(
                id=api_key.id,
                key_prefix=api_key.key_prefix,
                user_email=user.email,
                name=api_key.name,
                monthly_token_budget=api_key.monthly_token_budget,
                tokens_used_this_month=(
                    (usage.tokens_prompt + usage.tokens_completion) if usage else 0
                ),
                created_at=api_key.created_at,
                last_used_at=api_key.last_used_at,
                revoked_at=api_key.revoked_at,
            )
        )
    return out


@router.post("/admin/keys/{key_id}/revoke")
async def admin_revoke_key(
    key_id: uuid.UUID,
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    import datetime as dt

    authmod.assert_admin(request, settings.admin_token)
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id))
    ).scalar_one_or_none()
    if api_key is None:
        return Response(status_code=404)
    if api_key.revoked_at is None:
        api_key.revoked_at = dt.datetime.now(tz=dt.UTC)
    return {"id": str(api_key.id), "revoked_at": api_key.revoked_at.isoformat()}
