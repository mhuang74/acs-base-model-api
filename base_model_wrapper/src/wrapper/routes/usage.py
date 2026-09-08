"""Per-user usage route."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from .. import web_auth as webauth
from ..db import get_session
from ..models import User
from ..services.usage_reports import build_usage_context

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()

# --- usage tab ---------------------------------------------------------------


@router.get("/usage")
async def usage_tab(
    request: Request,
    key: str | None = None,
    user: User | None = webauth.CurrentUserDep,
    session: AsyncSession = Depends(get_session),
):
    """Per-user usage view: this-month total, 30-day chart, breakdowns, recent requests.

    The optional ``?key=<id>`` query param scopes the whole view to a single
    one of the caller's keys; absent or unrecognised, the view aggregates
    across every key (the "All keys" default).
    """
    if user is None:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse(
        request,
        "usage.html",
        await build_usage_context(session, user, selected_key=key),
    )
