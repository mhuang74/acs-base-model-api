"""Admin landing page."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ... import web_auth as webauth
from ...models import User
from .common import templates

router = APIRouter()


@router.get("/admin")
async def admin_dashboard(
    request: Request,
    admin: User = webauth.AdminRequiredDep,
):
    """Admin landing page — a menu linking to the admin subpages.

    Users live at /admin/users, models at /admin/models, the capacity probe at
    /admin/uptime, plus /admin/feedback and /admin/emails. This page is just the
    index that links to them.
    """
    flash_message = request.query_params.get("msg") or None
    flash_error = request.query_params.get("err") or None
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        {
            "user": admin,
            "flash_message": flash_message,
            "flash_error": flash_error,
        },
    )
