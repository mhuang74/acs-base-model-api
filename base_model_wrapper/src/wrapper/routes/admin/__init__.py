"""Admin route package aggregator."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    breakers,
    bulk_emails,
    debug,
    emails,
    feedback,
    home,
    models,
    probe,
    token_api,
    user_bulk,
    users,
    warm_windows,
)
from .common import _pending_key_serializer
from .models import (
    _resolve_model_states,
    admin_model_deploy,
    admin_model_keep_warm,
    admin_model_release,
    admin_model_stop,
    admin_models_page,
    admin_models_status,
)
from .probe import admin_probe_schedule
from .warm_windows import admin_warm_window_delete, admin_warm_window_upsert

router = APIRouter()
# NOTE: bulk_emails must register before emails — /admin/emails/{email_id}
# parses the segment as a UUID, so "/admin/emails/bulk" must match first.
# Same trap, same fix: user_bulk must register before users, because
# /admin/users/{user_id} is UUID-typed and would 422 on the literal "bulk".
for route_module in (
    token_api,
    breakers,
    home,
    probe,
    user_bulk,
    users,
    models,
    warm_windows,
    feedback,
    bulk_emails,
    emails,
    debug,
):
    router.include_router(route_module.router)

__all__ = [
    "_pending_key_serializer",
    "_resolve_model_states",
    "admin_model_deploy",
    "admin_model_keep_warm",
    "admin_model_release",
    "admin_model_stop",
    "admin_models_page",
    "admin_models_status",
    "admin_probe_schedule",
    "admin_warm_window_delete",
    "admin_warm_window_upsert",
    "router",
]
