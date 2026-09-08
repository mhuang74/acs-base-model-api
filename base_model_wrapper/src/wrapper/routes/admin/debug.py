"""Admin-only debug endpoints — deliberately exercise error paths.

Used to verify the global exception handler + Sentry wiring end-to-end in a
live environment (ACS-40). Admin-token gated; carries no user data.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ... import auth as authmod
from ...dependencies import get_settings
from ...settings import Settings

router = APIRouter()


@router.post("/admin/debug/boom")
async def admin_debug_boom(request: Request, settings: Settings = Depends(get_settings)):
    """Deliberately raise to verify the catch-all 500 handler + Sentry report.

    Never returns — it raises, which the global ``Exception`` handler turns into
    a generic ``500 {"error":…,"request_id":…}`` (no internals) and reports to
    Sentry (scrubbed). Admin-token gated, so only an operator can trip it; the
    error carries no user content. Trigger with:

        curl -X POST https://<host>/admin/debug/boom -H "X-Admin-Token: <token>"
    """
    authmod.assert_admin(request, settings.admin_token)
    raise ValueError("ACS-40 Sentry verification — deliberate test error (no real failure)")
