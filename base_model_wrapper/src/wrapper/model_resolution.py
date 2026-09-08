"""Shared model-resolution helpers for wrapper routes."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

def _resolve_model(
    request: Request, requested: str | None
) -> tuple[str, Any] | None:
    """Resolve a request's ``model`` field against the registry.

    Returns (model_id, entry) on success; None if the model is unknown or
    disabled. Caller surfaces 400.
    """
    registry = request.app.state.models
    model_id = requested or request.app.state.default_model_id
    entry = registry.get(model_id)
    if entry is None or entry.status == "disabled":
        return None
    return model_id, entry



def _unknown_model_response(request: Request, requested: str | None) -> JSONResponse:
    registry = request.app.state.models
    live = sorted(mid for mid, m in registry.items() if m.status == "live")
    if not requested:
        message = "No model specified and no default configured."
    else:
        # Pasting an HF repo id (the upstream ``served_model_name``) is a common
        # mistake — clients copy it from the model card. Suggest the canonical
        # short id rather than just listing every available model.
        hint = next(
            (m.model_id for m in registry.values() if m.served_model_name == requested),
            None,
        )
        suggest = f" Did you mean {hint!r}?" if hint else ""
        message = f"Unknown model {requested!r}.{suggest} Available: {', '.join(live)}."
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
    )


