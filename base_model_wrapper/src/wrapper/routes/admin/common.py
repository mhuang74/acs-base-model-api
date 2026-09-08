"""Shared helpers for admin route modules."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer

from ... import web_auth as webauth
from ...logging import get_logger
from ...settings import Settings

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
log = get_logger()
_FEEDBACK_CATEGORIES = ("bug", "feature", "general")


class _PendingKeySerializer:
    """Encrypt one-shot approval keys while allowing legacy signed rows.

    Before 2026-06-08 this helper used itsdangerous only, which authenticated
    the value but did not encrypt it. New rows use Fernet; old rows are still
    accepted so a pending user approved before the deploy can reveal the key
    once and clear the column.
    """

    _PREFIX = "fernet:"

    def __init__(self, session_secret: str) -> None:
        digest = hashlib.sha256(
            f"{session_secret}:{webauth.PENDING_KEY_SALT}".encode("utf-8")
        ).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))
        self._legacy = URLSafeSerializer(session_secret, salt=webauth.PENDING_KEY_SALT)

    def dumps(self, plaintext: str) -> str:
        token = self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")
        return f"{self._PREFIX}{token}"

    def loads(self, token: str) -> str:
        if token.startswith(self._PREFIX):
            try:
                return self._fernet.decrypt(token[len(self._PREFIX) :].encode("ascii")).decode(
                    "utf-8"
                )
            except (InvalidToken, UnicodeDecodeError) as exc:
                raise BadSignature("pending key ciphertext could not be decrypted") from exc
        return self._legacy.loads(token)


def _pending_key_serializer(session_secret: str) -> _PendingKeySerializer:
    return _PendingKeySerializer(session_secret)


def _redirect(path: str, *, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    from urllib.parse import urlencode

    params = {}
    if msg:
        params["msg"] = msg
    if err:
        params["err"] = err
    qs = ("?" + urlencode(params)) if params else ""
    return RedirectResponse(url=f"{path}{qs}", status_code=303)


def _admin_redirect(*, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    return _redirect("/admin", msg=msg, err=err)


def _uptime_redirect(*, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    return _redirect("/admin/uptime", msg=msg, err=err)


def _feedback_redirect(*, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    return _redirect("/admin/feedback", msg=msg, err=err)


def _lookup_model_or_404(request: Request, model_id: str) -> Any:
    entry = request.app.state.models.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="model not found")
    if entry.modal_app_name is None:
        raise HTTPException(
            status_code=400,
            detail=f"model {model_id!r} has no modal_app_name configured",
        )
    return entry


def _modal_ops_or_503(settings: Settings) -> None:
    if not (settings.modal_token_id and settings.modal_token_secret):
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": (
                        "Modal admin controls disabled - MODAL_TOKEN_ID / "
                        "MODAL_TOKEN_SECRET are not configured on this wrapper."
                    ),
                    "code": "modal_not_configured",
                }
            },
        )


def _modal_unavailable_msg(settings: Settings) -> str | None:
    """One-line user-facing message when Modal admin RPCs cannot run, else None."""
    if not settings.modal_token_id or not settings.modal_token_secret:
        return "Modal admin controls disabled: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET unset."
    return None
