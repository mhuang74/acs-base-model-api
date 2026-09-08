"""Shared SlowAPI limiter for app and route modules."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter


def _rate_key(request: Request) -> str:
    """Rate-limit bucket: per-key when available, else per-IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return f"key:{auth[7:][:48]}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


limiter = Limiter(key_func=_rate_key)
