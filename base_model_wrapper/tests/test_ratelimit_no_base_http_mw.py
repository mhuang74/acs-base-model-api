"""ACS-347: ``SlowAPIMiddleware`` (a Starlette ``BaseHTTPMiddleware``) was removed
from the app stack.

It buffered and re-emitted every response through an internal
``_StreamingResponse``, which intermittently raised ``RuntimeError: Response
content longer than Content-Length`` on ``/v1/completions`` (a prod 5xx) and
needlessly buffered the multi-GB full-vocab / activation bodies. Rate limiting
does not depend on it — the per-route ``@limiter.limit`` decorators enforce
inline. These tests guard both facts so the middleware can't creep back in.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from wrapper.app import create_app


def _app() -> FastAPI:
    async def lifespan(app):
        yield

    return create_app(lifespan=lifespan, limiter=Limiter(key_func=get_remote_address))


def test_no_base_http_middleware_in_the_response_path() -> None:
    """No middleware may be a ``BaseHTTPMiddleware`` — it buffers/re-emits the
    response body (ACS-347). The response path stays pure-ASGI."""
    app = _app()
    offenders = [
        mw.cls.__name__
        for mw in app.user_middleware
        if isinstance(mw.cls, type) and issubclass(mw.cls, BaseHTTPMiddleware)
    ]
    assert offenders == [], (
        f"BaseHTTPMiddleware in the stack: {offenders}. These buffer and re-emit "
        "response bodies (ACS-347: 'Response content longer than Content-Length' on "
        "/v1/completions). Keep the response path pure-ASGI."
    )


def test_rate_limit_decorator_still_enforces_without_the_middleware() -> None:
    """Enforcement lives in the ``@limiter.limit`` decorator, not the middleware:
    a 2/minute route returns 429 on the third call with no SlowAPIMiddleware."""
    limiter = Limiter(key_func=get_remote_address)

    async def lifespan(app):
        yield

    app = create_app(lifespan=lifespan, limiter=limiter)
    app.state.limiter.enabled = True
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/_limited")
    @limiter.limit("2/minute")
    async def limited(request: Request):  # slowapi requires the request arg
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/_limited").status_code == 200
    assert client.get("/_limited").status_code == 200
    assert client.get("/_limited").status_code == 429
