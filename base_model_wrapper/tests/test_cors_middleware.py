"""ACS-146: CORS middleware on /v1 — preflight and simple-request checks.

`/v1` is a token-in-Authorization-header API (like OpenAI's). The wildcard
default lets browser playgrounds (e.g. logitloom) call it cross-origin.
`allow_credentials=False` keeps the cookie-authenticated web/admin/workbench
surface off the CORS path entirely, so this adds no new CSRF reachability
(cf. ACS-102).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from wrapper.app import create_app


def _app(*, cors_allow_origins: str = "*") -> FastAPI:
    async def lifespan(app):
        yield

    limiter = Limiter(key_func=get_remote_address)
    app = create_app(
        lifespan=lifespan, limiter=limiter, cors_allow_origins=cors_allow_origins
    )
    app.state.limiter.enabled = False

    @app.post("/v1/completions")
    async def completions():
        return {"ok": True}

    return app


def test_preflight_options_returns_allow_origin_wildcard():
    with TestClient(_app()) as c:
        r = c.options(
            "/v1/completions",
            headers={
                "Origin": "https://vgel.me",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 200, r.text
        assert r.headers.get("access-control-allow-origin") == "*"
        allowed_methods = r.headers.get("access-control-allow-methods", "")
        assert "POST" in allowed_methods
        allowed_headers = r.headers.get("access-control-allow-headers", "").lower()
        assert "authorization" in allowed_headers
        assert "content-type" in allowed_headers


def test_simple_post_carries_allow_origin_header():
    with TestClient(_app()) as c:
        r = c.post(
            "/v1/completions",
            headers={"Origin": "https://vgel.me"},
            json={},
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"


def test_credentials_off_by_default_no_allow_credentials_header():
    # Pairing "*" with credentials would be a CORS spec violation AND would
    # opt the cookie-authenticated web surface into cross-origin reach. Pin
    # that the middleware never emits Access-Control-Allow-Credentials: true.
    with TestClient(_app()) as c:
        r = c.options(
            "/v1/completions",
            headers={
                "Origin": "https://vgel.me",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.headers.get("access-control-allow-credentials") != "true"


def test_explicit_origin_list_is_respected():
    with TestClient(_app(cors_allow_origins="https://vgel.me, https://other.example")) as c:
        r = c.options(
            "/v1/completions",
            headers={
                "Origin": "https://vgel.me",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "https://vgel.me"

        r2 = c.options(
            "/v1/completions",
            headers={
                "Origin": "https://blocked.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        # Starlette returns 400 for disallowed origins on preflight.
        assert r2.status_code == 400
