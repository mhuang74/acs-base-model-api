"""ACS-41: RequestIdMiddleware self-contained tests (no DB)."""

from __future__ import annotations

import asyncio
import re

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from wrapper.app import _REQUEST_ID_RE, RequestIdMiddleware, create_app, mint_request_id


def _bare_app() -> FastAPI:
    async def lifespan(app):
        yield

    limiter = Limiter(key_func=get_remote_address)
    app = create_app(lifespan=lifespan, limiter=limiter)
    app.state.limiter.enabled = False

    @app.get("/_probe")
    async def probe(request: Request):
        return {"rid_in_state": getattr(request.state, "request_id", None)}

    @app.get("/_error")
    async def error(request: Request):
        return JSONResponse(
            status_code=418,
            content={"error": {"message": "teapot"}, "request_id": request.state.request_id},
        )

    return app


def test_minted_request_id_matches_format_and_appears_in_header():
    with TestClient(_bare_app()) as c:
        r = c.get("/_probe")
        rid = r.headers["X-Request-Id"]
        assert re.fullmatch(r"req_[a-f0-9]{20}", rid), rid
        assert r.json()["rid_in_state"] == rid


def test_inbound_request_id_is_echoed_when_valid():
    with TestClient(_bare_app()) as c:
        r = c.get("/_probe", headers={"X-Request-Id": "my-trace-abc"})
        assert r.headers["X-Request-Id"] == "my-trace-abc"
        assert r.json()["rid_in_state"] == "my-trace-abc"


def test_inbound_request_id_rejected_when_malformed():
    with TestClient(_bare_app()) as c:
        r = c.get("/_probe", headers={"X-Request-Id": "bad id with spaces!"})
        rid = r.headers["X-Request-Id"]
        assert re.fullmatch(r"req_[a-f0-9]{20}", rid), rid


def test_error_body_request_id_matches_header():
    with TestClient(_bare_app()) as c:
        r = c.get("/_error")
        assert r.status_code == 418
        assert r.json()["request_id"] == r.headers["X-Request-Id"]


def test_mint_and_regex_agree():
    assert _REQUEST_ID_RE.match(mint_request_id())


# --- SSE / streaming regression guard (ACS-41 review follow-up) ---------------
#
# The reviewer flagged that BaseHTTPMiddleware is a known source of streaming
# regressions: it materializes the response body before forwarding, which would
# break /v1/completions stream:true and the workbench. The middleware is now
# pure-ASGI (wraps `send`, leaves body messages alone). This test pins that
# structural property by driving the middleware directly at the ASGI level and
# asserting:
#   (a) every http.response.body message the inner app emits flows through to
#       the outer send unmodified (no buffering / coalescing), and
#   (b) the X-Request-Id header is attached to http.response.start.
#
# (TestClient and httpx.ASGITransport both buffer chunks before exposing them,
# so they can't distinguish buffered vs. incremental at the wire — hence the
# ASGI-level test.)


@pytest.mark.asyncio
async def test_middleware_does_not_buffer_streaming_body_messages():
    async def inner_app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for i in range(5):
            await send(
                {
                    "type": "http.response.body",
                    "body": f"chunk{i}\n".encode(),
                    "more_body": i < 4,
                }
            )

    sent: list[dict] = []

    async def capture_send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/_stream",
        "headers": [],
    }

    middleware = RequestIdMiddleware(inner_app)
    await middleware(scope, receive, capture_send)

    # Exactly one start + five body messages should reach the outer send — same
    # count the inner app emitted. Coalescing would collapse the bodies into 1.
    starts = [m for m in sent if m["type"] == "http.response.start"]
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert len(starts) == 1, sent
    assert len(bodies) == 5, [b["body"] for b in bodies]
    assert [b["body"] for b in bodies] == [f"chunk{i}\n".encode() for i in range(5)]

    # X-Request-Id is attached on response.start.
    header_map = {k.lower(): v for k, v in starts[0]["headers"]}
    assert b"x-request-id" in header_map
    assert header_map[b"x-request-id"].decode().startswith("req_")
    # scope.state was populated for downstream handlers (catch-all 500, routes).
    assert scope["state"]["request_id"] == header_map[b"x-request-id"].decode()


@pytest.mark.asyncio
async def test_middleware_propagates_client_disconnect():
    # If the middleware swallowed the disconnect signal, _serve_stream's
    # `finally` couldn't release per-key semaphores on client cancel. Simulate a
    # disconnect: the inner app awaits its receive() and should see the
    # http.disconnect message we deliver.
    seen_disconnect = asyncio.Event()

    async def inner_app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send(
            {"type": "http.response.body", "body": b"first\n", "more_body": True}
        )
        msg = await receive()
        if msg["type"] == "http.disconnect":
            seen_disconnect.set()

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_msg):
        return None

    scope = {"type": "http", "method": "GET", "path": "/_stream", "headers": []}
    await RequestIdMiddleware(inner_app)(scope, receive, send)
    assert seen_disconnect.is_set()
