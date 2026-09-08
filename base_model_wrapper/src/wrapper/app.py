"""FastAPI application factory.

The production ASGI target remains ``wrapper.main:app``. This module keeps
the app shell separate from route registration so future router extraction can
reuse the same construction path without changing the deployment import.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from slowapi import Limiter

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# The OpenAPI spec is scoped to the public JSON API only: the `/v1/*` endpoints
# plus the unauthenticated `/health`. Everything else the app serves — the
# cookie-authed `/admin/*`, `/workbench`, `/loom`, `/dashboard`, and the HTML
# marketing/tutorial pages — is deliberately excluded so the published spec
# never advertises internal admin routes or leaks their request/response model
# shapes. We build the spec from an explicit route allowlist (not by
# post-filtering `paths`) so `components.schemas` is pruned to public models too.
_PUBLIC_API_PREFIXES = ("/v1/",)
_PUBLIC_API_EXACT = frozenset({"/health"})


def _is_public_api_route(path: str) -> bool:
    return path in _PUBLIC_API_EXACT or path.startswith(_PUBLIC_API_PREFIXES)


def _inject_request_body_schemas(
    schema: dict[str, Any], body_models: dict[str, Any]
) -> None:
    """Attach hand-validated request bodies to the OpenAPI operation objects.

    The `/v1/completions` and `/v1/harvest` handlers parse the raw JSON body and
    call ``model_validate`` themselves, so ``get_openapi`` emits no requestBody
    for them. We pull the JSON schema straight from the Pydantic model (single
    source of truth — it can't drift from what the route actually enforces),
    land the model plus any nested ``$defs`` (e.g. ``SteeringVector``) into
    ``components.schemas``, and point the operation's requestBody at the model
    by ``$ref``. Field constraints then render — including on Optional fields,
    where they sit inside ``anyOf`` (e.g. ``seed`` → ``anyOf[0].minimum: 0``).
    """
    if not body_models:
        return
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for path, model in body_models.items():
        path_item = schema.get("paths", {}).get(path)
        if not path_item:
            continue
        model_schema = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        # Promote nested model definitions ($defs) to top-level components so the
        # $refs inside model_schema resolve against the OpenAPI document root.
        for def_name, def_schema in model_schema.pop("$defs", {}).items():
            components.setdefault(def_name, def_schema)
        components[model.__name__] = model_schema
        body = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": f"#/components/schemas/{model.__name__}"}
                }
            },
        }
        # Both routes are POST-only; set the body on whatever operation(s) exist
        # rather than hard-coding "post", so this stays correct if a method is
        # added later.
        for method in ("post", "put", "patch"):
            if method in path_item:
                path_item[method]["requestBody"] = body


def mint_request_id() -> str:
    # `req_<hex20>` matches _REQUEST_ID_RE exactly, so a minted ID round-trips
    # through inbound validation in another hop without being rejected.
    return f"req_{uuid.uuid4().hex[:20]}"


class RequestIdMiddleware:
    # Pure-ASGI (not BaseHTTPMiddleware): the latter buffers StreamingResponse
    # bodies, breaking SSE on /v1/completions and the workbench. We only need
    # to read one inbound header and inject one outbound header, so wrap `send`
    # to attach X-Request-Id on http.response.start and let body messages pass
    # through untouched.

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = ""
        for name, value in scope.get("headers", ()):
            if name == b"x-request-id":
                inbound = value.decode("latin-1", "replace").strip()
                break
        request_id = inbound if _REQUEST_ID_RE.match(inbound) else mint_request_id()

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        rid_bytes = request_id.encode("latin-1")

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in message.get("headers", ())
                    if k.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", rid_bytes))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app(
    *,
    lifespan: Any,
    limiter: Limiter,
    cors_allow_origins: str = "*",
    request_body_models: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the wrapper FastAPI app with shared middleware installed.

    ``request_body_models`` maps a public route path (e.g. ``/v1/completions``)
    to the Pydantic model its body is validated against. Those routes read the
    raw JSON body and call ``Model.model_validate`` by hand (to keep the custom
    400 error contract), so FastAPI can't infer the request schema from the
    handler signature — we inject it into the OpenAPI spec explicitly so the
    field constraints (``seed>=0``, ``temperature<=100``, …) are discoverable.
    """
    body_models = request_body_models or {}
    app = FastAPI(
        title="ACS Infra API wrapper",
        version="0.1.0",
        lifespan=lifespan,
        # Swagger UI at /docs is safe to publish now that the spec is scoped to
        # the public API (see custom_openapi below): it renders only /v1/* +
        # /health, never the admin surface. ReDoc stays off (one UI is enough).
        docs_url="/docs",
        redoc_url=None,
    )

    def custom_openapi() -> dict[str, Any]:
        # Scope the published spec to the public API surface. Built once and
        # cached on app.openapi_schema (FastAPI's standard override pattern).
        # `app.routes` is read lazily here — by the time /openapi.json or /docs
        # is first hit, every router (incl. admin) is already registered, and we
        # keep only the public ones so admin paths and their models never appear.
        if app.openapi_schema is not None:
            return app.openapi_schema
        public_routes = [
            route
            for route in app.routes
            if isinstance(route, APIRoute) and _is_public_api_route(route.path)
        ]
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description or None,
            routes=public_routes,
        )
        _inject_request_body_schemas(schema, body_models)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    app.state.limiter = limiter
    # NB: we deliberately do NOT add slowapi's ``SlowAPIMiddleware`` (ACS-347).
    # It is a Starlette ``BaseHTTPMiddleware``, which buffers and *re-emits* every
    # response through an internal ``_StreamingResponse`` — the same reason
    # ``RequestIdMiddleware`` above is hand-written pure-ASGI. That re-emission
    # intermittently raised ``RuntimeError: Response content longer than
    # Content-Length`` on ``/v1/completions`` (a prod 5xx), and needlessly
    # buffers the multi-GB full-vocab / activation bodies. The middleware is not
    # needed for enforcement: rate limiting is applied by the per-route
    # ``@limiter.limit`` decorators (which raise ``RateLimitExceeded`` inline),
    # handled by the ``RateLimitExceeded`` handler, and keyed off
    # ``app.state.limiter`` set just above. There are no ``default_limits`` and
    # ``headers_enabled`` is off, so ``SlowAPIMiddleware`` was a no-op here apart
    # from the harmful re-emission.
    app.add_middleware(RequestIdMiddleware)
    # CORS for /v1 (ACS-146). allow_credentials=False is deliberate: with
    # credentials off the browser will not attach the acs_session cookie on
    # cross-origin requests, so the cookie-authenticated web/admin/workbench
    # surface gains no new CSRF reachability from this. Never pair "*" with
    # allow_credentials=True.
    origins = [o.strip() for o in cors_allow_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        # DELETE: the ACS-344 harvest-cancel route is unreachable from a browser
        # without it, which defeats ACS-146 for that endpoint.
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        # "*" not ["Authorization", "Content-Type"]: the OpenAI JS SDK (used by
        # browser tools like logitloom) attaches x-stainless-* telemetry headers,
        # which a narrow allowlist rejects at preflight with 400 "Disallowed CORS
        # headers" — defeating ACS-146's whole point. Safe here because
        # allow_credentials=False (no cookie reachability; "*" never pairs with
        # credentials=True).
        allow_headers=["*"],
        # Browser clients can only read non-simple response headers that are
        # explicitly exposed — without this, the headers our docs tell them to
        # check are invisible to fetch() (ACS-321, ACS-344).
        expose_headers=["X-Acs-Longpoll", "Retry-After"],
    )
    return app
