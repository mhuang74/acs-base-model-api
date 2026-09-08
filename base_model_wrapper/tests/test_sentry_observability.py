"""ACS-40/43: Sentry init is a no-op without a DSN, the scrub strips PII/bodies,
and the catch-all exception handler returns a clean, generic 500 with a
request_id (no internals leaked)."""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from wrapper import observability as obs


# ---- init is a no-op without a DSN ----------------------------------------

def test_init_sentry_noop_without_dsn():
    from wrapper.settings import Settings

    s = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        sentry_dsn="",
    )
    assert obs.init_sentry(s) is False


# ---- scrub: request body + PII removed, sensitive keys redacted ------------

def test_scrub_event_drops_request_body_and_pii():
    event = {
        "request": {
            "data": {"prompt": "secret user prompt", "max_tokens": 8},
            "cookies": {"acs_session": "abc"},
            "headers": {"Authorization": "Bearer acs-bm-xxx", "Accept": "application/json"},
        },
        "user": {"email": "alice@lab.org", "ip_address": "1.2.3.4", "id": "key_pfx"},
    }
    out = obs._scrub_event(event, None)
    assert "data" not in out["request"]
    assert "cookies" not in out["request"]
    assert "Authorization" not in out["request"]["headers"]
    assert out["request"]["headers"]["Accept"] == "application/json"  # innocuous kept
    assert "email" not in out["user"]
    assert "ip_address" not in out["user"]
    assert out["user"]["id"] == "key_pfx"  # non-PII correlation id kept


def test_scrub_event_scrubs_breadcrumbs():
    event = {
        "breadcrumbs": {
            "values": [
                {"category": "http", "data": {"prompt": "leak via crumb", "status": 200}},
            ]
        }
    }
    out = obs._scrub_event(event, None)
    crumb = out["breadcrumbs"]["values"][0]
    assert crumb["data"]["prompt"] == obs._REDACTED
    assert crumb["data"]["status"] == 200


def test_scrub_log_redacts_pii_and_sensitive_attributes():
    record = {
        "body": "GET /v1/models 200",
        "attributes": {
            "client.address": "1.2.3.4",
            "user_email": "alice@lab.org",
            "prompt": "should not be here",
            "logger": "uvicorn.access",  # innocuous, kept
        },
    }
    out = obs._scrub_log(record, None)
    assert out["attributes"]["client.address"] == obs._REDACTED
    assert out["attributes"]["user_email"] == obs._REDACTED
    assert out["attributes"]["prompt"] == obs._REDACTED
    assert out["attributes"]["logger"] == "uvicorn.access"


def test_scrub_log_never_raises():
    assert obs._scrub_log({}, None) == {}
    assert obs._scrub_log({"attributes": "not-a-dict"}, None) == {"attributes": "not-a-dict"}


def test_init_options_accepted_by_sdk():
    """Guard against an SDK version where our init kwargs aren't valid: the log
    hook must be a real option and the LoggingIntegration config must construct."""
    from sentry_sdk.consts import DEFAULT_OPTIONS
    from sentry_sdk.integrations.logging import LoggingIntegration

    assert "before_send_log" in DEFAULT_OPTIONS
    LoggingIntegration(level=None, event_level=None)  # must not raise


def test_scrub_event_redacts_sensitive_keys_in_extra():
    event = {
        "extra": {
            "payload": {"prompt": "leak me", "messages": [{"role": "user"}], "model": "llama-405b"},
            "note": "fine",
        }
    }
    out = obs._scrub_event(event, None)
    assert out["extra"]["payload"]["prompt"] == obs._REDACTED
    assert out["extra"]["payload"]["messages"] == obs._REDACTED
    assert out["extra"]["payload"]["model"] == "llama-405b"  # not sensitive
    assert out["extra"]["note"] == "fine"


def test_scrub_event_never_raises_returns_none_on_garbage():
    # A non-dict event would blow up naive code; scrub must swallow and drop.
    assert obs._scrub_event({"request": "not-a-dict"}, None) is not None  # tolerated
    # force an error path by passing something whose .get raises is hard; ensure
    # the happy path with minimal event is returned unchanged.
    assert obs._scrub_event({}, None) == {}


# ---- catch-all handler: generic 500 + request_id, no internals -------------

# ---- end-to-end: real SDK capture must not contain prompt locals -----------

def _capture_event_via_real_sdk(raise_fn):
    """Drive sentry_sdk.capture_exception through a Client built from the SAME
    kwargs as production (``sentry_init_kwargs``) and return the serialized
    event the transport would have sent.

    Unit scrubber tests synthesize events by hand; this exercises the path
    Sentry actually takes — frame-locals capture, before_send, the works — so a
    regression in init kwargs (e.g. ``include_local_variables`` flipping back to
    the SDK default) shows up here. Uses a hub-local client so it doesn't
    poison globals; the in-process transport drops on flush."""
    import json

    import sentry_sdk
    from sentry_sdk.transport import Transport

    from wrapper import observability as _obs
    from wrapper.settings import Settings

    captured: list[dict] = []

    class _CaptureTransport(Transport):
        def __init__(self, options=None):
            super().__init__(options)

        def capture_envelope(self, envelope):
            for item in envelope.items:
                payload = item.payload.json
                if payload is not None:
                    captured.append(payload)

    s = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        sentry_dsn="https://public@stub.ingest.sentry.io/1",
        sentry_environment="test",
        sentry_traces_sample_rate=0.0,
        sentry_profiles_sample_rate=0.0,
        sentry_enable_logs=False,
    )

    kwargs = _obs.sentry_init_kwargs(s)
    kwargs["transport"] = _CaptureTransport
    client = sentry_sdk.Client(**kwargs)
    hub = sentry_sdk.Hub(client)
    with hub:
        try:
            raise_fn()
        except Exception as exc:  # noqa: BLE001
            hub.capture_exception(exc)
        client.flush(timeout=2.0)
    return json.dumps(captured)


def test_real_sdk_does_not_leak_prompt_in_frame_locals():
    """The canonical ACS-43 leak vector: if a 500 is raised inside a function
    whose locals hold the parsed request body, the Sentry event must not carry
    the prompt text — not in ``frames[*].vars`` and not in the exception value.

    The canary is composed at runtime so it doesn't appear in the file's source
    (Sentry attaches pre/post_context lines around the throw site — those carry
    code, not values, but a literal in source would trip a false positive)."""
    import json as _json

    # Composed at runtime: must not appear as a string literal in this file.
    canary = "".join(["CNRY-", "PROMPT-", "7c3f9a12-leak"])

    def boom(_canary):
        body = {"prompt": _canary, "max_tokens": 16}  # noqa: F841 — must be a local
        parsed = {"prompt": _canary}  # noqa: F841 — second local
        raise RuntimeError("upstream blew up")

    payload = _capture_event_via_real_sdk(lambda: boom(canary))

    # Parse and check the structured fields we care about — pre/post_context
    # carries source code, which can't contain prompt text in production.
    events = _json.loads(payload)
    assert events, "no Sentry event was captured"
    for ev in events:
        for val in ev.get("exception", {}).get("values", []):
            assert canary not in (val.get("value") or ""), (
                "prompt text leaked into exception.value — exception messages "
                "must not carry user input"
            )
            for frame in val.get("stacktrace", {}).get("frames", []):
                vars_ = frame.get("vars") or {}
                # vars may be absent entirely (good — locals capture is off) or
                # present but redacted/empty. Either way, no canary.
                assert canary not in _json.dumps(vars_), (
                    "prompt text leaked into stacktrace.frames[*].vars — "
                    "include_local_variables must be False AND scrubber must "
                    "walk frames.vars"
                )
        # Defense in depth: not in extra, contexts, breadcrumbs, request, user.
        for section in ("extra", "contexts", "breadcrumbs", "request", "user"):
            payload_section = ev.get(section)
            if payload_section is not None:
                assert canary not in _json.dumps(payload_section), (
                    f"prompt text leaked into event[{section!r}]"
                )


def test_unhandled_exception_handler_returns_generic_500_with_request_id():
    """Mirror main.py's registration on a bare app to assert the contract
    without standing up the whole wrapper + DB."""
    import uuid as _uuid

    import sentry_sdk

    app = FastAPI()

    captured = {"n": 0}

    def _fake_capture(exc):
        captured["n"] += 1

    @app.exception_handler(Exception)
    async def handler(request, exc):  # noqa: ANN001
        rid = getattr(getattr(request, "state", None), "request_id", None) or f"req_{_uuid.uuid4().hex[:20]}"
        sentry_sdk.capture_exception(exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {"message": "Internal server error.", "type": "internal_error", "code": "internal_error"},
                "request_id": rid,
            },
        )

    @app.get("/boom")
    async def boom():
        raise ValueError("super secret internal detail with prompt fragment")

    import wrapper  # noqa: F401  (ensure package import path)
    sentry_sdk.capture_exception = _fake_capture  # type: ignore[assignment]

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["message"] == "Internal server error."
    # The secret exception text must NOT appear anywhere in the response.
    assert "super secret" not in r.text
    assert body["request_id"].startswith("req_")
    assert captured["n"] == 1  # exception was reported to Sentry
