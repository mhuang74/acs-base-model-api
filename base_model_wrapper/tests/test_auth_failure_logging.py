"""Auth failures must leave a structured trace (observability fix).

401/403s are raised before the normal per-request logging/DB path runs, so
without an explicit emit they produce only a bare uvicorn access line — no
request_id, no key_prefix — making auth-incident triage and key-guessing
detection impossible. These tests pin the behaviour that every rejection now
emits exactly one structured ``auth_failure`` line (request_id, endpoint,
error_kind, key_prefix, ip), while preserving the privacy invariant (never the
full token/secret/hash) and still raising the HTTPException.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, status

from wrapper import auth as authmod
from wrapper import logging as logmod


def _request(headers: dict[str, str], *, path: str = "/v1/completions",
             request_id: str = "req_abc", client_host: str = "203.0.113.7"):
    req = MagicMock()
    req.headers = headers
    req.url.path = path
    # _raise_auth_error reads the path from the ASGI scope (request.scope.get(
    # "path")), not request.url.path — auth failures fire before the full scope
    # is assembled, so it avoids building request.url. Mock the scope as a real
    # dict so .get() returns the path instead of an unconfigured MagicMock.
    req.scope = {"path": path}
    req.state.request_id = request_id
    req.client.host = client_host
    return req


def test_missing_authorization_emits_structured_line_and_raises():
    req = _request({})  # no Authorization header
    with patch.object(authmod, "log_auth_failure") as mock_log:
        with pytest.raises(HTTPException) as exc_info:
            authmod._extract_bearer(req, log_ip=True)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["error_kind"] == "missing_authorization"
    assert kwargs["request_id"] == "req_abc"
    assert kwargs["endpoint"] == "/v1/completions"
    assert kwargs["key_prefix"] is None  # nothing parseable was presented
    assert kwargs["ip"] == "203.0.113.7"  # log_ip=True


def test_malformed_bearer_emits_missing_authorization():
    req = _request({"authorization": "Token foo"})
    with patch.object(authmod, "log_auth_failure") as mock_log:
        with pytest.raises(HTTPException):
            authmod._extract_bearer(req, log_ip=False)
    kwargs = mock_log.call_args.kwargs
    assert kwargs["error_kind"] == "missing_authorization"
    assert kwargs["ip"] is None  # log_ip=False → ip suppressed


def test_log_ip_false_suppresses_ip():
    req = _request({})
    with patch.object(authmod, "log_auth_failure") as mock_log:
        with pytest.raises(HTTPException):
            authmod._extract_bearer(req, log_ip=False)
    assert mock_log.call_args.kwargs["ip"] is None


def test_raise_auth_error_logs_key_prefix_only_never_secret():
    """A presented prefix is logged for triage, but the helper signature makes
    it impossible to pass a full token/secret — only the public prefix flows."""
    req = _request({})
    with patch.object(authmod, "log_auth_failure") as mock_log:
        with pytest.raises(HTTPException):
            authmod._raise_auth_error(
                req,
                status_code=status.HTTP_401_UNAUTHORIZED,
                error_kind="invalid_api_key",
                message="Invalid API key.",
                code="invalid_api_key",
                key_prefix="ab12cd34",
                log_ip=False,
            )
    kwargs = mock_log.call_args.kwargs
    assert kwargs["key_prefix"] == "ab12cd34"
    # No way for a secret/hash to ride along: the only key-related kwarg is the
    # 8-char prefix.
    assert set(kwargs) == {
        "request_id", "endpoint", "error_kind", "status", "key_prefix", "ip"
    }


def test_log_auth_failure_emits_warning_with_metadata():
    """The helper emits one structured ``auth_failure`` warning carrying only
    metadata (request_id / endpoint / error_kind / key_prefix / ip)."""
    captured = {}

    class _Recorder:
        def warning(self, event, **kw):
            captured["event"] = event
            captured["kw"] = kw

    with patch.object(logmod.structlog, "get_logger", return_value=_Recorder()):
        logmod.log_auth_failure(
            request_id="req_xyz",
            endpoint="/v1/models",
            error_kind="key_revoked",
            status=401,
            key_prefix="deadbeef",
            ip=None,
        )

    assert captured["event"] == "auth_failure"
    assert captured["kw"]["error_kind"] == "key_revoked"
    assert captured["kw"]["request_id"] == "req_xyz"
    assert captured["kw"]["key_prefix"] == "deadbeef"
    # Privacy invariant: no body/secret fields.
    forbidden = {"prompt", "completion", "messages", "token", "secret", "key_hash"}
    assert not (set(captured["kw"]) & forbidden)
