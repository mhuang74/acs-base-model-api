"""Structured stdout logging.

Single helper for the privacy-critical per-request line. The plan's privacy
constraint is enforced by funnelling every request-meta log through this one
function, which only accepts pre-validated structured fields — no escape hatch
for body content.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def log_request_meta(
    *,
    request_id: str,
    key_id: str | None,
    key_prefix: str | None,
    user_email: str | None,
    ip: str | None,
    endpoint: str,
    model: str | None,
    n_prompt: int | None,
    n_completion: int | None,
    status: int,
    latency_ms: int,
    upstream_latency_ms: int | None,
    error_kind: str | None,
    stream: bool = False,
    cold_boot: bool = False,
    ttft_ms: int | None = None,
    workload_type: str | None = None,
) -> None:
    """Emit the single privacy-safe per-request log line.

    Only accepts structured metadata. No `prompt`, `completion`, `messages`, or
    body content is permitted in this signature — that is the guarantee the
    plan's privacy commitment rests on. The added beta-telemetry fields
    (``stream`` / ``cold_boot`` / ``ttft_ms`` / ``workload_type``) are likewise
    pure metadata; the richer sampling-param fields live only on the DB row to
    keep this stdout line readable.
    """
    log = structlog.get_logger("request")
    log.info(
        "request",
        request_id=request_id,
        key_id=key_id,
        key_prefix=key_prefix,
        user_email=user_email,
        ip=ip,
        endpoint=endpoint,
        model=model,
        n_prompt=n_prompt,
        n_completion=n_completion,
        status=status,
        latency_ms=latency_ms,
        upstream_latency_ms=upstream_latency_ms,
        error_kind=error_kind,
        stream=stream,
        cold_boot=cold_boot,
        ttft_ms=ttft_ms,
        workload_type=workload_type,
    )


def log_auth_failure(
    *,
    request_id: str,
    endpoint: str,
    error_kind: str,
    status: int,
    key_prefix: str | None = None,
    ip: str | None = None,
) -> None:
    """Emit one structured line for an auth rejection (401/403).

    Auth failures are raised before the normal per-request logging/DB path
    runs, so without this they leave no structured trace — only a bare uvicorn
    access line with no request_id/key_prefix, making auth-incident triage and
    key-guessing detection impossible.

    Privacy invariant matches ``log_request_meta``: only metadata. ``key_prefix``
    is the public ``acs-bm-<prefix>`` segment of a *presented* token (never the
    full token, secret, or hash) and is None when the token was missing or
    unparseable. No prompt/response/body content.
    """
    log = structlog.get_logger("auth")
    log.warning(
        "auth_failure",
        request_id=request_id,
        endpoint=endpoint,
        error_kind=error_kind,
        status=status,
        key_prefix=key_prefix,
        ip=ip,
    )


def get_logger(name: str = "wrapper") -> Any:
    return structlog.get_logger(name)
