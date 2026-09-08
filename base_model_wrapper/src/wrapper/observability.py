"""Sentry error tracking (ACS-40), wired with a hard privacy posture (ACS-43).

The headline product guarantee is that prompts / completions / logprobs are
never logged or stored off-box. Sentry is a third party, so the init here is
deliberately conservative:

- ``send_default_pii=False`` — Sentry will NOT auto-attach request bodies,
  headers, cookies, or client IP to events. This is the single most important
  flag; the wizard's default (`True`) would ship prompt text in a 500 trace.
- ``max_request_body_size="never"`` — belt: never capture request bodies.
- ``before_send`` / ``before_send_transaction`` — suspenders: drop any request
  data that slipped through, strip ``user.email`` / ``user.ip_address`` (PII we
  keep in Railway logs only), and recursively redact known prompt-bearing keys.

Log surfaces are locked down on three fronts (a code review flagged that the
SDK forwards logs via paths the event scrub doesn't cover):
- ``LoggingIntegration(level=None, event_level=None)`` — stdlib logs become
  neither Sentry breadcrumbs nor events. This kills the uvicorn-access-log
  (client IP) and third-party-lib paths entirely.
- ``before_send_log=_scrub_log`` — the Sentry Logs feature (``enable_logs``)
  has its OWN hook; we scrub PII/prompt-bearing attributes there too.
- ``_scrub_event`` also redacts ``breadcrumbs`` (e.g. HTTP/db integration crumbs).
Our own privacy-critical per-request log line already bypasses all of this: it
goes through structlog's ``PrintLogger`` (see ``logging.py``) straight to stdout,
never through stdlib ``logging``. Railway keeps the logs; Sentry's job is
exceptions + tracing.

Init is a no-op when ``SENTRY_DSN`` is unset, so local dev and tests never talk
to Sentry.
"""

from __future__ import annotations

from typing import Any

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from .logging import get_logger
from .settings import Settings

log = get_logger("sentry")

# Substrings that mark a PII-bearing attribute key on a forwarded log record
# (uvicorn access logs carry client IPs; some libs attach emails).
_PII_ATTR_SUBSTRINGS = ("email", "ip_address", "client_addr", "remote_addr", "client.address")

# Keys that may carry user content; redacted recursively anywhere they appear in
# an outgoing event (request data is already dropped, this is defense in depth).
_SENSITIVE_KEYS = frozenset(
    {
        "prompt",
        "completion",
        "completions",
        "messages",
        "logprobs",
        "prompt_logprobs",
        "input",
        "body",
    }
)
_REDACTED = "[redacted-acs]"


def _redact(obj: Any) -> Any:
    """Recursively replace values under sensitive keys. Bounded by event size."""
    if isinstance(obj, dict):
        return {
            k: (_REDACTED if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def _scrub_event(event: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """before_send / before_send_transaction hook. Strips request bodies + PII.

    Must never raise — a scrub failure should drop the event, not crash the app.
    """
    try:
        req = event.get("request")
        if isinstance(req, dict):
            req.pop("data", None)  # request body (prompts)
            req.pop("cookies", None)
            headers = req.get("headers")
            if isinstance(headers, dict):
                for h in ("Authorization", "authorization", "Cookie", "cookie"):
                    headers.pop(h, None)
        user = event.get("user")
        if isinstance(user, dict):
            # Keep nothing that identifies the person; key_prefix/request_id (set
            # as tags elsewhere) are enough to correlate with Railway logs.
            user.pop("email", None)
            user.pop("ip_address", None)
            user.pop("username", None)
        for section in ("extra", "contexts"):
            if isinstance(event.get(section), (dict, list)):
                event[section] = _redact(event[section])
        # Breadcrumbs (e.g. from HTTP/db integrations) can carry keyed data;
        # redact known sensitive keys within them too. Stdlib-log breadcrumbs
        # are already off (LoggingIntegration disabled in init_sentry).
        if "breadcrumbs" in event:
            event["breadcrumbs"] = _redact(event["breadcrumbs"])
        # Exception frame locals: narrow safety net. The load-bearing fix is
        # ``include_local_variables=False`` in the init kwargs — this walk only
        # blanks frame-var entries whose KEY is in ``_SENSITIVE_KEYS`` (e.g. a
        # local literally named ``prompt`` / ``body`` / ``messages``). It can't
        # see inside Sentry's serialized ``repr(value)`` strings, so a local
        # named ``parsed`` or ``raw_body`` holding prompt text would still
        # leak if locals capture is ever re-enabled. Do not read this as
        # latitude to flip the kwarg back on.
        exc = event.get("exception")
        if isinstance(exc, dict):
            for val in exc.get("values", []) or []:
                if not isinstance(val, dict):
                    continue
                stack = val.get("stacktrace")
                if not isinstance(stack, dict):
                    continue
                for frame in stack.get("frames", []) or []:
                    if isinstance(frame, dict) and isinstance(frame.get("vars"), dict):
                        frame["vars"] = _redact(frame["vars"])
        return event
    except Exception:  # noqa: BLE001 — never let scrubbing crash; drop instead
        return None


def _scrub_log(record: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """before_send_log hook for the Sentry Logs feature (``enable_logs``).

    The structured-logs product has its OWN hook, separate from ``before_send``
    — without this, forwarded log attributes would skip the event scrub. Redacts
    PII / prompt-bearing attribute keys. Free-text log bodies are not parsed
    (can't reliably); we rely on the app's own logs being body-free (structlog
    PrintLogger bypasses stdlib) and stdlib capture being disabled. Never raises.
    """
    try:
        attrs = record.get("attributes")
        if isinstance(attrs, dict):
            for k in list(attrs):
                kl = k.lower() if isinstance(k, str) else ""
                if kl in _SENSITIVE_KEYS or any(s in kl for s in _PII_ATTR_SUBSTRINGS):
                    attrs[k] = _REDACTED
            record["attributes"] = _redact(attrs)
        return record
    except Exception:  # noqa: BLE001 — never let scrubbing crash; drop instead
        return None


def sentry_init_kwargs(settings: Settings) -> dict[str, Any]:
    """The Sentry init config, factored out so tests can build an identical
    isolated Client and verify the real SDK doesn't leak prompts/locals.

    Any privacy-relevant flag added here applies to production AND to the
    end-to-end Sentry tests — they can't drift.
    """
    return dict(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=settings.sentry_release or None,
        # --- privacy (ACS-43) ---
        send_default_pii=False,
        max_request_body_size="never",
        # Frame-locals capture is OFF: Sentry's default attaches the value of
        # every local variable to each stack frame in an exception event. In the
        # completions path those locals include `body` / `parsed` / `prompt`, so
        # leaving this on would ship prompt text to Sentry on every 500. The
        # event scrubber also walks frame vars defensively in case this flips.
        include_local_variables=False,
        before_send=_scrub_event,
        before_send_transaction=_scrub_event,
        before_send_log=_scrub_log,
        # Disable stdlib-log capture entirely: no INFO→breadcrumb, no ERROR→event.
        # Stdlib logs (uvicorn access lines with client IPs, third-party libs)
        # must not become Sentry breadcrumbs/events unscrubbed. Sentry's job here
        # is exceptions + tracing; Railway keeps the logs. FastAPI/Starlette
        # integrations still auto-enable (default_integrations=True).
        integrations=[LoggingIntegration(level=None, event_level=None)],
        # --- volume / features ---
        traces_sample_rate=settings.sentry_traces_sample_rate,
        profiles_sample_rate=settings.sentry_profiles_sample_rate,
        enable_logs=settings.sentry_enable_logs,
    )


def init_sentry(settings: Settings) -> bool:
    """Initialize Sentry if a DSN is configured. Returns True if initialized.

    Safe to call once at startup. No-op (returns False) without ``SENTRY_DSN``.
    """
    if not settings.sentry_dsn:
        log.info("sentry_disabled", reason="no SENTRY_DSN")
        return False

    sentry_sdk.init(**sentry_init_kwargs(settings))
    log.info(
        "sentry_initialized",
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        enable_logs=settings.sentry_enable_logs,
    )
    return True
