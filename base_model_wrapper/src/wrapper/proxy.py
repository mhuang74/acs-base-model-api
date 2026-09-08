"""Proxy /v1/completions (and /v1/models) to the upstream Modal/vLLM endpoint.

Streaming path:
  - Open `httpx.AsyncClient.stream(...)` against Modal.
  - Yield SSE chunks as they arrive.
  - Inject `stream_options.include_usage=true` so the final chunk carries
    `usage` regardless of what the caller asked.
  - Parse the final `[DONE]`-adjacent usage chunk; commit to DB.

Non-streaming path is the same idea minus the generator.

Cold-boot handling:
  - Keep one upstream invocation alive. Modal returns HTTP 303 after a
    long-running Web Function hop and puts the same invocation's result URL in
    Location. Follow that URL as GET; never re-POST or apply a 25/60 s read
    timeout, because either can cancel/restart a large-model cold boot.
  - Public routes independently send JSON-whitespace / SSE-comment keepalives
    downstream while this proxy call waits.

Network / 5xx handling (reliability-hardening):
  - httpx ConnectError, ReadError, TimeoutException: caught and re-raised as
    ``UpstreamUnreachable`` so the caller produces a 502 with a structured
    error code instead of leaking an unhandled 500.
  - Upstream HTTP 5xx (502, 503, 504): retried up to ``RETRY_5XX_MAX_RETRIES``
    times with exponential backoff + jitter. Idempotent for vLLM's
    /v1/completions (same seed → same output; no side effects without seed).
  - 4xx is never retried.
  - The streaming variant only retries when no chunk has been yielded yet —
    once the client has bytes, restart would produce corrupt output.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import HTTPException, status

from .error_kinds import CLIENT_ISH_UPSTREAM_KINDS, ErrorKind

SSE_DONE = b"data: [DONE]"

# Modal Web Functions hold a request for up to 150 seconds, then return a 303
# whose Location is a result URL for *that same invocation*. Following that URL
# is the supported continuation protocol. Re-POSTing starts a second input and
# short read timeouts cancel the input before a 405B container can become warm.
MODAL_CONTINUATION_MAX_REDIRECTS = 20

# Railway's public HTTP hard ceiling is 15 minutes. Stop a little early so the
# wrapper can serialize a structured late-failure body instead of letting the
# edge reset the socket at exactly 900 seconds.
PUBLIC_REQUEST_TIMEOUT_S = 14 * 60.0

# Workbench status events are retained for its progress UI. The public routes
# now keep one Modal invocation alive and emit downstream keepalives instead of
# asking the client to retry a 503.
COLD_BOOT_MAX_RETRIES_WORKBENCH = 1_000_000
COLD_BOOT_BACKOFF_S = 30.0

# Status-event cadence during a cold-boot backoff. Workbench streaming path
# yields a `phase: cold_boot` event every ~5 s while sleeping so the banner
# timer stays alive and server-authoritative.
COLD_BOOT_STATUS_INTERVAL_S = 5.0

# Retry policy for non-cold-boot upstream 5xx and network errors. Kept low
# so a sustained outage doesn't quadruple our load on a struggling backend;
# the circuit breaker takes over after consecutive failures pile up.
RETRY_5XX_MAX_RETRIES = 3
RETRY_5XX_BASE_BACKOFF_S = 1.0  # exponential: 1s, 2s, 4s + jitter
RETRY_5XX_JITTER_S = 0.5


@dataclass(frozen=True)
class BackendContext:
    """Identity of the upstream a request is bound for.

    Threaded into ``post_nonstream`` / ``stream_post`` so error payloads,
    response headers, and structured logs can name *which* backend failed.
    Generic "upstream_5xx" without backend identity is useless during
    incidents.

    Constructed in the request handler from the matched ``ModelEntry``.
    Empty strings (rather than None) so the header values are always safe
    to serialise without conditionals.
    """

    model_id: str = ""
    gpu_shape: str = ""
    modal_app_name: str = ""
    # ``cold_hint`` = the request handler believes this model is currently
    # scaled-to-zero (warm-state heuristic / runner count == 0). When True, a
    # connect/read timeout or stall on the *first* attempt is classified as a
    # cold boot (→ ColdBootError → 503 modal_cold_boot) rather than as
    # ``UpstreamUnreachable`` (502). When False (model believed warm) a timeout
    # keeps the old meaning: a genuine outage, retried then surfaced as 502.
    # Default False so callers that don't know warm-state get the conservative
    # (502-on-timeout) behaviour and existing tests are unaffected.
    cold_hint: bool = False

    def as_log_fields(self) -> dict[str, str]:
        return {
            "upstream_model": self.model_id,
            "upstream_gpu": self.gpu_shape,
            "upstream_app": self.modal_app_name,
        }


class UpstreamError(HTTPException):
    pass


class UpstreamUnreachable(Exception):
    """Raised when httpx can't establish or maintain a connection to upstream.

    Distinct from ``ColdBootError`` (Modal is starting, retry might work) and
    ``UpstreamServerError`` (upstream answered with 5xx). This one means the
    network layer failed: DNS, TCP, TLS, read/write timeout on an active
    socket. The caller produces a 502.
    """

    def __init__(self, *, reason: str, attempts: int, ctx: BackendContext | None = None) -> None:
        super().__init__(f"upstream unreachable after {attempts} attempts: {reason}")
        self.reason = reason
        self.attempts = attempts
        self.ctx = ctx or BackendContext()


class UpstreamServerError(Exception):
    """Raised when upstream returns HTTP 5xx past the retry budget.

    The caller produces a 502 (the wrapper is acting as a gateway whose
    upstream is misbehaving). Kept separate from ``UpstreamUnreachable`` so
    the circuit breaker can weight them differently if we ever want to.
    """

    def __init__(
        self,
        *,
        upstream_status: int,
        attempts: int,
        body_excerpt: str = "",
        upstream_kind: str | None = None,
        ctx: BackendContext | None = None,
    ) -> None:
        super().__init__(
            f"upstream returned {upstream_status} after {attempts} attempts"
            + (f" ({upstream_kind})" if upstream_kind else "")
        )
        self.upstream_status = upstream_status
        self.attempts = attempts
        self.body_excerpt = body_excerpt
        self.upstream_kind = upstream_kind
        self.ctx = ctx or BackendContext()


# Patterns we look for in upstream error bodies to label vLLM-specific
# failures. Order matters: first match wins, so put the more specific
# patterns first. Conservative regex — false negatives are fine, false
# positives (mislabelled as a vllm error when it was a network blip) are
# not, because they'd trip the circuit breaker differently than intended.
_VLLM_ERROR_PATTERNS: tuple[tuple[ErrorKind, re.Pattern[str]], ...] = (
    (ErrorKind.vllm_context_length, re.compile(r"(maximum context length|max_model_len|exceeds.*token)", re.IGNORECASE)),
    (ErrorKind.vllm_oom, re.compile(r"(out of memory|CUDA out of memory|OOM|cuda_oom)", re.IGNORECASE)),
    (ErrorKind.vllm_engine_dead, re.compile(r"(engine.*(dead|crashed|terminated)|worker.*crashed)", re.IGNORECASE)),
    (ErrorKind.vllm_invalid_request, re.compile(r"(invalid_request|bad request)", re.IGNORECASE)),
    # vLLM-Lens raises ValueError -> 500 for an out-of-range steering
    # ``layer_index`` (activation engine). A deterministic client mistake, not a
    # server fault: classify it so the proxy stops retrying and the /v1 handler
    # returns 400, not a mislabelled upstream 5xx (ACS-322).
    (ErrorKind.vllm_invalid_request, re.compile(r"layer_index\s+\d+\s+out of range", re.IGNORECASE)),
)


def classify_upstream_error_body(body: Any) -> str | None:
    """Return a labelled error kind for known vLLM failure modes, else None.

    Inspects up to 4 KiB of the upstream's response body (parsed JSON, raw
    bytes, or str) and matches against ``_VLLM_ERROR_PATTERNS``. Used by the
    proxy to tag ``X-Acs-Upstream-Error-Kind`` and ``error_kind`` in the
    audit log so triage can tell "model OOMed" from "DNS failure" at a
    glance.

    Conservative: when the body doesn't match any pattern, returns None
    and the caller falls back to a generic upstream tag.
    """
    text = _excerpt_body(body, limit=4096)
    if not text:
        return None
    for label, pattern in _VLLM_ERROR_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _excerpt_body(body: Any, *, limit: int) -> str:
    """Coerce body into a short string for pattern-matching + log inclusion."""
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        try:
            return body.decode("utf-8", errors="replace")[:limit]
        except Exception:
            return ""
    if isinstance(body, str):
        return body[:limit]
    try:
        return json.dumps(body)[:limit]
    except Exception:
        return repr(body)[:limit]


def _exp_backoff(attempt: int) -> float:
    """1s, 2s, 4s, ... + uniform jitter. Cap at ~8s so a 3-retry sequence
    completes in <20s wall time. Module-level so tests can monkeypatch the
    sleep call (``asyncio.sleep``) to zero without monkey-patching this."""
    base = RETRY_5XX_BASE_BACKOFF_S * (2 ** min(attempt, 4))
    return base + random.uniform(0.0, RETRY_5XX_JITTER_S)


class ColdBootError(Exception):
    """Raised when the upstream keeps signalling "not ready" past our retry budget.

    Carries the last upstream status code observed (303 in practice) so the
    caller can include it in the structured error body it returns to the user.
    """

    def __init__(self, upstream_status: int, attempts: int) -> None:
        super().__init__(
            f"upstream not ready after {attempts} attempts (last status {upstream_status})"
        )
        self.upstream_status = upstream_status
        self.attempts = attempts


def _is_cold_boot_status(code: int) -> bool:
    # Modal returns 303 for "container starting"; treat any 3xx defensively
    # since we explicitly disabled redirect following on these POSTs.
    return 300 <= code < 400


def _looks_like_completion(payload: Any) -> bool:
    """Best-effort sanity check on a non-streaming /v1/completions body.

    The upstream is expected to return ``{"choices": [...], "usage": {...}, ...}``.
    If neither key is present, the body is almost certainly not a completion
    (e.g. a redirect HTML page or some other not-ready surface), and we want
    to treat it as a cold boot rather than forwarding ``n_completion=null``.
    """
    if not isinstance(payload, dict):
        return False
    return "choices" in payload or "usage" in payload


def _inject_include_usage(body: dict[str, Any]) -> None:
    if body.get("stream"):
        opts = body.get("stream_options") or {}
        if not isinstance(opts, dict):
            opts = {}
        opts["include_usage"] = True
        body["stream_options"] = opts


def clamp_max_tokens(body: dict[str, Any], remaining: int | None) -> tuple[int | None, int | None]:
    """Clamp body['max_tokens'] to `remaining`. Returns (original, clamped).

    `remaining` of None means unlimited — no-op.
    Caller is responsible for rejecting outright when remaining <= 0.
    """
    if remaining is None:
        return (body.get("max_tokens"), body.get("max_tokens"))
    original = body.get("max_tokens")
    if original is None:
        body["max_tokens"] = remaining
        return (None, remaining)
    if original > remaining:
        body["max_tokens"] = remaining
        return (original, remaining)
    return (original, original)


async def post_nonstream(
    client: httpx.AsyncClient,
    upstream_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
    *,
    ctx: BackendContext | None = None,
) -> tuple[int, dict[str, Any], int]:
    """POST once and follow Modal's 303 result URL until completion.

    Modal returns a 303 after a long-running Web Function has occupied one HTTP
    hop for 150 seconds. ``Location`` identifies the *same invocation*; it must
    be followed as GET. Re-POSTing or applying a short read timeout can cancel
    the cold input and start over indefinitely.
    """
    ctx = ctx or BackendContext()
    t0 = time.monotonic()
    deadline = t0 + min(timeout_s, PUBLIC_REQUEST_TIMEOUT_S)
    method = "POST"
    url = upstream_url
    request_body: dict[str, Any] | None = body
    redirects = 0
    last_status = 0

    while True:
        try:
            r = await _request_with_5xx_retry(
                client,
                method,
                url,
                api_key,
                request_body,
                deadline=deadline,
                ctx=ctx,
            )
        except TimeoutError as exc:
            if ctx.cold_hint or redirects:
                raise ColdBootError(last_status, max(1, redirects)) from exc
            raise UpstreamUnreachable(
                reason="overall upstream request deadline exceeded",
                attempts=1,
                ctx=ctx,
            ) from exc

        last_status = r.status_code
        if r.status_code == 303:
            redirects += 1
            if redirects > MODAL_CONTINUATION_MAX_REDIRECTS:
                raise ColdBootError(r.status_code, redirects)
            url = _modal_continuation_url(r, url, ctx)
            method = "GET"
            request_body = None
            continue
        if 300 <= r.status_code < 400:
            raise UpstreamUnreachable(
                reason=f"unexpected upstream redirect status {r.status_code}",
                attempts=1,
                ctx=ctx,
            )

        try:
            payload = r.json()
        except json.JSONDecodeError:
            payload = {"raw": r.text}
        if r.status_code == 200 and not _looks_like_completion(payload):
            raise UpstreamUnreachable(
                reason="upstream returned 200 with a non-completion body",
                attempts=1,
                ctx=ctx,
            )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return r.status_code, payload, elapsed_ms


def _modal_continuation_url(
    response: httpx.Response,
    current_url: str,
    ctx: BackendContext,
) -> str:
    """Resolve a same-origin Modal 303 Location without leaking credentials."""
    location = response.headers.get("location")
    if not location:
        raise UpstreamUnreachable(
            reason="Modal continuation response omitted Location",
            attempts=1,
            ctx=ctx,
        )
    try:
        target = urljoin(current_url, location)
        current = urlparse(current_url)
        parsed = urlparse(target)
        # Accessing hostname performs additional bracket/port validation.
        _ = parsed.hostname
    except ValueError as exc:
        raise UpstreamUnreachable(
            reason="Modal continuation returned invalid Location",
            attempts=1,
            ctx=ctx,
        ) from exc
    if parsed.scheme != current.scheme or parsed.netloc != current.netloc:
        raise UpstreamUnreachable(
            reason="Modal continuation redirected to a different origin",
            attempts=1,
            ctx=ctx,
        )
    return target


async def _request_with_5xx_retry(
    client: httpx.AsyncClient,
    method: str,
    upstream_url: str,
    api_key: str,
    body: dict[str, Any] | None,
    *,
    deadline: float,
    ctx: BackendContext,
) -> httpx.Response:
    """Issue one continuation hop, retrying only safe pre-response failures."""
    last_reason = ""
    last_response: httpx.Response | None = None
    for attempt in range(RETRY_5XX_MAX_RETRIES + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("overall upstream request deadline exceeded")
        timeout = httpx.Timeout(
            remaining,
            connect=min(30.0, remaining),
            pool=min(30.0, remaining),
            write=min(30.0, remaining),
            read=remaining,
        )
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "timeout": timeout,
            "follow_redirects": False,
        }
        if body is not None:
            request_kwargs["json"] = body
        try:
            r = await client.request(
                method,
                upstream_url,
                **request_kwargs,
            )
        except httpx.ReadTimeout as exc:
            raise TimeoutError("overall upstream request deadline exceeded") from exc
        except (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteTimeout) as exc:
            last_reason = f"{type(exc).__name__}: {exc}"
            if attempt < RETRY_5XX_MAX_RETRIES:
                await asyncio.sleep(_exp_backoff(attempt))
                continue
            raise UpstreamUnreachable(reason=last_reason, attempts=attempt + 1, ctx=ctx) from exc
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError) as exc:
            # Genuine network-layer failure (DNS, connection refused, TLS,
            # peer reset). NOT a cold boot even on a cold-hinted model — Modal's
            # router stays reachable while a container boots, so a hard connect
            # error means something is actually wrong. Retry then 502.
            last_reason = f"{type(exc).__name__}: {exc}"
            if attempt < RETRY_5XX_MAX_RETRIES:
                await asyncio.sleep(_exp_backoff(attempt))
                continue
            raise UpstreamUnreachable(reason=last_reason, attempts=attempt + 1, ctx=ctx) from exc

        last_response = r
        # 3xx (cold-boot) and 4xx never retry here; outer loop / caller handles.
        if r.status_code < 500:
            return r
        # 5xx. Classify the body first: some upstream 5xx are actually
        # deterministic CLIENT mistakes reflected as a 500 (e.g. vLLM-Lens raises
        # ValueError -> 500 for an out-of-range steering layer_index, ACS-322).
        # Retrying can't fix those, so raise straight away with the client-ish
        # kind rather than burning the retry budget; the /v1 handler turns it
        # into a 400.
        body_text = r.text
        upstream_kind = classify_upstream_error_body(body_text)
        client_ish = upstream_kind in CLIENT_ISH_UPSTREAM_KINDS
        if not client_ish and attempt < RETRY_5XX_MAX_RETRIES:
            # Transient/unknown 5xx: idempotent retry. /v1/completions is safe —
            # seed→deterministic; no side effects without seed.
            await asyncio.sleep(_exp_backoff(attempt))
            continue
        # Client-ish (never retried) or retries exhausted: raise structured
        # error with the real attempt count.
        raise UpstreamServerError(
            upstream_status=r.status_code,
            attempts=attempt + 1,
            body_excerpt=_excerpt_body(body_text, limit=512),
            upstream_kind=upstream_kind,
            ctx=ctx,
        )
    assert last_response is not None  # loop runs at least once
    return last_response


async def stream_post(
    client: httpx.AsyncClient,
    upstream_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
    *,
    ctx: BackendContext | None = None,
) -> AsyncIterator[tuple[bytes, dict[str, int] | None, int | None]]:
    """Stream one Modal invocation, following 303 result URLs as GET."""
    ctx = ctx or BackendContext()
    _inject_include_usage(body)
    deadline = time.monotonic() + min(timeout_s, PUBLIC_REQUEST_TIMEOUT_S)
    method = "POST"
    url = upstream_url
    request_body: dict[str, Any] | None = body
    redirects = 0
    # One retry budget for a continuation hop that 200s with an EMPTY body
    # (the observed shape behind the ACS-277 silent-EOF incident). Safe only
    # on a result-URL GET: nothing has been delivered downstream and a re-GET
    # does not start a second input (unlike a re-POST).
    empty_body_retried = False

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ColdBootError(303 if redirects else 0, max(1, redirects))
        timeout = httpx.Timeout(
            remaining,
            connect=min(30.0, remaining),
            pool=min(30.0, remaining),
            write=min(30.0, remaining),
            read=remaining,
        )
        stream_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "timeout": timeout,
            "follow_redirects": False,
        }
        if request_body is not None:
            stream_kwargs["json"] = request_body
        stream_cm = client.stream(method, url, **stream_kwargs)
        try:
            resp = await stream_cm.__aenter__()
        except httpx.ReadTimeout as exc:
            if ctx.cold_hint or redirects:
                raise ColdBootError(303 if redirects else 0, max(1, redirects)) from exc
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}", attempts=1, ctx=ctx
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
                httpx.ReadError, httpx.WriteError, httpx.WriteTimeout,
                httpx.RemoteProtocolError) as exc:
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}", attempts=1, ctx=ctx
            ) from exc
        try:
            status_code = resp.status_code
            if status_code == 303:
                await resp.aread()
                redirects += 1
                if redirects > MODAL_CONTINUATION_MAX_REDIRECTS:
                    raise ColdBootError(status_code, redirects)
                url = _modal_continuation_url(resp, url, ctx)
                method = "GET"
                request_body = None
                continue
            if 300 <= status_code < 400:
                raise UpstreamUnreachable(
                    reason=f"unexpected upstream redirect status {status_code}",
                    attempts=1,
                    ctx=ctx,
                )
            if status_code >= 500:
                text = await resp.aread()
                raise UpstreamServerError(
                    upstream_status=status_code,
                    attempts=1,
                    body_excerpt=_excerpt_body(text, limit=512),
                    upstream_kind=classify_upstream_error_body(text),
                    ctx=ctx,
                )
            if status_code >= 400:
                text = await resp.aread()
                yield text, None, status_code
                return

            sent_status = False
            try:
                async for raw in resp.aiter_raw():
                    usage = _extract_usage(raw)
                    if not sent_status:
                        yield raw, usage, status_code
                        sent_status = True
                    else:
                        yield raw, usage, None
            except (httpx.ReadTimeout, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError) as exc:
                # Body-phase failure (ACS-277). The except clauses above only
                # guard connection SETUP; without this, a mid-body network
                # error escaped the generator unhandled and the client saw a
                # silent EOF instead of a structured error frame.
                if not sent_status and (ctx.cold_hint or redirects):
                    raise ColdBootError(
                        303 if redirects else 0, max(1, redirects)
                    ) from exc
                raise UpstreamUnreachable(
                    reason=f"{type(exc).__name__} while reading upstream body: {exc}",
                    attempts=1,
                    ctx=ctx,
                ) from exc
            if not sent_status:
                # 200 with an empty body: a Modal continuation hop can end
                # empty-handed. Zero bytes went downstream, so re-GETting the
                # same result URL once is safe; a repeat (or an empty body on
                # the ORIGINAL POST, where retrying would start a second
                # input) becomes a typed error — never a silent clean end
                # (ACS-277; mirrors the non-streaming path's
                # "upstream body ended before any bytes").
                if redirects and not empty_body_retried:
                    empty_body_retried = True
                    continue
                raise UpstreamUnreachable(
                    reason="upstream body ended before any bytes",
                    attempts=2 if empty_body_retried else 1,
                    ctx=ctx,
                )
            return
        finally:
            await stream_cm.__aexit__(None, None, None)


def server_restarting_payload() -> dict[str, Any]:
    """Terminal frame for a stream severed by wrapper shutdown (ACS-277).

    Emitted best-effort when a deploy drain cancels an in-flight streaming
    response, so clients get a retryable error instead of a silent EOF. Not an
    ``api_requests.error_kind`` — the cancelled stream keeps its existing
    accounting; this payload is purely the client-facing goodbye.
    """
    return {
        "error": {
            "code": "server_restarting",
            "message": (
                "The API server restarted mid-request (deployment). Please "
                "retry; if a model cold boot was in progress it continues "
                "unaffected and the retry will attach to it."
            ),
            "type": "server_error",
            "retryable": True,
        }
    }


def _extract_usage(chunk: bytes) -> dict[str, int] | None:
    """Best-effort pluck of the `usage` block from an SSE chunk.

    vLLM emits one final `data: {... "usage": {...} ...}` chunk before
    `data: [DONE]` when `stream_options.include_usage=true`. We parse JSON
    out of each `data: ` line; if any has a usage block, that's our hit.
    """
    if b"usage" not in chunk:
        return None
    for line in chunk.splitlines():
        if not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: ") :].strip()
        if payload == b"[DONE]" or not payload:
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }
    return None


# How many leading body bytes ``passthrough_post`` buffers before deciding a
# 200 response really is a completion. vLLM's completion JSON opens with
# ``{"id":"cmpl-…","object":"text_completion",…,"choices":[…`` so the marker
# appears within the first few hundred bytes; 4 KiB leaves generous slack.
PASSTHROUGH_HEAD_CHECK_BYTES = 4096


def _head_looks_like_completion(head: bytes) -> bool:
    """Byte-level sibling of ``_looks_like_completion`` for unparsed bodies."""
    return b'"choices"' in head or b'"usage"' in head


async def passthrough_post(
    client: httpx.AsyncClient,
    upstream_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
    *,
    ctx: BackendContext | None = None,
) -> AsyncIterator[tuple[bytes, int | None]]:
    """Stream a non-streaming completion's body bytes through unparsed.

    Sibling of ``post_nonstream`` for bodies too large to materialize
    (full-vocab ``prompt_logprobs=-1``, ACS-198: ~22 MB of parsed dict per
    prompt position at a 128k vocab). Same Modal continuation protocol
    (follow 303 result URLs as GET, never re-POST), same pre-body error
    taxonomy (``ColdBootError`` / ``UpstreamUnreachable`` /
    ``UpstreamServerError``), same 5xx retry budget — but the body is never
    ``r.json()``-ed: decoded bytes are yielded as they arrive, so wrapper RAM
    stays O(chunk) instead of O(body).

    Yields ``(chunk, status_code)`` with the status set only on the first
    tuple. A 4xx body is small: it is read whole and yielded as one chunk so
    the caller can parse and re-shape it. A 200 whose head lacks a
    ``"choices"``/``"usage"`` marker raises ``UpstreamUnreachable`` (the
    unparsed mirror of ``_looks_like_completion``). Errors can only be
    raised before the first yield — once bytes flow, a failure surfaces to
    the caller as a truncated stream (mid-body retry would corrupt output).
    """
    ctx = ctx or BackendContext()
    deadline = time.monotonic() + min(timeout_s, PUBLIC_REQUEST_TIMEOUT_S)
    method = "POST"
    url = upstream_url
    request_body: dict[str, Any] | None = body
    redirects = 0
    attempts_5xx = 0
    attempts_net = 0
    last_status = 0

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if ctx.cold_hint or redirects:
                raise ColdBootError(last_status, max(1, redirects))
            raise UpstreamUnreachable(
                reason="overall upstream request deadline exceeded",
                attempts=1,
                ctx=ctx,
            )
        timeout = httpx.Timeout(
            remaining,
            connect=min(30.0, remaining),
            pool=min(30.0, remaining),
            write=min(30.0, remaining),
            read=remaining,
        )
        stream_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "timeout": timeout,
            "follow_redirects": False,
        }
        if request_body is not None:
            stream_kwargs["json"] = request_body
        stream_cm = client.stream(method, url, **stream_kwargs)
        try:
            resp = await stream_cm.__aenter__()
        except httpx.ReadTimeout as exc:
            if ctx.cold_hint or redirects:
                raise ColdBootError(last_status, max(1, redirects)) from exc
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}", attempts=1, ctx=ctx
            ) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
                httpx.ReadError, httpx.WriteError, httpx.WriteTimeout,
                httpx.RemoteProtocolError) as exc:
            # Pre-body network failure — retry with backoff like the buffered
            # path's _request_with_5xx_retry (no bytes have flowed, so a
            # re-issue is safe).
            if attempts_net < RETRY_5XX_MAX_RETRIES:
                await asyncio.sleep(_exp_backoff(attempts_net))
                attempts_net += 1
                continue
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}",
                attempts=attempts_net + 1,
                ctx=ctx,
            ) from exc
        sent_first = False
        try:
            status_code = resp.status_code
            last_status = status_code
            if status_code == 303:
                await resp.aread()
                redirects += 1
                if redirects > MODAL_CONTINUATION_MAX_REDIRECTS:
                    raise ColdBootError(status_code, redirects)
                url = _modal_continuation_url(resp, url, ctx)
                method = "GET"
                request_body = None
                continue
            if 300 <= status_code < 400:
                raise UpstreamUnreachable(
                    reason=f"unexpected upstream redirect status {status_code}",
                    attempts=1,
                    ctx=ctx,
                )
            if status_code >= 500:
                text = await resp.aread()
                # Same idempotent retry rationale as _request_with_5xx_retry:
                # /v1/completions is safe to re-issue (seed → deterministic,
                # no side effects without seed) and no body bytes have been
                # yielded yet.
                if attempts_5xx < RETRY_5XX_MAX_RETRIES:
                    await asyncio.sleep(_exp_backoff(attempts_5xx))
                    attempts_5xx += 1
                    continue
                raise UpstreamServerError(
                    upstream_status=status_code,
                    attempts=attempts_5xx + 1,
                    body_excerpt=_excerpt_body(text, limit=512),
                    upstream_kind=classify_upstream_error_body(text),
                    ctx=ctx,
                )
            if status_code >= 400:
                text = await resp.aread()
                yield text, status_code
                return

            # 200 — buffer a small head to sanity-check completion shape
            # before committing any bytes downstream. ``aiter_bytes`` (not
            # ``aiter_raw``) so a Content-Encoding the upstream applied is
            # transparently decoded; the wrapper applies its own gzip for
            # the client.
            head = bytearray()
            async for chunk in resp.aiter_bytes():
                if sent_first:
                    yield chunk, None
                    continue
                head.extend(chunk)
                head_ok = _head_looks_like_completion(head)
                if len(head) < PASSTHROUGH_HEAD_CHECK_BYTES and not head_ok:
                    continue
                if not head_ok:
                    raise UpstreamUnreachable(
                        reason="upstream returned 200 with a non-completion body",
                        attempts=1,
                        ctx=ctx,
                    )
                yield bytes(head), status_code
                sent_first = True
            if not sent_first:
                # Body ended before the head buffer filled (tiny body).
                if not _head_looks_like_completion(head):
                    raise UpstreamUnreachable(
                        reason="upstream returned 200 with a non-completion body",
                        attempts=1,
                        ctx=ctx,
                    )
                yield bytes(head), status_code
            return
        except httpx.ReadTimeout as exc:
            # Body read timed out before any bytes were forwarded — map to the
            # structured taxonomy instead of leaking an unhandled 500 (the
            # buffered path's client.request() reads the body inside the same
            # exception envelope). Once bytes have flowed, re-raise: the only
            # honest outcome mid-body is a truncated stream.
            if sent_first:
                raise
            if ctx.cold_hint or redirects:
                raise ColdBootError(last_status, max(1, redirects)) from exc
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}", attempts=1, ctx=ctx
            ) from exc
        except (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            if sent_first:
                raise
            raise UpstreamUnreachable(
                reason=f"{type(exc).__name__}: {exc}", attempts=1, ctx=ctx
            ) from exc
        finally:
            await stream_cm.__aexit__(None, None, None)


# Bytes of trailing body the pass-through route keeps for post-hoc ``usage``
# extraction. vLLM serializes ``usage`` after ``choices`` (last-but-one field
# of CompletionResponse), so the block lives in the final few hundred bytes;
# 16 KiB absorbs a trailing null field or two with a wide margin.
PASSTHROUGH_USAGE_TAIL_BYTES = 16384


def extract_usage_from_json_tail(tail: bytes) -> dict[str, int] | None:
    """Pluck ``usage`` token counts from the tail bytes of a completion body.

    The pass-through path (ACS-198) never parses the tens-of-MB full-vocab
    body, but budget accounting still needs ``prompt_tokens`` /
    ``completion_tokens``. vLLM emits ``usage`` as the last substantive key,
    so the route keeps the final ``PASSTHROUGH_USAGE_TAIL_BYTES`` and this
    decodes the *last* ``"usage"`` object in them (the completion text in
    ``choices`` precedes the real block, so last occurrence wins). Returns
    None when no parseable block is found — callers log and skip the commit
    rather than fail the response.
    """
    idx = tail.rfind(b'"usage"')
    if idx == -1:
        return None
    text = tail[idx:].decode("utf-8", errors="replace")
    brace = text.find("{")
    if brace == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[brace:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return {
        "prompt_tokens": int(obj.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(obj.get("completion_tokens", 0) or 0),
    }


async def stream_post_with_status(
    client: httpx.AsyncClient,
    upstream_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
    *,
    ctx: BackendContext | None = None,
    cancel_event: asyncio.Event | None = None,
    boot_stage_provider: Callable[[], Awaitable[dict[str, Any] | None]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Sibling of ``stream_post`` that yields discriminated status/chunk events.

    Used by the workbench streaming path, which always returns SSE 200 from
    the moment the client connects — so a cold-boot wait must be reported
    *inside* the stream rather than as a 503 JSONResponse. The generator
    yields dicts shaped like:

      ``{"kind": "status", "phase": "cold_boot", "attempt": int,
         "elapsed_ms": int, "next_retry_in_s": float | None}``

      ``{"kind": "chunk", "data": bytes, "usage": dict | None,
         "status": int | None}``

      ``{"kind": "error", "data": bytes, "status": int,
         "upstream_kind": str | None}``  (non-cold-boot 4xx/5xx)

    Retries are effectively unlimited (``COLD_BOOT_MAX_RETRIES_WORKBENCH``):
    Modal H200 allocation can legitimately take 30 min – 1 h on a constrained
    day. The progress banner stays up the whole time; the user aborts by
    closing the tab. No ``failed`` event is ever emitted on cold-boot — only
    real upstream 4xx/5xx surface as ``error`` events.

    Network errors (httpx ConnectError, TimeoutException, etc.) are caught and
    surfaced as an ``error`` event with status=0, so the workbench banner can
    distinguish "Modal cold-boot, keep waiting" from "actual outage, give up".
    Unlike the JSON path, we do NOT retry 5xx in the workbench — the user is
    watching and would rather see the error than a silent multi-second pause.
    """
    ctx = ctx or BackendContext()
    _inject_include_usage(body)
    t0 = time.monotonic()

    # Optional per-tick boot-stage lookup (ACS-272). The provider (typically
    # boot_stage.for_cold_wait bound to the model's Modal app) returns extra
    # fields to merge into each cold_boot status event — e.g. ``stage`` /
    # ``stage_label`` — so the banner can show what the container is actually
    # doing instead of an elapsed-time guess. Best-effort by contract: any
    # provider failure degrades to the plain event.
    async def _stage_fields() -> dict[str, Any]:
        if boot_stage_provider is None:
            return {}
        try:
            return (await boot_stage_provider()) or {}
        except Exception:  # noqa: BLE001 - progress info must never break the stream
            return {}

    for attempt in range(COLD_BOOT_MAX_RETRIES_WORKBENCH + 1):
        # Open the stream with try/except so a network error is surfaced as
        # an error event rather than bubbling as an unhandled 500. No retry
        # here — for workbench UX, fail-visible beats silent backoff.
        stream_cm = client.stream(
            "POST",
            upstream_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout_s,
            follow_redirects=False,
        )
        try:
            resp = await stream_cm.__aenter__()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError) as exc:
            yield {
                "kind": "error",
                "data": f"{type(exc).__name__}: {exc}".encode(),
                "status": 0,  # 0 = network layer, no HTTP response
                "upstream_kind": "upstream_unreachable",
            }
            return
        try:
            status_code = resp.status_code

            if _is_cold_boot_status(status_code):
                await resp.aread()
                # Emit one cold_boot status immediately per retry so the
                # banner updates as soon as the wrapper learns Modal is
                # cold (even before the backoff begins). Then tick more
                # events through the backoff at COLD_BOOT_STATUS_INTERVAL_S
                # so the timer + countdown stay alive while we wait.
                yield {
                    "kind": "status",
                    "phase": "cold_boot",
                    "attempt": attempt + 1,
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    "next_retry_in_s": COLD_BOOT_BACKOFF_S,
                    **(await _stage_fields()),
                }
                remaining = COLD_BOOT_BACKOFF_S
                while remaining > 0:
                    slice_s = min(COLD_BOOT_STATUS_INTERVAL_S, remaining)
                    # Race the slice against cancel_event so a workbench stop
                    # click unwinds within milliseconds instead of waiting out
                    # the slice. Without this, the bool flag the caller checks
                    # between yields wasn't observable until the next slice
                    # boundary (≤5 s) — the symptom in ACS-132.
                    if cancel_event is not None:
                        try:
                            await asyncio.wait_for(cancel_event.wait(), timeout=slice_s)
                        except TimeoutError:
                            pass
                        else:
                            return
                    else:
                        await asyncio.sleep(slice_s)
                    remaining -= slice_s
                    if remaining > 0:
                        yield {
                            "kind": "status",
                            "phase": "cold_boot",
                            "attempt": attempt + 1,
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                            "next_retry_in_s": remaining,
                            **(await _stage_fields()),
                        }
                continue

            if status_code >= 400:
                text = await resp.aread()
                yield {
                    "kind": "error",
                    "data": text,
                    "status": status_code,
                    "upstream_kind": classify_upstream_error_body(text),
                }
                return

            sent_status = False
            async for raw in resp.aiter_raw():
                usage = _extract_usage(raw)
                if not sent_status:
                    yield {
                        "kind": "chunk",
                        "data": raw,
                        "usage": usage,
                        "status": status_code,
                    }
                    sent_status = True
                else:
                    yield {
                        "kind": "chunk",
                        "data": raw,
                        "usage": usage,
                        "status": None,
                    }
            return
        finally:
            # Always release the upstream connection. The httpx context
            # manager would do this for us in the original ``async with``
            # form; we unrolled it to wrap the connect in try/except, so
            # we own the cleanup.
            await stream_cm.__aexit__(None, None, None)


def cold_boot_error_payload(upstream_status: int, *, retry_after_s: float | None = None) -> dict[str, Any]:
    """The JSON body returned to the wrapper's caller when upstream is cold.

    Code is the stable contract the workbench JS keys off of
    (``error.code === 'modal_cold_boot'`` triggers the progress banner).

    ``retry_after_s`` lets the caller propose a backoff to the client; when
    set it's surfaced both as ``error.retry_after_seconds`` and as an HTTP
    ``Retry-After`` header (the latter is the RFC 7231 standard hint). When
    None we default to ``COLD_BOOT_BACKOFF_S`` since that's the natural
    cadence of the next probe.
    """
    payload: dict[str, Any] = {
        "error": {
            "code": "modal_cold_boot",
            "message": (
                "Model is warming up after scale-to-zero. Please retry shortly; "
                "large models can take several minutes."
            ),
            "type": "upstream_not_ready",
            "upstream_status": upstream_status,
            "retryable": True,
            "retry_after_seconds": int(retry_after_s if retry_after_s is not None
                                       else COLD_BOOT_BACKOFF_S),
        }
    }
    return payload


def upstream_unreachable_payload(exc: UpstreamUnreachable) -> dict[str, Any]:
    """Structured 502 body for ``UpstreamUnreachable``.

    Tagged with the backend identity (model_id, gpu_shape) so triage during
    an incident can answer "which upstream failed?" from the response alone,
    no log dive required.
    """
    return {
        "error": {
            "code": "upstream_unreachable",
            "message": (
                "Could not reach the upstream model server "
                f"({exc.ctx.model_id or 'unknown'} on {exc.ctx.gpu_shape or 'unknown'}) "
                f"after {exc.attempts} attempts. Network or DNS issue."
            ),
            "type": "upstream_error",
            "reason": exc.reason,
            "attempts": exc.attempts,
            "model_id": exc.ctx.model_id,
            "gpu_shape": exc.ctx.gpu_shape,
            "retryable": True,
        }
    }


def upstream_server_error_payload(exc: UpstreamServerError) -> dict[str, Any]:
    """Structured 502 body for ``UpstreamServerError``.

    Includes the upstream status (so clients can distinguish 502 vs 503 vs
    504), a body excerpt (so logs can capture the upstream's message without
    forwarding 100 MB pages), and the vLLM-specific kind when we recognised
    it (vllm_oom, vllm_context_length, vllm_engine_dead, …).
    """
    return {
        "error": {
            "code": exc.upstream_kind or "upstream_server_error",
            "message": (
                f"Upstream model server ({exc.ctx.model_id or 'unknown'}) "
                f"returned {exc.upstream_status} after {exc.attempts} attempts."
            ),
            "type": "upstream_error",
            "upstream_status": exc.upstream_status,
            "upstream_kind": exc.upstream_kind,
            "attempts": exc.attempts,
            "body_excerpt": exc.body_excerpt,
            "model_id": exc.ctx.model_id,
            "gpu_shape": exc.ctx.gpu_shape,
            "retryable": exc.upstream_status in (502, 503, 504),
        }
    }


async def get_models(
    client: httpx.AsyncClient, upstream_url: str, api_key: str, timeout_s: float
) -> tuple[int, dict[str, Any]]:
    r = await client.get(
        upstream_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_s,
    )
    try:
        return r.status_code, r.json()
    except json.JSONDecodeError:
        raise UpstreamError(
            status.HTTP_502_BAD_GATEWAY,
            detail={"error": {"message": "Upstream returned non-JSON.", "code": "upstream_bad"}},
        )
