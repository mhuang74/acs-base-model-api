"""Completion request validation and response helper services."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from .. import auth as authmod
from .. import proxy as proxymod
from ..schemas import (
    DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS,
    FULL_VOCAB_MAX_PROMPT_TOKENS,
    FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED,
    FULL_VOCAB_SENTINEL,
    CompletionsRequest,
)
from ..settings import Settings
from ..tokenizer import get_token_counter

ErrorResponseFactory = Callable[
    [
        AsyncSession,
        str,
        authmod.AuthedCaller | None,
        str | None,
        str,
        float,
        int,
        str,
        str,
    ],
    Awaitable[JSONResponse],
]


def validation_message(exc: Exception) -> str:
    """Compose a compact user-facing message from a Pydantic ValidationError."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    try:
        items = errors()
    except Exception:
        return str(exc)
    parts: list[str] = []
    for err in items[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        kind = err.get("type", "invalid")
        msg = err.get("msg", "invalid value")
        if kind == "extra_forbidden":
            if loc == "best_of":
                parts.append(
                    "best_of is not supported by this API; "
                    "use n to request multiple completions."
                )
            else:
                parts.append(f"Unknown field {loc!r} - see /v1/models for accepted params.")
        else:
            parts.append(f"{loc}: {msg}")
    if len(items) > 5:
        parts.append(f"... and {len(items) - 5} more.")
    return " ".join(parts) or "Invalid request body."


def count_prompt_tokens(prompt: Any, counter: Any) -> int:
    """Total prompt tokens across a (possibly batched) ``/v1/completions`` prompt.

    Handles all four shapes: text, list of texts, a single pre-tokenized prompt
    (``list[int]``), or a batch of pre-tokenized prompts (``list[list[int]]``).
    Token-id prompts are counted by length (no re-tokenization); text is counted
    via the model tokenizer. Returns the *total* across a batch — what the input
    budget is charged, and a safe over-estimate for the context check.
    """
    if isinstance(prompt, str):
        return counter.count(prompt)
    if not isinstance(prompt, list) or not prompt:
        return 0
    if type(prompt[0]) is int:
        return len(prompt)  # one pre-tokenized prompt (token ids)
    total = 0
    for p in prompt:
        total += len(p) if isinstance(p, list) else counter.count(str(p))
    return total


async def check_sequence_length(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
    *,
    error_response: ErrorResponseFactory,
    token_counter_factory: Callable[[str, str | None], Any] = get_token_counter,
) -> JSONResponse | None:
    """Return a 400 response if prompt + max_tokens exceeds max_model_len."""
    if entry.max_model_len is None:
        return None
    prompt = parsed.prompt
    if isinstance(prompt, str) and not prompt:
        return None
    counter = token_counter_factory(entry.tokenizer_repo, settings.hf_token)
    prompt_tokens = count_prompt_tokens(prompt, counter)
    requested_max = parsed.max_tokens
    needed = prompt_tokens + (requested_max or 1)
    if needed <= entry.max_model_len:
        return None
    msg = (
        f"Request would exceed model context window: prompt={prompt_tokens} tokens"
        + (f" + max_tokens={requested_max}" if requested_max is not None else "")
        + f" > max_model_len={entry.max_model_len}."
        " Reduce the prompt or max_tokens."
    )
    return await error_response(
        session,
        request_id,
        caller,
        ip,
        "/v1/completions",
        t0,
        400,
        "context_length_exceeded",
        msg,
    )


async def check_full_vocab_prompt_logprobs(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
    *,
    error_response: ErrorResponseFactory,
    token_counter_factory: Callable[[str, str | None], Any] = get_token_counter,
) -> JSONResponse | None:
    """Return a 400 when a full-vocab ``prompt_logprobs=-1`` prompt is too long.

    Full-vocab prompt logprobs (ACS-191) return ``vocab_size`` values at every
    prompt position. Since the streaming pass-through (ACS-198) the wrapper no
    longer buffers the body, so the binding constraint moved UPSTREAM: the
    vLLM process materializes ~22 MB of parsed dict per prompt position at a
    128k vocab to serialize the response, plus the transfer itself must fit
    the public request window (see the sizing note at
    ``FULL_VOCAB_MAX_PROMPT_TOKENS``). We bound the one dimension we can
    measure here — prompt length — and reject early with a clear 400 rather
    than OOM the model server mid-request. No-op unless ``prompt_logprobs``
    is exactly ``-1``.

    Exception: a full-vocab request that also carries activation params is
    served by the activation engine over the BUFFERED path (the pass-through
    only handles plain completions), so it keeps the old wrapper-RAM-sized
    bound ``FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED``.
    """
    if parsed.prompt_logprobs != FULL_VOCAB_SENTINEL:
        return None
    cap = (
        FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED
        if parsed.has_activation_params
        else FULL_VOCAB_MAX_PROMPT_TOKENS
    )
    counter = token_counter_factory(entry.tokenizer_repo, settings.hf_token)
    prompt_tokens = count_prompt_tokens(parsed.prompt, counter)
    if prompt_tokens <= cap:
        return None
    msg = (
        "Full-vocabulary prompt_logprobs (prompt_logprobs=-1) is limited to "
        f"prompts of at most {cap} tokens"
        + (
            " when combined with activation params (that path buffers the "
            "response in memory)"
            if parsed.has_activation_params
            else ""
        )
        + f"; this prompt is {prompt_tokens} tokens. The full distribution is "
        "~vocab_size values per position, so a longer prompt would exhaust "
        "the server's memory. Use a shorter prompt, or request a fixed top-k "
        "(e.g. prompt_logprobs=20)."
    )
    return await error_response(
        session,
        request_id,
        caller,
        ip,
        "/v1/completions",
        t0,
        400,
        "full_vocab_prompt_too_long",
        msg,
    )


async def check_activation_prompt_length(
    session: AsyncSession,
    request_id: str,
    caller: authmod.AuthedCaller,
    ip: str | None,
    t0: float,
    entry: Any,
    parsed: CompletionsRequest,
    settings: Settings,
    *,
    error_response: ErrorResponseFactory,
    token_counter_factory: Callable[[str, str | None], Any] = get_token_counter,
    effective_max_tokens: int | None = None,
) -> JSONResponse | None:
    """Return a 400 when an activation-CAPTURE response would exceed the per-model cap.

    Capture returns the residual stream — a bf16 tensor of shape
    (n_layers, n_tokens, d_model) where ``n_tokens = prompt + generated − 1``
    (capture covers the generation trajectory too, minus the final sampled token
    that never gets a forward pass; ACS-252). Since ACS-250 streams the capture
    response (pass-through, no wrapper-RAM materialization), the binding
    constraint is the RESPONSE SIZE the client downloads, not wrapper memory.
    n_layers and d_model are fixed per model, so the dimensions we bound are the
    TOTAL POSITIONS (prompt + generated − 1), per model
    (``entry.activation_max_prompt_tokens``, falling back to
    ``DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS``). 405B (126 × 16384 ≈ 4 MB/token)
    stays tight; 8B (~0.27 MB/token) is generous. Counting the generated tokens
    matters because a short prompt with a large ``max_tokens`` would otherwise
    slip past a prompt-only cap and buffer a huge response (ACS-252/ACS-255) —
    this accounting is only valid because capture is single-sequence (batched
    prompt / n > 1 rejected in ``CompletionsRequest``), so there is no
    n × positions term. (The full-vocab+capture COMBO still buffers, but is
    bounded by the tighter full-vocab cap.) No-op unless the request captures
    (``output_residual_stream``). ``effective_max_tokens`` is the generated-token
    count the request will actually use once the None → per-model default is
    resolved at the call site; it falls back to ``parsed.max_tokens`` and then to
    1 (a single forward pass). Inline capture is for probing; bulk / long-context
    harvesting uses the dedicated batched pipeline (POST /v1/harvest).
    """
    requested = parsed.output_residual_stream
    if not requested:
        return None
    base_cap = (
        getattr(entry, "activation_max_prompt_tokens", None) or DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS
    )
    # The cap exists to bound RESPONSE SIZE, which scales with the number of
    # captured layers — so a subset request affords proportionally more prompt
    # (ACS-317). Needs the model's layer count; without it the flat cap stands.
    n_layers = getattr(entry, "n_layers", None)
    subset = len(requested) if isinstance(requested, list) else None
    cap = base_cap
    if subset and n_layers and subset < n_layers:
        cap = base_cap * n_layers // subset
        # A prompt can never exceed the context window anyway, so don't advertise
        # a cap the model couldn't accept.
        if entry.max_model_len:
            cap = min(cap, entry.max_model_len)
    counter = token_counter_factory(entry.tokenizer_repo, settings.hf_token)
    prompt_tokens = count_prompt_tokens(parsed.prompt, counter)
    # Capture covers prompt + generated − 1 positions (ACS-252), so the cap must
    # count generated tokens too — a short prompt with a large max_tokens would
    # otherwise slip past a prompt-only cap and buffer a huge response
    # (ACS-255). The generated count is the value the request will actually use:
    # the caller passes the resolved effective_max_tokens (None → per-model
    # default already applied); falling back to parsed.max_tokens, then to 1
    # (a single forward pass). Single-sequence only — batched prompt / n > 1 are
    # rejected upstream — so there is no n × positions term to account for.
    generated = effective_max_tokens
    if generated is None:
        generated = parsed.max_tokens
    if generated is None:
        generated = 1
    total_positions = prompt_tokens + max(0, generated - 1)
    if total_positions <= cap:
        return None
    scaled = bool(subset and n_layers and subset < n_layers)
    scope = f"the {subset} requested layer(s)" if scaled else "every layer's residual stream"
    # Only advertise layer subsetting as a remedy where it actually raises the
    # cap — it needs the model's layer count in the registry.
    if n_layers:
        remedy = "Request fewer layers to raise this cap, lower max_tokens, use"
    else:
        remedy = "Lower max_tokens, use"
    msg = (
        "Activation capture (output_residual_stream) is limited to "
        f"{cap} captured positions for this model; this request covers "
        f"{total_positions} (prompt {prompt_tokens} + generated {generated} − 1). "
        f"Capture returns {scope} at every position, so a long prompt or a large "
        f"max_tokens produces a very large response. {remedy} a shorter prompt, "
        "or for large / long-context captures use POST /v1/harvest."
    )
    return await error_response(
        session,
        request_id,
        caller,
        ip,
        "/v1/completions",
        t0,
        400,
        "activation_prompt_too_long",
        msg,
    )


def apply_extras_headers(response: Response, extras: dict[str, Any]) -> None:
    """Translate non-body completion extras into response headers."""
    clamp = extras.get("max_tokens_clamped")
    if isinstance(clamp, dict):
        requested = clamp.get("requested")
        applied = clamp.get("applied")
        reason = clamp.get("reason", "budget")
        response.headers["X-Acs-Max-Tokens-Clamped"] = (
            f"requested={requested},applied={applied},reason={reason}"
        )
    upstream_kind = extras.get("upstream_error_kind")
    if isinstance(upstream_kind, str) and upstream_kind:
        response.headers["X-Acs-Upstream-Error-Kind"] = upstream_kind


def _ascii_header_value(value: str) -> str:
    """Coerce a header value to ASCII so it's RFC 7230-legal on the wire.

    GPU shape labels use ``×`` (U+00D7) for readability — e.g. ``8×H200``.
    That's fine in UTF-8 JSON bodies, but a bare non-ASCII byte in an HTTP
    header value is illegal and strict clients/proxies reject or corrupt it.
    Map the multiplication sign to ASCII ``x`` (``8×H200`` → ``8xH200``) and
    strip any other non-ASCII as a backstop.
    """
    return value.replace("×", "x").encode("ascii", "ignore").decode("ascii")


def apply_backend_headers(response: Response, ctx: proxymod.BackendContext) -> None:
    """Tag a response with stable wrapper-side upstream identity."""
    if ctx.model_id:
        response.headers["X-Acs-Upstream-Model"] = _ascii_header_value(ctx.model_id)
    if ctx.gpu_shape:
        response.headers["X-Acs-Upstream-Gpu"] = _ascii_header_value(ctx.gpu_shape)


def budget_latency_ms(t0: float) -> int:
    """Shared latency calculation for budget/request-log paths."""
    return int((time.monotonic() - t0) * 1000)
