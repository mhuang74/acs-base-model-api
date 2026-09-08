"""Workbench generation state, SSE helpers, and background streaming task."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from . import auth as authmod
from . import boot_stage as boot_stage_mod
from . import modal_ops as modalops
from . import proxy as proxymod
from . import web_auth as webauth
from .error_kinds import ErrorKind
from .logging import get_logger
from .models import (
    ApiKey,
    ApiRequest,
    ChatGeneration,
    ChatSession,
    ChatSnapshot,
    CompareSnapshot,
)
from .tokenizer import get_token_counter

log = get_logger()


# Tab-independent workbench streaming. See ``GenerationState`` and the
# /workbench/<chat>/generations/* handlers. Kept module-level so tests can
# monkeypatch shorter values via wrapper.main's compatibility exports.
GENERATION_EVICT_S = 300.0
GENERATION_FLUSH_INTERVAL_S = 1.0
GENERATION_FLUSH_CHARS = 500

# SSE comment-line keepalive cadence on the workbench subscriber stream. Cold
# boots can leave the producer silent long enough for middleboxes to reset idle
# TCP sockets; comment-line SSE keepalives keep bytes moving.
HEARTBEAT_INTERVAL_S = 5.0

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


class GenerationState:
    """In-memory fan-out for one active workbench generation.

    The streaming task owns the upstream connection and appends to
    ``completion_text`` as chunks arrive; each subscribed SSE consumer
    receives a copy of every chunk on its own queue. Queues carry raw SSE frame
    bytes already shaped for the browser.
    """

    __slots__ = (
        "completion_text",
        "logprobs",
        "subscribers",
        "status",
        "error",
        "error_kind",
        "done_event",
        "usage",
        "cancel_requested",
        "cancel_event",
        "last_status_frame",
    )

    def __init__(self) -> None:
        self.completion_text: str = ""
        # Accumulated per-token logprobs for ``completion_text`` in the shared
        # normalised heatmap shape (``[{token, logprob, top:[...]}]``), when the
        # run requested logprobs. Empty otherwise. Persisted onto the
        # ``ChatSnapshot`` so a saved snapshot can be re-coloured (ACS-189).
        self.logprobs: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue[bytes]] = set()
        self.status: str = "running"
        self.error: str | None = None
        # Stable kind string for failed generations (``upstream_unreachable``,
        # ``vllm_oom``, ``internal_error``, …). Mirrors the SSE error frame's
        # ``code`` so late subscribers — who arrive after the real-time error
        # frame is gone and only see ``replay`` + ``done`` — still get the
        # structured failure-mode signal in the done payload.
        self.error_kind: str | None = None
        self.done_event: asyncio.Event = asyncio.Event()
        self.usage: dict[str, int] = {}
        self.cancel_requested: bool = False
        # Awaitable mirror of ``cancel_requested``: lets the proxy cold-boot
        # sleep loop unwind on stop click within milliseconds instead of
        # waiting up to ``COLD_BOOT_STATUS_INTERVAL_S`` for the next slice
        # boundary (ACS-132). The bool stays the source of truth for "should
        # this generation be marked cancelled in the finally block".
        #
        # Ordering invariant: callers MUST set ``cancel_requested = True``
        # before ``cancel_event.set()``. The producer wakes on the event and
        # the consumer routes the terminal status off the bool — flipping the
        # order would race the producer past the bool flip and the generation
        # could be marked ``completed`` instead of ``cancelled``.
        self.cancel_event: asyncio.Event = asyncio.Event()
        self.last_status_frame: bytes | None = None

    def subscribe(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self.subscribers.discard(q)

    def broadcast(self, frame: bytes) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except Exception:  # noqa: BLE001 — never let one bad queue kill the task
                pass

    def broadcast_status(self, frame: bytes) -> None:
        """Broadcast a status frame and remember it as the latest-known phase."""
        self.last_status_frame = frame
        self.broadcast(frame)

    def mark_done(
        self, status: str, error: str | None = None, error_kind: str | None = None
    ) -> None:
        self.status = status
        self.error = error
        self.error_kind = error_kind
        self.done_event.set()


def _friendly_upstream_error(status_code: int) -> str:
    if status_code == 404:
        return "This model is currently disabled. Email infra@acsresearch.org to request access."
    if status_code <= 0:
        # Network-layer failure (status==0 from ``stream_post_with_status``);
        # surfacing ``(HTTP 0)`` to the user would be confusing.
        return "Could not reach the upstream model server. Please try again shortly."
    return f"Upstream error ({status_code}). Please try again shortly."


def _derive_title(prompt: str) -> str:
    """First ~40 chars of the prompt, single-line, for auto-titling."""
    line = (prompt or "").strip().splitlines()[0] if prompt and prompt.strip() else ""
    line = line[:40].strip()
    return line or "Untitled"


def _sse_status_frame(payload: dict[str, Any]) -> bytes:
    return b"event: status\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _sse_error_frame(payload: dict[str, Any]) -> bytes:
    return b"event: error\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _sse_replay_frame(payload: dict[str, Any]) -> bytes:
    return b"event: replay\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _sse_done_frame(payload: dict[str, Any]) -> bytes:
    return b"event: done\ndata: " + json.dumps(payload).encode() + b"\n\n"


async def _gen_live(state: GenerationState, since: int | None, gen_id: uuid.UUID | None = None):
    """SSE subscriber generator with comment-line keepalive."""
    q = state.subscribe()
    t_connect = time.monotonic()
    saw_done = False
    try:
        full_text = state.completion_text
        offset = since or 0
        replay_text = full_text[offset:] if (offset and offset <= len(full_text)) else full_text
        yield _sse_replay_frame({"text": replay_text, "cursor": len(full_text)})
        if state.status == "running" and state.last_status_frame is not None:
            yield state.last_status_frame
        last_yield_at = time.monotonic()
        if gen_id is not None:
            log.info(
                "chat_generation_subscriber_connected",
                gen_id=str(gen_id),
                replay_chars=len(replay_text),
                status=state.status,
            )

        while True:
            if state.status != "running" and q.empty():
                yield _sse_done_frame(
                    {
                        "status": state.status,
                        "usage": dict(state.usage),
                        "error_message": state.error,
                        "code": state.error_kind,
                    }
                )
                break
            try:
                frame = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if time.monotonic() - last_yield_at >= HEARTBEAT_INTERVAL_S:
                    yield b": keepalive\n\n"
                    last_yield_at = time.monotonic()
                continue
            yield frame
            last_yield_at = time.monotonic()
            if frame.startswith(b"event: done"):
                saw_done = True
                break
    finally:
        state.unsubscribe(q)
        if gen_id is not None:
            log.info(
                "chat_generation_subscriber_disconnected",
                gen_id=str(gen_id),
                saw_done=saw_done,
                status=state.status,
                connected_s=round(time.monotonic() - t_connect, 1),
            )


def _absorb_chunk_text(chunk: bytes) -> tuple[str, bool, str | None]:
    """Extract appended completion text, the [DONE] sentinel, and the echoed
    ``model`` string from one single-pane SSE chunk.

    vLLM echoes its ``served_model_name`` on every chunk (OpenAI completions
    shape); the *last* non-empty echo wins, so a Run records the model string
    the upstream actually identified itself as. Returns ``(delta, saw_done,
    model_echo)`` with ``model_echo=None`` when the chunk carried none — the
    empty-string case is skipped so a mid-stream gap doesn't clobber a good
    earlier echo.
    """
    saw_done = False
    model_echo: str | None = None
    parts: list[str] = []
    for line in chunk.splitlines():
        if not line.startswith(b"data: "):
            continue
        payload_b = line[len(b"data: ") :].strip()
        if payload_b == b"[DONE]":
            saw_done = True
            continue
        if not payload_b:
            continue
        try:
            obj = json.loads(payload_b)
        except json.JSONDecodeError:
            continue
        m = obj.get("model")
        if isinstance(m, str) and m.strip():
            model_echo = m
        choices = obj.get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("text") or ""
            if delta:
                parts.append(delta)
    return "".join(parts), saw_done, model_echo


# Cap the per-snapshot persisted logprobs list so a runaway generation can't
# bloat the JSONB row. The workbench clamps max_tokens to 4000, so a normal run
# never reaches this; the cap only bounds a pathological upstream. Mirrors the
# spirit of ``_COMPARE_SNAPSHOT_COMPLETION_CAP`` (a size ceiling on persisted
# generation output).
_SNAPSHOT_LOGPROBS_TOKEN_CAP = 8000


def _absorb_chunk_logprobs(chunk: bytes) -> list[dict[str, Any]]:
    """Extract normalised per-token logprob entries from one single-pane SSE chunk.

    The single-pane relay forwards the raw upstream ``/v1/completions`` chunk
    verbatim, so ``choices[0].logprobs`` is present when the run requested
    logprobs. Returns the flattened heatmap shape (``[{token, logprob, top}]``)
    via the shared ``_normalise_logprobs`` — the same shape the loom stores and
    the client's live heatmap builds — or ``[]`` when the chunk carries none.
    """
    out: list[dict[str, Any]] = []
    for line in chunk.splitlines():
        if not line.startswith(b"data: "):
            continue
        payload_b = line[len(b"data: ") :].strip()
        if not payload_b or payload_b == b"[DONE]":
            continue
        try:
            obj = json.loads(payload_b)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if choices and isinstance(choices[0], dict):
            out.extend(_normalise_logprobs(choices[0].get("logprobs")))
    return out


def _parse_upstream_error_message(chunk: bytes, status_code: int) -> str:
    """Best-effort pluck of an OpenAI-shaped error message from an upstream body."""
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    msg = _friendly_upstream_error(status_code)
    try:
        j = json.loads(text)
        if isinstance(j, dict):
            err = j.get("error")
            if isinstance(err, dict) and err.get("message"):
                msg = str(err["message"])
            elif isinstance(err, str):
                msg = err
    except Exception:
        pass
    return msg


def _make_boot_stage_provider(app: FastAPI, model_id: str):
    """Boot-stage lookup for cold_boot status frames (ACS-272).

    Bound to the model's Modal app; passed into stream_post_with_status so
    each cold-boot tick can report what the container is actually doing
    (weights loading, engine init, …) instead of an elapsed-time guess.
    Returns None when the model has no Modal app (e.g. legacy env entries) —
    the proxy then emits plain events, exactly the pre-ACS-272 behaviour.
    """
    entry = app.state.models.get(model_id)
    modal_app_name = getattr(entry, "modal_app_name", "") if entry is not None else ""
    if not modal_app_name:
        return None

    async def provider() -> dict[str, Any] | None:
        st = await boot_stage_mod.for_cold_wait(modal_app_name)
        return st.as_payload()

    return provider


async def _run_generation_task(
    *,
    app: FastAPI,
    gen_id: uuid.UUID,
    chat_id: uuid.UUID,
    state: GenerationState,
    body: dict[str, Any],
    upstream_url: str,
    vllm_api_key: str,
    upstream_timeout_s: float,
    model_id: str,
    tokenizer_repo: str,
    hf_token: str | None,
    caller_key_id: uuid.UUID,
    clamped_max: int,
    clamped_temp: float,
    prompt: str,
    request_id: str,
    ip: str | None,
    t0: float,
    mark_model_warm: Callable[[FastAPI, str], None] | None = None,
    persist_session_state: bool = True,
    compare_run_id: uuid.UUID | None = None,
) -> None:
    """Drive one upstream stream, fan out SSE, and persist final state.

    ``persist_session_state`` (default True) controls whether a successful run
    writes back into the *session*: a ``ChatSnapshot`` row + the rolling
    ``chat_sessions.prompt_text``/title. The single-pane "Continue" workflow
    wants that (it's an iterative continuation workspace). Compare-mode lanes
    pass ``False`` — they're one-shot, exploratory fan-outs that must not
    clobber the session's rolling prompt or flood its snapshot history. The
    durable ``ChatGeneration`` row, usage commit, and ``ApiRequest`` audit log
    are always written regardless, so compare runs stay resumable and billed.

    ``compare_run_id`` (ACS-186) is the server-side batch identity shared by the
    N lanes of one Compare "Run all". When set, this task — after committing its
    own terminal state — runs a batch-completion *barrier*: it counts sibling
    lanes for the same run id, and if none are still ``running`` it assembles and
    writes ONE ``CompareSnapshot`` from the persisted lane rows. The strictly-
    after-commit ordering means the globally-last lane to finish sees zero
    running and writes; a UNIQUE constraint on ``compare_snapshots.compare_run_id``
    drops any duplicate from a near-simultaneous finisher or a racing client
    POST. NULL leaves single-pane / loom behaviour untouched.
    """
    http = app.state.http

    completion_chars_since_flush = 0
    last_flush_at = time.monotonic()
    upstream_status: int = 200
    # Stable error kind for the API-requests row + SSE error envelope.
    # Mirrors the ``code`` field that the non-streaming path emits (see
    # ``upstream_unreachable_payload`` / ``upstream_server_error_payload``
    # in ``proxy.py``) so dashboard triage + the workbench UI both see the
    # real failure mode instead of a generic ``upstream_error`` label.
    upstream_error_kind: str | None = None
    upstream_error_msg: str | None = None
    saw_done = False
    # Last non-empty ``model`` string the upstream echoed back on its completion
    # chunks (issue #12). Recorded on the Run alongside the requested short id
    # so exports carry both identities; stays None when upstream never echoed.
    upstream_model_echo: str | None = None
    # Whether this run asked upstream for per-token logprobs. Gate the (cheap but
    # non-free) per-chunk logprobs parse on it so a plain run never pays for it;
    # when set, we accumulate the normalised entries into ``state.logprobs`` for
    # the snapshot heatmap (ACS-189). Truncated defensively at the cap.
    logprobs_requested = bool(body.get("logprobs"))

    cold_boot_attempts = 0
    first_chunk_logged = False
    log.info(
        "chat_generation_task_started",
        gen_id=str(gen_id),
        chat_id=str(chat_id),
        model_id=model_id,
        max_tokens=clamped_max,
        prompt_chars=len(prompt),
        request_id=request_id,
    )

    state.broadcast_status(
        _sse_status_frame(
            {
                "phase": "sending",
                "attempt": 0,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "next_retry_in_s": None,
            }
        )
    )

    async def _flush_text() -> None:
        async with app.state.sessions() as s:
            await s.execute(
                ChatGeneration.__table__.update()
                .where(ChatGeneration.id == gen_id)
                .values(completion_text=state.completion_text)
            )
            await s.commit()

    try:
        upstream_iter = proxymod.stream_post_with_status(
            http,
            upstream_url,
            vllm_api_key,
            body,
            upstream_timeout_s,
            cancel_event=state.cancel_event,
            boot_stage_provider=_make_boot_stage_provider(app, model_id),
        )
        async for ev in upstream_iter:
            if state.cancel_requested:
                break
            kind = ev.get("kind")
            if kind == "status":
                if ev.get("phase") == "cold_boot":
                    cold_boot_attempts += 1
                    if cold_boot_attempts == 1 or cold_boot_attempts % 12 == 0:
                        log.info(
                            "chat_generation_cold_boot",
                            gen_id=str(gen_id),
                            model_id=model_id,
                            attempt=ev.get("attempt"),
                            elapsed_ms=ev.get("elapsed_ms"),
                        )
                state.broadcast_status(
                    _sse_status_frame(
                        {
                            "phase": ev.get("phase"),
                            "attempt": ev.get("attempt"),
                            "elapsed_ms": ev.get("elapsed_ms"),
                            "next_retry_in_s": ev.get("next_retry_in_s"),
                            "stage": ev.get("stage"),
                            "stage_label": ev.get("stage_label"),
                        }
                    )
                )
            elif kind == "chunk":
                chunk = ev.get("data") or b""
                usage = ev.get("usage")
                status_code = ev.get("status")
                if status_code is not None:
                    upstream_status = status_code
                if not first_chunk_logged:
                    first_chunk_logged = True
                    log.info(
                        "chat_generation_upstream_streaming",
                        gen_id=str(gen_id),
                        model_id=model_id,
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                        cold_boot_attempts=cold_boot_attempts,
                    )
                if usage:
                    state.usage.update(usage)
                delta, this_done, chunk_echo = _absorb_chunk_text(chunk)
                if this_done:
                    saw_done = True
                if chunk_echo:
                    upstream_model_echo = chunk_echo
                if delta:
                    state.completion_text += delta
                    completion_chars_since_flush += len(delta)
                if logprobs_requested and len(state.logprobs) < _SNAPSHOT_LOGPROBS_TOKEN_CAP:
                    chunk_lp = _absorb_chunk_logprobs(chunk)
                    if chunk_lp:
                        # Cap the persisted list; extra entries past the ceiling
                        # are dropped (the live client heatmap still streams them).
                        room = _SNAPSHOT_LOGPROBS_TOKEN_CAP - len(state.logprobs)
                        state.logprobs.extend(chunk_lp[:room])
                state.broadcast(chunk)
                now = time.monotonic()
                if (
                    completion_chars_since_flush >= GENERATION_FLUSH_CHARS
                    or (now - last_flush_at) >= GENERATION_FLUSH_INTERVAL_S
                ):
                    try:
                        await _flush_text()
                    except Exception as exc:  # noqa: BLE001 — log + keep streaming
                        log.warning(
                            "chat_generation_flush_failed",
                            gen_id=str(gen_id),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    completion_chars_since_flush = 0
                    last_flush_at = now
            elif kind == "error":
                chunk = ev.get("data") or b""
                # ``stream_post_with_status`` emits ``status=0`` for network-
                # layer failures (httpx ConnectError etc.) — the original
                # ``or 500`` lost that signal and reported a generic 500.
                # Preserve the upstream signal, but translate 0 → 502 for
                # the audit log + UI so neither shows ``(HTTP 0)``; that
                # matches the non-streaming path, which returns 502 from
                # ``upstream_unreachable_payload``.
                raw_status = ev.get("status")
                status_code = int(raw_status) if isinstance(raw_status, int) else 500
                upstream_kind = ev.get("upstream_kind")
                if status_code <= 0:
                    upstream_status = 502
                else:
                    upstream_status = status_code
                upstream_error_kind = upstream_kind or (
                    "upstream_unreachable"
                    if status_code <= 0
                    else ("upstream_5xx" if status_code >= 500 else "upstream_4xx")
                )
                upstream_error_msg = _parse_upstream_error_message(chunk, status_code)
                state.broadcast(
                    _sse_error_frame(
                        {
                            "status": upstream_status,
                            "code": upstream_error_kind,
                            "message": upstream_error_msg,
                        }
                    )
                )
                break
    except asyncio.CancelledError:
        state.cancel_requested = True
        raise
    except Exception as exc:  # noqa: BLE001 — record + surface as failed
        # Anything that escapes the upstream iterator is an in-wrapper bug
        # (DB hiccup, JSON crash, etc.). We tag it ``internal_error`` so the
        # dashboard can separate "we broke" from "upstream broke", and so
        # the SSE error frame carries a stable code the UI can render.
        upstream_status = 500
        upstream_error_kind = "internal_error"
        upstream_error_msg = f"{type(exc).__name__}: {exc}"
        log.warning(
            "chat_generation_task_error",
            gen_id=str(gen_id),
            error=upstream_error_msg,
        )
        state.broadcast(
            _sse_error_frame(
                {
                    "status": 500,
                    "code": upstream_error_kind,
                    "message": "Internal error while streaming generation.",
                }
            )
        )
    finally:
        if state.cancel_requested:
            terminal_status = "cancelled"
            error_message = None
        elif upstream_error_msg is not None or upstream_status >= 400:
            terminal_status = "failed"
            error_message = upstream_error_msg or f"Upstream returned HTTP {upstream_status}."
        else:
            terminal_status = "completed"
            error_message = None

        log.info(
            "chat_generation_task_finished",
            gen_id=str(gen_id),
            model_id=model_id,
            status=terminal_status,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            cold_boot_attempts=cold_boot_attempts,
            completion_chars=len(state.completion_text),
            saw_done=saw_done,
            error=error_message,
        )

        n_prompt = int(state.usage.get("prompt_tokens", 0) or 0)
        n_completion = int(state.usage.get("completion_tokens", 0) or 0)
        if n_completion == 0 and state.completion_text:
            try:
                counter = get_token_counter(tokenizer_repo, hf_token)
                n_completion = counter.count_completion(state.completion_text)
            except Exception:
                n_completion = len(state.completion_text) // 4

        try:
            async with app.state.sessions() as s:
                await s.execute(
                    ChatGeneration.__table__.update()
                    .where(ChatGeneration.id == gen_id)
                    .values(
                        status=terminal_status,
                        error_message=error_message,
                        completion_text=state.completion_text,
                        n_prompt_tokens=n_prompt or None,
                        n_completion_tokens=n_completion or None,
                        ended_at=dt.datetime.now(tz=dt.UTC),
                    )
                )

                if persist_session_state and (
                    terminal_status == "completed"
                    or (terminal_status == "cancelled" and state.completion_text)
                ):
                    s.add(
                        ChatSnapshot(
                            session_id=chat_id,
                            prompt_before=prompt,
                            completion_text=state.completion_text,
                            n_completion=n_completion,
                            max_tokens=clamped_max,
                            temperature=clamped_temp,
                            cancelled=(not saw_done),
                            model=model_id,
                            # NULL when logprobs were off (nothing accumulated),
                            # so the UI falls back to plain text; the normalised
                            # list drives the snapshot heatmap otherwise (ACS-189).
                            logprobs=(state.logprobs or None),
                            # --- Run record: full Sampling settings (issue #12) ---
                            # Read back off the clamped vLLM body _build_lane_body
                            # shaped, so the recipe recorded is exactly what the
                            # upstream saw (Draft values are intent; the Run is
                            # authoritative). Optional knobs keep NULL when the
                            # run didn't set them — the same only-when-set policy
                            # the body itself follows.
                            top_p=body.get("top_p"),
                            top_k=body.get("top_k"),
                            min_p=body.get("min_p"),
                            presence_penalty=body.get("presence_penalty"),
                            frequency_penalty=body.get("frequency_penalty"),
                            repetition_penalty=body.get("repetition_penalty"),
                            # Seed only when explicitly set (blank = the upstream
                            # drew randomly; vLLM never echoes the seed back).
                            seed=body.get("seed"),
                            # Single stop sequence as typed (the body carries [stop]).
                            stop=(body["stop"][0] if body.get("stop") else None),
                            logprobs_count=(body.get("logprobs") or None),
                            # A freshly-recorded Run is complete by construction;
                            # recorded_incomplete=True is reserved for pre-migration
                            # rows and v1-imported Runs.
                            recorded_incomplete=False,
                            # The model string the upstream echoed back per chunk.
                            model_echo=upstream_model_echo,
                        )
                    )
                    chat = (
                        await s.execute(select(ChatSession).where(ChatSession.id == chat_id))
                    ).scalar_one_or_none()
                    if chat is not None:
                        chat.prompt_text = prompt + state.completion_text
                        chat.last_max_tokens = clamped_max
                        chat.last_temperature = clamped_temp
                        chat.updated_at = dt.datetime.now(tz=dt.UTC)
                        if chat.title == "Untitled":
                            chat.title = _derive_title(prompt)

                if terminal_status == "completed" and (n_prompt + n_completion) > 0:
                    try:
                        await authmod.commit_usage(s, caller_key_id, n_prompt, n_completion)
                    except Exception:
                        log.warning("usage_commit_failed", key_id=str(caller_key_id))

                caller_row = (
                    await s.execute(select(ApiKey).where(ApiKey.id == caller_key_id))
                ).scalar_one_or_none()
                if caller_row is not None:
                    # The workbench attributes usage to this key but never goes
                    # through the API ``authenticate()`` path, which is the only
                    # other place ``last_used_at`` is stamped. Without this, a
                    # key used solely from the workbench shows "last used: never"
                    # despite accruing usage (ACS-13, Annie's alpha feedback).
                    caller_row.last_used_at = dt.datetime.now(tz=dt.UTC)
                    # Use the threaded ``upstream_error_kind`` so dashboard
                    # triage can distinguish ``upstream_unreachable`` /
                    # ``vllm_oom`` / ``internal_error`` instead of seeing a
                    # generic ``upstream_error`` for every workbench failure.
                    if terminal_status == "failed":
                        recorded_error_kind = upstream_error_kind or ErrorKind.upstream_error
                    else:
                        recorded_error_kind = None
                    s.add(
                        ApiRequest(
                            key_id=caller_row.id,
                            ip=webauth._safe_ip(ip),
                            endpoint="/workbench",
                            model=model_id,
                            n_prompt=n_prompt or None,
                            n_completion=n_completion or None,
                            status=upstream_status,
                            latency_ms=int((time.monotonic() - t0) * 1000),
                            # Workbench is always interactive + streaming (SSE).
                            # Tag it so the usage dashboard's workload mix is
                            # accurate instead of "undeclared".
                            workload_type="interactive",
                            stream=True,
                            error_kind=recorded_error_kind,
                        )
                    )

                await s.commit()
        except Exception as exc:  # noqa: BLE001 — never let DB failure swallow the broadcast
            log.warning(
                "chat_generation_final_persist_failed",
                gen_id=str(gen_id),
                error=f"{type(exc).__name__}: {exc}",
            )

        # Compare batch-completion barrier (ACS-186). Runs in its OWN session,
        # strictly AFTER the terminal-status commit above — that ordering is what
        # makes it correct: the globally-last lane to commit its terminal state
        # sees zero siblings still running and writes the snapshot. Kept out of
        # the terminal-persist txn so a losing racer's IntegrityError can't roll
        # back this lane's terminal write.
        #
        # ``asyncio.shield`` so a second CancelledError (e.g. the user hitting
        # "Stop all", which cancels every lane task) delivered while we're inside
        # this await can't abort the write half-done — the last cancelled lane is
        # exactly the one that must still persist the snapshot. ``except
        # Exception`` alone would NOT catch a CancelledError (it's a
        # BaseException), so without the shield a cancel here could silently skip
        # the barrier for the whole batch.
        #
        # Known gap (not fixed here): if a lane's terminal-status commit above
        # raises (caught at the persist block's ``except``), that row stays
        # ``running`` and every sibling's barrier read bails, so the batch gets
        # no snapshot. Same class as a wrapper restart mid-batch — the ticket
        # targets tab-close, not DB-hiccup/restart durability; both noted on PR.
        if compare_run_id is not None:
            try:
                await asyncio.shield(
                    _maybe_write_compare_snapshot(app, chat_id, compare_run_id)
                )
            except Exception as exc:  # noqa: BLE001 — barrier is best-effort
                log.warning(
                    "compare_snapshot_barrier_failed",
                    gen_id=str(gen_id),
                    compare_run_id=str(compare_run_id),
                    error=f"{type(exc).__name__}: {exc}",
                )

        if terminal_status == "completed" and mark_model_warm is not None:
            mark_model_warm(app, model_id)

        state.broadcast(
            _sse_done_frame(
                {
                    "status": terminal_status,
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_completion,
                    },
                    "error_message": error_message,
                    "code": (upstream_error_kind if terminal_status == "failed" else None),
                }
            )
        )
        state.mark_done(
            terminal_status,
            error_message,
            error_kind=(upstream_error_kind if terminal_status == "failed" else None),
        )

        async def _evict() -> None:
            try:
                await asyncio.sleep(GENERATION_EVICT_S)
            finally:
                app.state.generations.pop(gen_id, None)

        asyncio.create_task(_evict())


# Cap the persisted per-lane completion so a runaway generation can't bloat the
# JSONB snapshot row. Mirrors the client-POST sanitiser's 100k cap in routes.
_COMPARE_SNAPSHOT_COMPLETION_CAP = 100_000


def _lane_from_generation(gen: ChatGeneration) -> dict[str, Any]:
    """Assemble ONE compare-snapshot lane dict from a persisted lane row.

    Reads the sampling params from ``compare_config`` (the clamped vLLM body the
    route stored at launch — the base columns only carry model/max_tokens/
    temperature) and the final text + cancelled flag from the row itself. Shape
    matches the client-POST path's ``_sanitize_compare_lane`` so the history/
    restore UI renders server- and client-written snapshots identically.
    """
    cfg = gen.compare_config if isinstance(gen.compare_config, dict) else {}

    def _pick(key: str, default: Any = None) -> Any:
        val = cfg.get(key)
        return val if val is not None else default

    stop = cfg.get("stop")
    if isinstance(stop, list):
        stop = stop[0] if stop else None
    return {
        "model": gen.model,
        "max_tokens": gen.max_tokens,
        "temperature": gen.temperature,
        "top_p": _pick("top_p"),
        "top_k": _pick("top_k"),
        "min_p": _pick("min_p"),
        "presence_penalty": _pick("presence_penalty"),
        "frequency_penalty": _pick("frequency_penalty"),
        "repetition_penalty": _pick("repetition_penalty"),
        "seed": cfg.get("seed"),
        "stop": str(stop)[:200] if isinstance(stop, str) and stop != "" else None,
        "completion_text": (gen.completion_text or "")[:_COMPARE_SNAPSHOT_COMPLETION_CAP],
        # A lane is "cancelled" if the user stopped it; failed lanes keep their
        # (empty) text and are shown as such by the same flag the client used.
        "cancelled": gen.status in ("cancelled", "failed"),
    }


async def _maybe_write_compare_snapshot(
    app: FastAPI, chat_id: uuid.UUID, compare_run_id: uuid.UUID
) -> None:
    """Batch-completion barrier: write ONE CompareSnapshot when a batch finishes.

    Called by every lane's task after that lane commits its terminal status.
    Counts sibling lanes for ``compare_run_id`` still in ``running``; if none, the
    caller is the last finisher, so assemble the snapshot from the lane rows (in
    lane-index order, taken from ``compare_config['index']``) and insert it with
    ``ON CONFLICT (compare_run_id) DO NOTHING``. The unique constraint makes the
    write idempotent, so a near-simultaneous last-two-finishers race (or a later
    client POST for the same run) collapses to exactly one row.
    """
    async with app.state.sessions() as s:
        lanes_rows = list(
            (
                await s.execute(
                    select(ChatGeneration).where(
                        ChatGeneration.compare_run_id == compare_run_id
                    )
                )
            )
            .scalars()
            .all()
        )
        if not lanes_rows:
            return
        # Barrier: if any sibling is still running, we're not the last — bail and
        # let the actual last finisher write.
        if any(row.status == "running" for row in lanes_rows):
            return

        # Order lanes by the launch index stashed in compare_config so the
        # snapshot's lane order matches the UI (started_at can tie in the tight
        # insert loop, and rows have no explicit ordinal column).
        def _order_key(row: ChatGeneration) -> tuple[int, str]:
            cfg = row.compare_config if isinstance(row.compare_config, dict) else {}
            idx = cfg.get("index")
            return (int(idx) if isinstance(idx, int) else 1_000_000, str(row.id))

        lanes_rows.sort(key=_order_key)
        prompt = lanes_rows[0].prompt_before or ""
        lanes = [_lane_from_generation(row) for row in lanes_rows]

        stmt = (
            pg_insert(CompareSnapshot.__table__)
            .values(
                session_id=chat_id,
                prompt=prompt,
                lanes=lanes,
                n_lanes=len(lanes),
                compare_run_id=compare_run_id,
            )
            .on_conflict_do_nothing(index_elements=["compare_run_id"])
        )
        result = await s.execute(stmt)
        await s.commit()

        wrote = (result.rowcount or 0) > 0
        log.info(
            "compare_snapshot_barrier",
            compare_run_id=str(compare_run_id),
            chat_id=str(chat_id),
            n_lanes=len(lanes),
            wrote=wrote,
        )
        if wrote:
            # Prune this session's compare history back to the cap, matching the
            # client-POST path. Import here to avoid a route↔task import cycle.
            from .routes.workbench import _COMPARE_SNAPSHOT_LIMIT

            keep_ids = (
                select(CompareSnapshot.id)
                .where(CompareSnapshot.session_id == chat_id)
                .order_by(CompareSnapshot.ts.desc(), CompareSnapshot.id.desc())
                .limit(_COMPARE_SNAPSHOT_LIMIT)
            )
            await s.execute(
                delete(CompareSnapshot).where(
                    CompareSnapshot.session_id == chat_id,
                    CompareSnapshot.id.not_in(keep_ids),
                )
            )
            await s.commit()


def mark_model_warm_from_app(app: FastAPI, model_id: str | None) -> None:
    """Record a successful generation against app state without a Request."""
    if not model_id:
        return
    app.state.last_completion_at[model_id] = dt.datetime.now(tz=dt.UTC)
    entry = app.state.models.get(model_id)
    if entry is not None and entry.modal_app_name:
        modalops.mark_runner_warm(entry.modal_app_name)


# ---------------------------------------------------------------------------
# Loom (tree/branching exploration) — ACS-148
#
# The loom path is *additive*: it does NOT reuse ``GenerationState`` /
# ``_absorb_chunk_text`` / ``_run_generation_task`` (all of which assume a
# single completion and a single ``completion_text`` string threaded through
# replay/cursor/eviction/late-subscriber/error tests). Instead a loom
# "generate" is ONE upstream request with ``n=N`` whose response has N choices;
# we accumulate per-``index`` text + logprobs and persist N ``LoomNode`` rows on
# completion. One generation = one upstream call = no "one running per session"
# conflict with the single-stream path.
# ---------------------------------------------------------------------------


def _normalise_logprobs(lp: Any) -> list[dict[str, Any]]:
    """Turn vLLM's completions ``logprobs`` object into our heatmap shape.

    vLLM (OpenAI-compatible ``/v1/completions``) returns, per choice::

        "logprobs": {
            "tokens": ["he", "llo"],
            "token_logprobs": [-0.1, -2.3],
            "top_logprobs": [{"he": -0.1, " hi": -1.2}, {...}],
            ...
        }

    We flatten that into a list aligned with the chosen tokens::

        [{"token": "he", "logprob": -0.1,
          "top": [{"token": "he", "logprob": -0.1}, {"token": " hi", ...}]},
         ...]

    so the UI can colour each token by ``exp(logprob)`` and show ranked
    alternatives on hover. Defensive against partial/missing fields (a chunk
    may carry text without a logprobs block when logprobs weren't requested).
    """
    if not isinstance(lp, dict):
        return []
    tokens = lp.get("tokens") or []
    token_logprobs = lp.get("token_logprobs") or []
    top_logprobs = lp.get("top_logprobs") or []
    out: list[dict[str, Any]] = []
    for i, tok in enumerate(tokens):
        logprob = token_logprobs[i] if i < len(token_logprobs) else None
        top_map = top_logprobs[i] if i < len(top_logprobs) else None
        top: list[dict[str, Any]] = []
        if isinstance(top_map, dict):
            # Highest probability first (logprob descending).
            for alt_tok, alt_lp in sorted(
                top_map.items(),
                key=lambda kv: kv[1] if kv[1] is not None else float("-inf"),
                reverse=True,
            ):
                top.append({"token": alt_tok, "logprob": alt_lp})
        out.append({"token": tok, "logprob": logprob, "top": top})
    return out


def _absorb_loom_chunk(
    chunk: bytes,
) -> tuple[dict[int, dict[str, Any]], bool]:
    """Per-``index`` extraction for an n>1 completions SSE chunk.

    Returns ``(by_index, saw_done)`` where ``by_index`` maps a choice index to
    ``{"text": <delta>, "logprobs": [<entries>]}`` accumulated from THIS chunk
    (the caller appends to its running per-index buffers). A chunk may carry
    deltas for several indices at once, or just one — vLLM interleaves them.
    """
    saw_done = False
    by_index: dict[int, dict[str, Any]] = {}
    for line in chunk.splitlines():
        if not line.startswith(b"data: "):
            continue
        payload_b = line[len(b"data: ") :].strip()
        if payload_b == b"[DONE]":
            saw_done = True
            continue
        if not payload_b:
            continue
        try:
            obj = json.loads(payload_b)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            idx = int(choice.get("index", 0) or 0)
            slot = by_index.setdefault(idx, {"text": "", "logprobs": []})
            delta = choice.get("text") or ""
            if delta:
                slot["text"] += delta
            lp_entries = _normalise_logprobs(choice.get("logprobs"))
            if lp_entries:
                slot["logprobs"].extend(lp_entries)
    return by_index, saw_done


def _sse_loom_chunk_frame(payload: dict[str, Any]) -> bytes:
    """SSE frame carrying a per-branch text/logprob delta.

    Distinct event name (``loom_chunk``) from the single-stream ``chunk`` so the
    loom JS can route by ``index`` without colliding with the existing
    workbench EventSource handlers.
    """
    return b"event: loom_chunk\ndata: " + json.dumps(payload).encode() + b"\n\n"


class LoomGenerationState:
    """In-memory fan-out for one loom generate (one upstream call, N branches).

    Mirrors ``GenerationState`` but keyed per choice index: ``branches[i]`` holds
    the accumulating ``{"text", "logprobs"}`` for branch *i*. Kept deliberately
    simple (no replay-cursor protocol) — a loom generate is short-lived and the
    durable record is the ``LoomNode`` rows written on completion; a tab that
    misses the live stream just reloads the tree from the DB.
    """

    __slots__ = (
        "n",
        "loom_id",
        "parent_id",
        "branches",
        "subscribers",
        "status",
        "error",
        "error_kind",
        "done_event",
        "usage",
        "cancel_requested",
        "cancel_event",
        "last_status",
    )

    def __init__(
        self,
        n: int,
        loom_id: uuid.UUID,
        parent_id: uuid.UUID | None = None,
    ) -> None:
        self.n = n
        # Owning loom + branch parent: used to scope SSE/cancel access to the
        # generation's own loom (no cross-loom gen_id access) and to block
        # deleting a node that an in-flight generate is about to write under.
        self.loom_id = loom_id
        self.parent_id = parent_id
        self.branches: list[dict[str, Any]] = [{"text": "", "logprobs": []} for _ in range(n)]
        self.subscribers: set[asyncio.Queue[bytes]] = set()
        self.status: str = "running"
        self.error: str | None = None
        self.error_kind: str | None = None
        self.done_event: asyncio.Event = asyncio.Event()
        self.usage: dict[str, int] = {}
        self.cancel_requested: bool = False
        self.cancel_event: asyncio.Event = asyncio.Event()
        # Last cold-boot status frame, replayed to a tab that connects after the
        # boot already started so it still shows "warming up".
        self.last_status: bytes | None = None

    def subscribe(self) -> asyncio.Queue[bytes]:
        q: asyncio.Queue[bytes] = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        self.subscribers.discard(q)

    def broadcast(self, frame: bytes) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(frame)
            except Exception:  # noqa: BLE001
                pass

    def mark_done(
        self, status: str, error: str | None = None, error_kind: str | None = None
    ) -> None:
        self.status = status
        self.error = error
        self.error_kind = error_kind
        self.done_event.set()


async def _loom_gen_live(state: LoomGenerationState, gen_id: uuid.UUID | None = None):
    """SSE subscriber generator for a loom generate.

    Replays whatever each branch has accumulated so far (so a tab that connects
    mid-stream sees existing text), then tails live frames until ``done``. Uses
    the same comment-line keepalive cadence as the single-stream path.
    """
    q = state.subscribe()
    try:
        # Replay the last cold-boot status so a tab connecting mid-boot still
        # shows "warming up" rather than a silent gap before tokens arrive.
        if state.last_status is not None and state.status == "running":
            yield state.last_status
        # Replay current per-branch buffers as one loom_chunk each.
        for idx, branch in enumerate(state.branches):
            if branch["text"] or branch["logprobs"]:
                yield _sse_loom_chunk_frame(
                    {
                        "index": idx,
                        "text": branch["text"],
                        "logprobs": branch["logprobs"],
                        "replay": True,
                    }
                )
        last_yield_at = time.monotonic()
        while True:
            if state.status != "running" and q.empty():
                yield _sse_done_frame(
                    {
                        "status": state.status,
                        "usage": dict(state.usage),
                        "error_message": state.error,
                        "code": state.error_kind,
                    }
                )
                break
            try:
                frame = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if time.monotonic() - last_yield_at >= HEARTBEAT_INTERVAL_S:
                    yield b": keepalive\n\n"
                    last_yield_at = time.monotonic()
                continue
            yield frame
            last_yield_at = time.monotonic()
            if frame.startswith(b"event: done"):
                break
    finally:
        state.unsubscribe(q)
        if gen_id is not None:
            log.info("loom_generation_subscriber_disconnected", gen_id=str(gen_id))


async def _run_loom_generation_task(
    *,
    app: FastAPI,
    gen_id: uuid.UUID,
    loom_id: uuid.UUID,
    parent_id: uuid.UUID | None,
    state: LoomGenerationState,
    body: dict[str, Any],
    upstream_url: str,
    vllm_api_key: str,
    upstream_timeout_s: float,
    model_id: str,
    caller_key_id: uuid.UUID,
    seed: int | None,
    request_id: str,
    ip: str | None,
    t0: float,
    mark_model_warm: Callable[[FastAPI, str], None] | None = None,
) -> None:
    """Drive one n>1 upstream stream, fan out per-branch SSE, persist N nodes."""
    http = app.state.http
    upstream_status: int = 200
    upstream_error_kind: str | None = None
    upstream_error_msg: str | None = None
    saw_done = False
    cold_boot_attempts = 0

    log.info(
        "loom_generation_task_started",
        gen_id=str(gen_id),
        loom_id=str(loom_id),
        parent_id=str(parent_id) if parent_id else None,
        model_id=model_id,
        n=state.n,
        request_id=request_id,
    )

    try:
        upstream_iter = proxymod.stream_post_with_status(
            http,
            upstream_url,
            vllm_api_key,
            body,
            upstream_timeout_s,
            cancel_event=state.cancel_event,
            boot_stage_provider=_make_boot_stage_provider(app, model_id),
        )
        async for ev in upstream_iter:
            if state.cancel_requested:
                break
            kind = ev.get("kind")
            if kind == "status":
                # Forward cold-boot / warm-up progress so the loom UI shows a
                # "warming up" banner during a Modal cold boot instead of hanging
                # silently (mirrors the single-stream path). Store the latest so a
                # tab that connects after the boot started still sees it.
                if ev.get("phase") == "cold_boot":
                    cold_boot_attempts += 1
                    if cold_boot_attempts == 1 or cold_boot_attempts % 12 == 0:
                        log.info(
                            "loom_generation_cold_boot",
                            gen_id=str(gen_id),
                            model_id=model_id,
                            attempt=ev.get("attempt"),
                            elapsed_ms=ev.get("elapsed_ms"),
                        )
                status_frame = _sse_status_frame(
                    {
                        "phase": ev.get("phase"),
                        "attempt": ev.get("attempt"),
                        "elapsed_ms": ev.get("elapsed_ms"),
                        "next_retry_in_s": ev.get("next_retry_in_s"),
                        "stage": ev.get("stage"),
                        "stage_label": ev.get("stage_label"),
                    }
                )
                state.last_status = status_frame
                state.broadcast(status_frame)
            elif kind == "chunk":
                chunk = ev.get("data") or b""
                usage = ev.get("usage")
                status_code = ev.get("status")
                if status_code is not None:
                    upstream_status = status_code
                if usage:
                    state.usage.update(usage)
                by_index, this_done = _absorb_loom_chunk(chunk)
                if this_done:
                    saw_done = True
                for idx, slot in by_index.items():
                    if idx >= state.n:
                        continue
                    branch = state.branches[idx]
                    branch["text"] += slot["text"]
                    if slot["logprobs"]:
                        branch["logprobs"].extend(slot["logprobs"])
                    # Broadcast the FULL accumulated branch (not the delta) so the
                    # frame is idempotent: the client SETs slot.text = frame.text.
                    # A delta that lands between a late subscriber's replay snapshot
                    # and its first live frame can't then double-render (the old
                    # replay=SET + live=append model raced).
                    state.broadcast(
                        _sse_loom_chunk_frame(
                            {
                                "index": idx,
                                "text": branch["text"],
                                "logprobs": branch["logprobs"],
                            }
                        )
                    )
            elif kind == "error":
                chunk = ev.get("data") or b""
                raw_status = ev.get("status")
                status_code = int(raw_status) if isinstance(raw_status, int) else 500
                upstream_kind = ev.get("upstream_kind")
                upstream_status = 502 if status_code <= 0 else status_code
                upstream_error_kind = upstream_kind or (
                    "upstream_unreachable"
                    if status_code <= 0
                    else ("upstream_5xx" if status_code >= 500 else "upstream_4xx")
                )
                upstream_error_msg = _parse_upstream_error_message(chunk, status_code)
                state.broadcast(
                    _sse_error_frame(
                        {
                            "status": upstream_status,
                            "code": upstream_error_kind,
                            "message": upstream_error_msg,
                        }
                    )
                )
                break
    except asyncio.CancelledError:
        state.cancel_requested = True
        raise
    except Exception as exc:  # noqa: BLE001
        upstream_status = 500
        upstream_error_kind = "internal_error"
        upstream_error_msg = f"{type(exc).__name__}: {exc}"
        log.warning("loom_generation_task_error", gen_id=str(gen_id), error=upstream_error_msg)
        state.broadcast(
            _sse_error_frame(
                {
                    "status": 500,
                    "code": upstream_error_kind,
                    "message": "Internal error while streaming loom generation.",
                }
            )
        )
    finally:
        if state.cancel_requested:
            terminal_status = "cancelled"
            error_message = None
        elif upstream_error_msg is not None or upstream_status >= 400:
            terminal_status = "failed"
            error_message = upstream_error_msg or f"Upstream returned HTTP {upstream_status}."
        else:
            terminal_status = "completed"
            error_message = None

        n_prompt = int(state.usage.get("prompt_tokens", 0) or 0)
        n_completion = int(state.usage.get("completion_tokens", 0) or 0)

        created_node_ids: list[str] = []
        # Persist one LoomNode per branch that produced any text. On cancel we
        # still keep partial branches (matches the single-stream "keep partial
        # on cancel" behaviour) so a stopped explore isn't silently discarded.
        if terminal_status in ("completed", "cancelled"):
            try:
                from .models import LoomNode

                async with app.state.sessions() as s:
                    # Sibling ordinal base (ACS-340): the N branches of this batch
                    # share ``parent_id`` and a ``created_at`` to the microsecond,
                    # so give each a 0-based ``position`` after any existing
                    # children of the parent. Not unique-constrained — a
                    # concurrent batch under the same parent may read the same max
                    # and overlap; that rare case degrades to created_at order
                    # rather than losing a branch to an insert failure.
                    base_position = (
                        await s.execute(
                            select(func.max(LoomNode.position)).where(
                                LoomNode.loom_id == loom_id,
                                LoomNode.parent_id == parent_id,
                            )
                        )
                    ).scalar_one_or_none()
                    next_position = 0 if base_position is None else int(base_position) + 1
                    for branch in state.branches:
                        if not branch["text"]:
                            continue
                        node = LoomNode(
                            loom_id=loom_id,
                            parent_id=parent_id,
                            text=branch["text"],
                            model=model_id,
                            seed=seed,
                            position=next_position,
                            logprobs=(branch["logprobs"] or None),
                        )
                        s.add(node)
                        await s.flush()
                        created_node_ids.append(str(node.id))
                        next_position += 1
                    if created_node_ids and terminal_status == "completed":
                        try:
                            await authmod.commit_usage(s, caller_key_id, n_prompt, n_completion)
                        except Exception:
                            log.warning("loom_usage_commit_failed", key_id=str(caller_key_id))
                    caller_row = (
                        await s.execute(select(ApiKey).where(ApiKey.id == caller_key_id))
                    ).scalar_one_or_none()
                    if caller_row is not None:
                        caller_row.last_used_at = dt.datetime.now(tz=dt.UTC)
                        s.add(
                            ApiRequest(
                                key_id=caller_row.id,
                                ip=webauth._safe_ip(ip),
                                endpoint="/loom",
                                model=model_id,
                                n_prompt=n_prompt or None,
                                n_completion=n_completion or None,
                                status=upstream_status,
                                latency_ms=int((time.monotonic() - t0) * 1000),
                                workload_type="interactive",
                                stream=True,
                                error_kind=None,
                            )
                        )
                    await s.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "loom_generation_persist_failed",
                    gen_id=str(gen_id),
                    error=f"{type(exc).__name__}: {exc}",
                )
        elif terminal_status == "failed":
            # Record the failed attempt for dashboard triage (no nodes written).
            try:
                async with app.state.sessions() as s:
                    caller_row = (
                        await s.execute(select(ApiKey).where(ApiKey.id == caller_key_id))
                    ).scalar_one_or_none()
                    if caller_row is not None:
                        caller_row.last_used_at = dt.datetime.now(tz=dt.UTC)
                        s.add(
                            ApiRequest(
                                key_id=caller_row.id,
                                ip=webauth._safe_ip(ip),
                                endpoint="/loom",
                                model=model_id,
                                status=upstream_status,
                                latency_ms=int((time.monotonic() - t0) * 1000),
                                workload_type="interactive",
                                stream=True,
                                error_kind=upstream_error_kind or "upstream_error",
                            )
                        )
                    await s.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "loom_generation_failrec_failed",
                    gen_id=str(gen_id),
                    error=f"{type(exc).__name__}: {exc}",
                )

        log.info(
            "loom_generation_task_finished",
            gen_id=str(gen_id),
            model_id=model_id,
            status=terminal_status,
            nodes=len(created_node_ids),
            elapsed_ms=int((time.monotonic() - t0) * 1000),
            saw_done=saw_done,
        )

        if terminal_status == "completed" and mark_model_warm is not None:
            mark_model_warm(app, model_id)

        state.broadcast(
            _sse_done_frame(
                {
                    "status": terminal_status,
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_completion,
                    },
                    "error_message": error_message,
                    "code": (upstream_error_kind if terminal_status == "failed" else None),
                    "node_ids": created_node_ids,
                }
            )
        )
        state.mark_done(
            terminal_status,
            error_message,
            error_kind=(upstream_error_kind if terminal_status == "failed" else None),
        )

        async def _evict() -> None:
            try:
                await asyncio.sleep(GENERATION_EVICT_S)
            finally:
                app.state.loom_generations.pop(gen_id, None)

        asyncio.create_task(_evict())
