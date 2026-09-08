#!/usr/bin/env python
"""stub_upstream.py — a fake model backend for the take-home.

The real API forwards completion/steering/harvest requests to a GPU backend.
You have no GPU and no access to ours, so this stub stands in for it: a tiny
HTTP server that speaks the same ``/v1/completions`` shape and returns canned
text (up to ~35 tokens of lorem ipsum) instead of running a model.

It is deliberately dumb — it ignores the prompt semantics. Its only job is to
let the API server boot and answer requests end-to-end so you can exercise the
request-handling code that lives *before* the model is ever called. It does
honour the request fields that shape the *response envelope*, so what comes
back looks like a real backend's reply:

  max_tokens   caps how many tokens come back (finish_reason "length")
  logprobs: k  adds a per-token ``logprobs`` block with k alternatives
               (``tokens`` / ``token_logprobs`` / ``top_logprobs`` /
               ``text_offset``); the numbers are fake but deterministic
  usage        every reply carries prompt/completion/total token counts
               (prompt tokens are counted as whitespace-separated words)

You normally don't run this by hand — ``./setup-dev.sh up`` starts it for you and
points the API server's model registry at it. To run it standalone:

    (cd base_model_wrapper && uv run python ../stub_upstream.py)

It listens on http://localhost:8900.

Optional behaviour knobs (set as env vars before launching) are available for
general debugging without a real GPU:

  STUB_MODE=ok            (default) stream lorem ipsum, then a clean end
  STUB_MODE=drop_midway   stream a few tokens, then end without a clean marker
  STUB_MODE=error_500     return an HTTP 500 with an error body
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

_LOREM = ["Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo"]


def _tokens(body: dict) -> list[str]:
    """The token strings this reply is made of, capped by ``max_tokens``."""
    words = [" " + w for w in _LOREM]
    limit = body.get("max_tokens")
    if isinstance(limit, int) and limit >= 0:
        words = words[:limit]
    return words


def _top_k(body: dict) -> int | None:
    """How many alternatives per token the caller asked for, if any."""
    k = body.get("logprobs")
    if isinstance(k, bool):
        k = 1 if k else None
    if isinstance(k, int) and k >= 0:
        return max(k, 1)
    return None


def _logprobs(tokens: list[str], k: int, start_index: int, start_offset: int) -> dict:
    """A deterministic fake logprobs block in the OpenAI-completions shape."""
    chosen = [round(-0.1 - 0.05 * (start_index + i), 4) for i in range(len(tokens))]
    offsets, cursor = [], start_offset
    for tok in tokens:
        offsets.append(cursor)
        cursor += len(tok)
    return {
        "tokens": tokens,
        "token_logprobs": chosen,
        "top_logprobs": [
            {tok: lp, **{f"alt{j}": round(lp - 0.7 * j, 4) for j in range(1, k)}}
            for tok, lp in zip(tokens, chosen)
        ],
        "text_offset": offsets,
    }


def _usage(body: dict, n_completion: int) -> dict:
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    n_prompt = len(str(prompt).split())
    return {
        "prompt_tokens": n_prompt,
        "completion_tokens": n_completion,
        "total_tokens": n_prompt + n_completion,
    }


def _chunk(text: str, finish: str | None = None, logprobs: dict | None = None,
           usage: dict | None = None) -> bytes:
    choice: dict = {"index": 0, "text": text, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    payload = {
        "id": "cmpl-stub",
        "object": "text_completion",
        "model": "stub",
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


@app.post("/v1/completions")
async def completions(request: Request) -> object:
    mode = os.environ.get("STUB_MODE", "ok")
    body = await request.json()
    stream = bool(body.get("stream"))

    if mode == "error_500":
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "stub upstream failure", "type": "server_error"}},
        )

    toks = _tokens(body)
    k = _top_k(body)
    finish = "length" if len(toks) < len(_LOREM) else "stop"

    if not stream:
        choice: dict = {"index": 0, "text": "".join(toks), "finish_reason": finish}
        if k is not None:
            choice["logprobs"] = _logprobs(toks, k, 0, 0)
        return JSONResponse(
            content={
                "id": "cmpl-stub",
                "object": "text_completion",
                "model": "stub",
                "choices": [choice],
                "usage": _usage(body, len(toks)),
            }
        )

    async def gen():
        offset = 0
        for i, tok in enumerate(toks):
            if mode == "drop_midway" and i == 8:
                # Simulate the connection dying mid-stream: stop yielding
                # without ever sending a finish_reason or the [DONE] sentinel.
                return
            yield _chunk(tok, logprobs=_logprobs([tok], k, i, offset) if k else None)
            offset += len(tok)
            await asyncio.sleep(0.01)
        yield _chunk("", finish=finish, usage=_usage(body, len(toks)))
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "stub": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("STUB_PORT", "8900")))
