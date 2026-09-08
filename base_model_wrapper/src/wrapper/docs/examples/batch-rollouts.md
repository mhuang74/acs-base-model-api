# Batch rollouts

The per-key concurrency cap is 16; `asyncio.gather` over a single `AsyncOpenAI` client gives the maximum throughput without extra credentials.

## curl

```bash
# Shell version — xargs -P 16 fans out 16 concurrent requests (the per-key cap):
seq 1 32 | xargs -I{} -P 16 curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" -H "Content-Type: application/json" \
  -H "X-Acs-Workload: batch" \
  -d '{"model": "llama-8b", "prompt": "Sample {}: once upon a time", "max_tokens": 32}'
```

## Python

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.environ["ACS_API_BASE"],
    api_key=os.environ["ACS_API_KEY"],
    # Tag this whole client as batch traffic (see "Tag your workload" below).
    default_headers={"X-Acs-Workload": "batch"},
)
prompts = [f"Sample {i}: once upon a time" for i in range(32)]

async def one(p):
    r = await client.completions.create(model="llama-8b", prompt=p, max_tokens=32)
    return r.choices[0].text

async def _main():
    # asyncio.gather() must be called from inside a running event loop, so
    # wrap it in a coroutine and hand THAT to asyncio.run(...).
    return await asyncio.gather(*(one(p) for p in prompts))

results = asyncio.run(_main())
```

## Gotcha

Requests beyond the 16 active slots queue server-side, up to 64 waiters. A larger burst gets `429 queue_full`; sustained traffic above 600 request starts/minute gets `429 rate_limited`. Both include `Retry-After`, but the best fix is client-side backpressure: cap your own fan-out with `Semaphore(16)` so work does not pile up at the gateway.

A single request is also bounded: `prompt_count × n × max_tokens` must be at most 32,000, where `prompt_count` is the number of prompts in your list (a single string — or one pre-tokenized `list[int]` — counts as 1). If you send a batch of prompts or `n>1`, set `max_tokens` explicitly.

## Tag your workload

**Set an `X-Acs-Workload` header on your requests** — `batch` for bulk / offline / eval jobs (like these rollouts), `interactive` for live, latency-sensitive calls. Both examples above already set it (`default_headers=` on the client, or `-H` per curl call). It's a usage signal we record to understand traffic and plan capacity / keep-warm policy; it does *not* change how your request is scheduled or prioritised, and an unrecognised value is simply ignored. Setting it accurately is a small thing that genuinely helps us tune the service for everyone. The in-browser Workbench is tagged `interactive` automatically.
