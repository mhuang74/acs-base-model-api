# Cold-boot waiting

Large on-demand models sleep when idle. The first request after scale-to-zero takes a few minutes — usually under 10, occasionally longer for the largest models — while Modal allocates GPUs and loads the model. Keep that **one request** open: the wrapper sends harmless whitespace bytes for non-streaming JSON (or SSE comments for `stream=true`) so Railway and your client do not treat the connection as idle.

Set a client timeout of at least 15 minutes. Do not add a retry loop just to handle startup — the original request should eventually return the completion.

## curl

```bash
curl --fail-with-body --no-buffer --max-time 900 \
  "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-405b", "prompt": "Hello", "max_tokens": 4}'
```

## Python

```python
from openai import OpenAI

client = OpenAI(
    api_key=ACS_API_KEY,
    base_url=ACS_API_BASE,
    timeout=900,
    max_retries=0,  # the first request itself absorbs the cold boot
)

resp = client.completions.create(
    model="llama-405b",
    prompt="Hello",
    max_tokens=4,
)
print(resp.choices[0].text)
```

## Wire-format note

For `stream=false`, you may see whitespace arrive before the JSON object if you inspect raw bytes. Leading whitespace is valid JSON and normal clients buffer it before parsing. For `stream=true`, the keepalives are SSE comment frames and are ignored by OpenAI SDKs and EventSource clients.

If startup exceeds the wrapper's 14-minute safety budget, the already-open response ends with a JSON/SSE error payload. That is an allocation failure, not a signal to blindly retry forever.

## Interrupted request (rare): the connection drops mid-cold-boot

There is one case where retrying *is* the right move. The keepalive bytes commit an HTTP **200** status as soon as they start flowing — before the model has produced anything — and the body is sent as a chunked stream. If the wrapper process is restarted while still holding your request open (for example, a deploy lands during your cold boot), the connection closes **before the final chunk**, so you get a truncated response with no completion. How that surfaces depends on your client:

- **Python / OpenAI SDK (`stream=false`)**: a **connection error**, not a readable response — `openai.APIConnectionError` (an incomplete chunked read underneath, `httpx.RemoteProtocolError`). Catch it and retry.
- **`curl` (`stream=false`)**: partial bytes (often just whitespace) and a **non-zero exit** from the truncated transfer. Retry.
- **`stream=true` (SDK or EventSource)**: the SSE stream closes **without** a terminal completion or `error` frame. Treat "stream ended, no terminal frame" as interrupted and retry.

This is uncommon — it needs a server restart to coincide with an in-flight cold boot — and it is **safe to retry**.

**Do not confuse this with a real error response.** A genuine failure (allocation timeout, budget exceeded, upstream error) arrives as a complete, parseable OpenAI-shaped JSON `{"error": {...}}` body (with a `200` for late cold-boot failures, since the status was already committed — see above). That is a real outcome with a code to act on, **not** a retry signal. The retry guard is "the transfer was *interrupted*" (transport error / truncated body / no terminal SSE frame) — **not** merely "the JSON has no `choices`," which an `{"error": ...}` body also lacks.
