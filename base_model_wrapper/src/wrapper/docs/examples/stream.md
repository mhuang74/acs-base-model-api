# stream

Server-sent events: each chunk is a partial completion, terminated by `data: [DONE]`. Set the request-level `timeout` high enough to span the full generation.

## curl

```bash
curl -N -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "Five reasons to learn Rust:\n1.", "max_tokens": 80,
       "stream": true, "stream_options": {"include_usage": true}}'
```

## Python

```python
stream = client.completions.create(
    model="llama-8b",
    prompt="Five reasons to learn Rust:\n1.",
    max_tokens=80,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:                            # final usage chunk has choices=[]
        print(chunk.choices[0].text, end="", flush=True)
    elif chunk.usage:
        print(f"\n[tokens: {chunk.usage.total_tokens}]")
```

## Example response

First few SSE frames — the wire format; the Python SDK parses each `data:` payload into a chunk object. (Per-chunk `logprobs: null` and `stop_reason: null` keys are omitted here for brevity — they're present in the real wire output.)

```text
data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":" It","finish_reason":null}],"usage":null}

data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":"’s","finish_reason":null}],"usage":null}

data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":" fast","finish_reason":null}],"usage":null}

data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":".\n","finish_reason":null}],"usage":null}

... (one chunk per token until max_tokens or stop) ...

data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":" again","finish_reason":"length"}],"usage":null}

data: {"id":"cmpl-ae7c4833c82a33d6","object":"text_completion","model":"meta-llama/Llama-3.1-8B","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":80,"total_tokens":89}}

data: [DONE]
```

The final content chunk carries `finish_reason` alongside its token (not on an empty text). Then — because the request set `stream_options.include_usage: true` — a terminal chunk arrives with `choices: []` and a populated `usage` object, where you get the token counts for a streamed completion. Then `[DONE]` closes the stream. Drop `stream_options` and you'll skip the usage chunk and get only the content frames (the OpenAI spec requires this opt-in for usage).

## Gotcha

Use `curl -N` to disable buffering, otherwise you'll see the whole response arrive at once. Each SSE chunk's `finish_reason` is `null` until the last one.
