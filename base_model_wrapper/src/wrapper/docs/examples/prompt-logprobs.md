# prompt_logprobs

Return the model's top-*k* logprob predictions at each prompt position — useful for inspecting what the model expected at each step, and (with extra care, see the Gotcha) a building block for scoring how likely a given piece of text is under the model.

## curl

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b",
       "prompt": "The capital of France is",
       "max_tokens": 1,
       "prompt_logprobs": 5}' | jq '.choices[0].prompt_logprobs[:2]'
```

## Python

```python
resp = client.completions.create(
    model="llama-8b",
    prompt="The capital of France is",
    max_tokens=1,
    extra_body={"prompt_logprobs": 5},
)

# resp.choices[0].prompt_logprobs[i] is a {token_id: {logprob, rank, decoded_token}}
# dict for prompt position i, or None for the very first position (no
# conditioning context). Here we just print the top-5 at the first few positions.
for i, choices in enumerate(resp.choices[0].prompt_logprobs[:3]):
    if choices is None:
        continue
    print(f"position {i}:")
    for token_id, info in choices.items():
        print(f"  rank={info['rank']:>2}  logprob={info['logprob']:+.3f}  {info['decoded_token']!r}")
```

## Example response

Just `choices[0].prompt_logprobs[:2]` (what the curl `jq` filter shows):

```json
[
  null,
  {
    "14924": {"logprob": -1.179518699645996, "rank": 1, "decoded_token": "Question"},
    "755":   {"logprob": -2.179518699645996, "rank": 2, "decoded_token": "def"},
    "2":     {"logprob": -2.742018699645996, "rank": 3, "decoded_token": "#"},
    "791":   {"logprob": -3.679518699645996, "rank": 4, "decoded_token": "The"},
    "16309": {"logprob": -4.179518699645996, "rank": 5, "decoded_token": "Tags"}
  }
]
```

Position 0 is `null` (no conditioning context for the first token). At position 1 the model's top-1 prediction is `"Question"`, not the actual prompt token `"The"` — which here happens to show up at rank 4. If your prompt token isn't in the top-*k*, vLLM still appends it as an extra entry with `rank > k` so you can recover its logprob; see the Gotcha for how to use that to score a sequence.

## Full vocabulary (`prompt_logprobs=-1`)

Pass `prompt_logprobs=-1` to get the model's **entire** next-token distribution at each prompt position, not just the top *k* — the whole `vocab_size` (~128k for Llama) of logprobs per position. This is the raw material for exact surprisal, KL between models, or full-distribution interpretability, with no truncation.

A fixed top-*k* covers most uses — ranking candidates, classification by next-token probability, sequence scoring (see the Gotcha) — and works at any prompt length. Reach for `-1` when you need the tail of the distribution, and budget for the prompt-length cap below.

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept-Encoding: gzip" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is", "max_tokens": 1, "prompt_logprobs": -1}'
```

Two things to know:

- **Prompt-only, capped prompt length.** The full distribution is enormous (~vocab_size values *per token*), so `-1` is accepted only up to `capabilities.full_vocab_max_prompt_tokens` (from `GET /v1/models`) — currently **1024 tokens** for every model, a few pages of text (this page's example prompt is 6 tokens). A longer prompt returns `400 full_vocab_prompt_too_long`. Use a fixed top-*k* (e.g. `prompt_logprobs=20`) for longer prompts. Completion `logprobs` can't be `-1` — full vocab is prompt-only. Budget for the download on long prompts: at ~128k vocab the body is roughly 4 MB per prompt token before compression, and the response streams — first bytes can take a while while the model server serializes.
- **Send `Accept-Encoding: gzip`.** Most HTTP clients (the OpenAI SDK, `requests`, `httpx`, browsers, `curl --compressed`) do this by default and transparently decompress; the wrapper gzips the response, which shrinks these large logprobs bodies several-fold.

### Full distribution over generated text

`-1` covers prompt positions only — vLLM rejects `logprobs=-1` on generated tokens, so those stay top-*k*. To get the full distribution at every step of a generation anyway, do it in two passes: generate first, then re-score prompt + completion as one prompt.

```python
gen = client.completions.create(
    model="llama-8b", prompt=prompt, max_tokens=64, seed=7
)
full_text = prompt + gen.choices[0].text

rescored = client.completions.create(
    model="llama-8b",
    prompt=full_text,
    max_tokens=1,
    extra_body={"prompt_logprobs": -1},
)
# rescored.choices[0].prompt_logprobs[i] is now the full ~128k-entry
# distribution at position i — including every generated position.
```

It's the same forward pass over the same tokens, so the distributions match what the model saw while generating (up to float noise). Cost: one extra prefill. The combined text must fit the full-vocab prompt cap (`capabilities.full_vocab_max_prompt_tokens`).

## Gotcha

**Scoring a sequence.** `rank=1` is the model's top prediction at each position, *not* necessarily your actual prompt token — and if your token wasn't in the top-*k*, vLLM still appends it as an extra entry with `rank > k`, so its logprob is recoverable. To get the true log-likelihood of your prompt: set `echo=true` to recover the prompt tokens, then at each position pick the entry whose `decoded_token` matches (use `prompt_logprobs=20` to keep the top-*k* generous, and fall back to the appended extra entry when the actual token ranked lower). `max_tokens` must be `≥ 1`, so set it to `1` and ignore the single generated token.
