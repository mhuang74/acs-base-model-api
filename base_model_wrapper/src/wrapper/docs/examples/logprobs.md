# logprobs

Ask for the top-*k* alternative tokens at each generated position (`k ≤ 100`). Useful for calibration, classification by next-token probabilities, or just inspecting what the model "almost said". (The exact cap is per-model — read `capabilities.max_logprobs` from `GET /v1/models` rather than hard-coding it.)

## curl

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is", "max_tokens": 1, "logprobs": 5}'
```

## Python

```python
resp = client.completions.create(
    model="llama-8b",
    prompt="The capital of France is",
    max_tokens=1,
    logprobs=5,
)
choice = resp.choices[0]
print(choice.text, choice.logprobs.top_logprobs[0])  # {' a': -1.74, ' Paris': -1.99, ...}
```

## Example response

```json
{
  "id": "cmpl-91ea94c6ed0e5b75",
  "object": "text_completion",
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": " a",
      "logprobs": {
        "text_offset": [0],
        "tokens": [" a"],
        "token_logprobs": [-1.7437232732772827],
        "top_logprobs": [
          {
            " a": -1.7437232732772827,
            " Paris": -1.9937232732772827,
            " one": -2.5562233924865723,
            " the": -2.5562233924865723,
            " also": -3.1187233924865723
          }
        ]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7}
}
```

Note that base `llama-8b` ranks `" a"` above `" Paris"` — it's a continuation predictor, not a Q&A assistant, so it continues the sentence rather than answering it. `" Paris"` is a close second (0.25 logprob behind), so an instruction-tuned model would likely put it first. This base model won't, even at `temperature=0` — greedy decoding just takes the top-ranked token, here `" a"`.

## Gotcha

`logprobs` is an integer (top-*k*): the number of alternatives to return per token.
