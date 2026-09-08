# echo

Prepends the prompt to the generated text in `choices[0].text`. Combined with `prompt_logprobs` and `max_tokens=1` (the wrapper's minimum), it turns a request into pure text-scoring — only one token is generated, which you ignore.

## curl

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "Once upon a time", "max_tokens": 8, "echo": true}'
```

## Python

```python
resp = client.completions.create(
    model="llama-8b",
    prompt="Once upon a time",
    max_tokens=8,
    echo=True,
)
print(resp.choices[0].text)  # "Once upon a time there was a man who was very poor"
```

## Example response

```json
{
  "id": "cmpl-b986142b3fd7d215",
  "object": "text_completion",
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": "Once upon a time there was a man who was very poor",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
}
```

Note that `text` contains both the prompt and the 8 generated tokens, concatenated with no separator; `usage.completion_tokens` counts only the generated portion. (`prompt_tokens=5`, not 4, because the tokenizer auto-prepends a `<|begin_of_text|>` BOS token to Llama prompts.)

## Gotcha

`echo` only mirrors the prompt back; it doesn't add a separator, so split by length if you need just the continuation.
