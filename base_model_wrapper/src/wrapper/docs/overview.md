# Using the base-model API

> **Coding with an AI assistant?** Point it at [`{{SITE_BASE}}/llms.txt`]({{SITE_BASE}}/llms.txt) — the one-page version of the tutorial and API docs — and it can write your client for you.

This app provides access to **base models** — raw next-token prediction, no chat template, no instruction tuning — served over an OpenAI-compatible `/v1/completions` endpoint.

## What you get

- **Three base models** — a small one for quick tests plus two larger ones (see [Models](/tutorial/models)). Some are kept warm; others cold-start on first use.
- **Real next-token access** — arbitrary prefill/continuation, `logprobs` and `prompt_logprobs` for likelihood/surprisal and interpretability work, `echo`, and SSE streaming.
- **Sampling controls** — `temperature`, `top_p`, `top_k`, `min_p`, penalties, and a `seed` that's actually honored for reproducibility.
- **A predictable API** — OpenAI-compatible `/v1/completions` with strict input validation (a typo'd parameter fails loudly instead of silently defaulting) and clear, structured JSON errors.
- **Browser Workbench** — try prompts and manage API keys without writing code.
- **Per-key budgets & usage** — set token caps per key and track spend (see [Account](/tutorial/account)).
- **Feedback** — a one-click Feedback button in the Workbench for feature requests and bug reports.

## Two ways in

- **Workbench** — prompt the models straight from your browser. Good for getting a feel before you write any code.
- **The API** (below) — for anything programmatic. Create a key from your dashboard.

## Quick start

Create a key in your dashboard, then point any OpenAI-compatible client at the API:

```bash
export ACS_API_KEY="acs-bm-..."          # your key
export ACS_API_BASE="{{API_BASE}}"
```

A first request with `curl`:

```bash
curl -s "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "The capital of France is", "max_tokens": 8}'
```

Or with the Python SDK (`pip install openai`):

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["ACS_API_BASE"],
    api_key=os.environ["ACS_API_KEY"],
)

resp = client.completions.create(   # completions — not chat.completions
    model="llama-8b",
    prompt="The capital of France is",
    max_tokens=16,
    logprobs=5,
)
print(resp.choices[0].text)
```

## Worked examples

Short, copy-pasteable end-to-end snippets for each feature live under [Examples](/tutorial/examples) — one page per topic. The curl examples assume `ACS_API_KEY` + `ACS_API_BASE` are exported (see [Quick start](#quick-start)); the Python examples use the same `openai` SDK client as above.

New to base models? The hands-on [**Colab workshop**](https://colab.research.google.com/github/acsresearch/base-models-workshop/blob/main/workshop_student.ipynb) is the fastest way in — it runs entirely in your browser (no local setup) and walks the whole arc end-to-end: completions, sampling, `logprobs`, reading a model's activations, and steering it to talk like a pirate. The per-feature snippets under [Examples](/tutorial/examples) then take each topic one at a time.
