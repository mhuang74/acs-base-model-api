# Evaluate with Inspect

[Inspect](https://inspect.aisi.org.uk/) (UK AI Safety Institute) is a widely-used framework for LLM evals. Its **`openai-api-completions`** provider talks to a raw OpenAI-compatible `/v1/completions` endpoint — raw prompts, **no chat template** — which is exactly what this API is, so you can run base-model evals against any model we host.

Install Inspect (≥ 0.3.245) plus `openai`, which the provider needs:

```bash
pip install "inspect-ai>=0.3.245" openai
```

## Point Inspect at this API

Inspect's `openai-api` family reads credentials from `<PROVIDER>_API_KEY` / `<PROVIDER>_BASE_URL`, where `<PROVIDER>` is a name you pick. Using `acs`:

```bash
export ACS_API_KEY="sk-acs-..."                       # your key from the dashboard
export ACS_BASE_URL="https://infra.acsresearch.org/v1"

inspect eval your_task.py --model openai-api-completions/acs/llama-8b
```

The id after the last `/` is a **short model id from [`GET /v1/models`](/tutorial/models)** (e.g. `llama-8b`, `trinity-truebase`, `llama-405b`) — not the HF repo name.

## A minimal task

```python
# capitals.py
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

@task
def capitals():
    return Task(
        dataset=[
            Sample(
                input="The capital of Spain is Madrid.\n"
                "The capital of Germany is Berlin.\n"
                "The capital of France is",
                target="Paris",
            )
        ],
        solver=generate(),
        scorer=includes(),
    )
```

```bash
inspect eval capitals.py --model openai-api-completions/acs/llama-8b --temperature 0
```

Because these are **base models**, write prompts as raw text the model *continues*, not chat turns — few-shot prefixes and logprob scoring work well; instruction-style prompts often won't. The two example lines above are load-bearing: from the bare prompt `The capital of France is`, llama-8b's greedy continuation is ` a city of many faces. It is` and the eval scores 0. With the prefix its first sampled token is ` Paris` and the eval scores 1.

## Settings that matter here

- **Concurrency** — set `--max-connections` to the per-key cap (**16**); a higher value just queues server-side. Each key allows **16 active + 64 queued** requests and **600 starts/min**; a large eval fan-out otherwise hits `429 queue_full` / `rate_limited`. (Same limits as [Batch rollouts](/tutorial/examples/batch-rollouts).)
- **Pick an always-on model** (`llama-8b`, `trinity-truebase`) for big runs, or expect the first calls to wait through a **cold boot** on on-demand models like `llama-405b`. See [Cold-boot waiting](/tutorial/examples/cold-boot).
- **Sampling + logprobs** — Inspect's `GenerateConfig` maps onto our params: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `logprobs` / `top_logprobs`, and `num_choices` → `n` (≤ 16). Logprob-based scorers ride on [`logprobs`](/tutorial/examples/logprobs).
- **Pre-tokenized prompts** — the provider can send token ids via `prompt_token_ids` message metadata; we forward them to the model **verbatim** (no re-tokenization). See [`prompt`](/tutorial/api#prompt) in the API reference. To recover the ids the server uses for a text prompt, send it once with `prompt_logprobs=0`: each prompt position after the first then carries exactly its own token id. The first position has no logprobs entry (nothing precedes it) — on the Llama models it is the BOS token, so prepend `128000` yourself. vLLM's `return_token_ids` request field is rejected as an unknown param.

## Get help

Eval-specific question, or a provider bug? Email [infra@acsresearch.org](mailto:infra@acsresearch.org) or use the in-app **Feedback** button. If a specific call failed, include its `X-Request-Id`.
