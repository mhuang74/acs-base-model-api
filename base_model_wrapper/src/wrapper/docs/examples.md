# Examples

Short, copy-pasteable end-to-end snippets for each feature. The curl examples assume `ACS_API_KEY` + `ACS_API_BASE` are exported (see [Quick start](/tutorial/overview#quick-start)); the Python examples use the same `openai` SDK client.

Each page below shows one shared **example response** after the curl and Python snippets — both calls return the same JSON; the SDK just wraps it in typed objects. Always-null fields (`service_tier`, `system_fingerprint`, `kv_transfer_params`, etc.) are dropped from the shown responses for brevity.

Your own output won't match token-for-token: the displayed requests don't pin a `seed`, so under the default `temperature=1.0` sampling varies. Add a `seed` to reproduce a specific run, or `temperature=0` for greedy decoding.

## Guided walkthrough

- [**Hands-on Colab workshop**](https://colab.research.google.com/github/acsresearch/base-models-workshop/blob/main/workshop_student.ipynb) — the from-zero path through most of what's below, run end-to-end in your browser: completions → sampling → `logprobs` → reading activations → building a steering vector, with exercises and an instructor solutions notebook — both in [our tutorial repo](https://github.com/acsresearch/base-models-workshop). Start here if the per-feature snippets feel piecemeal.

## Token-level inspection

- [`logprobs`](/tutorial/examples/logprobs) — top-*k* logprobs per generated token.
- [`prompt_logprobs`](/tutorial/examples/prompt-logprobs) — top-*k* logprobs at each *prompt* position (gotchas around rank-vs-actual-token).
- [`echo`](/tutorial/examples/echo) — include the prompt in the response.

## Streaming + concurrency

- [`stream`](/tutorial/examples/stream) — SSE token-by-token.
- [Batch rollouts](/tutorial/examples/batch-rollouts) — 16 concurrent per key, with the canonical `asyncio.gather` pattern.

## Recovery patterns

- [Cold-boot waiting](/tutorial/examples/cold-boot) — keep one request open through a large-model startup.
- [Budget-cap recovery](/tutorial/examples/budget-cap) — handle `429 budget_exceeded` (the non-retryable kind).

## Evaluation

- [Evaluate with Inspect](/tutorial/examples/evaluate-with-inspect) — run base-model evals through Inspect's `openai-api-completions` provider (raw prompts, no chat template).
