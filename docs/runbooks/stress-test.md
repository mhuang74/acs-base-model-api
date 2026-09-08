---
title: Pre-beta stress test (ACS-45)
status: current
updated: 2026-06-23
owner: platform@example.org
---

# ACS-45 — Pre-beta stress test

Two scripts to run before opening base-model-api to beta testers. Both write
researchlog-shaped `Result:` blocks to stdout (paste into `researchlog.md` after
the run).

## What each suite covers

### `benchmarks/s4_param_matrix.py` — observability + determinism sweep

Functional check that `logprobs`, `prompt_logprobs`, and `seed` round-trip
end-to-end under concurrency. The throughput benchmarks (`s1`, `s2`, `s3`,
`concurrency_sweep`) only fire vanilla completions; this fills the gap.

- **Matrix**: 11 cells of (`logprobs` ∈ {None, 5, 20}) × (`prompt_logprobs` ∈
  {None, 5}) × (`seed` ∈ {None, 42}), all-None skipped. Each cell fires
  `--prompts-per-cell` (default 20) requests at `--concurrency` (default 32).
  Per-request CSV row records shape checks: `logprobs_shape_ok`,
  `prompt_logprobs_shape_ok`.
- **Determinism phase**: 5 prompts × 2 sequential fires at `seed=42`,
  `temperature=0.7`. Records both byte-identical match and first-20-token
  prefix match. The prefix metric is the realistic signal — vLLM seed
  reproducibility is best-effort under continuous batching.
- **Route**: Modal-direct (`MODAL_BASE_URL` + `VLLM_API_KEY`), same as
  `concurrency_sweep.py` — bypasses the wrapper's 120/min slowapi cap.
- **Cost**: `AbortGuard("s4_param_matrix", $40)`. Default config ~$15–20.

### `benchmarks/fuzz_api.py` — adversarial / try-to-break-it suite

Data-driven case list. Each case is a `FuzzCase(name, category, severity, fn,
mutating)`. The runner fires cases sequentially, records every result, and
exits 1 if any 500 was seen or any case failed.

Categories: `sampler_boundary`, `float_edge` (NaN/±Inf via raw JSON bytes;
huge ints and far-out numeric values), `unknown_field`, `wrong_type`,
`payload_extreme`, `unicode`, `streaming_abuse`, `authn`, `rate_limit`
(optional, behind `--include-burst`), `web_ui` (GET-only by default).

- **Route**: wrapper (`ACS_API_BASE` + `ACS_API_KEY`). We want the full stack
  exercised: auth, rate-limit, validation, budget clamp, proxy reliability,
  vLLM, error mapping.
- **Read-only by default**: cases tagged `mutating=True` (real signup POST,
  password-reset emails) are skipped unless `--allow-mutating` is passed.
- **Cost**: `AbortGuard("fuzz_api", $30)`. Most cases short-circuit at the
  Pydantic validator and never reach Modal compute.

## How to run

Both against prod. The schemas live at
`base_model_wrapper/src/wrapper/schemas.py` if you need to add a boundary case.

```bash
# 1. Param matrix — Modal-direct
export MODAL_BASE_URL=https://<workspace>--acs-llama-405b-serve.example.modal.run
export VLLM_API_KEY=sk-...

python -m benchmarks.s4_param_matrix --help
python -m benchmarks.s4_param_matrix --dry-run          # free, prints plan
python -m benchmarks.s4_param_matrix                    # ~$15–20

# 2. Fuzz suite — wrapper
export ACS_API_BASE=https://infra.acsresearch.org
export ACS_API_KEY=acsk_...

python -m benchmarks.fuzz_api --help
python -m benchmarks.fuzz_api --dry-run                 # free, prints cases
python -m benchmarks.fuzz_api                           # ~$1–2 (most cases free)
python -m benchmarks.fuzz_api --include-burst           # +$8, fires --burst-n requests
```

Both scripts print a one-line `PREFLIGHT:` echo before launching (per the
global "paid runs — echo and proceed" rule).

## How to triage failures

**Param matrix:**

- `seed_match_exact = 0/5` but `seed_match_prefix = 5/5` → known-limitation
  signal. vLLM continuous batching breaks bit-identity but tokens stay
  determined. The 2026-06-16 prod run got 5/5 byte-identical on a warm
  container, so byte-identity *is* achievable; treat a regression to 0/5 as a
  flag.
- `seed_match_prefix < 5/5` → seed isn't reproducible at all. **Escalate.** Open
  a bug, gate beta on the fix (or document the limitation explicitly for
  research users).
- `lp_ok < n_requested` at `logprobs=k` with k near `MAX_LOGPROBS=100`: known
  behavior under concurrency. At batch size 32, ~25% of responses have at
  least one output position where the `top_logprobs` dict has fewer than `k`
  entries — vLLM occasionally returns two distinct token IDs that decode to
  the same string, and the JSON object dedupes them. Sequential probes
  return 20/20. Document for research users; not a fixable wrapper bug.
- `lp_ok < n_requested` at `logprobs=5`: should be 100%. If not, vLLM is
  truncating mid-distribution — file a bug.
- `plp_ok < n_requested`: prompt_logprobs is supposed to live under
  `choices[0].prompt_logprobs` in vLLM 0.19.x. If this regresses to 0/n,
  vLLM probably bumped to a version that moves the field again — adjust the
  shape check.
- A cell with `n_ok = 0` → likely an upstream vLLM rejection. Look at one
  failing request in the Sentry feed or Modal logs.

**Fuzz suite:**

- **Any 500** → critical. The no-500 invariant is broken. Look up the case
  name in the CSV; the `body_excerpt` column has the first 200 chars of the
  response — usually enough to find the unhandled exception path in the
  wrapper.
- **`authn/*` failure** → potential auth bypass. **Critical**, block beta.
- **`web/get_admin_users_unauth` returns 200** → admin list exposed unauth.
  **Critical**, block beta.
- **`payload/*` returns 200 or timeout** → context-window guard or body-size
  limit missing. High-priority but not necessarily blocking.
- **`unicode/*` returns non-200** → tokenizer or JSON-encoding bug. High-priority.
- **`streaming/disconnect_then_followup` follow-up status ≠ 200** → slot leak.
  Critical — one flaky client could pin every wrapper slot.

When you find a real bug, fix it as its own PR. The plan for ACS-45 doesn't
include any wrapper changes — these scripts are pure observers.

## Smoking on the dev wrapper (`scripts/dev-local.sh`)

The dev wrapper points at `MODAL_BASE_URL=https://example.invalid` — no real
vLLM upstream is wired. Running the fuzz suite against it is still useful as a
framework smoke (case-runner code paths, validation, auth, web-UI redirects),
but cases that need a real upstream will fail.

Expected dev-run failure set, all explained by "no upstream":

- `unicode/*` → 502/503 (`upstream_unreachable` then `circuit_open` after a few)
- `streaming/disconnect_then_followup` → 503
- `payload/1000_entry_prompt_list` → 502 (passes validation, tries upstream)

Everything else (61/69 cases at time of writing) should pass on dev. If the
boundary / unknown-field / wrong-type / authn / web-UI categories show
unexpected failures on dev, that's a real bug — the upstream is irrelevant to
those code paths.

To mint a key for the dev run:

```bash
cd base_model_wrapper && DATABASE_URL="postgresql://pg:pg@localhost:5434/acs" \\
  uv run python -m cli.acs_keys create dev@acsresearch.org --name fuzz-smoke
# copies the printed acs-bm-… key
cd <repo-root>
ACS_API_BASE=http://localhost:5173 ACS_API_KEY=<paste> \\
  uv run python -m benchmarks.fuzz_api
```

## Recurring vs one-shot

These are **one-shot pre-launch checks**, not CI. If we want recurring
adversarial coverage post-launch, that's a separate ticket — talk to the
admin about how to size a low-cost daily run.

## Reference

- Plan file: `~/.claude/plans/delightful-waddling-owl.md` (local).
- Linear issue: ACS-45.
- Schemas under test: `base_model_wrapper/src/wrapper/schemas.py`.
- Existing benchmark infra reused: `benchmarks/harness.py`, `benchmarks/billing.py`.
