# Budget-cap recovery

When a token budget runs out, requests return `429 budget_exceeded`. There is no automatic refill mid-period — monthly caps reset on the 1st (UTC), the daily cap at midnight (UTC). The `message` always names which budget was hit (see [Account → Budgets](/tutorial/account)).

## curl

```bash
curl -i "$ACS_API_BASE/completions" \
  -H "Authorization: Bearer $ACS_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "llama-8b", "prompt": "Hi", "max_tokens": 8}'
# HTTP/1.1 429 Too Many Requests
# The message is dimension-specific — one of:
#   "Monthly token budget exhausted for this API key."
#   "Monthly token budget exhausted for this user (aggregate across keys)."
#   "Daily token budget exhausted for this API key."
#   "Monthly input-token budget exhausted for this API key."   (or output-token)
# {"error": {"code": "budget_exceeded", "message": "Daily token budget exhausted for this API key.", ...}}
```

## Python

```python
from openai import APIStatusError

try:
    resp = client.completions.create(model="llama-8b", prompt="Hi", max_tokens=8)
except APIStatusError as e:
    # The wrapper nests the error code under body["error"]["code"], not at
    # the top level — so e.code (which the openai SDK reads from
    # body["code"]) is None here. Dig into e.body instead.
    err = (e.body or {}).get("error", {}) if isinstance(e.body, dict) else {}
    if e.status_code == 429 and err.get("code") == "budget_exceeded":
        # Don't retry — monthly caps are sticky until the 1st (UTC), the daily
        # cap until tomorrow. err["message"] names which dimension was hit.
        # Email infra@acsresearch.org to raise the budget.
        raise SystemExit(f"Budget exhausted ({err.get('message')}) — emailing the team for a raise.")
    raise
```

## Gotcha

`429 budget_exceeded` is *not* a transient rate-limit — exponential backoff just burns time. Inspect `error.code` and short-circuit non-retryable cases (concurrency limits, by contrast, simply queue).
