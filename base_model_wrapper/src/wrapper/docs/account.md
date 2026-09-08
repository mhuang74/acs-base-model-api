# Account

Where to find your keys, your spend, and your monthly budget.

## API keys

You create keys from the Dashboard once your account is approved. Each key is shown **once** at creation time — copy it then; we only ever store a hash. If you lose it, revoke the key and create a new one.

Keys can be **paused** (reversibly disabled) or **revoked** (permanent). Paused keys come back with one click; revoked keys are gone.

## Budgets

A request is checked against several token budgets before it runs. Any one of them can refuse it — and the `429 budget_exceeded` message **names the dimension that was hit**, so you know which cap to ask us to raise.

- **Per-key monthly budget** — a token cap (`prompt + completion`) that resets on the 1st (UTC). You can set this per key when you create it; "unbounded" means no per-key sub-cap (the account-level total still applies).
- **Account total (monthly)** — an aggregate cap across all your keys. Set by an admin; visible on the dashboard. A per-key cap can't exceed the account total.

On top of those two totals, a key can carry optional **sub-budgets** that scope the cap more tightly. Each is opt-in — when unset, only the monthly totals above apply:

- **Daily budget** — a per-key `prompt + completion` ceiling that resets every day (UTC), for pacing a key across a longer run rather than letting it burn the whole month in one batch.
- **Monthly input-token budget** — a per-key cap on **prompt** tokens only.
- **Monthly output-token budget** — a per-key cap on **completion** tokens only.

When a request would exceed **any** of these, it returns `429 budget_exceeded` with a message naming the dimension that was hit. [Budget-cap recovery](/tutorial/examples/budget-cap) shows how to handle it in code — the short version: don't retry. Monthly, input, and output caps are sticky until the 1st; the daily cap until tomorrow (UTC).

## Where things live

- **Dashboard** — keys, this-month's spend per key, create / pause / revoke buttons.
- **Usage** — usage chart with per-model breakdown for the current month, plus the running totals you'll need for budget planning.
- **Workbench** — in-browser prompt UI; saved prompts persist across browsers.

Once you're signed in, all three pages are linked from the top nav.

## Asking for more

If you need a bigger budget, an always-on warm window, or access to a model that isn't on the list — use the in-app **Feedback** button or [email infra@acsresearch.org](mailto:infra@acsresearch.org). We read everything; please include your key prefix or name so we can find your account.
