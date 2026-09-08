# Repository Guidelines

## Project Overview

ACS base-model platform: an OpenAI-compatible API wrapper in front of a Modal/vLLM base-model endpoint, plus auth, key/budget management, usage accounting, and a server-rendered web UI (Workbench, Loom, Compare). Two packages: the repo-root Modal app (`modal_app.py`, `serving/`) is the GPU side; `base_model_wrapper/` is the FastAPI "wrapper" package where API work happens. Take-home-style repo: `SOLUTION-NOTES.md` is an empty template for candidate write-up.

## Architecture & Data Flow

Request flow (API server):

```
request → RequestIdMiddleware → route (slowapi @limiter.limit) → bearer-key auth + budget check (auth.py)
→ proxy.py / modal_ops.py forwards via app.state.http (httpx) to ModelEntry.upstream_url
  (or activation_upstream_url for harvest/steering)
→ SSE stream or gzip'd JSON response → usage commit (usage_monthly/usage_daily) + request logging
```

Key modules (all under `base_model_wrapper/src/wrapper/` unless noted):

- `main.py` — ASGI target (`wrapper.main:app`); composition root + large compat re-export/shim surface (tests patch private names via `wrapper.main`).
- `app.py` — `create_app()` factory; hand-written pure-ASGI `RequestIdMiddleware` (BaseHTTPMiddleware avoided: it buffers StreamingResponse and breaks SSE); CORS `allow_credentials=False`; custom OpenAPI scoped to `/v1/*` + `/health`.
- `lifespan.py` — builds all runtime state once at startup (engine, httpx client, model registry, tokenizers, APScheduler jobs) into `WrapperRuntime` on `app.state`; disposes engine at shutdown.
- `runtime.py` — `WrapperRuntime` dataclass on `app.state` (settings/engine/sessions/http/models/generations/breakers/scheduler, with legacy aliases).
- `routes/api.py` — public OpenAI-compat API: `/health`, `/v1/models`, `/v1/completions` (SSE streaming, logprobs, steering), `/v1/harvest` submit/status/cancel; `/v1/chat/completions` is an explicit 501.
- `routes/web.py`, `routes/admin/`, `routes/workbench.py`, `routes/loom.py`, `routes/usage.py`, `routes/discord.py` — cookie-authed web/admin surface (Jinja2 templates in `src/wrapper/templates/`, no JS build; `static/` = favicon only).
- `auth.py` / `web_auth.py` — bearer API keys with multi-dimensional budgets (per-key monthly/daily/per-direction + per-user aggregate); cookie sessions for web/admin.
- `db.py` — async SQLAlchemy (asyncpg); `_to_async_dsn` rewrites `postgres://` → `postgresql+asyncpg://`; `get_session` request-scoped dep.
- `services/` — logic extracted from big route modules (completions, availability, workbench, request_log, usage_reports, health, customer360, user_tags).
- `stub_upstream.py` (repo root) — fake GPU backend on :8900 (canned tokens, fake logprobs, usage counts); `STUB_MODE=ok|drop_midway|error_500` for misbehavior drills; dev-runtime only, NOT used by tests.
- `serving/` + root `modal_app.py` — Modal serving side; `modal deploy modal_app.py` is the canonical deploy entrypoint.

## Key Directories

- `base_model_wrapper/` — the API server package (src layout: `src/wrapper/`, plus root-level `cli/` and `acs_model_registry/` inside it), `tests/`, `alembic/` (migrations 0001–0045), `pyproject.toml`, `uv.lock`, `Dockerfile`, `entrypoint.sh` (prod/Railway: alembic + uvicorn, graceful shutdown tuned < Railway drainingSeconds).
- `serving/` — secondary Modal operator scripts (capacity probes, prestage, smoke); see its README.
- `docs/` — `_index.md` docs map; `architecture/` (living reference: modal-app, modal-phases, cost-monitoring, activation-harvesting), `runbooks/` (error-troubleshooting, activation-go-live, stress-test, Metabase SQL), `agents/` (AI-agent guidance: domain.md, issue-tracker.md, triage-labels.md).
- `base_model_wrapper/src/wrapper/docs/` — in-app tutorial content served at `/tutorial` (suite-tested; not part of `docs/`); uses `{{SITE_BASE}}`/`{{API_BASE}}` placeholders.
- `setup-dev.sh` — all local dev orchestration.

## Development Commands

```bash
# Local dev (repo root; requires Docker + uv; first boot needs internet)
./setup-dev.sh up        # Postgres(5434) + stub backend(8900) + API(5173), foreground; prints API key + web login
./setup-dev.sh key       # mint a fresh API key
./setup-dev.sh login     # re-print web-UI login
./setup-dev.sh down      # stop DB + stub

# Port-override example
PG_CONTAINER=acs-devdb-2 PG_PORT=5435 STUB_PORT=8901 APP_PORT=5174 ./setup-dev.sh up

# Tests (from base_model_wrapper/, dev server still up)
TEST_DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  uv run --python 3.12 --extra dev pytest                    # full suite ~1059 tests, 3–5 min
TEST_DATABASE_URL=postgresql://pg:pg@localhost:5434/acs \
  uv run --python 3.12 --extra dev pytest tests/test_harvest_api.py   # single file

# Lint (ruff, configured in base_model_wrapper/pyproject.toml: line-length 100, target py312)
uv run --python 3.12 --extra dev ruff check .        # currently passes; this is the lint gate
# NOTE: the repo is NOT format-clean (~96 files would be reformatted). Do NOT run `ruff format .`
# or blanket-reformat; if needed, format only the specific files you touched:
uv run --python 3.12 --extra dev ruff format <files-you-touched>

# Migrations (as setup-dev.sh does on up)
uv run alembic upgrade head

# CLI (installed via [project.scripts])
uv run acs-keys create <email> --create-user --name <key-name>   # plaintext key printed once

# Build / lock
uv build && uv lock --upgrade   # modal is capped <1.6 (ACS-125); minor bump is a tracked migration
```

No CI, no mypy/coverage config — don't invent typecheck/coverage commands. All commands run from `base_model_wrapper/` unless noted.

## Code Conventions & Common Patterns

- **Async everywhere**: asyncpg + `httpx.AsyncClient`; `asyncio.to_thread` for CPU-bound work (gzip/JSON serialization); per-key semaphores for inflight caps; per-backend circuit breakers (`<model>` and `<model>::activation` keys).
- **State via `app.state` + DI**: everything is built once in lifespan into `WrapperRuntime`; deps `get_settings`/`get_http` (`dependencies.py`) and `get_session` (`db.py`) read off `request.app.state`. Don't reach for globals.
- **Errors**: OpenAI-style envelope `{"error": {message, type, code}}` via `HTTPException` detail; `ErrorKind` vocabulary in `error_kinds.py`; structured auth-failure logging (secrets never logged).
- **Logging**: module-level `log = get_logger()` (structlog); structured kwargs, e.g. `log.warning("x", error=...)`.
- **Comments cite ticket IDs** (`ACS-###`) — house convention for "why" context; follow it.
- **Naming**: snake_case modules; routers in `routes/`, extracted logic in `services/`; DB tables snake_case; test files `tests/test_<area>.py`.
- **SlowAPI**: per-route `@limiter.limit` decorators; `SlowAPIMiddleware` deliberately NOT installed (buffers responses, broke SSE); `app.state.limiter.enabled=False` in tests.
- **CORS** read at import time in `main.py` (before lifespan) — a subtle ordering constraint.

## Important Files

- `base_model_wrapper/src/wrapper/main.py` — ASGI target; the supported patch surface (compat shims propagate `wrapper.main` patches into `routes/api.py`/admin module globals per call).
- `base_model_wrapper/src/wrapper/app.py` — app factory, middleware, OpenAPI shaping.
- `base_model_wrapper/src/wrapper/settings.py` — pydantic-settings; required: `DATABASE_URL`, `MODAL_BASE_URL`, `VLLM_API_KEY`, `ADMIN_TOKEN`; `MODELS_REGISTRY_JSON` → `ModelEntry` registry (upstream, activation side-car, tokenizer, Modal app names).
- `base_model_wrapper/src/wrapper/db.py` / `dependencies.py` / `runtime.py` / `lifespan.py` — DB engine, DI, runtime state.
- `base_model_wrapper/README.md` — load-bearing 365-line API contract (endpoint surface, sampling-param table, logprobs, limits, error codes, retry policy, circuit breaker, /health, acs-keys CLI). Note: its "Local dev" section is the older manual path — root `README.md` + `setup-dev.sh` are operative.
- `base_model_wrapper/pyproject.toml` — package, dev extra, pytest + ruff config.
- `base_model_wrapper/uv.lock` — committed; Dockerfile builds `--frozen`, so regen on any dependency change.
- `setup-dev.sh` — single local-dev entrypoint.
- `stub_upstream.py` — fake upstream; see its docstring for `STUB_MODE` behaviors.
- `docs/_index.md` — docs taxonomy/map.

## Runtime/Tooling Preferences

- **Python ≥3.12 everywhere** (wrapper, Dockerfile `python:3.12-slim`, `UV_PYTHON=3.12`); use `uv run --python 3.12` in commands.
- **uv** is the only sanctioned package manager (no pip/pip-tools); two-project layout with **no uv workspace**: root `pyproject.toml` is a bare dependency manifest for the Modal side (not installable, no lockfile); `base_model_wrapper/` is the real setuptools package with the committed `uv.lock`.
- **Dev extra is opt-in**: `pytest`/`ruff`/`respx`/`aiosqlite` live in `--extra dev`; `uv sync`/`uv run` without `--extra dev` silently omits them. Production installs (`Dockerfile`) use `uv sync --frozen --no-dev`.
- **Docker** for local Postgres (`postgres:16`, container `acs-hiring-devdb`, pg/pg on 5434); dev credentials are non-secret placeholders.
- Env vars: see `setup-dev.sh` (dev defaults) and `base_model_wrapper/.env.example` / root `.env.example` (runtime/Modal side). `materials/`, `scratchpad/`, `notebooks/` are gitignored local working areas — never commit them.

## Testing & QA

- Framework: **pytest ≥8.3 + pytest-asyncio** (`asyncio_mode = "auto"`), `testpaths = ["tests"]` in `base_model_wrapper/pyproject.toml`. `respx` for HTTP stubbing (dev-extra-only: `test_sse_routing.py` silently self-skips without it). No coverage config.
- **Two test trees**:
  - `base_model_wrapper/tests/` (~69 files, the main suite) — DB-gated e2e + unit.
  - Root `tests/` (5 files) — DB-free unit tests of `serving.*`; outside `testpaths`, run explicitly: `pytest ../tests/`.
- **Run whole test files, never single tests by node id** (some tests depend on neighbour state — `test_harvest_api.py` truncates `harvest_jobs` and resets `_LONGPOLL_INFLIGHT` in autouse fixtures; node-id runs still trigger module fixtures and files relying on conftest env defaults fail alone).
- **`TEST_DATABASE_URL` is mandatory** for the wrapper suite; without it 450 DB-backed tests skip and **3 tests in `tests/test_startup_prewarm.py` fail** (lifespan's `make_engine('')` raises) — that failure means the variable is missing, not a code bug. DB must be migrated first (`alembic -x url=$TEST_DATABASE_URL upgrade head`, documented in `test_auth_lifecycle.py`).
- **Fixtures are per-file by convention**: each DB-gated file re-declares `TEST_DATABASE_URL` + a `dbtest = pytest.mark.skipif(...)` marker and its own `TestClient` client fixture (mutates `os.environ`, lazily imports `wrapper.main.app`, disables limiter, restores env). `conftest.py` is env-only (collection-time `setdefault`s so `wrapper.main` imports without a DB). New DB tests: copy an existing file's preamble; pure-unit tests: set required Settings env at module top before importing `wrapper.main`.
- **Stubbing conventions**: monkeypatch module seams (`modal_ops.spawn/poll/cancel_harvest`, `proxymod.post_nonstream`) with recording fakes, or `respx` for real-HTTP surfaces; no factories library. Fresh uuid rows per test instead of truncation; only globally-asserted tables (`harvest_jobs`) get truncated. `stub_upstream.py` is NOT used by tests.
- Full-suite duration 3–5 min; `RuntimeError: No active exception to reraise` printed after the summary comes from a dependency's shutdown handler — the exit code is what counts.

## Agent skills

### Issue tracker

Issues are tracked as GitHub Issues in `mhuang74-learning-playground/acs-base-model-api` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Defaults: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.