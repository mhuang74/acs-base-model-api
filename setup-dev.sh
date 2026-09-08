#!/usr/bin/env bash
#
# setup-dev.sh — one command to boot the whole app locally for this take-home.
#
# Starts, in order:
#   1. a throwaway local Postgres (Docker)
#   2. a fake model backend (stub_upstream.py) in the background
#   3. the API server, with its model registry pointed at the fake backend
#
# It also mints an API key and prints it, plus a ready-to-run curl command, so
# you can hit /v1/completions immediately, and seeds a web-UI login so you can
# open the Workbench / Loom / Compare pages in a browser. There is no real GPU behind this —
# the fake backend returns lorem ipsum. That is intentional: every task here is
# about the API server's own request handling, which runs before any model is
# called.
#
#   ./setup-dev.sh up      # (default) bring everything up
#   ./setup-dev.sh key     # just print a fresh API key
#   ./setup-dev.sh login   # re-seed + print the web-UI login
#   ./setup-dev.sh down    # stop Postgres + the stub
#
# Requirements: Docker running, and `uv` (https://docs.astral.sh/uv/).
# First boot needs internet (pulls the postgres image + a public tokenizer used
# for token counting). After that it runs offline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$REPO_ROOT/base_model_wrapper"

: "${PG_CONTAINER:=acs-hiring-devdb}"
: "${PG_PORT:=5434}"
: "${APP_PORT:=5173}"
: "${STUB_PORT:=8900}"
: "${STUB_MODE:=ok}"
: "${DEV_EMAIL:=candidate@example.com}"
: "${DEV_PASSWORD:=devpassword12345}"
: "${KEY_NAME:=take-home}"
: "${UV_PYTHON:=3.12}"
export UV_PYTHON

export DATABASE_URL="postgresql://pg:pg@localhost:${PG_PORT}/acs"
export MODAL_BASE_URL="https://example.invalid"
export VLLM_API_KEY="dev-vllm"
export ADMIN_TOKEN="dev-admin-token"
export SESSION_SECRET="dev-session-secret-please-rotate-0123456789abcdef"
export COOKIE_SECURE="false"
# Point the model's upstream at the local fake backend. tokenizer_repo is the
# public gpt2 so startup never needs a gated download (token counts are
# approximate locally — fine for these tasks).
export MODELS_REGISTRY_JSON="{\"llama-8b\":{\"upstream_url\":\"http://localhost:${STUB_PORT}\",\"served_model_name\":\"meta-llama/Llama-3.1-8B\",\"tokenizer_repo\":\"gpt2\",\"gpu_shape_label\":\"1xL40S\",\"status\":\"live\",\"n_layers\":32,\"activation_upstream_url\":\"http://localhost:${STUB_PORT}\"}}"

# Per-port so that two copies of this repo on different ports don't stop
# each other's stub.
STUB_PIDFILE="/tmp/acs-hiring-stub-${STUB_PORT}.pid"
STUB_LOG="/tmp/acs-hiring-stub-${STUB_PORT}.log"

ensure_db() {
  if ! docker info >/dev/null 2>&1; then
    echo "✖ Docker isn't running — start Docker Desktop first." >&2; exit 1
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    docker start "$PG_CONTAINER" >/dev/null
  else
    echo "→ creating Postgres container '$PG_CONTAINER' on :$PG_PORT"
    docker run -d --name "$PG_CONTAINER" \
      -e POSTGRES_USER=pg -e POSTGRES_PASSWORD=pg -e POSTGRES_DB=acs \
      -p "${PG_PORT}:5432" postgres:16 >/dev/null
  fi
  echo -n "→ waiting for Postgres"
  for _ in $(seq 1 30); do
    # -h 127.0.0.1: probe the TCP listener the app actually uses. Without it
    # pg_isready checks the unix socket, which answers during the image's
    # init phase while TCP is still closed — so the app could race the DB.
    docker exec "$PG_CONTAINER" pg_isready -U pg -h 127.0.0.1 >/dev/null 2>&1 && { echo " — ready"; return; }
    echo -n "."; sleep 1
  done
  echo; echo "✖ Postgres did not become ready" >&2; exit 1
}

migrate() { echo "→ alembic upgrade head"; (cd "$WRAPPER" && uv run alembic upgrade head >/dev/null); }

start_stub() {
  stop_stub
  echo "→ fake model backend on http://localhost:${STUB_PORT} (STUB_MODE=$STUB_MODE)"
  # `cd` on its own line, so `$!` is the pid of the backgrounded `uv run`
  # itself and not of a wrapping subshell that exits immediately (which would
  # leave a stale pidfile and an unstoppable stub).
  (
    cd "$WRAPPER"
    STUB_PORT="$STUB_PORT" STUB_MODE="$STUB_MODE" \
      uv run python "$REPO_ROOT/stub_upstream.py" >"$STUB_LOG" 2>&1 &
    echo $! > "$STUB_PIDFILE"
  )
  sleep 2
}

stop_stub() {
  [ -f "$STUB_PIDFILE" ] || return 0
  local pid; pid="$(cat "$STUB_PIDFILE")"
  # `uv run` forwards the signal, but reap its child too in case it doesn't.
  pkill -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
  rm -f "$STUB_PIDFILE"
}

seed_login() {
  # The web UI (Workbench, Loom, Compare) authenticates with email + password.
  # `acs-keys create` does not set one you could know, so give the dev user a
  # known password and make sure the account is approved — otherwise the
  # printed login can't actually open a browser session. A plain (non-admin)
  # user: nothing in Workbench or Loom needs the admin role.
  echo "→ seeding web login for $DEV_EMAIL"
  (cd "$WRAPPER" && DEV_EMAIL="$DEV_EMAIL" DEV_PASSWORD="$DEV_PASSWORD" uv run python - <<'PYEOF'
import asyncio, os
from sqlalchemy import select
from wrapper.db import make_engine, make_session_factory, session_scope
from wrapper.web_auth import set_password
from wrapper.models import User


async def go():
    engine = make_engine(os.environ["DATABASE_URL"])
    try:
        sf = make_session_factory(engine)
        async with session_scope(sf) as s:
            email = os.environ["DEV_EMAIL"]
            u = (await s.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if u is None:
                u = User(email=email)
                s.add(u)
                await s.flush()
            await set_password(s, u, os.environ["DEV_PASSWORD"])
            u.status = "approved"
    finally:
        await engine.dispose()


asyncio.run(go())
PYEOF
)
}

mint_key() {
  local out key
  # Keep stderr (diagnostics) visible; don't let a no-match grep abort the script.
  out="$(cd "$WRAPPER" && uv run acs-keys create "$DEV_EMAIL" --create-user --name "$KEY_NAME" 2>&1)" || {
    echo "✖ acs-keys failed:" >&2; echo "$out" >&2; return 1
  }
  key="$(printf '%s\n' "$out" | grep -oE 'acs-bm-[A-Za-z0-9_-]+' | head -1 || true)"
  if [ -z "$key" ]; then
    echo "✖ could not parse an API key from acs-keys output:" >&2; echo "$out" >&2; return 1
  fi
  printf '%s\n' "$key"
}

run() {
  # Tolerate a key-mint hiccup — the server should still boot; the candidate can
  # mint one later with `./setup-dev.sh key`. (mint_key prints its own error.)
  local key; key="$(mint_key || true)"
  echo "──────────────────────────────────────────────"
  echo "  Web UI:   http://localhost:${APP_PORT}/  (Workbench, Loom, Compare)"
  echo "  login:    ${DEV_EMAIL} / ${DEV_PASSWORD}"
  echo ""
  echo "  API key:  ${key:-<run: ./setup-dev.sh key>}"
  echo "  Try it:"
  echo "    curl -s http://localhost:${APP_PORT}/v1/completions \\"
  echo "      -H 'Authorization: Bearer ${key}' \\"
  echo "      -H 'Content-Type: application/json' \\"
  echo "      -d '{\"model\":\"llama-8b\",\"prompt\":\"hello\",\"max_tokens\":16}'"
  echo "──────────────────────────────────────────────"
  cd "$WRAPPER"
  exec uv run uvicorn wrapper.main:app --host 127.0.0.1 --port "$APP_PORT"
}

case "${1:-up}" in
  up)    ensure_db; migrate; seed_login; start_stub; run ;;
  key)   mint_key ;;
  login) seed_login; echo "${DEV_EMAIL} / ${DEV_PASSWORD}  ->  http://localhost:${APP_PORT}/" ;;
  down)  stop_stub; docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 && echo "removed $PG_CONTAINER" || true ;;
  *)     echo "usage: $0 {up|key|login|down}" >&2; exit 2 ;;
esac
