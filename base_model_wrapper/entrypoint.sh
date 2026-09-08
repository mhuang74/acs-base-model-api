#!/usr/bin/env bash
set -euo pipefail

# Silence transformers' advisory "PyTorch was not found..." notice, which it
# writes to STDERR on import — Railway's stream classifier would otherwise tag
# that benign line as level:error. We only need the slow tokenizer, so the
# missing-framework advisory is irrelevant.
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"

# Apply DB migrations on every boot — idempotent.
alembic upgrade head

# Bind to Railway's $PORT (defaults to 8000 locally).
#
# --log-config routes uvicorn's own lifecycle + access logging to STDOUT.
# uvicorn defaults those handlers to STDERR, where Railway's stream classifier
# tags everything as level:error — so benign boot/shutdown lines ("INFO:
# Application startup complete", "Uvicorn running", "Shutting down") polluted
# @level:error alerts. The config keeps levels intact; it only moves the
# stream. The app's own structlog lines already go to stdout (logging.py).
#
# Graceful drain on redeploy (zero-downtime fix, ACS finding 2026-06):
#   On a redeploy Railway sends SIGTERM to the old container, waits
#   `drainingSeconds` (set to 130 in railway.json), then SIGKILL. `exec` makes
#   uvicorn PID 1, so SIGTERM reaches it directly (no shell swallowing it).
#   `--timeout-graceful-shutdown` tells uvicorn: stop accepting new connections,
#   let in-flight requests finish for up to this many seconds, then cancel the
#   rest cleanly. We set it to 120s — just under Railway's 130s SIGKILL window,
#   so uvicorn always gets to close out (or cleanly 5xx) in-flight work itself
#   instead of being hard-killed mid-request (which is what produced the opaque
#   edge 502s). 120s covers the common request; truly long cold-boot calls
#   (upstream_timeout_s is 1200s) can still be cut — a 20-min drain per deploy
#   is not worth it. Tune via GRACEFUL_SHUTDOWN_S if needed (keep it < the
#   railway.json drainingSeconds value).
exec uvicorn wrapper.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --log-config "$(dirname "$0")/uvicorn_log_config.json" \
  --timeout-graceful-shutdown "${GRACEFUL_SHUTDOWN_S:-120}"
