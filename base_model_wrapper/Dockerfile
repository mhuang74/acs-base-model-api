FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Reproducible installs: pin uv and resolve strictly from the committed
# uv.lock (`--frozen`), instead of `pip install .` which ignored the lock and
# re-resolved dependency ranges from PyPI at build time. That float is what
# silently shipped modal 1.5.0 and broke the admin model-state lookups
# (ACS-123); honoring the lock is the durable fix (ACS-124).
RUN pip install --no-cache-dir uv==0.10.11
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1

# Dependency layer: install everything in the lock except the project itself,
# so this (expensive) layer is cached unless pyproject.toml / uv.lock change.
COPY base_model_wrapper/pyproject.toml base_model_wrapper/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Project source. setuptools package discovery uses where=["src", "."], so the
# wrapper package plus the cli/ and acs_model_registry/ root packages must all
# be present before the project is installed below.
COPY base_model_wrapper/src ./src
COPY base_model_wrapper/cli ./cli
COPY acs_model_registry ./acs_model_registry
COPY base_model_wrapper/alembic ./alembic
COPY base_model_wrapper/alembic.ini ./
COPY base_model_wrapper/entrypoint.sh ./
COPY base_model_wrapper/uvicorn_log_config.json ./

# Needed by the admin Deploy button — modal_ops.deploy_app subprocesses
# `modal deploy modal_app.py` with cwd=/app/modal_source/.
COPY modal_app.py ./modal_source/modal_app.py
COPY acs_model_registry/__init__.py ./modal_source/acs_model_registry/__init__.py
COPY serving/__init__.py ./modal_source/serving/__init__.py
COPY serving/modal_config.py ./modal_source/serving/modal_config.py
COPY serving/modal_lifecycle.py ./modal_source/serving/modal_lifecycle.py
COPY serving/modal_resources.py ./modal_source/serving/modal_resources.py
COPY serving/staging.py ./modal_source/serving/staging.py
COPY serving/vllm_runtime.py ./modal_source/serving/vllm_runtime.py

# Install the project itself into the frozen environment.
RUN uv sync --frozen --no-dev

# Put the uv-managed venv first on PATH so entrypoint.sh's bare `alembic` /
# `uvicorn` invocations resolve to the locked versions without `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

RUN chmod +x ./entrypoint.sh

# Railway injects PORT; bind there.
ENV PORT=8000
EXPOSE 8000

CMD ["./entrypoint.sh"]
