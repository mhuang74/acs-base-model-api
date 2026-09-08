"""Shared pytest setup for the wrapper test suite.

``Settings`` (pydantic-settings) has four **required** fields — ``database_url``,
``modal_base_url``, ``vllm_api_key``, ``admin_token`` — so importing
``wrapper.main`` (which builds ``Settings()`` at module load) raises a
``ValidationError`` unless they're in the environment. Most test files set these
at module top, but files whose *unit* tests import ``wrapper.main`` outside a DB
fixture (e.g. ``test_workbench_generations.py``) fail when run first / alone —
the failure is purely a function of collection order (ACS-309).

Setting them here, at collection time, makes the whole suite order-independent
and lets the pure-unit tests run without a database (the ACS-213 precondition
for running pytest in CI). ``setdefault`` means any file that sets its own
values still wins — this only fills the gaps.
"""

from __future__ import annotations

import os

# Local-dev / CI-only stubs — never real secrets (matches the values the
# per-file fixtures already use, e.g. tests/test_chat_history.py).
os.environ.setdefault("DATABASE_URL", os.environ.get("TEST_DATABASE_URL", ""))
os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example/")
os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
os.environ.setdefault("SESSION_SECRET", "test-session-secret")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.setdefault("COOKIE_SECURE", "false")
os.environ.pop("HF_TOKEN", None)
