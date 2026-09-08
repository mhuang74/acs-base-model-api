"""Beta event-logging telemetry: RequestTelemetry mapping + record_request
persistence + the X-Acs-Workload parser.

These lock the contract that per-request metadata (sampling shape, stream,
cold-boot, ttft, workload) lands on the api_requests row — without any body
content ever entering the logging/persistence path.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
os.environ.pop("HF_TOKEN", None)

from wrapper.auth import AuthedCaller
from wrapper.schemas import CompletionsRequest
from wrapper.services.request_log import RequestTelemetry, record_request


def _caller() -> AuthedCaller:
    return AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="testpfx0",
        user_email="test@example.local",
        monthly_token_budget=0,
        tokens_used_this_month=0,
    )


class _CaptureSession:
    """Records objects passed to .add() so we can inspect the ApiRequest row."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)


def test_telemetry_from_completions_request_maps_fields():
    parsed = CompletionsRequest.model_validate(
        {
            "prompt": "hi",
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 123,
            "logprobs": 5,
            "prompt_logprobs": 2,
            "echo": True,
        }
    )
    tel = RequestTelemetry.from_completions_request(
        parsed, stream=True, workload_type="batch"
    )
    assert tel.stream is True
    assert tel.workload_type == "batch"
    assert tel.req_max_tokens == 64
    assert tel.temperature == 0.7
    assert tel.top_p == 0.9
    assert tel.seed_set is True
    assert tel.logprobs_set is True
    assert tel.prompt_logprobs_set is True
    assert tel.echo_set is True


def test_telemetry_defaults_when_params_absent():
    parsed = CompletionsRequest.model_validate({"prompt": "hi"})
    tel = RequestTelemetry.from_completions_request(
        parsed, stream=False, workload_type=None
    )
    assert tel.stream is False
    assert tel.workload_type is None
    assert tel.req_max_tokens is None  # not sent
    assert tel.seed_set is False
    assert tel.logprobs_set is False
    assert tel.prompt_logprobs_set is False
    assert tel.echo_set is False


async def test_record_request_persists_telemetry_fields():
    session = _CaptureSession()
    tel = RequestTelemetry(
        stream=True,
        workload_type="interactive",
        req_max_tokens=128,
        temperature=0.5,
        top_p=0.8,
        logprobs_set=True,
        seed_set=True,
    )
    await record_request(
        session,
        request_id="req_test",
        caller=_caller(),
        ip=None,
        endpoint="/v1/completions",
        model="llama-405b",
        n_prompt=3,
        n_completion=7,
        status_code=200,
        latency_ms=42,
        upstream_latency_ms=30,
        error_kind=None,
        telemetry=tel,
        cold_boot=False,
        ttft_ms=11,
    )
    assert len(session.added) == 1
    row = session.added[0]
    assert row.stream is True
    assert row.workload_type == "interactive"
    assert row.req_max_tokens == 128
    assert row.temperature == 0.5
    assert row.top_p == 0.8
    assert row.logprobs_set is True
    assert row.prompt_logprobs_set is False
    assert row.seed_set is True
    assert row.upstream_latency_ms == 30
    assert row.ttft_ms == 11
    assert row.cold_boot is False


async def test_record_request_defaults_when_no_telemetry():
    """An error path that records without telemetry still produces a valid row
    with the new columns at their defaults (no AttributeError)."""
    session = _CaptureSession()
    await record_request(
        session,
        request_id="req_test",
        caller=_caller(),
        ip=None,
        endpoint="/v1/completions",
        model=None,
        n_prompt=None,
        n_completion=None,
        status_code=400,
        latency_ms=1,
        upstream_latency_ms=None,
        error_kind="bad_json",
    )
    row = session.added[0]
    assert row.stream is False
    assert row.cold_boot is False
    assert row.workload_type is None
    assert row.req_max_tokens is None
    assert row.logprobs_set is False


def test_parse_workload_validates():
    from wrapper.routes.api import _parse_workload

    assert _parse_workload("batch") == "batch"
    assert _parse_workload("Interactive") == "interactive"
    assert _parse_workload("  batch ") == "batch"
    assert _parse_workload("nonsense") is None
    assert _parse_workload("") is None
    assert _parse_workload(None) is None
