"""The privacy guarantee: log_request_meta must accept no body-content kwargs.

Inspecting the function signature is a hard guard — if someone adds a `prompt`
or `completion` parameter to log_request_meta, this test fails.
"""

import inspect

from wrapper.logging import log_request_meta


FORBIDDEN = {"prompt", "completion", "messages", "body", "text", "content", "logprobs"}


def test_log_signature_has_no_body_fields():
    sig = inspect.signature(log_request_meta)
    params = set(sig.parameters)
    leaks = params & FORBIDDEN
    assert not leaks, f"log_request_meta leaks body fields: {leaks}"


def test_log_signature_only_takes_metadata():
    expected = {
        "request_id", "key_id", "key_prefix", "user_email", "ip",
        "endpoint", "model", "n_prompt", "n_completion",
        "status", "latency_ms", "upstream_latency_ms", "error_kind",
        # Beta event-logging metadata (no body content — pure request shape).
        "stream", "cold_boot", "ttft_ms", "workload_type",
    }
    sig = inspect.signature(log_request_meta)
    assert set(sig.parameters) == expected
