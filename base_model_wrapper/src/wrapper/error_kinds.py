"""Canonical ``error_kind`` taxonomy — the single source of truth.

``error_kind`` is the string the proxy / route / workbench paths stamp on a
failed request so dashboards and the troubleshooting runbook can tell failure
modes apart. It used to be a free-form string scattered across ``proxy.py``,
``routes/api.py`` and ``workbench_generations.py`` with no enforcement, so a
typo (``vllm_oom`` → ``vlllm_oom``) would silently fragment a dashboard
grouping, break the runbook lookup, and slip past any alert keyed on the
correct spelling (ACS-94).

This module is now the **one place** the taxonomy is defined. Producers
reference ``ErrorKind`` members instead of bare strings (so a typo is an
``AttributeError`` at import, not a silent new value), the recorder validates
against it at write time (``is_known`` — a loud warning on drift), and
``tests/test_error_kinds.py`` pins the enum to the documented taxonomy table in
``docs/runbooks/error-troubleshooting.md`` so the two can't drift apart.

``StrEnum`` members ARE ``str`` (``ErrorKind.vllm_oom == "vllm_oom"`` and
``str``/``json``/SQLAlchemy serialise to the bare value), so members can be used
anywhere a plain string was before — no call-site type changes needed.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorKind(StrEnum):
    """Every value that may be written to ``api_requests.error_kind`` (and the
    closely-related kinds emitted before a request row exists).

    Keep in lockstep with the taxonomy table in
    ``docs/runbooks/error-troubleshooting.md`` — the test enforces parity.
    """

    # --- Expected / allocation-incident (not worth alarming on alone) ---
    cold_boot = "cold_boot"
    budget_exceeded = "budget_exceeded"
    activation_quota_exceeded = "activation_quota_exceeded"
    # Self-serve bulk harvest quota gates (ACS-245): key over its monthly
    # harvest-job budget / already at its running-job concurrency cap.
    harvest_quota_exceeded = "harvest_quota_exceeded"
    harvest_concurrency_exceeded = "harvest_concurrency_exceeded"
    # Global (cross-key) cap on concurrent BIG-MODEL (multi-GPU) harvests — a
    # transient capacity limit that stops 10 keys stampeding 80×H200 at once.
    harvest_capacity_exceeded = "harvest_capacity_exceeded"
    queue_full = "queue_full"
    rate_limited = "rate_limited"

    # --- Derived symptom (chase the root cause, not this) ---
    circuit_open = "circuit_open"

    # --- Real upstream failures ---
    upstream_unreachable = "upstream_unreachable"
    # The Modal harvest app (acs-<model>-harvest) isn't deployed / reachable,
    # so POST /v1/harvest can't spawn a job (ACS-245).
    harvest_unavailable = "harvest_unavailable"
    vllm_oom = "vllm_oom"
    vllm_engine_dead = "vllm_engine_dead"
    upstream_5xx = "upstream_5xx"
    # Generic upstream failure the workbench path couldn't classify further
    # (non-5xx-split fallback). Distinct from ``upstream_5xx`` which is the
    # /v1 streaming/non-streaming fallback.
    upstream_error = "upstream_error"

    # --- Client-ish (reflected from the upstream error body) ---
    vllm_context_length = "vllm_context_length"
    vllm_invalid_request = "vllm_invalid_request"
    upstream_4xx = "upstream_4xx"

    # --- Client (wrapper-side validation) ---
    invalid_request = "invalid_request"
    activations_unsupported = "activations_unsupported"
    activation_prompt_too_long = "activation_prompt_too_long"
    # POST /v1/harvest against a model with no Modal-backed harvest app
    # (registry entry has no modal_app_name) — ACS-245.
    harvest_unsupported = "harvest_unsupported"
    # GET /v1/harvest/<id> for a job that doesn't exist or belongs to another
    # key/user (deliberately indistinguishable) — ACS-245.
    harvest_job_not_found = "harvest_job_not_found"
    # DELETE /v1/harvest/<id> for a job already in a terminal state
    # (done/failed/cancelled) — there is nothing left to cancel (409) — ACS-344.
    harvest_not_cancellable = "harvest_not_cancellable"
    # Request body exceeds the accepted Content-Length (413; currently the
    # POST /v1/harvest corpus cap, ``harvest_max_body_bytes``) — ACS-245.
    request_too_large = "request_too_large"
    bad_json = "bad_json"
    context_length_exceeded = "context_length_exceeded"

    # --- Raised before a request row exists (logs/Sentry only, never in
    #     api_requests) but still part of the taxonomy users see ---
    chat_completions_unsupported = "chat_completions_unsupported"
    invalid_api_key = "invalid_api_key"
    # Key is live but its owner's account status != 'approved' — e.g. a
    # previously-approved user who was rejected (ACS-212).
    account_not_approved = "account_not_approved"
    # Owner's account is specifically 'suspended' — a reversible, often
    # time-boxed park rather than a decision about them (ACS-353). Split out of
    # ``account_not_approved`` so the client-facing message can say what's
    # actually true and how to get access back.
    account_suspended = "account_suspended"
    internal_error = "internal_error"


#: Fast membership set of the canonical string values.
KNOWN_ERROR_KINDS: frozenset[str] = frozenset(k.value for k in ErrorKind)


#: Upstream 5xx bodies that actually reflect a deterministic CLIENT mistake
#: (bad input the caller must fix), reflected from vLLM's error body — not a
#: server fault. The proxy skips retries for these (retrying can't fix a bad
#: request) and the /v1 handler returns 400 instead of 502 and does NOT trip
#: the circuit breaker — a user input error must not read as a server outage
#: or open the breaker for every caller on the model (ACS-322).
CLIENT_ISH_UPSTREAM_KINDS: frozenset[ErrorKind] = frozenset(
    {ErrorKind.vllm_context_length, ErrorKind.vllm_invalid_request}
)


def is_known(error_kind: str | None) -> bool:
    """True if ``error_kind`` is None (no error) or a canonical taxonomy value.

    Used by the request recorder as a write-time guard: an unknown value means a
    producer drifted from the enum and should be caught loudly rather than
    fragmenting a dashboard silently.
    """
    return error_kind is None or error_kind in KNOWN_ERROR_KINDS
