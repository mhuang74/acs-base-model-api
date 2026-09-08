"""Tests for the beta-grade /v1/completions request validation + capability
surfaces added in beta-api-hardening.

Covers four concerns:
  1. ``CompletionsRequest`` rejects unknown fields and out-of-range values
     with explicit errors (no silent drop to vLLM).
  2. ``_check_sequence_length`` returns a 400 when prompt + max_tokens would
     exceed the registry's ``max_model_len``, and passes through otherwise.
  3. ``X-Acs-Max-Tokens-Clamped`` is set when the budget clamp reduces
     ``max_tokens``.
  4. ``/v1/models`` includes the ``capabilities`` block per model.

These are mostly pure unit tests; no DB needed. Same env-stub pattern as
``test_resolve_model.py`` for cheap import.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

os.environ.setdefault("DATABASE_URL", "postgresql://stub")
os.environ.setdefault("MODAL_BASE_URL", "https://stub")
os.environ.setdefault("VLLM_API_KEY", "stub")
os.environ.setdefault("ADMIN_TOKEN", "stub")
os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")

from wrapper.auth import AuthedCaller
from wrapper.schemas import MAX_LOGPROBS, MAX_OUTPUT_WORK_TOKENS, CompletionsRequest
from wrapper.settings import ModelEntry

# --- Schema: extra="forbid" + range validation ------------------------------


def test_schema_accepts_minimal_body():
    req = CompletionsRequest.model_validate({"prompt": "hi"})
    assert req.prompt == "hi"
    # Defaults are not sent upstream when caller didn't set them.
    assert "temperature" not in req.to_upstream_body()
    assert req.to_upstream_body() == {"prompt": "hi"}


def test_add_special_tokens_forwarded_only_when_set():
    """ACS-319: the field is accepted (no longer dropped by extra='forbid'), and
    reaches the upstream body ONLY when the caller sets it — an omitted value
    sends nothing, so vLLM's own default (add_special_tokens=True) still applies."""
    # Omitted → absent from the upstream body (exclude_unset).
    assert "add_special_tokens" not in CompletionsRequest.model_validate(
        {"prompt": "hi"}
    ).to_upstream_body()
    # Explicit False → forwarded verbatim (the double-BOS fix for client-side templating).
    req = CompletionsRequest.model_validate({"prompt": "hi", "add_special_tokens": False})
    assert req.add_special_tokens is False
    assert req.to_upstream_body()["add_special_tokens"] is False
    # Explicit True → forwarded (honors the explicit set even though it matches the default).
    req_true = CompletionsRequest.model_validate({"prompt": "hi", "add_special_tokens": True})
    assert req_true.to_upstream_body()["add_special_tokens"] is True


def test_add_special_tokens_flows_to_activation_upstream():
    """It is a normal sampler-adjacent field, NOT an activation control, so it
    rides the allowlist onto the vLLM-Lens body too and is never smuggled into
    vllm_xargs (ACS-319) — a capture with a client-side BOS can suppress the double."""
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "add_special_tokens": False, "output_residual_stream": True}
    )
    body = req.to_activation_upstream_body()
    assert body["add_special_tokens"] is False
    assert "add_special_tokens" not in body["vllm_xargs"]


def test_schema_rejects_unknown_field():
    """``temprature`` (typo) must fail loudly, not silently drop."""
    with pytest.raises(ValidationError) as exc:
        CompletionsRequest.model_validate({"prompt": "hi", "temprature": 0.5})
    # The error path includes the offending key so the user can fix it.
    locs = [".".join(str(p) for p in e["loc"]) for e in exc.value.errors()]
    assert "temprature" in locs


def test_best_of_rejected_with_clear_user_message():
    """``best_of`` is unsupported by current vLLM; callers should use ``n``."""
    from wrapper.main import _validation_message

    with pytest.raises(ValidationError) as exc:
        CompletionsRequest.model_validate({"prompt": "hi", "best_of": 2})

    msg = _validation_message(exc.value)
    assert msg == "best_of is not supported by this API; use n to request multiple completions."


def test_schema_temperature_range():
    # ge=0.0, le=100.0 — cap raised from the old OpenAI-inherited 2.0 (ACS-178).
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "temperature": -0.1})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "temperature": 100.1})
    # Values above the old 2.0 cap are now accepted.
    assert CompletionsRequest.model_validate({"prompt": "hi", "temperature": 2.1}).temperature == 2.1
    assert CompletionsRequest.model_validate({"prompt": "hi", "temperature": 50}).temperature == 50


# --- Prompt shapes: text / batch / pre-tokenized token ids (ACS-168) ---------


class _LenCounter:
    """Stub token counter: one token per character (deterministic)."""

    def count(self, text: str) -> int:
        return len(text)


def test_schema_accepts_token_id_prompt():
    req = CompletionsRequest.model_validate({"prompt": [1, 2, 3]})
    assert req.prompt == [1, 2, 3]
    # Forwarded verbatim to vLLM (no re-tokenization, no str coercion).
    assert req.to_upstream_body()["prompt"] == [1, 2, 3]


def test_token_id_prompt_not_coerced_to_strings():
    """A list[int] must stay ints, not become list[str] via lax union coercion."""
    req = CompletionsRequest.model_validate({"prompt": [128, 9, 220]})
    assert all(type(x) is int for x in req.prompt)


def test_schema_accepts_token_id_batch():
    req = CompletionsRequest.model_validate({"prompt": [[1, 2], [3, 4]], "max_tokens": 5})
    assert req.prompt == [[1, 2], [3, 4]]
    assert req.to_upstream_body()["prompt"] == [[1, 2], [3, 4]]


def test_single_token_id_prompt_is_one_prompt():
    """A lone list[int] is ONE prompt, so n=1 needs no max_tokens (no fan-out)."""
    req = CompletionsRequest.model_validate({"prompt": [1, 2, 3, 4, 5]})
    assert req.max_tokens is None  # would be required if mis-counted as a batch of 5


def test_token_id_batch_requires_max_tokens():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": [[1, 2], [3, 4]]})


def test_schema_rejects_mixed_prompt_list():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": [1, "two", 3]})


def test_schema_rejects_empty_prompt_list():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": []})


def test_prompt_batch_size_helper():
    from wrapper.schemas import prompt_batch_size

    assert prompt_batch_size("hi") == 1
    assert prompt_batch_size(["a", "b"]) == 2
    assert prompt_batch_size([1, 2, 3]) == 1  # one tokenized prompt
    assert prompt_batch_size([[1, 2], [3, 4]]) == 2


def test_count_prompt_tokens_helper():
    from wrapper.services.completions import count_prompt_tokens

    c = _LenCounter()
    assert count_prompt_tokens("abc", c) == 3
    assert count_prompt_tokens(["ab", "c"], c) == 3  # batch of texts: summed
    assert count_prompt_tokens([1, 2, 3], c) == 3  # token ids: len, no tokenize
    assert count_prompt_tokens([[1, 2], [3, 4, 5]], c) == 5  # batch of token-id prompts


def test_schema_rejects_out_of_range_top_p():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "top_p": 0.0})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "top_p": 1.1})


def test_schema_top_k_disable_or_positive():
    # -1 is the documented "disabled" sentinel.
    CompletionsRequest.model_validate({"prompt": "hi", "top_k": -1})
    CompletionsRequest.model_validate({"prompt": "hi", "top_k": 10})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "top_k": 0})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "top_k": -2})


def test_schema_rejects_logprobs_over_cap():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "logprobs": MAX_LOGPROBS + 1})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "prompt_logprobs": MAX_LOGPROBS + 1})


def test_schema_prompt_logprobs_accepts_full_vocab_sentinel():
    """ACS-191: ``prompt_logprobs=-1`` (full vocab) is accepted; the prompt-length
    gate that keeps it from OOMing the server lives in the route, not the schema."""
    req = CompletionsRequest.model_validate({"prompt": "hi", "prompt_logprobs": -1})
    assert req.prompt_logprobs == -1


def test_schema_completion_logprobs_rejects_full_vocab_sentinel():
    """Full vocab is prompt-only: vLLM's OpenAI validator rejects ``logprobs=-1``
    on generated tokens, so the wrapper rejects it up front (still ge=0)."""
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "logprobs": -1})


def test_schema_prompt_logprobs_rejects_below_sentinel():
    """-1 is the only negative accepted; -2 and below are still invalid."""
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "prompt_logprobs": -2})


def test_logprobs_cap_is_100():
    """ACS-84 Tier 1: positive top-k cap raised 20 → 100. This value is pinned in
    lockstep with ``tests/test_logprobs_cap_lockstep.py`` (which enforces the
    wrapper-cap ≤ vLLM ``--max-logprobs`` invariant) and is advertised via
    ``/v1/models``. If you change it, update both.
    """
    assert MAX_LOGPROBS == 100


def test_schema_accepts_logprobs_at_new_cap():
    """The raised cap is honored at the boundary for both logprobs fields."""
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "logprobs": MAX_LOGPROBS, "prompt_logprobs": MAX_LOGPROBS}
    )
    assert req.logprobs == MAX_LOGPROBS
    assert req.prompt_logprobs == MAX_LOGPROBS


def test_schema_stop_caps_at_4():
    CompletionsRequest.model_validate({"prompt": "hi", "stop": ["a", "b", "c", "d"]})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "stop": ["a", "b", "c", "d", "e"]})


def test_schema_seed_must_be_non_negative():
    CompletionsRequest.model_validate({"prompt": "hi", "seed": 0})
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "seed": -1})


def test_schema_max_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "max_tokens": 0})


def test_schema_caps_n_times_max_tokens_output_work():
    CompletionsRequest.model_validate(
        {"prompt": "hi", "n": 16, "max_tokens": MAX_OUTPUT_WORK_TOKENS // 16}
    )
    with pytest.raises(ValidationError) as exc:
        CompletionsRequest.model_validate(
            {
                "prompt": "hi",
                "n": 16,
                "max_tokens": MAX_OUTPUT_WORK_TOKENS // 16 + 1,
            }
        )
    assert "prompt_count" in str(exc.value)
    assert str(MAX_OUTPUT_WORK_TOKENS) in str(exc.value)


def test_schema_caps_prompt_batch_total_output_work():
    CompletionsRequest.model_validate(
        {"prompt": ["a", "b"], "n": 2, "max_tokens": MAX_OUTPUT_WORK_TOKENS // 4}
    )
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate(
            {
                "prompt": ["a", "b"],
                "n": 2,
                "max_tokens": MAX_OUTPUT_WORK_TOKENS // 4 + 1,
            }
        )


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "hi", "n": 2},
        {"prompt": ["a", "b"]},
    ],
)
def test_schema_fanout_requires_explicit_max_tokens(body):
    with pytest.raises(ValidationError) as exc:
        CompletionsRequest.model_validate(body)
    assert "max_tokens is required" in str(exc.value)


def test_schema_to_upstream_body_omits_unset():
    """vLLM has its own defaults — we must not overwrite them by sending
    every default value the caller didn't actually set."""
    req = CompletionsRequest.model_validate({"prompt": "hi", "max_tokens": 32, "temperature": 0.5})
    body = req.to_upstream_body()
    assert body == {"prompt": "hi", "max_tokens": 32, "temperature": 0.5}
    assert "top_p" not in body
    assert "presence_penalty" not in body


# --- Validation message formatting ------------------------------------------


def test_validation_message_unknown_field_friendly():
    """The helper should produce a one-liner that includes the bad key name
    rather than Pydantic's verbose default."""
    from wrapper.main import _validation_message

    try:
        CompletionsRequest.model_validate({"prompt": "hi", "temprature": 0.5})
    except ValidationError as exc:
        msg = _validation_message(exc)
        assert "temprature" in msg
        assert "Unknown field" in msg


def test_validation_message_caps_at_5_errors():
    """A megaobject with 100 bad fields shouldn't dump 100 lines."""
    from wrapper.main import _validation_message

    bad = {f"junk_{i}": i for i in range(20)}
    bad["prompt"] = "hi"
    try:
        CompletionsRequest.model_validate(bad)
    except ValidationError as exc:
        msg = _validation_message(exc)
        assert "more" in msg.lower()


# --- Pre-flight sequence-length check ---------------------------------------


def _entry_with_max_len(max_len: int | None) -> ModelEntry:
    return ModelEntry(
        model_id="test-model",
        upstream_url="https://upstream.example",
        served_model_name="test/served",
        tokenizer_repo="gpt2",
        gpu_shape_label="1×L40S",
        status="live",
        max_model_len=max_len,
    )


def _caller():
    return AuthedCaller(
        key_id=uuid.uuid4(),
        key_prefix="seqcheck",
        user_email="seqcheck@example.local",
        monthly_token_budget=0,
        tokens_used_this_month=0,
    )


def test_omitted_max_tokens_gets_bounded_default(monkeypatch):
    from wrapper.routes import api as api_routes

    parsed = CompletionsRequest.model_validate({"prompt": "hello"})
    monkeypatch.setattr(
        api_routes,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda text: 1_000),
    )

    assert (
        api_routes._default_max_tokens(
            _entry_with_max_len(8_192),
            parsed,
            MagicMock(hf_token=None),
        )
        == 7_192
    )
    assert (
        api_routes._default_max_tokens(
            _entry_with_max_len(None),
            parsed,
            MagicMock(hf_token=None),
        )
        == MAX_OUTPUT_WORK_TOKENS
    )


async def test_sequence_length_no_max_skips(monkeypatch):
    """Legacy entries without max_model_len declared bypass the check."""
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    parsed = CompletionsRequest.model_validate({"prompt": "hello", "max_tokens": 100})
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(None),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is None


async def test_sequence_length_empty_prompt_skips(monkeypatch):
    """vLLM accepts empty prompts — wrapper should not 400 on them."""
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    monkeypatch.setattr(
        mainmod,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda t: 0),
    )
    parsed = CompletionsRequest.model_validate({"prompt": "", "max_tokens": 100})
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(8192),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is None


async def test_sequence_length_within_window_passes(monkeypatch):
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    monkeypatch.setattr(
        mainmod,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda t: 100),
    )
    parsed = CompletionsRequest.model_validate({"prompt": "long prompt", "max_tokens": 500})
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(8192),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is None


async def test_sequence_length_overflow_returns_400(monkeypatch):
    """Caller's prompt + max_tokens > model context → wrapper-side 400."""
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    monkeypatch.setattr(
        mainmod,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda t: 8000),
    )
    parsed = CompletionsRequest.model_validate({"prompt": "x" * 100, "max_tokens": 500})
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(8192),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is not None
    assert result.status_code == 400
    body = json.loads(bytes(result.body))
    assert body["error"]["code"] == "context_length_exceeded"
    msg = body["error"]["message"]
    assert "8000" in msg and "500" in msg and "8192" in msg


async def test_sequence_length_unset_max_tokens_only_checks_prompt(monkeypatch):
    """When max_tokens is None, vLLM defaults to 'fill remaining context';
    the check only needs to confirm the prompt itself fits (+1 token for any
    completion)."""
    from wrapper import main as mainmod

    monkeypatch.setattr(mainmod, "_record_request", AsyncMock())
    monkeypatch.setattr(
        mainmod,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda t: 8191),
    )
    parsed = CompletionsRequest.model_validate({"prompt": "x"})  # max_tokens unset
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(8192),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is None  # 8191 + 1 = 8192 fits

    monkeypatch.setattr(
        mainmod,
        "get_token_counter",
        lambda *a, **kw: MagicMock(count=lambda t: 8192),
    )
    result = await mainmod._check_sequence_length(
        session=MagicMock(),
        request_id="rq",
        caller=_caller(),
        ip=None,
        t0=0.0,
        entry=_entry_with_max_len(8192),
        parsed=parsed,
        settings=MagicMock(hf_token=None),
    )
    assert result is not None
    assert result.status_code == 400


# --- _apply_extras_headers -------------------------------------------------


def test_apply_extras_sets_clamp_header():
    from fastapi.responses import JSONResponse

    from wrapper.main import _apply_extras_headers

    resp = JSONResponse(content={"ok": True})
    _apply_extras_headers(
        resp,
        {"max_tokens_clamped": {"requested": 500, "applied": 98, "reason": "budget"}},
    )
    header = resp.headers["X-Acs-Max-Tokens-Clamped"]
    assert "requested=500" in header
    assert "applied=98" in header
    assert "reason=budget" in header


def test_apply_extras_no_clamp_no_header():
    from fastapi.responses import JSONResponse

    from wrapper.main import _apply_extras_headers

    resp = JSONResponse(content={"ok": True})
    _apply_extras_headers(resp, {})
    assert "X-Acs-Max-Tokens-Clamped" not in resp.headers


# --- /v1/models capability surfacing ---------------------------------------


def test_models_registry_parses_max_model_len():
    """The MODELS_REGISTRY_JSON ``max_model_len`` field is plumbed into ModelEntry."""
    from wrapper.settings import Settings

    settings = Settings(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        served_model_name="gpt2",
        models_registry_json=json.dumps(
            {
                "tiny": {
                    "upstream_url": "https://tiny.example",
                    "served_model_name": "tiny/model",
                    "tokenizer_repo": "tiny/model",
                    "gpu_shape_label": "1×L40S",
                    "max_model_len": 8192,
                },
                "no-max": {
                    "upstream_url": "https://nm.example",
                    "served_model_name": "nm/model",
                    "tokenizer_repo": "nm/model",
                    "gpu_shape_label": "1×L40S",
                },
            }
        ),
    )
    registry = settings.parsed_models_registry()
    assert registry["tiny"].max_model_len == 8192
    assert registry["no-max"].max_model_len is None


def test_models_registry_rejects_bad_max_model_len():
    """A non-integer or non-positive max_model_len fails at startup, loudly."""
    from wrapper.settings import Settings

    base = dict(
        database_url="postgresql://stub",
        modal_base_url="https://stub",
        vllm_api_key="stub",
        admin_token="stub",
        served_model_name="gpt2",
    )
    settings = Settings(
        **base,
        models_registry_json=json.dumps(
            {
                "bad": {
                    "upstream_url": "https://b.example",
                    "served_model_name": "b/m",
                    "tokenizer_repo": "b/m",
                    "max_model_len": "not-a-number",
                },
            }
        ),
    )
    with pytest.raises(RuntimeError, match="non-integer max_model_len"):
        settings.parsed_models_registry()

    settings_neg = Settings(
        **base,
        models_registry_json=json.dumps(
            {
                "bad": {
                    "upstream_url": "https://b.example",
                    "served_model_name": "b/m",
                    "tokenizer_repo": "b/m",
                    "max_model_len": 0,
                },
            }
        ),
    )
    with pytest.raises(RuntimeError, match="max_model_len"):
        settings_neg.parsed_models_registry()


# --- Activation harvesting + steering params (ACS-199) -----------------------
#
# Contract verified against the live 0.19.1 vLLM-Lens engine (2026-07-06):
# capture is all-or-nothing (``output_residual_stream: true`` → all layers; a
# list 400s), and steering vectors are the vLLM-Lens codec dicts, forwarded as a
# json.dumps'd string under ``vllm_xargs.apply_steering_vectors``.


def _codec_dict(shape=(1, 4096), data="AAAA"):
    """A minimal vLLM-Lens tensor codec dict (the wrapper never decodes it)."""
    return {
        "data": data,
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(shape),
        "compression": "zstd",
    }


def _steer(**over):
    sv = {"activations": _codec_dict(), "layer_indices": [16], "scale": 1.0}
    sv.update(over)
    return sv


def test_plain_completion_has_no_activation_params_and_excludes_them():
    req = CompletionsRequest.model_validate({"prompt": "hi", "temperature": 0.7})
    assert req.has_activation_params is False
    body = req.to_upstream_body()
    assert "output_residual_stream" not in body
    assert "apply_steering_vectors" not in body
    assert "vllm_xargs" not in body


def test_capture_sends_true_not_a_list():
    req = CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": True})
    assert req.has_activation_params is True
    # Normal body must NOT carry the activation control.
    assert "output_residual_stream" not in req.to_upstream_body()
    # Activation body sends the bare bool `true` (a list 400s at the engine).
    act = req.to_activation_upstream_body()
    assert act["vllm_xargs"] == {"output_residual_stream": True}
    assert act["prompt"] == "hi"


def test_capture_false_is_not_an_activation_request():
    req = CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": False})
    assert req.has_activation_params is False


def test_capture_layer_subset_sent_as_json_string(caplog):
    # ACS-266: a list of layer indices captures only those layers on the engine.
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "output_residual_stream": [16, 0, 8, 8]}
    )
    assert req.has_activation_params is True
    # Normalized: deduped + sorted.
    assert req.output_residual_stream == [0, 8, 16]
    act = req.to_activation_upstream_body()
    raw = act["vllm_xargs"]["output_residual_stream"]
    # Must be a JSON *string* (a bare list 400s at vLLM's vllm_xargs validation;
    # vLLM-Lens >= 1.2.0 json.loads it worker-side).
    assert isinstance(raw, str)
    assert json.loads(raw) == [0, 8, 16]
    assert "output_residual_stream" not in req.to_upstream_body()


def test_capture_empty_layer_list_rejected():
    with pytest.raises(ValidationError, match="non-empty list of layer indices"):
        CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": []})


def test_capture_layer_list_rejects_non_int_and_negative():
    with pytest.raises(ValidationError, match="must be integers"):
        CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": [0, "3"]})
    with pytest.raises(ValidationError, match="must be integers"):
        # bool is a subclass of int — must be rejected, not read as [1].
        CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": [True]})
    with pytest.raises(ValidationError, match=">= 0"):
        CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": [-1, 2]})


def test_capture_invalid_type_rejected():
    with pytest.raises(ValidationError, match="true, false, or a list"):
        CompletionsRequest.model_validate({"prompt": "hi", "output_residual_stream": "all"})


def test_steering_serialized_as_json_string_under_vllm_xargs():
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "apply_steering_vectors": [_steer(scale=2.0)]}
    )
    assert req.has_activation_params is True
    act = req.to_activation_upstream_body()
    raw = act["vllm_xargs"]["apply_steering_vectors"]
    # Must be a JSON *string* (the plugin json.loads it worker-side), not a list.
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed[0]["layer_indices"] == [16]
    assert parsed[0]["scale"] == 2.0
    assert parsed[0]["activations"]["dtype"] == "int16"


def test_capture_and_steer_combine():
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "output_residual_stream": True, "apply_steering_vectors": [_steer()]}
    )
    xargs = req.to_activation_upstream_body()["vllm_xargs"]
    assert xargs["output_residual_stream"] is True
    assert isinstance(xargs["apply_steering_vectors"], str)


def test_capture_subset_and_steer_combine():
    # ACS-266: a layer subset + steering both land in vllm_xargs as json strings.
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "output_residual_stream": [7, 16], "apply_steering_vectors": [_steer()]}
    )
    xargs = req.to_activation_upstream_body()["vllm_xargs"]
    assert isinstance(xargs["output_residual_stream"], str)
    assert json.loads(xargs["output_residual_stream"]) == [7, 16]
    assert isinstance(xargs["apply_steering_vectors"], str)


def test_steering_vector_requires_activations_and_layer_indices():
    with pytest.raises(ValidationError, match="activations"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "apply_steering_vectors": [{"layer_indices": [1], "scale": 1.0}]}
        )
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "apply_steering_vectors": [{"activations": _codec_dict()}]}
        )


def test_steering_vector_layer_indices_must_match_shape0():
    # shape[0]=2 but only one layer index → rejected.
    bad = _steer(activations=_codec_dict(shape=(2, 4096)), layer_indices=[16])
    with pytest.raises(ValidationError, match="layer_indices"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [bad]})


def test_steering_vector_rejects_oversized_data():
    from wrapper.schemas import MAX_STEERING_VECTOR_DATA_LEN

    huge = _steer(activations=_codec_dict(data="A" * (MAX_STEERING_VECTOR_DATA_LEN + 1)))
    with pytest.raises(ValidationError, match="cap"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [huge]})


def test_steering_vector_rejects_bad_shape_and_bool_scale():
    with pytest.raises(ValidationError, match="shape"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "apply_steering_vectors": [_steer(activations={"data": "AAAA"})]}
        )
    with pytest.raises(ValidationError, match="not a boolean"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "apply_steering_vectors": [_steer(scale=True)]}
        )


def test_steering_vector_rejects_unknown_field():
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "apply_steering_vectors": [_steer(bogus=1)]}
        )


def test_steering_rejects_too_many_vectors():
    from wrapper.schemas import MAX_STEERING_VECTORS

    vectors = [_steer() for _ in range(MAX_STEERING_VECTORS + 1)]
    with pytest.raises(ValidationError, match="at most"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": vectors})


def test_capture_with_stream_is_rejected():
    with pytest.raises(ValidationError, match="stream=true"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "output_residual_stream": True, "stream": True}
        )


def test_steering_with_stream_is_allowed():
    req = CompletionsRequest.model_validate(
        {"prompt": "hi", "apply_steering_vectors": [_steer()], "stream": True}
    )
    assert req.stream is True
    assert req.has_activation_params is True


# --- Steering single-sequence guards (ACS-251) --------------------------------
#
# The engine steers only the first sequence of a request: a batched prompt comes
# back half-steered and n>1 sampled fan-out is unsteered, both with HTTP 200
# (re-verified live 2026-07-18). The wrapper rejects both shapes.


def test_steering_with_batched_text_prompt_is_rejected():
    with pytest.raises(ValidationError, match="batched prompt"):
        CompletionsRequest.model_validate(
            {
                "prompt": ["one", "two"],
                "max_tokens": 8,
                "apply_steering_vectors": [_steer()],
            }
        )


def test_steering_with_batched_token_id_prompt_is_rejected():
    with pytest.raises(ValidationError, match="batched prompt"):
        CompletionsRequest.model_validate(
            {
                "prompt": [[1, 2], [3, 4]],
                "max_tokens": 8,
                "apply_steering_vectors": [_steer()],
            }
        )


def test_steering_with_single_token_id_prompt_is_allowed():
    # One pre-tokenized prompt (list[int]) is a single sequence, not a batch.
    req = CompletionsRequest.model_validate(
        {"prompt": [1, 2, 3], "apply_steering_vectors": [_steer()]}
    )
    assert req.has_activation_params is True


def test_steering_with_n_gt_1_is_rejected():
    with pytest.raises(ValidationError, match="n > 1"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "n": 3, "max_tokens": 8, "apply_steering_vectors": [_steer()]}
        )


def test_batch_and_n_stay_allowed_without_steering():
    req = CompletionsRequest.model_validate({"prompt": ["one", "two"], "n": 3, "max_tokens": 8})
    assert req.n == 3


# Capture (output_residual_stream) is single-sequence only, mirroring the
# steering guard: the response tensor is one sequence's (n_layers, n_tokens,
# hidden). A batched prompt or n>1 has no unambiguous single-sequence tensor and
# breaks the response-size accounting, so both are rejected pre-flight (ACS-255).


def test_capture_with_batched_text_prompt_is_rejected():
    with pytest.raises(ValidationError, match="batched prompt"):
        CompletionsRequest.model_validate(
            {"prompt": ["one", "two"], "max_tokens": 8, "output_residual_stream": True}
        )


def test_capture_with_batched_token_id_prompt_is_rejected():
    with pytest.raises(ValidationError, match="batched prompt"):
        CompletionsRequest.model_validate(
            {"prompt": [[1, 2], [3, 4]], "max_tokens": 8, "output_residual_stream": [3, 7]}
        )


def test_capture_with_n_gt_1_is_rejected():
    with pytest.raises(ValidationError, match="n > 1"):
        CompletionsRequest.model_validate(
            {"prompt": "hi", "n": 3, "max_tokens": 8, "output_residual_stream": True}
        )


def test_capture_with_single_token_id_prompt_is_allowed():
    # One pre-tokenized prompt (list[int]) is a single sequence, not a batch.
    req = CompletionsRequest.model_validate(
        {"prompt": [1, 2, 3], "max_tokens": 4, "output_residual_stream": True}
    )
    assert req.has_activation_params is True


def test_capture_false_with_batch_and_n_stays_allowed():
    # The guard only fires when capture is actually requested.
    req = CompletionsRequest.model_validate(
        {"prompt": ["one", "two"], "n": 3, "max_tokens": 8, "output_residual_stream": False}
    )
    assert req.n == 3


# --- Position-specific steering shape (ACS-251) -------------------------------


def test_steering_position_indices_with_2d_vector_is_rejected():
    # A 2-D vector broadcasts to all positions; the engine ignores
    # position_indices, so the combination is a silent no-op position filter.
    # Uses a valid (non-negative) index so this pins the 2-D rule alone — the
    # negative-index rule now fires first and has its own test.
    bad = _steer(position_indices=[0])
    with pytest.raises(ValidationError, match="3-D"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [bad]})


def test_steering_position_indices_with_matching_3d_vector_is_allowed():
    ok = _steer(activations=_codec_dict(shape=(1, 2, 4096)), position_indices=[0, 5])
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [ok]})
    assert req.apply_steering_vectors[0].position_indices == [0, 5]


def test_steering_position_indices_rejects_empty_list():
    # [] is neither "all positions" (that's null) nor a position selection.
    bad = _steer(activations=_codec_dict(shape=(1, 1, 4096)), position_indices=[])
    with pytest.raises(ValidationError):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [bad]})


def test_steering_position_indices_length_must_match_position_axis():
    bad = _steer(activations=_codec_dict(shape=(1, 3, 4096)), position_indices=[0])
    with pytest.raises(ValidationError, match="position axis"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [bad]})


def test_steering_3d_vector_without_position_indices_is_allowed():
    # null position_indices = broadcast; a 3-D tensor then targets the first
    # n_positions of the sequence — engine-side semantics, not the wrapper's
    # concern. Only the set-but-inconsistent combination is rejected.
    ok = _steer(activations=_codec_dict(shape=(1, 2, 4096)))
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [ok]})
    assert req.apply_steering_vectors[0].position_indices is None


# --- Span-based position selection -------------------------------------------
#
# ``position_spans`` broadcasts a 2-D vector across half-open [start, end) spans.
# The wrapper expands the spans into explicit position_indices AND tiles the 2-D
# vector into the 3-D tensor the engine already accepts, so correctness reduces
# to the existing validated explicit-indices path (verified transitively — no
# GPU needed).


def _real_2d_codec(n_layers=3, d_model=4, itemsize=2):
    """A REAL zstd codec with a distinct byte pattern per layer.

    Unlike ``_codec_dict`` (fake ``data``), this round-trips through zstd so the
    tiling logic can decode it — and the per-layer pattern lets a test detect a
    mis-tile (a slot holding the wrong layer's bytes).
    """
    import base64

    import zstandard

    row = d_model * itemsize
    raw = b"".join(bytes([i + 1]) * row for i in range(n_layers))
    packed = zstandard.ZstdCompressor(level=1).compress(raw)
    return {
        "data": base64.b64encode(packed).decode(),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": [n_layers, d_model],
        "compression": "zstd",
    }, row


def _decode_codec(codec):
    import base64

    import zstandard

    raw = base64.b64decode(codec["data"])
    if codec.get("compression") == "zstd":
        raw = zstandard.ZstdDecompressor().decompress(raw)
    return raw


def test_span_expands_to_position_indices():
    codec, _ = _real_2d_codec(n_layers=3)
    sv = _steer(activations=codec, layer_indices=[0, 1, 2],
                position_spans=[[10, 12], [20, 23]])
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})
    payload = req.apply_steering_vectors[0].engine_payload()
    assert payload["position_indices"] == [10, 11, 20, 21, 22]  # 2 + 3, in order


def test_span_tiles_2d_vector_to_correct_3d_shape_and_values():
    n_layers, d_model = 3, 4
    codec, row = _real_2d_codec(n_layers=n_layers, d_model=d_model)
    sv = _steer(activations=codec, layer_indices=[0, 1, 2],
                position_spans=[[0, 5]])  # 5 positions
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})
    payload = req.apply_steering_vectors[0].engine_payload()
    tiled = payload["activations"]
    assert tiled["shape"] == [n_layers, 5, d_model]
    assert tiled["compression"] == "zstd"
    # VALUES, not just shape: every position slot must equal its layer's row.
    out = _decode_codec(tiled)
    assert len(out) == n_layers * 5 * row
    for i in range(n_layers):
        layer_bytes = out[i * 5 * row:(i + 1) * 5 * row]
        for p in range(5):
            assert layer_bytes[p * row:(p + 1) * row] == bytes([i + 1]) * row


def test_span_engine_payload_omits_position_spans_and_matches_explicit_shape():
    # The transitive-correctness guarantee: the upstream dict must be the exact
    # 5-field explicit-indices shape — no ``position_spans`` key leaks to the
    # engine (which would reject an unknown vllm_xarg key).
    codec, _ = _real_2d_codec()
    sv = _steer(activations=codec, layer_indices=[0, 1, 2], position_spans=[[0, 3]])
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})
    payload = req.apply_steering_vectors[0].engine_payload()
    assert set(payload) == {
        "activations", "layer_indices", "scale", "norm_match", "position_indices"
    }
    # And it flows through to_activation_upstream_body as a json string.
    act = req.to_activation_upstream_body()
    forwarded = json.loads(act["vllm_xargs"]["apply_steering_vectors"])[0]
    assert "position_spans" not in forwarded
    assert forwarded["position_indices"] == [0, 1, 2]


def test_span_tiled_output_passes_steering_vector_validation():
    # The tiled result must itself satisfy every existing SteeringVector rule
    # (3-D shape, len(position_indices)==shape[1], layers match) — that is the
    # engine validation, standing in for a live GPU.
    from wrapper.schemas import SteeringVector

    codec, _ = _real_2d_codec(n_layers=2)
    sv = _steer(activations=codec, layer_indices=[0, 1], position_spans=[[4, 9]])
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})
    payload = req.apply_steering_vectors[0].engine_payload()
    reconstructed = SteeringVector.model_validate(payload)
    assert reconstructed.position_indices == [4, 5, 6, 7, 8]
    assert reconstructed.activations["shape"] == [2, 5, 4]


def test_span_rejects_negative_endpoint():
    codec, _ = _real_2d_codec()
    sv = _steer(activations=codec, layer_indices=[0, 1, 2], position_spans=[[-1, 5]])
    with pytest.raises(ValidationError, match="non-negative"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})


def test_span_rejects_end_le_start():
    codec, _ = _real_2d_codec()
    sv = _steer(activations=codec, layer_indices=[0, 1, 2], position_spans=[[5, 5]])
    with pytest.raises(ValidationError, match="end > start"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})
    sv2 = _steer(activations=codec, layer_indices=[0, 1, 2], position_spans=[[8, 3]])
    with pytest.raises(ValidationError, match="end > start"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv2]})


def test_span_and_position_indices_together_rejected():
    codec, _ = _real_2d_codec()
    sv = _steer(activations=codec, layer_indices=[0, 1, 2],
                position_spans=[[0, 3]], position_indices=[0])
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})


def test_span_requires_2d_vector():
    # A 3-D tensor + spans is ambiguous and out of scope.
    sv = _steer(activations=_codec_dict(shape=(1, 2, 4096)), position_spans=[[0, 3]])
    with pytest.raises(ValidationError, match="2-D"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})


def test_span_rejects_overlap():
    codec, _ = _real_2d_codec()
    sv = _steer(activations=codec, layer_indices=[0, 1, 2],
                position_spans=[[0, 5], [3, 8]])
    with pytest.raises(ValidationError, match="overlap"):
        CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [sv]})


def test_explicit_indices_path_unaffected_by_span_change():
    # Regression guard: the pre-existing explicit position_indices + 3-D path
    # still forwards verbatim, with no position_spans key.
    ok = _steer(activations=_codec_dict(shape=(1, 2, 4096)), position_indices=[0, 5])
    req = CompletionsRequest.model_validate({"prompt": "hi", "apply_steering_vectors": [ok]})
    payload = req.apply_steering_vectors[0].engine_payload()
    assert "position_spans" not in payload
    assert payload["position_indices"] == [0, 5]
    assert payload["activations"]["shape"] == [1, 2, 4096]


def test_registry_parses_activation_upstream_url():
    from wrapper.settings import Settings

    base = dict(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
    )
    settings = Settings(
        **base,
        models_registry_json=json.dumps(
            {
                "llama-8b": {
                    "upstream_url": "https://a.example/serve",
                    "served_model_name": "meta-llama/Llama-3.1-8B",
                    "tokenizer_repo": "meta-llama/Llama-3.1-8B",
                    "activation_upstream_url": "https://a.example-activation/serve",
                },
                "plain": {
                    "upstream_url": "https://b.example/serve",
                    "served_model_name": "b/m",
                    "tokenizer_repo": "b/m",
                },
            }
        ),
    )
    registry = settings.parsed_models_registry()
    assert registry["llama-8b"].activation_upstream_url == "https://a.example-activation/serve"
    # Absent → None (no activation support advertised for this model).
    assert registry["plain"].activation_upstream_url is None


def test_registry_rejects_non_string_activation_upstream_url():
    from wrapper.settings import Settings

    base = dict(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
    )
    settings = Settings(
        **base,
        models_registry_json=json.dumps(
            {
                "bad": {
                    "upstream_url": "https://b.example",
                    "served_model_name": "b/m",
                    "tokenizer_repo": "b/m",
                    "activation_upstream_url": 123,
                },
            }
        ),
    )
    with pytest.raises(RuntimeError, match="activation_upstream_url"):
        settings.parsed_models_registry()


# --- ACS-317: activation boundary bugs (usersonas feedback) ------------------


def test_token_counter_includes_bos_like_the_engine():
    """vLLM tokenizes text prompts with add_special_tokens=True. Counting
    without them undercounted every text prompt by one, so a prompt sitting
    exactly on a limit passed our pre-flight and failed upstream instead."""
    from wrapper.tokenizer import TokenCounter

    counter = TokenCounter.__new__(TokenCounter)  # no HF download
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    counter._tok = tok
    assert counter.count("hi") == 3
    assert tok.encode.call_args.kwargs["add_special_tokens"] is True

    # …but a COMPLETION gets none: the engine never prepends BOS to its own
    # output, and this value is billed (workbench fallback path).
    assert counter.count_completion("hi") == 3
    assert tok.encode.call_args.kwargs["add_special_tokens"] is False


def _activation_entry(*, cap: int, n_layers: int | None, max_len: int | None = None) -> ModelEntry:
    return ModelEntry(
        model_id="act-model",
        upstream_url="https://upstream.example",
        served_model_name="test/served",
        tokenizer_repo="gpt2",
        gpu_shape_label="1×L40S",
        status="live",
        max_model_len=max_len,
        activation_upstream_url="https://activation.example",
        activation_max_prompt_tokens=cap,
        n_layers=n_layers,
    )


@pytest.mark.parametrize(
    "requested,n_layers,prompt_tokens,expect_error",
    [
        (True, 32, 600, True),      # all layers, over the flat cap
        (True, 32, 500, False),     # all layers, under it
        ([14], 32, 600, False),     # 1-of-32 → cap scales to 16384
        ([14], 32, 20_000, True),   # …but not without limit
        ([1, 2], 32, 8_000, False), # 2-of-32 → 8192
        ([14], None, 600, True),    # unknown layer count → flat cap stands
        (list(range(32)), 32, 600, True),   # explicit all-layers list == True
        (list(range(32)), 32, 500, False),
    ],
)
def test_capture_cap_scales_with_requested_layers(
    monkeypatch, requested, n_layers, prompt_tokens, expect_error
):
    """The cap bounds RESPONSE SIZE, which scales with captured layers — so a
    subset request affords proportionally more prompt (ACS-317 flag E)."""
    import asyncio

    from wrapper.services import completions as completions_svc

    parsed = CompletionsRequest.model_validate(
        {"prompt": "x", "max_tokens": 1, "output_residual_stream": requested}
    )
    entry = _activation_entry(cap=512, n_layers=n_layers)
    result = asyncio.run(
        completions_svc.check_activation_prompt_length(
            AsyncMock(),
            "r",
            _caller(),
            "1.2.3.4",
            0.0,
            entry,
            parsed,
            MagicMock(hf_token=None),
            error_response=AsyncMock(return_value="ERR"),
            token_counter_factory=lambda *a, **kw: MagicMock(
                count=lambda text: prompt_tokens
            ),
        )
    )
    assert (result is not None) is expect_error


def test_capture_cap_never_exceeds_the_context_window(monkeypatch):
    """Scaling must not advertise a cap the model couldn't accept anyway."""
    import asyncio

    from wrapper.services import completions as completions_svc

    parsed = CompletionsRequest.model_validate(
        {"prompt": "x", "max_tokens": 1, "output_residual_stream": [14]}
    )
    entry = _activation_entry(cap=512, n_layers=32, max_len=4096)
    result = asyncio.run(
        completions_svc.check_activation_prompt_length(
            AsyncMock(),
            "r",
            _caller(),
            "1.2.3.4",
            0.0,
            entry,
            parsed,
            MagicMock(hf_token=None),
            error_response=AsyncMock(return_value="ERR"),
            token_counter_factory=lambda *a, **kw: MagicMock(count=lambda text: 5_000),
        )
    )
    assert result is not None  # 5000 > min(16384, 4096)


def _run_activation_gate(parsed, entry, *, prompt_tokens, effective_max_tokens=None):
    """Drive check_activation_prompt_length with a fake token counter."""
    import asyncio

    from wrapper.services import completions as completions_svc

    return asyncio.run(
        completions_svc.check_activation_prompt_length(
            AsyncMock(),
            "r",
            _caller(),
            "1.2.3.4",
            0.0,
            entry,
            parsed,
            MagicMock(hf_token=None),
            error_response=AsyncMock(return_value="ERR"),
            token_counter_factory=lambda *a, **kw: MagicMock(count=lambda text: prompt_tokens),
            effective_max_tokens=effective_max_tokens,
        )
    )


def test_capture_cap_counts_generated_tokens():
    """A short prompt with a large max_tokens must NOT slip past the cap: the
    capture covers prompt + generated − 1 positions, so a 10-token prompt with
    max_tokens=10_000 buffers ~10k positions and is rejected (ACS-252/ACS-255)."""
    parsed = CompletionsRequest.model_validate(
        {"prompt": "x", "max_tokens": 10_000, "output_residual_stream": True}
    )
    entry = _activation_entry(cap=512, n_layers=32)
    result = _run_activation_gate(
        parsed, entry, prompt_tokens=10, effective_max_tokens=10_000
    )
    assert result is not None  # 10 + 10000 - 1 = 10009 > 512


def test_capture_short_prompt_small_max_tokens_passes():
    """The common probing shape — short prompt, few generated tokens — stays
    under the cap and is accepted."""
    parsed = CompletionsRequest.model_validate(
        {"prompt": "x", "max_tokens": 8, "output_residual_stream": True}
    )
    entry = _activation_entry(cap=512, n_layers=32)
    result = _run_activation_gate(parsed, entry, prompt_tokens=10, effective_max_tokens=8)
    assert result is None  # 10 + 8 - 1 = 17 <= 512


def test_capture_cap_uses_resolved_default_when_max_tokens_omitted():
    """An omitted max_tokens resolves to the per-model default at the call site;
    the caller passes that in, so an unbounded-generation capture is rejected
    rather than slipping past on prompt length alone (ACS-255)."""
    parsed = CompletionsRequest.model_validate({"prompt": "x", "output_residual_stream": True})
    entry = _activation_entry(cap=512, n_layers=32)
    # Call site resolved None → 32_000 (MAX_OUTPUT_WORK_TOKENS default).
    result = _run_activation_gate(
        parsed, entry, prompt_tokens=10, effective_max_tokens=32_000
    )
    assert result is not None  # 10 + 32000 - 1 = 32009 > 512


def test_capture_max_tokens_1_is_prompt_only():
    """max_tokens=1 generates one position, so total == prompt tokens — the
    accounting is backward-compatible with the old prompt-only cap."""
    parsed = CompletionsRequest.model_validate(
        {"prompt": "x", "max_tokens": 1, "output_residual_stream": True}
    )
    entry = _activation_entry(cap=512, n_layers=32)
    under = _run_activation_gate(parsed, entry, prompt_tokens=512, effective_max_tokens=1)
    assert under is None  # 512 + 1 - 1 = 512 == cap
    over = _run_activation_gate(parsed, entry, prompt_tokens=513, effective_max_tokens=1)
    assert over is not None  # 513 > 512


def test_steering_rejects_negative_position_indices():
    """The docs advertised [-1]; engine semantics are unverified and silently
    steering the wrong token is the worst outcome (ACS-317 flag F)."""
    from wrapper.schemas import SteeringVector

    body = {
        "activations": {"shape": [1, 1, 4], "dtype": "float32", "data": "AAAAAA=="},
        "layer_indices": [14],
        "scale": 1.0,
        "position_indices": [-1],
    }
    with pytest.raises(ValidationError) as exc:
        SteeringVector.model_validate(body)
    assert "non-negative" in str(exc.value)


def _registry_settings(entry_extra: dict):
    """Settings built from a one-entry MODELS_REGISTRY_JSON."""
    from wrapper.settings import Settings

    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        modal_base_url="https://legacy.example.invalid",
        vllm_api_key="test-vllm-key",
        admin_token="test-admin-token",
        models_registry_json=json.dumps(
            {
                "m": {
                    "upstream_url": "https://b.example",
                    "served_model_name": "b/m",
                    "tokenizer_repo": "b/m",
                    **entry_extra,
                }
            }
        ),
    )


def test_registry_parses_n_layers():
    """The cap scaling is inert until this field is set in prod, so the parse
    is the one path that must not break (ACS-317 review finding)."""
    for raw, expected in ((32, 32), ("32", 32), (None, None)):
        entry = _registry_settings({"n_layers": raw} if raw is not None else {})
        assert entry.parsed_models_registry()["m"].n_layers == expected


@pytest.mark.parametrize("bad", ["abc", 0, -5, True])
def test_registry_rejects_bad_n_layers(bad):
    settings = _registry_settings({"n_layers": bad})
    with pytest.raises(RuntimeError, match="n_layers"):
        settings.parsed_models_registry()


# --- ACS-320: harvest projection request validation -------------------------


def _projection_body(**over):
    import base64

    body = {
        "model": "llama-8b",
        "prompts": ["hello"],
        "project_onto": {
            # 2 x 4 float16 = 16 bytes. Keep this consistent with `shape`:
            # the validator now checks the byte length, which is the point.
            "data": base64.b64encode(b"\x01" * (2 * 4 * 2)).decode(),
            "dtype": "float16",
            "shape": [2, 4],
            "compression": "none",
        },
    }
    body["project_onto"].update(over.pop("project_onto", {}))
    body.update(over)
    return body


def test_harvest_accepts_projection_directions():
    from wrapper.schemas import HarvestRequest

    req = HarvestRequest.model_validate(_projection_body())
    assert req.project_onto["shape"] == [2, 4]


def test_harvest_projection_defaults_to_none():
    """Raw-activation harvests stay the default — projection is opt-in."""
    from wrapper.schemas import HarvestRequest

    assert HarvestRequest.model_validate({"model": "m", "prompts": ["p"]}).project_onto is None


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"shape": [4]}, "2-D"),
        ({"shape": [0, 4]}, "1..64"),
        ({"shape": [65, 4]}, "1..64"),
        ({"shape": [2, 0]}, "positive"),
        ({"dtype": "int8"}, "float32, float16 or bfloat16"),
        ({"compression": "gzip"}, "compression"),
        ({"data": 123}, "base64 string"),
    ],
)
def test_harvest_projection_rejects_bad_payloads(over, msg):
    """Caught at the boundary rather than after a GPU job has started."""
    from wrapper.schemas import HarvestRequest

    with pytest.raises(ValidationError) as exc:
        HarvestRequest.model_validate(_projection_body(project_onto=over))
    assert msg in str(exc.value)
