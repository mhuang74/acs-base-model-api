"""Pydantic types for admin endpoints + the loud-fail error response + the
beta-grade ``/v1/completions`` request validator.

Historically the /v1/completions request body was passed opaquely to vLLM so we
didn't have to track the OpenAI surface as it evolves. For the beta we tighten
this: ``CompletionsRequest`` defines an allowlisted, range-validated set of
sampling parameters and rejects anything unknown with an explicit 400 (rather
than silently forwarding to vLLM, which often accepts-and-drops). The trade-off
is that adding new params now needs a wrapper change, but the alternative —
typos like ``temprature=0.5`` silently using the default — is unacceptable for a
research API where reproducibility hinges on sampler state.

Responses are still passed through unchanged so logprobs/token_ids/prompt_logprobs
surface verbatim from vLLM.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import json
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class CreateUserBody(BaseModel):
    email: EmailStr
    name: str | None = None
    org: str | None = None
    notes: str | None = None


class CreateKeyBody(BaseModel):
    user_email: EmailStr
    name: str | None = Field(None, description="Free-form label e.g. 'laptop', 'rollout-batch'")
    monthly_token_budget: int = Field(0, ge=0, description="0 = unlimited")


class CreateKeyResponse(BaseModel):
    key: str = Field(..., description="Plaintext key. Shown ONCE. Never recoverable.")
    key_id: uuid.UUID
    key_prefix: str
    user_email: EmailStr
    monthly_token_budget: int


class KeySummary(BaseModel):
    id: uuid.UUID
    key_prefix: str
    user_email: EmailStr
    name: str | None
    monthly_token_budget: int
    tokens_used_this_month: int
    created_at: dt.datetime
    last_used_at: dt.datetime | None
    revoked_at: dt.datetime | None


class OpenAIErrorBody(BaseModel):
    """Mirror of OpenAI's error envelope so SDK clients display it sensibly."""

    error: dict


# --- /v1/completions request schema -----------------------------------------
#
# Allowlisted sampling params. Range bounds match vLLM's accepted ranges for
# vllm==0.19.1 so a value that passes wrapper validation also passes vLLM's
# own checks (the wrapper just gives a clearer error path). NB: for logprobs
# this parity holds only once the serve image is deployed with
# ``--max-logprobs 100`` (see MAX_LOGPROBS below) — vLLM's own default is 20.

# Per-position cap on the ``logprobs`` / ``prompt_logprobs`` *count* the wrapper
# accepts. Surfaced in /v1/models as the per-model ``max_logprobs`` capability.
# This is the ceiling for a POSITIVE (top-k) request; full-vocab is handled
# separately by the ``-1`` sentinel below.
#
# Relationship to the upstream ``--max-logprobs`` flag (``modal_app.py``:
# ``MAX_LOGPROBS_CAP``, default ``-1`` = uncapped): the wrapper is the guardrail,
# not the serve flag. vLLM is deployed uncapped so it accepts whatever the
# wrapper forwards (including ``prompt_logprobs=-1``); every real bound —
# the positive top-k ceiling here, the full-vocab prompt-length gate, and the
# output-work bound — is enforced at the wrapper boundary. The invariant is
# therefore "wrapper cap ≤ server cap", satisfied trivially by the uncapped
# server (see ``tests/test_logprobs_cap_lockstep.py``). See ACS-84 / ACS-191.
MAX_LOGPROBS = 100
# Full-vocab logprobs (ACS-191). vLLM 0.23 (V1 engine) returns the model's whole
# next-token distribution — vocab_size (~128k for Llama) floats PER position —
# when ``prompt_logprobs=-1``. Completion ``logprobs`` cannot be ``-1`` (vLLM's
# OpenAI validator rejects it), so full-vocab is a PROMPT-only capability.
#
# The payload is enormous. Since ACS-198 the wrapper is no longer the binding
# constraint: full-vocab responses take the streaming pass-through
# (``proxy.passthrough_post`` → ``routes.api._serve_fullvocab_passthrough``),
# which forwards bytes without ``r.json()``-ing the body, so wrapper RAM stays
# O(chunk) regardless of prompt length. What still scales with prompt length:
#   • the UPSTREAM vLLM process materializes the whole response to serialize
#     it — a parsed Python dict of ~175 B × 128k ≈ 22 MB PER POSITION (the
#     Modal containers set no explicit memory limit, so headroom is
#     host-dependent and unverified);
#   • ~4 MB of JSON on the wire per prompt position upstream → wrapper
#     (~10-15× smaller wrapper → client when the client accepts gzip), all of
#     which must fit inside the 14-minute public request window.
#
# Hence the gate below is now sized against upstream serialization + transfer,
# not wrapper RAM: 1024 tokens ≈ ~22 GB peak inside the model server and ~4 GB
# on the upstream wire — conservative for the 8×H200 boxes, plausible for the
# single-GPU ones, and 64× the old cap. Raisable further (toward
# ``max_model_len``) once real upstream memory behavior is observed; the
# wrapper-side sequence-length check still bounds everything at max_model_len.
FULL_VOCAB_SENTINEL = -1
FULL_VOCAB_MAX_PROMPT_TOKENS = 1024
# A full-vocab request that ALSO carries activation params routes to the
# activation engine over the BUFFERED path (activation responses are
# materialized whole in wrapper RAM, see the activation block below), so it
# must keep the old wrapper-RAM-sized bound: 16 tokens ≈ ~5.6 GB worst-case
# peak on the 24 GB Railway box (22 MB/position × ~2 × 8-way concurrency).
FULL_VOCAB_MAX_PROMPT_TOKENS_BUFFERED = 16
# Bound worst-case generated output across prompt batches and ``n`` fan-out.
# 32k keeps ordinary eval/sweep requests viable while rejecting the
# n=16 × max_tokens=10k abuse shape measured in ACS-93.
MAX_OUTPUT_WORK_TOKENS = 32_000

# --- Activation harvesting + steering (ACS-199) ------------------------------
#
# Capture is all layers OR a subset. The vLLM-Lens engine takes
# ``vllm_xargs.output_residual_stream: true`` (capture EVERY decoder layer) or a
# ``json.dumps``'d **string** list of layer indices (capture only those — ACS-266,
# vLLM-Lens >= 1.2.0, which json.loads the list worker-side). A bare list 400s at
# vLLM's ``vllm_xargs`` validation (it only accepts scalars/strings), so a subset
# is sent as a string — same convention as ``apply_steering_vectors``. The wrapper
# accepts ``output_residual_stream`` as a bool or a list of ints; the subset is
# applied on the engine, so only those layers cross the wire (before ACS-266 the
# wrapper captured all layers and clients filtered client-side).
#
# The response carries a base64(zstd(bf16)) tensor of shape
# (n_layers, n_tokens, d_model) under a top-level ``activations`` key, where
# ``n_tokens = prompt + generated − 1`` (capture covers the generation
# trajectory too, minus the final sampled token that never gets a forward pass;
# ACS-252). Passed through verbatim. Since ACS-250 the capture response is
# STREAMED (no wrapper-RAM materialization), so the constraint is the RESPONSE
# SIZE the client downloads: payload = captured_layers × n_tokens × d_model × 2 B.
# All three are per-request: prompt length, generated length (``max_tokens``),
# and — since ACS-317 — the requested layer subset. Hence a PER-MODEL token cap
# (``ModelEntry.activation_max_prompt_tokens``), scaled up when a request asks
# for fewer layers (needs ``ModelEntry.n_layers``): 405B (126 × 16384 ≈
# 4 MB/token) gets a tight all-layer cap, 8B (32 × 4096 ≈ 256 KB/token) a
# generous one. Long-context / bulk capture is the offline path (ACS-198).

# Cap concurrent steering vectors per request (bounds request-body size + engine
# work). Each vector is a small base64(zstd(bf16)) codec dict built client-side.
MAX_STEERING_VECTORS = 8
# Projection directions per harvest job (ACS-320). A probe set is a handful;
# this bounds the request body and the per-token output width.
MAX_PROJECTION_DIRECTIONS = 64
# Fallback per-model activation prompt-token cap when a model's registry entry
# doesn't set ``activation_max_prompt_tokens``. Conservative (sized so even a
# big model stays well under the Railway RAM budget); tune per model in the
# registry. 0 / None on the entry means "inherit this default".
DEFAULT_ACTIVATION_MAX_PROMPT_TOKENS = 64
# Upper bound on one steering vector's base64 ``activations.data`` length. A
# (n_layers, d_model) bf16 direction is small; 512 KB is generous headroom while
# rejecting absurd payloads. For the plain (non-span) path the wrapper forwards
# the codec dict verbatim — it never decodes the tensor. The ONLY exception is
# ``position_spans``, which tiles a 2-D vector across the span: that
# path touches the container format (zstd frame + C-order layer-major bytes)
# but NOT the tensor semantics (no dtype/element interpretation). See
# ``_tile_2d_codec_to_3d``.
MAX_STEERING_VECTOR_DATA_LEN = 512 * 1024


class SteeringVector(BaseModel):
    """One steering directive, forwarded verbatim to vLLM-Lens.

    Matches the vLLM-Lens ``SteeringVector`` wire schema (v1.1.0, verified from
    source + a live engine 2026-07-06): the direction ``activations`` is itself a
    base64(zstd(bf16-as-int16)) **codec dict** — the client builds it (see the
    runbook's encode recipe). The wrapper does NOT decode the tensor on the plain
    path; it only validates the envelope + bounds size, then ``json.dumps`` the
    list under ``vllm_xargs.apply_steering_vectors``. The one exception is
    ``position_spans``: the wrapper tiles a 2-D vector across the span
    into the 3-D tensor the engine already accepts, touching the container format
    (zstd frame + byte layout) only. Steering mechanics (S1–S6) are
    validated upstream (ACS-156).
    """

    model_config = ConfigDict(extra="forbid")

    activations: dict[str, Any] = Field(
        ...,
        description=(
            "Steering direction as a vLLM-Lens tensor codec dict: {data (base64), "
            "dtype, original_dtype, shape, compression}. shape (n_layers, d_model) "
            "broadcasts to all positions; (n_layers, n_positions, d_model) is "
            "position-specific."
        ),
    )
    layer_indices: list[int] = Field(
        ...,
        min_length=1,
        description="Decoder layer indices; len(layer_indices) must equal activations.shape[0].",
    )
    scale: float = Field(default=1.0, description="Scalar multiplier applied before addition.")
    norm_match: bool = Field(
        default=False,
        description="Rescale the steered hidden state to preserve the original per-token L2 norm.",
    )
    position_indices: list[int] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Absolute token positions to steer (0-indexed, non-negative); "
            "null = broadcast / all positions."
        ),
    )
    position_spans: list[tuple[int, int]] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Half-open [start, end) token-position spans to steer, as a "
            "convenience over building an explicit position_indices list + 3-D "
            "tensor. Requires a 2-D (n_layers, d_model) activations "
            "vector, which the wrapper broadcasts (tiles) across every position "
            "in the spans. Endpoints are 0-indexed and non-negative; end must be "
            "> start. Mutually exclusive with position_indices. Expanded "
            "server-side into position_indices + a 3-D tensor, so the engine "
            "sees the already-validated position-specific path."
        ),
    )

    @field_validator("scale", mode="before")
    @classmethod
    def _reject_bool_scale(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("scale must be a number, not a boolean")
        return v

    @field_validator("activations")
    @classmethod
    def _validate_codec_envelope(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Validate the envelope only — the wrapper forwards the tensor verbatim
        # (no decode). Must be the vLLM-Lens codec dict, with a sane data size and
        # a shape whose leading dim matches the vector's layer count.
        missing = {"data", "shape"} - v.keys()
        if missing:
            raise ValueError(f"activations must be a tensor codec dict; missing keys {missing}")
        data = v.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError("activations.data must be a non-empty base64 string")
        if len(data) > MAX_STEERING_VECTOR_DATA_LEN:
            raise ValueError(
                f"activations.data exceeds the {MAX_STEERING_VECTOR_DATA_LEN}-byte cap"
            )
        shape = v.get("shape")
        if not isinstance(shape, list) or not shape or not all(isinstance(x, int) for x in shape):
            raise ValueError("activations.shape must be a non-empty list of ints")
        return v

    @model_validator(mode="after")
    def _layers_match_shape(self) -> SteeringVector:
        n = self.activations["shape"][0]
        if len(self.layer_indices) != n:
            raise ValueError(
                f"len(layer_indices)={len(self.layer_indices)} must equal "
                f"activations.shape[0]={n}"
            )
        return self

    @model_validator(mode="after")
    def _positions_require_3d(self) -> SteeringVector:
        # Position-specific steering needs a 3-D (n_layers, n_positions, hidden)
        # tensor. The engine accepts a 2-D broadcast vector combined with
        # ``position_indices`` but ignores the position selection (verified in
        # beta feedback, 2026-07-18) — the result looks like a successful
        # position experiment while steering every position. Reject the
        # combination, and a position-axis/indices length mismatch, at the
        # boundary instead.
        if self.position_indices is None:
            return self
        # Defer to ``_validate_position_spans`` when spans are ALSO set: the
        # mutual-exclusivity error is clearer than a 3-D-shape error here (the
        # spans path deliberately uses a 2-D vector).
        if self.position_spans is not None:
            return self
        # Checked before the shape rules: someone migrating from the [-1] the
        # docs used to advertise must be told about THIS rule, not shown a
        # shape error that sends them the wrong way.
        negatives = [i for i in self.position_indices if i < 0]
        if negatives:
            raise ValueError(
                f"position_indices must be non-negative; got {negatives}. "
                "Negative (from-the-end) indices are not supported — pass the "
                "absolute 0-indexed position instead, e.g. the last token of an "
                "n-token prompt is n-1."
            )
        shape = self.activations["shape"]
        if len(shape) != 3:
            raise ValueError(
                "position_indices requires a position-specific 3-D activations "
                f"shape (n_layers, n_positions, hidden); got shape {shape} - a 2-D "
                "(n_layers, hidden) vector broadcasts to ALL positions and the "
                "engine ignores position_indices"
            )
        if len(self.position_indices) != shape[1]:
            raise ValueError(
                f"len(position_indices)={len(self.position_indices)} must equal "
                f"the position axis activations.shape[1]={shape[1]}"
            )
        return self

    @model_validator(mode="after")
    def _validate_position_spans(self) -> SteeringVector:
        # ``position_spans`` is a convenience over an explicit
        # ``position_indices`` list + 3-D tensor: the caller sends a 2-D vector
        # and half-open [start, end) spans, and the wrapper broadcasts (tiles)
        # the vector across every position in the spans at serialization time
        # (see ``to_activation_upstream_body`` / ``_tile_2d_codec_to_3d``).
        if self.position_spans is None:
            return self
        # Mutually exclusive: two ways to name the same axis is ambiguous, so
        # reject rather than guess a precedence.
        if self.position_indices is not None:
            raise ValueError(
                "position_spans and position_indices are mutually exclusive; "
                "pass exactly one (spans broadcast a 2-D vector, indices need a "
                "matching 3-D tensor)"
            )
        # Spans broadcast ONE 2-D (n_layers, d_model) vector. A 3-D tensor would
        # be ambiguous (which position slice tiles?) and is out of scope — point
        # the caller at the explicit path instead of guessing.
        shape = self.activations["shape"]
        if len(shape) != 2:
            raise ValueError(
                "position_spans requires a 2-D (n_layers, d_model) activations "
                f"vector to broadcast across the span; got shape {shape}. Use "
                "position_indices with a matching 3-D tensor for per-position "
                "directions."
            )
        # Endpoint rules, kept consistent with the position_indices negative
        # rejection at ``_positions_require_3d`` (ACS-317 flag F): non-negative,
        # half-open, end strictly after start.
        for start, end in self.position_spans:
            if start < 0 or end < 0:
                raise ValueError(
                    f"position_spans endpoints must be non-negative; got "
                    f"({start}, {end}). Negative (from-the-end) positions are not "
                    "supported — pass absolute 0-indexed positions."
                )
            if end <= start:
                raise ValueError(
                    f"position_spans are half-open [start, end) and require "
                    f"end > start; got ({start}, {end})."
                )
        # Reject overlaps/duplicates: overlapping spans would list a position
        # twice in the expanded ``position_indices``, which the engine may steer
        # twice. Sort by start and check adjacency.
        ordered = sorted(self.position_spans)
        for (_, prev_end), (nxt_start, _) in zip(ordered, ordered[1:]):
            if nxt_start < prev_end:
                raise ValueError(
                    "position_spans must not overlap; merge overlapping ranges "
                    f"(got overlapping spans near position {nxt_start})."
                )
        return self

    def expanded_position_indices(self) -> list[int]:
        """Flatten ``position_spans`` into explicit 0-indexed positions.

        Preserves caller order across spans (each span is [start, end)). Only
        valid after validation, which guarantees non-overlap.
        """
        assert self.position_spans is not None
        return [i for start, end in self.position_spans for i in range(start, end)]

    def engine_payload(self) -> dict[str, Any]:
        """The single steering-vector dict to send under ``vllm_xargs``.

        Plain path: the verbatim 5-field ``model_dump`` (``position_spans``
        excluded — the engine never sees it). Span path: tile the 2-D vector
        across the expanded positions into the 3-D tensor + explicit
        ``position_indices`` the engine already accepts, so the upstream payload
        is byte-shape-identical to the validated explicit-indices path.
        """
        base = self.model_dump(exclude={"position_spans"})
        if self.position_spans is None:
            return base
        indices = self.expanded_position_indices()
        base["activations"] = _tile_2d_codec_to_3d(self.activations, len(indices))
        base["position_indices"] = indices
        return base


def _decode_2d_codec(codec: dict[str, Any]) -> bytes:
    """Decode a 2-D steering codec to its raw layer-major bytes."""
    import zstandard  # local import: only the span path needs the dep

    try:
        # Tolerate MIME-wrapped (line-broken) base64, like the project_onto decoder.
        raw = base64.b64decode("".join(str(codec["data"]).split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"activations.data is not valid base64: {exc}") from None
    if codec.get("compression") == "zstd":
        raw = zstandard.ZstdDecompressor().decompress(raw)
    return raw


def _tile_2d_codec_to_3d(codec: dict[str, Any], n_positions: int) -> dict[str, Any]:
    """Broadcast a 2-D (n_layers, d_model) codec tensor to 3-D over n_positions.

    Repeats each layer's row ``n_positions`` times so the result is
    (n_layers, n_positions, d_model) with every position holding the same layer
    vector — the shape the vLLM-Lens engine uses for position-specific steering.

    Byte layout only: decode → repeat each of ``shape[0]`` contiguous
    C-order layer rows → zstd-compress → base64. Output is always zstd (repeated
    rows compress to ~constant, keeping the tiled payload small). Assumes a
    well-formed 2-D codec.
    """
    import zstandard  # local import: only the span path needs the dep

    shape = codec["shape"]
    n_layers, d_model = int(shape[0]), int(shape[1])
    raw = _decode_2d_codec(codec)
    row = len(raw) // n_layers
    tiled = b"".join(raw[i * row : (i + 1) * row] * n_positions for i in range(n_layers))
    packed = zstandard.ZstdCompressor(level=1).compress(tiled)
    return {
        **codec,
        "data": base64.b64encode(packed).decode(),
        "shape": [n_layers, n_positions, d_model],
        "compression": "zstd",
    }


def prompt_batch_size(prompt: str | list[Any]) -> int:
    """Number of independent prompts in a ``/v1/completions`` ``prompt``.

    A single text or a single pre-tokenized prompt (``list[int]``) is ONE prompt;
    a ``list[str]`` or ``list[list[int]]`` is a batch. Used to bound output work
    (batch_size × n × max_tokens) — NOT to count tokens. ``type(x) is int``
    excludes bools.
    """
    if isinstance(prompt, list):
        if prompt and type(prompt[0]) is int:
            return 1  # one pre-tokenized prompt (token ids)
        return max(1, len(prompt))
    return 1


class CompletionsRequest(BaseModel):
    """Validated body for ``POST /v1/completions``.

    ``extra="forbid"`` means unknown fields produce a 422 from FastAPI (we map
    that to a 400 in the handler). The intent is that callers paste-from-Modal
    or copy-from-OpenAI-tutorial don't get silent drops on sampler params we
    don't yet support — they learn about it at the wrapper boundary.

    Field semantics that aren't obvious from the type:
      - ``prompt`` is **raw text**; the wrapper never applies a chat template.
        Lists of strings are accepted (vLLM treats them as a batch).
      - ``max_tokens=None`` is accepted only for a single prompt with ``n=1``.
        The route injects a bounded default no larger than 32k or the remaining
        model context. The wrapper's budget clamp may still reduce this further;
        see X-Acs-Max-Tokens-Clamped response header.
      - ``logprobs`` is an integer count of top alternatives per position
        (vLLM's OpenAI-compatible semantics). 0/None disables, otherwise the
        value is the top-k size.
      - ``stop`` accepts up to 4 strings; vLLM rejects longer lists.
    """

    model_config = ConfigDict(extra="forbid")

    # Routing / addressing
    model: str | None = None

    # Prompt — text, a batch of texts, OR pre-tokenized token ids (one prompt as
    # ``list[int]``, a batch as ``list[list[int]]``). Token-id prompts are sent to
    # vLLM verbatim, with no re-tokenization (avoids the lossy id → str → id
    # round-trip on non-bijective tokenizers — ACS-168). The ``mode="before"``
    # validator rejects mixed/empty shapes before Pydantic's lax union coercion
    # can silently turn a ``list[int]`` into a batch of text prompts.
    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        ..., description="Raw prompt(s): text, list of texts, or token ids. No chat templating."
    )

    # Length
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    n: int = Field(default=1, ge=1, le=16)

    # Core sampler
    temperature: float = Field(default=1.0, ge=0.0, le=100.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, description="-1 disables; otherwise >= 1.")
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)

    # Penalties
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)

    # Determinism
    seed: int | None = Field(default=None, ge=0)

    # Stop sequences (OpenAI shape: a string or a small list). The list branch
    # carries ``Field(max_length=4)`` so the 4-item cap renders as ``maxItems``
    # in the OpenAPI schema (a bare field_validator is invisible to the schema);
    # ``_stop_max_4`` below stays as belt-and-suspenders runtime enforcement.
    stop: str | Annotated[list[str], Field(max_length=4)] | None = None

    # Observability
    #
    # ``logprobs`` is a per-generated-token top-k count (0..MAX_LOGPROBS). It is
    # positive-only: vLLM's OpenAI validator rejects ``-1`` on the completion
    # side, and full-vocab × up-to-32k output positions would be catastrophic,
    # so full-vocab is deliberately NOT offered for generated tokens.
    logprobs: int | None = Field(default=None, ge=0, le=MAX_LOGPROBS)
    echo: bool = False
    # ``prompt_logprobs`` additionally accepts the ``-1`` full-vocab sentinel
    # (FULL_VOCAB_SENTINEL): return the whole distribution at every prompt
    # position. The prompt-length gate that keeps this from OOMing the server
    # lives in ``services/completions.check_full_vocab_prompt_logprobs`` (it
    # needs the tokenizer, which the schema doesn't have). Positive values are
    # a top-k count capped at MAX_LOGPROBS, exactly like ``logprobs``.
    prompt_logprobs: int | None = Field(default=None, ge=FULL_VOCAB_SENTINEL, le=MAX_LOGPROBS)

    # Streaming
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    # Pass-through user tag (OpenAI shape, useful for client-side bookkeeping).
    user: str | None = None

    # Special-token handling. vLLM's completion endpoint prepends BOS by default
    # (add_special_tokens=True). A text prompt that ALREADY carries a BOS — e.g.
    # a caller chat-templating client-side and emitting ``<|begin_of_text|>`` —
    # then gets a DOUBLE BOS, shifting every token alignment by one (ACS-319, cost
    # the usersonas team real debugging time). Expose the control so such callers
    # can pass ``add_special_tokens=false``. Default True matches the engine and
    # every existing caller's current behavior; forwarded verbatim to vLLM via
    # ``to_upstream_body()`` (``exclude_unset`` means a caller who omits it sends
    # nothing, preserving the engine default) — on both the normal and activation
    # upstreams, whose vLLM/vLLM-Lens builds both accept the field.
    add_special_tokens: bool = True

    # Activation harvesting + steering (ACS-199). Presence of EITHER field routes
    # the request to the model's activation upstream (acs-<id>-activation) rather
    # than the normal workbench upstream, and both are forwarded to vllm-lens
    # nested under ``vllm_xargs``. They are NOT vLLM sampler params — the normal
    # ``to_upstream_body()`` excludes them so a plain completion never leaks them.
    output_residual_stream: bool | list[int] = Field(
        default=False,
        description=(
            "Capture the residual stream. Pass ``true`` for ALL decoder layers, or "
            "a list of layer indices (e.g. ``[15, 20]``) to capture only those — "
            "the subset is applied on the engine, so only those layers cross the "
            "wire (ACS-266, requires vLLM-Lens >= 1.2.0). Returned under a top-level "
            "``activations`` key. Prompt length is capped per model."
        ),
    )
    apply_steering_vectors: list[SteeringVector] | None = Field(
        default=None,
        description="Steering vectors added to the residual stream during generation.",
    )

    @field_validator("top_k")
    @classmethod
    def _top_k_disable_or_positive(cls, v: int) -> int:
        if v == -1 or v >= 1:
            return v
        raise ValueError("top_k must be -1 (disabled) or >= 1")

    @field_validator("stop")
    @classmethod
    def _stop_max_4(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("stop accepts at most 4 sequences")
        return v

    @field_validator("output_residual_stream", mode="before")
    @classmethod
    def _normalize_capture(cls, v: Any) -> Any:
        # Accept a bool (all / none) or a list of layer indices (subset, ACS-266).
        # A bare bool captures all-or-nothing; a list captures exactly those layers
        # on the engine (vLLM-Lens >= 1.2.0 json.loads the list worker-side and
        # filters by layer index). Normalize the list here: ints only, >= 0,
        # deduped and sorted, non-empty. Range (idx < n_layers) is model-specific,
        # so the engine enforces the upper bound and returns its own 4xx.
        if isinstance(v, bool):
            return v
        if isinstance(v, list):
            if not v:
                raise ValueError(
                    "output_residual_stream must be a non-empty list of layer "
                    "indices (or true for all layers / false for none)"
                )
            layers: list[int] = []
            for x in v:
                # bool is a subclass of int — exclude it so [true] isn't read as [1].
                if isinstance(x, bool) or not isinstance(x, int):
                    raise ValueError(
                        "output_residual_stream layer indices must be integers"
                    )
                if x < 0:
                    raise ValueError(
                        "output_residual_stream layer indices must be >= 0"
                    )
                layers.append(x)
            return sorted(set(layers))
        raise ValueError(
            "output_residual_stream must be true, false, or a list of layer indices"
        )

    @field_validator("apply_steering_vectors")
    @classmethod
    def _validate_steering(cls, v: list[SteeringVector] | None) -> list[SteeringVector] | None:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "apply_steering_vectors must not be empty; omit it to disable steering"
            )
        if len(v) > MAX_STEERING_VECTORS:
            raise ValueError(
                f"apply_steering_vectors has {len(v)} vectors; at most "
                f"{MAX_STEERING_VECTORS} may be applied per request"
            )
        return v

    @model_validator(mode="after")
    def _no_stream_capture(self) -> CompletionsRequest:
        # Captured activations are attached to the final non-streaming completion
        # body (vllm-lens returns the tensor in the JSON response, not in SSE
        # chunks), so streaming a capture would silently drop them. Reject it
        # loudly. Steering does NOT return activations, so stream + steering is
        # fine and stays allowed.
        if self.stream and self.output_residual_stream:
            raise ValueError(
                "output_residual_stream (activation capture) is not supported with "
                "stream=true — captured activations are returned only in the "
                "non-streaming response body; set stream=false to capture"
            )
        return self

    @model_validator(mode="after")
    def _steering_single_sequence_only(self) -> CompletionsRequest:
        # The engine applies steering vectors to the FIRST sequence of a request
        # only: with a batched prompt the remaining prompts come back unsteered,
        # and with n > 1 the sampled fan-out bypasses steering entirely — both
        # with HTTP 200 (re-verified against the live engine in beta feedback,
        # 2026-07-18; the greedy n>1 case is already rejected by vLLM itself).
        # A silently half-steered batch invalidates an experiment, so reject
        # both shapes until the engine supports fan-out.
        if self.apply_steering_vectors is None:
            return self
        if prompt_batch_size(self.prompt) > 1:
            raise ValueError(
                "apply_steering_vectors is not supported with a batched prompt - "
                "the engine steers only the first prompt of a batch; send one "
                "request per prompt"
            )
        if self.n > 1:
            raise ValueError(
                "apply_steering_vectors is not supported with n > 1 - steering is "
                "bypassed on the sampled completions; send n single-completion "
                "requests with different seeds instead"
            )
        return self

    @model_validator(mode="after")
    def _capture_single_sequence_only(self) -> CompletionsRequest:
        # Capture (output_residual_stream) is single-sequence only, mirroring the
        # steering guard above (ACS-251/ACS-255). The response tensor is
        # (n_layers, n_tokens, hidden) for ONE sequence; a batched prompt or n > 1
        # has no unambiguous single-sequence tensor to return (the engine's
        # multi-sequence capture behaviour is unverified — no GPU in CI), and it
        # also breaks the RAM/response-size accounting in
        # ``check_activation_prompt_length``, which bounds prompt + generated − 1
        # positions for ONE sequence. Reject both shapes pre-flight so the
        # silent-wrong case can never reach the engine; send one request per
        # prompt / per sample instead.
        if not self.output_residual_stream:
            return self
        if prompt_batch_size(self.prompt) > 1:
            raise ValueError(
                "output_residual_stream (activation capture) is not supported with "
                "a batched prompt - capture returns a single-sequence tensor; send "
                "one request per prompt"
            )
        if self.n > 1:
            raise ValueError(
                "output_residual_stream (activation capture) is not supported with "
                "n > 1 - capture returns a single-sequence tensor; send n "
                "single-completion requests with different seeds instead"
            )
        return self

    @field_validator("prompt", mode="before")
    @classmethod
    def _validate_prompt_shape(cls, v: Any) -> Any:
        """Accept str | list[str] | list[int] | list[list[int]]; reject mixed/empty.

        Runs *before* Pydantic's union coercion: a raw ``list[int]`` would
        otherwise be lax-coerced element-by-element into ``list[str]`` (token ids
        silently becoming a *batch of text prompts*). We classify the raw value
        here and pass valid shapes through unchanged; everything else fails
        loudly. ``type(x) is int`` excludes bools (``True``/``False`` are ints).
        """
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if not v:
                raise ValueError("prompt must not be empty")
            if all(type(x) is str for x in v):
                return v
            if all(type(x) is int for x in v):
                return v  # one pre-tokenized prompt (token ids)
            if all(isinstance(x, list) and x and all(type(y) is int for y in x) for x in v):
                return v  # batch of pre-tokenized prompts
            raise ValueError(
                "prompt must be a string, a list of strings, a list of token ids "
                "(integers), or a list of lists of token ids - not a mix"
            )
        raise ValueError("prompt must be a string or a list")

    @model_validator(mode="after")
    def _cap_output_work(self) -> CompletionsRequest:
        prompt_count = prompt_batch_size(self.prompt)
        fans_out = prompt_count > 1 or self.n > 1
        if fans_out and self.max_tokens is None:
            raise ValueError(
                "max_tokens is required when prompt is a list or n > 1 "
                f"(maximum output work is {MAX_OUTPUT_WORK_TOKENS} tokens)"
            )
        if self.max_tokens is None:
            return self
        requested_work = prompt_count * self.n * self.max_tokens
        if requested_work > MAX_OUTPUT_WORK_TOKENS:
            raise ValueError(
                "prompt_count × n × max_tokens must be "
                f"<= {MAX_OUTPUT_WORK_TOKENS}; got "
                f"{prompt_count} × {self.n} × {self.max_tokens} = "
                f"{requested_work}"
            )
        return self

    # Fields that are wrapper-level activation controls, never top-level vLLM
    # sampler params. Excluded from every upstream body; the activation path
    # re-nests them under ``vllm_xargs`` (see ``to_activation_upstream_body``).
    _ACTIVATION_FIELDS = ("output_residual_stream", "apply_steering_vectors")

    @property
    def has_activation_params(self) -> bool:
        """True when the request asks for activation capture and/or steering.

        Used by the route to pick the activation upstream instead of the normal
        workbench upstream.
        """
        return bool(self.output_residual_stream) or self.apply_steering_vectors is not None

    def to_upstream_body(self) -> dict[str, Any]:
        """Dict to forward to vLLM, omitting fields the caller didn't set.

        ``exclude_unset=True`` is important: vLLM has its own defaults and we
        don't want to override them by always sending ``temperature=1.0`` when
        the caller passed no temperature. Activation controls are excluded — a
        plain completion must never carry them to the workbench upstream.
        """
        return self.model_dump(
            exclude_unset=True, exclude_none=True, exclude=set(self._ACTIVATION_FIELDS)
        )

    def to_activation_upstream_body(self) -> dict[str, Any]:
        """Dict to forward to the activation (vLLM-Lens) upstream.

        Same allowlisted sampler fields as ``to_upstream_body()`` plus a
        ``vllm_xargs`` dict in the EXACT shape the 0.19.1 vLLM-Lens engine expects
        (verified 2026-07-06; layer-subset path added ACS-266):
          - ``output_residual_stream: true`` — a bare bool; captures all layers.
          - ``output_residual_stream: "[15, 20]"`` — a ``json.dumps``'d **string**
            of layer indices for a subset (vLLM-Lens >= 1.2.0 json.loads it
            worker-side and captures only those layers). A bare list 400s at vLLM's
            vllm_xargs validation, so it MUST be a string — same convention as
            ``apply_steering_vectors``.
          - ``apply_steering_vectors`` — a ``json.dumps``'d **string** of the
            SteeringVector dict list (the plugin json.loads it worker-side).
        """
        body = self.to_upstream_body()
        xargs: dict[str, Any] = {}
        ors = self.output_residual_stream
        if ors is True:
            xargs["output_residual_stream"] = True
        elif isinstance(ors, list) and ors:
            xargs["output_residual_stream"] = json.dumps(ors)
        if self.apply_steering_vectors is not None:
            # ``engine_payload`` returns the exact 5-field shape verified
            # end-to-end against the live engine (2026-07-06): activations,
            # layer_indices, scale, norm_match, position_indices — and NEVER
            # ``position_spans`` (excluded so the engine sees only fields it
            # knows). For a span vector it also tiles the 2-D direction into the
            # 3-D tensor + explicit position_indices, so the upstream payload is
            # byte-shape-identical to the already-validated explicit path.
            # Full dump (NOT exclude_none) keeps an explicit
            # ``position_indices: null`` present, removing absent-vs-null
            # ambiguity.
            xargs["apply_steering_vectors"] = json.dumps(
                [sv.engine_payload() for sv in self.apply_steering_vectors]
            )
        body["vllm_xargs"] = xargs
        return body


class HarvestRequest(BaseModel):
    """Body of ``POST /v1/harvest`` — one self-serve bulk-harvest job (ACS-245).

    Shape validation only (types, ranges, non-empty prompts). The
    settings-dependent caps — prompt COUNT (``harvest_max_prompts``) and the
    chars/4 estimated-token bound (``harvest_max_est_tokens``) — live in the
    route, which has ``Settings``; same split as the completions caps.
    ``extra="forbid"`` for the same reason as ``CompletionsRequest``: a typo'd
    field must 400, not silently fall back to a default on a paid GPU job.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    prompts: list[str] = Field(min_length=1)
    # Residual-stream layer indices to keep (block-output convention, see
    # docs/design/activation-offline-harvest.md). None = the engine's default
    # quartile subset (blocks at ~25/50/75% depth — the middle-depth band interp
    # work targets, and ~10–40× less storage than every block); "all" = every
    # block output. Bounds are sanity caps, not per-model truth: 512 comfortably
    # exceeds any served model's depth, and an out-of-range-for-THIS-model index
    # fails fast remotely before the model loads.
    layers: Annotated[list[int], Field(max_length=256)] | Literal["all"] | None = None
    # Prompts per output safetensors shard.
    shard_size: int = Field(default=32, ge=1, le=4096)
    # Forward-pass batch size inside the harvest function. None (default) defers
    # to the function's own resolution — batched on single-GPU, sequential on
    # multi-GPU — so big-model batching stays opt-in; the wrapper never invents
    # a batch size.
    batch_size: int | None = Field(default=None, ge=1, le=64)
    # Optional: project the residual stream onto these directions and write the
    # PROJECTIONS instead of raw activations (ACS-320). Same tensor-codec dict as
    # steering vectors, shape (n_directions, hidden). Collapses shard size by
    # hidden/n_directions — ~500x for a handful of directions — which is what
    # makes corpus-scale probing practical. Directions are L2-normalized
    # server-side; the manifest records that the run holds projections.
    project_onto: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Tensor-codec dict {data (base64), dtype, shape} of shape (n_directions, "
            "hidden), uncompressed — unlike steering vectors, which accept zstd. "
            "When set, shards hold per-token projections onto these directions "
            "instead of the raw residual stream."
        ),
    )

    @field_validator("project_onto")
    @classmethod
    def _projection_shape(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Shape/size checks the wrapper can do without knowing the model.

        The hidden-dim match is checked remotely (the harvester reads it from the
        model config) — but everything catchable here is caught here, because the
        alternative is finding out after a GPU job has already started.
        """
        if v is None:
            return v
        shape = v.get("shape")
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
            raise ValueError(
                "project_onto.shape must be 2-D (n_directions, hidden); got "
                f"{shape!r}"
            )
        n_dirs, hidden = shape
        if not all(
            isinstance(d, int) and not isinstance(d, bool) for d in (n_dirs, hidden)
        ):
            raise ValueError("project_onto.shape entries must be integers")
        if not 1 <= n_dirs <= MAX_PROJECTION_DIRECTIONS:
            raise ValueError(
                f"project_onto must hold 1..{MAX_PROJECTION_DIRECTIONS} directions; "
                f"got {n_dirs}"
            )
        if hidden < 1:
            raise ValueError(f"project_onto hidden dim must be positive; got {hidden}")
        data = v.get("data")
        if not isinstance(data, str):
            raise ValueError("project_onto.data must be a base64 string")
        # Before the length check: a genuinely zstd-compressed payload has the
        # wrong byte count too, and "needs 1048576 bytes" would send the caller
        # chasing their shape instead of the compression they chose.
        if v.get("compression") not in (None, "none", ""):
            raise ValueError(
                "project_onto compression is not supported; send raw codec data"
            )
        dtype = str(v.get("dtype", "")).lower()
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                "project_onto.dtype must be float32, float16 or bfloat16; got "
                f"{v.get('dtype')!r}"
            )
        # Decode here rather than letting it fail on the GPU: a remote ValueError
        # comes back as an opaque "job failed on Modal; details in server logs"
        # (modal_ops.classify_poll_exception), which is a terrible way to learn
        # you sent a truncated tensor.
        try:
            # Line-wrapped base64 is what `base64.encodebytes`, the coreutils
            # `base64` CLI and Java's MIME encoder all produce, so strip
            # whitespace before validating rather than rejecting a payload that
            # is merely wrapped at 76 columns.
            raw = base64.b64decode("".join(data.split()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"project_onto.data is not valid base64: {exc}") from None
        itemsize = 4 if dtype == "float32" else 2
        expected = n_dirs * hidden * itemsize
        if len(raw) != expected:
            raise ValueError(
                f"project_onto.data is {len(raw)} bytes; shape {n_dirs}x{hidden} "
                f"of {dtype} needs {expected}"
            )
        return v

    # See ``CompletionsRequest.add_special_tokens``. Harvest tokenizes prompts
    # server-side (HF / vLLM tokenizer, ``add_special_tokens=True`` by default),
    # so a prompt already carrying a BOS gets a double BOS in the captured token
    # stream — and harvest has NO pre-tokenized escape hatch (prompts are
    # ``list[str]`` only), so this is the only clean fix for client-side chat
    # templating (ACS-319). Default True (current behavior). The route forwards
    # it to the Modal harvest function only when the caller explicitly set it, so
    # existing traffic stays compatible with harvest apps deployed before the
    # ``harvest()`` signature gained this parameter.
    add_special_tokens: bool = True

    @field_validator("prompts")
    @classmethod
    def _prompts_non_empty(cls, v: list[str]) -> list[str]:
        if any(not p for p in v):
            raise ValueError("prompts must be non-empty strings")
        return v

    @field_validator("layers")
    @classmethod
    def _layers_in_range(cls, v: list[int] | str | None) -> list[int] | str | None:
        if v is not None and v != "all":
            if not v:
                raise ValueError('layers must be null, "all", or a non-empty list of ints')
            if any((not isinstance(i, int)) or isinstance(i, bool) or i < 0 or i > 512 for i in v):
                raise ValueError("layers must be integers in [0, 512]")
        return v

    @property
    def estimated_tokens(self) -> int:
        """Cheap total-token estimate: chars/4 across all prompts.

        A pre-flight cost bound, not billing — the workbench uses the same
        chars/4 heuristic. Avoids a tokenizer pass over thousands of prompts on
        the submit path.
        """
        return sum(len(p) for p in self.prompts) // 4
