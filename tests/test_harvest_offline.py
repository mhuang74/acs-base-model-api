"""Unit tests for the pure (torch/boto3-free) helpers in serving.harvest_offline.

These cover the shard/manifest/layer-parsing logic that determines file layout
and downloader-facing metadata — the parts that must stay correct without a GPU.
The Modal ``harvest`` function itself is exercised by the live GPU smoke
(``modal run serving/harvest_offline.py``), not here.
"""

from __future__ import annotations

import importlib.util
import io
import threading

import pytest

from serving import harvest_offline as ho


def test_chunk_prompts_even_split():
    assert ho.chunk_prompts(["a", "b", "c", "d"], 2) == [[0, 1], [2, 3]]


def test_chunk_prompts_ragged_last_shard():
    assert ho.chunk_prompts(["a", "b", "c"], 2) == [[0, 1], [2]]


def test_chunk_prompts_shard_larger_than_input():
    assert ho.chunk_prompts(["a", "b"], 32) == [[0, 1]]


def test_chunk_prompts_empty():
    assert ho.chunk_prompts([], 4) == []


def test_chunk_prompts_rejects_zero_shard():
    with pytest.raises(ValueError):
        ho.chunk_prompts(["a"], 0)


def test_shard_and_manifest_keys_are_zero_padded_and_scoped():
    assert ho.shard_key("run42", 7) == "harvest/run42/shard_00007.safetensors"
    assert ho.manifest_key("run42") == "harvest/run42/manifest.json"


def test_parse_layers_none_and_empty_mean_default_subset():
    assert ho.parse_layers(None) is None
    assert ho.parse_layers("") is None


def test_parse_layers_all_sentinel():
    assert ho.parse_layers("all") == "all"
    assert ho.parse_layers(" ALL ") == "all"


def test_default_layer_indices_quartiles():
    assert ho.default_layer_indices(32) == [8, 16, 24]  # llama-8b
    assert ho.default_layer_indices(126) == [31, 63, 94]  # llama-405b
    # degenerate small n still returns valid in-range, deduped indices
    assert ho.default_layer_indices(2) == [0, 1]


def test_parse_layers_list_and_whitespace():
    assert ho.parse_layers("8,12,16") == [8, 12, 16]
    assert ho.parse_layers("8, 12 , 16,") == [8, 12, 16]


def test_build_manifest_shape_and_native_convention():
    m = ho.build_manifest(
        run_id="r",
        model_id="llama-8b",
        hf_repo="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        layer_indices=[1, 2, 3],
        prompts=["p0", "p1"],
        shards=[{"key": "harvest/r/shard_00000.safetensors", "prompt_indices": [0, 1]}],
        batch_size=8,
        mean_token_norm={"1": 12.5},
    )
    assert m["n_prompts"] == 2
    assert m["layout_version"] == 2
    assert m["layer_indices"] == [1, 2, 3]
    assert m["capture"] == "hf-transformers output_hidden_states"
    # HF loop uses hidden_states[k+1] (output of block k); the LAST layer is
    # POST-final-norm (documented for downloaders — differs from the inline engine).
    assert "hidden_states[k+1]" in m["residual_stream_convention"]
    assert "POST-final-norm" in m["residual_stream_convention"]
    assert m["prompts"] == ["p0", "p1"]
    assert m["shards"][0]["prompt_indices"] == [0, 1]
    assert m["batch_size"] == 8
    # tokens + statistics travel with the run (Activault layout) and must be
    # documented for downloaders.
    assert "tokens_{i}" in m["tokens_layout"]
    assert "stats.safetensors" in m["statistics"]
    assert m["mean_token_norm"] == {"1": 12.5}
    # add_special_tokens provenance (ACS-319): defaults True and is recorded so a
    # downloader knows whether a leading BOS is present in tokens_{i}.
    assert m["add_special_tokens"] is True


def test_build_manifest_records_add_special_tokens_false():
    # ACS-319: a harvest run tokenized with add_special_tokens=False (client-side
    # BOS) records that in the manifest, and the token-layout doc no longer
    # hard-claims a leading BOS.
    m = ho.build_manifest(
        run_id="r",
        model_id="llama-8b",
        hf_repo="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        layer_indices=[1],
        prompts=["p0"],
        shards=[{"key": "harvest/r/shard_00000.safetensors", "prompt_indices": [0]}],
        add_special_tokens=False,
    )
    assert m["add_special_tokens"] is False
    assert "iff add_special_tokens" in m["tokens_layout"]


def test_build_manifest_vllm_capture_documents_prenorm_last_layer():
    # ACS-268: the vLLM offline path captures the PRE-final-norm residual for every
    # layer (incl. the last), consistent with the inline engine. The manifest must
    # record the backend + convention so downloaders know the last layer differs
    # from the old HF harvest.
    m = ho.build_manifest(
        run_id="r",
        model_id="llama-405b",
        hf_repo="meta-llama/Llama-3.1-405B",
        dtype="bfloat16",
        layer_indices=[0, 63, 125],
        prompts=["p0"],
        shards=[{"key": "harvest/r/shard_00000.safetensors", "prompt_indices": [0]}],
        capture="vllm-lens offline output_residual_stream",
    )
    assert m["capture"] == "vllm-lens offline output_residual_stream"
    conv = m["residual_stream_convention"]
    assert "hidden_states[k+1]" in conv
    assert "PRE-final-norm" in conv
    assert "POST-final-norm" not in conv  # vLLM last layer is pre-norm, not post


def test_batch_prompt_indices_longest_first_groups():
    # sizes: idx0=5, idx1=50, idx2=20, idx3=1 -> longest-first order 1,2,0,3
    batches = ho.batch_prompt_indices([0, 1, 2, 3], [5, 50, 20, 1], 2)
    assert batches == [[1, 2], [0, 3]]
    # every index appears exactly once
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3]


def test_batch_prompt_indices_token_budget_caps_padded_footprint():
    # 4 prompts of ~100 est tokens; padded footprint of a pair = 2*100 = 200.
    # budget 250 forbids pairs (200 <= 250 ok for 2, 300 > 250 for 3) with
    # batch_size 3: batches grow only while (len+1)*first <= budget.
    batches = ho.batch_prompt_indices([0, 1, 2, 3], [100, 100, 100, 100], 3, token_budget=250)
    assert batches == [[0, 1], [2, 3]]
    # a single over-budget prompt still forms its own batch (never dropped)
    assert ho.batch_prompt_indices([0], [10_000], 8, token_budget=250) == [[0]]


def test_batch_prompt_indices_batch_of_one_preserves_indices():
    # sizes is indexed by GLOBAL prompt index (a shard's idxs are a subset)
    sizes = [0, 0, 0, 0, 10, 0, 0, 10]
    assert ho.batch_prompt_indices([4, 7], sizes, 1) == [[4], [7]]


def test_batch_prompt_indices_rejects_zero():
    with pytest.raises(ValueError):
        ho.batch_prompt_indices([0], [1], 0)


def test_url_ttl_clamped_to_seven_days():
    assert ho.URL_TTL_S <= 7 * 24 * 3600


def test_harvest_max_containers_default_is_n_gpu_aware():
    # Big models (>1 GPU) cap the fleet at the wrapper's big-model cap so admitted
    # jobs never queue behind a warm container (which would trip the 180-min stale
    # janitor). 8B stays uncapped-ish for throughput.
    assert ho._harvest_max_containers_default(8) == 2
    assert ho._harvest_max_containers_default(2) == 2
    assert ho._harvest_max_containers_default(1) == 16


def test_harvest_big_model_fleet_default_matches_wrapper_big_model_cap():
    """Coupling guard (ACS-265): the big-model harvest fleet default must be >= the
    wrapper's cross-key big-model cap, or a second admitted big-model job queues and
    the 180-min stale janitor false-fails it while it's still legitimately running.
    Locks the two defaults together so raising one without the other is caught here.
    """
    import re
    from pathlib import Path

    settings_path = (
        Path(__file__).resolve().parents[1]
        / "base_model_wrapper"
        / "src"
        / "wrapper"
        / "settings.py"
    )
    src = settings_path.read_text()
    # Parse the field default without importing the wrapper package (pydantic-settings
    # + env deps aren't guaranteed in this test env). The field is a plain
    # `harvest_max_running_big_model: int = Field(default=N, ...)`.
    m = re.search(r"harvest_max_running_big_model:\s*int\s*=\s*Field\(\s*default=(\d+)", src)
    assert m, "could not locate harvest_max_running_big_model default in wrapper settings"
    wrapper_big_model_cap = int(m.group(1))
    assert ho._harvest_max_containers_default(8) >= wrapper_big_model_cap


# ---- projection directions (ACS-320) ---------------------------------------

# Scoped to the tests that need them — a module-level importorskip would skip
# this file's pure helpers too (they must run without torch installed).

needs_np = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None, reason="needs numpy"
)
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch"
)


def _codec(arr, dtype="float32"):
    """Encode a numpy array the way a client would (raw base64, no compression)."""
    import base64

    return {
        "data": base64.b64encode(arr.tobytes()).decode(),
        "dtype": dtype,
        "shape": list(arr.shape),
        "compression": "none",
    }


@needs_np
def test_decode_projection_normalizes_directions():
    import numpy as _NP

    """Projecting onto a non-unit vector silently rescales every number and the
    caller can't see it in the output — so we normalize server-side."""
    arr = _NP.array([[3.0, 4.0, 0.0, 0.0]], dtype="float32")  # norm 5
    dirs = ho.decode_projection_directions_np(_codec(arr), hidden_size=4)
    assert dirs.shape == (1, 4)
    assert abs(float(_NP.linalg.norm(dirs, axis=1)[0]) - 1.0) < 1e-6
    assert abs(float(dirs[0, 0]) - 0.6) < 1e-6


@needs_np
def test_decode_projection_matches_manual_dot_product():
    import numpy as _NP

    """The whole point is that server-side projection equals what the client
    would have computed from the raw tensor."""
    dirs_raw = _NP.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype="float32")
    dirs = ho.decode_projection_directions_np(_codec(dirs_raw), hidden_size=4)
    acts = _NP.array([[[1.0, 3.0, 5.0, 7.0]]], dtype="float32")  # [1 layer, 1 tok, hidden]
    got = acts @ dirs.T
    assert got.shape == (1, 1, 2)
    assert abs(float(got[0, 0, 0]) - 1.0) < 1e-6   # onto x̂
    assert abs(float(got[0, 0, 1]) - 3.0) < 1e-6   # onto ŷ (unit-normalized)


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda c: c.update(shape=[4]), "2-D"),
        (lambda c: c.update(shape=[1, 999]), "hidden size"),
        (lambda c: c.update(dtype="int8"), "unsupported"),
        (lambda c: c.update(compression="gzip"), "compression"),
        (lambda c: c.update(data="not base64!!"), "base64"),
        (lambda c: c.update(shape=[2, 4]), "needs"),  # data too small for shape
    ],
)
@needs_np
def test_decode_projection_rejects_bad_payloads(mutate, msg):
    import numpy as _NP

    """Every one of these must fail BEFORE the model loads — a bad payload
    should cost seconds, not a 50-minute 405B load."""
    c = _codec(_NP.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    mutate(c)
    with pytest.raises(ValueError, match=msg):
        ho.decode_projection_directions_np(c, hidden_size=4)


@needs_np
def test_decode_projection_rejects_zero_norm_direction():
    import numpy as _NP

    arr = _NP.zeros((1, 4), dtype="float32")
    with pytest.raises(ValueError, match="zero-norm"):
        ho.decode_projection_directions_np(_codec(arr), hidden_size=4)


def test_manifest_marks_projected_runs_distinctly():
    """A projected shard must never be mistakable for raw activations."""
    raw = ho.build_manifest(
        run_id="r", model_id="m", hf_repo="h", dtype="bfloat16",
        layer_indices=[1], prompts=["p"], shards=[],
    )
    assert raw["contents"] == "activations"
    assert "projection" not in raw
    assert "hidden]" in raw["tensor_layout"]

    proj = ho.build_manifest(
        run_id="r", model_id="m", hf_repo="h", dtype="bfloat16",
        layer_indices=[1], prompts=["p"], shards=[], n_projection_dirs=8,
    )
    assert proj["contents"] == "projections"
    assert proj["projection"]["n_directions"] == 8
    assert proj["projection"]["normalized"] is True
    assert "n_directions] float32" in proj["tensor_layout"]
    assert "PROJECTED" in proj["tensor_layout"]


@needs_torch
def test_decode_projection_torch_wrapper_matches_numpy():
    """The torch entry point the harvester calls must agree with the numpy one."""
    import numpy as _NP

    c = _codec(_NP.array([[3.0, 4.0, 0.0, 0.0]], dtype="float32"))
    assert _NP.allclose(
        ho.decode_projection_directions(c, 4).numpy(),
        ho.decode_projection_directions_np(c, 4),
    )


@needs_np
def test_decode_projection_reads_bfloat16_bit_patterns():
    """bf16 is what activations are, so it's the dtype callers will send.

    The values are chosen so a naive `astype(float32)` of the raw uint16 words
    survives normalization with a *different* ratio: correct decoding gives
    [1, 2] -> [0.447, 0.894]; reading the bit patterns as integers gives
    [16256, 16384] -> [0.704, 0.710]. Without this, dropping the bit-widen is
    invisible to the suite.
    """
    import numpy as _NP

    # bf16 bit patterns: 1.0 == 0x3F80, 2.0 == 0x4000.
    bits = _NP.array([0x3F80, 0x4000], dtype="uint16")
    codec = _codec(bits, dtype="bfloat16")
    codec["shape"] = [1, 2]
    dirs = ho.decode_projection_directions_np(codec, hidden_size=2)
    assert abs(float(dirs[0, 0]) - 0.4472136) < 1e-5
    assert abs(float(dirs[0, 1]) - 0.8944272) < 1e-5


@needs_np
def test_decode_projection_rejects_non_finite_values():
    """An inf/NaN direction would poison every projection in the run with NaN,
    silently — the shards would look well-formed and be entirely useless."""
    import numpy as _NP

    for bad in (_NP.inf, _NP.nan):
        arr = _NP.array([[1.0, bad, 0.0, 0.0]], dtype="float32")
        with pytest.raises(ValueError, match="non-finite"):
            ho.decode_projection_directions_np(_codec(arr), hidden_size=4)


@needs_np
def test_decode_projection_survives_float32_norm_overflow():
    """A direction whose squared norm overflows float32: computing the norm in
    float32 gives inf, and arr/inf is an all-zero direction that raises nothing
    and returns 0.0 for the whole run. float64 norms keep it a unit vector."""
    import numpy as _NP

    arr = _NP.array([[3e38, 3e38, 0.0, 0.0]], dtype="float32")
    out = ho.decode_projection_directions_np(_codec(arr), hidden_size=4)
    assert _NP.isfinite(out).all()
    assert out.any(), "direction collapsed to all zeros"
    assert abs(float(_NP.linalg.norm(out.astype("float64"))) - 1.0) < 1e-6


@needs_np
def test_decode_projection_reads_float16():
    """float16 is an advertised dtype; nothing else covers the 2-byte path."""
    import numpy as _NP

    arr = _NP.array([[1.0, 2.0, 3.0, 4.0]], dtype="float16")
    out = ho.decode_projection_directions_np(_codec(arr, dtype="float16"), hidden_size=4)
    expected = arr.astype("float64")[0]
    expected = expected / _NP.linalg.norm(expected)
    assert _NP.allclose(out[0], expected, atol=1e-6)


@needs_np
def test_decode_projection_accepts_line_wrapped_base64():
    """`base64.encodebytes`, the coreutils `base64` CLI and Java's MIME encoder
    all wrap at 76 columns. Rejecting that would be a decoding puzzle for anyone
    who built their payload with a shell pipeline."""
    import base64 as _b64

    import numpy as _NP

    arr = _NP.arange(8, dtype="float32").reshape(1, 8) + 1.0
    wrapped = _b64.encodebytes(arr.tobytes()).decode()  # contains newlines
    assert "\n" in wrapped
    codec = {"data": wrapped, "dtype": "float32", "shape": [1, 8], "compression": "none"}
    out = ho.decode_projection_directions_np(codec, hidden_size=8)
    assert out.shape == (1, 8)


@needs_torch
def test_project_activations_matches_a_manual_projection():
    """The chunked, preallocated implementation must equal the obvious one — a
    dropped transpose or a mis-sliced chunk would silently ship every shard with
    the wrong last axis under a manifest that says nothing is wrong."""
    import torch

    torch.manual_seed(0)
    x = torch.randn(3, 37, 16, dtype=torch.bfloat16)
    d = torch.randn(5, 16, dtype=torch.float32)
    got = ho.project_activations(x, d)
    assert got.shape == (3, 37, 5)
    assert got.dtype == torch.float32
    expected = x.float() @ d.T
    assert torch.allclose(got, expected, atol=1e-5)


@needs_torch
def test_project_activations_chunk_boundary_is_not_a_seam():
    """n_tok deliberately straddles the token-chunk boundary: an off-by-one in
    the chunk loop would leave a band of uninitialized `torch.empty` memory,
    which is garbage rather than an error."""
    import torch

    chunk = ho._PROJECTION_TOKEN_CHUNK
    x = torch.ones(1, chunk + 3, 4, dtype=torch.bfloat16)
    d = torch.eye(4, dtype=torch.float32)[:2]
    got = ho.project_activations(x, d)
    assert got.shape == (1, chunk + 3, 2)
    assert torch.equal(got, torch.ones(1, chunk + 3, 2))


# ---- straight-to-S3 storage path + uploader (ACS-278) ----------------------
#
# These are torch/boto3-free: the uploader orchestration takes an injected
# serializer + an s3 client, so a fake client (below) exercises the pool,
# per-object streaming, backpressure, and — critically — the no-data-loss
# invariant that the manifest is uploaded LAST and ONLY on a clean drain.



class _FakeS3:
    """Records uploads in call order; can be told to fail specific keys.

    Shared across the uploader's worker threads (real code builds one client per
    thread; a shared fake with a lock is equivalent for the test) — the ordered
    ``uploads`` list is the ground truth for "what landed, and in what order".
    """

    def __init__(self, fail_keys=frozenset()):
        self.uploads: list[tuple[str, bytes]] = []
        self._fail = set(fail_keys)
        self._lock = threading.Lock()

    def upload_fileobj(self, fileobj, bucket, key, Config=None):
        data = fileobj.read()  # read before failing, mirroring a real streamed PUT
        if key in self._fail:
            raise RuntimeError(f"injected upload failure for {key}")
        with self._lock:
            self.uploads.append((key, data))

    def keys(self) -> list[str]:
        return [k for k, _ in self.uploads]


def test_select_storage_backend_is_mutually_exclusive():
    up = ho.select_storage_backend(True)
    assert up == {"to_s3": True, "to_volume": False, "commit": False}
    smoke = ho.select_storage_backend(False)
    assert smoke == {"to_s3": False, "to_volume": True, "commit": True}
    # exactly one durable path per run — never both, never neither
    for plan in (up, smoke):
        assert plan["to_s3"] != plan["to_volume"]
        assert plan["commit"] == plan["to_volume"]  # commit belongs to the Volume path


def test_shard_key_compress_suffix_and_stats_key():
    assert ho.shard_key("r", 7) == "harvest/r/shard_00007.safetensors"
    # compressed shards are self-describing on the wire (.gz), so the downloader
    # gunzips before load_file — the manifest.compression field agrees.
    assert ho.shard_key("r", 7, compress=True) == "harvest/r/shard_00007.safetensors.gz"
    assert ho.stats_key("r") == "harvest/r/stats.safetensors"


def test_build_manifest_records_compression():
    m = ho.build_manifest(
        run_id="r", model_id="m", hf_repo="h", dtype="bfloat16",
        layer_indices=[1], prompts=["p"], shards=[],
    )
    assert m["compression"] == "none"
    mz = ho.build_manifest(
        run_id="r", model_id="m", hf_repo="h", dtype="bfloat16",
        layer_indices=[1], prompts=["p"], shards=[], compression="gzip",
    )
    assert mz["compression"] == "gzip"


def test_maybe_compress_is_noop_when_off_and_gzips_when_on():
    import gzip

    raw = b"activation-bytes" * 128
    assert ho._maybe_compress(raw, False) is raw  # off: exact same object, zero cost
    comp = ho._maybe_compress(raw, True)
    assert comp != raw
    assert gzip.decompress(comp) == raw  # lossless round-trip


def test_put_bytes_streams_exact_bytes_from_memory():
    fake = _FakeS3()
    n = ho._put_bytes(fake, "bucket", "some/key", b"hello world", transfer_config=None)
    assert n == 11
    assert fake.uploads == [("some/key", b"hello world")]


def test_s3_uploader_uploads_every_shard_and_records_metrics():
    fake = _FakeS3()
    up = ho.S3ShardUploader(
        s3_factory=lambda: fake, bucket="b", serialize=lambda p: p, concurrency=3
    )
    up.start()
    payloads = {ho.shard_key("run", i): b"x" * (100 + i) for i in range(7)}
    for k, v in payloads.items():
        up.submit(k, v)
    clean = up.drain(timeout=10)
    assert clean is True
    assert up.errors == []
    assert set(fake.keys()) == set(payloads)  # every shard landed exactly once
    assert len(fake.uploads) == len(payloads)
    assert up.bytes == sum(len(v) for v in payloads.values())
    assert up.upload_wall_s >= 0.0  # metric populated for MB/s reporting


def test_finalize_uploads_manifest_strictly_after_all_shards():
    fake = _FakeS3()
    up = ho.S3ShardUploader(
        s3_factory=lambda: fake, bucket="b", serialize=lambda p: p, concurrency=2
    )
    up.start()
    for i in range(4):
        up.submit(ho.shard_key("r", i), b"shard-data")

    def _put(k, b):
        fake.upload_fileobj(io.BytesIO(b), "b", k)

    ho.finalize_s3_run(
        up, _put,
        stats_key=ho.stats_key("r"), stats_bytes=b"STATS",
        manifest_key=ho.manifest_key("r"), manifest_bytes=b"MANIFEST",
    )
    keys = fake.keys()
    # manifest LAST, stats just before it, all shards before those two.
    assert keys[-1] == ho.manifest_key("r")
    assert keys[-2] == ho.stats_key("r")
    assert {ho.shard_key("r", i) for i in range(4)}.issubset(set(keys[:-2]))


def test_finalize_does_not_upload_manifest_when_a_shard_fails():
    """The no-data-loss invariant: a manifest referencing a missing shard is
    silent corruption for the downloader, so a failed shard must abort BEFORE the
    manifest is written — and on the upload path there's no Volume fallback."""
    failing = ho.shard_key("r", 2)
    fake = _FakeS3(fail_keys={failing})
    up = ho.S3ShardUploader(
        s3_factory=lambda: fake, bucket="b", serialize=lambda p: p, concurrency=2
    )
    up.start()
    for i in range(4):
        up.submit(ho.shard_key("r", i), b"shard-data")

    def _put(k, b):
        fake.upload_fileobj(io.BytesIO(b), "b", k)

    with pytest.raises(RuntimeError, match="re-run"):
        ho.finalize_s3_run(
            up, _put,
            stats_key=ho.stats_key("r"), stats_bytes=b"STATS",
            manifest_key=ho.manifest_key("r"), manifest_bytes=b"MANIFEST",
        )
    assert up.errors  # the failed shard was recorded
    assert ho.manifest_key("r") not in fake.keys()  # manifest NEVER written
    assert ho.stats_key("r") not in fake.keys()  # stats NEVER written


def test_s3_uploader_surfaces_a_dead_worker_without_deadlocking():
    """If the per-thread client can't be built, submit() must not hang forever and
    the error must surface at drain (no silently-lost shards)."""
    def _boom():
        raise RuntimeError("credentials rejected")

    up = ho.S3ShardUploader(
        s3_factory=_boom, bucket="b", serialize=lambda p: p, concurrency=2, maxsize=2
    )
    up.start()
    for i in range(3):
        up.submit(ho.shard_key("r", i), b"data")  # must not block indefinitely
    clean = up.drain(timeout=10)
    assert clean is True
    assert up.errors  # worker-init failure recorded


@needs_torch
def test_serialize_shard_roundtrips_in_memory_and_optionally_gzips():
    """serialize_shard must produce bytes that safetensors can load back — the
    whole point of streaming straight to S3 without a temp file."""
    import gzip

    import torch
    from safetensors.torch import load

    tensors = {
        "prompt_0": torch.arange(12, dtype=torch.bfloat16).reshape(2, 3, 2),
        "tokens_0": torch.tensor([1, 2, 3], dtype=torch.int32),
    }
    raw = ho.serialize_shard(tensors, compress=False)
    back = load(raw)
    assert torch.equal(back["tokens_0"], tensors["tokens_0"])
    assert back["prompt_0"].shape == (2, 3, 2)
    gz = ho.serialize_shard(tensors, compress=True)
    assert load(gzip.decompress(gz))["tokens_0"].tolist() == [1, 2, 3]
