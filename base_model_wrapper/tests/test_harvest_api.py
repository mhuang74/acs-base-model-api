"""End-to-end tests for the self-serve bulk-harvest endpoints (ACS-245).

``POST /v1/harvest`` validates + quota-gates a job, spawns the Modal harvest
function, and persists a ``harvest_jobs`` row; ``GET /v1/harvest/<id>`` polls
the job (lazily reconciling ``running`` rows with Modal) and returns the
result once done.

The ONLY thing faked is the modal_ops seam (``spawn_harvest`` /
``poll_harvest`` — the module boundary that owns all Modal interaction), so
these tests exercise the real auth, validation, quota SQL, and job
persistence. DB-gated like ``test_activation_routing.py`` (needs a migrated
Postgres via ``TEST_DATABASE_URL``).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from wrapper.routes import api as api_routes

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

dbtest = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a migrated Postgres to run harvest API tests",
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    os.environ["DATABASE_URL"] = TEST_DATABASE_URL or ""
    os.environ.setdefault("SESSION_SECRET", "test-harvest-secret")
    os.environ.setdefault("COOKIE_SECURE", "false")
    os.environ.setdefault("MODAL_BASE_URL", "https://upstream.example")
    os.environ.setdefault("VLLM_API_KEY", "vllm-test-key")
    os.environ.setdefault("ADMIN_TOKEN", "admin-test-token")
    os.environ.setdefault("SERVED_MODEL_NAME", "gpt2")
    # Tight caps so the validation tests stay small.
    os.environ["HARVEST_MAX_PROMPTS"] = "8"
    os.environ["HARVEST_MAX_EST_TOKENS"] = "1000"
    os.environ["HARVEST_MAX_RUNNING_PER_KEY"] = "1"
    # 2, not the shipped default of 3, so the small lane is genuinely >1 while
    # the tests that fill it stay short.
    os.environ["HARVEST_MAX_RUNNING_PER_KEY_SMALL"] = "2"
    os.environ["HARVEST_MAX_RUNNING_BIG_MODEL"] = "1"
    os.environ["HARVEST_MAX_BODY_BYTES"] = "100000"
    os.environ["MODELS_REGISTRY_JSON"] = json.dumps(
        {
            # Modal-backed model → harvest-capable (modal_app_name set).
            "harv": {
                "upstream_url": "https://harv-workbench.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
                "modal_app_name": "acs-harv",
                "n_layers": 32,  # enables per-model harvest layer-range check (ACS-342)
            },
            # Multi-GPU (8×H200) → subject to the GLOBAL big-model harvest cap.
            "harv-big": {
                "upstream_url": "https://harv-big-workbench.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "8xH200",
                "status": "live",
                "modal_app_name": "acs-harv-big",
            },
            # No modal_app_name → no harvest path (400 harvest_unsupported).
            "plain": {
                "upstream_url": "https://plain-workbench.example",
                "served_model_name": "gpt2",
                "tokenizer_repo": "gpt2",
                "gpu_shape_label": "test",
                "status": "live",
            },
        }
    )
    os.environ["DEFAULT_MODEL_ID"] = "harv"
    os.environ.pop("HF_TOKEN", None)

    from wrapper.main import app

    app.state.limiter.enabled = False
    with TestClient(app) as c:
        yield c
    app.state.limiter.enabled = True
    for var in (
        "MODELS_REGISTRY_JSON",
        "DEFAULT_MODEL_ID",
        "HARVEST_MAX_PROMPTS",
        "HARVEST_MAX_EST_TOKENS",
        "HARVEST_MAX_RUNNING_PER_KEY",
        "HARVEST_MAX_RUNNING_BIG_MODEL",
        "HARVEST_MAX_BODY_BYTES",
    ):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def _reset_longpoll_inflight():
    """Clear the in-process long-poll counter between tests (ACS-321).

    Same reasoning as ``_clean_harvest_jobs``: module-level state in a suite that
    reuses one process leaks, and a leaked count here surfaces as some unrelated
    test being unexpectedly declined — far worse to debug than the leak."""
    api_routes._LONGPOLL_INFLIGHT.clear()
    yield
    api_routes._LONGPOLL_INFLIGHT.clear()


@pytest.fixture(autouse=True)
def _clean_harvest_jobs():
    """Empty ``harvest_jobs`` before every test (ACS-326).

    Several tests assert on GLOBAL harvest state, not just their own rows: the
    big-model cap counts running jobs across all keys
    (``test_global_big_model_harvest_cap``), and ``test_spawn_unavailable_...``
    reads the latest-created row (``ORDER BY created_at DESC LIMIT 1``). Nothing
    cleans up, so on a reused database rows leaked by earlier tests — or by a
    prior run of this file — poison those globals and fail tests that have
    nothing to do with the change under test. Truncating before each test makes
    every case start from a known-empty table. Only ``harvest_jobs`` needs
    clearing: the fresh user + api_key each test creates never collide (and a
    broad CASCADE across shared tables mid-suite would be needless risk)."""
    if not TEST_DATABASE_URL:
        return

    from sqlalchemy import text

    from wrapper.db import make_engine, make_session_factory, session_scope

    async def _truncate() -> None:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                await s.execute(text("TRUNCATE TABLE harvest_jobs"))
        finally:
            await engine.dispose()

    asyncio.run(_truncate())


def _make_key(harvest_budget: int = 0) -> str:
    """Create a fresh user + API key; returns the bearer plaintext."""
    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.keys import generate as generate_key
    from wrapper.models import ApiKey, User

    async def _setup() -> str:
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                u = User(email=f"harvest-{uuid.uuid4().hex[:6]}@example.local")
                s.add(u)
                await s.flush()
                gk = generate_key()
                s.add(
                    ApiKey(
                        user_id=u.id,
                        key_hash=gk.hash_,
                        key_prefix=gk.prefix,
                        monthly_token_budget=0,  # unlimited
                        monthly_harvest_budget=harvest_budget,  # 0 = unlimited
                    )
                )
                return gk.plaintext
        finally:
            await engine.dispose()

    return asyncio.run(_setup())


def _key_row_ids(plaintext: str):
    """(key_id, user_id) for a bearer key created by _make_key."""
    from sqlalchemy import select

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.keys import hash_key
    from wrapper.models import ApiKey

    async def _fetch():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                row = (
                    await s.execute(select(ApiKey).where(ApiKey.key_hash == hash_key(plaintext)))
                ).scalar_one()
                return row.id, row.user_id
        finally:
            await engine.dispose()

    return asyncio.run(_fetch())


def _insert_job(
    plaintext: str,
    *,
    status: str,
    age_minutes: int = 0,
    call_id: str | None = None,
    model_id: str = "harv",
):
    """Insert a harvest_jobs row directly (bypassing the route); returns job id."""
    import datetime as dt
    import uuid as uuidmod

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import HarvestJob

    key_id, user_id = _key_row_ids(plaintext)

    async def _go():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                job = HarvestJob(
                    id=str(uuidmod.uuid4()),
                    key_id=key_id,
                    user_id=user_id,
                    model_id=model_id,
                    run_id=f"hv-{uuidmod.uuid4().hex[:12]}",
                    status=status,
                    modal_call_id=call_id,
                    params={"n_prompts": 1},
                    created_at=dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=age_minutes),
                )
                s.add(job)
                return job.id
        finally:
            await engine.dispose()

    return asyncio.run(_go())


def _job_row(job_id: str):
    from sqlalchemy import select

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import HarvestJob

    async def _fetch():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                return (
                    await s.execute(select(HarvestJob).where(HarvestJob.id == job_id))
                ).scalar_one_or_none()
        finally:
            await engine.dispose()

    return asyncio.run(_fetch())


def _stub_spawn(monkeypatch, call_id: str = "fc-test-123"):
    """Replace modal_ops.spawn_harvest, recording the kwargs it was called with."""
    from wrapper import modal_ops

    calls: list[dict] = []

    async def fake_spawn(
        *,
        app_name,
        model_id,
        prompts,
        layer_indices,
        shard_size,
        batch_size,
        run_id,
        project_onto=None,
        add_special_tokens=None,
    ):
        # add_special_tokens is keyword-only with a default because the route
        # forwards it ONLY when the caller explicitly set it (ACS-319) — omitted
        # otherwise, matching spawn_harvest's own None-means-don't-send contract.
        calls.append(
            {
                "app_name": app_name,
                "model_id": model_id,
                "prompts": prompts,
                "layer_indices": layer_indices,
                "shard_size": shard_size,
                "batch_size": batch_size,
                "run_id": run_id,
                "project_onto": project_onto,
                "add_special_tokens": add_special_tokens,
            }
        )
        return call_id

    monkeypatch.setattr(modal_ops, "spawn_harvest", fake_spawn)
    return calls


def _post(client, key: str, body: dict):
    return client.post("/v1/harvest", json=body, headers={"Authorization": f"Bearer {key}"})


def _get(client, key: str, job_id: str):
    return client.get(f"/v1/harvest/{job_id}", headers={"Authorization": f"Bearer {key}"})


# --- request validation ------------------------------------------------------


@dbtest
def test_unknown_model_rejected(client):
    key = _make_key()
    r = _post(client, key, {"model": "nope", "prompts": ["hi"]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_found"


@dbtest
def test_model_without_modal_app_rejected(client):
    key = _make_key()
    r = _post(client, key, {"model": "plain", "prompts": ["hi"]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "harvest_unsupported"


@dbtest
def test_empty_prompts_rejected(client):
    key = _make_key()
    for prompts in ([], ["ok", ""]):
        r = _post(client, key, {"model": "harv", "prompts": prompts})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_request"


@dbtest
def test_prompt_count_cap(client):
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["p"] * 9})  # cap is 8
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert "too many prompts" in body["message"]


@dbtest
def test_estimated_token_cap(client):
    key = _make_key()
    # 2 prompts x 2500 chars = 5000 chars → ~1250 est tokens > 1000 cap.
    r = _post(client, key, {"model": "harv", "prompts": ["x" * 2500] * 2})
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "invalid_request"
    assert "exceeds the per-job cap" in body["message"]


@dbtest
def test_bad_layers_rejected(client):
    key = _make_key()
    # negatives, empty list, non-int, above the 512 sanity bound, > 256 entries,
    # and strings other than the "all" sentinel
    for layers in ([-1], [], ["a"], [513], list(range(257)), "some", "ALL"):
        r = _post(client, key, {"model": "harv", "prompts": ["hi"], "layers": layers})
        assert r.status_code == 400, layers
        assert r.json()["error"]["code"] == "invalid_request"


@dbtest
def test_harvest_layers_out_of_range_rejected(client, monkeypatch):
    """ACS-342: a harvest ``layers`` index that is valid-shaped but out of range
    for THIS model is rejected pre-flight with a clean 400 and NEVER spawns a
    Modal job (no wasted GPU boot). ``harv`` has ``n_layers=32`` so 0-31 are
    valid. The schema caps at 512, so use [32] (the boundary, in [n_layers, 512])
    to exercise the per-model check rather than the schema's [0, 512] bound."""
    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    for bad in ([32], [0, 32], [45]):
        r = _post(client, key, {"model": "harv", "prompts": ["hi"], "layers": bad})
        assert r.status_code == 400, (bad, r.text)
        body = r.json()["error"]
        assert body["code"] == "invalid_request"
        assert "out of range" in body["message"]
    assert spawn_calls == []  # no Modal spawn for any out-of-range request


@dbtest
def test_harvest_in_range_layers_pass_through(client, monkeypatch):
    """A valid explicit ``layers`` list (all indices < n_layers) passes the
    per-model check and reaches spawn — the guard rejects only out-of-range
    indices."""
    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"], "layers": [0, 15, 31]})
    assert r.status_code == 202, r.text
    assert spawn_calls[0]["layer_indices"] == [0, 15, 31]


@dbtest
def test_layers_all_sentinel_passes_through(client, monkeypatch):
    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"], "layers": "all"})
    assert r.status_code == 202
    assert spawn_calls[0]["layer_indices"] == "all"


@dbtest
def test_batch_size_bounds(client):
    key = _make_key()
    for bs in (0, 65, -1):
        r = _post(client, key, {"model": "harv", "prompts": ["hi"], "batch_size": bs})
        assert r.status_code == 400, bs
        assert r.json()["error"]["code"] == "invalid_request"


@dbtest
def test_batch_size_default_none_passthrough(client, monkeypatch):
    """Omitted batch_size must reach spawn as None — the harvest function's own
    default applies (batched single-GPU / sequential multi-GPU), never a
    wrapper-invented number."""
    calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 202
    assert calls[0]["batch_size"] is None


@dbtest
def test_add_special_tokens_omitted_not_forwarded_to_spawn(client, monkeypatch):
    """ACS-319: when the caller omits add_special_tokens, the route must NOT pass
    it to spawn (None sentinel) — so the spawn stays byte-compatible with harvest
    apps deployed before harvest() gained the parameter. Its recorded metadata
    reflects the effective default (True)."""
    calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 202
    assert calls[0]["add_special_tokens"] is None  # not forwarded
    assert _job_row(r.json()["job_id"]).params["add_special_tokens"] is True


@dbtest
def test_add_special_tokens_forwarded_when_set(client, monkeypatch):
    """ACS-319: an explicit value is forwarded to spawn verbatim and recorded in
    the job params (metadata only, no prompt text)."""
    calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"], "add_special_tokens": False})
    assert r.status_code == 202
    assert calls[0]["add_special_tokens"] is False
    assert _job_row(r.json()["job_id"]).params["add_special_tokens"] is False


@dbtest
def test_oversized_body_rejected_413(client):
    key = _make_key()
    # Fixture caps HARVEST_MAX_BODY_BYTES at 100 kB; this body is ~150 kB.
    r = _post(client, key, {"model": "harv", "prompts": ["x" * 150_000]})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"


@dbtest
def test_unknown_field_rejected(client):
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"], "shard_sizes": 4})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


# --- quota gating ------------------------------------------------------------


@dbtest
def test_monthly_harvest_quota(client, monkeypatch):
    from wrapper import modal_ops

    _stub_spawn(monkeypatch)

    async def fake_poll(call_id):
        return "done", {"manifest_url": None}

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)

    key = _make_key(harvest_budget=1)
    r1 = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r1.status_code == 202
    # Reconcile job 1 to done so the second POST hits the MONTHLY quota, not
    # the concurrency cap (jobs count against the month regardless of status).
    assert _get(client, key, r1.json()["job_id"]).json()["status"] == "done"

    r2 = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r2.status_code == 429
    body = r2.json()["error"]
    assert body["code"] == "harvest_quota_exceeded"
    assert "1/1" in body["message"]


@dbtest
def test_running_job_concurrency_cap(client, monkeypatch):
    _stub_spawn(monkeypatch)
    key = _make_key()  # unlimited monthly budget
    # Small-model lane (ACS-321): the cap is 2 here, so the second submit is
    # allowed — that is the whole point of the lane — and the third is not.
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202

    r3 = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r3.status_code == 429
    body = r3.json()["error"]
    assert body["code"] == "harvest_concurrency_exceeded"
    assert "small-model" in body["message"]


def test_harvest_is_big_model_parses_gpu_shape():
    # The global big-model cap keys off the multi-GPU shape (parsed from
    # gpu_shape_label). Hermetic — no DB needed.
    from types import SimpleNamespace

    from wrapper.routes.api import _harvest_is_big_model

    def big(label):
        return _harvest_is_big_model(SimpleNamespace(gpu_shape_label=label))

    assert big("8xH200") is True
    assert big("8×H200") is True  # prod/dev-local use the × (U+00D7) form
    assert big("16xH200") is True
    assert big("1×L40S") is False
    assert big("1xL40S") is False
    assert big("test") is False  # unparseable → fail-open (small; never blocks)
    assert big("") is False


def test_resolve_harvest_app_derives_from_model_id():
    # ACS-273: derivation is keyed on the MODEL ID (matching harvest_offline's
    # acs-<MODEL_ID>-harvest), NOT modal_app_name — Trinity's serving app kept
    # the pre-rename name, so {modal_app_name}-harvest resolved the nonexistent
    # acs-trinity-base-harvest. Hermetic — no DB needed.
    from types import SimpleNamespace

    from wrapper.routes.api import _resolve_harvest_app

    # Trinity shape: aliased serving app, no explicit override → model-id name.
    trinity = SimpleNamespace(modal_app_name="acs-trinity-base", harvest_app_name=None)
    assert (
        _resolve_harvest_app(trinity, "trinity-truebase") == "acs-trinity-truebase-harvest"
    )
    # Explicit registry override still wins.
    trinity.harvest_app_name = "acs-custom-harvest"
    assert _resolve_harvest_app(trinity, "trinity-truebase") == "acs-custom-harvest"
    # No Modal serving app → unsupported (400 gate), not a derived name.
    plain = SimpleNamespace(modal_app_name=None, harvest_app_name=None)
    assert _resolve_harvest_app(plain, "plain") is None


@dbtest
def test_global_big_model_harvest_cap(client, monkeypatch):
    # Global cap of 1 (fixture env): one big-model job running → a DIFFERENT
    # key's big-model submit is rejected, even though that key is under its own
    # per-key cap. Small-model harvests stay uncapped.
    _stub_spawn(monkeypatch)
    key_a, key_b = _make_key(), _make_key()
    r1 = _post(client, key_a, {"model": "harv-big", "prompts": ["hi"]})
    assert r1.status_code == 202
    # key_b is under its own per-key cap (0 running) but the GLOBAL cap is full.
    r2 = _post(client, key_b, {"model": "harv-big", "prompts": ["hi"]})
    assert r2.status_code == 429
    assert r2.json()["error"]["code"] == "harvest_capacity_exceeded"
    # A SMALL-model submit from key_b is NOT blocked by the big-model cap.
    r3 = _post(client, key_b, {"model": "harv", "prompts": ["hi"]})
    assert r3.status_code == 202


@dbtest
def test_pending_job_occupies_concurrency_slot(client, monkeypatch):
    """A 'pending' row (inserted inside the gate transaction, spawn not yet
    confirmed) must hold a slot — that is what closes the count-then-insert
    race: a concurrent submit that passes the lock sees it and 429s."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    _insert_job(key, status="pending")
    _insert_job(key, status="pending")  # fills the 2-slot small lane
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "harvest_concurrency_exceeded"


@dbtest
def test_concurrent_submits_only_one_passes_cap(client, monkeypatch):
    """Two overlapping POSTs against a cap of 1 → exactly one 202, one 429.

    The FOR UPDATE lock on the caller's api_keys row serializes the gate:
    whichever transaction commits first inserts the pending row; the other
    blocks on the lock, then counts it. A slow stubbed spawn widens the old
    race window to make a regression loud."""
    import threading

    from wrapper import modal_ops

    async def slow_spawn(**kwargs):
        import asyncio as aio

        await aio.sleep(0.3)  # keep job 1 in 'pending' while job 2 races the gate
        return "fc-race-1"

    monkeypatch.setattr(modal_ops, "spawn_harvest", slow_spawn)
    key = _make_key()
    _insert_job(key, status="running")  # leaves exactly one free small-lane slot
    results: list[int] = []
    lock = threading.Lock()

    def submit():
        r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [202, 429], results


@dbtest
def test_stale_running_job_aged_out_and_does_not_block(client, monkeypatch):
    """A 'running' row older than the max harvest runtime is provably dead
    (Modal hard-kills at the function timeout): the submit path lazily fails it
    and the concurrency cap no longer counts it — no permanent deadlock."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    stale_id = _insert_job(key, status="running", age_minutes=200, call_id="fc-stale")

    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 202

    stale = _job_row(stale_id)
    assert stale.status == "failed"
    assert "maximum harvest runtime" in stale.error
    assert stale.completed_at is not None


# --- job read authorization --------------------------------------------------


@dbtest
def test_other_key_cannot_read_job(client, monkeypatch):
    _stub_spawn(monkeypatch)
    owner, stranger = _make_key(), _make_key()
    job_id = _post(client, owner, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]

    r = _get(client, stranger, job_id)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "harvest_job_not_found"
    # Same shape as a genuinely nonexistent id — existence must not leak.
    r_missing = _get(client, stranger, "no-such-job")
    assert r_missing.status_code == 404
    assert r_missing.json()["error"]["code"] == "harvest_job_not_found"


# --- POST → GET happy path (modal_ops seam stubbed) --------------------------


@dbtest
def test_submit_then_poll_to_done(client, monkeypatch):
    from wrapper import modal_ops

    spawn_calls = _stub_spawn(monkeypatch, call_id="fc-happy-1")

    result_dict = {
        "manifest_url": "https://bucket.example/hv/manifest.json",
        "shard_urls": ["https://bucket.example/hv/shard_00000.safetensors"],
        "stats_url": "https://bucket.example/hv/stats.json",
        "volume_path": "/harvest/hv-abc",
        "timings": {"total_s": 12.5},
    }
    poll_results = [("running", None), ("done", result_dict)]

    async def fake_poll(call_id):
        assert call_id == "fc-happy-1"
        return poll_results.pop(0)

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)

    key = _make_key()
    r = _post(
        client,
        key,
        {
            "model": "harv",
            "prompts": ["the cat sat", "on the mat"],
            "layers": [8, 16],
            "shard_size": 2,
            "batch_size": 4,
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"
    assert body["model"] == "harv"
    assert body["n_prompts"] == 2
    job_id = body["job_id"]

    # The spawn got the validated params verbatim + a hv- run id, aimed at the
    # app derived from the registry's modal_app_name.
    assert len(spawn_calls) == 1
    call = spawn_calls[0]
    assert call["app_name"] == "acs-harv-harvest"
    assert call["model_id"] == "harv"
    assert call["prompts"] == ["the cat sat", "on the mat"]
    assert call["layer_indices"] == [8, 16]
    assert call["shard_size"] == 2
    assert call["batch_size"] == 4
    assert call["run_id"].startswith("hv-") and len(call["run_id"]) == 15

    # First poll: Modal says still running → returned as-is, nothing persisted.
    g1 = _get(client, key, job_id)
    assert g1.status_code == 200
    assert g1.json()["status"] == "running"
    assert "result" not in g1.json()

    # Second poll: finished → result persisted on the row and returned, with
    # the presigned-URL freshness window anchored on completed_at.
    g2 = _get(client, key, job_id)
    assert g2.status_code == 200
    body2 = g2.json()
    assert body2["status"] == "done"
    assert body2["result"] == result_dict
    assert body2["completed_at"] is not None
    assert body2["urls_expire_at"] > body2["completed_at"]
    assert "urls_expired" not in body2  # fresh URLs carry no expired flag

    row = _job_row(job_id)
    assert row.status == "done"
    assert row.result == result_dict
    assert row.modal_call_id == "fc-happy-1"
    assert row.completed_at is not None
    # Privacy: params hold the prompt COUNT and sizes, never prompt text.
    assert row.params["n_prompts"] == 2
    assert "prompts" not in row.params

    # Terminal rows are served from Postgres — no further Modal polls.
    g3 = _get(client, key, job_id)
    assert g3.json()["status"] == "done"
    assert poll_results == []  # both stub results consumed, none left / needed


@dbtest
def test_poll_failure_marks_job_failed(client, monkeypatch):
    from wrapper import modal_ops

    _stub_spawn(monkeypatch)

    # What modal_ops.classify_poll_exception hands back for a remote raise —
    # class name only, no raw exception text.
    safe_msg = "job failed on Modal (RuntimeError); details in server logs"

    async def fake_poll(call_id):
        return "failed", safe_msg

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)

    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    g = _get(client, key, job_id)
    assert g.status_code == 200
    assert g.json()["status"] == "failed"
    assert g.json()["error"] == safe_msg
    row = _job_row(job_id)
    assert row.status == "failed"
    assert row.completed_at is not None


@dbtest
def test_poll_route_survives_unexpected_poll_error(client, monkeypatch):
    """If poll_harvest itself raises (wrapper bug / unclassified infra state),
    the GET must serve the stored row, not 500 — and the job stays running."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch)

    async def exploding_poll(call_id):
        raise RuntimeError("totally unexpected")

    monkeypatch.setattr(modal_ops, "poll_harvest", exploding_poll)

    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    g = _get(client, key, job_id)
    assert g.status_code == 200
    assert g.json()["status"] == "running"
    assert _job_row(job_id).status == "running"


@dbtest
def test_spawn_unavailable_returns_503_failed_job_no_quota_burn(client, monkeypatch):
    from wrapper import modal_ops

    async def failing_spawn(**kwargs):
        raise modal_ops.HarvestUnavailableError("harvest app 'acs-harv-harvest' is not deployed")

    monkeypatch.setattr(modal_ops, "spawn_harvest", failing_spawn)

    key = _make_key(harvest_budget=1)
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 503
    body = r.json()["error"]
    assert body["code"] == "harvest_unavailable"
    # Honest message: the job row exists (failed) — never "no job was started".
    assert "recorded" in body["message"] and "failed" in body["message"]

    # The pending row was flipped to failed with a sanitized error (class name
    # only) and no call id — pollable, but excluded from the monthly tally.
    from sqlalchemy import select

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import HarvestJob

    async def _fetch_failed():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                return (
                    await s.execute(
                        select(HarvestJob).order_by(HarvestJob.created_at.desc()).limit(1)
                    )
                ).scalar_one()
        finally:
            await engine.dispose()

    row = asyncio.run(_fetch_failed())
    assert row.status == "failed"
    assert row.modal_call_id is None
    assert row.error == "harvest spawn failed (HarvestUnavailableError); see server logs"
    # And it's readable through the API like any other job.
    g = _get(client, key, row.id)
    assert g.status_code == 200
    assert g.json()["status"] == "failed"

    # budget=1 but the never-spawned failure doesn't count → a retry reaches
    # the spawn again (503), it is NOT rejected with 429 harvest_quota_exceeded.
    r2 = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r2.status_code == 503


@dbtest
def test_terminal_status_never_overwritten(client, monkeypatch):
    """The reconcile UPDATE is guarded WHERE status='running': applying a
    'failed' outcome to a row that a racing poll already committed as 'done'
    must be a no-op."""
    import datetime as dt

    from sqlalchemy import update

    from wrapper.db import make_engine, make_session_factory, session_scope
    from wrapper.models import HarvestJob

    _stub_spawn(monkeypatch)
    key = _make_key()
    job_id = _insert_job(key, status="done", call_id="fc-done")

    async def _guarded_fail():
        engine = make_engine(TEST_DATABASE_URL)
        try:
            factory = make_session_factory(engine)
            async with session_scope(factory) as s:
                now = dt.datetime.now(tz=dt.UTC)
                result = await s.execute(
                    update(HarvestJob)
                    .where(HarvestJob.id == job_id, HarvestJob.status == "running")
                    .values(status="failed", error="late loser", completed_at=now)
                )
                return result.rowcount
        finally:
            await engine.dispose()

    assert asyncio.run(_guarded_fail()) == 0
    row = _job_row(job_id)
    assert row.status == "done"
    assert row.error is None


# --- poll-exception classification (pure, no DB) -----------------------------


def test_classify_poll_exception():
    """Transient infra → running (retry next poll); only function-level
    failures are terminal; client-safe strings carry the class name at most."""
    import grpclib.exceptions
    import modal.exception
    from grpclib.const import Status as GrpcStatus

    from wrapper.modal_ops import classify_poll_exception

    # Still executing — modal's own TimeoutError (NOT the builtin).
    assert classify_poll_exception(modal.exception.TimeoutError("x")) == ("running", None)
    assert classify_poll_exception(TimeoutError()) == ("running", None)
    # Transient infrastructure → running.
    grpc_exc = grpclib.exceptions.GRPCError(GrpcStatus.UNAVAILABLE, "upstream unavailable")
    assert classify_poll_exception(grpc_exc) == ("running", None)
    assert classify_poll_exception(grpclib.exceptions.StreamTerminatedError()) == (
        "running",
        None,
    )
    assert classify_poll_exception(ConnectionResetError("peer reset")) == ("running", None)
    assert classify_poll_exception(modal.exception.ClientClosed()) == ("running", None)
    # Terminal: remote execution timeout / expired output.
    status, msg = classify_poll_exception(modal.exception.FunctionTimeoutError("t"))
    assert status == "failed" and "timeout" in msg
    status, msg = classify_poll_exception(modal.exception.OutputExpiredError("e"))
    assert status == "failed" and "expired" in msg
    # Terminal: a deserialized remote raise (arbitrary class) — class name only,
    # never the raw exception text.
    status, msg = classify_poll_exception(ValueError("CUDA secret /path/leak"))
    assert status == "failed"
    assert "ValueError" in msg
    assert "CUDA" not in msg and "leak" not in msg


def _projection_codec(n_dirs=2, hidden=8, dtype="float32"):
    """A well-formed project_onto payload (float32, uncompressed)."""
    import base64
    import struct

    vals = [float(i + 1) for i in range(n_dirs * hidden)]
    raw = struct.pack(f"<{len(vals)}f", *vals)
    return {
        "data": base64.b64encode(raw).decode(),
        "dtype": dtype,
        "shape": [n_dirs, hidden],
        "compression": "none",
    }


@dbtest
def test_project_onto_is_forwarded_to_the_harvester(client, monkeypatch):
    """The whole feature is the payload reaching Modal. Accepting the request
    and then dropping the directions would return raw activations under a
    manifest that says nothing is wrong — so assert the handoff, not the 202."""
    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    codec = _projection_codec()
    r = _post(
        client,
        key,
        {"model": "harv", "prompts": ["hi"], "layers": [1], "project_onto": codec},
    )
    assert r.status_code == 202, r.text
    assert spawn_calls[0]["project_onto"] == codec


@dbtest
def test_harvest_without_projection_forwards_none(client, monkeypatch):
    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    r = _post(client, key, {"model": "harv", "prompts": ["hi"], "layers": [1]})
    assert r.status_code == 202
    assert spawn_calls[0]["project_onto"] is None


@dbtest
@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda c: c.update(data="not base64!!"), "valid base64"),
        (lambda c: c.update(shape=[2, 9]), "bytes"),
        (lambda c: c.update(dtype="float16"), "bytes"),
        (lambda c: c.update(compression="zstd"), "compression"),
        (lambda c: c.update(shape=[0, 8]), "1..64"),
    ],
)
def test_malformed_projection_is_rejected_at_the_boundary(client, mutate, msg):
    """Rejected here with a readable message, not on the GPU — a remote
    ValueError comes back as an opaque 'job failed on Modal'."""
    key = _make_key()
    codec = _projection_codec()
    mutate(codec)
    r = _post(
        client,
        key,
        {"model": "harv", "prompts": ["hi"], "layers": [1], "project_onto": codec},
    )
    # 400, not 422: the loud-fail envelope translates validation errors.
    assert r.status_code == 400, r.text
    assert msg in r.text


@dbtest
def test_compressed_projection_says_compression_not_byte_count(client):
    """A client who genuinely zstd-compressed their directions has a payload of
    the wrong length too. Checking the length first would answer a question they
    didn't ask ("needs 1048576 bytes") and send them off to debug their shape."""
    key = _make_key()
    codec = _projection_codec()
    import base64

    codec["compression"] = "zstd"
    codec["data"] = base64.b64encode(b"\x28\xb5\x2f\xfd" + b"\x00" * 40).decode()
    r = _post(
        client,
        key,
        {"model": "harv", "prompts": ["hi"], "layers": [1], "project_onto": codec},
    )
    assert r.status_code == 400, r.text
    assert "compression is not supported" in r.text
    assert "bytes" not in r.text


@dbtest
def test_line_wrapped_projection_base64_is_accepted(client, monkeypatch):
    """Shell pipelines and `base64.encodebytes` wrap at 76 columns; that is a
    formatting detail, not a malformed payload."""
    import base64
    import struct

    spawn_calls = _stub_spawn(monkeypatch)
    key = _make_key()
    raw = struct.pack("<16f", *[float(i) for i in range(16)])
    codec = {
        "data": base64.encodebytes(raw).decode(),  # newline-wrapped
        "dtype": "float32",
        "shape": [2, 8],
        "compression": "none",
    }
    assert "\n" in codec["data"]
    r = _post(
        client,
        key,
        {"model": "harv", "prompts": ["hi"], "layers": [1], "project_onto": codec},
    )
    assert r.status_code == 202, r.text
    # Forwarded verbatim — the harvester does the same whitespace strip.
    assert spawn_calls[0]["project_onto"]["data"] == codec["data"]

# --- ACS-321: per-key concurrency lanes -------------------------------------


@dbtest
def test_running_big_job_does_not_block_a_small_submit(client, monkeypatch):
    """The complaint this fixes: an interactive 8-prompt job queued behind a
    corpus run. The lanes are counted independently, so a full big-model lane
    leaves the small lane untouched."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    # Fill the big lane past BOTH caps, so a shared counter (the pre-ACS-321
    # flat behaviour) would block the small submit below. With a single running
    # big job this test passes even with the lanes removed — the small cap alone
    # would admit it, which is not what the test claims to prove.
    for _ in range(3):
        _insert_job(key, status="running", model_id="harv-big")
    assert _post(client, key, {"model": "harv-big", "prompts": ["hi"]}).status_code == 429
    # ...but the small lane is empty, so the interactive job goes through.
    r3 = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r3.status_code == 202, r3.text


@dbtest
def test_full_small_lane_does_not_block_a_big_submit(client, monkeypatch):
    """The converse: cheap jobs must not consume the expensive lane's budget."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    _insert_job(key, status="running", model_id="harv")
    _insert_job(key, status="running", model_id="harv")  # small lane full (cap 2)
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 429
    r = _post(client, key, {"model": "harv-big", "prompts": ["hi"]})
    assert r.status_code == 202, r.text


@dbtest
def test_big_lane_message_names_the_lane(client, monkeypatch):
    _stub_spawn(monkeypatch)
    key = _make_key()
    assert _post(client, key, {"model": "harv-big", "prompts": ["hi"]}).status_code == 202
    body = _post(client, key, {"model": "harv-big", "prompts": ["hi"]}).json()["error"]
    assert body["code"] == "harvest_concurrency_exceeded"
    assert "large-model" in body["message"]


@dbtest
def test_retired_model_job_still_holds_a_small_lane_slot(client, monkeypatch):
    """A job whose model has left the registry is in neither id set. It must
    still consume a slot — otherwise retiring a model silently hands every key
    that was running one an extra concurrent job."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    _insert_job(key, status="running", model_id="harv")
    _insert_job(key, status="running", model_id="model-retired-last-week")
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 429, r.text
    assert r.json()["error"]["code"] == "harvest_concurrency_exceeded"


# --- ACS-321: long-poll on GET /v1/harvest/<id> ------------------------------


@dbtest
def test_wait_returns_as_soon_as_the_job_finishes(client, monkeypatch):
    """The point of ?wait= is that the client stops writing polling loops. It
    must return the moment the job is terminal, not after the full budget."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-wait-1")
    polls = {"n": 0}
    result_dict = {"run_id": "hv-wait", "shard_urls": [], "timings": {"total_s": 1.0}}

    async def fake_poll(call_id):
        polls["n"] += 1
        return ("done", result_dict) if polls["n"] >= 3 else ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)
    # Don't actually sleep between reconciles — assert the loop, not the clock.
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(api_routes.asyncio, "sleep", fake_sleep)

    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    r = client.get(
        f"/v1/harvest/{job_id}?wait=30", headers={"Authorization": f"Bearer {key}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done", body
    assert body["result"]["run_id"] == "hv-wait"
    assert polls["n"] == 3, "should have kept polling until terminal"
    assert len(slept) == 2, "one sleep between each reconcile, none after the last"


@dbtest
def test_wait_gives_up_at_the_deadline_and_returns_running(client, monkeypatch):
    """A job that outlives the budget is not an error — the client gets the
    same 'running' payload an immediate poll would have given."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-wait-2")

    async def never_done(call_id):
        return ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", never_done)

    async def fake_sleep(s):
        return None

    monkeypatch.setattr(api_routes.asyncio, "sleep", fake_sleep)
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    r = client.get(
        f"/v1/harvest/{job_id}?wait=1", headers={"Authorization": f"Bearer {key}"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "running"


@dbtest
def test_wait_zero_is_the_default_and_polls_once(client, monkeypatch):
    """Backwards compatibility: no wait param == exactly one reconcile."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-wait-3")
    polls = {"n": 0}

    async def fake_poll(call_id):
        polls["n"] += 1
        return ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    r = _get(client, key, job_id)
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    assert polls["n"] == 1


@dbtest
def test_wait_beyond_the_maximum_is_rejected(client, monkeypatch):
    """An unbounded wait would pin a worker and a pooled connection for as long
    as the caller likes."""
    _stub_spawn(monkeypatch, call_id="fc-wait-4")
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    for bad in (api_routes.HARVEST_MAX_WAIT_S + 1, -1):
        r = client.get(
            f"/v1/harvest/{job_id}?wait={bad}",
            headers={"Authorization": f"Bearer {key}"},
        )
        # 400 in this API's error envelope, not FastAPI's bare 422 `detail`
        # shape — /v1 answers every bad request the same way.
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "invalid_request"
        assert "wait must be between 0 and 60" in r.json()["error"]["message"]


@dbtest
def test_wait_releases_the_db_connection_between_polls(client, monkeypatch):
    """The pool is SQLAlchemy's default 5+10. If a long-poll held its connection
    for the whole wait, a handful of them would starve every other request in
    the app — including the login page. Assert the session is closed around each
    sleep rather than trusting a comment."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-wait-5")
    polls = {"n": 0}

    async def fake_poll(call_id):
        polls["n"] += 1
        return ("done", {"run_id": "x"}) if polls["n"] >= 3 else ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)

    events: list[str] = []
    real_close = AsyncSession.close

    async def spy_close(self):
        events.append("close")
        return await real_close(self)

    async def fake_sleep(s):
        events.append("sleep")

    monkeypatch.setattr(AsyncSession, "close", spy_close)
    monkeypatch.setattr(api_routes.asyncio, "sleep", fake_sleep)

    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    r = client.get(
        f"/v1/harvest/{job_id}?wait=30", headers={"Authorization": f"Bearer {key}"}
    )
    assert r.status_code == 200
    # Every sleep must be immediately preceded by a close.
    for i, ev in enumerate(events):
        if ev == "sleep":
            assert i > 0 and events[i - 1] == "close", events
    assert events.count("sleep") == 2, events


# --- Retry-After on the retryable harvest 429s (ACS-344) ---------------------


def _delete(client, key: str, job_id: str):
    return client.delete(
        f"/v1/harvest/{job_id}", headers={"Authorization": f"Bearer {key}"}
    )


@dbtest
def test_concurrency_429_carries_retry_after(client, monkeypatch):
    """The per-key concurrency 429 now advertises Retry-After so a client paces
    its retries instead of hammering (ACS-344 — a tester saw 44 un-paced 429s
    over 915s)."""
    _stub_spawn(monkeypatch)
    key = _make_key()
    # Fill the 2-slot small lane, then trip the cap.
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202
    r = _post(client, key, {"model": "harv", "prompts": ["hi"]})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "harvest_concurrency_exceeded"
    assert r.headers.get("Retry-After") == "30"


@dbtest
def test_capacity_429_carries_retry_after(client, monkeypatch):
    """The global big-model capacity 429 (transient) also advertises Retry-After."""
    _stub_spawn(monkeypatch)
    key_a, key_b = _make_key(), _make_key()
    assert _post(client, key_a, {"model": "harv-big", "prompts": ["hi"]}).status_code == 202
    r = _post(client, key_b, {"model": "harv-big", "prompts": ["hi"]})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "harvest_capacity_exceeded"
    assert r.headers.get("Retry-After") == "30"


# --- DELETE /v1/harvest/{job_id} cancel (ACS-344) ----------------------------


def _stub_cancel(monkeypatch, *, returns: bool = True):
    """Replace modal_ops.cancel_harvest, recording the call ids it was asked to
    cancel. Returns the recording list."""
    from wrapper import modal_ops

    seen: list[str] = []

    async def fake_cancel(call_id):
        seen.append(call_id)
        return returns

    monkeypatch.setattr(modal_ops, "cancel_harvest", fake_cancel)
    return seen


@dbtest
def test_cancel_running_job_frees_slot(client, monkeypatch):
    """Cancelling a running job returns 200 cancelled, signals Modal to stop,
    and frees the concurrency slot so the next submit succeeds."""
    _stub_spawn(monkeypatch)
    cancels = _stub_cancel(monkeypatch)
    key = _make_key()
    # Fill the 2-slot small lane with real running jobs (via the route).
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    # Lane is full → a third submit 429s.
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 429

    r = _delete(client, key, job_id)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled"
    assert body["job_id"] == job_id
    # The stubbed spawn returns "fc-test-123" — the route asked Modal to stop it.
    assert cancels == ["fc-test-123"]
    # Row is terminal in Postgres.
    assert _job_row(job_id).status == "cancelled"
    # Slot freed → the previously-rejected submit now goes through.
    assert _post(client, key, {"model": "harv", "prompts": ["hi"]}).status_code == 202


@dbtest
def test_cancel_pending_job_no_modal_call(client, monkeypatch):
    """A still-pending job (spawn not confirmed, no call id) cancels without any
    Modal round trip."""
    cancels = _stub_cancel(monkeypatch)
    key = _make_key()
    job_id = _insert_job(key, status="pending")  # no call_id
    r = _delete(client, key, job_id)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert cancels == []  # no modal_call_id → nothing to cancel upstream


@dbtest
def test_cancel_unknown_job_404(client, monkeypatch):
    import uuid as uuidmod

    _stub_cancel(monkeypatch)
    key = _make_key()
    r = _delete(client, key, str(uuidmod.uuid4()))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "harvest_job_not_found"


@dbtest
def test_cancel_other_keys_job_404(client, monkeypatch):
    """A job owned by another user's key is indistinguishable from nonexistent."""
    _stub_cancel(monkeypatch)
    owner, other = _make_key(), _make_key()
    job_id = _insert_job(owner, status="running", call_id="fc-owner")
    r = _delete(client, other, job_id)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "harvest_job_not_found"
    # Untouched — still running for the owner.
    assert _job_row(job_id).status == "running"


@dbtest
def test_cancel_terminal_job_409(client, monkeypatch):
    """A job already in a terminal state cannot be cancelled (409)."""
    cancels = _stub_cancel(monkeypatch)
    key = _make_key()
    for terminal in ("done", "failed", "cancelled"):
        job_id = _insert_job(key, status=terminal, call_id="fc-x")
        r = _delete(client, key, job_id)
        assert r.status_code == 409, terminal
        assert r.json()["error"]["code"] == "harvest_not_cancellable"
    assert cancels == []  # never signalled Modal for an already-terminal job


@dbtest
def test_cancel_best_effort_modal_failure_still_cancels(client, monkeypatch):
    """If the Modal terminate RPC fails, the DB is authoritative: the job is
    still cancelled and the slot freed; only the container may linger."""
    _stub_spawn(monkeypatch)
    _stub_cancel(monkeypatch, returns=False)  # simulate a failed terminate
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    r = _delete(client, key, job_id)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert _job_row(job_id).status == "cancelled"


def test_cancel_harvest_swallows_upstream_errors(monkeypatch):
    """modal_ops.cancel_harvest never raises — a failing terminate RPC returns
    False so the route can rely on the DB being authoritative. Hermetic."""
    from wrapper import modal_ops

    monkeypatch.setattr(modal_ops, "_auth", lambda: None)

    def boom(call_id):
        raise RuntimeError("control-plane blip")

    monkeypatch.setattr(modal_ops, "_cancel_harvest_sync", boom)
    assert asyncio.run(modal_ops.cancel_harvest("fc-x")) is False


def test_cancel_harvest_returns_false_when_auth_fails(monkeypatch):
    """A missing-credentials ``_auth()`` must classify as best-effort False, not
    bubble out and 500 the DELETE route after the job is already cancelled.
    ``_auth()`` runs INSIDE cancel_harvest's try for exactly this reason."""
    from wrapper import modal_ops
    from wrapper.modal_ops import ModalOpsError

    def no_creds():
        raise ModalOpsError("MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set")

    monkeypatch.setattr(modal_ops, "_auth", no_creds)
    assert asyncio.run(modal_ops.cancel_harvest("fc-x")) is False


# --- Wall-clock accounting on the done payload (ACS-343) ----------------------


@dbtest
def test_done_payload_surfaces_wall_clock_accounting(client, monkeypatch):
    """A done job attributes wall clock vs. in-container work so a long wall
    clock on trivial work is explicable (ACS-343). Mirrors the reported case:
    load_s=0, ~1.6s of timings, a multi-minute wall clock — nearly all of it
    unaccounted (Modal queue/scheduling, not GPU work). ``forward_tok_s`` is a
    RATE and must NOT be summed into in_container_s."""
    from wrapper import modal_ops

    key = _make_key()
    # A running job created 5 minutes ago (created_at anchors wall_clock_s).
    job_id = _insert_job(key, status="running", age_minutes=5, call_id="fc-acct")
    result_dict = {
        "shard_urls": [],
        "timings": {
            "load_s": 0.0,
            "forward_s": 1.59,
            "save_s": 0.0,
            "commit_s": 0.0,
            "upload_s": 0.0,
            "forward_tok_s": 13.2,  # a RATE — must be excluded from the sum
            "tokens": 21,
        },
    }

    async def fake_poll(call_id):
        assert call_id == "fc-acct"
        return "done", result_dict

    monkeypatch.setattr(modal_ops, "poll_harvest", fake_poll)

    body = _get(client, key, job_id).json()
    assert body["status"] == "done"
    acct = body["accounting"]
    # Only the five phase timings are summed — the rate is excluded.
    assert acct["in_container_s"] == 1.59
    # created_at was 5 min ago; completed_at is ~now → a ~300s wall clock.
    assert acct["wall_clock_s"] >= 290
    # ~99% of the wall clock is unaccounted — the ACS-343 signature.
    assert acct["unaccounted_s"] == round(acct["wall_clock_s"] - 1.59, 2)
    assert acct["unaccounted_s"] > 250


# --- ACS-321 follow-up: long-poll admission control --------------------------


@dbtest
def test_long_polls_are_capped_per_key_and_answered_early(client, monkeypatch):
    """`poll_harvest` runs on the event loop's DEFAULT thread pool, shared with
    logprob serialization, zstd and spawn_harvest — so unbounded waiters add
    seconds of latency to /v1/completions traffic. Over the ceiling we answer
    early rather than 429, because an early status poll is within contract while
    a 429 could break a client's loop."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-cap-1")

    async def never_done(call_id):
        return ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", never_done)

    async def fake_sleep(s):
        return None

    # Shrink the pacing interval rather than stubbing asyncio.sleep: stubbing it
    # made "was it paced at all?" invisible, so deleting the sleep passed.
    monkeypatch.setattr(api_routes, "HARVEST_WAIT_POLL_INTERVAL_S", 1.0)
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    key_id = _key_row_ids(key)[0]
    settings = client.app.state.settings
    api_routes._LONGPOLL_INFLIGHT[key_id] = settings.harvest_max_longpoll_per_key

    started = time.monotonic()
    r = client.get(
        f"/v1/harvest/{job_id}?wait=30", headers={"Authorization": f"Bearer {key}"}
    )
    elapsed = time.monotonic() - started

    assert r.status_code == 200
    assert r.json()["status"] == "running"
    assert r.headers.get("X-Acs-Longpoll") == "declined"
    assert r.headers.get("Retry-After") == "1"
    # Paced (not instant) — an instant answer turns a declined client into a hot
    # loop — but far short of the 30s budget it asked for.
    assert 0.9 <= elapsed < 5, elapsed


@dbtest
def test_a_waiting_request_occupies_a_slot_while_it_waits(client, monkeypatch):
    """Exercises the INCREMENT, which the hand-seeded cap test never touches:
    deleting it removes the ceiling entirely and leaves the suite green.
    Observed from inside the reconcile — the only moment the count is non-zero."""
    from wrapper import modal_ops

    _stub_spawn(monkeypatch, call_id="fc-cap-2")

    async def never_done(call_id):
        return ("running", None)

    monkeypatch.setattr(modal_ops, "poll_harvest", never_done)
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    key_id = _key_row_ids(key)[0]

    seen: list[int] = []
    real_reconcile = api_routes._reconcile_harvest_job_until

    async def spy(*a, **kw):
        seen.append(api_routes._LONGPOLL_INFLIGHT.get(key_id, 0))
        return await real_reconcile(*a, **kw)

    async def fake_sleep(s):
        return None

    monkeypatch.setattr(api_routes, "_reconcile_harvest_job_until", spy)
    monkeypatch.setattr(api_routes.asyncio, "sleep", fake_sleep)
    r = client.get(
        f"/v1/harvest/{job_id}?wait=5", headers={"Authorization": f"Bearer {key}"}
    )
    assert r.status_code == 200
    assert r.headers.get("X-Acs-Longpoll") is None, "should have been admitted"
    assert seen == [1], f"slot not held during the wait: {seen}"
    assert key_id not in api_routes._LONGPOLL_INFLIGHT, "slot not released"


@dbtest
def test_long_poll_slot_is_released_even_when_the_reconcile_raises(client, monkeypatch):
    """A leaked slot is permanent: the key would be locked out of long-polling
    for the process's lifetime, and the global ceiling would erode with it."""
    _stub_spawn(monkeypatch, call_id="fc-cap-3")
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    key_id = _key_row_ids(key)[0]

    async def exploding_reconcile(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_routes, "_reconcile_harvest_job_until", exploding_reconcile)
    with pytest.raises(Exception):
        client.get(
            f"/v1/harvest/{job_id}?wait=30", headers={"Authorization": f"Bearer {key}"}
        )
    assert key_id not in api_routes._LONGPOLL_INFLIGHT, api_routes._LONGPOLL_INFLIGHT


@dbtest
def test_wait_zero_never_occupies_a_long_poll_slot(client, monkeypatch):
    """Ordinary polling must not be able to exhaust the long-poll ceiling."""
    _stub_spawn(monkeypatch, call_id="fc-cap-4")
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    before = dict(api_routes._LONGPOLL_INFLIGHT)
    r = _get(client, key, job_id)
    assert r.status_code == 200
    assert r.headers.get("X-Acs-Longpoll") is None
    assert dict(api_routes._LONGPOLL_INFLIGHT) == before


@dbtest
def test_the_global_long_poll_ceiling_declines_other_keys(client, monkeypatch):
    """The per-key cap protects a key from itself; THIS one protects
    /v1/completions from everyone else's keys — it is the bound the thread-pool
    argument rests on, and nothing else exercises it."""
    _stub_spawn(monkeypatch, call_id="fc-cap-5")
    monkeypatch.setattr(api_routes, "HARVEST_WAIT_POLL_INTERVAL_S", 0.05)
    settings = client.app.state.settings

    # Fill the GLOBAL ceiling with other keys, each below its own per-key cap.
    others = [uuid.uuid4() for _ in range(settings.harvest_max_longpoll_total)]
    for k in others:
        api_routes._LONGPOLL_INFLIGHT[k] = 1

    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    key_id = _key_row_ids(key)[0]
    assert api_routes._LONGPOLL_INFLIGHT.get(key_id, 0) == 0, "this key is at 0 of its own cap"

    r = client.get(
        f"/v1/harvest/{job_id}?wait=30", headers={"Authorization": f"Bearer {key}"}
    )
    assert r.status_code == 200
    assert r.headers.get("X-Acs-Longpoll") == "declined", (
        "under its per-key cap but over the global one — must still be declined"
    )


@dbtest
def test_declined_wait_is_never_held_longer_than_requested(client, monkeypatch):
    """`?wait=1` over the cap must not take 3s. A caller who asks for a short
    beat and sets a matching client timeout would start timing out exactly when
    the service is busiest — strictly worse than not offering the feature."""
    _stub_spawn(monkeypatch, call_id="fc-cap-6")
    monkeypatch.setattr(api_routes, "HARVEST_WAIT_POLL_INTERVAL_S", 3.0)
    key = _make_key()
    job_id = _post(client, key, {"model": "harv", "prompts": ["hi"]}).json()["job_id"]
    key_id = _key_row_ids(key)[0]
    settings = client.app.state.settings
    api_routes._LONGPOLL_INFLIGHT[key_id] = settings.harvest_max_longpoll_per_key

    started = time.monotonic()
    r = client.get(
        f"/v1/harvest/{job_id}?wait=1", headers={"Authorization": f"Bearer {key}"}
    )
    elapsed = time.monotonic() - started
    assert r.status_code == 200
    assert r.headers.get("X-Acs-Longpoll") == "declined"
    assert elapsed < 2.0, f"held {elapsed:.2f}s for a wait=1 request"
    assert r.headers.get("Retry-After") == "1"


@dbtest
def test_the_new_response_header_is_exposed_to_browsers(client):
    """Browser clients can only read non-simple headers that CORS exposes; the
    docs tell them to check this one."""
    r = client.get("/v1/models", headers={"Origin": "https://example.com"})
    exposed = r.headers.get("access-control-expose-headers", "")
    assert "X-Acs-Longpoll" in exposed, exposed
    assert "Retry-After" in exposed, exposed


@dbtest
def test_a_key_can_wait_on_every_job_it_may_legally_run(client):
    """The shipped default must cover small lane + big lane, not just the small
    one: a key running 3 small + 1 big otherwise cannot wait on the fourth."""
    from wrapper.settings import Settings

    shipped = Settings.model_fields
    per_key = shipped["harvest_max_longpoll_per_key"].default
    small = shipped["harvest_max_running_per_key_small"].default
    big = shipped["harvest_max_running_per_key"].default
    assert per_key >= small + big, f"{per_key} < {small} + {big}"
    assert shipped["harvest_max_longpoll_total"].default >= per_key
