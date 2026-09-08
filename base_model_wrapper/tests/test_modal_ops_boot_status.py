"""Unit tests for modal_ops.get_boot_status (ACS-276, review follow-up on #278).

This is the fail-silent raw-RPC read of the boot-status Dict. Because every
failure deliberately degrades to None (callers render the waiting-for-GPU
fallback), a Modal SDK change that broke it would show up in prod only as
stages quietly disappearing — these tests exist to make that a red CI instead.

The fake stub mirrors the two RPCs the real path makes (DictGetOrCreate →
DictGet) and values round-trip through the REAL modal serialize/deserialize,
so a serialization-format change in the pinned SDK fails here too.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from modal._serialization import serialize

from wrapper import modal_ops


class _FakeStub:
    def __init__(self, *, value=None, found=True, exc=None, delay_s=0.0):
        self.value = value
        self.found = found
        self.exc = exc
        self.delay_s = delay_s
        self.get_or_create_calls = 0
        self.get_calls = 0

    async def DictGetOrCreate(self, req):
        self.get_or_create_calls += 1
        return SimpleNamespace(dict_id="di-test")

    async def DictGet(self, req):
        self.get_calls += 1
        if self.exc is not None:
            raise self.exc
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return SimpleNamespace(found=self.found, value=self.value)


@pytest.fixture
def arm(monkeypatch):
    """Reset module state, satisfy _auth, and install a fake async client."""

    def _arm(stub: _FakeStub) -> _FakeStub:
        monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
        modal_ops._BOOT_STATUS_CACHE.clear()
        modal_ops._BOOT_STATUS_DICT_ID = None

        async def fake_from_env():
            return SimpleNamespace(stub=stub)

        monkeypatch.setattr(modal_ops._ModalAsyncClient, "from_env", fake_from_env)
        return stub

    yield _arm
    modal_ops._BOOT_STATUS_CACHE.clear()
    modal_ops._BOOT_STATUS_DICT_ID = None


async def test_found_entry_roundtrips_through_real_serialization(arm):
    entry = {"stage": "weights_loading", "ts": "2026-07-22T10:00:00+00:00"}
    arm(_FakeStub(value=serialize(entry)))
    got = await modal_ops.get_boot_status("acs-llama-405b")
    assert got == entry


async def test_not_found_returns_none(arm):
    arm(_FakeStub(found=False))
    assert await modal_ops.get_boot_status("acs-llama-405b") is None


async def test_non_dict_value_returns_none(arm):
    arm(_FakeStub(value=serialize("not a dict")))
    assert await modal_ops.get_boot_status("acs-llama-405b") is None


async def test_rpc_error_degrades_to_none_and_never_raises(arm):
    arm(_FakeStub(exc=RuntimeError("control plane down")))
    assert await modal_ops.get_boot_status("acs-llama-405b") is None


async def test_slow_rpc_is_time_bounded(arm, monkeypatch):
    # A hung control plane must not stall the caller (keepalive loops run on a
    # 5 s cadence) — the wait_for bound turns it into a None.
    monkeypatch.setattr(modal_ops, "_BOOT_STATUS_TIMEOUT_S", 0.05)
    arm(_FakeStub(value=serialize({"stage": "serving"}), delay_s=1.0))
    assert await modal_ops.get_boot_status("acs-llama-405b") is None


async def test_ttl_cache_dedupes_rpcs_including_failures(arm):
    entry = {"stage": "serving", "ts": "2026-07-22T10:00:00+00:00"}
    stub = arm(_FakeStub(value=serialize(entry)))
    assert await modal_ops.get_boot_status("acs-llama-405b") == entry
    assert await modal_ops.get_boot_status("acs-llama-405b") == entry
    assert stub.get_calls == 1  # second call served from the TTL cache

    # Failures are cached too — a broken control plane is probed at most once
    # per TTL window per app, not once per keepalive tick.
    fail = arm(_FakeStub(exc=RuntimeError("down")))
    assert await modal_ops.get_boot_status("acs-other") is None
    assert await modal_ops.get_boot_status("acs-other") is None
    assert fail.get_calls == 1


async def test_dict_id_resolved_once_across_apps(arm):
    stub = arm(_FakeStub(found=False))
    await modal_ops.get_boot_status("acs-a")
    await modal_ops.get_boot_status("acs-b")
    assert stub.get_or_create_calls == 1  # dict id memoized module-wide
    assert stub.get_calls == 2


async def test_missing_modal_tokens_degrade_to_none(arm, monkeypatch):
    stub = arm(_FakeStub(found=True, value=serialize({"stage": "serving"})))
    monkeypatch.delenv("MODAL_TOKEN_ID")
    assert await modal_ops.get_boot_status("acs-llama-405b") is None
    assert stub.get_calls == 0  # _auth failed before any RPC
