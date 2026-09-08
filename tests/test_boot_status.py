"""Unit tests for serving/boot_status.py (ACS-272).

The publisher's contract: authored stages only, monotonic within a boot,
``vllm_exited`` → ``failed`` only while the boot is still in progress, and
lifetime-CSV events are never polluted with synthetic ones.
"""

from __future__ import annotations

from serving import boot_status


class _FakeDict:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict]] = []

    def put(self, key: str, value: dict) -> None:
        self.puts.append((key, value))


class _ExplodingDict:
    def put(self, key: str, value: dict) -> None:
        raise RuntimeError("control plane down")


def _arm(app_name: str = "acs-test-app") -> _FakeDict:
    fake = _FakeDict()
    boot_status.configure(app_name)
    boot_status._state["dict"] = fake
    return fake


def _stages(fake: _FakeDict) -> list[str]:
    return [v["stage"] for _, v in fake.puts]


def test_happy_path_boot_publishes_ordered_stages():
    fake = _arm()
    for ev in (
        "started",  # lifetime-only, publishes nothing
        "container_up",
        "weights_load_start",
        "weights_load_complete",
        "engine_init_complete",
        "vllm_port_open",
    ):
        boot_status.publish_event(ev)
    assert _stages(fake) == [
        "container_started",
        "weights_loading",
        "weights_loaded",
        "engine_ready",
        "serving",
    ]
    assert all(k == "acs-test-app" for k, _ in fake.puts)
    assert all(set(v) == {"stage", "detail", "ts", "container_id"} for _, v in fake.puts)


def test_stage_regressions_are_dropped():
    fake = _arm()
    boot_status.publish_event("weights_load_complete")
    boot_status.publish_event("weights_load_start")  # TP>1 late worker line
    boot_status.publish_event("weights_load_complete")  # duplicate marker
    assert _stages(fake) == ["weights_loaded"]


def test_vllm_exited_mid_boot_publishes_failed():
    fake = _arm()
    boot_status.publish_event("container_up")
    boot_status.publish_event("weights_load_start")
    boot_status.publish_event("vllm_exited")
    assert _stages(fake)[-1] == "failed"
    assert fake.puts[-1][1]["detail"] == "vLLM exited during startup"
    # Terminal: nothing publishes after failed until the next configure().
    boot_status.publish_event("vllm_port_open")
    assert _stages(fake)[-1] == "failed"


def test_vllm_exited_after_serving_is_normal_shutdown():
    fake = _arm()
    for ev in ("container_up", "weights_load_start", "weights_load_complete",
               "engine_init_complete", "vllm_port_open"):
        boot_status.publish_event(ev)
    boot_status.publish_event("vllm_exited")
    assert "failed" not in _stages(fake)


def test_direct_publish_serving_marks_boot_complete():
    # Snapshot-restore path: publish_stage("serving") without the event chain
    # must still suppress a later vllm_exited → failed.
    fake = _arm()
    boot_status.publish_stage("serving")
    boot_status.publish_event("vllm_exited")
    assert _stages(fake) == ["serving"]


def test_unconfigured_or_broken_dict_never_raises():
    boot_status.configure("")  # unbound
    boot_status.publish_stage("serving")  # no-op, no raise
    boot_status.configure("acs-test-app")
    boot_status._state["dict"] = _ExplodingDict()
    boot_status.publish_event("container_up")  # swallowed, no raise


def test_make_emitter_routes_both_sinks_and_filters_synthetic():
    fake = _arm()
    lifetime_events: list[str] = []
    emit = boot_status.make_emitter(lifetime_events.append)
    emit("container_up")
    emit("vllm_exited")
    # Lifetime CSV sees only the real event; the Dict saw the failure too.
    assert lifetime_events == ["container_up"]
    assert _stages(fake) == ["container_started", "failed"]
