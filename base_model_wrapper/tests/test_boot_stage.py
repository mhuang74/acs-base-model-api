"""Unit tests for wrapper.boot_stage (ACS-272) — freshness rules + rendering."""

from __future__ import annotations

import datetime as dt

import pytest

from wrapper import boot_stage


def _entry(stage: str, age_s: float = 10.0) -> dict:
    ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s)
    return {"stage": stage, "detail": "", "ts": ts.isoformat(), "container_id": "ta-1"}


def test_interpret_fresh_in_progress_stage():
    st = boot_stage.interpret(_entry("weights_loading", age_s=120))
    assert st is not None
    assert st.stage == "weights_loading"
    assert st.label == boot_stage.STAGE_LABELS["weights_loading"]
    assert 115 <= st.age_s <= 130


@pytest.mark.parametrize(
    ("stage", "age_s", "expect_shown"),
    [
        ("weights_loading", boot_stage.MAX_BOOT_AGE_S + 60, False),
        ("serving", 60, True),
        ("serving", boot_stage.SERVING_FRESH_S + 60, False),
        ("failed", 60, True),
        ("failed", boot_stage.FAILED_FRESH_S + 60, False),
    ],
)
def test_interpret_freshness_horizons(stage, age_s, expect_shown):
    st = boot_stage.interpret(_entry(stage, age_s=age_s))
    assert (st is not None) is expect_shown


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"stage": "weights_loading"},  # no ts
        {"stage": "weights_loading", "ts": "not-a-date"},
        {"stage": "rm -rf /", "ts": dt.datetime.now(dt.timezone.utc).isoformat()},
        {"stage": "waiting_for_gpu", "ts": dt.datetime.now(dt.timezone.utc).isoformat()},
    ],
)
def test_interpret_rejects_malformed_or_unknown(entry):
    # Unknown stage keys are rejected outright — only authored keys render,
    # so a poisoned Dict entry can never inject text into user surfaces.
    assert boot_stage.interpret(entry) is None


async def test_for_cold_wait_falls_back_to_waiting_for_gpu(monkeypatch):
    async def _none(app_name):
        return None

    monkeypatch.setattr(boot_stage.modalops, "get_boot_status", _none)
    st = await boot_stage.for_cold_wait("acs-llama-405b")
    assert st.stage == "waiting_for_gpu"
    assert st.age_s is None
    assert st.as_payload()["stage_label"] == boot_stage.STAGE_LABELS["waiting_for_gpu"]


async def test_for_cold_wait_uses_fresh_entry(monkeypatch):
    async def _fresh(app_name):
        return _entry("engine_ready", age_s=5)

    monkeypatch.setattr(boot_stage.modalops, "get_boot_status", _fresh)
    st = await boot_stage.for_cold_wait("acs-llama-405b")
    assert st.stage == "engine_ready"


def test_sse_comment_is_cr_terminated_progress_bar():
    st = boot_stage.BootStage(
        stage="weights_loading",
        label=boot_stage.STAGE_LABELS["weights_loading"],
        age_s=3,
    )
    line = boot_stage.sse_comment(st, elapsed_s=130)
    # Pure SSE comment, CR-terminated (valid WHATWG line ending): compliant
    # parsers ignore it; terminals redraw in place (ACS-280).
    assert line.startswith(b": [")
    assert line.endswith(b"\r")
    assert b"\n" not in line
    assert b"data:" not in line
    text = line[:-1].decode()
    assert len(text) == boot_stage._BAR_LINE_WIDTH  # fixed-width redraw
    assert "Loading model weights" in text
    assert "2m10s" in text
    assert "\u2588" in text and "\u2591" in text  # filled + empty bar cells


@pytest.mark.parametrize(
    ("stage", "more_filled_than"),
    [("container_started", "waiting_for_gpu"), ("weights_loaded", "weights_loading"),
     ("engine_ready", "weights_loaded")],
)
def test_bar_fill_is_monotonic_across_stages(stage, more_filled_than):
    def filled(stage_key: str) -> int:
        st = boot_stage.BootStage(stage=stage_key, label="x", age_s=1)
        return boot_stage.sse_comment(st, 10).decode().count("\u2588")

    assert filled(stage) > filled(more_filled_than)


def test_bar_width_fits_every_stage():
    # ljust() silently no-ops on overflow, which would leave redraw residue —
    # pin that every stage's frame (at a pessimistic elapsed) fits the fixed
    # width (review #284 nit: the failed label overflowed the original 72).
    for stage, label in boot_stage.STAGE_LABELS.items():
        st = boot_stage.BootStage(stage=stage, label=label, age_s=1)
        text = boot_stage.sse_comment(st, elapsed_s=100 * 60).decode()[:-1]
        assert len(text) == boot_stage._BAR_LINE_WIDTH, (stage, len(text))
