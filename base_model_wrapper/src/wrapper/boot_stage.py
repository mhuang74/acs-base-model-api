"""Interpret container-published cold-boot stages for user surfaces (ACS-272).

Serving containers publish authored stage keys into a shared Modal Dict (see
``serving/boot_status.py`` in the repo root; read via
``modal_ops.get_boot_status``). This module turns a raw Dict entry into what a
user should actually see, applying the freshness rules below, and provides the
tiny render helpers the three surfaces share (workbench status frames, public
SSE keepalive comments, the per-model status endpoint).

Freshness — the Dict is a mailbox, not a lifecycle: entries survive scale-down,
so age decides whether an entry describes *this* boot:

- A boot may have been started minutes ago by ANOTHER user's request, so "entry
  older than my wait" is NOT staleness. Instead: in-progress stages count as
  live within ``MAX_BOOT_AGE_S`` (no real weight-load runs longer); ``serving``
  only within ``SERVING_FRESH_S`` (older means a previous boot that likely
  scaled down since); ``failed`` within ``FAILED_FRESH_S``.
- During an active cold-boot wait, no fresh entry means Modal hasn't started
  our container yet — the one phase the container can't report — so we infer
  ``waiting_for_gpu``. That's also the honest default when the Dict is
  unreachable (get_boot_status returns None on any failure).

Only authored stage keys and labels leave this module — never raw container
log content.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import modal_ops as modalops

STAGE_LABELS: dict[str, str] = {
    "waiting_for_gpu": "Waiting for Modal to allocate GPUs",
    "container_started": "Container started",
    "weights_loading": "Loading model weights",
    "weights_loaded": "Weights loaded — initializing engine",
    "engine_ready": "Engine ready — opening port",
    "serving": "Model is up — request in flight",
    "failed": "Boot failed — vLLM exited during startup",
}

MAX_BOOT_AGE_S = 45 * 60  # in-progress stage older than this = previous boot
SERVING_FRESH_S = 10 * 60  # "serving" goes stale fast (scale-down window)
FAILED_FRESH_S = 30 * 60


@dataclass(frozen=True)
class BootStage:
    stage: str
    label: str
    age_s: int | None  # None when inferred (waiting_for_gpu), not read from Dict

    def as_payload(self) -> dict[str, object]:
        """JSON shape shared by status frames and the status endpoint."""
        return {"stage": self.stage, "stage_label": self.label, "stage_age_s": self.age_s}


def interpret(entry: dict | None) -> BootStage | None:
    """Freshness-filter a raw Dict entry; None = nothing current to show."""
    if not entry:
        return None
    stage = entry.get("stage")
    if stage not in STAGE_LABELS or stage == "waiting_for_gpu":
        return None
    try:
        ts = dt.datetime.fromisoformat(entry["ts"])
        age_s = max(0, int((dt.datetime.now(dt.timezone.utc) - ts).total_seconds()))
    except (KeyError, TypeError, ValueError):
        return None
    horizon = {
        "serving": SERVING_FRESH_S,
        "failed": FAILED_FRESH_S,
    }.get(stage, MAX_BOOT_AGE_S)
    if age_s > horizon:
        return None
    return BootStage(stage=stage, label=STAGE_LABELS[stage], age_s=age_s)


async def for_cold_wait(app_name: str) -> BootStage:
    """Stage to show while a request is actively waiting out a cold boot.

    Always returns something renderable: a fresh container-reported stage, or
    the inferred ``waiting_for_gpu`` when there's no fresh entry (container
    not started yet, or Dict unreachable).
    """
    st = interpret(await modalops.get_boot_status(app_name))
    if st is not None:
        return st
    return BootStage(
        stage="waiting_for_gpu",
        label=STAGE_LABELS["waiting_for_gpu"],
        age_s=None,
    )


async def snapshot(app_name: str) -> BootStage | None:
    """Stage for the status endpoint — no cold-wait context, so no inference."""
    return interpret(await modalops.get_boot_status(app_name))


# Bar fill per stage — coarse, honest checkpoints. We don't know wall-clock
# totals, so the bar advances on real container milestones rather than faking
# smooth progress; it never claims 100% before the first token.
_STAGE_FRACTION = {
    "waiting_for_gpu": 0.05,
    "container_started": 0.25,
    "weights_loading": 0.45,
    "weights_loaded": 0.70,
    "engine_ready": 0.90,
    "serving": 0.95,
    "failed": 1.0,
}
_BAR_WIDTH = 20
# Redraws must fully cover the previous (possibly longer) line, so every
# frame is padded to a fixed width before the CR. Must exceed the longest
# label + bar + a three-digit-minutes elapsed (the failed-stage label peaks
# at 75 chars; pinned by test_bar_width_fits_every_stage).
_BAR_LINE_WIDTH = 80


def _fmt_elapsed(elapsed_s: int) -> str:
    return f"{elapsed_s // 60}m{elapsed_s % 60:02d}s"


def sse_comment(st: BootStage, elapsed_s: int) -> bytes:
    """One CR-terminated SSE comment rendering an in-place progress bar.

    A lone CR is a valid SSE line terminator (WHATWG server-sent-events
    spec), so compliant parsers (EventSource, openai-python, anthropic
    SDKs) see an ignorable comment — while a raw terminal treats ``\r`` as
    "redraw this line", so ``curl -N`` shows a single live-updating
    HF-style bar instead of a scrolling wall (ACS-280):

        : [████████░░░░░░░░░░░░] Loading model weights · 4m32s

    Contract for safety with naive parsers that split only on ``\n``: the
    frame must stay a PURE comment — starts with ':', never contains a
    newline or a ``data:`` token — so however many frames accumulate before
    the next real newline, they parse as one big ignorable comment. The
    keepalive loop emits a bare ``\n`` handoff before the first token or
    error frame (see routes/api.py), which both terminates that comment and
    preserves the finished bar as one line above the output.
    """
    frac = _STAGE_FRACTION.get(st.stage, 0.05)
    filled = round(frac * _BAR_WIDTH)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    text = f": [{bar}] {st.label} · {_fmt_elapsed(elapsed_s)}"
    return text.ljust(_BAR_LINE_WIDTH).encode() + b"\r"
