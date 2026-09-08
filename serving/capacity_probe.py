"""Shared GPU capacity-probe helpers for Modal entrypoints."""

from __future__ import annotations

import math
import subprocess
from typing import Any

NVIDIA_SMI_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader",
]

NVIDIA_SMI_TELEMETRY_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,driver_version",
    "--format=csv,noheader,nounits",
]

TELEMETRY_EMPTY: dict[str, Any] = {
    "gpu_count": None,
    "gpu_memory_total_mb": None,
    "gpu_memory_total_std_mb": None,
    "gpu_memory_used_mb": None,
    "gpu_memory_used_std_mb": None,
    "gpu_utilization_pct": None,
    "gpu_utilization_std_pct": None,
    "gpu_temperature_c": None,
    "gpu_temperature_std_c": None,
    "gpu_power_w": None,
    "gpu_power_std_w": None,
    "driver_version": None,
}


def query_gpu_inventory() -> str:
    """Return ``nvidia-smi`` GPU inventory output as CSV lines.

    Preserve the previous probe behavior: do not raise on a non-zero exit code,
    and return stdout exactly as emitted.
    """
    result = subprocess.run(
        NVIDIA_SMI_GPU_QUERY,
        capture_output=True,
        text=True,
    )
    return result.stdout


def query_gpu_telemetry() -> str:
    """Return best-effort ``nvidia-smi`` telemetry output as CSV lines."""
    result = subprocess.run(
        NVIDIA_SMI_TELEMETRY_QUERY,
        capture_output=True,
        text=True,
    )
    return result.stdout


def gpu_type_from_inventory(output: str) -> str:
    """Extract the first GPU name from ``query_gpu_inventory`` output."""
    if not output:
        return "unknown"
    return output.split(",", 1)[0].strip()


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def summarize_gpu_telemetry(output: str) -> dict[str, Any]:
    """Summarize per-GPU telemetry as mean + stddev values.

    The capacity probe is primarily an allocation check, so telemetry must stay
    tolerant of older ``nvidia-smi`` output and missing fields.
    """
    if not output.strip():
        return dict(TELEMETRY_EMPTY)

    memory_total: list[float] = []
    memory_used: list[float] = []
    utilization: list[float] = []
    temperature: list[float] = []
    power: list[float] = []
    driver_version: str | None = None
    gpu_count = 0

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        gpu_count += 1
        if driver_version is None and parts[6]:
            driver_version = parts[6]
        for bucket, value in (
            (memory_total, _to_float(parts[1])),
            (memory_used, _to_float(parts[2])),
            (utilization, _to_float(parts[3])),
            (temperature, _to_float(parts[4])),
            (power, _to_float(parts[5])),
        ):
            if value is not None:
                bucket.append(value)

    total_mean, total_std = _mean_std(memory_total)
    used_mean, used_std = _mean_std(memory_used)
    util_mean, util_std = _mean_std(utilization)
    temp_mean, temp_std = _mean_std(temperature)
    power_mean, power_std = _mean_std(power)
    return {
        "gpu_count": gpu_count or None,
        "gpu_memory_total_mb": total_mean,
        "gpu_memory_total_std_mb": total_std,
        "gpu_memory_used_mb": used_mean,
        "gpu_memory_used_std_mb": used_std,
        "gpu_utilization_pct": util_mean,
        "gpu_utilization_std_pct": util_std,
        "gpu_temperature_c": temp_mean,
        "gpu_temperature_std_c": temp_std,
        "gpu_power_w": power_mean,
        "gpu_power_std_w": power_std,
        "driver_version": driver_version,
    }
