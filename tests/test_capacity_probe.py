from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from serving.capacity_probe import (
    NVIDIA_SMI_GPU_QUERY,
    NVIDIA_SMI_TELEMETRY_QUERY,
    gpu_type_from_inventory,
    query_gpu_inventory,
    query_gpu_telemetry,
    summarize_gpu_telemetry,
)


class CapacityProbeTests(unittest.TestCase):
    def test_query_gpu_inventory_uses_nvidia_smi_csv_query(self) -> None:
        with patch("serving.capacity_probe.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                stdout="NVIDIA H200, 143771 MiB\nNVIDIA H200, 143771 MiB\n"
            )

            output = query_gpu_inventory()

        self.assertEqual(
            output,
            "NVIDIA H200, 143771 MiB\nNVIDIA H200, 143771 MiB\n",
        )
        run.assert_called_once_with(
            NVIDIA_SMI_GPU_QUERY,
            capture_output=True,
            text=True,
        )

    def test_query_gpu_telemetry_uses_nvidia_smi_csv_query(self) -> None:
        with patch("serving.capacity_probe.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                stdout=(
                    "NVIDIA H200, 143771, 100, 0, 31, 70.5, 550.54.15\n"
                    "NVIDIA H200, 143771, 300, 50, 35, 71.5, 550.54.15\n"
                )
            )

            output = query_gpu_telemetry()

        self.assertEqual(
            output,
            (
                "NVIDIA H200, 143771, 100, 0, 31, 70.5, 550.54.15\n"
                "NVIDIA H200, 143771, 300, 50, 35, 71.5, 550.54.15\n"
            ),
        )
        run.assert_called_once_with(
            NVIDIA_SMI_TELEMETRY_QUERY,
            capture_output=True,
            text=True,
        )

    def test_gpu_type_from_inventory_reads_first_csv_field(self) -> None:
        output = " NVIDIA H200 , 143771 MiB\nNVIDIA H200, 143771 MiB\n"

        self.assertEqual(gpu_type_from_inventory(output), "NVIDIA H200")

    def test_gpu_type_from_inventory_preserves_scheduled_empty_output_behavior(
        self,
    ) -> None:
        self.assertEqual(gpu_type_from_inventory(""), "unknown")
        self.assertEqual(gpu_type_from_inventory(" , 143771 MiB"), "")

    def test_summarize_gpu_telemetry_returns_mean_and_std(self) -> None:
        output = (
            "NVIDIA H200, 143771, 100, 0, 31, 70.5, 550.54.15\n"
            "NVIDIA H200, 143771, 300, 50, 35, 71.5, 550.54.15\n"
        )

        summary = summarize_gpu_telemetry(output)

        self.assertEqual(summary["gpu_count"], 2)
        self.assertEqual(summary["driver_version"], "550.54.15")
        self.assertAlmostEqual(summary["gpu_memory_total_mb"], 143771.0)
        self.assertAlmostEqual(summary["gpu_memory_total_std_mb"], 0.0)
        self.assertAlmostEqual(summary["gpu_memory_used_mb"], 200.0)
        self.assertAlmostEqual(summary["gpu_memory_used_std_mb"], 100.0)
        self.assertAlmostEqual(summary["gpu_utilization_pct"], 25.0)
        self.assertAlmostEqual(summary["gpu_utilization_std_pct"], 25.0)
        self.assertAlmostEqual(summary["gpu_temperature_c"], 33.0)
        self.assertAlmostEqual(summary["gpu_temperature_std_c"], 2.0)
        self.assertAlmostEqual(summary["gpu_power_w"], 71.0)
        self.assertAlmostEqual(summary["gpu_power_std_w"], 0.5)

    def test_summarize_gpu_telemetry_tolerates_empty_output(self) -> None:
        summary = summarize_gpu_telemetry("")

        self.assertIsNone(summary["gpu_count"])
        self.assertIsNone(summary["gpu_utilization_pct"])


if __name__ == "__main__":
    unittest.main()
