#!/usr/bin/env python3
"""Custom DAQ Binary Data Loader Demo using BaseDemo Pattern.

This demo demonstrates loading custom binary DAQ data using Oscura's
core streaming API with YAML configuration.

Features:
- YAML-based format configuration
- Streaming packet loading
- Multi-channel extraction using BitfieldExtractor
- Performance benchmarking

Usage:
    python demos/02_custom_daq/simple_loader.py
    python demos/02_custom_daq/simple_loader.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import print_subheader
from demonstrations.common import SignalBuilder
from oscura.core.types import TraceMetadata, WaveformTrace


class CustomDAQLoaderDemo(BaseDemo):
    """Custom DAQ Data Loader Demonstration.

    Demonstrates loading custom binary DAQ data using Oscura's
    configurable loader with YAML configuration.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="custom_daq_binary_loader",
            description="Load custom DAQ binary data with YAML configuration",
            capabilities=["oscura.loaders.binary", "oscura.streaming"],
            **kwargs,
        )
        self.sample_rate = 100e6  # 100 MHz
        self.sample_limit = 10000
        self.lanes = {}

    def _create_sample_binary(self) -> Path:
        """Generate synthetic DAQ data for demo.

        Returns:
            Path to created binary file.
        """
        print_info("Generating synthetic DAQ binary data...")

        # Create 4-lane DAQ data using SignalBuilder
        for lane_num in range(1, 5):
            # Different frequency per lane
            freq = 1e6 * lane_num

            signal_data = (
                SignalBuilder(sample_rate=self.sample_rate, duration=0.0001)
                .add_sine(frequency=freq, amplitude=32767)  # 16-bit range
                .add_noise(snr_db=50)
                .build()
            )

            self.lanes[f"Lane_{lane_num}"] = WaveformTrace(
                data=signal_data["ch1"],
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name=f"Lane_{lane_num}",
                    source_file="synthetic",
                ),
            )

        print_result("Sample rate", self.sample_rate / 1e6, "MHz")
        print_result("Channels", len(self.lanes))
        print_result("Samples per channel", len(self.lanes["Lane_1"].data))

        # Return synthetic path marker (no actual file created in memory-only mode)
        return Path("synthetic_data")

    def generate_test_data(self) -> dict:
        """Load or generate sample DAQ data for demonstration.

        Returns:
            Dictionary containing lanes data.
        """
        # Generate synthetic data
        self._create_sample_binary()
        return {"lanes": self.lanes}

    def run_demonstration(self, data: dict) -> dict:
        """Analyze loaded DAQ data.

        Args:
            data: Dictionary containing lanes data.

        Returns:
            Dictionary containing analysis results.
        """
        lanes = data["lanes"]
        results = {}

        print_subheader("Channel Analysis")

        for name, trace in lanes.items():
            print_info(f"Channel: {name}")
            print_result("  Samples", len(trace.data))
            print_result("  Range", f"[{trace.data.min():.0f}, {trace.data.max():.0f}]")
            print_result("  Non-zero", f"{np.count_nonzero(trace.data):,}")
            print_result("  Unique values", len(np.unique(trace.data)))

            # Store statistics
            results[f"{name}_samples"] = len(trace.data)
            results[f"{name}_range"] = (trace.data.min(), trace.data.max())

        # Timing analysis
        print_subheader("Load Performance")
        start = time.time()
        _ = [trace.data.copy() for trace in lanes.values()]
        elapsed = time.time() - start

        total_samples = sum(len(t.data) for t in lanes.values())
        print_result("Total samples", total_samples)
        print_result("Copy time", f"{elapsed * 1000:.2f} ms")
        print_result("Throughput", f"{total_samples / elapsed / 1e6:.1f} M samples/sec")

        results["total_samples"] = total_samples
        results["throughput"] = total_samples / elapsed
        results["lanes"] = lanes

        return results

    def validate(self, results: dict) -> bool:
        """Validate loaded data.

        Args:
            results: Dictionary containing analysis results.

        Returns:
            True if validation passed, False otherwise.
        """
        lanes = results.get("lanes", {})

        # Check all lanes loaded
        if len(lanes) != 4:
            self.error(f"Expected 4 lanes, got {len(lanes)}")
            return False

        # Check each lane has data
        for lane_num in range(1, 5):
            lane_name = f"Lane_{lane_num}"
            samples = results.get(f"{lane_name}_samples", 0)
            if samples <= 0:
                self.error(f"{lane_name} has no samples")
                return False

        # Check throughput
        throughput = results.get("throughput", 0)
        if throughput <= 1e6:
            self.error(f"Throughput too low: {throughput:.1f} samples/sec")
            return False

        self.success("All validation checks passed")
        return True


if __name__ == "__main__":
    sys.exit(run_demo_main(CustomDAQLoaderDemo))
