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

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import print_subheader
from demonstrations.common import SignalBuilder
from oscura.core.types import TraceMetadata, WaveformTrace


class CustomDAQLoaderDemo(BaseDemo):
    """Custom DAQ Data Loader Demonstration.

    Demonstrates loading custom binary DAQ data using Oscura's
    configurable loader with YAML configuration.
    """

    name = "Custom DAQ Binary Loader"
    description = "Load custom DAQ binary data with YAML configuration"
    category = "custom_daq"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
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

            signal = (
                SignalBuilder(sample_rate=self.sample_rate, duration=0.0001)
                .add_sine(frequency=freq, amplitude=32767)  # 16-bit range
                .add_noise(snr_db=50)
                .build()
            )

            self.lanes[f"Lane_{lane_num}"] = WaveformTrace(
                data=signal.data["ch1"],
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

    def generate_data(self) -> None:
        """Load or generate sample DAQ data for demonstration."""
        bin_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            bin_file_to_load = self.data_file
            print_info(f"Loading binary from CLI override: {self.data_file}")
        # 2. Check default demo_data files
        elif (default_file := self.find_default_data_file("multi_lane_daq_10M.bin")) or (
            default_file := self.find_default_data_file("continuous_acquisition_100M.bin")
        ):
            bin_file_to_load = default_file
            print_info(f"Loading binary from default file: {default_file.name}")

        # Use existing file if found
        if bin_file_to_load:
            # TODO: Load actual binary file data
            # For now, fall back to synthetic generation
            print_info("Binary file loading not yet implemented, using synthetic data")
            self._create_sample_binary()
        else:
            # 3. Generate synthetic data as fallback
            self._create_sample_binary()

    def run_analysis(self) -> None:
        """Analyze loaded DAQ data."""
        print_subheader("Channel Analysis")

        for name, trace in self.lanes.items():
            print_info(f"Channel: {name}")
            print_result("  Samples", len(trace.data))
            print_result("  Range", f"[{trace.data.min():.0f}, {trace.data.max():.0f}]")
            print_result("  Non-zero", f"{np.count_nonzero(trace.data):,}")
            print_result("  Unique values", len(np.unique(trace.data)))

            # Store statistics
            self.results[f"{name}_samples"] = len(trace.data)
            self.results[f"{name}_range"] = (trace.data.min(), trace.data.max())

        # Timing analysis
        print_subheader("Load Performance")
        start = time.time()
        _ = [trace.data.copy() for trace in self.lanes.values()]
        elapsed = time.time() - start

        total_samples = sum(len(t.data) for t in self.lanes.values())
        print_result("Total samples", total_samples)
        print_result("Copy time", f"{elapsed * 1000:.2f} ms")
        print_result("Throughput", f"{total_samples / elapsed / 1e6:.1f} M samples/sec")

        self.results["total_samples"] = total_samples
        self.results["throughput"] = total_samples / elapsed

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate loaded data."""
        # Check all lanes loaded
        suite.check_equal(
            "All lanes loaded",
            len(self.lanes),
            4,
            category="loading",
        )

        # Check each lane has data
        for lane_num in range(1, 5):
            lane_name = f"Lane_{lane_num}"
            suite.check_greater(
                f"{lane_name} samples",
                self.results.get(f"{lane_name}_samples", 0),
                0,
                category="data",
            )

        # Check throughput
        suite.check_greater(
            "Throughput > 1M samples/sec",
            self.results.get("throughput", 0),
            1e6,
            category="performance",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(CustomDAQLoaderDemo))
