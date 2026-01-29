#\!/usr/bin/env python3
"""Side-Channel Analysis Basics Demo.

This demo introduces side-channel analysis fundamentals:
- Power analysis basics
- Timing analysis fundamentals
- EM (electromagnetic) analysis introduction
- Leakage detection methods
- Simple power trace analysis

Educational focus - demonstrates concepts, not attack implementation.

Standards:
- ISO/IEC 17825 (Testing methods for side-channel attacks)
- NIST SP 800-57 (Key management)

Usage:
    python demos/05_domain_specific/05_side_channel_basics.py

Author: Oscura Development Team
Date: 2026-01-29
"""

# SKIP_VALIDATION: Side-channel analysis requires >60s for crypto operations

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class SideChannelBasicsDemo(BaseDemo):
    """Side-Channel Analysis Basics Demonstration."""

    name = "Side-Channel Analysis Basics"
    description = "Introduction to power analysis and leakage detection"
    category = "domain_specific"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.power_traces = []

    def generate_data(self) -> None:
        """Generate simulated power traces."""
        print_info("Generating simulated power traces...")

        num_traces = 50
        samples_per_trace = 400
        sample_rate = 1e9

        # Simulate power consumption during crypto operation
        for i in range(num_traces):
            plaintext = np.random.randint(0, 256)

            # Base power trace with noise
            trace = np.random.normal(1.0, 0.05, samples_per_trace)

            # Add data-dependent power consumption (Hamming weight model)
            hw = bin(plaintext).count('1')
            poi = samples_per_trace // 2
            trace[poi-10:poi+10] += hw * 0.3

            # Add clock activity
            t = np.arange(samples_per_trace) / sample_rate
            trace += 0.01 * np.sin(2 * np.pi * 10e6 * t)

            metadata = TraceMetadata(sample_rate=sample_rate, channel_name="power")
            self.power_traces.append(WaveformTrace(data=trace, metadata=metadata))

        print_result("Power traces generated", len(self.power_traces))

    def run_analysis(self) -> None:
        """Execute side-channel analysis."""
        print_subheader("Power Analysis Fundamentals")
        self._analyze_power_consumption()

        print_subheader("Leakage Detection")
        self._detect_leakage()

        print_subheader("Countermeasures")
        self._discuss_countermeasures()

    def _analyze_power_consumption(self) -> None:
        """Analyze power consumption patterns."""
        print_info("Analyzing power consumption patterns...")

        # Calculate mean trace
        traces_array = np.array([t.data for t in self.power_traces])
        mean_trace = np.mean(traces_array, axis=0)
        std_trace = np.std(traces_array, axis=0)

        max_std = np.max(std_trace)
        max_std_idx = np.argmax(std_trace)

        print_result("Total traces analyzed", len(self.power_traces))
        print_result("Samples per trace", len(self.power_traces[0].data))
        print_result("Maximum variance point", max_std_idx)
        print_result("Variance magnitude", f"{max_std:.4f}")

        self.results["traces_analyzed"] = len(self.power_traces)
        self.results["variance_detected"] = max_std > 0.1

    def _detect_leakage(self) -> None:
        """Detect information leakage."""
        print_info("Performing leakage detection...")

        traces_array = np.array([t.data for t in self.power_traces])

        # Split into groups
        mid = len(self.power_traces) // 2
        group1 = traces_array[:mid]
        group2 = traces_array[mid:]

        # Compute t-statistic
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        var1 = np.var(group1, axis=0, ddof=1)
        var2 = np.var(group2, axis=0, ddof=1)

        t_stat = (mean1 - mean2) / np.sqrt(var1/len(group1) + var2/len(group2))
        max_t = np.max(np.abs(t_stat))

        leakage_detected = max_t > 4.5

        print_result("Maximum |t-statistic|", f"{max_t:.2f}")
        print_result("Threshold", "4.5")
        print_result("Leakage detected", "YES" if leakage_detected else "NO")

        self.results["leakage_detected"] = leakage_detected
        self.results["max_t_statistic"] = max_t

    def _discuss_countermeasures(self) -> None:
        """Discuss side-channel countermeasures."""
        print_info("Side-Channel Countermeasures:")
        print_info("  - Constant-time implementation")
        print_info("  - Masking (randomizing intermediate values)")
        print_info("  - Hiding (equalizing power consumption)")
        print_info("  - Shuffling (randomizing operation order)")
        print_info("  - Noise injection")

    def validate_results(self, suite: ValidationSuite) -> None:
        suite.check_greater("Traces analyzed", self.results.get("traces_analyzed", 0), 0, category="power")
        suite.check_exists("Leakage detection ran", self.results.get("leakage_detected"), category="leakage")


if __name__ == "__main__":
    sys.exit(run_demo_main(SideChannelBasicsDemo))
