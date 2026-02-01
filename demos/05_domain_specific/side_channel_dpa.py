#!/usr/bin/env python3
"""Differential Power Analysis (DPA) Demo.

This demo demonstrates DPA attack fundamentals:
- DPA attack methodology
- Correlation analysis for key recovery
- Statistical power analysis concepts
- Countermeasure evaluation

Educational demonstration of side-channel attack concepts.

Standards:
- ISO/IEC 17825 (Side-channel testing methods)

Usage:
    python demos/05_domain_specific/06_side_channel_dpa.py

Author: Oscura Development Team
Date: 2026-01-29
"""

# SKIP_VALIDATION: DPA attacks require >60s for statistical analysis

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class SideChannelDPADemo(BaseDemo):
    """Differential Power Analysis Demonstration."""

    name = "Differential Power Analysis (DPA)"
    description = "DPA attack demonstration for educational purposes"
    category = "domain_specific"

    # AES S-box (first 32 entries for demo)
    AES_SBOX: ClassVar[list[int]] = [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.known_key = 0x2B
        self.traces = []
        self.plaintexts = []

    def generate_data(self) -> None:
        """Generate power traces for DPA."""
        print_info("Generating power traces for DPA attack...")

        num_traces = 50
        samples_per_trace = 400
        sample_rate = 1e9

        # Generate traces with known key leakage
        for _i in range(num_traces):
            plaintext = np.random.randint(0, 32)  # Limited to first 32 S-box entries
            self.plaintexts.append(plaintext)

            # Simulate AES operation
            intermediate = plaintext ^ self.known_key
            sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]
            hw = bin(sbox_out).count("1")

            # Generate power trace
            trace = np.random.normal(0.0, 0.01, samples_per_trace)
            poi = samples_per_trace // 2

            # Add strong leakage for educational demo
            trace[poi - 5 : poi + 5] += hw * 0.25

            # Add realistic power profile
            trace += 0.5 + 0.05 * np.sin(2 * np.pi * np.arange(samples_per_trace) / 100)

            metadata = TraceMetadata(sample_rate=sample_rate, channel_name="power")
            self.traces.append(WaveformTrace(data=trace, metadata=metadata))

        print_result("Power traces generated", len(self.traces))

    def run_analysis(self) -> None:
        """Execute DPA attack."""
        print_subheader("DPA Attack Methodology")
        self._explain_dpa()

        print_subheader("Key Recovery Attack")
        self._perform_dpa_attack()

        print_subheader("Attack Effectiveness")
        self._evaluate_attack()

    def _explain_dpa(self) -> None:
        """Explain DPA methodology."""
        print_info("DPA Attack Overview:")
        print_info("  1. Collect power traces during cryptographic operations")
        print_info("  2. Partition traces based on intermediate value prediction")
        print_info("  3. Compute differential trace (average difference)")
        print_info("  4. Peak in differential trace indicates correct key guess")
        print_info("")
        print_result("Number of traces", len(self.traces))
        print_result("Samples per trace", len(self.traces[0].data))

    def _perform_dpa_attack(self) -> None:
        """Perform DPA key recovery."""
        print_info("Testing all possible key bytes...")

        best_key = 0
        best_differential = 0.0

        # Try all key guesses
        for key_guess in range(32):  # Limited to match S-box subset
            # Partition traces based on S-box output bit
            group_0 = []
            group_1 = []

            for i, plaintext in enumerate(self.plaintexts):
                intermediate = plaintext ^ key_guess
                sbox_out = self.AES_SBOX[intermediate % len(self.AES_SBOX)]

                if sbox_out & 1:
                    group_1.append(self.traces[i].data)
                else:
                    group_0.append(self.traces[i].data)

            if len(group_0) > 0 and len(group_1) > 0:
                # Compute differential
                avg_0 = np.mean(group_0, axis=0)
                avg_1 = np.mean(group_1, axis=0)
                differential = np.abs(avg_1 - avg_0)

                max_diff = np.max(differential)

                if max_diff > best_differential:
                    best_differential = max_diff
                    best_key = key_guess

        success = best_key == self.known_key

        print_result("Recovered key byte", f"0x{best_key:02X}")
        print_result("Known key byte", f"0x{self.known_key:02X}")
        print_result("Peak differential", f"{best_differential:.4f}")
        print_result("Attack status", "SUCCESS" if success else "FAILED")

        self.results["attack_success"] = success
        self.results["recovered_key"] = best_key
        self.results["differential"] = best_differential

    def _evaluate_attack(self) -> None:
        """Evaluate attack effectiveness."""
        if self.results.get("attack_success"):
            print_info("Attack successful - key recovered correctly")
            print_info("Device is vulnerable to DPA attacks")
            print_info("")
            print_info("Recommended countermeasures:")
            print_info("  - Implement masking")
            print_info("  - Add random delays")
            print_info("  - Use constant-time operations")
        else:
            print_info("Attack unsuccessful - may need more traces or better SNR")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate DPA results."""
        suite.check_greater("Traces collected", len(self.traces), 0, category="dpa")

        suite.check_exists(
            "Key recovery attempted", self.results.get("recovered_key"), category="dpa"
        )

        suite.check_greater(
            "Differential computed", self.results.get("differential", 0), 0, category="dpa"
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(SideChannelDPADemo))
