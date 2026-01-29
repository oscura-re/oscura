#!/usr/bin/env python3
"""Vintage Logic Family Detection Demo.

This demo demonstrates vintage logic family detection and analysis:
- TTL (74xx, 74LSxx, 74Sxx) family detection
- CMOS (4000 series, 74HCxx) detection
- ECL (10K series) detection
- Voltage level analysis
- IC identification and replacement recommendations

Logic Families:
- TTL (Transistor-Transistor Logic)
- CMOS (Complementary Metal-Oxide-Semiconductor)
- ECL (Emitter-Coupled Logic)
- RTL (Resistor-Transistor Logic)

Usage:
    python demos/05_domain_specific/08_vintage_logic.py

Author: Oscura Development Team
Date: 2026-01-29
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class VintageLogicDemo(BaseDemo):
    """Vintage Logic Family Detection Demonstration."""

    name = "Vintage Logic Family Detection"
    description = "Detect and analyze vintage logic families"
    category = "domain_specific"

    # Logic family specifications
    LOGIC_FAMILIES = {
        "TTL": {
            "description": "Transistor-Transistor Logic (74xx)",
            "v_il_max": 0.8,
            "v_ih_min": 2.0,
            "v_ol_max": 0.4,
            "v_oh_min": 2.4,
            "vcc": 5.0,
            "typical_tpd": 10e-9,
        },
        "74LS": {
            "description": "Low-Power Schottky TTL (74LSxx)",
            "v_il_max": 0.8,
            "v_ih_min": 2.0,
            "v_ol_max": 0.5,
            "v_oh_min": 2.7,
            "vcc": 5.0,
            "typical_tpd": 9e-9,
        },
        "74HC": {
            "description": "High-Speed CMOS (74HCxx)",
            "v_il_max": 1.5,
            "v_ih_min": 3.5,
            "v_ol_max": 0.1,
            "v_oh_min": 4.9,
            "vcc": 5.0,
            "typical_tpd": 8e-9,
        },
        "4000B": {
            "description": "CMOS 4000B series",
            "v_il_max": 1.5,
            "v_ih_min": 3.5,
            "v_ol_max": 0.05,
            "v_oh_min": 4.95,
            "vcc": 5.0,
            "typical_tpd": 50e-9,
        },
    }

    # IC replacement database
    REPLACEMENTS = {
        "7400": {"modern": "74HCT00", "notes": "TTL-compatible CMOS, lower power"},
        "74LS00": {"modern": "74HCT00", "notes": "Direct replacement, CMOS"},
        "7404": {"modern": "74HCT04", "notes": "TTL-compatible hex inverter"},
        "7408": {"modern": "74HCT08", "notes": "Quad 2-input AND gate"},
        "4011": {"modern": "74HC132", "notes": "Schmitt trigger NAND"},
    }

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.logic_signals = {}

    def generate_data(self) -> None:
        """Generate vintage logic signals."""
        print_info("Generating vintage logic family signals...")

        sample_rate = 1e9
        duration = 1e-6

        # TTL signal
        ttl_signal = self._generate_logic_signal(
            family="TTL",
            frequency=1e6,
            rise_time=3e-9,
            fall_time=3e-9,
            sample_rate=sample_rate,
            duration=duration,
        )
        self.logic_signals["TTL"] = ttl_signal

        # 74LS signal
        ls_signal = self._generate_logic_signal(
            family="74LS",
            frequency=2e6,
            rise_time=5e-9,
            fall_time=5e-9,
            sample_rate=sample_rate,
            duration=duration,
        )
        self.logic_signals["74LS"] = ls_signal

        # 74HC signal
        hc_signal = self._generate_logic_signal(
            family="74HC",
            frequency=5e6,
            rise_time=6e-9,
            fall_time=6e-9,
            sample_rate=sample_rate,
            duration=duration,
        )
        self.logic_signals["74HC"] = hc_signal

        print_result("Logic families generated", len(self.logic_signals))

    def _generate_logic_signal(
        self, family: str, frequency: float, rise_time: float, fall_time: float,
        sample_rate: float, duration: float
    ) -> WaveformTrace:
        """Generate logic signal for family."""
        spec = self.LOGIC_FAMILIES[family]
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate

        period = 1.0 / frequency
        phase = (t % period) / period

        v_low = spec["v_ol_max"]
        v_high = spec["v_oh_min"]

        signal = np.zeros(n_samples)

        for i in range(n_samples):
            p = phase[i]
            if p < 0.5:
                if p < rise_time / period:
                    progress = p / (rise_time / period)
                    signal[i] = v_low + (v_high - v_low) * (1 - np.exp(-5 * progress))
                else:
                    signal[i] = v_high
            else:
                time_since_fall = p - 0.5
                if time_since_fall < fall_time / period:
                    progress = time_since_fall / (fall_time / period)
                    signal[i] = v_high - (v_high - v_low) * (1 - np.exp(-5 * progress))
                else:
                    signal[i] = v_low

        # Add noise
        noise_level = (v_high - v_low) * 0.02
        signal += np.random.normal(0, noise_level, n_samples)

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name=f"{family}_output"),
        )

    def run_analysis(self) -> None:
        """Execute vintage logic analysis."""
        print_subheader("Logic Family Overview")
        self._display_families()

        print_subheader("Logic Family Detection")
        self._detect_families()

        print_subheader("IC Replacement Guide")
        self._show_replacements()

    def _display_families(self) -> None:
        """Display logic family specifications."""
        print_info("Vintage Logic Family Specifications:\n")

        for family, spec in self.LOGIC_FAMILIES.items():
            print_info(f"{family}: {spec['description']}")
            print_info(f"  V_OL (max): {spec['v_ol_max']:.2f} V")
            print_info(f"  V_OH (min): {spec['v_oh_min']:.2f} V")
            print_info(f"  Propagation delay: {spec['typical_tpd']*1e9:.1f} ns\n")

    def _detect_families(self) -> None:
        """Detect logic families from signals."""
        print_info("Analyzing logic signals...\n")

        for family_name, signal in self.logic_signals.items():
            v_high = np.percentile(signal.data, 95)
            v_low = np.percentile(signal.data, 5)

            detected_family = self._detect_family_from_voltages(v_low, v_high)

            print_info(f"Signal: {family_name}")
            print_result("  V_LOW", f"{v_low:.3f}", "V")
            print_result("  V_HIGH", f"{v_high:.3f}", "V")
            print_result("  Detected as", detected_family)

            match = "✓ Correct" if detected_family == family_name else "⚠ Mismatch"
            print_info(f"  {match}\n")

            self.results[f"{family_name}_detected"] = detected_family
            self.results[f"{family_name}_match"] = (detected_family == family_name)

    def _detect_family_from_voltages(self, v_low: float, v_high: float) -> str:
        """Detect logic family from voltage levels."""
        best_match = "Unknown"
        best_score = float("inf")

        for family, spec in self.LOGIC_FAMILIES.items():
            v_low_error = abs(v_low - spec["v_ol_max"])
            v_high_error = abs(v_high - spec["v_oh_min"])
            score = v_low_error + v_high_error

            if score < best_score:
                best_score = score
                best_match = family

        return best_match

    def _show_replacements(self) -> None:
        """Show IC replacement recommendations."""
        print_info("Vintage IC → Modern CMOS Replacement Guide:\n")

        for vintage_ic, info in self.REPLACEMENTS.items():
            print_info(f"{vintage_ic:12s} → {info['modern']:12s}")
            print_info(f"  Notes: {info['notes']}\n")

        print_info("Replacement Benefits:")
        print_info("  - Lower power consumption (1000x less static power)")
        print_info("  - Wider supply voltage range (2V-6V)")
        print_info("  - Better noise immunity")
        print_info("  - Pin-compatible drop-in replacement")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate vintage logic analysis."""
        suite.check_greater(
            "Logic families generated", len(self.logic_signals), 0, category="generation"
        )

        # Check detection accuracy
        matches = sum(
            1 for k, v in self.results.items() if k.endswith("_match") and v
        )
        suite.check_greater("Families correctly detected", matches, 0, category="detection")


if __name__ == "__main__":
    sys.exit(run_demo_main(VintageLogicDemo))
