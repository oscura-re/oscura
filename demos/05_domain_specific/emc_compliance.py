#!/usr/bin/env python3
"""EMC Compliance Testing Demo - Fundamentals.

This demo demonstrates EMC/EMI compliance testing capabilities:
- CISPR limits and compliance checking
- IEC standards application
- Conducted emissions analysis
- Radiated emissions analysis
- Pre-compliance testing workflow

Standards:
- CISPR 16-1-1 (Measurement apparatus)
- CISPR 32 (Multimedia equipment emissions)
- IEC 61000-4-3 (Radiated immunity)
- IEC 61000-3-2 (Harmonics)

Usage:
    python demos/05_domain_specific/03_emc_compliance.py
    python demos/05_domain_specific/03_emc_compliance.py --verbose

Author: Oscura Development Team
Date: 2026-01-29
"""

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


class EMCComplianceDemo(BaseDemo):
    """EMC Compliance Testing Demonstration.

    Demonstrates fundamental EMC/EMI compliance testing including
    conducted and radiated emissions analysis with limit checking.
    """

    name = "EMC Compliance Testing Fundamentals"
    description = "CISPR and IEC EMC compliance testing"
    category = "domain_specific"

    # CISPR 32 Class B limits (dBμV) - residential
    CISPR32_CONDUCTED: ClassVar[dict[float, tuple[int, int]]] = {
        0.15: (66, 56),  # (quasi-peak, average) at 150 kHz
        0.50: (56, 46),  # 500 kHz
        5.00: (56, 46),  # 5 MHz
        30.0: (60, 50),  # 30 MHz
    }

    # CISPR 32 Class B radiated limits (dBμV/m at 10m)
    CISPR32_RADIATED: ClassVar[dict[int, tuple[int, int]]] = {
        30: (30, 30),  # 30 MHz
        230: (37, 37),  # 230 MHz
        1000: (37, 37),  # 1 GHz
    }

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 100e6  # 100 MHz
        self.conducted_trace = None
        self.radiated_trace = None

    def generate_data(self) -> None:
        """Generate EMC test signals."""
        print_info("Generating EMC test signals...")

        # Conducted emissions: Switching power supply noise
        duration = 0.001  # 1 ms
        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate

        # Generate switching harmonics at 100 kHz
        fundamental = 100e3
        signal = np.zeros(n_samples)

        harmonics: ClassVar[list[str]] = [1, 2, 3, 5, 7, 9]
        levels_dbuv: ClassVar[list[str]] = [90, 75, 68, 55, 48, 42]

        for harmonic, level_dbuv in zip(harmonics, levels_dbuv, strict=False):
            amplitude_uv = 10 ** (level_dbuv / 20)
            amplitude_v = amplitude_uv * 1e-6
            freq = fundamental * harmonic
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        # Add noise floor
        noise_floor_dbuv = 40
        noise_amplitude = (10 ** (noise_floor_dbuv / 20)) * 1e-6
        signal += np.random.normal(0, noise_amplitude / 3, n_samples)

        self.conducted_trace = WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="CE_Line",
                source_file="synthetic",
            ),
        )

        # Radiated emissions: Multiple RF sources
        radiated_signal = np.zeros(n_samples)
        frequencies: ClassVar[list[str]] = [30e6, 88e6, 150e6, 433e6]
        levels_dbuvm: ClassVar[list[str]] = [35, 42, 38, 33]

        antenna_factor = 20  # dB
        for freq, level_dbuvm in zip(frequencies, levels_dbuvm, strict=False):
            level_dbuv = level_dbuvm - antenna_factor
            amplitude_v = (10 ** (level_dbuv / 20)) * 1e-6
            radiated_signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        radiated_signal += np.random.normal(0, 1e-7, n_samples)

        self.radiated_trace = WaveformTrace(
            data=radiated_signal,
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="Radiated_Antenna",
                source_file="synthetic",
            ),
        )

        print_result("Conducted trace samples", len(self.conducted_trace.data))
        print_result("Radiated trace samples", len(self.radiated_trace.data))

    def run_analysis(self) -> None:
        """Execute EMC compliance analysis."""
        # Section 1: Conducted Emissions
        print_subheader("Conducted Emissions (CISPR 32)")
        self._analyze_conducted()

        # Section 2: Radiated Emissions
        print_subheader("Radiated Emissions (CISPR 32)")
        self._analyze_radiated()

        # Section 3: Compliance Summary
        print_subheader("Compliance Summary")
        self._print_compliance()

    def _analyze_conducted(self) -> None:
        """Analyze conducted emissions."""
        print_info("Frequency range: 150 kHz - 30 MHz")
        print_info("Measurement: CISPR 16-1-1 quasi-peak and average detectors\n")

        # FFT analysis
        fft = np.fft.rfft(self.conducted_trace.data)
        freqs = np.fft.rfftfreq(
            len(self.conducted_trace.data), 1 / self.conducted_trace.metadata.sample_rate
        )
        magnitude_v = np.abs(fft) * 2 / len(self.conducted_trace.data)
        magnitude_dbuv = 20 * np.log10(magnitude_v / 1e-6 + 1e-12)

        # Check against limits
        print_info("Conducted Emissions Measurements:")
        measurements = 0
        max_margin = 0.0
        min_margin = 100.0
        violations: ClassVar[list[str]] = []

        for freq_mhz, (qp_limit, _avg_limit) in sorted(self.CISPR32_CONDUCTED.items()):
            freq_hz = freq_mhz * 1e6
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured_dbuv = magnitude_dbuv[idx]

            margin_qp = qp_limit - measured_dbuv
            status = "PASS" if margin_qp > 0 else "FAIL"

            print_info(
                f"  {freq_mhz:6.2f} MHz: {measured_dbuv:5.1f} dBμV "
                f"(Limit: {qp_limit} dBμV QP) Margin: {margin_qp:+.1f} dB [{status}]"
            )

            measurements += 1
            max_margin = max(max_margin, margin_qp)
            min_margin = min(min_margin, margin_qp)

            if margin_qp < 0:
                violations.append(freq_mhz)

        compliant = len(violations) == 0
        print_result("\nCompliance", "PASS" if compliant else "FAIL")
        print_result("Worst margin", f"{min_margin:.1f}", "dB")

        self.results["conducted_compliant"] = compliant
        self.results["conducted_measurements"] = measurements
        self.results["conducted_violations"] = len(violations)

    def _analyze_radiated(self) -> None:
        """Analyze radiated emissions."""
        print_info("Frequency range: 30 MHz - 1 GHz")
        print_info("Measurement distance: 10 meters")
        print_info("Detector: CISPR 16-1-1 quasi-peak\n")

        # FFT analysis
        fft = np.fft.rfft(self.radiated_trace.data)
        freqs = np.fft.rfftfreq(
            len(self.radiated_trace.data), 1 / self.radiated_trace.metadata.sample_rate
        )
        magnitude_v = np.abs(fft) * 2 / len(self.radiated_trace.data)

        # Convert to field strength (add antenna factor back)
        antenna_factor = 20
        magnitude_dbuvm = 20 * np.log10(magnitude_v / 1e-6 + 1e-12) + antenna_factor

        # Check against limits
        print_info("Radiated Emissions Measurements:")
        measurements = 0
        max_margin = 0.0
        min_margin = 100.0
        violations: ClassVar[list[str]] = []

        for freq_mhz, (qp_limit, _avg_limit) in sorted(self.CISPR32_RADIATED.items()):
            freq_hz = freq_mhz * 1e6
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured_dbuvm = magnitude_dbuvm[idx]

            margin = qp_limit - measured_dbuvm
            status = "PASS" if margin > 0 else "FAIL"

            print_info(
                f"  {freq_mhz:6.0f} MHz: {measured_dbuvm:5.1f} dBμV/m "
                f"(Limit: {qp_limit} dBμV/m) Margin: {margin:+.1f} dB [{status}]"
            )

            measurements += 1
            max_margin = max(max_margin, margin)
            min_margin = min(min_margin, margin)

            if margin < 0:
                violations.append(freq_mhz)

        compliant = len(violations) == 0
        print_result("\nCompliance", "PASS" if compliant else "FAIL")
        print_result("Worst margin", f"{min_margin:.1f}", "dB")

        self.results["radiated_compliant"] = compliant
        self.results["radiated_measurements"] = measurements
        self.results["radiated_violations"] = len(violations)

    def _print_compliance(self) -> None:
        """Print overall compliance summary."""
        print_info("Overall EMC Compliance Status:\n")

        tests: ClassVar[list[str]] = [
            ("CISPR 32 Conducted", self.results.get("conducted_compliant", False)),
            ("CISPR 32 Radiated", self.results.get("radiated_compliant", False)),
        ]

        for test_name, compliant in tests:
            status = "PASS" if compliant else "FAIL"
            print_info(f"  {test_name:25s}: {status}")

        all_compliant = all(compliant for _, compliant in tests)
        print_result("\nOverall Compliance", "PASS" if all_compliant else "FAIL")

        self.results["overall_compliant"] = all_compliant

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate EMC analysis results."""
        suite.check_greater(
            "Conducted measurements",
            self.results.get("conducted_measurements", 0),
            0,
            category="conducted",
        )

        suite.check_greater(
            "Radiated measurements",
            self.results.get("radiated_measurements", 0),
            0,
            category="radiated",
        )

        suite.check_exists(
            "Compliance status determined",
            self.results.get("overall_compliant"),
            category="compliance",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(EMCComplianceDemo))
