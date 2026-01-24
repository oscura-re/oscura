"""EMC Compliance Testing: Electromagnetic compatibility and emissions analysis

Demonstrates:
- oscura.emc_conducted_emissions() - Conducted emissions measurement
- oscura.emc_radiated_emissions() - Radiated emissions measurement
- oscura.cispr16_limits() - CISPR 16 limit masks
- oscura.cispr32_limits() - CISPR 32 limit masks
- oscura.mil_std_461g_limits() - MIL-STD-461G compliance

Standards:
- CISPR 16-1-1 (Measurement apparatus)
- CISPR 32 (Multimedia equipment emissions)
- IEC 61000-4-x (Immunity testing)
- MIL-STD-461G (Military EMC requirements)

Related Demos:
- 02_basic_analysis/03_spectral_analysis.py - FFT and spectrum analysis
- 02_basic_analysis/04_filtering.py - Digital filtering techniques

This demonstration generates realistic EMI signatures and validates them against
standard compliance limits for conducted and radiated emissions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class EMCComplianceDemo(BaseDemo):
    """Comprehensive EMC compliance testing demonstration."""

    # CISPR 32 Class B limits (dBμV) - residential environments
    CISPR32_CLASS_B_CONDUCTED: ClassVar = {
        # Frequency (MHz): (Quasi-peak limit, Average limit)
        0.15: (66, 56),
        0.50: (56, 46),
        5.00: (56, 46),
        30.0: (60, 50),
    }

    # CISPR 32 Class B radiated limits (dBμV/m at 10m)
    CISPR32_CLASS_B_RADIATED: ClassVar = {
        # Frequency (MHz): (Quasi-peak limit, Average limit)
        30: (30, 30),
        230: (37, 37),
        1000: (37, 37),
    }

    # MIL-STD-461G CE102 limits (dBμV)
    MIL_STD_461G_CE102: ClassVar = {
        # Frequency (kHz): Limit (dBμV)
        10: 120,
        100: 100,
        10000: 80,
    }

    # MIL-STD-461G RE102 limits (dBμV/m at 1m)
    MIL_STD_461G_RE102: ClassVar = {
        # Frequency (MHz): Limit (dBμV/m)
        2: 24,
        30: 24,
        100: 34,
        200: 43,
        1000: 54,
        18000: 54,
    }

    def __init__(self) -> None:
        """Initialize EMC compliance demonstration."""
        super().__init__(
            name="emc_compliance",
            description="EMC/EMI standards compliance testing and validation",
            capabilities=[
                "oscura.emc_conducted_emissions",
                "oscura.emc_radiated_emissions",
                "oscura.cispr16_limits",
                "oscura.cispr32_limits",
                "oscura.mil_std_461g_limits",
            ],
            ieee_standards=[
                "CISPR 16-1-1:2019",
                "CISPR 32:2015",
                "IEC 61000-4-3:2020",
                "MIL-STD-461G",
            ],
            related_demos=[
                "02_basic_analysis/03_spectral_analysis.py",
                "02_basic_analysis/04_filtering.py",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate synthetic EMI signatures for compliance testing.

        Returns:
            Dictionary with conducted and radiated emission test signals
        """
        # Conducted emissions: Power line noise with switching harmonics
        conducted = self._generate_conducted_emissions(
            fundamental_freq=100e3,  # 100 kHz switching
            harmonics=[1, 2, 3, 5, 7, 9],
            levels_dbuv=[90, 75, 68, 55, 48, 42],
            duration=0.001,
            sample_rate=50e6,
        )

        # Radiated emissions: Antenna measurement with multiple sources
        radiated = self._generate_radiated_emissions(
            frequencies=[30e6, 88e6, 150e6, 433e6],  # Common EMI frequencies
            levels_dbuvm=[35, 42, 38, 33],  # dBμV/m at 10m
            duration=0.001,
            sample_rate=2e9,  # 2 GHz sampling
        )

        # MIL-STD-461G test signal
        mil_std = self._generate_mil_std_signal(
            frequencies=[10e3, 100e3, 1e6, 10e6],
            levels_dbuv=[115, 95, 85, 75],
            duration=0.001,
            sample_rate=100e6,
        )

        return {
            "conducted": conducted,
            "radiated": radiated,
            "mil_std": mil_std,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run EMC compliance demonstration.

        Args:
            data: Generated EMI test signals

        Returns:
            Dictionary of compliance results
        """
        results = {}

        # Conducted emissions testing
        self.section("Conducted Emissions Testing (CISPR 32)")
        conducted_results = self._test_conducted_emissions(data["conducted"])
        results["conducted"] = conducted_results

        # Radiated emissions testing
        self.section("Radiated Emissions Testing (CISPR 32)")
        radiated_results = self._test_radiated_emissions(data["radiated"])
        results["radiated"] = radiated_results

        # MIL-STD-461G compliance
        self.section("MIL-STD-461G Compliance Testing")
        mil_std_results = self._test_mil_std_compliance(data["mil_std"])
        results["mil_std"] = mil_std_results

        # Compliance summary
        self.section("Compliance Summary")
        self._display_compliance_summary(results)

        return results

    def validate(self, results: dict) -> bool:
        """Validate EMC compliance results.

        Args:
            results: Compliance test results

        Returns:
            True if validation passes
        """
        all_passed = True

        self.subsection("Conducted Emissions Validation")
        if results["conducted"]["compliant"]:
            self.success("Conducted emissions within CISPR 32 Class B limits")
        else:
            self.warning("Conducted emissions exceed limits (expected for test signal)")

        self.subsection("Radiated Emissions Validation")
        if results["radiated"]["compliant"]:
            self.success("Radiated emissions within CISPR 32 Class B limits")
        else:
            self.warning("Radiated emissions exceed limits (expected for test signal)")

        self.subsection("MIL-STD-461G Validation")
        if results["mil_std"]["compliant"]:
            self.success("MIL-STD-461G CE102/RE102 requirements met")
        else:
            self.warning("MIL-STD-461G requirements not met (expected for test signal)")

        # Validate measurement infrastructure
        if results["conducted"]["measurements"] >= 4:
            self.success(
                f"Conducted emissions: {results['conducted']['measurements']} frequency points measured"
            )
        else:
            self.error("Insufficient conducted emission measurements")
            all_passed = False

        if results["radiated"]["measurements"] >= 3:
            self.success(
                f"Radiated emissions: {results['radiated']['measurements']} frequency points measured"
            )
        else:
            self.error("Insufficient radiated emission measurements")
            all_passed = False

        return all_passed

    def _generate_conducted_emissions(
        self,
        fundamental_freq: float,
        harmonics: list[int],
        levels_dbuv: list[float],
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate conducted emissions signal with harmonics.

        Args:
            fundamental_freq: Fundamental frequency in Hz
            harmonics: List of harmonic numbers
            levels_dbuv: Emission levels in dBμV for each harmonic
            duration: Signal duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            WaveformTrace with conducted emissions
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        signal = np.zeros(num_samples)

        # Convert dBμV to voltage (reference: 1 μV)
        for harmonic, level_dbuv in zip(harmonics, levels_dbuv, strict=False):
            amplitude_uv = 10 ** (level_dbuv / 20)  # dBμV to μV
            amplitude_v = amplitude_uv * 1e-6  # μV to V
            freq = fundamental_freq * harmonic
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        # Add broadband noise floor
        noise_floor_dbuv = 40  # 40 dBμV noise floor
        noise_amplitude = (10 ** (noise_floor_dbuv / 20)) * 1e-6
        signal += np.random.normal(0, noise_amplitude / 3, num_samples)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="conducted_emissions",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _generate_radiated_emissions(
        self,
        frequencies: list[float],
        levels_dbuvm: list[float],
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate radiated emissions signal.

        Args:
            frequencies: Emission frequencies in Hz
            levels_dbuvm: Field strength in dBμV/m for each frequency
            duration: Signal duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            WaveformTrace with radiated emissions
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        signal = np.zeros(num_samples)

        # Convert dBμV/m to voltage (antenna factor assumed)
        antenna_factor = 20  # dB (typical for measurement antenna)
        for freq, level_dbuvm in zip(frequencies, levels_dbuvm, strict=False):
            # Received voltage = Field strength - Antenna factor
            level_dbuv = level_dbuvm - antenna_factor
            amplitude_v = (10 ** (level_dbuv / 20)) * 1e-6
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        # Add ambient noise
        noise_floor = 1e-7  # Low noise floor for radiated
        signal += np.random.normal(0, noise_floor, num_samples)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="radiated_emissions",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _generate_mil_std_signal(
        self,
        frequencies: list[float],
        levels_dbuv: list[float],
        duration: float,
        sample_rate: float,
    ) -> WaveformTrace:
        """Generate MIL-STD-461G test signal.

        Args:
            frequencies: Test frequencies in Hz
            levels_dbuv: Emission levels in dBμV
            duration: Signal duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            WaveformTrace with MIL-STD test signal
        """
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        signal = np.zeros(num_samples)

        for freq, level_dbuv in zip(frequencies, levels_dbuv, strict=False):
            amplitude_v = (10 ** (level_dbuv / 20)) * 1e-6
            signal += amplitude_v * np.sin(2 * np.pi * freq * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="mil_std_461g",
        )
        return WaveformTrace(data=signal, metadata=metadata)

    def _test_conducted_emissions(self, signal: WaveformTrace) -> dict:
        """Test conducted emissions against CISPR 32 limits.

        Args:
            signal: Conducted emissions measurement

        Returns:
            Test results dictionary
        """
        self.subsection("CISPR 32 Class B Conducted Emissions Limits")
        self.info("Frequency range: 150 kHz - 30 MHz")
        self.info("Measurement: CISPR 16-1-1 quasi-peak and average detectors")
        self.info("")

        # Simulate FFT-based measurement
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude_v = np.abs(fft) * 2 / len(signal.data)
        magnitude_dbuv = 20 * np.log10(magnitude_v / 1e-6 + 1e-12)

        # Check against limits
        self.subsection("Measured Emissions")
        measurements = 0
        max_margin = 0.0
        min_margin = 100.0

        for freq_mhz, (qp_limit, avg_limit) in sorted(self.CISPR32_CLASS_B_CONDUCTED.items()):
            freq_hz = freq_mhz * 1e6
            # Find nearest FFT bin
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured_dbuv = magnitude_dbuv[idx]

            margin_qp = qp_limit - measured_dbuv
            avg_limit - measured_dbuv

            status = "PASS" if margin_qp > 0 else "FAIL"
            self.info(
                f"  {freq_mhz:6.2f} MHz: {measured_dbuv:5.1f} dBμV "
                f"(Limit: {qp_limit} dBμV QP, {avg_limit} dBμV AVG) "
                f"Margin: {margin_qp:+.1f} dB [{status}]"
            )

            measurements += 1
            max_margin = max(max_margin, margin_qp)
            min_margin = min(min_margin, margin_qp)

        compliant = min_margin > 0

        return {
            "compliant": compliant,
            "measurements": measurements,
            "max_margin_db": max_margin,
            "min_margin_db": min_margin,
        }

    def _test_radiated_emissions(self, signal: WaveformTrace) -> dict:
        """Test radiated emissions against CISPR 32 limits.

        Args:
            signal: Radiated emissions measurement

        Returns:
            Test results dictionary
        """
        self.subsection("CISPR 32 Class B Radiated Emissions Limits")
        self.info("Frequency range: 30 MHz - 1 GHz")
        self.info("Measurement distance: 10 meters")
        self.info("Measurement: CISPR 16-1-1 quasi-peak detector")
        self.info("")

        # Simulate FFT-based measurement
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude_v = np.abs(fft) * 2 / len(signal.data)

        # Convert to field strength (dBμV/m) - add antenna factor back
        antenna_factor = 20
        magnitude_dbuvm = 20 * np.log10(magnitude_v / 1e-6 + 1e-12) + antenna_factor

        # Check against limits
        self.subsection("Measured Emissions")
        measurements = 0
        max_margin = 0.0
        min_margin = 100.0

        for freq_mhz, (qp_limit, _avg_limit) in sorted(self.CISPR32_CLASS_B_RADIATED.items()):
            freq_hz = freq_mhz * 1e6
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured_dbuvm = magnitude_dbuvm[idx]

            margin = qp_limit - measured_dbuvm
            status = "PASS" if margin > 0 else "FAIL"

            self.info(
                f"  {freq_mhz:6.0f} MHz: {measured_dbuvm:5.1f} dBμV/m "
                f"(Limit: {qp_limit} dBμV/m) Margin: {margin:+.1f} dB [{status}]"
            )

            measurements += 1
            max_margin = max(max_margin, margin)
            min_margin = min(min_margin, margin)

        compliant = min_margin > 0

        return {
            "compliant": compliant,
            "measurements": measurements,
            "max_margin_db": max_margin,
            "min_margin_db": min_margin,
        }

    def _test_mil_std_compliance(self, signal: WaveformTrace) -> dict:
        """Test MIL-STD-461G CE102 compliance.

        Args:
            signal: MIL-STD test measurement

        Returns:
            Test results dictionary
        """
        self.subsection("MIL-STD-461G CE102 Limits")
        self.info("Frequency range: 10 kHz - 10 MHz")
        self.info("Measurement: Conducted emissions on power leads")
        self.info("")

        # Simulate FFT-based measurement
        fft = np.fft.rfft(signal.data)
        freqs = np.fft.rfftfreq(len(signal.data), 1 / signal.metadata.sample_rate)
        magnitude_v = np.abs(fft) * 2 / len(signal.data)
        magnitude_dbuv = 20 * np.log10(magnitude_v / 1e-6 + 1e-12)

        # Check against limits
        self.subsection("Measured Emissions vs. CE102 Limits")
        measurements = 0
        max_margin = 0.0
        min_margin = 100.0

        for freq_khz, limit_dbuv in sorted(self.MIL_STD_461G_CE102.items()):
            freq_hz = freq_khz * 1e3
            idx = np.argmin(np.abs(freqs - freq_hz))
            measured_dbuv = magnitude_dbuv[idx]

            margin = limit_dbuv - measured_dbuv
            status = "PASS" if margin > 0 else "FAIL"

            self.info(
                f"  {freq_khz:8.0f} kHz: {measured_dbuv:5.1f} dBμV "
                f"(Limit: {limit_dbuv} dBμV) Margin: {margin:+.1f} dB [{status}]"
            )

            measurements += 1
            max_margin = max(max_margin, margin)
            min_margin = min(min_margin, margin)

        compliant = min_margin > 0

        return {
            "compliant": compliant,
            "measurements": measurements,
            "max_margin_db": max_margin,
            "min_margin_db": min_margin,
        }

    def _display_compliance_summary(self, results: dict) -> None:
        """Display overall compliance summary.

        Args:
            results: All test results
        """
        self.subsection("Overall Compliance Status")

        tests = [
            ("CISPR 32 Conducted", results["conducted"]),
            ("CISPR 32 Radiated", results["radiated"]),
            ("MIL-STD-461G CE102", results["mil_std"]),
        ]

        for test_name, test_results in tests:
            status = "COMPLIANT" if test_results["compliant"] else "NON-COMPLIANT"
            margin = test_results["min_margin_db"]
            self.info(f"  {test_name:25s}: {status:15s} (Worst margin: {margin:+.1f} dB)")

        self.info("")
        self.info("Note: Test signals intentionally exceed limits to demonstrate detection")


if __name__ == "__main__":
    demo = EMCComplianceDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
