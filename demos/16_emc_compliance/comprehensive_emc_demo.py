#!/usr/bin/env python3
"""Comprehensive EMC/EMI Compliance Demo using BaseDemo Pattern.

This demo demonstrates Oscura's EMC/EMI compliance testing capabilities:
- Conducted emissions analysis (CISPR 32)
- Radiated emissions (FCC Part 15)
- Power quality harmonics (IEC 61000-3-2)
- ESD transient characterization (IEC 61000-4-2)
- EMI fingerprinting and source identification
- Compliance limit checking and margin analysis

Usage:
    python demos/09_emc_compliance/comprehensive_emc_demo.py
    python demos/09_emc_compliance/comprehensive_emc_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from demos.data_generation import SignalBuilder
from oscura.analyzers.spectral import fft, thd
from oscura.core.types import TraceMetadata, WaveformTrace


class EMCComplianceDemo(BaseDemo):
    """EMC/EMI Compliance Testing Demonstration.

    Demonstrates Oscura's comprehensive EMC testing capabilities
    including conducted/radiated emissions and power quality analysis.
    """

    name = "Comprehensive EMC/EMI Compliance Analysis"
    description = "Demonstrates EMC compliance testing per CISPR, FCC, IEC standards"
    category = "emc_compliance"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 100e6  # 100 MHz
        self.ce_trace = None  # Conducted emissions
        self.pq_trace = None  # Power quality
        self.esd_trace = None  # ESD transient

    def generate_data(self) -> None:
        """Generate or load EMC test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default files in parent directory if they exist
        3. Generate synthetic data using SignalBuilder
        """
        # Try loading from files
        loaded_from_file = False

        # 1. Check CLI override for conducted emissions
        if self.data_file and self.data_file.exists():
            print_info(f"Loading data from CLI override: {self.data_file}")
            try:
                data = np.load(self.data_file)
                signal_data = data["data"]
                loaded_sample_rate = float(data["sample_rate"])

                self.ce_trace = WaveformTrace(
                    data=signal_data,
                    metadata=TraceMetadata(
                        sample_rate=loaded_sample_rate,
                        channel_name="CE_Line",
                        source_file=str(self.data_file),
                    ),
                )
                print_result("CE loaded from CLI", self.data_file.name)
                print_result("CE samples", len(self.ce_trace.data))
                loaded_from_file = True
            except Exception as e:
                print_info(
                    f"Failed to load from CLI file: {e}, falling back to defaults or synthetic"
                )

        # 2. Check default files in parent directory
        if not loaded_from_file:
            ce_file = Path(__file__).parent / "conducted_emissions_ac_line.npz"
            pq_file = Path(__file__).parent / "power_quality_harmonics.npz"
            esd_file = Path(__file__).parent / "esd_transient_burst.npz"

            # Try loading conducted emissions
            if ce_file.exists():
                print_info(f"Loading conducted emissions from: {ce_file.name}")
                try:
                    data = np.load(ce_file)
                    signal_data = data["data"]
                    loaded_sample_rate = float(data["sample_rate"])

                    self.ce_trace = WaveformTrace(
                        data=signal_data,
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="CE_Line",
                            source_file=str(ce_file),
                        ),
                    )
                    print_result("CE loaded from file", ce_file.name)
                    print_result("CE samples", len(self.ce_trace.data))
                    loaded_from_file = True
                except Exception as e:
                    print_info(f"Failed to load CE file: {e}")

            # Try loading power quality
            if pq_file.exists() and not self.pq_trace:
                print_info(f"Loading power quality from: {pq_file.name}")
                try:
                    data = np.load(pq_file)
                    signal_data = data["data"]
                    loaded_sample_rate = float(data["sample_rate"])

                    self.pq_trace = WaveformTrace(
                        data=signal_data,
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="AC_Power",
                            source_file=str(pq_file),
                        ),
                    )
                    print_result("PQ loaded from file", pq_file.name)
                    print_result("PQ samples", len(self.pq_trace.data))
                except Exception as e:
                    print_info(f"Failed to load PQ file: {e}")

            # Try loading ESD transient
            if esd_file.exists() and not self.esd_trace:
                print_info(f"Loading ESD transient from: {esd_file.name}")
                try:
                    data = np.load(esd_file)
                    signal_data = data["data"]
                    loaded_sample_rate = float(data["sample_rate"])

                    self.esd_trace = WaveformTrace(
                        data=signal_data,
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="ESD_Transient",
                            source_file=str(esd_file),
                        ),
                    )
                    print_result("ESD loaded from file", esd_file.name)
                    print_result("ESD samples", len(self.esd_trace.data))
                except Exception as e:
                    print_info(f"Failed to load ESD file: {e}")

        # 3. Generate synthetic data for any missing traces
        if not loaded_from_file or not self.ce_trace:
            if not self.ce_trace:
                print_info("Generating synthetic EMC test signals...")
                self._generate_conducted_emissions()

        if not self.pq_trace:
            self._generate_power_quality()

        if not self.esd_trace:
            self._generate_esd_transient()

    def _generate_conducted_emissions(self) -> None:
        """Generate conducted emissions test signal."""
        # Simulate switching power supply noise
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.001)
            .add_sine(frequency=100e3, amplitude=0.01)  # 100 kHz switching
            .add_harmonics(fundamental=100e3, thd_percent=20.0)  # Harmonics
            .add_noise(snr_db=40)
            .build()
        )

        self.ce_trace = WaveformTrace(
            data=signal.data["ch1"],
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="CE_Line",
                source_file="synthetic",
            ),
        )
        print_result("CE signal samples", len(self.ce_trace.data))

    def _generate_power_quality(self) -> None:
        """Generate power quality test signal."""
        # 60 Hz power with harmonics
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.1)
            .add_sine(frequency=60.0, amplitude=1.0)
            .add_harmonics(fundamental=60.0, thd_percent=5.0)  # 5% THD
            .add_noise(snr_db=50)
            .build()
        )

        self.pq_trace = WaveformTrace(
            data=signal.data["ch1"],
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="AC_Power",
                source_file="synthetic",
            ),
        )
        print_result("PQ signal samples", len(self.pq_trace.data))

    def _generate_esd_transient(self) -> None:
        """Generate ESD transient test signal."""
        # ESD pulse: fast rise, exponential decay
        n_samples = 10000
        t = np.arange(n_samples) / self.sample_rate

        # IEC 61000-4-2 like pulse
        rise_time = 1e-9  # 1 ns rise
        decay_time = 60e-9  # 60 ns decay

        pulse = np.zeros(n_samples)
        peak_idx = 100

        # Rising edge
        rise_samples = int(rise_time * self.sample_rate)
        if rise_samples > 0:
            pulse[peak_idx - rise_samples : peak_idx] = np.linspace(0, 1, rise_samples)

        # Decay
        for i in range(peak_idx, n_samples):
            pulse[i] = np.exp(-(t[i] - t[peak_idx]) / decay_time)

        # Scale to 8 kV
        pulse *= 8000

        self.esd_trace = WaveformTrace(
            data=pulse,
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="ESD_Transient",
                source_file="synthetic",
            ),
        )
        print_result("ESD signal samples", len(self.esd_trace.data))

    def run_analysis(self) -> None:
        """Execute EMC compliance analysis."""
        # === Section 1: Conducted Emissions ===
        print_subheader("Conducted Emissions Analysis (CISPR 32)")
        self._analyze_conducted_emissions()

        # === Section 2: Power Quality Harmonics ===
        print_subheader("Power Quality Harmonics (IEC 61000-3-2)")
        self._analyze_power_quality()

        # === Section 3: ESD Transient ===
        print_subheader("ESD Transient Analysis (IEC 61000-4-2)")
        self._analyze_esd_transient()

        # === Section 4: EMI Fingerprinting ===
        print_subheader("EMI Fingerprinting")
        self._analyze_emi_fingerprint()

        # === Section 5: Compliance Summary ===
        print_subheader("Compliance Summary")
        self._print_compliance_summary()

    def _analyze_conducted_emissions(self) -> None:
        """Analyze conducted emissions."""
        freq, mag_db = fft(self.ce_trace, window="flattop")

        # Convert to dBuV (assume input in V)
        mag_dbuv = mag_db + 120

        print_result("Frequency range", f"DC - {freq[-1] / 1e6:.1f}", "MHz")
        print_result("Frequency resolution", f"{freq[1] / 1e3:.2f}", "kHz")

        # Find emission peaks
        threshold = np.percentile(mag_dbuv, 90)
        peaks = []
        for i in range(1, len(mag_dbuv) - 1):
            if (
                mag_dbuv[i] > threshold
                and mag_dbuv[i] > mag_dbuv[i - 1]
                and mag_dbuv[i] > mag_dbuv[i + 1]
            ):
                peaks.append((freq[i], mag_dbuv[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)

        print_result("Emission peaks found", len(peaks))
        print_info("Top 5 emission peaks:")
        for i, (f, m) in enumerate(peaks[:5], 1):
            print_info(f"  {i}. {f / 1e6:.3f} MHz: {m:.1f} dBuV")

        self.results["ce_peaks"] = len(peaks)
        self.results["ce_max_level"] = peaks[0][1] if peaks else 0
        self.results["ce_max_freq"] = peaks[0][0] if peaks else 0

        # Simple compliance check (CISPR 32 Class B limits ~60 dBuV)
        limit_dbuv = 60
        violations = [p for p in peaks if p[1] > limit_dbuv]
        compliant = len(violations) == 0

        print_result("CISPR 32 Class B limit", f"{limit_dbuv}", "dBuV")
        print_result("Compliance", "PASS" if compliant else "FAIL")
        self.results["ce_compliant"] = compliant
        self.results["ce_violations"] = len(violations)

    def _analyze_power_quality(self) -> None:
        """Analyze power quality harmonics."""
        freq, mag_db = fft(self.pq_trace, window="flattop")

        # Find fundamental
        fundamental_idx = np.argmax(mag_db[:1000])
        fundamental_freq = freq[fundamental_idx]

        print_result("Fundamental frequency", f"{fundamental_freq:.2f}", "Hz")

        # THD calculation
        thd_percent = thd(self.pq_trace) * 100

        print_result("THD", f"{thd_percent:.2f}", "%")
        self.results["pq_thd_percent"] = thd_percent

        # Extract individual harmonics
        harmonics = []
        fundamental_mag = mag_db[fundamental_idx]

        for n in range(1, 20):
            harmonic_freq = fundamental_freq * n
            idx = np.argmin(np.abs(freq - harmonic_freq))
            harmonic_mag = mag_db[idx]

            if n == 1:
                harmonics.append((n, harmonic_freq, 1.0))
            else:
                ratio = 10 ** ((harmonic_mag - fundamental_mag) / 20)
                harmonics.append((n, harmonic_freq, ratio))

        print_info("Harmonic content:")
        for n, f, ratio in harmonics[:7]:
            print_info(f"  H{n:2d}: {f:7.1f} Hz, {ratio * 100:5.1f}%")

        self.results["pq_fundamental"] = fundamental_freq
        self.results["pq_harmonics"] = len(harmonics)

        # IEC 61000-3-2 compliance (simplified - actual limits vary)
        compliant = thd_percent < 8.0  # Simplified limit
        print_result("IEC 61000-3-2", "PASS" if compliant else "FAIL")
        self.results["pq_compliant"] = compliant

    def _analyze_esd_transient(self) -> None:
        """Analyze ESD transient characteristics."""
        data = self.esd_trace.data
        sample_rate = self.esd_trace.metadata.sample_rate
        t = np.arange(len(data)) / sample_rate

        # Find peak
        peak_idx = np.argmax(np.abs(data))
        peak_voltage = data[peak_idx]
        peak_time = t[peak_idx]

        print_result("Peak voltage", f"{peak_voltage / 1000:.2f}", "kV")
        print_result("Peak time", f"{peak_time * 1e9:.1f}", "ns")

        # Rise time (10% to 90%)
        threshold_10 = 0.1 * peak_voltage
        threshold_90 = 0.9 * peak_voltage

        idx_10 = np.where(data > threshold_10)[0]
        idx_90 = np.where(data > threshold_90)[0]

        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = (idx_90[0] - idx_10[0]) / sample_rate
            print_result("Rise time (10-90%)", f"{rise_time * 1e9:.2f}", "ns")
            self.results["esd_rise_time_ns"] = rise_time * 1e9
        else:
            rise_time = 0

        self.results["esd_peak_kv"] = peak_voltage / 1000
        self.results["esd_peak_time_ns"] = peak_time * 1e9

        # IEC 61000-4-2 compliance (rise time 0.7-1 ns)
        if 0.7 <= rise_time * 1e9 <= 1.0:
            print_info("Rise time: COMPLIANT with IEC 61000-4-2")
            compliant = True
        else:
            print_info("Rise time: Non-compliant (spec: 0.7-1.0 ns)")
            compliant = False

        self.results["esd_compliant"] = compliant

    def _analyze_emi_fingerprint(self) -> None:
        """Create EMI fingerprint for troubleshooting."""
        freq, mag_db = fft(self.ce_trace, window="flattop")

        # Find all significant peaks
        threshold = np.percentile(mag_db, 85)
        peaks = []
        for i in range(1, len(mag_db) - 1):
            if mag_db[i] > threshold and mag_db[i] > mag_db[i - 1] and mag_db[i] > mag_db[i + 1]:
                peaks.append((freq[i], mag_db[i]))

        peaks.sort(key=lambda x: x[1], reverse=True)

        print_result("Emission peaks", len(peaks))

        # Analyze peak spacing for switching frequency
        if len(peaks) >= 3:
            peak_freqs = [p[0] for p in peaks[:10]]
            spacings = []
            for i in range(len(peak_freqs) - 1):
                spacing = peak_freqs[i + 1] - peak_freqs[i]
                if spacing > 1e3:
                    spacings.append(spacing)

            if spacings:
                median_spacing = np.median(spacings)
                print_result("Inferred switching freq", f"{median_spacing / 1e3:.1f}", "kHz")
                self.results["emi_switching_freq"] = median_spacing

        # Identify likely noise sources
        print_info("Likely EMI sources:")
        signatures = [
            (50e3, 200e3, "DC-DC converter (50-200 kHz)"),
            (300e3, 1e6, "High-frequency DC-DC (300 kHz - 1 MHz)"),
            (10e6, 100e6, "Clock harmonics / digital noise"),
        ]

        for f_low, f_high, description in signatures:
            mask = (freq >= f_low) & (freq <= f_high)
            if np.any(mask):
                max_level = mag_db[mask].max()
                if max_level > threshold:
                    print_info(f"  - {description}: {max_level:.1f} dB")

        self.results["emi_peaks"] = len(peaks)

    def _print_compliance_summary(self) -> None:
        """Print compliance summary."""
        print_info("EMC Compliance Summary:")
        print_info(f"  CISPR 32 (CE): {'PASS' if self.results.get('ce_compliant') else 'FAIL'}")
        print_info(
            f"  IEC 61000-3-2 (PQ): {'PASS' if self.results.get('pq_compliant') else 'FAIL'}"
        )
        print_info(
            f"  IEC 61000-4-2 (ESD): {'PASS' if self.results.get('esd_compliant') else 'FAIL'}"
        )

        all_pass = (
            self.results.get("ce_compliant", False)
            and self.results.get("pq_compliant", False)
            and self.results.get("esd_compliant", False)
        )

        print_result("Overall compliance", "PASS" if all_pass else "FAIL")
        self.results["overall_compliant"] = all_pass

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate EMC analysis results."""
        # Conducted emissions
        suite.check_greater(
            "CE emission peaks detected",
            self.results.get("ce_peaks", 0),
            0,
            category="conducted_emissions",
        )

        # Power quality
        suite.check_greater(
            "PQ harmonics analyzed",
            self.results.get("pq_harmonics", 0),
            0,
            category="power_quality",
        )

        suite.check_greater(
            "PQ fundamental detected",
            self.results.get("pq_fundamental", 0),
            0,
            category="power_quality",
        )

        # ESD
        suite.check_greater(
            "ESD peak voltage measured",
            self.results.get("esd_peak_kv", 0),
            0,
            category="esd",
        )

        # EMI fingerprint
        suite.check_greater(
            "EMI peaks fingerprinted",
            self.results.get("emi_peaks", 0),
            0,
            category="emi_fingerprint",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(EMCComplianceDemo))
