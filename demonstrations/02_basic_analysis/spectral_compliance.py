#!/usr/bin/env python3
"""Comprehensive Spectral and Compliance Demonstration using BaseDemo Pattern.

This demo demonstrates Oscura's spectral analysis and IEEE 1241-2010
compliance validation capabilities:
- FFT with proper windowing
- Power Spectral Density (PSD, Welch, Bartlett)
- THD, SNR, SINAD, ENOB, SFDR (IEEE 1241-2010)
- Spectrogram time-frequency analysis
- Compliance validation

Usage:
    python demos/06_spectral_compliance/comprehensive_spectral_demo.py
    python demos/06_spectral_compliance/comprehensive_spectral_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import print_subheader
from demonstrations.common import SignalBuilder
from oscura.analyzers.waveform.spectral import (
    bartlett_psd,
    clear_fft_cache,
    enob,
    fft,
    get_fft_cache_stats,
    periodogram,
    psd,
    sfdr,
    sinad,
    snr,
    spectrogram,
    thd,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class SpectralComplianceDemo(BaseDemo):
    """Spectral Analysis and IEEE 1241-2010 Compliance Demonstration.

    Demonstrates Oscura's comprehensive spectral analysis capabilities
    including FFT, PSD, and IEEE 1241-2010 quality metrics.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="comprehensive_spectral_and_compliance_analysis",
            description="Demonstrates spectral analysis and IEEE 1241-2010 compliance",
            capabilities=["oscura.spectral_analysis", "oscura.ieee1241_metrics"],
            ieee_standards=["IEEE 1241-2010"],
            **kwargs,
        )
        self.sample_rate = 1e6  # 1 MHz
        self.signal_freq = 1000.0  # 1 kHz fundamental
        self.trace = None

    def generate_test_data(self) -> dict:
        """Generate synthetic spectral test signal data using SignalBuilder."""
        print_info("Generating synthetic test signal with harmonics...")

        # Create signal with harmonics for THD analysis
        signal = (
            SignalBuilder(sample_rate=self.sample_rate, duration=0.01)
            .add_sine(frequency=self.signal_freq, amplitude=1.0)
            .add_harmonics(fundamental=self.signal_freq, thd_percent=1.0)
            .add_noise(snr_db=60)
            .build()
        )

        self.trace = WaveformTrace(
            data=signal["ch1"],
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="Spectral_Test",
                source_file="synthetic",
            ),
        )

        print_result("Sample rate", f"{self.sample_rate / 1e6:.1f}", "MHz")
        print_result("Samples", len(self.trace.data))
        print_result("Fundamental", f"{self.signal_freq:.0f}", "Hz")

        return {"trace": self.trace}

    def run_demonstration(self, data: dict) -> dict:
        """Execute spectral analysis."""
        self.trace = data["trace"]

        # === Section 1: FFT Analysis ===
        print_subheader("FFT Analysis")
        self._analyze_fft()

        # === Section 2: Power Spectral Density ===
        print_subheader("Power Spectral Density")
        self._analyze_psd()

        # === Section 3: IEEE 1241-2010 Quality Metrics ===
        print_subheader("IEEE 1241-2010 Quality Metrics")
        self._analyze_quality_metrics()

        # === Section 4: Spectrogram ===
        print_subheader("Time-Frequency Analysis")
        self._analyze_spectrogram()

        # === Section 5: Compliance Validation ===
        print_subheader("IEEE 1241-2010 Compliance")
        self._validate_compliance()

        return self.results

    def _analyze_fft(self) -> None:
        """Perform FFT analysis."""
        clear_fft_cache()

        freq, mag_db = fft(self.trace, window="hann", detrend="mean")

        # Find fundamental peak
        dc_bins = 5
        peak_idx = dc_bins + np.argmax(mag_db[dc_bins:])
        fundamental_freq = freq[peak_idx]
        fundamental_mag = mag_db[peak_idx]

        print_result("FFT bins", len(freq))
        print_result("Frequency resolution", f"{freq[1] - freq[0]:.2f}", "Hz")
        print_result("Fundamental detected", f"{fundamental_freq:.1f}", "Hz")
        print_result("Fundamental magnitude", f"{fundamental_mag:.1f}", "dB")

        self.results["fft_bins"] = len(freq)
        self.results["fundamental_freq"] = fundamental_freq
        self.results["fundamental_mag"] = fundamental_mag

        # Cache statistics
        cache_stats = get_fft_cache_stats()
        print_info(f"FFT cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses")

    def _analyze_psd(self) -> None:
        """Perform Power Spectral Density analysis."""
        # Welch's method
        freq_welch, psd_welch = psd(self.trace, window="hann", scaling="density")
        print_result("Welch PSD bins", len(freq_welch))
        self.results["psd_welch_bins"] = len(freq_welch)

        # Periodogram
        freq_period, psd_period = periodogram(self.trace, window="hann", scaling="density")
        print_result("Periodogram bins", len(freq_period))
        self.results["psd_periodogram_bins"] = len(freq_period)

        # Bartlett's method
        if len(self.trace.data) >= 1024:
            freq_bartlett, psd_bartlett = bartlett_psd(self.trace, n_segments=8)
            print_result("Bartlett PSD bins", len(freq_bartlett))
            self.results["psd_bartlett_bins"] = len(freq_bartlett)
        else:
            print_info("Insufficient samples for Bartlett method")

    def _analyze_quality_metrics(self) -> None:
        """Compute IEEE 1241-2010 quality metrics."""
        # THD - Total Harmonic Distortion
        thd_db = thd(self.trace, n_harmonics=10, window="hann", return_db=True)
        thd_pct = thd(self.trace, n_harmonics=10, window="hann", return_db=False)

        if np.isfinite(thd_db):
            print_result("THD", f"{thd_db:.2f}", "dB")
            print_result("THD", f"{thd_pct:.4f}", "%")
            self.results["thd_db"] = thd_db
            self.results["thd_pct"] = thd_pct
        else:
            print_info("THD: Could not compute")

        # SNR - Signal-to-Noise Ratio
        snr_db = snr(self.trace, n_harmonics=10, window="hann")
        if np.isfinite(snr_db):
            print_result("SNR", f"{snr_db:.2f}", "dB")
            self.results["snr_db"] = snr_db
        else:
            print_info("SNR: Could not compute")

        # SINAD
        sinad_db = sinad(self.trace, window="hann")
        if np.isfinite(sinad_db):
            print_result("SINAD", f"{sinad_db:.2f}", "dB")
            self.results["sinad_db"] = sinad_db
        else:
            print_info("SINAD: Could not compute")

        # ENOB
        enob_bits = enob(self.trace, window="hann")
        if np.isfinite(enob_bits):
            print_result("ENOB", f"{enob_bits:.2f}", "bits")
            self.results["enob_bits"] = enob_bits
        else:
            print_info("ENOB: Could not compute")

        # SFDR
        sfdr_db = sfdr(self.trace, window="hann")
        if np.isfinite(sfdr_db):
            print_result("SFDR", f"{sfdr_db:.2f}", "dBc")
            self.results["sfdr_db"] = sfdr_db
        else:
            print_info("SFDR: Could not compute")

    def _analyze_spectrogram(self) -> None:
        """Perform time-frequency analysis."""
        if len(self.trace.data) < 1024:
            print_info("Insufficient samples for spectrogram")
            return

        t, f, Sxx = spectrogram(self.trace, window="hann")

        print_result("Time bins", len(t))
        print_result("Frequency bins", len(f))
        print_result("Time resolution", f"{t[1] - t[0] if len(t) > 1 else 0:.6f}", "s")
        print_result("Freq resolution", f"{f[1] - f[0] if len(f) > 1 else 0:.2f}", "Hz")

        self.results["spectrogram_time_bins"] = len(t)
        self.results["spectrogram_freq_bins"] = len(f)

    def _validate_compliance(self) -> None:
        """Validate IEEE 1241-2010 compliance."""
        compliant = True
        violations = []

        # Check oversampling ratio
        fundamental_freq = self.results.get("fundamental_freq", 0)
        if fundamental_freq > 0:
            nyquist = self.sample_rate / 2
            oversampling = nyquist / fundamental_freq

            print_result("Nyquist frequency", f"{nyquist / 1e3:.1f}", "kHz")
            print_result("Oversampling ratio", f"{oversampling:.1f}x")

            if oversampling < 5:
                compliant = False
                violations.append("Insufficient oversampling (<5x)")
                print_info("Warning: Undersampling detected")
            else:
                print_info("Adequate oversampling ratio")

        # Check coherent sampling
        if fundamental_freq > 0:
            n_cycles = (len(self.trace.data) / self.sample_rate) * fundamental_freq
            coherence_error = abs(n_cycles - round(n_cycles))

            if coherence_error < 0.01:
                print_info(f"Coherent sampling ({n_cycles:.1f} cycles)")
            else:
                print_info(f"Non-coherent sampling ({n_cycles:.2f} cycles)")

        # Check SFDR
        sfdr_value = self.results.get("sfdr_db", 0)
        if sfdr_value > 60:
            print_info(f"Adequate SFDR ({sfdr_value:.1f} dBc)")
        elif sfdr_value > 0:
            print_info(f"Low SFDR ({sfdr_value:.1f} dBc, recommend >60 dBc)")

        self.results["ieee1241_compliant"] = compliant
        self.results["ieee1241_violations"] = violations

        if compliant:
            print_info("Signal meets IEEE 1241-2010 guidelines")
        else:
            print_info(f"{len(violations)} compliance issues detected")

    def validate(self, results: dict) -> bool:
        """Validate spectral analysis results."""
        suite = ValidationSuite()

        # FFT computed
        fft_bins = results.get("fft_bins", 0)
        suite.add_check(
            "FFT bins computed",
            fft_bins > 0,
            f"Got {fft_bins} bins",
        )

        # Fundamental detected
        fundamental_freq = results.get("fundamental_freq", 0)
        suite.add_check(
            "Fundamental frequency detected",
            fundamental_freq > 0,
            f"Got {fundamental_freq} Hz",
        )

        # PSD computed
        psd_bins = results.get("psd_welch_bins", 0)
        suite.add_check(
            "PSD Welch computed",
            psd_bins > 0,
            f"Got {psd_bins} bins",
        )

        # Quality metrics
        if "thd_db" in results:
            thd_db = results["thd_db"]
            suite.add_check(
                "THD is negative dB",
                thd_db < 0,
                f"Got {thd_db} dB",
            )

        if "snr_db" in results:
            snr_db = results["snr_db"]
            suite.add_check(
                "SNR is positive",
                snr_db > 0,
                f"Got {snr_db} dB",
            )

        if "enob_bits" in results:
            enob_bits = results["enob_bits"]
            suite.add_check(
                "ENOB is positive",
                enob_bits > 0,
                f"Got {enob_bits} bits",
            )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(SpectralComplianceDemo))
