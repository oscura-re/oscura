#!/usr/bin/env python3
"""Spectral Analysis Basics: FFT and frequency domain analysis.

This demo demonstrates fundamental spectral analysis techniques:
- FFT computation and visualization
- Power Spectral Density (PSD)
- Total Harmonic Distortion (THD)
- Harmonic analysis and detection
- Frequency peak identification
- Signal-to-Noise Ratio (SNR)

IEEE Standards: IEEE 1241-2010 (ADC testing), IEEE 1057 (Digitizing waveform recorders)
Related demos:
- 01_waveform_basics.py - Time domain measurements
- ../04_advanced_analysis/03_fft_spectral.py - Advanced spectral techniques
- ../12_standards_compliance/spectral_compliance.py - IEEE compliance

Usage:
    python demos/02_basic_analysis/03_spectral_basics.py
    python demos/02_basic_analysis/03_spectral_basics.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import add_noise, generate_complex_signal, generate_sine_wave
from demos.common.formatting import print_subheader


class SpectralBasicsDemo(BaseDemo):
    """Fundamental spectral analysis demonstration."""

    name = "Spectral Analysis Basics"
    description = "FFT, PSD, THD, and harmonic analysis fundamentals"
    category = "basic_analysis"

    capabilities = [
        "oscura.fft",
        "oscura.psd",
        "oscura.thd",
        "oscura.find_peaks",
        "oscura.snr",
    ]

    ieee_standards = ["IEEE 1241-2010", "IEEE 1057"]

    related_demos = [
        "01_waveform_basics.py",
        "../04_advanced_analysis/03_fft_spectral.py",
        "../12_standards_compliance/spectral_compliance.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.pure_sine = None
        self.noisy_sine = None
        self.harmonic_signal = None

    def generate_data(self) -> None:
        """Generate test signals for spectral analysis."""
        print_info("Generating spectral test signals...")

        # 1. Pure sine wave (10 kHz)
        self.pure_sine = generate_sine_wave(
            frequency=10e3,  # 10 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=1e6,  # 1 MHz sampling
        )

        # 2. Noisy sine wave (same frequency, with noise)
        self.noisy_sine = add_noise(
            self.pure_sine,
            snr_db=40.0,  # 40 dB SNR
        )

        # 3. Signal with harmonics (fundamental + 2nd and 3rd harmonics)
        self.harmonic_signal = generate_complex_signal(
            fundamentals=[5e3, 10e3, 15e3],  # 5 kHz + harmonics at 10k, 15k
            amplitudes=[1.0, 0.3, 0.1],  # Fundamental + weaker harmonics
            duration=0.01,  # 10 ms
            sample_rate=1e6,  # 1 MHz sampling
        )

        print_result("Pure sine wave", "10 kHz, 2V peak")
        print_result("Noisy sine wave", "10 kHz, 2V peak, 40 dB SNR")
        print_result("Harmonic signal", "5 kHz fundamental + 2nd/3rd harmonics")

    def run_analysis(self) -> None:
        """Execute spectral analysis."""
        # ========== PART 1: FFT ANALYSIS ==========
        print_subheader("Part 1: FFT Analysis")
        print_info("Computing FFT of pure sine wave")

        # Compute FFT
        fft_result = osc.fft(self.pure_sine)
        frequencies = fft_result["frequencies"]
        magnitudes = fft_result["magnitudes"]

        # Find peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = frequencies[peak_idx]
        peak_mag = magnitudes[peak_idx]

        self.results["fft_peak_frequency"] = peak_freq
        self.results["fft_peak_magnitude"] = peak_mag
        self.results["fft_num_bins"] = len(frequencies)

        print_result("FFT bins", len(frequencies))
        print_result("Peak frequency", f"{peak_freq / 1e3:.3f} kHz")
        print_result("Peak magnitude", f"{peak_mag:.4f} V")

        # ========== PART 2: POWER SPECTRAL DENSITY ==========
        print_subheader("Part 2: Power Spectral Density (PSD)")
        print_info("Computing PSD for noisy signal")

        psd_result = osc.psd(self.noisy_sine)
        psd_freqs = psd_result["frequencies"]
        psd_power = psd_result["power"]

        # Find peak in PSD
        psd_peak_idx = np.argmax(psd_power)
        psd_peak_freq = psd_freqs[psd_peak_idx]
        psd_peak_power = psd_power[psd_peak_idx]

        self.results["psd_peak_frequency"] = psd_peak_freq
        self.results["psd_peak_power"] = psd_peak_power

        print_result("PSD peak frequency", f"{psd_peak_freq / 1e3:.3f} kHz")
        print_result("PSD peak power", f"{10 * np.log10(psd_peak_power):.2f} dB")

        # Calculate total power
        total_power = np.sum(psd_power) * (psd_freqs[1] - psd_freqs[0])
        self.results["total_power"] = total_power
        print_result("Total signal power", f"{total_power:.6f} W")

        # ========== PART 3: SNR MEASUREMENT ==========
        print_subheader("Part 3: Signal-to-Noise Ratio (SNR)")
        print_info("Measuring SNR of noisy signal")

        snr = osc.snr(self.noisy_sine)
        self.results["measured_snr"] = snr
        print_result("Measured SNR", f"{snr:.2f} dB")
        print_info(f"Expected SNR: 40 dB (measured: {snr:.2f} dB)")

        # ========== PART 4: HARMONIC ANALYSIS ==========
        print_subheader("Part 4: Harmonic Analysis")
        print_info("Detecting harmonics in complex signal")

        # Find frequency peaks
        harmonic_fft = osc.fft(self.harmonic_signal)
        peaks = osc.find_peaks(
            harmonic_fft["magnitudes"],
            height=0.05,  # Minimum peak height
            distance=100,  # Minimum separation in bins
        )

        peak_frequencies = harmonic_fft["frequencies"][peaks["indices"]]
        peak_magnitudes = harmonic_fft["magnitudes"][peaks["indices"]]

        self.results["num_harmonics"] = len(peak_frequencies)
        self.results["harmonic_frequencies"] = peak_frequencies.tolist()
        self.results["harmonic_magnitudes"] = peak_magnitudes.tolist()

        print_result("Harmonics detected", len(peak_frequencies))
        for i, (freq, mag) in enumerate(zip(peak_frequencies, peak_magnitudes, strict=False)):
            print_result(f"  Harmonic {i + 1}", f"{freq / 1e3:.3f} kHz @ {mag:.4f} V")

        # ========== PART 5: TOTAL HARMONIC DISTORTION ==========
        print_subheader("Part 5: Total Harmonic Distortion (THD)")
        print_info("Computing THD of harmonic signal")

        thd = osc.thd(self.harmonic_signal, fundamental_freq=5e3)
        self.results["thd"] = thd
        print_result("THD", f"{thd:.2f}%")

        # Calculate theoretical THD from known amplitudes
        # THD = sqrt(H2^2 + H3^2 + ...) / H1 * 100%
        # With amplitudes [1.0, 0.3, 0.1], THD should be sqrt(0.3^2 + 0.1^2)/1.0 * 100% ≈ 31.6%
        theoretical_thd = np.sqrt(0.3**2 + 0.1**2) / 1.0 * 100
        print_info(f"Theoretical THD: {theoretical_thd:.2f}% (measured: {thd:.2f}%)")

        # ========== PART 6: FREQUENCY RESOLUTION ==========
        print_subheader("Part 6: Frequency Resolution")
        print_info("Analyzing frequency resolution capabilities")

        sample_rate = self.pure_sine.metadata.sample_rate
        num_samples = len(self.pure_sine.data)
        freq_resolution = sample_rate / num_samples

        self.results["frequency_resolution"] = freq_resolution
        self.results["max_frequency"] = sample_rate / 2  # Nyquist frequency

        print_result("Frequency resolution", f"{freq_resolution:.2f} Hz")
        print_result("Maximum frequency (Nyquist)", f"{sample_rate / 2 / 1e3:.1f} kHz")
        print_result("Number of samples", num_samples)

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Measurement Interpretation")

        print_info("\n[FFT Analysis]")
        print_info(f"  Peak at {peak_freq / 1e3:.3f}kHz matches input (10kHz)")
        print_info(
            f"  Frequency resolution: {freq_resolution:.2f}Hz → accuracy ±{freq_resolution / 2:.1f}Hz"
        )

        print_info("\n[PSD Analysis]")
        print_info("  PSD reveals signal power distribution across frequencies")
        print_info(
            f"  Peak power: {10 * np.log10(psd_peak_power):.2f}dB at {psd_peak_freq / 1e3:.3f}kHz"
        )
        print_info(f"  Total power: {total_power:.6f}W")

        print_info("\n[Harmonic Analysis]")
        print_info(f"  Detected {len(peak_frequencies)} harmonics")
        print_info(f"  Fundamental: {peak_frequencies[0] / 1e3:.1f}kHz")
        print_info(
            f"  THD: {thd:.2f}% (quality: {'Good' if thd < 5 else 'Fair' if thd < 20 else 'Poor'})"
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate spectral analysis results."""
        # FFT validation - peak should be at 10 kHz
        suite.check_range("FFT peak frequency", self.results["fft_peak_frequency"], 9.9e3, 10.1e3)
        suite.check_range("FFT peak magnitude", self.results["fft_peak_magnitude"], 1.9, 2.1)

        # PSD validation
        suite.check_range("PSD peak frequency", self.results["psd_peak_frequency"], 9.9e3, 10.1e3)
        suite.check_range("Total power", self.results["total_power"], 0.01, 10.0)

        # SNR validation - should be around 40 dB
        suite.check_range("Measured SNR", self.results["measured_snr"], 35, 45)

        # Harmonic detection - should find 3 harmonics (5k, 10k, 15k)
        suite.check_range("Number of harmonics", self.results["num_harmonics"], 2, 4)

        # THD validation - theoretical is ~31.6%
        suite.check_range("THD", self.results["thd"], 25, 40)

        # Frequency resolution validation
        suite.check_range("Frequency resolution", self.results["frequency_resolution"], 50, 150)
        suite.check_equal("Max frequency", self.results["max_frequency"], 500e3)


if __name__ == "__main__":
    sys.exit(run_demo_main(SpectralBasicsDemo))
