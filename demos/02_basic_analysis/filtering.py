#!/usr/bin/env python3
"""Signal Filtering: Comprehensive filtering techniques.

This demo demonstrates signal filtering capabilities:
- Low-pass filtering (remove high-frequency noise)
- High-pass filtering (remove DC offset and low-frequency drift)
- Band-pass filtering (isolate frequency band)
- Band-stop/notch filtering (remove specific frequencies)
- Moving average filtering
- Savitzky-Golay filtering

IEEE Standards: IEEE 181-2011 (Waveform measurement)
Related demos:
- 01_waveform_basics.py - Time domain measurements
- 03_spectral_basics.py - Frequency domain analysis
- ../04_advanced_analysis/04_filtering_advanced.py - Advanced techniques

Usage:
    python demos/02_basic_analysis/05_filtering.py
    python demos/02_basic_analysis/05_filtering.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import add_noise, generate_sine_wave
from demos.common.formatting import print_subheader


class FilteringDemo(BaseDemo):
    """Signal filtering techniques demonstration."""

    name = "Signal Filtering Techniques"
    description = "Low-pass, high-pass, band-pass, and notch filtering"
    category = "basic_analysis"

    capabilities: ClassVar[list[str]] = [
        "oscura.low_pass",
        "oscura.high_pass",
        "oscura.band_pass",
        "oscura.band_stop",
        "oscura.moving_average",
        "oscura.savgol_filter",
    ]

    ieee_standards: ClassVar[list[str]] = ["IEEE 181-2011"]

    related_demos: ClassVar[list[str]] = [
        "01_waveform_basics.py",
        "03_spectral_basics.py",
        "../04_advanced_analysis/04_filtering_advanced.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.noisy_signal = None
        self.dc_signal = None
        self.multi_freq = None
        self.powerline_noise = None

    def generate_data(self) -> None:
        """Generate test signals for filtering demonstrations."""
        print_info("Generating test signals with noise and interference...")

        # 1. Noisy 1 kHz sine wave (for low-pass filtering)
        clean_sine = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        self.noisy_signal = add_noise(clean_sine, snr_db=10.0)  # 10 dB SNR

        # 2. Signal with DC offset (for high-pass filtering)
        self.dc_signal = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz
            offset=2.0,  # 2V DC offset
        )

        # 3. Multi-frequency signal (for band-pass filtering)
        # Combine 1 kHz, 5 kHz, and 20 kHz components
        sig_1k = generate_sine_wave(1000.0, 0.5, 0.01, 100e3)
        sig_5k = generate_sine_wave(5000.0, 1.0, 0.01, 100e3)  # Target frequency
        sig_20k = generate_sine_wave(20000.0, 0.3, 0.01, 100e3)

        multi_data = sig_1k.data + sig_5k.data + sig_20k.data
        self.multi_freq = osc.WaveformTrace(
            data=multi_data,
            metadata=osc.TraceMetadata(sample_rate=100e3, channel_name="multi_freq"),
        )

        # 4. Signal with 60 Hz powerline interference (for notch filtering)
        signal = generate_sine_wave(5000.0, 1.0, 0.1, 100e3)  # 5 kHz signal
        noise = generate_sine_wave(60.0, 0.5, 0.1, 100e3)  # 60 Hz interference

        noise_data = signal.data + noise.data
        self.powerline_noise = osc.WaveformTrace(
            data=noise_data, metadata=osc.TraceMetadata(sample_rate=100e3, channel_name="powerline")
        )

        print_result("Noisy signal", "1 kHz + noise (SNR: 10 dB)")
        print_result("DC offset signal", "1 kHz + 2V DC")
        print_result("Multi-frequency", "1k + 5k + 20k Hz combined")
        print_result("Powerline interference", "5 kHz + 60 Hz noise")

    def run_analysis(self) -> None:
        """Execute filtering demonstrations."""
        # ========== PART 1: LOW-PASS FILTERING ==========
        print_subheader("Part 1: Low-Pass Filtering")
        print_info("Remove high-frequency noise from signal")

        # Original RMS
        original_rms = osc.rms(self.noisy_signal)
        print_result("Original RMS", f"{original_rms:.4f} V")

        # Apply low-pass filter (cutoff at 2 kHz)
        filtered_lp = osc.low_pass(self.noisy_signal, cutoff=2000.0)
        filtered_rms = osc.rms(filtered_lp)

        self.results["lp_original_rms"] = original_rms
        self.results["lp_filtered_rms"] = filtered_rms
        self.results["lp_noise_reduction"] = (1 - filtered_rms / original_rms) * 100

        print_result("Filtered RMS", f"{filtered_rms:.4f} V")
        print_result("Noise reduction", f"{self.results['lp_noise_reduction']:.1f}%")

        # ========== PART 2: HIGH-PASS FILTERING ==========
        print_subheader("Part 2: High-Pass Filtering")
        print_info("Remove DC offset and low-frequency drift")

        # Original mean (DC offset)
        original_mean = osc.mean(self.dc_signal)
        print_result("Original DC offset", f"{original_mean:.4f} V")

        # Apply high-pass filter (cutoff at 500 Hz)
        filtered_hp = osc.high_pass(self.dc_signal, cutoff=500.0)
        filtered_mean = osc.mean(filtered_hp)

        self.results["hp_original_mean"] = original_mean
        self.results["hp_filtered_mean"] = filtered_mean
        self.results["hp_dc_removal"] = abs(filtered_mean / original_mean) * 100

        print_result("Filtered DC offset", f"{filtered_mean:.6f} V")
        print_result("DC removal", f"{100 - self.results['hp_dc_removal']:.1f}%")

        # ========== PART 3: BAND-PASS FILTERING ==========
        print_subheader("Part 3: Band-Pass Filtering")
        print_info("Isolate 5 kHz component from multi-frequency signal")

        # Analyze original signal
        fft_original = osc.fft(self.multi_freq)
        freqs = fft_original["frequencies"]
        mags = fft_original["magnitudes"]

        # Find peaks in original
        peak_1k_idx = np.argmin(np.abs(freqs - 1000))
        peak_5k_idx = np.argmin(np.abs(freqs - 5000))
        peak_20k_idx = np.argmin(np.abs(freqs - 20000))

        print_result("1 kHz component", f"{mags[peak_1k_idx]:.4f} V")
        print_result("5 kHz component", f"{mags[peak_5k_idx]:.4f} V (target)")
        print_result("20 kHz component", f"{mags[peak_20k_idx]:.4f} V")

        # Apply band-pass filter (3-7 kHz to isolate 5 kHz)
        filtered_bp = osc.band_pass(self.multi_freq, low_cutoff=3000.0, high_cutoff=7000.0)

        # Analyze filtered signal
        fft_filtered = osc.fft(filtered_bp)
        mags_filt = fft_filtered["magnitudes"]

        self.results["bp_5k_original"] = mags[peak_5k_idx]
        self.results["bp_5k_filtered"] = mags_filt[peak_5k_idx]
        self.results["bp_1k_suppression"] = (1 - mags_filt[peak_1k_idx] / mags[peak_1k_idx]) * 100
        self.results["bp_20k_suppression"] = (
            1 - mags_filt[peak_20k_idx] / mags[peak_20k_idx]
        ) * 100

        print_result("After band-pass (5 kHz)", f"{mags_filt[peak_5k_idx]:.4f} V")
        print_result("1 kHz suppression", f"{self.results['bp_1k_suppression']:.1f}%")
        print_result("20 kHz suppression", f"{self.results['bp_20k_suppression']:.1f}%")

        # ========== PART 4: BAND-STOP (NOTCH) FILTERING ==========
        print_subheader("Part 4: Band-Stop/Notch Filtering")
        print_info("Remove 60 Hz powerline interference")

        # Analyze original
        fft_noise = osc.fft(self.powerline_noise)
        freqs_noise = fft_noise["frequencies"]
        mags_noise = fft_noise["magnitudes"]

        idx_60hz = np.argmin(np.abs(freqs_noise - 60))
        idx_5khz = np.argmin(np.abs(freqs_noise - 5000))

        print_result("60 Hz interference", f"{mags_noise[idx_60hz]:.4f} V")
        print_result("5 kHz signal", f"{mags_noise[idx_5khz]:.4f} V")

        # Apply notch filter (remove 55-65 Hz)
        filtered_notch = osc.band_stop(self.powerline_noise, low_cutoff=55.0, high_cutoff=65.0)

        # Analyze filtered
        fft_notched = osc.fft(filtered_notch)
        mags_notched = fft_notched["magnitudes"]

        self.results["notch_60hz_suppression"] = (
            1 - mags_notched[idx_60hz] / mags_noise[idx_60hz]
        ) * 100
        self.results["notch_5khz_preserved"] = (mags_notched[idx_5khz] / mags_noise[idx_5khz]) * 100

        print_result("60 Hz after notch", f"{mags_notched[idx_60hz]:.4f} V")
        print_result("60 Hz suppression", f"{self.results['notch_60hz_suppression']:.1f}%")
        print_result("5 kHz preservation", f"{self.results['notch_5khz_preserved']:.1f}%")

        # ========== PART 5: MOVING AVERAGE ==========
        print_subheader("Part 5: Moving Average Filter")
        print_info("Simple noise reduction using moving average")

        # Apply moving average
        filtered_ma = osc.moving_average(self.noisy_signal, window_size=10)
        ma_rms = osc.rms(filtered_ma)

        self.results["ma_rms"] = ma_rms
        self.results["ma_smoothing"] = (1 - ma_rms / original_rms) * 100

        print_result("Moving average RMS", f"{ma_rms:.4f} V")
        print_result("Smoothing effect", f"{self.results['ma_smoothing']:.1f}%")

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Filter Performance Summary")

        print_info("\n[Low-Pass Filter]")
        print_info(f"  Noise reduction: {self.results['lp_noise_reduction']:.1f}%")
        print_info("  Use case: Remove high-frequency noise above signal bandwidth")

        print_info("\n[High-Pass Filter]")
        print_info(f"  DC removal: {100 - self.results['hp_dc_removal']:.1f}%")
        print_info("  Use case: Remove DC offset and low-frequency drift")

        print_info("\n[Band-Pass Filter]")
        print_info(f"  Target preserved: {self.results['bp_5k_filtered']:.4f}V")
        print_info(
            f"  Out-of-band suppression: {min(self.results['bp_1k_suppression'], self.results['bp_20k_suppression']):.1f}%"
        )
        print_info("  Use case: Isolate specific frequency band")

        print_info("\n[Notch Filter]")
        print_info(f"  Interference suppression: {self.results['notch_60hz_suppression']:.1f}%")
        print_info(f"  Signal preservation: {self.results['notch_5khz_preserved']:.1f}%")
        print_info("  Use case: Remove powerline interference")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate filtering results."""
        # Low-pass filtering should reduce RMS (remove noise)
        suite.check_range("Low-pass noise reduction", self.results["lp_noise_reduction"], 5, 50)

        # High-pass filtering should remove DC
        suite.check_range("High-pass DC removal", 100 - self.results["hp_dc_removal"], 90, 100)

        # Band-pass should suppress out-of-band frequencies
        suite.check_range("Band-pass 1k suppression", self.results["bp_1k_suppression"], 70, 100)
        suite.check_range("Band-pass 20k suppression", self.results["bp_20k_suppression"], 70, 100)

        # Notch filter should suppress 60 Hz while preserving signal
        suite.check_range("Notch 60Hz suppression", self.results["notch_60hz_suppression"], 80, 100)
        suite.check_range(
            "Notch signal preservation", self.results["notch_5khz_preserved"], 85, 105
        )

        # Moving average should reduce noise
        suite.check_range("Moving average smoothing", self.results["ma_smoothing"], 5, 50)


if __name__ == "__main__":
    sys.exit(run_demo_main(FilteringDemo))
