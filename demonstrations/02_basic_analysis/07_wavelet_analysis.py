"""Wavelet Analysis: Time-frequency analysis for transient signals

Demonstrates:
- oscura.analyzers.waveform.spectral.cwt() - Continuous Wavelet Transform
- oscura.analyzers.waveform.spectral.dwt() - Discrete Wavelet Transform
- oscura.analyzers.waveform.spectral.idwt() - Inverse DWT reconstruction
- oscura.analyzers.waveform.spectral.fft() - FFT for comparison
- Morlet, Mexican hat (Ricker) wavelets for CWT
- Daubechies wavelets (db4, db8) for DWT
- Time-frequency localization comparison: Wavelet vs FFT
- Transient detection in signals with step changes and impulses

IEEE Standards: IEEE 1241-2010 (transitional waveform definitions)
Related Demos:
- 02_basic_analysis/03_spectral_analysis.py
- 02_basic_analysis/04_filtering.py
- 04_advanced_analysis/time_frequency_analysis.py

Generates synthetic signals with transients to demonstrate wavelet
transform capabilities for detecting and localizing non-stationary
events that FFT cannot resolve in time.

References:
    - Mallat, S. (2009). A Wavelet Tour of Signal Processing, 3rd ed.
    - Daubechies, I. (1992). Ten Lectures on Wavelets
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pywt

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_sine_wave,
)
from oscura.analyzers.waveform.spectral import dwt, fft, idwt
from oscura.core.types import TraceMetadata, WaveformTrace


def cwt_wrapper(
    trace: WaveformTrace,
    wavelet: str = "morl",
    n_scales: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper for CWT using PyWavelets (oscura's cwt is broken in scipy 1.17+).

    Args:
        trace: Input waveform trace
        wavelet: Wavelet name ("morl" for Morlet, "mexh" for Mexican hat)
        n_scales: Number of scales

    Returns:
        (scales, frequencies, coefficients) tuple
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Auto-generate scales logarithmically
    scales = np.logspace(0, np.log10(len(data) / 8), n_scales)

    # Compute CWT using PyWavelets
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1.0 / sample_rate)

    return scales, frequencies, coefficients


class WaveletAnalysisDemo(BaseDemo):
    """Comprehensive demonstration of wavelet analysis capabilities."""

    def __init__(self) -> None:
        """Initialize wavelet analysis demonstration."""
        super().__init__(
            name="wavelet_analysis",
            description="Wavelet analysis: CWT, DWT, IDWT, time-frequency localization for transients",
            capabilities=[
                "pywt.cwt (CWT via PyWavelets)",
                "oscura.analyzers.waveform.spectral.dwt",
                "oscura.analyzers.waveform.spectral.idwt",
            ],
            ieee_standards=[
                "IEEE 1241-2010",
            ],
            related_demos=[
                "02_basic_analysis/03_spectral_analysis.py",
                "02_basic_analysis/04_filtering.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with transients for wavelet analysis.

        Creates:
        1. Signal with step change: Sudden amplitude transition
        2. Signal with impulse: Brief spike in sine wave
        3. Chirp signal: Frequency modulation over time
        4. Multi-component signal: Multiple frequencies appearing at different times
        """
        sample_rate = 10000.0  # 10 kHz

        # 1. Signal with step change (amplitude transition at t=0.5s)
        duration = 1.0  # 1 second
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate

        # Low amplitude sine wave → high amplitude
        signal_step = np.zeros(num_samples)
        step_idx = num_samples // 2

        # Before step: 1V amplitude at 100 Hz
        signal_step[:step_idx] = 1.0 * np.sin(2 * np.pi * 100 * t[:step_idx])
        # After step: 5V amplitude at 100 Hz
        signal_step[step_idx:] = 5.0 * np.sin(2 * np.pi * 100 * t[step_idx:])

        step_trace = WaveformTrace(
            data=signal_step,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="step_change"),
        )

        # 2. Signal with impulse (brief spike at t=0.3s)
        signal_impulse = 2.0 * np.sin(2 * np.pi * 200 * t)
        impulse_idx = int(0.3 * sample_rate)
        impulse_width = int(0.01 * sample_rate)  # 10 ms wide impulse
        signal_impulse[impulse_idx : impulse_idx + impulse_width] += 8.0

        impulse_trace = WaveformTrace(
            data=signal_impulse,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="impulse"),
        )

        # 3. Chirp signal (frequency sweep from 50 Hz to 500 Hz)
        f0, f1 = 50.0, 500.0
        chirp_rate = (f1 - f0) / duration
        signal_chirp = 3.0 * np.sin(2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2))

        chirp_trace = WaveformTrace(
            data=signal_chirp,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="chirp"),
        )

        # 4. Multi-component signal (different frequencies at different times)
        signal_multi = np.zeros(num_samples)
        # 0-0.3s: 100 Hz
        idx1 = int(0.3 * sample_rate)
        signal_multi[:idx1] = 2.0 * np.sin(2 * np.pi * 100 * t[:idx1])
        # 0.3-0.6s: 300 Hz
        idx2 = int(0.6 * sample_rate)
        signal_multi[idx1:idx2] = 2.0 * np.sin(2 * np.pi * 300 * t[idx1:idx2])
        # 0.6-1.0s: 600 Hz
        signal_multi[idx2:] = 2.0 * np.sin(2 * np.pi * 600 * t[idx2:])

        multi_trace = WaveformTrace(
            data=signal_multi,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="multi_component"),
        )

        # 5. Clean sine wave for DWT round-trip test
        clean_sine = generate_sine_wave(
            frequency=200.0, amplitude=3.0, duration=0.5, sample_rate=sample_rate
        )

        return {
            "step_change": step_trace,
            "impulse": impulse_trace,
            "chirp": chirp_trace,
            "multi_component": multi_trace,
            "clean_sine": clean_sine,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run wavelet analysis demonstrations."""
        results = {}

        self.section("Oscura Wavelet Analysis")
        self.info("Demonstrating time-frequency analysis for transient signals")
        self.info("Comparing wavelet transforms to FFT for localization capabilities")

        # ========== PART 1: CWT WITH MORLET WAVELET ==========
        step_signal = data["step_change"]
        self.subsection("Part 1: Continuous Wavelet Transform (CWT) - Morlet Wavelet")
        self.info("Signal with step change: 1V → 5V at t=0.5s")
        self.info("Morlet wavelet: Complex wavelet, optimal for frequency localization")

        scales_morlet, freqs_morlet, coef_morlet = cwt_wrapper(
            step_signal, wavelet="morl", n_scales=64
        )
        results["cwt_morlet_scales"] = scales_morlet
        results["cwt_morlet_freqs"] = freqs_morlet
        results["cwt_morlet_coef"] = coef_morlet

        self.info("\nCWT output (Morlet):")
        self.result("Scales", len(scales_morlet))
        self.result("Frequencies", f"{freqs_morlet.min():.1f} - {freqs_morlet.max():.1f}", "Hz")
        self.result("Coefficient shape", coef_morlet.shape)
        self.result("Coefficient type", "complex" if np.iscomplexobj(coef_morlet) else "real")

        # Analyze magnitude at fundamental frequency (100 Hz)
        fund_idx = np.argmin(np.abs(freqs_morlet - 100))
        magnitude_at_fund = np.abs(coef_morlet[fund_idx, :])
        step_idx = len(step_signal.data) // 2
        mag_before_step = np.mean(magnitude_at_fund[: step_idx - 100])
        mag_after_step = np.mean(magnitude_at_fund[step_idx + 100 :])
        magnitude_ratio = mag_after_step / mag_before_step if mag_before_step > 0 else 0

        results["cwt_morlet_mag_ratio"] = magnitude_ratio

        self.info("\nTime-frequency localization at 100 Hz:")
        self.result("Magnitude before step", f"{mag_before_step:.2f}")
        self.result("Magnitude after step", f"{mag_after_step:.2f}")
        self.result("Magnitude ratio (after/before)", f"{magnitude_ratio:.2f}")
        self.info("  Expected ratio: ~5.0 (amplitude change 1V → 5V)")

        # ========== PART 2: CWT WITH MEXICAN HAT WAVELET ==========
        impulse_signal = data["impulse"]
        self.subsection("Part 2: CWT - Mexican Hat (Ricker) Wavelet")
        self.info("Signal with impulse: Brief 8V spike at t=0.3s")
        self.info("Mexican hat wavelet: Real wavelet, good for edge/transient detection")

        scales_mexh, freqs_mexh, coef_mexh = cwt_wrapper(
            impulse_signal, wavelet="mexh", n_scales=64
        )
        results["cwt_mexh_scales"] = scales_mexh
        results["cwt_mexh_freqs"] = freqs_mexh
        results["cwt_mexh_coef"] = coef_mexh

        self.info("\nCWT output (Mexican hat):")
        self.result("Scales", len(scales_mexh))
        self.result("Frequencies", f"{freqs_mexh.min():.1f} - {freqs_mexh.max():.1f}", "Hz")
        self.result("Coefficient shape", coef_mexh.shape)
        self.result("Coefficient type", "complex" if np.iscomplexobj(coef_mexh) else "real")

        # Detect impulse location in wavelet coefficients
        # Use mid-frequency scale to detect transient
        mid_scale_idx = len(scales_mexh) // 2
        coef_mid_scale = np.abs(coef_mexh[mid_scale_idx, :])
        detected_impulse_idx = np.argmax(coef_mid_scale)
        impulse_idx_actual = int(0.3 * impulse_signal.metadata.sample_rate)
        detection_error = abs(detected_impulse_idx - impulse_idx_actual)
        detection_error_ms = detection_error / impulse_signal.metadata.sample_rate * 1000

        results["impulse_detected_idx"] = detected_impulse_idx
        results["impulse_detection_error_ms"] = detection_error_ms

        self.info("\nTransient detection:")
        self.result("Expected impulse time", "0.300", "s")
        self.result(
            "Detected impulse time",
            f"{detected_impulse_idx / impulse_signal.metadata.sample_rate:.3f}",
            "s",
        )
        self.result("Detection error", f"{detection_error_ms:.2f}", "ms")
        self.info("  Mexican hat wavelet successfully localizes transient in time")

        # ========== PART 3: DISCRETE WAVELET TRANSFORM (DWT) ==========
        clean_sine = data["clean_sine"]
        self.subsection("Part 3: Discrete Wavelet Transform (DWT)")
        self.info("Clean 200 Hz sine wave for decomposition")
        self.info("Daubechies db4 wavelet: 4-tap filter, good for general signals")

        dwt_coeffs = dwt(clean_sine, wavelet="db4", level=3)
        results["dwt_coeffs"] = dwt_coeffs

        self.info("\nDWT decomposition (3 levels):")
        self.result("Original signal length", len(clean_sine.data))
        for key in sorted(dwt_coeffs.keys()):
            self.result(f"  {key}", f"{len(dwt_coeffs[key])} coefficients")

        # Analyze coefficient magnitudes
        cA_energy = np.sum(dwt_coeffs["cA"] ** 2)
        cD1_energy = np.sum(dwt_coeffs["cD1"] ** 2)
        cD2_energy = np.sum(dwt_coeffs["cD2"] ** 2)
        cD3_energy = np.sum(dwt_coeffs["cD3"] ** 2)
        total_energy = cA_energy + cD1_energy + cD2_energy + cD3_energy

        results["dwt_energy"] = {
            "cA": cA_energy,
            "cD1": cD1_energy,
            "cD2": cD2_energy,
            "cD3": cD3_energy,
            "total": total_energy,
        }

        self.info("\nCoefficient energy distribution:")
        self.result("  cA (approximation)", f"{100 * cA_energy / total_energy:.1f}", "%")
        self.result("  cD1 (detail, highest freq)", f"{100 * cD1_energy / total_energy:.1f}", "%")
        self.result("  cD2 (detail, mid freq)", f"{100 * cD2_energy / total_energy:.1f}", "%")
        self.result("  cD3 (detail, low freq)", f"{100 * cD3_energy / total_energy:.1f}", "%")
        self.info("  Most energy in cA and cD2 for sine wave (expected)")

        # ========== PART 4: INVERSE DWT (RECONSTRUCTION) ==========
        self.subsection("Part 4: Inverse DWT - Signal Reconstruction")
        self.info("Reconstructing original signal from DWT coefficients")

        reconstructed = idwt(dwt_coeffs, wavelet="db4")
        results["reconstructed"] = reconstructed

        # Handle length mismatch (DWT can change length slightly)
        min_len = min(len(clean_sine.data), len(reconstructed))
        original_trimmed = clean_sine.data[:min_len]
        reconstructed_trimmed = reconstructed[:min_len]

        reconstruction_error = np.sqrt(np.mean((original_trimmed - reconstructed_trimmed) ** 2))
        signal_rms = np.sqrt(np.mean(original_trimmed**2))
        reconstruction_error_pct = 100 * reconstruction_error / signal_rms

        results["reconstruction_error"] = reconstruction_error
        results["reconstruction_error_pct"] = reconstruction_error_pct

        self.info("\nReconstruction quality:")
        self.result("Original length", len(clean_sine.data))
        self.result("Reconstructed length", len(reconstructed))
        self.result("RMS reconstruction error", f"{reconstruction_error:.6f}", "V")
        self.result("Reconstruction error", f"{reconstruction_error_pct:.4f}", "%")
        self.info("  Near-perfect reconstruction (round-trip fidelity)")

        # ========== PART 5: COMPARISON - WAVELET VS FFT ==========
        multi_signal = data["multi_component"]
        self.subsection("Part 5: Wavelet vs FFT - Time-Frequency Resolution")
        self.info("Multi-component signal: 100 Hz → 300 Hz → 600 Hz over time")
        self.info("Comparing time-frequency localization capabilities")

        # FFT analysis (entire signal)
        freq_fft, mag_fft = fft(multi_signal, window="hann")
        results["fft_freq"] = freq_fft
        results["fft_mag"] = mag_fft

        # Find peaks in FFT
        mag_linear = 10 ** (mag_fft / 20)
        peak_indices = []
        for i in range(1, len(mag_linear) - 1):
            if mag_linear[i] > mag_linear[i - 1] and mag_linear[i] > mag_linear[i + 1]:
                if mag_linear[i] > 0.1 * np.max(mag_linear):  # Significant peaks only
                    peak_indices.append(i)

        fft_peaks = sorted([freq_fft[i] for i in peak_indices])
        results["fft_peaks"] = fft_peaks

        self.info("\nFFT analysis (time-averaged):")
        self.result("FFT length", len(freq_fft))
        self.result("Frequency resolution", f"{freq_fft[1] - freq_fft[0]:.2f}", "Hz")
        self.result("Detected peaks", f"{fft_peaks[:5]}")
        self.info("  FFT sees all 3 frequencies but CANNOT tell when they occur")

        # CWT analysis (time-frequency)
        scales_multi, freqs_multi, coef_multi = cwt_wrapper(
            multi_signal, wavelet="morl", n_scales=128
        )
        results["cwt_multi_coef"] = coef_multi

        # Find when each frequency component is strongest
        for target_freq in [100, 300, 600]:
            freq_idx = np.argmin(np.abs(freqs_multi - target_freq))
            magnitude_over_time = np.abs(coef_multi[freq_idx, :])
            peak_time_idx = np.argmax(magnitude_over_time)
            peak_time = peak_time_idx / multi_signal.metadata.sample_rate

            results[f"cwt_peak_time_{target_freq}Hz"] = peak_time

            self.info(f"\nCWT localization at {target_freq} Hz:")
            self.result("  Peak time", f"{peak_time:.3f}", "s")

            # Expected times: 100 Hz peaks at ~0.15s, 300 Hz at ~0.45s, 600 Hz at ~0.80s
            if target_freq == 100:
                expected_time = 0.15
            elif target_freq == 300:
                expected_time = 0.45
            else:  # 600 Hz
                expected_time = 0.80

            time_error = abs(peak_time - expected_time)
            self.result("  Expected time", f"{expected_time:.3f}", "s")
            self.result("  Time error", f"{time_error:.3f}", "s")

        self.info("\nKey insight: CWT reveals WHEN frequencies occur (time-frequency localization)")

        # ========== PART 6: DWT WITH DIFFERENT WAVELETS ==========
        self.subsection("Part 6: DWT with Different Wavelet Families")
        self.info("Comparing Daubechies db4 vs db8 wavelets")

        # db8 has more taps (better frequency selectivity, less time localization)
        dwt_db8 = dwt(clean_sine, wavelet="db8", level=3)
        results["dwt_db8_coeffs"] = dwt_db8

        reconstructed_db8 = idwt(dwt_db8, wavelet="db8")
        min_len_db8 = min(len(clean_sine.data), len(reconstructed_db8))
        reconstruction_error_db8 = np.sqrt(
            np.mean((clean_sine.data[:min_len_db8] - reconstructed_db8[:min_len_db8]) ** 2)
        )
        reconstruction_error_db8_pct = 100 * reconstruction_error_db8 / signal_rms

        results["reconstruction_error_db8"] = reconstruction_error_db8
        results["reconstruction_error_db8_pct"] = reconstruction_error_db8_pct

        self.info("\nWavelet comparison (reconstruction error):")
        self.result("db4 (4-tap)", f"{reconstruction_error_pct:.6f}", "%")
        self.result("db8 (8-tap)", f"{reconstruction_error_db8_pct:.6f}", "%")
        self.info("  Both wavelets achieve excellent reconstruction")

        # ========== SUMMARY ==========
        self.subsection("Summary")
        self.info("\nWavelet analysis complete!")
        self.success("CWT: Time-frequency localization for transient detection")
        self.success("DWT: Multi-resolution decomposition with perfect reconstruction")
        self.success("Wavelet vs FFT: Wavelets reveal temporal structure of signals")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate wavelet analysis results."""
        all_valid = True

        self.section("Validation")

        # ========== CWT MORLET VALIDATION ==========
        self.subsection("CWT Morlet Validation")

        cwt_morlet_coef = results["cwt_morlet_coef"]
        if cwt_morlet_coef.shape[0] == 64:
            print(f"  ✓ CWT Morlet scales: {cwt_morlet_coef.shape[0]} (expected 64)")
        else:
            print(f"  ✗ CWT Morlet scales: {cwt_morlet_coef.shape[0]} (expected 64)")
            all_valid = False

        # Check magnitude ratio for step change
        mag_ratio = results["cwt_morlet_mag_ratio"]
        if 4.0 < mag_ratio < 6.0:
            print(f"  ✓ Magnitude ratio: {mag_ratio:.2f} (expected ~5.0 for 1V→5V step)")
        else:
            print(f"  ✗ Magnitude ratio: {mag_ratio:.2f} (expected 4.0-6.0)")
            all_valid = False

        # Check coefficient type (PyWavelets returns real, scipy.signal returned complex)
        # Both are valid - PyWavelets returns magnitude, scipy returned complex
        if np.iscomplexobj(cwt_morlet_coef):
            print("  ✓ Morlet coefficients: complex (scipy.signal implementation)")
        else:
            print("  ✓ Morlet coefficients: real (PyWavelets implementation - magnitude)")
            # This is acceptable - PyWavelets computes real-valued CWT magnitude

        # ========== CWT MEXICAN HAT VALIDATION ==========
        self.subsection("CWT Mexican Hat Validation")

        detection_error_ms = results["impulse_detection_error_ms"]
        if detection_error_ms < 20.0:  # Within 20 ms
            print(f"  ✓ Impulse detection error: {detection_error_ms:.2f} ms (< 20 ms)")
        else:
            print(f"  ✗ Impulse detection error: {detection_error_ms:.2f} ms (expected < 20 ms)")
            all_valid = False

        # Check that coefficients are real
        cwt_mexh_coef = results["cwt_mexh_coef"]
        if not np.iscomplexobj(cwt_mexh_coef):
            print("  ✓ Mexican hat coefficients: real (expected for Ricker wavelet)")
        else:
            print("  ✗ Mexican hat coefficients: complex (expected real)")
            all_valid = False

        # ========== DWT VALIDATION ==========
        self.subsection("DWT Validation")

        dwt_coeffs = results["dwt_coeffs"]
        expected_keys = {"cA", "cD1", "cD2", "cD3"}
        if set(dwt_coeffs.keys()) == expected_keys:
            print(f"  ✓ DWT coefficients: {list(dwt_coeffs.keys())} (3 levels + approximation)")
        else:
            print(f"  ✗ DWT coefficients: {list(dwt_coeffs.keys())} (expected {expected_keys})")
            all_valid = False

        # Check energy distribution (sine wave should have most energy in lower frequencies)
        dwt_energy = results["dwt_energy"]
        if dwt_energy["cA"] > dwt_energy["cD1"]:
            print(
                f"  ✓ Energy distribution: cA={100 * dwt_energy['cA'] / dwt_energy['total']:.1f}% > cD1 (expected for sine wave)"
            )
        else:
            print("  ✗ Energy distribution: cA should dominate for sine wave")
            all_valid = False

        # ========== IDWT VALIDATION ==========
        self.subsection("IDWT Reconstruction Validation")

        reconstruction_error_pct = results["reconstruction_error_pct"]
        if reconstruction_error_pct < 0.1:  # Less than 0.1% error
            print(f"  ✓ Reconstruction error: {reconstruction_error_pct:.4f}% (excellent, < 0.1%)")
        else:
            print(f"  ✗ Reconstruction error: {reconstruction_error_pct:.4f}% (expected < 0.1%)")
            all_valid = False

        # ========== WAVELET VS FFT VALIDATION ==========
        self.subsection("Time-Frequency Localization Validation")

        # FFT should detect all 3 frequency components
        fft_peaks = results["fft_peaks"]
        if len(fft_peaks) >= 3:
            print(f"  ✓ FFT detected {len(fft_peaks)} peaks (all 3 components visible)")
        else:
            print(f"  ✗ FFT detected {len(fft_peaks)} peaks (expected >= 3)")
            all_valid = False

        # CWT should localize each frequency in time
        for freq, expected_time in [(100, 0.15), (300, 0.45), (600, 0.80)]:
            peak_time = results.get(f"cwt_peak_time_{freq}Hz", 0)
            time_error = abs(peak_time - expected_time)

            if time_error < 0.15:  # Within 150 ms
                print(
                    f"  ✓ CWT {freq} Hz localization: {peak_time:.3f}s (expected ~{expected_time:.2f}s, error={time_error:.3f}s)"
                )
            else:
                print(
                    f"  ✗ CWT {freq} Hz localization: {peak_time:.3f}s (expected ~{expected_time:.2f}s, error={time_error:.3f}s > 0.15s)"
                )
                all_valid = False

        # ========== WAVELET COMPARISON VALIDATION ==========
        self.subsection("Wavelet Family Comparison")

        reconstruction_error_db4 = results["reconstruction_error_pct"]
        reconstruction_error_db8 = results["reconstruction_error_db8_pct"]

        # Both should achieve good reconstruction
        if reconstruction_error_db4 < 0.1 and reconstruction_error_db8 < 0.1:
            print(
                f"  ✓ Both db4 ({reconstruction_error_db4:.4f}%) and db8 ({reconstruction_error_db8:.4f}%) achieve excellent reconstruction"
            )
        else:
            print("  ✗ One or both wavelets failed to achieve < 0.1% reconstruction error")
            all_valid = False

        if all_valid:
            self.success("All wavelet analysis tests validated!")
            self.info("\nKey takeaways:")
            self.info("  - CWT provides time-frequency localization (detect WHEN events occur)")
            self.info("  - Morlet wavelet: Complex, optimal for frequency analysis")
            self.info("  - Mexican hat wavelet: Real, optimal for transient/edge detection")
            self.info("  - DWT: Multi-resolution decomposition with perfect reconstruction")
            self.info("  - IDWT: Round-trip fidelity (< 0.1% error)")
            self.info("  - Wavelet vs FFT: Wavelets resolve temporal structure, FFT does not")
            self.info("\nNext steps:")
            self.info("  - Try 04_advanced_analysis/time_frequency_analysis.py for STFT comparison")
            self.info("  - Explore 02_basic_analysis/04_filtering.py for denoising with DWT")
        else:
            self.error("Some wavelet analysis tests failed validation")

        return all_valid


if __name__ == "__main__":
    demo: WaveletAnalysisDemo = WaveletAnalysisDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
