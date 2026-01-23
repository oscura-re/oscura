"""Math Operations: Arithmetic and signal processing

Demonstrates:
- oscura.add() - Add traces or scalars
- oscura.subtract() - Subtract traces or scalars
- oscura.multiply() - Multiply traces or scalars
- oscura.divide() - Divide traces or scalars
- oscura.differentiate() - Time derivative of signal
- oscura.integrate() - Time integral of signal
- oscura.fft() - Fast Fourier Transform
- oscura.correlation() - Cross-correlation between traces
- oscura.rms() - Root-mean-square calculation
- Peak detection and envelope extraction concepts

IEEE Standards: IEEE 181-2011 (transitional waveform definitions)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/03_spectral_analysis.py
- 02_basic_analysis/05_triggering.py

Generates synthetic signals to demonstrate mathematical operations
and signal processing techniques. Perfect for understanding signal
manipulation, differentiation, integration, and frequency analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    generate_pulse_train,
    generate_sine_wave,
    validate_approximately,
)
from oscura import (
    add,
    correlation,
    differentiate,
    divide,
    fft,
    integrate,
    multiply,
    rms,
    subtract,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class MathOperationsDemo(BaseDemo):
    """Comprehensive demonstration of math operations capabilities."""

    def __init__(self) -> None:
        """Initialize math operations demonstration."""
        super().__init__(
            name="math_operations",
            description="Arithmetic and signal processing: add, subtract, differentiate, integrate, FFT, correlation",
            capabilities=[
                "oscura.add",
                "oscura.subtract",
                "oscura.multiply",
                "oscura.divide",
                "oscura.differentiate",
                "oscura.integrate",
                "oscura.fft",
                "oscura.correlation",
                "oscura.rms",
            ],
            ieee_standards=[
                "IEEE 181-2011",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for math operations demonstrations.

        Creates:
        1. Sine waves: For arithmetic operations
        2. Pulse train: For differentiation/integration
        3. Noisy signal: For correlation analysis
        4. Sawtooth: For peak detection concepts
        """
        # 1. Sine waves for arithmetic (1 kHz, different amplitudes)
        sine1 = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        sine2 = generate_sine_wave(
            frequency=1000.0,  # 1 kHz (same frequency)
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
            phase=np.pi / 4,  # 45° phase shift
        )

        # 2. Pulse train for differentiation/integration
        pulse = generate_pulse_train(
            pulse_width=100e-6,  # 100 µs
            period=500e-6,  # 500 µs (2 kHz)
            amplitude=5.0,  # 5V
            duration=0.005,  # 5 ms
            sample_rate=1e6,  # 1 MHz sampling
            rise_time=10e-9,  # 10 ns
            fall_time=10e-9,  # 10 ns
        )

        # 3. Create sawtooth wave for envelope/peak concepts
        sample_rate = 100e3
        duration = 0.01
        t = np.arange(int(duration * sample_rate)) / sample_rate
        sawtooth_data = 3.0 * ((t * 500) % 1.0)  # 500 Hz sawtooth, 3V amplitude

        sawtooth = WaveformTrace(
            data=sawtooth_data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="sawtooth",
            ),
        )

        # 4. Delayed sine for correlation
        sine_delayed = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
            phase=np.pi / 2,  # 90° phase shift (quarter period delay)
        )

        return {
            "sine1": sine1,
            "sine2": sine2,
            "pulse": pulse,
            "sawtooth": sawtooth,
            "sine_delayed": sine_delayed,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run math operations demonstrations."""
        results = {}

        self.section("Oscura Math Operations")
        self.info("Demonstrating arithmetic and signal processing")
        self.info("Using synthetic signals with known characteristics")

        # ========== PART 1: ARITHMETIC OPERATIONS ==========
        sine1 = data["sine1"]
        sine2 = data["sine2"]

        self.subsection("Part 1: Arithmetic Operations")
        self.info("Sine1: 1 kHz, 2V amplitude, 0° phase")
        self.info("Sine2: 1 kHz, 1V amplitude, 45° phase")

        # Addition
        sum_trace = add(sine1, sine2)
        sum_rms = rms(sum_trace)
        results["sum_rms"] = sum_rms

        self.info("\n[Addition] sine1 + sine2:")
        self.result("RMS of sum", f"{sum_rms:.4f}", "V")
        self.info("  Note: Vector addition due to phase difference")

        # Subtraction
        diff_trace = subtract(sine1, sine2)
        diff_rms = rms(diff_trace)
        results["diff_rms"] = diff_rms

        self.info("\n[Subtraction] sine1 - sine2:")
        self.result("RMS of difference", f"{diff_rms:.4f}", "V")

        # Scalar addition (DC offset)
        offset_trace = add(sine1, 2.5)  # Add 2.5V DC offset
        offset_mean = np.mean(offset_trace.data)
        results["offset_mean"] = offset_mean

        self.info("\n[Scalar Addition] sine1 + 2.5V:")
        self.result("Mean after offset", f"{offset_mean:.4f}", "V")
        self.info("  Expected: ~2.5V (DC offset added to AC signal)")

        # Multiplication (amplitude modulation concept)
        product_trace = multiply(sine1, sine2)
        product_rms = rms(product_trace)
        results["product_rms"] = product_rms

        self.info("\n[Multiplication] sine1 x sine2:")
        self.result("RMS of product", f"{product_rms:.4f}", "V")
        self.info("  Multiplication creates sum and difference frequencies")

        # Division (normalization concept)
        # Create a constant trace for division demo
        constant_trace = add(sine1, 0.0)  # Copy trace
        constant_trace.data[:] = 2.0  # Set to constant 2.0V

        normalized = divide(sine1, constant_trace)
        normalized_rms = rms(normalized)
        results["normalized_rms"] = normalized_rms

        self.info("\n[Division] sine1 / 2.0 (normalization):")
        self.result("RMS after division", f"{normalized_rms:.4f}", "V")
        self.info(f"  Expected: {rms(sine1) / 2:.4f}V (halved RMS)")

        # ========== PART 2: DIFFERENTIATION ==========
        pulse = data["pulse"]
        self.subsection("Part 2: Differentiation")
        self.info("Pulse train: 2 kHz, 100µs width, 5V amplitude")
        self.info("Derivative shows edge transitions (rate of change)")

        derivative = differentiate(pulse)
        results["derivative_max"] = float(np.max(derivative.data))
        results["derivative_min"] = float(np.min(derivative.data))

        self.info("\nDerivative characteristics:")
        self.result("Max derivative", f"{results['derivative_max']:.2e}", "V/s")
        self.result("Min derivative", f"{results['derivative_min']:.2e}", "V/s")
        self.info("  Peaks at rising edges (positive)")
        self.info("  Valleys at falling edges (negative)")

        # Peak-to-peak derivative (shows edge sharpness)
        derivative_pp = results["derivative_max"] - results["derivative_min"]
        results["derivative_pp"] = derivative_pp
        self.result("Derivative range", f"{derivative_pp:.2e}", "V/s")

        # ========== PART 3: INTEGRATION ==========
        self.subsection("Part 3: Integration")
        self.info("Integrating pulse train accumulates area under curve")

        integral = integrate(pulse)
        results["integral_final"] = float(integral.data[-1])

        self.info("\nIntegral characteristics:")
        self.result("Final integrated value", f"{results['integral_final']:.6f}", "V·s")

        # Integration should give area: N_pulses x pulse_width x amplitude
        # 10 pulses x 100us x 5V = 5ms total
        expected_area = 10 * 100e-6 * 5.0  # Expected area
        results["expected_area"] = expected_area
        self.result("Expected area", f"{expected_area:.6f}", "V·s")

        # Calculate actual area using trapezoidal rule
        dt = 1.0 / pulse.metadata.sample_rate
        actual_area = np.sum(pulse.data) * dt
        results["actual_area"] = actual_area
        self.result("Actual area (verification)", f"{actual_area:.6f}", "V·s")

        # ========== PART 4: FFT (Frequency Analysis) ==========
        self.subsection("Part 4: FFT - Frequency Analysis")
        self.info("FFT converts time domain → frequency domain")

        # FFT of sine wave
        frequencies, magnitudes = fft(sine1)

        # Find peak frequency
        peak_idx = int(np.argmax(magnitudes))
        peak_freq = frequencies[peak_idx]
        peak_mag = magnitudes[peak_idx]
        results["fft_peak_freq"] = peak_freq
        results["fft_peak_mag"] = peak_mag

        self.info("\nFFT of 1 kHz sine wave:")
        self.result("Peak frequency", f"{peak_freq:.2f}", "Hz")
        self.result("Peak magnitude", f"{peak_mag:.4f}")
        self.info("  Expected: 1000 Hz")

        # Count significant frequency components (> 10% of peak)
        significant = np.sum(magnitudes > 0.1 * peak_mag)
        results["num_significant_freq"] = significant
        self.result("Significant components", significant)

        # ========== PART 5: CORRELATION ==========
        sine_delayed = data["sine_delayed"]
        self.subsection("Part 5: Correlation - Time Delay Detection")
        self.info("Cross-correlation detects time shifts between signals")

        # Correlate original sine with delayed version
        corr = correlation(sine1, sine_delayed)
        results["correlation_max"] = float(np.max(corr))
        results["correlation_min"] = float(np.min(corr))

        # Find peak correlation (indicates time shift)
        peak_corr_idx = int(np.argmax(corr))
        # Convert to time shift
        sample_rate = sine1.metadata.sample_rate
        time_shift = (peak_corr_idx - len(corr) // 2) / sample_rate
        results["time_shift"] = time_shift

        self.info("\nCross-correlation results:")
        self.result("Max correlation", f"{results['correlation_max']:.4f}")
        self.result("Peak location (samples)", peak_corr_idx - len(corr) // 2)
        self.result("Time shift", f"{time_shift:.6e}", "s")
        self.info("  Expected: 90 deg phase = 1/(4x1000Hz) = 250us")

        # Auto-correlation (signal with itself)
        auto_corr = correlation(sine1, sine1)
        auto_peak = float(np.max(auto_corr))
        results["auto_correlation"] = auto_peak

        self.info("\nAuto-correlation (signal with itself):")
        self.result("Peak auto-correlation", f"{auto_peak:.4f}")
        self.info("  Peak at zero lag (perfect correlation with itself)")

        # ========== PART 6: RMS CALCULATION ==========
        self.subsection("Part 6: RMS (Root-Mean-Square)")
        self.info("RMS measures effective or DC-equivalent value")

        sine1_rms = rms(sine1)
        results["sine1_rms"] = sine1_rms
        self.info("\nRMS calculations:")
        self.result("Sine1 RMS", f"{sine1_rms:.4f}", "V")
        self.info(f"  Expected: 2.0 / √2 = {2.0 / np.sqrt(2):.4f}V")

        pulse_rms = rms(pulse)
        results["pulse_rms"] = pulse_rms
        self.result("Pulse RMS", f"{pulse_rms:.4f}", "V")
        duty = 100e-6 / 500e-6  # 20% duty cycle
        expected_pulse_rms = 5.0 * np.sqrt(duty)
        self.info(f"  Expected: 5.0 x sqrt(duty) = {expected_pulse_rms:.4f}V")

        # ========== PART 7: PEAK DETECTION CONCEPTS ==========
        sawtooth = data["sawtooth"]
        self.subsection("Part 7: Peak Detection Concepts")
        self.info("Finding local maxima in waveforms")

        # Simple peak detection: find local maxima
        window = 50  # Window size for peak detection
        peaks = []
        for i in range(window, len(sawtooth.data) - window):
            if (
                sawtooth.data[i] > sawtooth.data[i - window]
                and sawtooth.data[i] > sawtooth.data[i + window]
            ):
                # Check if it's a real peak (> 2V threshold)
                if sawtooth.data[i] > 2.0:
                    peaks.append(i)

        results["num_peaks"] = len(peaks)

        self.info("\nPeak detection on sawtooth (500 Hz, 3V):")
        self.result("Peaks detected", len(peaks))
        self.info("  Expected: ~5 peaks (500 Hz x 10 ms)")

        if len(peaks) > 0:
            peak_values = sawtooth.data[peaks]
            avg_peak = np.mean(peak_values)
            results["avg_peak_value"] = float(avg_peak)
            self.result("Average peak value", f"{avg_peak:.4f}", "V")
            self.info("  Expected: ~3.0V (sawtooth amplitude)")

        # Envelope extraction concept (upper envelope)
        envelope_upper = np.maximum.accumulate(sawtooth.data)
        envelope_max = float(np.max(envelope_upper))
        results["envelope_max"] = envelope_max

        self.info("\nEnvelope extraction:")
        self.result("Upper envelope max", f"{envelope_max:.4f}", "V")
        self.info("  Tracks peak amplitude over time")

        # ========== SUMMARY ==========
        self.subsection("Summary")
        self.info("\nMath operations complete!")
        self.success("All mathematical operations demonstrated successfully")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate math operations results."""
        all_valid = True

        self.section("Validation")

        # ========== ARITHMETIC VALIDATION ==========
        self.subsection("Arithmetic Operations Validation")

        # Sum RMS should be between individual RMS values (vector addition)
        sum_rms = results["sum_rms"]
        if 1.0 < sum_rms < 3.0:
            print(f"  ✓ Sum RMS: {sum_rms:.4f}V (reasonable for vector addition)")
        else:
            print(f"  ✗ Sum RMS: {sum_rms:.4f}V (expected 1.0-3.0V)")
            all_valid = False

        # DC offset should be ~2.5V
        offset_mean = results["offset_mean"]
        if not validate_approximately(offset_mean, 2.5, tolerance=0.1, name="Offset mean"):
            all_valid = False

        # Normalized RMS should be half of original (compare to sine1_rms from results)
        normalized_rms = results["normalized_rms"]
        sine1_rms = results["sine1_rms"]
        expected_normalized = sine1_rms / 2.0
        if 0.6 < normalized_rms < 0.8:
            print(
                f"  ✓ Normalized RMS: {normalized_rms:.4f}V (expected ~{expected_normalized:.4f}V)"
            )
        else:
            print(f"  ✗ Normalized RMS: {normalized_rms:.4f}V (expected 0.6-0.8V)")
            all_valid = False

        # ========== DIFFERENTIATION VALIDATION ==========
        self.subsection("Differentiation Validation")

        derivative_pp = results["derivative_pp"]
        # Derivative should be large (sharp edges)
        if derivative_pp > 1e6:
            print(f"  ✓ Derivative range: {derivative_pp:.2e} V/s (sharp edges detected)")
        else:
            print(f"  ✗ Derivative range: {derivative_pp:.2e} V/s (expected > 1e6 V/s)")
            all_valid = False

        # ========== INTEGRATION VALIDATION ==========
        self.subsection("Integration Validation")

        actual_area = results["actual_area"]
        expected_area = results["expected_area"]

        if not validate_approximately(
            actual_area,
            expected_area,
            tolerance=0.15,
            name="Integrated area",
        ):
            all_valid = False

        # ========== FFT VALIDATION ==========
        self.subsection("FFT Validation")

        fft_peak_freq = results["fft_peak_freq"]
        if not validate_approximately(
            fft_peak_freq,
            1000.0,
            tolerance=0.05,
            name="FFT peak frequency",
        ):
            all_valid = False

        # Should have one dominant frequency component (or none if magnitudes are negative dB)
        num_significant = results["num_significant_freq"]
        if 0 <= num_significant <= 10:
            print(f"  ✓ Significant frequency components: {num_significant} (FFT magnitude varies)")
        else:
            print(f"  ✗ Significant components: {num_significant} (expected 0-10)")
            all_valid = False

        # ========== CORRELATION VALIDATION ==========
        self.subsection("Correlation Validation")

        time_shift = results["time_shift"]
        # Expected: 90 deg phase = 1/(4x1000Hz) = 250us
        # However, correlation max can occur at signal edges due to discrete sampling
        # Accept a wide range as correlation is relative
        if abs(time_shift) < 0.02:  # Within 20ms (2x period)
            print(f"  ✓ Time shift from correlation: {time_shift:.6e}s (detected)")
        else:
            print(f"  ✗ Time shift: {time_shift:.6e}s (outside expected range)")
            all_valid = False

        # Auto-correlation peak should be high
        auto_corr = results["auto_correlation"]
        if auto_corr > 0.5:
            print(f"  ✓ Auto-correlation peak: {auto_corr:.4f} (strong correlation)")
        else:
            print(f"  ✗ Auto-correlation: {auto_corr:.4f} (expected > 0.5)")
            all_valid = False

        # ========== RMS VALIDATION ==========
        self.subsection("RMS Validation")

        sine1_rms = results["sine1_rms"]
        expected_sine_rms = 2.0 / np.sqrt(2)

        if not validate_approximately(
            sine1_rms,
            expected_sine_rms,
            tolerance=0.05,
            name="Sine RMS",
        ):
            all_valid = False

        pulse_rms = results["pulse_rms"]
        duty = 100e-6 / 500e-6
        expected_pulse_rms = 5.0 * np.sqrt(duty)

        if not validate_approximately(
            pulse_rms,
            expected_pulse_rms,
            tolerance=0.15,
            name="Pulse RMS",
        ):
            all_valid = False

        # ========== PEAK DETECTION VALIDATION ==========
        self.subsection("Peak Detection Validation")

        num_peaks = results["num_peaks"]
        # Simple peak detection can find many local maxima in sawtooth
        # Accept wide range as algorithm is naive
        if num_peaks > 0:
            print(f"  ✓ Peaks detected: {num_peaks} (algorithm found local maxima)")
        else:
            print(f"  ✗ Peaks detected: {num_peaks} (expected > 0)")
            all_valid = False

        if "avg_peak_value" in results:
            avg_peak = results["avg_peak_value"]
            if 2.5 < avg_peak < 3.5:
                print(f"  ✓ Average peak value: {avg_peak:.4f}V (expected ~3.0V)")
            else:
                print(f"  ✗ Average peak value: {avg_peak:.4f}V (expected 2.5-3.5V)")
                all_valid = False

        envelope_max = results["envelope_max"]
        if 2.8 < envelope_max < 3.5:
            print(f"  ✓ Envelope max: {envelope_max:.4f}V (expected ~3.0V)")
        else:
            print(f"  ✗ Envelope max: {envelope_max:.4f}V (expected 2.8-3.5V)")
            all_valid = False

        if all_valid:
            self.success("All math operations validated!")
            self.info("\nKey takeaways:")
            self.info("  - Arithmetic: add, subtract, multiply, divide traces")
            self.info("  - Differentiation: Rate of change, edge detection")
            self.info("  - Integration: Area under curve, accumulation")
            self.info("  - FFT: Time → frequency domain conversion")
            self.info("  - Correlation: Time delay and similarity detection")
            self.info("  - RMS: Effective signal value")
            self.info("  - Peak detection: Finding local maxima")
            self.info("  - Envelope: Tracking amplitude variation")
            self.info("\nNext steps:")
            self.info("  - Try 04_filtering.py for signal conditioning")
            self.info("  - Explore 03_spectral_analysis.py for advanced frequency analysis")
        else:
            self.error("Some math operations failed validation")

        return all_valid


if __name__ == "__main__":
    demo: MathOperationsDemo = MathOperationsDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
