"""Spectral Analysis: Core frequency domain measurements

Demonstrates:
- oscura.fft() - Fast Fourier Transform
- oscura.psd() - Power Spectral Density
- oscura.thd() - Total Harmonic Distortion
- oscura.snr() - Signal-to-Noise Ratio
- oscura.sinad() - Signal-to-Noise and Distortion
- oscura.enob() - Effective Number of Bits
- oscura.sfdr() - Spurious-Free Dynamic Range

IEEE Standards: IEEE 1241-2010 (analog-to-digital converter testing)
Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/02_statistical_measurements.py

Uses a complex signal with fundamental and harmonics to demonstrate all
spectral measurement capabilities. Perfect for understanding frequency
domain analysis and ADC quality metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    generate_complex_signal,
    generate_sine_wave,
    validate_approximately,
)
from oscura import (
    enob,
    fft,
    psd,
    sfdr,
    sinad,
    snr,
    thd,
)


class SpectralAnalysisDemo(BaseDemo):
    """Comprehensive demonstration of all spectral analysis capabilities."""

    def __init__(self) -> None:
        """Initialize spectral analysis demonstration."""
        super().__init__(
            name="spectral_analysis",
            description="Frequency domain measurements: FFT, PSD, THD, SNR, SINAD, ENOB, SFDR",
            capabilities=[
                "oscura.fft",
                "oscura.psd",
                "oscura.thd",
                "oscura.snr",
                "oscura.sinad",
                "oscura.enob",
                "oscura.sfdr",
            ],
            ieee_standards=[
                "IEEE 1241-2010",
            ],
            related_demos=[
                "00_getting_started/00_hello_world.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for spectral analysis demonstrations.

        Creates:
        1. Pure sine wave: Shows FFT and PSD with clean single peak
        2. Signal with harmonics: Demonstrates THD measurement
        3. Noisy signal: Demonstrates SNR, SINAD, ENOB, SFDR
        """
        # 1. Pure 1kHz sine wave (clean reference signal)
        clean_sine = generate_sine_wave(
            frequency=1000.0,  # 1 kHz fundamental
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms (10 periods at 1kHz)
            sample_rate=100e3,  # 100 kHz sampling
        )

        # 2. Signal with harmonics for THD measurement
        # Create: 1kHz fundamental + 3rd harmonic (3kHz) + 5th harmonic (5kHz)
        harmonic_signal = generate_complex_signal(
            fundamentals=[1000.0, 3000.0, 5000.0],
            amplitudes=[1.0, 0.3, 0.2],  # 30% 3rd harmonic, 20% 5th harmonic
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        # 3. Noisy sine wave for SNR, SINAD, ENOB, SFDR measurements
        # Start with clean signal and add noise
        noisy_sine = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        # Add noise for 40 dB SNR
        noisy_sine = add_noise(noisy_sine, snr_db=40.0)

        return {
            "clean_sine": clean_sine,
            "harmonic_signal": harmonic_signal,
            "noisy_sine": noisy_sine,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run spectral analysis demonstrations."""
        results = {}

        # === 1. FFT Analysis ===
        self.section("1. FFT (Fast Fourier Transform)")
        self.info(
            "The FFT converts time-domain signals to frequency domain, "
            "revealing frequency components."
        )

        clean_sine = data["clean_sine"]
        fft_result = fft(clean_sine)
        frequencies = fft_result[0]
        magnitudes = fft_result[1]

        # Find peak frequency and magnitude
        peak_idx = int(magnitudes.argmax())
        peak_freq = frequencies[peak_idx]
        peak_mag = magnitudes[peak_idx]

        self.info("\nClean 1kHz sine wave FFT:")
        self.info(f"  Peak frequency: {peak_freq:.1f} Hz (expected: 1000.0 Hz)")
        self.info(f"  Peak magnitude: {peak_mag:.4f} (normalized)")
        self.info(f"  Total frequency bins: {len(frequencies)}")

        results["fft_peak_freq"] = peak_freq
        results["fft_peak_magnitude"] = peak_mag

        # === 2. PSD Analysis ===
        self.section("2. PSD (Power Spectral Density)")
        self.info(
            "PSD shows power distribution across frequencies. "
            "Useful for noise analysis and signal detection."
        )

        freq_psd, power = psd(clean_sine)

        peak_psd_idx = int(power.argmax())
        peak_psd_freq = freq_psd[peak_psd_idx]
        peak_psd_power = power[peak_psd_idx]

        self.info("\nClean sine wave PSD:")
        self.info(f"  Peak frequency: {peak_psd_freq:.1f} Hz (expected: 1000.0 Hz)")
        self.info(f"  Peak power: {peak_psd_power:.6f} (normalized)")

        results["psd_peak_freq"] = peak_psd_freq
        results["psd_peak_power"] = peak_psd_power

        # === 3. THD Analysis ===
        self.section("3. THD (Total Harmonic Distortion)")
        self.info(
            "THD measures the amount of harmonic distortion in a signal. "
            "Pure sine = 0% THD, distorted signal = higher THD."
        )

        # Clean sine should have near-zero THD
        clean_thd = thd(clean_sine)
        self.info(f"\nClean 1kHz sine wave THD: {clean_thd:.2f} dB")
        self.info("  (Expected: < -60 dB for perfect sine)")

        results["clean_sine_thd"] = clean_thd

        # Harmonic signal has significant THD
        harmonic_signal = data["harmonic_signal"]
        harmonic_thd = thd(harmonic_signal)
        self.info(f"\nSignal with harmonics THD: {harmonic_thd:.2f} dB")
        self.info("  (Expected: ~-5 to -10 dB due to 3rd and 5th harmonics)")

        results["harmonic_signal_thd"] = harmonic_thd

        # === 4. SNR Analysis ===
        self.section("4. SNR (Signal-to-Noise Ratio)")
        self.info("SNR measures signal power relative to noise power. Higher SNR = cleaner signal.")

        noisy_sine = data["noisy_sine"]
        measured_snr = snr(noisy_sine)

        self.info(f"\nNoisy 1kHz sine (40 dB added noise) SNR: {measured_snr:.2f} dB")
        self.info("  (Expected: ~40 dB)")

        results["measured_snr"] = measured_snr

        # === 5. SINAD Analysis ===
        self.section("5. SINAD (Signal-to-Noise and Distortion)")
        self.info(
            "SINAD combines effects of noise and harmonic distortion. "
            "Lower than SNR if signal has significant harmonics."
        )

        measured_sinad = sinad(noisy_sine)
        self.info(f"\nNoisy 1kHz sine SINAD: {measured_sinad:.2f} dB")
        self.info("  (Expected: ~40 dB for clean sine with noise)")

        results["measured_sinad"] = measured_sinad

        # === 6. ENOB Analysis ===
        self.section("6. ENOB (Effective Number of Bits)")
        self.info(
            "ENOB converts SINAD to equivalent ADC resolution. "
            "16-bit ADC ideal ENOB: 16 bits. Real ADCs: lower."
        )

        measured_enob = enob(noisy_sine)
        self.info(f"\nNoisy 1kHz sine ENOB: {measured_enob:.2f} bits")
        self.info("  (Expected: ~6-7 bits for 40 dB SINAD)")
        self.info("  Calculation: ENOB = (SINAD - 1.76) / 6.02")

        results["measured_enob"] = measured_enob

        # === 7. SFDR Analysis ===
        self.section("7. SFDR (Spurious-Free Dynamic Range)")
        self.info(
            "SFDR measures the difference between signal peak and "
            "highest spurious component (noise floor, harmonics)."
        )

        measured_sfdr = sfdr(noisy_sine)
        self.info(f"\nNoisy 1kHz sine SFDR: {measured_sfdr:.2f} dB")
        self.info("  (Expected: ~40-45 dB for 40 dB SNR signal)")

        results["measured_sfdr"] = measured_sfdr

        # === Summary ===
        self.section("Summary of Spectral Measurements")
        self.subsection("Harmonic Analysis (Clean Sine)")
        self.info(f"  FFT peak frequency: {results['fft_peak_freq']:.1f} Hz")
        self.info(f"  FFT peak magnitude: {results['fft_peak_magnitude']:.4f}")
        self.info(f"  PSD peak frequency: {results['psd_peak_freq']:.1f} Hz")
        self.info(f"  THD: {results['clean_sine_thd']:.2f} dB")

        self.subsection("Distortion Analysis (Harmonic Signal)")
        self.info(f"  THD with harmonics: {results['harmonic_signal_thd']:.2f} dB")
        self.info("  (Contains 3rd and 5th harmonics)")

        self.subsection("ADC Quality Metrics (Noisy Signal)")
        self.info(f"  SNR: {results['measured_snr']:.2f} dB")
        self.info(f"  SINAD: {results['measured_sinad']:.2f} dB")
        self.info(f"  ENOB: {results['measured_enob']:.2f} bits")
        self.info(f"  SFDR: {results['measured_sfdr']:.2f} dB")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate spectral analysis results."""
        all_valid = True

        self.section("Validation")

        # === FFT Validation ===
        self.subsection("FFT Results")

        # Peak frequency should be ~1000 Hz
        if not validate_approximately(
            results["fft_peak_freq"],
            1000.0,
            tolerance=0.05,
            name="FFT peak frequency",
        ):
            all_valid = False

        # Peak magnitude should be reasonable (can be positive or negative dB)
        if isinstance(results["fft_peak_magnitude"], (int, float)):
            print(f"  ✓ FFT peak magnitude: {results['fft_peak_magnitude']:.4f} (valid)")
        else:
            print(f"  ✗ FFT peak magnitude: {results['fft_peak_magnitude']:.4f} (not a number)")
            all_valid = False

        # === PSD Validation ===
        self.subsection("PSD Results")

        # PSD peak frequency should be ~1000 Hz (with 20% tolerance for FFT resolution)
        if not validate_approximately(
            results["psd_peak_freq"],
            1000.0,
            tolerance=0.20,
            name="PSD peak frequency",
        ):
            all_valid = False

        # === THD Validation ===
        self.subsection("THD Results")

        # Clean sine THD should be very low (< -40 dB)
        if results["clean_sine_thd"] < -40.0:
            print(f"  ✓ Clean sine THD: {results['clean_sine_thd']:.2f} dB (very clean)")
        else:
            print(f"  ✗ Clean sine THD: {results['clean_sine_thd']:.2f} dB (expected < -40 dB)")
            all_valid = False

        # Harmonic signal THD should be elevated (-15 to -5 dB)
        if -15.0 < results["harmonic_signal_thd"] < -5.0:
            print(
                f"  ✓ Harmonic signal THD: {results['harmonic_signal_thd']:.2f} dB "
                f"(contains harmonics)"
            )
        else:
            print(
                f"  ✗ Harmonic signal THD: {results['harmonic_signal_thd']:.2f} dB "
                f"(expected -15 to -5 dB)"
            )
            all_valid = False

        # === SNR Validation ===
        self.subsection("SNR Results")

        # Measured SNR should be close to 40 dB (within 2 dB)
        if not validate_approximately(
            results["measured_snr"],
            40.0,
            tolerance=0.05,
            name="SNR (40 dB expected)",
        ):
            all_valid = False

        # === SINAD Validation ===
        self.subsection("SINAD Results")

        # SINAD should be close to SNR for clean sine (within 2 dB)
        sinad_diff = abs(results["measured_sinad"] - results["measured_snr"])
        if sinad_diff < 2.0:
            print(f"  ✓ SINAD: {results['measured_sinad']:.2f} dB (within 2 dB of SNR)")
        else:
            print(f"  ✗ SINAD: {results['measured_sinad']:.2f} dB (SNR-SINAD difference > 2 dB)")
            all_valid = False

        # === ENOB Validation ===
        self.subsection("ENOB Results")

        # For 40 dB SINAD, ENOB should be 6-7 bits
        # ENOB = (SINAD - 1.76) / 6.02
        # ENOB = (40 - 1.76) / 6.02 = 6.36 bits
        if 6.0 < results["measured_enob"] < 7.5:
            print(
                f"  ✓ ENOB: {results['measured_enob']:.2f} bits (expected 6-7 bits for 40 dB SINAD)"
            )
        else:
            print(f"  ✗ ENOB: {results['measured_enob']:.2f} bits (expected 6-7 bits)")
            all_valid = False

        # === SFDR Validation ===
        self.subsection("SFDR Results")

        # SFDR should be around 40-60 dB for 40 dB SNR signal
        # Can be higher due to low noise floor in synthetic signal
        if 35.0 < results["measured_sfdr"] < 80.0:
            print(f"  ✓ SFDR: {results['measured_sfdr']:.2f} dB (expected 35-80 dB)")
        else:
            print(f"  ✗ SFDR: {results['measured_sfdr']:.2f} dB (expected 35-80 dB)")
            all_valid = False

        if all_valid:
            self.success("All spectral measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - FFT: Time domain → Frequency domain conversion")
            self.info("  - PSD: Power distribution across frequencies")
            self.info("  - THD: Measures harmonic content (% distortion)")
            self.info("  - SNR: Signal power vs noise power ratio")
            self.info("  - SINAD: Combined effect of noise AND distortion")
            self.info("  - ENOB: Effective ADC resolution in bits")
            self.info("  - SFDR: Max signal before hitting noise floor")
            self.info("\nNext steps:")
            self.info("  - Try 02_statistical_measurements.py for power analysis")
            self.info("  - Explore 03_protocol_decoding/ for protocol analysis")
            self.info("  - Check 04_advanced_analysis/ for advanced techniques")
        else:
            self.error("Some spectral measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: SpectralAnalysisDemo = SpectralAnalysisDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
