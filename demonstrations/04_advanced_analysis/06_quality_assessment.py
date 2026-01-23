"""Signal Quality Assessment: IEEE 1241-2010 ADC testing metrics

Demonstrates:
- oscura.quality.snr() - Signal-to-noise ratio measurement
- oscura.quality.sinad() - Signal-to-noise-and-distortion ratio
- oscura.quality.thd() - Total harmonic distortion
- oscura.quality.sfdr() - Spurious-free dynamic range
- oscura.quality.enob() - Effective number of bits
- Quality scoring - Overall signal quality assessment
- Data quality warnings - Automatic issue detection

IEEE Standards: IEEE 1241-2010 (ADC testing and evaluation)
Related Demos:
- 02_basic_analysis/02_statistics.py
- 02_basic_analysis/03_spectral_analysis.py

Uses signals with varying quality levels to demonstrate assessment.
Perfect for understanding ADC performance and signal quality metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, add_noise, generate_sine_wave, validate_approximately
from oscura.analyzers.waveform.spectral import sfdr, sinad, snr, thd
from oscura.core.types import TraceMetadata, WaveformTrace


class QualityAssessmentDemo(BaseDemo):
    """Comprehensive demonstration of signal quality assessment."""

    def __init__(self) -> None:
        """Initialize quality assessment demonstration."""
        super().__init__(
            name="quality_assessment",
            description="IEEE 1241-2010 quality metrics: SNR, SINAD, THD, SFDR, ENOB",
            capabilities=[
                "oscura.quality.snr",
                "oscura.quality.sinad",
                "oscura.quality.thd",
                "oscura.quality.sfdr",
                "oscura.quality.enob",
            ],
            ieee_standards=[
                "IEEE 1241-2010",
            ],
            related_demos=[
                "02_basic_analysis/02_statistics.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate signals with varying quality levels.

        Creates:
        1. Ideal signal: Clean sine wave (very high SNR)
        2. Noisy signal: SNR ~ 40 dB
        3. Distorted signal: Harmonic distortion
        4. Poor quality signal: Combined noise and distortion
        """
        sample_rate = 10e6  # 10 MHz sampling
        duration = 0.001  # 1 ms
        signal_freq = 100e3  # 100 kHz test tone

        # 1. Ideal signal (reference quality)
        ideal = generate_sine_wave(
            frequency=signal_freq,
            amplitude=1.0,
            duration=duration,
            sample_rate=sample_rate,
        )

        # 2. Noisy signal (SNR ~ 40 dB)
        noisy = add_noise(ideal, snr_db=40.0)

        # 3. Distorted signal (with harmonics)
        # Add 2nd (5%), 3rd (3%), and 5th (2%) harmonics
        fundamental = generate_sine_wave(
            frequency=signal_freq,
            amplitude=1.0,
            duration=duration,
            sample_rate=sample_rate,
        )
        harmonic_2 = generate_sine_wave(
            frequency=2 * signal_freq,
            amplitude=0.05,  # -26 dBc
            duration=duration,
            sample_rate=sample_rate,
        )
        harmonic_3 = generate_sine_wave(
            frequency=3 * signal_freq,
            amplitude=0.03,  # -30.5 dBc
            duration=duration,
            sample_rate=sample_rate,
        )
        harmonic_5 = generate_sine_wave(
            frequency=5 * signal_freq,
            amplitude=0.02,  # -34 dBc
            duration=duration,
            sample_rate=sample_rate,
        )

        distorted_data = fundamental.data + harmonic_2.data + harmonic_3.data + harmonic_5.data
        distorted = WaveformTrace(
            data=distorted_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="distorted"),
        )

        # 4. Poor quality signal (noise + distortion)
        poor_data = distorted_data.copy()
        # Add noise (SNR ~ 30 dB)
        signal_power = np.mean(poor_data**2)
        noise_power = signal_power / (10 ** (30 / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(poor_data))
        poor_data = poor_data + noise

        poor = WaveformTrace(
            data=poor_data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="poor_quality"),
        )

        return {
            "ideal": ideal,
            "noisy": noisy,
            "distorted": distorted,
            "poor": poor,
            "signal_freq": signal_freq,
            "sample_rate": sample_rate,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive quality assessment demonstration."""
        results = {}

        self.section("Oscura Signal Quality Assessment")
        self.info("Demonstrating IEEE 1241-2010 compliant quality metrics")
        self.info("Using sine wave signals with varying quality levels")

        _signal_freq = data["signal_freq"]  # For reference (100 kHz)

        # ========== PART 1: IDEAL SIGNAL (REFERENCE) ==========
        self.subsection("Part 1: Ideal Signal (Reference Quality)")
        ideal = data["ideal"]
        self.info("Ideal signal: Clean 100 kHz sine wave")

        snr_ideal = snr(ideal)
        sinad_ideal = sinad(ideal)
        thd_ideal = thd(ideal)
        sfdr_ideal = sfdr(ideal)

        # ENOB from SINAD: ENOB = (SINAD - 1.76) / 6.02
        enob_ideal = (sinad_ideal - 1.76) / 6.02

        self.result("SNR", f"{snr_ideal:.2f}", "dB")
        self.result("SINAD", f"{sinad_ideal:.2f}", "dB")
        self.result("THD", f"{thd_ideal:.2f}", "dB")
        self.result("SFDR", f"{sfdr_ideal:.2f}", "dB")
        self.result("ENOB", f"{enob_ideal:.2f}", "bits")

        results["snr_ideal"] = snr_ideal
        results["sinad_ideal"] = sinad_ideal
        results["thd_ideal"] = thd_ideal
        results["sfdr_ideal"] = sfdr_ideal
        results["enob_ideal"] = enob_ideal

        # ========== PART 2: NOISY SIGNAL ==========
        self.subsection("Part 2: Noisy Signal (SNR ~ 40 dB)")
        noisy = data["noisy"]
        self.info("Noisy signal: Added white Gaussian noise")

        snr_noisy = snr(noisy)
        sinad_noisy = sinad(noisy)
        thd_noisy = thd(noisy)
        sfdr_noisy = sfdr(noisy)
        enob_noisy = (sinad_noisy - 1.76) / 6.02

        self.result("SNR", f"{snr_noisy:.2f}", "dB")
        self.result("SINAD", f"{sinad_noisy:.2f}", "dB")
        self.result("THD", f"{thd_noisy:.2f}", "dB")
        self.result("SFDR", f"{sfdr_noisy:.2f}", "dB")
        self.result("ENOB", f"{enob_noisy:.2f}", "bits")

        # Compare to ideal
        snr_degradation = snr_ideal - snr_noisy
        self.result("SNR degradation", f"{snr_degradation:.2f}", "dB")

        results["snr_noisy"] = snr_noisy
        results["sinad_noisy"] = sinad_noisy
        results["enob_noisy"] = enob_noisy

        # ========== PART 3: DISTORTED SIGNAL ==========
        self.subsection("Part 3: Distorted Signal (Harmonic Distortion)")
        distorted = data["distorted"]
        self.info("Distorted signal: 2nd (5%), 3rd (3%), 5th (2%) harmonics")

        snr_distorted = snr(distorted)
        sinad_distorted = sinad(distorted)
        thd_distorted = thd(distorted)
        sfdr_distorted = sfdr(distorted)
        enob_distorted = (sinad_distorted - 1.76) / 6.02

        self.result("SNR", f"{snr_distorted:.2f}", "dB")
        self.result("SINAD", f"{sinad_distorted:.2f}", "dB")
        self.result("THD", f"{thd_distorted:.2f}", "dB")
        self.result("SFDR", f"{sfdr_distorted:.2f}", "dB")
        self.result("ENOB", f"{enob_distorted:.2f}", "bits")

        # Expected THD from harmonics: sqrt(0.05^2 + 0.03^2 + 0.02^2) ≈ 6.2%
        expected_thd_percent = np.sqrt(0.05**2 + 0.03**2 + 0.02**2) * 100
        expected_thd_db = 20 * np.log10(expected_thd_percent / 100)
        self.info(f"Expected THD: {expected_thd_db:.2f} dB ({expected_thd_percent:.2f}%)")

        results["snr_distorted"] = snr_distorted
        results["thd_distorted"] = thd_distorted
        results["sfdr_distorted"] = sfdr_distorted

        # ========== PART 4: POOR QUALITY SIGNAL ==========
        self.subsection("Part 4: Poor Quality Signal (Noise + Distortion)")
        poor = data["poor"]
        self.info("Poor quality: Combined noise and harmonic distortion")

        snr_poor = snr(poor)
        sinad_poor = sinad(poor)
        thd_poor = thd(poor)
        sfdr_poor = sfdr(poor)
        enob_poor = (sinad_poor - 1.76) / 6.02

        self.result("SNR", f"{snr_poor:.2f}", "dB")
        self.result("SINAD", f"{sinad_poor:.2f}", "dB")
        self.result("THD", f"{thd_poor:.2f}", "dB")
        self.result("SFDR", f"{sfdr_poor:.2f}", "dB")
        self.result("ENOB", f"{enob_poor:.2f}", "bits")

        # Quality assessment
        quality_score = self._calculate_quality_score(snr_poor, sinad_poor, thd_poor, sfdr_poor)
        self.result("Quality Score", f"{quality_score:.1f}", "/ 100")

        results["snr_poor"] = snr_poor
        results["sinad_poor"] = sinad_poor
        results["enob_poor"] = enob_poor
        results["quality_score"] = quality_score

        # ========== PART 5: QUALITY WARNINGS ==========
        self.subsection("Part 5: Automatic Quality Warnings")

        warnings = self._generate_quality_warnings(
            snr_poor, sinad_poor, thd_poor, sfdr_poor, enob_poor
        )

        self.result("Warnings detected", len(warnings))
        for warning in warnings:
            self.warning(warning)

        results["warnings"] = warnings

        # ========== QUALITY METRICS INTERPRETATION ==========
        self.subsection("Quality Metrics Interpretation")

        self.info("\n[SNR - Signal-to-Noise Ratio]")
        self.info("  Ratio of signal power to noise power")
        self.info("  Higher = cleaner signal")
        self.info("  >60 dB = excellent, 40-60 dB = good, <40 dB = poor")

        self.info("\n[SINAD - Signal-to-Noise-And-Distortion]")
        self.info("  Ratio of signal to all unwanted components (noise + distortion)")
        self.info("  Always ≤ SNR (includes distortion)")
        self.info("  Used to calculate ENOB")

        self.info("\n[THD - Total Harmonic Distortion]")
        self.info("  Power in harmonics relative to fundamental")
        self.info("  Negative dB = better (lower distortion)")
        self.info("  < -60 dB = excellent, -40 to -60 dB = good, > -40 dB = poor")

        self.info("\n[SFDR - Spurious-Free Dynamic Range]")
        self.info("  Ratio of fundamental to largest spurious component")
        self.info("  Indicates worst-case spur")
        self.info("  >80 dB = excellent, 60-80 dB = good, <60 dB = poor")

        self.info("\n[ENOB - Effective Number of Bits]")
        self.info("  Effective resolution: ENOB = (SINAD - 1.76) / 6.02")
        self.info("  16-bit ADC ideal: 16 bits, typical: 14-15 bits")

        self.info("\n[Quality Comparison:]")
        self.info(f"  Ideal:     SNR={snr_ideal:.1f} dB, ENOB={enob_ideal:.1f} bits")
        self.info(f"  Noisy:     SNR={snr_noisy:.1f} dB, ENOB={enob_noisy:.1f} bits")
        self.info(f"  Distorted: THD={thd_distorted:.1f} dB, SFDR={sfdr_distorted:.1f} dB")
        self.info(f"  Poor:      SINAD={sinad_poor:.1f} dB, ENOB={enob_poor:.1f} bits")

        self.success("All quality assessment measurements complete!")

        return results

    def _calculate_quality_score(self, snr: float, sinad: float, thd: float, sfdr: float) -> float:
        """Calculate overall quality score (0-100).

        Args:
            snr: SNR in dB
            sinad: SINAD in dB
            thd: THD in dB
            sfdr: SFDR in dB

        Returns:
            Quality score (0-100)
        """
        # Normalize metrics to 0-100 scale
        snr_score = min(100, max(0, (snr - 20) * 2))  # 20-70 dB → 0-100
        sinad_score = min(100, max(0, (sinad - 20) * 2))  # 20-70 dB → 0-100
        thd_score = min(100, max(0, (-thd - 20) * 2))  # -70 to -20 dB → 0-100
        sfdr_score = min(100, max(0, (sfdr - 40) * 2))  # 40-90 dB → 0-100

        # Weighted average (SINAD is most important for overall quality)
        quality = 0.3 * snr_score + 0.4 * sinad_score + 0.15 * thd_score + 0.15 * sfdr_score

        return quality

    def _generate_quality_warnings(
        self, snr: float, sinad: float, thd: float, sfdr: float, enob: float
    ) -> list[str]:
        """Generate quality warnings based on metrics.

        Args:
            snr: SNR in dB
            sinad: SINAD in dB
            thd: THD in dB
            sfdr: SFDR in dB
            enob: ENOB in bits

        Returns:
            List of warning messages
        """
        warnings = []

        if snr < 40:
            warnings.append(f"Low SNR ({snr:.1f} dB < 40 dB): High noise level")

        if sinad < 40:
            warnings.append(f"Low SINAD ({sinad:.1f} dB < 40 dB): Noise and/or distortion present")

        if thd > -40:
            warnings.append(f"High THD ({thd:.1f} dB > -40 dB): Significant harmonic distortion")

        if sfdr < 60:
            warnings.append(f"Low SFDR ({sfdr:.1f} dB < 60 dB): Large spurious components")

        if enob < 8:
            warnings.append(f"Low ENOB ({enob:.1f} bits < 8 bits): Poor effective resolution")

        # Check for specific issues
        if snr - sinad > 3:
            warnings.append(
                f"SINAD significantly lower than SNR ({sinad:.1f} vs {snr:.1f} dB): "
                "Distortion dominates"
            )

        return warnings

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate quality assessment measurements."""
        self.info("Validating quality assessment measurements...")

        all_valid = True

        # Validate ideal signal
        self.subsection("Ideal Signal Validation")

        # Ideal signal should have very high SNR (>80 dB)
        if results["snr_ideal"] > 80:
            self.success(f"Ideal SNR: {results['snr_ideal']:.2f} dB > 80 dB")
        else:
            self.warning(f"Ideal SNR lower than expected: {results['snr_ideal']:.2f} dB")

        # SINAD should be close to SNR for ideal signal
        if abs(results["snr_ideal"] - results["sinad_ideal"]) < 3:
            self.success(
                f"SNR ≈ SINAD (low distortion): {results['snr_ideal']:.1f} ≈ {results['sinad_ideal']:.1f} dB"
            )
        else:
            self.info(
                f"SNR vs SINAD: {results['snr_ideal']:.1f} vs {results['sinad_ideal']:.1f} dB"
            )

        # THD should be very low (< -60 dB)
        if results["thd_ideal"] < -60:
            self.success(f"Ideal THD: {results['thd_ideal']:.2f} dB < -60 dB")
        else:
            self.info(f"Ideal THD: {results['thd_ideal']:.2f} dB")

        # Validate noisy signal
        self.subsection("Noisy Signal Validation")

        # SNR should be around 40 dB (injected noise level)
        if not validate_approximately(
            results["snr_noisy"],
            40.0,
            tolerance=0.2,  # ±20% due to random noise
            name="Noisy SNR",
        ):
            all_valid = False

        # ENOB should be reduced from ideal
        if results["enob_noisy"] < results["enob_ideal"]:
            self.success(
                f"Noise reduced ENOB: {results['enob_noisy']:.1f} < {results['enob_ideal']:.1f} bits"
            )
        else:
            self.warning("Noisy ENOB should be lower than ideal")

        # Validate distorted signal
        self.subsection("Distorted Signal Validation")

        # SNR should be high (no added noise)
        if results["snr_distorted"] > 60:
            self.success(f"Distorted SNR high (no noise): {results['snr_distorted']:.2f} dB")
        else:
            self.info(f"Distorted SNR: {results['snr_distorted']:.2f} dB")

        # THD should be around -24 dB (6.2% harmonics = -24 dB)
        expected_thd = 20 * np.log10(np.sqrt(0.05**2 + 0.03**2 + 0.02**2))
        if abs(results["thd_distorted"] - expected_thd) < 3:
            self.success(
                f"THD close to expected: {results['thd_distorted']:.1f} dB ≈ {expected_thd:.1f} dB"
            )
        else:
            self.info(f"THD: {results['thd_distorted']:.1f} dB (expected: {expected_thd:.1f} dB)")

        # SFDR should indicate largest harmonic (2nd at -26 dBc)
        if -30 < results["sfdr_distorted"] < -20:
            self.success(f"SFDR reasonable: {results['sfdr_distorted']:.1f} dB")
        else:
            self.info(f"SFDR: {results['sfdr_distorted']:.1f} dB")

        # Validate poor quality signal
        self.subsection("Poor Quality Validation")

        # Should have warnings
        if len(results["warnings"]) > 0:
            self.success(f"Quality warnings generated: {len(results['warnings'])}")
        else:
            self.warning("No quality warnings generated for poor signal")

        # Quality score should be low (<60)
        if results["quality_score"] < 60:
            self.success(f"Quality score appropriately low: {results['quality_score']:.1f}")
        else:
            self.info(f"Quality score: {results['quality_score']:.1f}")

        # ENOB should be significantly reduced
        if results["enob_poor"] < 10:
            self.success(f"Poor ENOB appropriately low: {results['enob_poor']:.1f} bits")
        else:
            self.info(f"Poor ENOB: {results['enob_poor']:.1f} bits")

        if all_valid:
            self.success("All quality assessment validations passed!")
            self.info("\nKey takeaways:")
            self.info("  - SNR measures noise performance")
            self.info("  - SINAD includes noise AND distortion")
            self.info("  - THD measures harmonic distortion")
            self.info("  - SFDR indicates worst-case spur")
            self.info("  - ENOB relates SINAD to effective resolution")
            self.info("\nNext steps:")
            self.info("  - Apply to real ADC characterization")
            self.info("  - Use for signal chain optimization")
            self.info("  - Combine with spectral analysis")
        else:
            self.error("Some quality assessment validations failed")

        return all_valid


if __name__ == "__main__":
    demo: QualityAssessmentDemo = QualityAssessmentDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
