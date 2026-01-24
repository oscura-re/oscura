"""Signal Quality Scoring

Demonstrates comprehensive signal quality assessment:
- Multi-factor quality scoring (0-100 scale)
- SNR, distortion, and noise floor analysis
- Automatic quality warnings
- Quality-based processing decisions

This demonstration shows:
1. How to calculate comprehensive quality scores
2. How to identify quality issues automatically
3. How to make processing decisions based on quality
4. Practical quality thresholds for different applications
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING

from demonstrations.common import (
    BaseDemo,
    add_noise,
    generate_sine_wave,
)

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class QualityScoringDemo(BaseDemo):
    """Demonstrate signal quality scoring and assessment."""

    def __init__(self) -> None:
        """Initialize quality scoring demonstration."""
        super().__init__(
            name="quality_scoring",
            description="Comprehensive signal quality assessment and scoring",
            capabilities=[
                "oscura.quality.overall_score",
                "oscura.quality.snr_estimation",
                "oscura.quality.distortion_analysis",
                "oscura.quality.automatic_warnings",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "12_quality_tools/03_warning_system.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate signals with different quality levels."""
        self.info("Creating test signals with varying quality...")

        # Excellent quality - clean sine wave with very high SNR
        excellent = self._create_signal(freq=1000.0, amplitude=1.0, snr_db=60.0, distortion=0.0)
        self.info("  ✓ Excellent quality (SNR≈60dB, THD<0.1%)")

        # Good quality - low noise
        good = self._create_signal(freq=1000.0, amplitude=1.0, snr_db=40.0, distortion=0.01)
        self.info("  ✓ Good quality (SNR≈40dB, THD≈1%)")

        # Fair quality - moderate noise
        fair = self._create_signal(freq=1000.0, amplitude=1.0, snr_db=26.0, distortion=0.03)
        self.info("  ✓ Fair quality (SNR≈26dB, THD≈3%)")

        # Poor quality - high noise
        poor = self._create_signal(freq=1000.0, amplitude=1.0, snr_db=16.0, distortion=0.1)
        self.info("  ✓ Poor quality (SNR≈16dB, THD≈10%)")

        # Clipped signal
        clipped = self._create_clipped_signal(freq=1000.0, amplitude=2.0)
        self.info("  ✓ Clipped signal (saturation)")

        # Low level signal
        low_level = self._create_signal(freq=1000.0, amplitude=0.05, snr_db=40.0, distortion=0.0)
        self.info("  ✓ Low level signal (poor dynamic range)")

        return {
            "excellent": excellent,
            "good": good,
            "fair": fair,
            "poor": poor,
            "clipped": clipped,
            "low_level": low_level,
        }

    def _create_signal(
        self, freq: float, amplitude: float, snr_db: float, distortion: float
    ) -> WaveformTrace:
        """Create signal with controlled quality parameters.

        Args:
            freq: Signal frequency in Hz
            amplitude: Signal amplitude
            snr_db: Signal-to-noise ratio in dB (higher = less noise)
            distortion: Distortion level (0.0 to 1.0, where 0.1 = 10% THD)

        Returns:
            WaveformTrace with specified quality characteristics
        """
        sample_rate = 100_000.0
        duration = 0.1

        # Base signal - CRITICAL: generate_sine_wave expects (freq, amplitude, duration, sample_rate)
        signal = generate_sine_wave(freq, amplitude, duration, sample_rate)

        # Add harmonic distortion
        if distortion > 0:
            t = np.arange(len(signal.data)) / sample_rate
            harmonics = distortion * amplitude * np.sin(4 * np.pi * freq * t)  # 2nd harmonic
            harmonics += distortion * 0.5 * amplitude * np.sin(6 * np.pi * freq * t)  # 3rd harmonic
            signal.data = signal.data + harmonics

        # Add noise using SNR in dB
        if snr_db < 100:  # Don't add noise for very high SNR (essentially clean)
            signal = add_noise(signal, snr_db)

        return signal

    def _create_clipped_signal(self, freq: float, amplitude: float) -> WaveformTrace:
        """Create clipped signal."""
        sample_rate = 100_000.0
        duration = 0.1
        # CRITICAL: generate_sine_wave expects (freq, amplitude, duration, sample_rate)
        signal = generate_sine_wave(freq, amplitude, duration, sample_rate)
        # Hard clip at ±1.0
        signal.data = np.clip(signal.data, -1.0, 1.0)
        return signal

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate quality scoring."""
        results: dict[str, Any] = {}

        # Part 1: Individual quality metrics
        self.section("Part 1: Individual Quality Metrics")

        signal = data["good"]
        metrics = self._calculate_quality_metrics(signal)

        self.info("Quality metrics for 'good' signal:")
        self.info(f"  SNR:                {metrics['snr']:.1f} dB")
        self.info(f"  SINAD:              {metrics['sinad']:.1f} dB")
        self.info(f"  THD:                {metrics['thd']:.2f}%")
        self.info(f"  Peak amplitude:     {metrics['peak_amplitude']:.3f}")
        self.info(f"  RMS amplitude:      {metrics['rms_amplitude']:.3f}")
        self.info(f"  Crest factor:       {metrics['crest_factor']:.2f}")
        self.info(f"  Clipping detected:  {metrics['clipping']}")
        self.info(f"  DC offset:          {metrics['dc_offset']:.4f}")

        results["individual_metrics"] = metrics

        # Part 2: Composite quality scores
        self.section("Part 2: Composite Quality Scores")

        self.info("Quality scores for all test signals:\n")
        all_scores = {}

        for name, signal in data.items():
            metrics = self._calculate_quality_metrics(signal)
            score = self._calculate_overall_score(metrics)
            grade = self._get_quality_grade(score)

            all_scores[name] = {"score": score, "grade": grade, "metrics": metrics}

            self.info(f"{name:15s}: {score:5.1f}/100 ({grade})")

        results["quality_scores"] = all_scores

        # Part 3: Quality warnings
        self.section("Part 3: Automatic Quality Warnings")

        for name, signal in data.items():
            self.subsection(f"Signal: {name}")
            warnings = self._check_quality_warnings(signal)

            if warnings:
                for warning in warnings:
                    self.warning(warning)
            else:
                self.success("No quality issues detected")

        results["warnings"] = {
            name: self._check_quality_warnings(signal) for name, signal in data.items()
        }

        # Part 4: Quality-based processing decisions
        self.section("Part 4: Quality-Based Processing Decisions")

        self.info("Recommended processing based on quality:\n")

        for name, score_data in all_scores.items():
            score = score_data["score"]
            self.info(f"{name:15s} (score={score:.1f}):")

            recommendations = self._get_processing_recommendations(score)
            for rec in recommendations:
                self.info(f"  - {rec}")
            self.info("")

        results["recommendations"] = {
            name: self._get_processing_recommendations(data["score"])
            for name, data in all_scores.items()
        }

        # Part 5: Quality trends
        self.section("Part 5: Quality Factor Analysis")

        self.info("Quality factor contributions:\n")

        signal_name = "fair"
        metrics = all_scores[signal_name]["metrics"]
        contributions = self._analyze_quality_contributions(metrics)

        self.info(f"Analysis for '{signal_name}' signal:")
        for factor, contribution in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
            impact = "HIGH" if contribution < 0.7 else "MEDIUM" if contribution < 0.9 else "LOW"
            self.info(f"  {factor:20s}: {contribution:.2f} (impact: {impact})")

        results["quality_factors"] = contributions

        return results

    def _calculate_quality_metrics(self, signal: WaveformTrace) -> dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        data = signal.data

        # Amplitude metrics
        peak_amp = np.max(np.abs(data))
        rms_amp = np.sqrt(np.mean(data**2))
        crest_factor = peak_amp / rms_amp if rms_amp > 0 else 0

        # DC offset
        dc_offset = np.mean(data)

        # Clipping detection - check for flat regions at extremes
        max_val = np.max(data)
        min_val = np.min(data)
        clipping = self._detect_clipping(data, max_val, min_val)

        # Frequency domain analysis
        fft = np.fft.rfft(data)
        _freqs = np.fft.rfftfreq(len(data), 1 / signal.metadata.sample_rate)  # For reference
        magnitude = np.abs(fft)

        # Find fundamental
        fundamental_idx = np.argmax(magnitude[1:]) + 1
        fundamental_power = magnitude[fundamental_idx] ** 2

        # Noise floor (exclude DC and fundamental region)
        noise_region = magnitude.copy()
        noise_region[0] = 0  # Exclude DC
        noise_region[max(1, fundamental_idx - 5) : fundamental_idx + 5] = 0  # Exclude fundamental
        noise_power = np.mean(noise_region**2)

        # SNR
        snr_db = 10 * np.log10(fundamental_power / noise_power) if noise_power > 0 else 100

        # THD - find harmonics
        harmonic_power = 0
        for n in range(2, 6):  # 2nd to 5th harmonic
            harm_idx = fundamental_idx * n
            if harm_idx < len(magnitude):
                # Sum power in window around harmonic
                start = max(0, harm_idx - 2)
                end = min(len(magnitude), harm_idx + 3)
                harmonic_power += np.sum(magnitude[start:end] ** 2)

        thd = 100 * np.sqrt(harmonic_power / fundamental_power) if fundamental_power > 0 else 0

        # SINAD
        signal_power = fundamental_power
        noise_distortion_power = noise_power + harmonic_power
        sinad_db = (
            10 * np.log10(signal_power / noise_distortion_power)
            if noise_distortion_power > 0
            else 100
        )

        return {
            "snr": snr_db,
            "sinad": sinad_db,
            "thd": thd,
            "peak_amplitude": peak_amp,
            "rms_amplitude": rms_amp,
            "crest_factor": crest_factor,
            "clipping": clipping,
            "dc_offset": dc_offset,
            "noise_floor": 10 * np.log10(noise_power) if noise_power > 0 else -100,
        }

    def _detect_clipping(self, data: np.ndarray, max_val: float, min_val: float) -> bool:
        """Detect clipping by looking for flat regions at signal extremes.

        Clipping is detected when:
        1. Multiple consecutive samples are at the same extreme value, OR
        2. A significant percentage of samples are at the extreme values
        """
        # Tolerance for considering values as "at the limit"
        clip_tolerance = 0.001

        # Check positive clipping
        at_max = np.abs(data - max_val) < clip_tolerance
        at_min = np.abs(data - min_val) < clip_tolerance

        # Count samples at extremes
        max_count = np.sum(at_max)
        min_count = np.sum(at_min)

        # Clipping if more than 1% of samples are at extremes
        clipping_threshold = 0.01 * len(data)

        if max_count > clipping_threshold or min_count > clipping_threshold:
            return True

        # Also check for consecutive samples at extremes (flat tops)
        max_consecutive_at_max = self._max_consecutive_true(at_max)
        max_consecutive_at_min = self._max_consecutive_true(at_min)

        # If more than 5 consecutive samples at extreme, likely clipping
        return max_consecutive_at_max > 5 or max_consecutive_at_min > 5

    def _max_consecutive_true(self, mask: np.ndarray) -> int:
        """Count maximum consecutive True values in boolean array."""
        if not np.any(mask):
            return 0
        # Find runs of True values
        changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def _calculate_overall_score(self, metrics: dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0.0

        # SNR contribution (max 40 points)
        snr = metrics["snr"]
        if snr >= 60:
            snr_score = 40.0
        elif snr >= 40:
            snr_score = 30.0 + (snr - 40) / 2
        elif snr >= 20:
            snr_score = 10.0 + (snr - 20)
        else:
            snr_score = max(0, snr / 2)
        score += snr_score

        # THD contribution (max 25 points)
        thd = metrics["thd"]
        if thd <= 1.0:
            thd_score = 25.0
        elif thd <= 5.0:
            thd_score = 20.0 - (thd - 1.0) * 1.25
        elif thd <= 10.0:
            thd_score = 15.0 - (thd - 5.0) * 2
        else:
            thd_score = max(0, 5.0 - (thd - 10.0))
        score += thd_score

        # Dynamic range contribution (max 20 points)
        peak = metrics["peak_amplitude"]
        if peak >= 0.5:
            dr_score = 20.0
        elif peak >= 0.1:
            dr_score = 10.0 + (peak - 0.1) / 0.04
        else:
            dr_score = max(0, peak * 100)
        score += dr_score

        # Crest factor contribution (max 15 points)
        crest = metrics["crest_factor"]
        if 1.2 <= crest <= 1.5:  # Good crest factor for sine wave
            score += 15.0
        elif 1.0 <= crest <= 2.0:
            score += 10.0
        else:
            score += 5.0

        # Clipping penalty (max -30 points)
        if metrics["clipping"]:
            score -= 30

        # DC offset penalty (max -10 points)
        dc_penalty = min(10, abs(metrics["dc_offset"]) * 100)
        score -= dc_penalty

        return max(0, min(100, score))

    def _get_quality_grade(self, score: float) -> str:
        """Get letter grade from score."""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 75:
            return "B (Good)"
        elif score >= 60:
            return "C (Fair)"
        elif score >= 40:
            return "D (Poor)"
        else:
            return "F (Unacceptable)"

    def _check_quality_warnings(self, signal: WaveformTrace) -> list[str]:
        """Check for quality issues."""
        warnings = []
        metrics = self._calculate_quality_metrics(signal)

        if metrics["snr"] < 20:
            warnings.append(
                f"Low SNR detected ({metrics['snr']:.1f} dB) - signal may be unreliable"
            )

        if metrics["thd"] > 10:
            warnings.append(
                f"High distortion detected (THD={metrics['thd']:.1f}%) - consider filtering"
            )

        if metrics["clipping"]:
            warnings.append("Signal clipping detected - reduce input amplitude")

        if metrics["peak_amplitude"] < 0.1:
            warnings.append(f"Low signal level ({metrics['peak_amplitude']:.3f}) - increase gain")

        if abs(metrics["dc_offset"]) > 0.1:
            warnings.append(
                f"Significant DC offset ({metrics['dc_offset']:.3f}) - consider AC coupling"
            )

        if metrics["crest_factor"] > 10:
            warnings.append(
                f"Unusual crest factor ({metrics['crest_factor']:.1f}) - check for spikes"
            )

        return warnings

    def _get_processing_recommendations(self, score: float) -> list[str]:
        """Get processing recommendations based on quality score."""
        recommendations = []

        if score >= 90:
            recommendations.append(
                "Signal quality excellent - proceed with high-precision analysis"
            )
            recommendations.append("Suitable for all measurement types")
        elif score >= 75:
            recommendations.append("Signal quality good - reliable for standard analysis")
            recommendations.append("May use advanced processing techniques")
        elif score >= 60:
            recommendations.append("Signal quality fair - apply noise reduction")
            recommendations.append("Use averaging or filtering for better results")
        elif score >= 40:
            recommendations.append("Signal quality poor - significant preprocessing required")
            recommendations.append("Consider signal source improvements")
            recommendations.append("Use robust measurement methods")
        else:
            recommendations.append("Signal quality unacceptable - not recommended for analysis")
            recommendations.append("Check signal source and acquisition settings")
            recommendations.append("May need hardware-level fixes")

        return recommendations

    def _analyze_quality_contributions(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Analyze which factors contribute to quality score."""
        contributions = {}

        # SNR contribution (normalized 0-1)
        contributions["SNR"] = min(1.0, metrics["snr"] / 60.0)

        # THD contribution
        contributions["Low Distortion"] = max(0, 1.0 - metrics["thd"] / 10.0)

        # Clipping contribution
        contributions["No Clipping"] = 0.0 if metrics["clipping"] else 1.0

        # Dynamic range contribution
        contributions["Good Level"] = min(1.0, metrics["peak_amplitude"] / 0.5)

        # DC offset contribution
        contributions["Low DC Offset"] = max(0, 1.0 - abs(metrics["dc_offset"]) * 10)

        return contributions

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating quality scoring...")
        all_valid = True

        # Check quality scores exist
        if "quality_scores" not in results:
            self.error("Missing quality scores")
            return False

        scores = results["quality_scores"]

        # Validate score ordering - excellent MUST score higher than poor
        excellent_score = scores["excellent"]["score"]
        good_score = scores["good"]["score"]
        fair_score = scores["fair"]["score"]
        poor_score = scores["poor"]["score"]

        self.info(f"  Excellent: {excellent_score:.1f}")
        self.info(f"  Good: {good_score:.1f}")
        self.info(f"  Fair: {fair_score:.1f}")
        self.info(f"  Poor: {poor_score:.1f}")

        if excellent_score <= poor_score:
            self.error(
                f"Quality scores not properly ordered: "
                f"excellent ({excellent_score:.1f}) should be > poor ({poor_score:.1f})"
            )
            all_valid = False
        else:
            self.success("Quality scores properly ordered (excellent > poor)")

        # Validate ordering: excellent > good > fair > poor
        if not (excellent_score >= good_score >= fair_score >= poor_score):
            self.warning("Quality scores not strictly ordered")
        else:
            self.success("Quality scores strictly ordered")

        # Validate score range
        for name, data in scores.items():
            score = data["score"]
            if not (0 <= score <= 100):
                self.error(f"Score out of range for {name}: {score}")
                all_valid = False

        # Validate warnings for clipped signal
        if "warnings" in results:
            clipped_warnings = results["warnings"]["clipped"]
            if not any("clipping" in w.lower() for w in clipped_warnings):
                self.error("Clipping not detected in clipped signal")
                all_valid = False
            else:
                self.success("Clipping correctly detected in clipped signal")

        if all_valid:
            self.success("All quality scoring validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = QualityScoringDemo()
    success = demo.execute()
    exit(0 if success else 1)
