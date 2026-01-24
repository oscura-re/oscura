"""Automatic Warning System

Demonstrates intelligent warning and anomaly detection:
- Automatic anomaly detection in signals
- Configurable warning thresholds
- Warning categories and severity levels
- Warning aggregation and reporting

This demonstration shows:
1. How to detect various signal anomalies automatically
2. How to configure warning thresholds for different scenarios
3. How to categorize and prioritize warnings
4. How to generate comprehensive warning reports
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
    add_noise,
    generate_sine_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class WarningSeverity:
    """Warning severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SignalWarning:
    """Signal warning with category and severity."""

    def __init__(
        self,
        category: str,
        message: str,
        severity: str,
        value: float | None = None,
        threshold: float | None = None,
    ):
        """Initialize warning."""
        self.category = category
        self.message = message
        self.severity = severity
        self.value = value
        self.threshold = threshold

    def __repr__(self) -> str:
        """String representation."""
        details = (
            f" (value={self.value:.3f}, threshold={self.threshold:.3f})"
            if self.value is not None
            else ""
        )
        return f"[{self.severity}] {self.category}: {self.message}{details}"


class WarningSystemDemo(BaseDemo):
    """Demonstrate automatic warning and anomaly detection."""

    def __init__(self) -> None:
        """Initialize warning system demonstration."""
        super().__init__(
            name="warning_system",
            description="Automatic signal anomaly detection and warning system",
            capabilities=[
                "oscura.warnings.anomaly_detection",
                "oscura.warnings.threshold_checking",
                "oscura.warnings.warning_categories",
                "oscura.warnings.severity_levels",
            ],
            related_demos=[
                "12_quality_tools/02_quality_scoring.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with various anomalies."""
        self.info("Creating test signals with different anomalies...")

        # Clean signal
        clean = self._create_clean_signal()
        self.info("  ✓ Clean reference signal")

        # Clipped signal
        clipped = self._create_clipped_signal()
        self.info("  ✓ Signal with clipping")

        # Signal with glitches
        glitches = self._create_glitch_signal()
        self.info("  ✓ Signal with transient glitches")

        # Signal with dropout
        dropout = self._create_dropout_signal()
        self.info("  ✓ Signal with data dropout")

        # Noisy signal
        noisy = self._create_noisy_signal()
        self.info("  ✓ High noise signal")

        # Frequency drift
        drift = self._create_frequency_drift_signal()
        self.info("  ✓ Signal with frequency drift")

        # DC offset
        dc_offset = self._create_dc_offset_signal()
        self.info("  ✓ Signal with DC offset")

        return {
            "clean": clean,
            "clipped": clipped,
            "glitches": glitches,
            "dropout": dropout,
            "noisy": noisy,
            "drift": drift,
            "dc_offset": dc_offset,
        }

    def _create_clean_signal(self) -> WaveformTrace:
        """Create clean signal."""
        return generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)

    def _create_clipped_signal(self) -> WaveformTrace:
        """Create clipped signal."""
        # CRITICAL: generate_sine_wave expects (freq, amplitude, duration, sample_rate)
        signal = generate_sine_wave(1000.0, 2.0, 0.1, 100_000.0)
        signal.data = np.clip(signal.data, -1.0, 1.0)
        return signal

    def _create_glitch_signal(self) -> WaveformTrace:
        """Create signal with glitches."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        # Add random spikes
        np.random.seed(42)  # Reproducible glitches
        glitch_indices = np.random.choice(len(signal.data), size=5, replace=False)
        signal.data[glitch_indices] += np.random.uniform(2.0, 5.0, size=5) * np.random.choice(
            [-1, 1], size=5
        )
        return signal

    def _create_dropout_signal(self) -> WaveformTrace:
        """Create signal with dropout."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        # Zero out a section
        dropout_start = len(signal.data) // 3
        dropout_end = dropout_start + 500
        signal.data[dropout_start:dropout_end] = 0
        return signal

    def _create_noisy_signal(self) -> WaveformTrace:
        """Create noisy signal."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        return add_noise(signal, 10.0)  # 10 dB SNR = high noise

    def _create_frequency_drift_signal(self) -> WaveformTrace:
        """Create signal with frequency drift."""
        sample_rate = 100_000.0
        duration = 0.1
        t = np.arange(int(sample_rate * duration)) / sample_rate
        # Frequency sweeps from 1000 to 1100 Hz
        freq_drift = 1000 + 100 * t / duration
        phase = 2 * np.pi * np.cumsum(freq_drift) / sample_rate
        data = np.sin(phase)
        metadata = TraceMetadata(sample_rate=sample_rate)
        return WaveformTrace(data=data, metadata=metadata)

    def _create_dc_offset_signal(self) -> WaveformTrace:
        """Create signal with DC offset."""
        signal = generate_sine_wave(1000.0, 1.0, 0.1, 100_000.0)
        signal.data = signal.data + 0.3  # Add DC offset
        return signal

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate warning system."""
        results: dict[str, Any] = {}

        # Part 1: Individual anomaly detection
        self.section("Part 1: Anomaly Detection Methods")

        signal = data["glitches"]
        self.info("Testing anomaly detection on signal with glitches...")

        anomalies = self._detect_anomalies(signal)
        self.info(f"Detected {len(anomalies)} anomalies:")
        for idx, value in anomalies:
            self.info(f"  Sample {idx}: {value:.3f} (threshold exceeded)")

        results["anomalies"] = anomalies

        # Part 2: Comprehensive warning checks
        self.section("Part 2: Comprehensive Warning Checks")

        all_warnings: dict[str, list[SignalWarning]] = {}

        for name, signal in data.items():
            warnings = self._check_all_warnings(signal)
            all_warnings[name] = warnings

            self.subsection(f"Signal: {name}")
            if warnings:
                for warning in warnings:
                    severity_symbol = {
                        "INFO": "i",
                        "WARNING": "!",
                        "ERROR": "X",
                        "CRITICAL": "!!",
                    }.get(warning.severity, "*")
                    self.info(f"{severity_symbol} {warning}")
            else:
                self.success("No warnings")

        results["all_warnings"] = all_warnings

        # Part 3: Warning aggregation and statistics
        self.section("Part 3: Warning Statistics")

        total_warnings = sum(len(w) for w in all_warnings.values())
        self.info(f"Total warnings: {total_warnings}")

        # Count by severity
        severity_counts = {
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }

        for warnings in all_warnings.values():
            for warning in warnings:
                severity_counts[warning.severity] += 1

        self.info("\nWarnings by severity:")
        for severity, count in severity_counts.items():
            self.info(f"  {severity:10s}: {count}")

        # Count by category
        category_counts: dict[str, int] = {}
        for warnings in all_warnings.values():
            for warning in warnings:
                category_counts[warning.category] = category_counts.get(warning.category, 0) + 1

        self.info("\nWarnings by category:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            self.info(f"  {category:20s}: {count}")

        results["severity_counts"] = severity_counts
        results["category_counts"] = category_counts

        # Part 4: Configurable thresholds
        self.section("Part 4: Configurable Warning Thresholds")

        self.info("Testing with different threshold configurations...\n")

        # Strict thresholds
        strict_config = {
            "max_amplitude": 0.8,
            "max_noise_level": 0.05,
            "max_dc_offset": 0.05,
            "min_snr_db": 40.0,
        }

        strict_warnings = self._check_with_config(data["clean"], strict_config)
        self.info(f"Strict thresholds: {len(strict_warnings)} warnings")

        # Relaxed thresholds
        relaxed_config = {
            "max_amplitude": 1.5,
            "max_noise_level": 0.2,
            "max_dc_offset": 0.3,
            "min_snr_db": 10.0,
        }

        relaxed_warnings = self._check_with_config(data["noisy"], relaxed_config)
        self.info(f"Relaxed thresholds: {len(relaxed_warnings)} warnings")

        results["strict_warnings"] = strict_warnings
        results["relaxed_warnings"] = relaxed_warnings

        # Part 5: Warning report generation
        self.section("Part 5: Comprehensive Warning Report")

        report = self._generate_warning_report(all_warnings)
        self.info("\n" + report)

        results["report"] = report

        return results

    def _detect_anomalies(
        self, signal: WaveformTrace, threshold: float = 3.0
    ) -> list[tuple[int, float]]:
        """Detect anomalies using statistical methods."""
        data = signal.data
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / (mad + 1e-10)

        anomalies = []
        for i, z_score in enumerate(modified_z_scores):
            if abs(z_score) > threshold:
                anomalies.append((i, data[i]))

        return anomalies

    def _check_all_warnings(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check all warning conditions."""
        warnings = []

        # Check clipping
        clipping_warnings = self._check_clipping(signal)
        warnings.extend(clipping_warnings)

        # Check noise level
        noise_warnings = self._check_noise_level(signal)
        warnings.extend(noise_warnings)

        # Check DC offset
        dc_warnings = self._check_dc_offset(signal)
        warnings.extend(dc_warnings)

        # Check dropouts
        dropout_warnings = self._check_dropouts(signal)
        warnings.extend(dropout_warnings)

        # Check glitches
        glitch_warnings = self._check_glitches(signal)
        warnings.extend(glitch_warnings)

        # Check frequency stability
        freq_warnings = self._check_frequency_stability(signal)
        warnings.extend(freq_warnings)

        return warnings

    def _check_clipping(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check for signal clipping.

        Clipping is detected by:
        1. Simple threshold: max/min values near ±1.0
        2. Flat top detection: consecutive samples at the same extreme value
        """
        warnings = []
        data = signal.data
        max_val = np.max(data)
        min_val = np.min(data)

        # Simple threshold check
        clip_threshold = 0.99

        # More robust: check for flat tops (consecutive samples at extreme)
        has_positive_clipping = False
        has_negative_clipping = False

        # Check if signal reaches extremes AND has flat regions
        if max_val >= clip_threshold:
            # Count consecutive samples at max value
            at_max = np.abs(data - max_val) < 0.001
            consecutive = self._count_max_consecutive(at_max)
            # If more than 3 consecutive samples at max, it's clipping
            if consecutive > 3 or np.sum(at_max) > 0.01 * len(data):
                has_positive_clipping = True

        if min_val <= -clip_threshold:
            at_min = np.abs(data - min_val) < 0.001
            consecutive = self._count_max_consecutive(at_min)
            if consecutive > 3 or np.sum(at_min) > 0.01 * len(data):
                has_negative_clipping = True

        if has_positive_clipping:
            warnings.append(
                SignalWarning(
                    "Clipping",
                    "Positive clipping detected",
                    WarningSeverity.ERROR,
                    max_val,
                    clip_threshold,
                )
            )

        if has_negative_clipping:
            warnings.append(
                SignalWarning(
                    "Clipping",
                    "Negative clipping detected",
                    WarningSeverity.ERROR,
                    abs(min_val),
                    clip_threshold,
                )
            )

        return warnings

    def _count_max_consecutive(self, mask: np.ndarray) -> int:
        """Count maximum consecutive True values in a boolean array."""
        if not np.any(mask):
            return 0
        changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def _check_noise_level(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check noise level."""
        warnings = []

        # Estimate noise from high-frequency components
        fft = np.fft.rfft(signal.data)
        magnitude = np.abs(fft)
        noise_floor = np.median(magnitude[len(magnitude) // 2 :])
        signal_power = np.max(magnitude)

        if signal_power > 0:
            snr_db = 20 * np.log10(signal_power / (noise_floor + 1e-10))

            if snr_db < 20:
                warnings.append(
                    SignalWarning(
                        "Noise",
                        "High noise level detected",
                        WarningSeverity.WARNING,
                        snr_db,
                        20.0,
                    )
                )

        return warnings

    def _check_dc_offset(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check DC offset."""
        warnings = []
        dc_offset = np.mean(signal.data)
        threshold = 0.1

        if abs(dc_offset) > threshold:
            severity = WarningSeverity.WARNING if abs(dc_offset) < 0.3 else WarningSeverity.ERROR
            warnings.append(
                SignalWarning(
                    "DC Offset",
                    "Significant DC offset detected",
                    severity,
                    abs(dc_offset),
                    threshold,
                )
            )

        return warnings

    def _check_dropouts(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check for data dropouts."""
        warnings = []

        # Find sections with near-zero values
        threshold = 0.01
        zero_mask = np.abs(signal.data) < threshold

        # Find consecutive zeros
        zero_runs = []
        in_run = False
        run_start = 0

        for i, is_zero in enumerate(zero_mask):
            if is_zero and not in_run:
                in_run = True
                run_start = i
            elif not is_zero and in_run:
                in_run = False
                zero_runs.append((run_start, i - run_start))

        # Warn about long runs of zeros
        for start, length in zero_runs:
            if length > 100:
                warnings.append(
                    SignalWarning(
                        "Dropout",
                        f"Data dropout detected at sample {start}",
                        WarningSeverity.ERROR,
                        float(length),
                        100.0,
                    )
                )

        return warnings

    def _check_glitches(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check for transient glitches."""
        warnings = []
        anomalies = self._detect_anomalies(signal, threshold=4.0)

        if len(anomalies) > 0:
            severity = WarningSeverity.WARNING if len(anomalies) < 5 else WarningSeverity.ERROR
            warnings.append(
                SignalWarning(
                    "Glitch",
                    f"{len(anomalies)} transient glitches detected",
                    severity,
                    float(len(anomalies)),
                    5.0,
                )
            )

        return warnings

    def _check_frequency_stability(self, signal: WaveformTrace) -> list[SignalWarning]:
        """Check frequency stability."""
        warnings = []

        # Simple check: compare frequency in first and second half
        mid = len(signal.data) // 2

        def estimate_freq(data: np.ndarray) -> float:
            fft = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1 / signal.metadata.sample_rate)
            peak_idx = np.argmax(np.abs(fft)[1:]) + 1
            return freqs[peak_idx]

        freq1 = estimate_freq(signal.data[:mid])
        freq2 = estimate_freq(signal.data[mid:])

        drift_pct = 100 * abs(freq2 - freq1) / (freq1 + 1e-10)

        if drift_pct > 5.0:
            warnings.append(
                SignalWarning(
                    "Frequency Drift",
                    f"Frequency drift detected ({drift_pct:.1f}%)",
                    WarningSeverity.WARNING,
                    drift_pct,
                    5.0,
                )
            )

        return warnings

    def _check_with_config(
        self, signal: WaveformTrace, config: dict[str, float]
    ) -> list[SignalWarning]:
        """Check warnings with custom configuration."""
        warnings = []

        peak = np.max(np.abs(signal.data))
        if peak > config["max_amplitude"]:
            warnings.append(
                SignalWarning(
                    "Amplitude",
                    "Amplitude exceeds threshold",
                    WarningSeverity.WARNING,
                    peak,
                    config["max_amplitude"],
                )
            )

        dc_offset = abs(np.mean(signal.data))
        if dc_offset > config["max_dc_offset"]:
            warnings.append(
                SignalWarning(
                    "DC Offset",
                    "DC offset exceeds threshold",
                    WarningSeverity.WARNING,
                    dc_offset,
                    config["max_dc_offset"],
                )
            )

        return warnings

    def _generate_warning_report(self, all_warnings: dict[str, list[SignalWarning]]) -> str:
        """Generate comprehensive warning report."""
        lines = []
        lines.append("=" * 80)
        lines.append("SIGNAL QUALITY WARNING REPORT")
        lines.append("=" * 80)

        for signal_name, warnings in all_warnings.items():
            lines.append(f"\n{signal_name.upper()}")
            lines.append("-" * 80)

            if not warnings:
                lines.append("  No warnings")
            else:
                # Group by severity
                critical = [w for w in warnings if w.severity == "CRITICAL"]
                errors = [w for w in warnings if w.severity == "ERROR"]
                warnings_list = [w for w in warnings if w.severity == "WARNING"]
                info = [w for w in warnings if w.severity == "INFO"]

                if critical:
                    lines.append("  CRITICAL:")
                    for w in critical:
                        lines.append(f"    - {w.message}")

                if errors:
                    lines.append("  ERRORS:")
                    for w in errors:
                        lines.append(f"    - {w.message}")

                if warnings_list:
                    lines.append("  WARNINGS:")
                    for w in warnings_list:
                        lines.append(f"    - {w.message}")

                if info:
                    lines.append("  INFO:")
                    for w in info:
                        lines.append(f"    - {w.message}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating warning system...")
        all_valid = True

        # Check warnings detected
        if "all_warnings" not in results:
            self.error("Missing warning results")
            return False

        all_warnings = results["all_warnings"]

        # Validate specific signals have expected warnings
        if "clipped" in all_warnings:
            clipped_warnings = all_warnings["clipped"]
            if not any(w.category == "Clipping" for w in clipped_warnings):
                self.error("Clipping not detected in clipped signal")
                all_valid = False
            else:
                self.success("Clipping correctly detected in clipped signal")

        if "glitches" in all_warnings:
            glitch_warnings = all_warnings["glitches"]
            if not any(w.category == "Glitch" for w in glitch_warnings):
                self.warning("Glitches not detected in glitch signal")
            else:
                self.success("Glitches correctly detected")

        if "dc_offset" in all_warnings:
            dc_warnings = all_warnings["dc_offset"]
            if not any(w.category == "DC Offset" for w in dc_warnings):
                self.warning("DC offset not detected")
            else:
                self.success("DC offset correctly detected")

        # Validate clean signal has few/no warnings
        if "clean" in all_warnings:
            clean_warnings = all_warnings["clean"]
            if len(clean_warnings) > 2:
                self.warning(f"Clean signal has {len(clean_warnings)} warnings (expected few)")
            else:
                self.success("Clean signal has minimal warnings")

        if all_valid:
            self.success("All warning system checks validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = WarningSystemDemo()
    success = demo.execute()
    exit(0 if success else 1)
