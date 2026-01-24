"""Anomaly Detection and Data Quality Assessment for Unknown Protocols

Demonstrates:
- oscura.discovery.find_anomalies - Automatic signal anomaly detection
- oscura.discovery.assess_data_quality - Data quality scoring
- oscura.analyzers.statistics.outliers.detect_outliers - Statistical outlier detection
- oscura.analyzers.statistics.outliers.zscore_outliers - Z-score based outlier detection
- oscura.analyzers.statistics.outliers.iqr_outliers - IQR-based outlier detection
- oscura.analyzers.statistics.outliers.modified_zscore_outliers - MAD-based robust detection
- Practical workflow: finding anomalies in captured protocol traffic
- Method comparison and selection

IEEE Standards: IEEE 1057-2017 (Digitizing Waveform Recorders), IEEE 1241-2010 (ADC Terminology)
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/07_entropy_analysis.py
- 12_quality_tools/01_ensemble_methods.py

This demonstration shows how to automatically detect anomalies, outliers, and
assess data quality when analyzing unknown protocols. Essential for identifying
signal integrity issues, transmission errors, and data corruption.

This is a CRITICAL feature for reverse engineering - anomaly detection reveals
protocol errors, timing violations, and helps focus analysis on interesting events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class AnomalyDetectionDemo(BaseDemo):
    """Demonstrates anomaly detection and data quality assessment."""

    def __init__(self) -> None:
        """Initialize anomaly detection demonstration."""
        super().__init__(
            name="anomaly_detection",
            description="Automatic anomaly detection and data quality assessment",
            capabilities=[
                "oscura.discovery.find_anomalies",
                "oscura.discovery.assess_data_quality",
                "oscura.analyzers.statistics.outliers.detect_outliers",
                "oscura.analyzers.statistics.outliers.zscore_outliers",
                "oscura.analyzers.statistics.outliers.iqr_outliers",
                "oscura.analyzers.statistics.outliers.modified_zscore_outliers",
            ],
            ieee_standards=["IEEE 1057-2017", "IEEE 1241-2010"],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/07_entropy_analysis.py",
                "12_quality_tools/01_ensemble_methods.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test waveforms with known anomalies and outliers.

        Creates multiple test scenarios:
        1. Clean signal with statistical outliers
        2. Signal with glitches and noise spikes
        3. Signal with timing violations
        4. Protocol data with transmission errors

        Returns:
            Dictionary with test waveforms and ground truth
        """
        self.section("Generating Test Data with Known Anomalies")

        # === Scenario 1: Statistical Outliers in Measurement Data ===
        self.subsection("Scenario 1: Measurement Data with Outliers")

        # Generate mostly normal data with some outliers
        np.random.seed(42)
        n_samples = 1000
        normal_data = np.random.normal(loc=5.0, scale=0.5, size=n_samples)

        # Insert known outliers at specific positions
        outlier_indices = [100, 250, 500, 750]
        outlier_values = [10.0, -2.0, 12.0, -3.0]

        measurement_data = normal_data.copy()
        for idx, val in zip(outlier_indices, outlier_values, strict=True):
            measurement_data[idx] = val

        self.info(f"Generated {n_samples} measurement samples")
        self.info(f"Inserted {len(outlier_indices)} known outliers")
        self.info(f"  Outlier positions: {outlier_indices}")
        self.info(f"  Outlier values: {outlier_values}")

        # === Scenario 2: Digital Signal with Anomalies ===
        self.subsection("Scenario 2: Digital Signal with Glitches and Noise")

        # Generate clean square wave
        sample_rate = 10e6  # 10 MS/s
        duration = 0.001  # 1 ms
        n_samples_digital = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples_digital)

        # 100 kHz square wave
        signal_freq = 100e3
        clean_signal = np.sign(np.sin(2 * np.pi * signal_freq * t)) * 3.3

        # Add some noise
        noisy_signal = clean_signal + np.random.normal(0, 0.1, n_samples_digital)

        # Insert glitches (brief spikes)
        glitch_indices = [1500, 3200, 6800]
        for idx in glitch_indices:
            if idx < len(noisy_signal) - 5:
                # 50ns glitch (5 samples at 10 MS/s)
                noisy_signal[idx : idx + 5] = 5.0  # Spike above normal high level

        # Insert noise spikes
        noise_spike_indices = [2500, 4500, 5500, 8000]
        for idx in noise_spike_indices:
            if idx < len(noisy_signal):
                noisy_signal[idx] += np.random.choice([-1.5, 1.5])

        self.info(f"Generated {n_samples_digital} samples at {sample_rate / 1e6:.0f} MS/s")
        self.info(f"Inserted {len(glitch_indices)} glitches")
        self.info(f"Inserted {len(noise_spike_indices)} noise spikes")

        # === Scenario 3: Signal with Timing Violations ===
        self.subsection("Scenario 3: Protocol Signal with Timing Violations")

        # Generate mostly regular clock with some timing violations
        n_pulses = 50
        nominal_period_samples = 200
        timing_signal = np.zeros(n_pulses * nominal_period_samples)

        pulse_starts = []
        current_pos = 0

        for i in range(n_pulses):
            # Most pulses are regular
            if i in [10, 25, 40]:  # Timing violations
                # Irregular timing (±20%)
                period = int(nominal_period_samples * np.random.choice([0.8, 1.25]))
            else:
                period = nominal_period_samples

            if current_pos + period // 2 < len(timing_signal):
                # Create pulse (50% duty cycle)
                timing_signal[current_pos : current_pos + period // 2] = 3.3
                pulse_starts.append(current_pos)
                current_pos += period

        timing_signal = timing_signal[:current_pos]

        self.info(f"Generated {n_pulses} clock pulses")
        self.info("Inserted 3 timing violations (±20% period deviation)")

        # === Scenario 4: Protocol Data with Transmission Errors ===
        self.subsection("Scenario 4: Protocol Data with Errors")

        # Simulate captured protocol bytes with some errors
        n_bytes = 100
        protocol_data = np.random.randint(0, 256, size=n_bytes, dtype=np.uint8)

        # Insert some error patterns
        # Corrupted bytes (stuck bits, bit flips)
        protocol_data[20] = 0xFF  # All bits high (stuck)
        protocol_data[45] = 0x00  # All bits low (stuck)
        protocol_data[70] = protocol_data[69] ^ 0xFF  # Bit flip

        # Convert to voltage levels for waveform representation
        protocol_waveform = np.repeat(protocol_data.astype(np.float64) / 255.0 * 3.3, 10)

        self.info(f"Generated {n_bytes} protocol bytes")
        self.info("Inserted 3 transmission errors (stuck bits, bit flips)")

        return {
            "measurement_data": measurement_data,
            "measurement_outliers": outlier_indices,
            "digital_signal": noisy_signal,
            "digital_sample_rate": sample_rate,
            "digital_time": t,
            "glitch_indices": glitch_indices,
            "noise_spike_indices": noise_spike_indices,
            "timing_signal": timing_signal,
            "timing_sample_rate": sample_rate,
            "protocol_waveform": protocol_waveform,
            "protocol_data": protocol_data,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute anomaly detection demonstration."""
        from oscura.analyzers.statistics.outliers import (
            detect_outliers,
            iqr_outliers,
            modified_zscore_outliers,
            zscore_outliers,
        )
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.discovery import assess_data_quality, find_anomalies

        results: dict[str, Any] = {}

        # ===== Part 1: Statistical Outlier Detection =====
        self.section("Part 1: Statistical Outlier Detection on Measurement Data")

        measurement_data = data["measurement_data"]
        known_outliers = set(data["measurement_outliers"])

        # Method 1: Z-Score
        self.subsection("Method 1: Standard Z-Score Outliers")

        zscore_result = zscore_outliers(measurement_data, threshold=3.0)

        self.result("Outliers detected", zscore_result.count)
        self.result("Detection method", zscore_result.method)
        self.result("Threshold", zscore_result.threshold)

        if zscore_result.count > 0:
            self.info(f"Outlier indices: {zscore_result.indices.tolist()[:10]}")
            self.info(f"Outlier z-scores: {zscore_result.scores[:10]}")

        # Check detection accuracy
        detected_known = sum(1 for idx in zscore_result.indices if idx in known_outliers)
        self.result("Known outliers detected", f"{detected_known}/{len(known_outliers)}")

        results["zscore_detected"] = zscore_result.count
        results["zscore_accuracy"] = detected_known / len(known_outliers) if known_outliers else 0

        # Method 2: Modified Z-Score (MAD-based, more robust)
        self.subsection("Method 2: Modified Z-Score (MAD-based)")

        mod_zscore_result = modified_zscore_outliers(measurement_data, threshold=3.5)

        self.result("Outliers detected", mod_zscore_result.count)
        self.result("Detection method", mod_zscore_result.method)

        if mod_zscore_result.count > 0:
            self.info(f"Outlier indices: {mod_zscore_result.indices.tolist()[:10]}")

        detected_known = sum(1 for idx in mod_zscore_result.indices if idx in known_outliers)
        self.result("Known outliers detected", f"{detected_known}/{len(known_outliers)}")

        results["modified_zscore_detected"] = mod_zscore_result.count
        results["modified_zscore_accuracy"] = (
            detected_known / len(known_outliers) if known_outliers else 0
        )

        # Method 3: IQR (Interquartile Range)
        self.subsection("Method 3: IQR Method")

        iqr_result, fences = iqr_outliers(measurement_data, multiplier=1.5, return_fences=True)

        self.result("Outliers detected", iqr_result.count)
        self.result("Q1", f"{fences['q1']:.2f}")
        self.result("Q3", f"{fences['q3']:.2f}")
        self.result("IQR", f"{fences['iqr']:.2f}")
        self.result("Lower fence", f"{fences['lower_fence']:.2f}")
        self.result("Upper fence", f"{fences['upper_fence']:.2f}")

        if iqr_result.count > 0:
            self.info(f"Outlier indices: {iqr_result.indices.tolist()[:10]}")

        detected_known = sum(1 for idx in iqr_result.indices if idx in known_outliers)
        self.result("Known outliers detected", f"{detected_known}/{len(known_outliers)}")

        results["iqr_detected"] = iqr_result.count
        results["iqr_accuracy"] = detected_known / len(known_outliers) if known_outliers else 0

        # Method Comparison
        self.subsection("Method Comparison Summary")

        self.info("Detection rates:")
        self.info(f"  Z-Score:          {zscore_result.count:3d} outliers")
        self.info(f"  Modified Z-Score: {mod_zscore_result.count:3d} outliers")
        self.info(f"  IQR:              {iqr_result.count:3d} outliers")
        self.info("")
        self.info("Accuracy (known outliers detected):")
        self.info(f"  Z-Score:          {results['zscore_accuracy']:.1%}")
        self.info(f"  Modified Z-Score: {results['modified_zscore_accuracy']:.1%}")
        self.info(f"  IQR:              {results['iqr_accuracy']:.1%}")

        # ===== Part 2: Signal Anomaly Detection =====
        self.section("Part 2: Automatic Signal Anomaly Detection")

        digital_signal = data["digital_signal"]
        sample_rate = data["digital_sample_rate"]

        # Create WaveformTrace for anomaly detection
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="Digital Signal",
        )
        digital_trace = WaveformTrace(data=digital_signal, metadata=metadata)

        self.subsection("Signal-Specific Anomaly Detection")

        anomalies = find_anomalies(digital_trace, min_confidence=0.7)

        self.result("Total anomalies detected", len(anomalies))

        # Initialize by_type before use
        by_type: dict[str, int] = {}

        if anomalies:
            # Count by type
            for anom in anomalies:
                by_type[anom.type] = by_type.get(anom.type, 0) + 1

            self.subsection("Anomalies by Type")
            for anom_type, count in sorted(by_type.items()):
                self.info(f"  {anom_type:20s}: {count:3d}")

            # Show first few anomalies
            self.subsection("Sample Anomalies (first 5)")
            for i, anom in enumerate(anomalies[:5]):
                self.info(f"Anomaly {i + 1}:")
                self.info(f"  Timestamp:   {anom.timestamp_us:.2f} μs")
                self.info(f"  Type:        {anom.type}")
                self.info(f"  Severity:    {anom.severity}")
                self.info(f"  Duration:    {anom.duration_ns:.0f} ns")
                self.info(f"  Confidence:  {anom.confidence:.2f}")
                self.info(f"  Description: {anom.description}")
                self.info("")

        results["total_anomalies"] = len(anomalies)
        results["anomalies_by_type"] = by_type if anomalies else {}

        # Check detection of known glitches
        glitch_count = by_type.get("glitch", 0) + by_type.get("noise_spike", 0)
        expected_glitches = len(data["glitch_indices"]) + len(data["noise_spike_indices"])
        self.subsection("Known Anomaly Detection Rate")
        self.info(
            f"Expected glitches/spikes: {expected_glitches}, "
            f"Detected: {glitch_count} (~{glitch_count / expected_glitches:.0%} recall)"
        )

        results["glitch_detection_rate"] = (
            glitch_count / expected_glitches if expected_glitches else 0
        )

        # ===== Part 3: Data Quality Assessment =====
        self.section("Part 3: Data Quality Assessment")

        self.subsection("Protocol Decode Quality Assessment")

        # Assess quality for protocol decoding scenario
        quality = assess_data_quality(
            digital_trace, scenario="protocol_decode", protocol_params={"clock_freq_mhz": 0.1}
        )

        self.result("Overall Status", quality.status)
        self.result("Confidence", f"{quality.confidence:.2f}")

        self.subsection("Quality Metrics")
        for metric in quality.metrics:
            status_icon = "✓" if metric.passed else "✗"
            self.info(f"{status_icon} {metric.name}:")
            self.info(f"    Current: {metric.current_value:.2f} {metric.unit}")
            self.info(f"    Required: {metric.required_value:.2f} {metric.unit}")
            self.info(f"    Status: {metric.status}")
            if metric.explanation:
                self.info(f"    Issue: {metric.explanation}")
            if metric.recommendation:
                self.info(f"    Fix: {metric.recommendation}")
            self.info("")

        results["quality_status"] = quality.status
        results["quality_confidence"] = quality.confidence
        results["quality_metrics_passed"] = sum(1 for m in quality.metrics if m.passed)
        results["quality_metrics_total"] = len(quality.metrics)

        # Show improvement suggestions if any
        if quality.improvement_suggestions:
            self.subsection("Improvement Suggestions")
            for i, suggestion in enumerate(quality.improvement_suggestions):
                self.info(f"{i + 1}. {suggestion['action']}")
                self.info(f"   Benefit: {suggestion['expected_benefit']}")
                self.info(f"   Difficulty: {suggestion['difficulty_level']}")
                self.info("")

        # ===== Part 4: Practical Workflow - Unknown Protocol Analysis =====
        self.section("Part 4: Practical Workflow - Unknown Protocol Anomaly Detection")

        self.subsection("Scenario: Captured Unknown Protocol Traffic")
        self.info("Task: Identify transmission errors and signal integrity issues")
        self.info("")

        # Step 1: Quality check
        self.info("Step 1: Assess data quality")
        protocol_metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="Protocol Capture",
        )
        protocol_trace = WaveformTrace(data=data["protocol_waveform"], metadata=protocol_metadata)

        protocol_quality = assess_data_quality(
            protocol_trace, scenario="protocol_decode", protocol_params={"clock_freq_mhz": 0.1}
        )
        self.result("  Capture quality", protocol_quality.status)

        # Step 2: Find anomalies
        self.info("Step 2: Detect signal anomalies")
        protocol_anomalies = find_anomalies(protocol_trace, min_confidence=0.6)
        self.result("  Anomalies found", len(protocol_anomalies))

        if protocol_anomalies:
            # Focus on critical/warning severity
            critical = [a for a in protocol_anomalies if a.severity in ["CRITICAL", "WARNING"]]
            self.result("  Critical/Warning", len(critical))

            if critical:
                self.info("  High-priority anomalies:")
                for anom in critical[:3]:
                    self.info(
                        f"    - {anom.timestamp_us:.1f}μs: {anom.type} "
                        f"({anom.severity}) - {anom.description}"
                    )

        # Step 3: Statistical outlier check on byte values
        self.info("Step 3: Check for statistical outliers in data bytes")
        protocol_bytes = data["protocol_data"].astype(np.float64)
        byte_outliers = detect_outliers(protocol_bytes, method="modified_zscore", threshold=3.5)
        self.result("  Outlier bytes", byte_outliers.count)

        if byte_outliers.count > 0:
            self.info(f"  Outlier positions: {byte_outliers.indices.tolist()}")
            self.info(f"  Outlier values: {byte_outliers.values.astype(int).tolist()}")

        results["protocol_anomalies"] = len(protocol_anomalies)
        results["protocol_critical"] = len(
            [a for a in protocol_anomalies if a.severity == "CRITICAL"]
        )
        results["protocol_outlier_bytes"] = byte_outliers.count

        # Step 4: Analysis recommendation
        self.subsection("Analysis Recommendation")
        if protocol_quality.status == "FAIL":
            self.warning("Data quality is poor - consider recapturing with better settings")
        elif len(protocol_anomalies) > 10:
            self.info("Multiple anomalies detected - focus analysis on anomaly-free regions")
        else:
            self.success("Data quality is good - proceed with protocol analysis")

        # ===== Part 5: Method Selection Guide =====
        self.section("Part 5: Anomaly Detection Method Selection Guide")

        self.info("When to use each method:")
        self.info("")
        self.info("Z-Score (zscore_outliers):")
        self.info("  ✓ Use for: Normally distributed data")
        self.info("  ✓ Use for: Low contamination (<5% outliers)")
        self.info("  ✗ Avoid: Skewed distributions, high contamination")
        self.info("  Threshold: 2.0 (liberal), 3.0 (standard), 4.0 (conservative)")
        self.info("")
        self.info("Modified Z-Score (modified_zscore_outliers):")
        self.info("  ✓ Use for: Contaminated data (up to 50% outliers)")
        self.info("  ✓ Use for: Non-normal distributions")
        self.info("  ✓ Use for: General-purpose robust detection")
        self.info("  Threshold: 3.5 (recommended by Iglewicz & Hoaglin)")
        self.info("")
        self.info("IQR (iqr_outliers):")
        self.info("  ✓ Use for: Skewed distributions")
        self.info("  ✓ Use for: Exploratory data analysis")
        self.info("  ✓ Use for: Visualization (box plots)")
        self.info("  Multiplier: 1.5 (standard), 3.0 (extreme outliers only)")
        self.info("")
        self.info("Signal Anomaly Detection (find_anomalies):")
        self.info("  ✓ Use for: Waveform/digital signal analysis")
        self.info("  ✓ Use for: Detecting glitches, timing issues, ringing")
        self.info("  ✓ Use for: Protocol reverse engineering")
        self.info("  Min confidence: 0.5 (sensitive), 0.7 (balanced), 0.9 (specific)")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate anomaly detection results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        all_valid = True

        # Validate outlier detection methods
        self.subsection("Validating Outlier Detection")

        # At least one method should detect most outliers (>50% accuracy)
        best_accuracy = max(
            results.get("zscore_accuracy", 0),
            results.get("modified_zscore_accuracy", 0),
            results.get("iqr_accuracy", 0),
        )

        if best_accuracy >= 0.5:
            self.success(f"Best detection accuracy: {best_accuracy:.1%} (>= 50%)")
        else:
            self.error(f"Poor outlier detection accuracy: {best_accuracy:.1%} (expected >= 50%)")
            all_valid = False

        # Validate signal anomaly detection
        self.subsection("Validating Signal Anomaly Detection")

        total_anomalies = results.get("total_anomalies", 0)
        if total_anomalies > 0:
            self.success(f"Detected {total_anomalies} signal anomalies")
        else:
            self.warning("No signal anomalies detected (expected some)")

        # Check glitch detection rate
        glitch_rate = results.get("glitch_detection_rate", 0)
        if glitch_rate >= 0.3:  # At least 30% recall
            self.success(f"Glitch detection rate: {glitch_rate:.0%} (>= 30%)")
        else:
            self.warning(f"Low glitch detection rate: {glitch_rate:.0%} (expected >= 30%)")

        # Validate data quality assessment
        self.subsection("Validating Data Quality Assessment")

        quality_status = results.get("quality_status", "FAIL")
        if quality_status in ["PASS", "WARNING"]:
            self.success(f"Quality assessment completed: {quality_status}")
        else:
            self.warning(f"Quality assessment result: {quality_status}")

        metrics_passed = results.get("quality_metrics_passed", 0)
        metrics_total = results.get("quality_metrics_total", 0)
        if metrics_total > 0:
            pass_rate = metrics_passed / metrics_total
            self.result(
                "Quality metrics passed", f"{metrics_passed}/{metrics_total} ({pass_rate:.0%})"
            )

        # Validate practical workflow
        self.subsection("Validating Practical Workflow")

        protocol_anomalies = results.get("protocol_anomalies", 0)
        if protocol_anomalies >= 0:
            self.success(f"Protocol workflow completed: {protocol_anomalies} anomalies detected")
        else:
            self.error("Protocol workflow failed")
            all_valid = False

        # Overall validation
        if all_valid:
            self.success("All anomaly detection validations passed!")
        else:
            self.error("Some anomaly detection validations failed")

        return all_valid


if __name__ == "__main__":
    demo = AnomalyDetectionDemo()
    success = demo.execute()
    exit(0 if success else 1)
