"""Signal Classification and Logic Family Detection: Automatic signal characterization

Demonstrates:
- oscura.inference.classify_signal - Automatic signal type detection (digital/analog/mixed)
- oscura.inference.detect_logic_family - Logic family identification (TTL, CMOS, LVDS)
- oscura.inference.detect_protocol - Protocol family inference from signal patterns
- oscura.discovery.characterize_signal - Complete signal characterization workflow
- Automatic unknown signal analysis
- Confidence scoring and alternative suggestions

IEEE Standards: IEEE 181-2011 (Transitional Waveform Definitions)
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 03_protocol_decoding/06_auto_detection.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows how to automatically classify unknown signals without
prior knowledge. It generates various signal types (analog, digital, mixed-signal),
identifies their characteristics, detects logic families from voltage levels, and
infers likely protocol families. This is essential for reverse engineering unknown
hardware interfaces.

This is a P0 CRITICAL feature - demonstrates automatic signal intelligence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class SignalClassificationDemo(BaseDemo):
    """Demonstrates automatic signal classification and logic family detection."""

    def __init__(self) -> None:
        """Initialize signal classification demonstration."""
        super().__init__(
            name="signal_classification",
            description="Automatic signal type classification and logic family detection",
            capabilities=[
                "oscura.inference.classify_signal",
                "oscura.inference.detect_logic_family",
                "oscura.inference.detect_protocol",
                "oscura.discovery.characterize_signal",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "03_protocol_decoding/06_auto_detection.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )
        self.test_signals: dict[str, WaveformTrace] = {}

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with various types and logic families.

        Creates signals representing:
        - Pure analog signal (sine wave)
        - Digital TTL signal (5V logic)
        - Digital CMOS 3.3V signal
        - Digital LVCMOS 1.8V signal
        - Mixed-signal (PWM-like)
        - UART-like serial data

        Returns:
            Dictionary with test signal traces
        """
        self.section("Generating Test Signals")

        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10ms
        n_samples = int(sample_rate * duration)
        time_axis = np.linspace(0, duration, n_samples, endpoint=False)

        # ===== Signal 1: Pure Analog (Sine Wave) =====
        self.subsection("Signal 1: Pure Analog Signal")
        freq = 1000  # 1 kHz
        analog_data = 2.5 + 1.0 * np.sin(2 * np.pi * freq * time_axis)
        # Add noise
        analog_data += np.random.normal(0, 0.01, len(analog_data))

        analog_metadata = TraceMetadata(sample_rate=sample_rate)
        analog_trace = WaveformTrace(data=analog_data, metadata=analog_metadata)
        self.test_signals["analog"] = analog_trace
        self.info(f"Analog: {len(analog_data)} samples, {1000}Hz sine wave, 2.5V ± 1V")

        # ===== Signal 2: Digital TTL (5V logic) =====
        self.subsection("Signal 2: Digital TTL Signal")
        # 100 kHz square wave with TTL levels
        ttl_freq = 100e3
        ttl_data = np.where(
            np.sin(2 * np.pi * ttl_freq * time_axis) > 0,
            5.0,  # VOH typical
            0.2,  # VOL typical
        )
        # Add small noise
        ttl_data += np.random.normal(0, 0.05, len(ttl_data))

        ttl_metadata = TraceMetadata(sample_rate=sample_rate)
        ttl_trace = WaveformTrace(data=ttl_data, metadata=ttl_metadata)
        self.test_signals["ttl"] = ttl_trace
        self.info(f"TTL: {len(ttl_data)} samples, {100e3}Hz square wave, 0.2V/5.0V levels")

        # ===== Signal 3: Digital CMOS 3.3V =====
        self.subsection("Signal 3: Digital CMOS 3.3V Signal")
        # 50 kHz square wave with CMOS 3.3V levels
        cmos_freq = 50e3
        cmos_data = np.where(
            np.sin(2 * np.pi * cmos_freq * time_axis) > 0,
            3.3,  # VOH
            0.0,  # VOL
        )
        # Add small noise
        cmos_data += np.random.normal(0, 0.03, len(cmos_data))

        cmos_metadata = TraceMetadata(sample_rate=sample_rate)
        cmos_trace = WaveformTrace(data=cmos_data, metadata=cmos_metadata)
        self.test_signals["cmos_3v3"] = cmos_trace
        self.info(f"CMOS 3.3V: {len(cmos_data)} samples, {50e3}Hz square wave, 0V/3.3V levels")

        # ===== Signal 4: Digital LVCMOS 1.8V =====
        self.subsection("Signal 4: Digital LVCMOS 1.8V Signal")
        # 200 kHz square wave with LVCMOS 1.8V levels
        lvcmos_freq = 200e3
        lvcmos_data = np.where(
            np.sin(2 * np.pi * lvcmos_freq * time_axis) > 0,
            1.8,  # VOH
            0.0,  # VOL
        )
        # Add small noise
        lvcmos_data += np.random.normal(0, 0.02, len(lvcmos_data))

        lvcmos_metadata = TraceMetadata(sample_rate=sample_rate)
        lvcmos_trace = WaveformTrace(data=lvcmos_data, metadata=lvcmos_metadata)
        self.test_signals["lvcmos_1v8"] = lvcmos_trace
        self.info(f"LVCMOS 1.8V: {len(lvcmos_data)} samples, {200e3}Hz square wave, 0V/1.8V levels")

        # ===== Signal 5: Mixed Signal (PWM-like) =====
        self.subsection("Signal 5: Mixed Signal (PWM)")
        # Variable duty cycle PWM with analog filtering
        pwm_freq = 10e3
        duty_cycle = 0.5 + 0.3 * np.sin(2 * np.pi * 100 * time_axis)  # Varying duty cycle
        pwm_carrier = np.sin(2 * np.pi * pwm_freq * time_axis)
        pwm_data = np.where(pwm_carrier > (1 - 2 * duty_cycle), 3.3, 0.0)
        # Add analog filtering (RC low-pass effect)
        from scipy import signal as sp_signal

        b, a = sp_signal.butter(2, 0.1)
        pwm_data = sp_signal.filtfilt(b, a, pwm_data)
        pwm_data += np.random.normal(0, 0.05, len(pwm_data))

        pwm_metadata = TraceMetadata(sample_rate=sample_rate)
        pwm_trace = WaveformTrace(data=pwm_data, metadata=pwm_metadata)
        self.test_signals["pwm"] = pwm_trace
        self.info(f"PWM Mixed: {len(pwm_data)} samples, {10e3}Hz carrier with analog filtering")

        # ===== Signal 6: UART-like Serial Data =====
        self.subsection("Signal 6: UART-like Serial Data")
        # Simulate UART transmission at 9600 baud
        baud_rate = 9600
        bit_duration = 1.0 / baud_rate
        samples_per_bit = int(sample_rate * bit_duration)

        # Generate a few bytes: "ABC" = 0x41, 0x42, 0x43
        # UART frame: start bit (0), 8 data bits (LSB first), stop bit (1)
        uart_bits = []
        for byte_val in [0x41, 0x42, 0x43]:
            uart_bits.append(0)  # Start bit
            for i in range(8):
                uart_bits.append((byte_val >> i) & 1)  # Data bits LSB first
            uart_bits.append(1)  # Stop bit

        # Create signal with TTL levels
        uart_data = np.ones(n_samples) * 3.3  # Idle high
        for i, bit in enumerate(uart_bits):
            start_sample = i * samples_per_bit
            end_sample = min(start_sample + samples_per_bit, n_samples)
            if start_sample < n_samples:
                uart_data[start_sample:end_sample] = 3.3 if bit else 0.0

        # Add noise
        uart_data += np.random.normal(0, 0.03, len(uart_data))

        uart_metadata = TraceMetadata(sample_rate=sample_rate)
        uart_trace = WaveformTrace(data=uart_data, metadata=uart_metadata)
        self.test_signals["uart"] = uart_trace
        self.info(f"UART: {len(uart_data)} samples, 9600 baud, 3 bytes transmitted")

        self.result("Total test signals", len(self.test_signals))
        return {"signals": self.test_signals}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute signal classification and logic family detection."""
        results: dict[str, Any] = {}
        signals = data["signals"]

        # ===== Phase 1: Automatic Signal Type Classification =====
        self.section("Part 1: Automatic Signal Type Classification")

        from oscura.inference import classify_signal

        classifications = {}
        for name, trace in signals.items():
            self.subsection(f"Classifying: {name}")

            classification = classify_signal(trace.data, trace.metadata.sample_rate)

            self.info(f"Signal type:    {classification['signal_type']}")
            self.info(f"Is digital:     {classification['is_digital']}")
            self.info(f"Is periodic:    {classification['is_periodic']}")
            self.info(f"Confidence:     {classification['confidence']:.2f}")
            self.info(f"Characteristics: {', '.join(classification['characteristics'])}")

            if classification["frequency_estimate"] is not None:
                self.info(f"Frequency:      {classification['frequency_estimate']:.3e} Hz")

            if classification["snr_db"] is not None:
                self.info(f"SNR:            {classification['snr_db']:.1f} dB")

            if classification["levels"] is not None:
                self.info(
                    f"Logic levels:   Low={classification['levels']['low']:.2f}V, "
                    f"High={classification['levels']['high']:.2f}V"
                )

            classifications[name] = classification

        results["classifications"] = classifications

        # ===== Phase 2: Logic Family Detection =====
        self.section("Part 2: Logic Family Identification")

        from oscura.inference import detect_logic_family

        logic_families = {}
        for name, trace in signals.items():
            # Only analyze digital/mixed signals
            if classifications[name]["is_digital"] or classifications[name]["signal_type"] in [
                "mixed",
                "pwm",
            ]:
                self.subsection(f"Detecting Logic Family: {name}")

                try:
                    family_result = detect_logic_family(trace, return_candidates=True)

                    if family_result["primary"]:
                        primary = family_result["primary"]
                        self.success(f"Primary: {primary['name']}")
                        self.info(f"Confidence:  {primary['confidence']:.2%}")
                        self.info(f"VDD:         {primary['vdd']:.2f}V")
                        self.info(f"Detected VOH: {primary['detected_voh']:.2f}V")
                        self.info(f"Detected VOL: {primary['detected_vol']:.2f}V")

                        # Show candidates if available
                        if "candidates" in family_result and len(family_result["candidates"]) > 1:
                            self.info("Alternative candidates:")
                            for candidate in family_result["candidates"][:3]:
                                if candidate["name"] != primary["name"]:
                                    self.info(
                                        f"  - {candidate['name']}: "
                                        f"confidence={candidate['confidence']:.2%}"
                                    )

                        logic_families[name] = family_result
                    else:
                        self.warning(f"Could not determine logic family for {name}")
                        logic_families[name] = None

                except Exception as e:
                    self.warning(f"Logic family detection failed for {name}: {e}")
                    logic_families[name] = None
            else:
                self.info(f"Skipping {name} (not digital)")
                logic_families[name] = None

        results["logic_families"] = logic_families

        # ===== Phase 3: Protocol Family Inference =====
        self.section("Part 3: Protocol Family Inference")

        from oscura.inference import detect_protocol

        protocol_inferences = {}
        for name, trace in signals.items():
            # Only analyze digital signals that might be protocols
            if classifications[name]["is_digital"]:
                self.subsection(f"Inferring Protocol: {name}")

                try:
                    protocol_result = detect_protocol(trace)

                    if protocol_result["likely_protocol"] != "unknown":
                        self.success(f"Likely protocol: {protocol_result['likely_protocol']}")
                        self.info(f"Confidence:      {protocol_result['confidence']:.2f}")
                        self.info(f"Baud rate:       {protocol_result.get('baud_rate', 'N/A')}")

                        if protocol_result.get("characteristics"):
                            self.info("Characteristics:")
                            for char in protocol_result["characteristics"]:
                                self.info(f"  - {char}")

                        if protocol_result.get("alternatives"):
                            self.info("Alternative protocols:")
                            for alt_name, alt_conf in protocol_result["alternatives"][:3]:
                                self.info(f"  - {alt_name}: {alt_conf:.2f}")

                        protocol_inferences[name] = protocol_result
                    else:
                        self.info(f"Protocol type unknown for {name}")
                        protocol_inferences[name] = protocol_result

                except Exception as e:
                    self.warning(f"Protocol detection failed for {name}: {e}")
                    protocol_inferences[name] = {"likely_protocol": "unknown", "confidence": 0.0}
            else:
                self.info(f"Skipping {name} (not digital protocol)")
                protocol_inferences[name] = None

        results["protocols"] = protocol_inferences

        # ===== Phase 4: Complete Signal Characterization Workflow =====
        self.section("Part 4: Complete Unknown Signal Characterization")

        from oscura.discovery import characterize_signal

        # Demonstrate workflow on a "mystery" signal (UART)
        self.subsection("Mystery Signal Analysis")
        mystery_signal = signals["uart"]

        self.info("Analyzing unknown signal with no prior information...")
        characterization = characterize_signal(
            mystery_signal, confidence_threshold=0.6, include_alternatives=True
        )

        self.info("")
        self.success("Signal Characterization Complete!")
        self.result("Signal type", characterization.signal_type)
        self.result("Confidence", f"{characterization.confidence:.2%}")
        self.result(
            "Voltage range",
            f"{characterization.voltage_low:.2f}V - {characterization.voltage_high:.2f}V",
        )
        self.result("Dominant frequency", f"{characterization.frequency_hz:.3e} Hz")

        if characterization.parameters:
            self.info("Signal parameters:")
            for key, value in characterization.parameters.items():
                self.info(f"  {key}: {value}")

        if characterization.quality_metrics:
            self.info("Quality metrics:")
            for key, value in characterization.quality_metrics.items():
                self.info(f"  {key}: {value:.3f}")

        if characterization.alternatives:
            self.info("Alternative interpretations:")
            for alt_type, alt_conf in characterization.alternatives:
                self.info(f"  {alt_type}: {alt_conf:.2%}")

        results["characterization"] = {
            "signal_type": characterization.signal_type,
            "confidence": characterization.confidence,
            "voltage_low": characterization.voltage_low,
            "voltage_high": characterization.voltage_high,
        }

        # ===== Phase 5: Confidence Scoring Summary =====
        self.section("Part 5: Confidence Scoring Summary")

        self.subsection("Classification Confidence Matrix")
        self.info(f"{'Signal':<15} {'Type':<10} {'Confidence':<12} {'Logic Family':<15}")
        self.info("-" * 60)

        for name in signals:
            sig_type = classifications[name]["signal_type"]
            confidence = classifications[name]["confidence"]
            logic_fam = (
                logic_families[name]["primary"]["name"]
                if logic_families[name] and logic_families[name]["primary"]
                else "N/A"
            )

            self.info(f"{name:<15} {sig_type:<10} {confidence:>6.2f}       {logic_fam:<15}")

        results["summary_created"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate signal classification results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Check that classifications were generated
        if "classifications" not in results or len(results["classifications"]) != 6:
            self.error(f"Expected 6 classifications, got {len(results.get('classifications', {}))}")
            return False

        classifications = results["classifications"]

        # Validate analog signal classification
        if classifications["analog"]["signal_type"] not in ["analog", "dc"]:
            self.error(f"Analog signal misclassified as {classifications['analog']['signal_type']}")
            return False

        # Validate digital signals are detected as digital
        digital_signals = ["ttl", "cmos_3v3", "lvcmos_1v8", "uart"]
        for sig_name in digital_signals:
            if not classifications[sig_name]["is_digital"]:
                self.warning(f"{sig_name} not detected as digital")

        # Validate TTL logic family detection
        logic_families = results.get("logic_families", {})
        if logic_families.get("ttl"):
            ttl_family = logic_families["ttl"]["primary"]["name"]
            if "TTL" not in ttl_family and "5V" not in ttl_family:
                self.warning(f"TTL signal detected as {ttl_family}")

        # Validate CMOS 3.3V detection
        if logic_families.get("cmos_3v3"):
            cmos_family = logic_families["cmos_3v3"]["primary"]["name"]
            if "3" not in cmos_family:
                self.warning(f"CMOS 3.3V signal detected as {cmos_family}")

        # Validate LVCMOS 1.8V detection
        if logic_families.get("lvcmos_1v8"):
            lvcmos_family = logic_families["lvcmos_1v8"]["primary"]["name"]
            if "1V8" not in lvcmos_family and "1.8" not in lvcmos_family:
                self.warning(f"LVCMOS 1.8V signal detected as {lvcmos_family}")

        # Validate protocol inference on UART
        protocols = results.get("protocols", {})
        if protocols.get("uart"):
            uart_protocol = protocols["uart"]["likely_protocol"]
            # UART might be detected or unknown depending on implementation
            if uart_protocol == "uart":
                self.success("UART protocol correctly identified!")
            else:
                self.info(f"UART detected as: {uart_protocol} (protocol inference is heuristic)")

        # Validate characterization workflow completed
        if "characterization" not in results:
            self.error("Complete characterization workflow not executed")
            return False

        char_result = results["characterization"]
        if char_result["confidence"] < 0.3:
            self.warning(f"Low confidence in characterization: {char_result['confidence']:.2f}")

        # Validate confidence scores are reasonable
        for name, classification in classifications.items():
            if classification["confidence"] < 0.3:
                self.warning(f"Low confidence for {name}: {classification['confidence']:.2f}")
            if classification["confidence"] > 1.0:
                self.error(f"Invalid confidence for {name}: {classification['confidence']}")
                return False

        self.success("Signal classification and logic family detection successful!")
        return True


if __name__ == "__main__":
    demo = SignalClassificationDemo()
    success = demo.execute()
    exit(0 if success else 1)
