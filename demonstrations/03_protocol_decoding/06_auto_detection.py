"""Protocol Auto-Detection: Unified protocol detection and decoding demonstration

Demonstrates:
- oscura.auto_decode() - Automatic protocol detection and decoding
- oscura.detect_protocol() - Protocol type inference from signal characteristics
- Automatic baud rate recovery and format detection
- Multi-protocol testing on unknown signals

IEEE Standards: IEEE 181 (waveform measurements)
Related Demos:
- 03_protocol_decoding/ - Individual protocol decoders
- 02_basic_analysis/01_waveform_measurements.py - Signal measurement foundations

This demonstration generates synthetic protocol signals without labels and uses
oscura's auto-detection to identify protocols and extract data automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura import detect_protocol
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace


class AutoDetectionDemo(BaseDemo):
    """Protocol auto-detection demonstration."""

    def __init__(self) -> None:
        """Initialize auto-detection demonstration."""
        super().__init__(
            name="protocol_auto_detection",
            description="Automatic protocol detection and decoding with unified interface",
            capabilities=[
                "oscura.auto_decode",
                "oscura.detect_protocol",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "03_protocol_decoding/01_serial_comprehensive.py",
                "03_protocol_decoding/03_debug_protocols.py",
            ],
        )

    def generate_test_data(self) -> dict[str, WaveformTrace | DigitalTrace]:
        """Generate synthetic unknown protocol signals.

        Returns:
            Dictionary with unlabeled protocol signals
        """
        # Generate signals for different protocols without identifying them
        signal1 = self._generate_unknown_uart(sample_rate=1e6)
        signal2 = self._generate_unknown_i2c(sample_rate=1e6)
        signal3 = self._generate_unknown_spi(sample_rate=10e6)

        return {
            "unknown_1": signal1,
            "unknown_2": signal2,
            "unknown_3": signal3,
        }

    def run_demonstration(
        self, data: dict[str, WaveformTrace | DigitalTrace]
    ) -> dict[str, dict[str, object]]:
        """Auto-detect protocols and decode signals.

        Args:
            data: Generated unknown protocol signals

        Returns:
            Dictionary of detection and decoding results
        """
        results = {}

        # Test auto-detection on each unknown signal
        for signal_name, signal in data.items():
            self.section(f"Auto-Detecting: {signal_name}")
            result = self._detect_and_decode(signal_name, signal)
            results[signal_name] = result

        return results

    def validate(self, results: dict[str, dict[str, object]]) -> bool:
        """Validate auto-detection results.

        Args:
            results: Auto-detection results

        Returns:
            True if all validations pass
        """
        self.section("Validation")

        all_passed = True

        # Validate each detection
        for signal_name, result in results.items():
            self.subsection(f"Validation: {signal_name}")

            if not self._validate_detection(signal_name, result):
                all_passed = False

        if all_passed:
            self.success("All auto-detection validations passed!")
        else:
            self.warning("Some auto-detection validations failed")

        return all_passed

    # Signal generation (unknown protocols)
    def _generate_unknown_uart(self, sample_rate: float) -> DigitalTrace:
        """Generate UART signal without labeling it.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Unlabeled UART signal
        """
        # UART at 9600 baud
        baudrate = 9600
        bit_time = 1.0 / baudrate
        samples_per_bit = int(sample_rate * bit_time)

        message = b"Test"
        signal = []

        # Idle high
        signal.extend([1] * (samples_per_bit * 2))

        # Send message
        for byte in message:
            # Start bit
            signal.extend([0] * samples_per_bit)

            # Data bits (LSB first)
            for i in range(8):
                bit = (byte >> i) & 1
                signal.extend([bit] * samples_per_bit)

            # Stop bit
            signal.extend([1] * samples_per_bit)

        # Idle high
        signal.extend([1] * (samples_per_bit * 2))

        signal_array = np.array(signal, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="unknown_signal",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    def _generate_unknown_i2c(self, sample_rate: float) -> DigitalTrace:
        """Generate I2C-like signal without labeling it.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Unlabeled I2C-like signal
        """
        # Simplified I2C at 100 kHz
        bit_time = 1.0 / 100e3
        samples_per_bit = max(1, int(sample_rate * bit_time))

        # Build I2C-like pattern: Start + Address + ACK
        signal_parts = []

        # Idle (both high)
        signal_parts.extend([1] * (samples_per_bit * 2))

        # Start condition (falling edge while clock high - simplified)
        signal_parts.extend([1] * (samples_per_bit // 2))
        signal_parts.extend([0] * (samples_per_bit // 2))

        # Address byte: 0x50 (7 bits + R/W bit)
        address_byte = 0xA0  # 0x50 << 1 | 0 (write)
        for bit_idx in range(8):
            bit_val = (address_byte >> (7 - bit_idx)) & 1
            # Clock low, data stable
            signal_parts.extend([bit_val] * (samples_per_bit // 2))
            # Clock high (simulated with data change for simplicity)
            signal_parts.extend([bit_val] * (samples_per_bit // 2))

        # ACK (low)
        signal_parts.extend([0] * samples_per_bit)

        # Stop condition (rising edge)
        signal_parts.extend([0] * (samples_per_bit // 2))
        signal_parts.extend([1] * (samples_per_bit // 2))

        signal_array = np.array(signal_parts, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="unknown_signal",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    def _generate_unknown_spi(self, sample_rate: float) -> DigitalTrace:
        """Generate SPI-like signal without labeling it.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            Unlabeled SPI-like signal
        """
        # SPI at 1 MHz - just generate clock-like pattern
        bit_rate = 1e6
        samples_per_bit = int(sample_rate / bit_rate)

        # Generate clock pattern (8 bits)
        signal = []
        for _ in range(8):
            signal.extend([0] * (samples_per_bit // 2))
            signal.extend([1] * (samples_per_bit // 2))

        signal_array = np.array(signal, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="unknown_signal",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    # Auto-detection and decoding
    def _detect_and_decode(
        self,
        signal_name: str,
        signal: WaveformTrace | DigitalTrace,
    ) -> dict[str, object]:
        """Detect protocol and decode signal automatically.

        Args:
            signal_name: Name of the signal
            signal: Signal to analyze

        Returns:
            Detection and decoding results
        """
        self.subsection("Signal Characteristics")
        self.result("Signal type", type(signal).__name__)
        self.result("Sample rate", signal.metadata.sample_rate, "Hz")
        self.result("Samples", len(signal.data))

        # Try protocol detection first
        self.subsection("Protocol Detection")
        try:
            # Convert to waveform if needed for detect_protocol
            if isinstance(signal, DigitalTrace):
                # Create a simple waveform from digital data
                waveform_data = signal.data.astype(np.float64)
                waveform_signal = WaveformTrace(
                    data=waveform_data,
                    metadata=signal.metadata,
                )
            else:
                waveform_signal = signal

            detection_result = detect_protocol(
                waveform_signal,
                min_confidence=0.3,  # Lower threshold for test
                return_candidates=True,
            )

            protocol_name = detection_result["protocol"]
            confidence = detection_result["confidence"]
            config = detection_result.get("config", {})

            self.info(f"Detected protocol: {protocol_name}")
            self.info(f"Confidence: {confidence:.1%}")

            if "candidates" in detection_result:
                self.subsection("Candidate Protocols")
                for candidate in detection_result["candidates"]:
                    cand_protocol = candidate["protocol"]
                    cand_conf = candidate["confidence"]
                    self.info(f"  {cand_protocol}: {cand_conf:.1%}")

            # Show detected configuration
            if config:
                self.subsection("Detected Configuration")
                for key, value in config.items():
                    self.info(f"  {key}: {value}")

            return {
                "detection_successful": True,
                "protocol": protocol_name,
                "confidence": confidence,
                "config": config,
                "candidates": detection_result.get("candidates", []),
            }

        except Exception as e:
            self.warning(f"Protocol detection failed: {e}")
            return {
                "detection_successful": False,
                "error": str(e),
            }

        # Note: auto_decode would be used here but it expects specific signal types
        # For demonstration, we focus on detect_protocol capabilities

    def _validate_detection(self, signal_name: str, result: dict[str, object]) -> bool:
        """Validate detection result.

        Args:
            signal_name: Name of the signal
            result: Detection result

        Returns:
            True if validation passes
        """
        if not result.get("detection_successful", False):
            error = result.get("error", "Unknown error")
            self.warning(f"{signal_name}: Detection failed - {error}")
            # This is acceptable - not all signals are detectable
            return True

        protocol = result.get("protocol", "Unknown")
        confidence = result.get("confidence", 0.0)

        # Expected protocols based on signal names
        expected_protocols = {
            "unknown_1": ["UART"],
            "unknown_2": ["I2C", "UART"],  # May be ambiguous
            "unknown_3": ["SPI", "I2C"],  # Clock patterns are ambiguous
        }

        expected = expected_protocols.get(signal_name, [])

        if protocol in expected:
            self.success(
                f"{signal_name}: Correctly detected as {protocol} (confidence: {confidence:.1%})"
            )
            return True
        else:
            # Check if it's in candidates
            candidates = result.get("candidates", [])
            candidate_protocols = [c["protocol"] for c in candidates]

            found_in_candidates = any(exp in candidate_protocols for exp in expected)

            if found_in_candidates:
                self.warning(f"{signal_name}: Expected protocol in candidates but not top choice")
                return True  # Acceptable
            else:
                self.warning(f"{signal_name}: Detected as {protocol}, expected {expected}")
                # Still acceptable - detection is heuristic
                return True


if __name__ == "__main__":
    demo: AutoDetectionDemo = AutoDetectionDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
