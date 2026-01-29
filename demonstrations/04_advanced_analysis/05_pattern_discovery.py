"""Pattern Discovery: Automatic pattern recognition in signals

Demonstrates:
- oscura.patterns.discover_signatures() - Automatic header/delimiter discovery
- oscura.patterns.find_periodic_patterns() - Repetition detection
- oscura.patterns.extract_sequences() - Sequence identification
- oscura.patterns.correlation_analysis() - Pattern correlation
- State extraction - Digital state machine inference

Related Demos:
- 03_protocol_decoding/01_uart_decoding.py
- 03_protocol_decoding/02_spi_decoding.py

Uses synthetic signals with embedded patterns for discovery demonstration.
Perfect for reverse engineering unknown protocols and signal patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.analyzers.patterns.discovery import CandidateSignature, SignatureDiscovery


class PatternDiscoveryDemo(BaseDemo):
    """Comprehensive demonstration of pattern discovery techniques."""

    def __init__(self) -> None:
        """Initialize pattern discovery demonstration."""
        super().__init__(
            name="pattern_discovery",
            description="Automatic pattern recognition: signatures, headers, repetitions, sequences",
            capabilities=[
                "oscura.patterns.discover_signatures",
                "oscura.patterns.find_periodic_patterns",
                "oscura.patterns.extract_sequences",
                "oscura.patterns.correlation_analysis",
            ],
            related_demos=[
                "03_protocol_decoding/01_uart_decoding.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate signals with embedded patterns.

        Creates:
        1. Packet stream: Periodic headers + varying payloads
        2. Counter signal: Incrementing counter pattern
        3. Toggle sequence: Specific toggle patterns
        4. Mixed patterns: Combined periodic and aperiodic patterns
        """
        # 1. Packet stream with headers
        # Header: 0xAA55 (sync bytes), followed by length, data, CRC
        packets = []
        header = bytes([0xAA, 0x55])  # Sync pattern

        for _i in range(20):
            # Variable length packets (16-64 bytes)
            payload_len = np.random.randint(16, 64)
            payload = np.random.randint(0, 256, payload_len, dtype=np.uint8).tobytes()

            # Simple CRC (XOR of all bytes)
            crc = 0
            for b in payload:
                crc ^= b

            packet = header + bytes([payload_len]) + payload + bytes([crc])
            packets.append(packet)

        packet_stream = b"".join(packets)

        # 2. Counter signal (8-bit incrementing)
        counter_data = bytes(np.arange(0, 256, dtype=np.uint8)) * 10  # 10 cycles

        # 3. Toggle sequence (specific pattern: 0x01, 0x02, 0x04, 0x08, 0x10...)
        toggle_pattern = [1 << i for i in range(8)] * 20
        toggle_data = bytes(toggle_pattern)

        # 4. Mixed patterns
        # Combine: header + counter + random + toggle
        mixed_data = b""
        for _ in range(10):
            mixed_data += header
            mixed_data += bytes(np.arange(0, 32, dtype=np.uint8))
            mixed_data += bytes(np.random.randint(0, 256, 16, dtype=np.uint8))
            mixed_data += bytes([1, 2, 4, 8, 16, 32, 64, 128])

        return {
            "packet_stream": packet_stream,
            "counter_data": counter_data,
            "toggle_data": toggle_data,
            "mixed_data": mixed_data,
            "header_pattern": header,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run comprehensive pattern discovery demonstration."""
        results = {}

        self.section("Oscura Pattern Discovery")
        self.info("Demonstrating automatic pattern recognition and discovery")
        self.info("Using synthetic data with embedded patterns")

        # ========== PART 1: SIGNATURE DISCOVERY IN PACKET STREAM ==========
        self.subsection("Part 1: Signature Discovery in Packet Stream")
        packet_stream = data["packet_stream"]
        self.info(f"Packet stream: {len(packet_stream)} bytes with 0xAA55 headers")

        # Discover signatures
        discoverer = SignatureDiscovery(
            min_length=2,
            max_length=8,
            min_occurrences=3,
        )

        signatures: list[CandidateSignature] = discoverer.discover_signatures(packet_stream)

        self.result("Signatures found", len(signatures))

        # Find the header pattern (0xAA55)
        header_found = False
        for sig in signatures[:5]:  # Show top 5
            pattern_hex = sig.pattern.hex().upper()
            self.info(
                f"  Pattern: 0x{pattern_hex}, "
                f"Length: {sig.length}, "
                f"Occurrences: {sig.occurrences}, "
                f"Score: {sig.score:.3f}"
            )

            if sig.pattern == data["header_pattern"]:
                header_found = True
                self.success(f"Header pattern 0xAA55 discovered! (score: {sig.score:.3f})")
                results["header_found"] = True
                results["header_occurrences"] = sig.occurrences
                results["header_score"] = sig.score

        if not header_found:
            self.warning("Header pattern 0xAA55 not in top signatures")
            results["header_found"] = False

        # ========== PART 2: COUNTER PATTERN DETECTION ==========
        self.subsection("Part 2: Counter Pattern Detection")
        counter_data = data["counter_data"]
        self.info(f"Counter data: {len(counter_data)} bytes (0x00 to 0xFF incrementing)")

        # Detect incrementing sequences
        counter_sequences = self._find_incrementing_sequences(counter_data)

        self.result("Incrementing sequences found", len(counter_sequences))

        if len(counter_sequences) > 0:
            longest = max(counter_sequences, key=lambda x: x[1])
            self.result("Longest sequence start", f"0x{longest[0]:04X}")
            self.result("Longest sequence length", longest[1])
            results["counter_sequences"] = len(counter_sequences)
            results["longest_counter"] = longest[1]
        else:
            results["counter_sequences"] = 0

        # ========== PART 3: TOGGLE SEQUENCE DETECTION ==========
        self.subsection("Part 3: Toggle Sequence Detection")
        toggle_data = data["toggle_data"]
        self.info(f"Toggle data: {len(toggle_data)} bytes (power-of-2 pattern)")

        # Detect power-of-2 sequences
        toggle_sequences = self._find_power_of_2_sequences(toggle_data)

        self.result("Power-of-2 sequences found", len(toggle_sequences))

        if len(toggle_sequences) > 0:
            longest = max(toggle_sequences, key=lambda x: x[1])
            self.result("Longest toggle sequence start", f"0x{longest[0]:04X}")
            self.result("Longest toggle sequence length", longest[1])
            results["toggle_sequences"] = len(toggle_sequences)
            results["longest_toggle"] = longest[1]
        else:
            results["toggle_sequences"] = 0

        # ========== PART 4: MIXED PATTERN ANALYSIS ==========
        self.subsection("Part 4: Mixed Pattern Analysis")
        mixed_data = data["mixed_data"]
        self.info(f"Mixed data: {len(mixed_data)} bytes with multiple pattern types")

        # Discover all patterns
        mixed_sigs: list[CandidateSignature] = discoverer.discover_signatures(mixed_data)

        self.result("Total signatures found", len(mixed_sigs))

        # Analyze pattern types
        header_patterns = [s for s in mixed_sigs if s.pattern == data["header_pattern"]]
        if header_patterns:
            self.success(f"Header pattern found: {header_patterns[0].occurrences} occurrences")
            results["mixed_header_found"] = True

        # Find periodic patterns (those with consistent intervals)
        periodic = [s for s in mixed_sigs if s.interval_std < 0.1 * s.interval_mean]
        self.result("Periodic patterns", len(periodic))

        results["mixed_signatures"] = len(mixed_sigs)
        results["periodic_patterns"] = len(periodic)

        # ========== PART 5: CORRELATION ANALYSIS ==========
        self.subsection("Part 5: Correlation Analysis")
        self.info("Analyzing pattern correlation and repetition")

        # Analyze autocorrelation to find repetition periods
        if len(counter_data) > 256:
            sample_data = np.frombuffer(counter_data[:512], dtype=np.uint8)
            autocorr = self._autocorrelation(sample_data, max_lag=256)

            # Find peaks in autocorrelation (repetition periods)
            peaks = self._find_peaks(autocorr[1:], threshold=0.5)  # Skip lag=0

            if len(peaks) > 0:
                self.result("Repetition periods detected", len(peaks))
                self.result("Primary period", peaks[0] + 1, "bytes")
                results["primary_period"] = int(peaks[0] + 1)
            else:
                self.info("No strong repetition periods detected")
                results["primary_period"] = 0

        # ========== PATTERN INTERPRETATION ==========
        self.subsection("Pattern Discovery Interpretation")

        self.info("\n[Signature Discovery]")
        self.info("  Finds repeating byte sequences (headers, delimiters)")
        self.info("  Useful for protocol reverse engineering")
        self.info("  High score = distinctive, regularly spaced pattern")

        self.info("\n[Sequence Detection]")
        self.info("  Identifies structured patterns (counters, toggles)")
        self.info("  Useful for finding state machines and counters")

        self.info("\n[Correlation Analysis]")
        self.info("  Finds repetition periods and periodicities")
        self.info("  Useful for packet boundary detection")

        self.info("\n[Applications:]")
        self.info("  - Unknown protocol reverse engineering")
        self.info("  - Packet boundary detection")
        self.info("  - State machine extraction")
        self.info("  - CRC/checksum field identification")

        self.success("All pattern discovery demonstrations complete!")

        return results

    def _find_incrementing_sequences(
        self, data: bytes, min_length: int = 8
    ) -> list[tuple[int, int]]:
        """Find incrementing byte sequences.

        Args:
            data: Input byte data
            min_length: Minimum sequence length

        Returns:
            List of (start_offset, length) tuples
        """
        sequences = []
        i = 0

        while i < len(data) - min_length:
            # Check if incrementing
            length = 1
            while i + length < len(data):
                expected = (data[i] + length) & 0xFF
                if data[i + length] == expected:
                    length += 1
                else:
                    break

            if length >= min_length:
                sequences.append((i, length))
                i += length
            else:
                i += 1

        return sequences

    def _find_power_of_2_sequences(self, data: bytes, min_length: int = 4) -> list[tuple[int, int]]:
        """Find power-of-2 sequences (bit shifting patterns).

        Args:
            data: Input byte data
            min_length: Minimum sequence length

        Returns:
            List of (start_offset, length) tuples
        """
        sequences = []
        i = 0

        while i < len(data) - min_length:
            # Check if power of 2 sequence
            if data[i] > 0 and (data[i] & (data[i] - 1)) == 0:
                # First byte is power of 2
                length = 1
                current_val = data[i]

                while i + length < len(data):
                    # Check for left shift
                    next_val = (current_val << 1) & 0xFF
                    if next_val == 0:
                        next_val = 1  # Wrap around

                    if data[i + length] == next_val:
                        length += 1
                        current_val = next_val
                    else:
                        break

                if length >= min_length:
                    sequences.append((i, length))
                    i += length
                    continue

            i += 1

        return sequences

    def _autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation of data.

        Args:
            data: Input data array
            max_lag: Maximum lag to compute

        Returns:
            Autocorrelation values for lags 0 to max_lag
        """
        n = len(data)
        data_centered = data - np.mean(data)
        variance = np.var(data)

        if variance == 0:
            return np.zeros(max_lag + 1)

        autocorr = np.zeros(max_lag + 1)

        for lag in range(max_lag + 1):
            if lag >= n:
                break
            autocorr[lag] = np.sum(data_centered[: n - lag] * data_centered[lag:]) / (n * variance)

        return autocorr

    def _find_peaks(self, data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Find peaks in data above threshold.

        Args:
            data: Input data
            threshold: Minimum peak height

        Returns:
            Array of peak indices
        """
        peaks = []

        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(i)

        return np.array(peaks, dtype=np.int64)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate pattern discovery results."""
        self.info("Validating pattern discovery results...")

        all_valid = True

        # Validate signature discovery
        self.subsection("Signature Discovery Validation")

        # Header should be found
        if results.get("header_found", False):
            self.success("Header pattern 0xAA55 successfully discovered")
        else:
            self.error("Header pattern 0xAA55 not found")
            all_valid = False

        # Header should have reasonable score (> 0.5)
        if results.get("header_score", 0) > 0.5:
            self.success(f"Header score: {results['header_score']:.3f} > 0.5")
        else:
            self.warning(f"Header score low: {results.get('header_score', 0):.3f}")

        # Validate counter detection
        self.subsection("Counter Detection Validation")

        # Should find counter sequences
        if results.get("counter_sequences", 0) > 0:
            self.success(f"Found {results['counter_sequences']} counter sequences")
        else:
            self.warning("No counter sequences found")

        # Longest counter should be significant
        if results.get("longest_counter", 0) >= 128:
            self.success(f"Longest counter: {results['longest_counter']} bytes >= 128")
        else:
            self.info(f"Longest counter: {results.get('longest_counter', 0)} bytes")

        # Validate toggle detection
        self.subsection("Toggle Detection Validation")

        # Should find toggle sequences
        if results.get("toggle_sequences", 0) > 0:
            self.success(f"Found {results['toggle_sequences']} toggle sequences")
        else:
            self.warning("No toggle sequences found")

        # Validate mixed pattern analysis
        self.subsection("Mixed Pattern Analysis Validation")

        # Should find multiple signatures in mixed data
        if results.get("mixed_signatures", 0) > 5:
            self.success(f"Found {results['mixed_signatures']} signatures in mixed data")
        else:
            self.info(f"Mixed signatures: {results.get('mixed_signatures', 0)}")

        # Should detect periodic patterns
        if results.get("periodic_patterns", 0) > 0:
            self.success(f"Found {results['periodic_patterns']} periodic patterns")
        else:
            self.info("No periodic patterns detected")

        # Validate correlation analysis
        self.subsection("Correlation Analysis Validation")

        # Should detect primary period (256 for counter)
        if results.get("primary_period", 0) > 0:
            self.success(f"Primary period detected: {results['primary_period']} bytes")
            if 200 < results["primary_period"] < 300:
                self.success("Period close to expected 256 bytes")
        else:
            self.info("No primary period detected")

        if all_valid:
            self.success("All pattern discovery validations passed!")
            self.info("\nKey takeaways:")
            self.info("  - Signature discovery finds repeating headers/delimiters")
            self.info("  - Sequence detection identifies structured patterns")
            self.info("  - Correlation analysis reveals periodicities")
            self.info("  - Useful for protocol reverse engineering")
            self.info("\nNext steps:")
            self.info("  - Apply to real protocol captures")
            self.info("  - Combine with statistical analysis")
            self.info("  - Try 03_protocol_decoding demos for known protocols")
        else:
            self.error("Some pattern discovery validations failed")

        return all_valid


if __name__ == "__main__":
    demo: PatternDiscoveryDemo = PatternDiscoveryDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
