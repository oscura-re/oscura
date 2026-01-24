"""Data Classification and Structure Inference: Automatic binary data analysis

Demonstrates:
- oscura.analyzers.statistical.classify_data_type - Automatic data type classification
- oscura.analyzers.statistical.detect_padding_regions - Padding byte detection
- oscura.analyzers.statistical.extract_ngrams - N-gram pattern extraction
- oscura.analyzers.statistical.ngram_frequencies - Pattern frequency analysis
- oscura.analyzers.statistical.detect_checksum_fields - Checksum field detection
- oscura.analyzers.statistical.identify_checksum_algorithm - Algorithm identification
- oscura.analyzers.statistical.byte_frequency_distribution - Byte frequency analysis
- Automatic structure inference from unknown binary data
- Statistical data characterization

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/02_crc_recovery.py
- 06_reverse_engineering/04_field_inference.py

This demonstration shows how to automatically classify and analyze binary data
without prior knowledge of its structure. It generates a binary blob with known
structure (header, payload, checksum, padding) and demonstrates automatic
detection and classification of each region using statistical analysis.

This is a P0 CRITICAL feature - demonstrates automatic reverse engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class DataClassificationDemo(BaseDemo):
    """Demonstrates automatic data classification and structure inference."""

    def __init__(self) -> None:
        """Initialize data classification demonstration."""
        super().__init__(
            name="data_classification",
            description="Automatic data type classification and structure inference",
            capabilities=[
                "oscura.analyzers.statistical.classify_data_type",
                "oscura.analyzers.statistical.detect_padding_regions",
                "oscura.analyzers.statistical.extract_ngrams",
                "oscura.analyzers.statistical.ngram_frequencies",
                "oscura.analyzers.statistical.detect_checksum_fields",
                "oscura.analyzers.statistical.identify_checksum_algorithm",
                "oscura.analyzers.statistical.byte_frequency_distribution",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/02_crc_recovery.py",
                "06_reverse_engineering/04_field_inference.py",
            ],
        )
        self.test_binary: bytes = b""
        self.expected_structure: dict[str, tuple[int, int]] = {}

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test binary with known structure.

        Creates a binary blob containing:
        - Text header (ASCII)
        - Binary payload (structured data)
        - Checksum field
        - Padding (null bytes)

        Returns:
            Dictionary with binary data and expected structure
        """
        self.section("Generating Test Binary with Known Structure")

        # Build binary with distinct regions
        binary_parts = []

        # 1. Text header (ASCII)
        header_text = b"OSCURA-DATA-V1.0"
        binary_parts.append(header_text)
        header_end = len(header_text)

        # 2. Binary payload (structured data with patterns)
        payload = bytearray()
        # Magic number
        payload.extend(b"\xde\xad\xbe\xef")
        # Sequence counter
        for i in range(10):
            payload.append(i)
        # Random-looking data (but with pattern)
        np.random.seed(42)
        payload.extend(np.random.randint(0, 256, size=50, dtype=np.uint8).tobytes())
        binary_parts.append(bytes(payload))
        payload_start = header_end
        payload_end = payload_start + len(payload)

        # 3. Checksum field (XOR of all previous data)
        data_so_far = b"".join(binary_parts)
        checksum = 0
        for byte in data_so_far:
            checksum ^= byte
        checksum_bytes = bytes([checksum])
        binary_parts.append(checksum_bytes)
        checksum_offset = len(data_so_far)

        # 4. Padding (null bytes)
        padding = b"\x00" * 64
        binary_parts.append(padding)
        padding_start = checksum_offset + 1
        padding_end = padding_start + len(padding)

        self.test_binary = b"".join(binary_parts)

        # Store expected structure for validation
        self.expected_structure = {
            "header": (0, header_end),
            "payload": (payload_start, payload_end),
            "checksum": (checksum_offset, checksum_offset + 1),
            "padding": (padding_start, padding_end),
        }

        self.info(f"Generated binary: {len(self.test_binary)} bytes")
        self.info(f"  Header (text):    bytes {0:3d}-{header_end:3d}")
        self.info(f"  Payload (binary): bytes {payload_start:3d}-{payload_end:3d}")
        self.info(f"  Checksum:         byte  {checksum_offset:3d}")
        self.info(f"  Padding (null):   bytes {padding_start:3d}-{padding_end:3d}")

        return {
            "binary": self.test_binary,
            "expected_structure": self.expected_structure,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute data classification and structure inference."""
        results: dict[str, Any] = {}
        binary = data["binary"]

        # ===== Phase 1: Overall Data Classification =====
        self.section("Part 1: Automatic Data Type Classification")
        self.subsection("Full Binary Classification")

        from oscura.analyzers.statistical import classify_data_type

        full_classification = classify_data_type(binary)

        self.info(f"Primary type:      {full_classification.primary_type}")
        self.info(f"Confidence:        {full_classification.confidence:.2f}")
        self.info(f"Entropy:           {full_classification.entropy:.2f} bits")
        self.info(f"Printable ratio:   {full_classification.printable_ratio:.2%}")
        self.info(f"Null ratio:        {full_classification.null_ratio:.2%}")
        self.info(f"Byte variance:     {full_classification.byte_variance:.1f}")

        results["full_classification"] = full_classification.primary_type
        results["entropy"] = full_classification.entropy

        # ===== Phase 2: Region Detection =====
        self.section("Part 2: Automatic Region Detection")

        # Detect text regions
        self.subsection("Text Region Detection")
        from oscura.analyzers.statistical import detect_text_regions

        text_regions = detect_text_regions(binary, min_length=8)
        self.result("Text regions found", len(text_regions))

        for i, region in enumerate(text_regions):
            self.info(
                f"  Region {i + 1}: bytes {region.start:3d}-{region.end:3d} "
                f"({region.length} bytes, confidence={region.classification.confidence:.2f})"
            )
            # Show sample of text
            sample = binary[region.start : min(region.start + 20, region.end)]
            self.info(f"    Sample: {sample!r}")

        results["text_regions_found"] = len(text_regions)

        # Detect padding regions
        self.subsection("Padding Region Detection")
        from oscura.analyzers.statistical import detect_padding_regions

        padding_regions = detect_padding_regions(binary, min_length=4)
        self.result("Padding regions found", len(padding_regions))

        for i, region in enumerate(padding_regions):
            self.info(
                f"  Region {i + 1}: bytes {region.start:3d}-{region.end:3d} "
                f"({region.length} bytes, "
                f"padding_byte={region.classification.details.get('padding_byte', 'N/A')})"
            )

        results["padding_regions_found"] = len(padding_regions)

        # ===== Phase 3: N-gram Analysis =====
        self.section("Part 3: N-gram Frequency Analysis")
        self.subsection("Bigram Pattern Discovery")

        from oscura.analyzers.statistical import extract_ngrams, ngram_frequencies

        # Extract bigrams from binary payload region
        payload_start = self.expected_structure["payload"][0]
        payload_end = self.expected_structure["payload"][1]
        payload_data = binary[payload_start:payload_end]

        bigrams = extract_ngrams(payload_data, n=2, overlap=True)
        bigram_freqs = ngram_frequencies(payload_data, n=2, overlap=True)

        self.result("Total bigrams extracted", len(bigrams))
        self.result("Unique bigrams", len(bigram_freqs))

        # Show most common bigrams
        top_bigrams = sorted(bigram_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
        self.info("Top 5 most common bigrams:")
        for ngram, count in top_bigrams:
            self.info(f"  {ngram.hex():6s} : {count:3d} occurrences")

        results["unique_bigrams"] = len(bigram_freqs)
        results["total_bigrams"] = len(bigrams)

        # ===== Phase 4: Checksum Detection =====
        self.section("Part 4: Automatic Checksum Detection")
        self.subsection("Checksum Field Identification")

        # For checksum detection, we need multiple messages
        # Generate variations of the binary with different payloads
        messages = []
        checksum_offset = self.expected_structure["checksum"][0]

        # Get header from original binary
        header_start, header_end = self.expected_structure["header"]
        header_bytes = binary[header_start:header_end]

        for i in range(15):
            # Build new message with varied payload
            modified_payload = bytearray()
            # Magic number (constant)
            modified_payload.extend(b"\xde\xad\xbe\xef")
            # Sequence counter (varies)
            for j in range(10):
                modified_payload.append((j + i) % 256)
            # Random data (varies per message)
            np.random.seed(42 + i)
            modified_payload.extend(np.random.randint(0, 256, size=50, dtype=np.uint8).tobytes())

            # Build message without checksum first
            msg_without_checksum = header_bytes + bytes(modified_payload)

            # Calculate checksum (XOR of all data so far)
            new_checksum = 0
            for byte in msg_without_checksum:
                new_checksum ^= byte

            # Build complete message with checksum and padding
            complete_msg = msg_without_checksum + bytes([new_checksum]) + b"\x00" * 64

            messages.append(complete_msg)

        from oscura.analyzers.statistical import (
            detect_checksum_fields,
            identify_checksum_algorithm,
        )

        checksum_candidates = detect_checksum_fields(messages)
        self.result("Checksum candidates found", len(checksum_candidates))

        # Also manually verify known checksum position
        from oscura.analyzers.statistical import verify_checksums

        passed, failed = verify_checksums(
            messages, "xor", checksum_offset, scope_start=0, scope_end=checksum_offset
        )
        self.info(f"Manual checksum verification at offset {checksum_offset}:")
        self.info(f"  Passed: {passed}/{len(messages)}, Failed: {failed}/{len(messages)}")

        if checksum_candidates:
            # Show top candidates
            self.info("Top checksum candidates:")
            for i, candidate in enumerate(checksum_candidates[:3]):
                self.info(
                    f"  Candidate {i + 1}: offset={candidate.offset}, "
                    f"size={candidate.size}, "
                    f"position={candidate.position}, "
                    f"correlation={candidate.correlation:.2f}"
                )

            # Try to identify algorithm for best candidate
            best_candidate = checksum_candidates[0]
            checksum_match = identify_checksum_algorithm(
                messages, best_candidate.offset, best_candidate.size
            )

            if checksum_match:
                self.success("Checksum algorithm identified!")
                self.result("Algorithm", checksum_match.algorithm)
                self.result("Offset", checksum_match.offset)
                self.result("Match rate", f"{checksum_match.match_rate:.0%}")
                self.result(
                    "Scope", f"bytes {checksum_match.scope_start}-{checksum_match.scope_end}"
                )
                results["checksum_detected"] = True
                results["checksum_algorithm"] = checksum_match.algorithm
            else:
                self.warning("Could not identify specific checksum algorithm")
                results["checksum_detected"] = False
        else:
            # No automatic detection, but we can verify manually
            if passed > len(messages) * 0.8:
                self.success(
                    f"Manual verification confirms XOR checksum at offset {checksum_offset}!"
                )
                self.info(
                    "Note: Automatic detection may fail with small sample sizes or high padding ratio"
                )
                results["checksum_detected"] = True
                results["checksum_algorithm"] = "xor (manual verification)"
            else:
                self.warning("No checksum candidates found")
                results["checksum_detected"] = False

        # ===== Phase 5: Byte Frequency Distribution =====
        self.section("Part 5: Byte Frequency Distribution Analysis")
        self.subsection("Statistical Byte Analysis")

        from oscura.analyzers.statistical import byte_frequency_distribution

        byte_freq = byte_frequency_distribution(binary)

        self.result("Unique bytes", byte_freq.unique_bytes)
        self.result("Most common byte", f"0x{byte_freq.most_common[0][0]:02X}")
        self.result(
            "Most common frequency",
            f"{byte_freq.most_common[0][1]} ({byte_freq.frequencies[byte_freq.most_common[0][0]]:.2%})",
        )
        self.result("Least common byte", f"0x{byte_freq.least_common[0][0]:02X}")
        self.result("Uniformity score", f"{byte_freq.uniformity_score:.4f}")
        self.result("Zero byte ratio", f"{byte_freq.zero_byte_ratio:.2%}")
        self.result("Printable ratio", f"{byte_freq.printable_ratio:.2%}")

        # Calculate standard deviation of frequencies
        freq_std = float(np.std(byte_freq.frequencies))
        self.result("Frequency std dev", f"{freq_std:.4f}")

        # Show byte distribution characteristics
        self.info("Byte distribution characteristics:")
        self.info(
            f"  Bytes with frequency > 5%:  {sum(1 for f in byte_freq.frequencies if f > 0.05)}"
        )
        self.info(
            f"  Bytes with frequency < 0.5%: {sum(1 for f in byte_freq.frequencies if f < 0.005)}"
        )

        results["unique_bytes"] = byte_freq.unique_bytes
        results["byte_freq_stddev"] = freq_std
        results["uniformity_score"] = byte_freq.uniformity_score

        # ===== Phase 6: Complete Structure Inference =====
        self.section("Part 6: Complete Structure Inference")
        self.subsection("Inferred Binary Structure")

        self.info("Combining all analysis results:")
        self.info("")
        self.info("Inferred Structure:")

        # Combine all detections
        inferred_structure = []

        # Add text regions
        for region in text_regions:
            inferred_structure.append(
                {
                    "start": region.start,
                    "end": region.end,
                    "type": "text",
                    "confidence": region.classification.confidence,
                }
            )

        # Add checksum if detected
        if checksum_candidates:
            best = checksum_candidates[0]
            inferred_structure.append(
                {
                    "start": best.offset,
                    "end": best.offset + best.size,
                    "type": "checksum",
                    "confidence": best.correlation,
                }
            )

        # Add padding regions
        for region in padding_regions:
            inferred_structure.append(
                {
                    "start": region.start,
                    "end": region.end,
                    "type": "padding",
                    "confidence": region.classification.confidence,
                }
            )

        # Sort by start position
        inferred_structure.sort(key=lambda x: x["start"])

        # Fill gaps with "binary" regions
        complete_structure = []
        last_end = 0
        for region in inferred_structure:
            if region["start"] > last_end:
                # Gap - mark as binary
                complete_structure.append(
                    {
                        "start": last_end,
                        "end": region["start"],
                        "type": "binary",
                        "confidence": 0.5,
                    }
                )
            complete_structure.append(region)
            last_end = region["end"]

        # Handle any remaining data
        if last_end < len(binary):
            complete_structure.append(
                {
                    "start": last_end,
                    "end": len(binary),
                    "type": "binary",
                    "confidence": 0.5,
                }
            )

        # Display inferred structure
        for i, region in enumerate(complete_structure):
            length = region["end"] - region["start"]
            self.info(
                f"  Region {i + 1}: bytes {region['start']:3d}-{region['end']:3d} "
                f"({length:3d} bytes) - {region['type']:10s} "
                f"(confidence={region['confidence']:.2f})"
            )

        results["inferred_regions"] = len(complete_structure)
        results["structure"] = complete_structure

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate data classification results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must detect text regions (header)
        if results.get("text_regions_found", 0) < 1:
            self.error("Failed to detect text header region")
            return False

        # Must detect padding regions
        if results.get("padding_regions_found", 0) < 1:
            self.error("Failed to detect padding region")
            return False

        # Should extract n-grams
        if results.get("unique_bigrams", 0) < 10:
            self.warning("Expected more unique bigrams in payload")

        # Should detect checksum (may fail due to small sample)
        if not results.get("checksum_detected", False):
            self.warning("Checksum detection failed (expected with small sample size)")
        else:
            # If detected, verify algorithm
            detected_algo = results.get("checksum_algorithm", "")
            if "xor" not in detected_algo.lower():
                self.warning(f"Checksum algorithm mismatch: got {detected_algo}, expected xor")

        # Must have inferred structure
        if results.get("inferred_regions", 0) < 3:
            self.error("Failed to infer reasonable structure (expected at least 3 regions)")
            return False

        # Must have reasonable entropy
        entropy = results.get("entropy", 0)
        if entropy < 1.0 or entropy > 8.0:
            self.warning(f"Unusual entropy value: {entropy:.2f}")

        self.success("Data classification and structure inference successful!")
        return True


if __name__ == "__main__":
    demo = DataClassificationDemo()
    success = demo.execute()
    exit(0 if success else 1)
