"""Entropy Analysis for Data Segmentation and Protocol Reverse Engineering

Demonstrates:
- oscura.analyzers.statistical.shannon_entropy() - Shannon entropy calculation
- oscura.analyzers.statistical.sliding_entropy() - Sliding window entropy profiles
- oscura.analyzers.statistical.detect_entropy_transitions() - Boundary detection
- oscura.analyzers.statistical.classify_by_entropy() - Data type classification
- oscura.analyzers.statistical.classification.detect_encrypted_regions()
- oscura.analyzers.statistical.classification.detect_compressed_regions()
- Entropy-based protocol segmentation
- Encrypted vs compressed vs plaintext detection

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/04_field_inference.py
- 06_reverse_engineering/05_pattern_discovery.py

This demonstration shows how Shannon entropy analysis enables automatic
segmentation of unknown binary protocols into distinct regions (plaintext,
compressed, encrypted). Entropy transitions identify field boundaries and
protocol structure without prior knowledge.

This is a CRITICAL feature for reverse engineering - entropy is the primary
tool for understanding unknown binary data structure.
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class EntropyAnalysisDemo(BaseDemo):
    """Demonstrates entropy analysis for protocol reverse engineering."""

    def __init__(self) -> None:
        """Initialize entropy analysis demonstration."""
        super().__init__(
            name="entropy_analysis",
            description="Shannon entropy analysis for data segmentation and classification",
            capabilities=[
                "oscura.analyzers.statistical.shannon_entropy",
                "oscura.analyzers.statistical.sliding_entropy",
                "oscura.analyzers.statistical.detect_entropy_transitions",
                "oscura.analyzers.statistical.classify_by_entropy",
                "oscura.analyzers.statistical.classification.detect_encrypted_regions",
                "oscura.analyzers.statistical.classification.detect_compressed_regions",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/04_field_inference.py",
                "06_reverse_engineering/05_pattern_discovery.py",
            ],
        )
        self.test_data: bytes = b""

    def generate_test_data(self) -> dict[str, Any]:
        """Generate mixed data with different entropy regions.

        Creates:
        - Plaintext region (low entropy, ~4-5 bits/byte)
        - Compressed region (medium entropy, ~6-7 bits/byte)
        - Encrypted region (high entropy, >7.5 bits/byte)
        - Structured binary (medium-low entropy)

        Returns:
            Dictionary with test data and ground truth
        """
        self.section("Generating Mixed Entropy Test Data")

        # Region 1: Plaintext ASCII (low entropy)
        plaintext = b"Hello, this is a plaintext message. " * 20
        plaintext = plaintext[:256]  # Fixed size for consistency

        # Region 2: Structured binary protocol (medium-low entropy)
        # Simulates a protocol with magic, counters, types
        structured = bytearray()
        for i in range(64):
            structured.extend([0xDE, 0xAD])  # Magic (constant)
            structured.append(i & 0xFF)  # Counter
            structured.append(i % 5)  # Type (enum with 5 values)

        # Region 3: Compressed data (medium-high entropy)
        # Compress random-looking but compressible data
        compressible = b"ABCD" * 100 + b"1234" * 100 + b"XYZW" * 100
        compressed = gzip.compress(compressible)[:512]  # Take first 512 bytes

        # Region 4: Encrypted/Random data (high entropy)
        # Generate truly random data - larger size for more reliable entropy
        encrypted = os.urandom(512)

        # Combine all regions with padding for clear transitions
        padding = b"\x00" * 32
        self.test_data = (
            plaintext + padding + structured + padding + compressed + padding + encrypted
        )

        # Store ground truth boundaries
        ground_truth = {
            "plaintext": (0, len(plaintext)),
            "padding1": (len(plaintext), len(plaintext) + 32),
            "structured": (len(plaintext) + 32, len(plaintext) + 32 + len(structured)),
            "padding2": (
                len(plaintext) + 32 + len(structured),
                len(plaintext) + 64 + len(structured),
            ),
            "compressed": (
                len(plaintext) + 64 + len(structured),
                len(plaintext) + 64 + len(structured) + len(compressed),
            ),
            "padding3": (
                len(plaintext) + 64 + len(structured) + len(compressed),
                len(plaintext) + 96 + len(structured) + len(compressed),
            ),
            "encrypted": (
                len(plaintext) + 96 + len(structured) + len(compressed),
                len(self.test_data),
            ),
        }

        self.info(f"Total data size: {len(self.test_data)} bytes")
        self.info("Region breakdown:")
        for name, (start, end) in ground_truth.items():
            self.info(f"  {name}: bytes {start}-{end} ({end - start} bytes)")

        return {
            "data": self.test_data,
            "ground_truth": ground_truth,
            "plaintext": plaintext,
            "structured": structured,
            "compressed": compressed,
            "encrypted": encrypted,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute entropy analysis demonstration."""
        from oscura.analyzers.statistical import (
            classify_by_entropy,
            detect_entropy_transitions,
            shannon_entropy,
            sliding_entropy,
        )
        from oscura.analyzers.statistical.classification import (
            detect_compressed_regions,
            detect_encrypted_regions,
        )

        results: dict[str, Any] = {}
        test_data = data["data"]
        ground_truth = data["ground_truth"]

        # ===== Part 1: Basic Entropy Calculation =====
        self.section("Part 1: Shannon Entropy Calculation")
        self.subsection("Per-Region Entropy Analysis")

        region_entropies = {}
        for region_name in ["plaintext", "structured", "compressed", "encrypted"]:
            region_data = data[region_name]
            entropy = shannon_entropy(region_data)
            region_entropies[region_name] = entropy
            self.result(f"{region_name} entropy", f"{entropy:.3f}", "bits/byte")

        results["region_entropies"] = region_entropies

        # Validate entropy ranges
        self.subsection("Entropy Validation")
        plaintext_entropy = region_entropies["plaintext"]
        compressed_entropy = region_entropies["compressed"]
        encrypted_entropy = region_entropies["encrypted"]

        # Plaintext should be low entropy
        if plaintext_entropy < 5.0:
            self.success(f"Plaintext has low entropy ({plaintext_entropy:.2f} < 5.0)")
        else:
            self.warning(f"Plaintext entropy unexpectedly high: {plaintext_entropy:.2f}")

        # Compressed should be medium-high
        if 5.0 <= compressed_entropy <= 7.5:
            self.success(
                f"Compressed has medium-high entropy ({compressed_entropy:.2f} in 5.0-7.5)"
            )
        else:
            self.warning(f"Compressed entropy out of range: {compressed_entropy:.2f}")

        # Encrypted should be very high (>= 7.0 for small samples)
        if encrypted_entropy >= 7.0:
            self.success(f"Encrypted has very high entropy ({encrypted_entropy:.2f} >= 7.0)")
        else:
            self.warning(f"Encrypted entropy too low: {encrypted_entropy:.2f}")

        # ===== Part 2: Sliding Entropy Windows =====
        self.section("Part 2: Sliding Entropy Window Analysis")
        self.subsection("Entropy Profile Generation")

        window_size = 128
        step_size = 32
        entropy_profile = sliding_entropy(test_data, window=window_size, step=step_size)

        self.result("Window size", window_size, "bytes")
        self.result("Step size", step_size, "bytes")
        self.result("Profile length", len(entropy_profile), "windows")
        self.result("Min entropy", f"{np.min(entropy_profile):.3f}", "bits/byte")
        self.result("Max entropy", f"{np.max(entropy_profile):.3f}", "bits/byte")
        self.result("Mean entropy", f"{np.mean(entropy_profile):.3f}", "bits/byte")

        results["entropy_profile"] = entropy_profile
        results["profile_min"] = float(np.min(entropy_profile))
        results["profile_max"] = float(np.max(entropy_profile))

        # ===== Part 3: Entropy Transition Detection =====
        self.section("Part 3: Entropy Transition Detection")
        self.subsection("Automatic Boundary Detection")

        transitions = detect_entropy_transitions(test_data, window=128, threshold=1.0, min_gap=64)

        self.result("Transitions detected", len(transitions))
        results["transitions_detected"] = len(transitions)

        if transitions:
            self.subsection("Detected Transitions")
            for i, trans in enumerate(transitions):
                self.info(
                    f"Transition {i + 1}: offset={trans.offset}, "
                    f"before={trans.entropy_before:.2f}, "
                    f"after={trans.entropy_after:.2f}, "
                    f"delta={trans.delta:.2f}, "
                    f"type={trans.transition_type}"
                )

            results["transition_offsets"] = [t.offset for t in transitions]
            results["transition_deltas"] = [t.delta for t in transitions]

        # Validate transitions align with ground truth
        self.subsection("Transition Validation")
        expected_boundaries = [
            ground_truth["plaintext"][1],  # End of plaintext
            ground_truth["structured"][1],  # End of structured
            ground_truth["compressed"][1],  # End of compressed
        ]

        # Check if detected transitions are near expected boundaries
        detected_near_expected = 0
        tolerance = 128  # Allow 128 byte tolerance for window effects

        for expected in expected_boundaries:
            for trans in transitions:
                if abs(trans.offset - expected) <= tolerance:
                    detected_near_expected += 1
                    break

        if detected_near_expected > 0:
            self.success(f"Detected {detected_near_expected} transitions near expected boundaries")
        else:
            self.warning("No transitions detected near expected boundaries")

        results["transitions_near_expected"] = detected_near_expected

        # ===== Part 4: Data Classification by Entropy =====
        self.section("Part 4: Entropy-Based Data Classification")
        self.subsection("Per-Region Classification")

        classifications = {}
        for region_name in ["plaintext", "structured", "compressed", "encrypted"]:
            region_data = data[region_name]
            result = classify_by_entropy(region_data)
            classifications[region_name] = result
            self.info(
                f"{region_name}: {result.classification} "
                f"(entropy={result.entropy:.2f}, "
                f"confidence={result.confidence:.2f})"
            )

        results["classifications"] = {
            name: res.classification for name, res in classifications.items()
        }

        # Validate classifications
        self.subsection("Classification Validation")

        # Plaintext should be classified as text or structured
        plaintext_class = classifications["plaintext"].classification
        if plaintext_class in ["text", "structured"]:
            self.success(f"Plaintext correctly classified as '{plaintext_class}'")
        else:
            self.warning(f"Plaintext classified as '{plaintext_class}' (expected text/structured)")

        # Encrypted should be classified as random or compressed (both high entropy)
        encrypted_class = classifications["encrypted"].classification
        if encrypted_class in ["random", "compressed"]:
            self.success(f"Encrypted data classified as high-entropy '{encrypted_class}'")
        else:
            self.warning(
                f"Encrypted classified as '{encrypted_class}' (expected random/compressed)"
            )

        # Compressed should be classified as compressed
        compressed_class = classifications["compressed"].classification
        if compressed_class in ["compressed", "random"]:
            self.success(f"Compressed data classified as '{compressed_class}'")
        else:
            self.warning(f"Compressed classified as '{compressed_class}'")

        # ===== Part 5: Region Detection =====
        self.section("Part 5: Automatic Region Detection")
        self.subsection("Encrypted Region Detection")

        encrypted_regions = detect_encrypted_regions(test_data, min_length=64, min_entropy=7.5)
        self.result("Encrypted regions found", len(encrypted_regions))

        if encrypted_regions:
            for i, region in enumerate(encrypted_regions):
                self.info(
                    f"  Region {i + 1}: bytes {region.start}-{region.end} "
                    f"(length={region.length}, "
                    f"entropy={region.classification.entropy:.2f})"
                )

        results["encrypted_regions_found"] = len(encrypted_regions)

        # Check if detected encrypted region overlaps with actual encrypted data
        encrypted_detected = False
        actual_encrypted_start, actual_encrypted_end = ground_truth["encrypted"]
        for region in encrypted_regions:
            # Check for overlap
            if not (region.end <= actual_encrypted_start or region.start >= actual_encrypted_end):
                encrypted_detected = True
                break

        if encrypted_detected:
            self.success("Successfully detected encrypted region!")
        else:
            self.warning("Encrypted region not detected")

        results["encrypted_detected"] = encrypted_detected

        self.subsection("Compressed Region Detection")

        compressed_regions = detect_compressed_regions(test_data, min_length=64)
        self.result("Compressed regions found", len(compressed_regions))

        if compressed_regions:
            for i, region in enumerate(compressed_regions):
                self.info(
                    f"  Region {i + 1}: bytes {region.start}-{region.end} "
                    f"(length={region.length}, "
                    f"type={region.classification.details.get('compression_type', 'unknown')})"
                )

        results["compressed_regions_found"] = len(compressed_regions)

        # Check if detected compressed region matches actual
        compressed_detected = False
        actual_compressed_start, actual_compressed_end = ground_truth["compressed"]
        for region in compressed_regions:
            # Check for overlap
            if not (region.end <= actual_compressed_start or region.start >= actual_compressed_end):
                compressed_detected = True
                break

        if compressed_detected:
            self.success("Successfully detected compressed region!")
        else:
            self.warning("Compressed region not detected")

        results["compressed_detected"] = compressed_detected

        # ===== Part 6: Practical Application - Protocol Segmentation =====
        self.section("Part 6: Practical Application - Unknown Protocol Segmentation")
        self.subsection("Entropy-Based Protocol Analysis")

        self.info("Scenario: Captured unknown protocol traffic, no prior knowledge")
        self.info("Task: Segment into meaningful regions for further analysis")
        self.info("")
        self.info("Analysis strategy:")
        self.info("  1. Compute sliding entropy profile")
        self.info("  2. Detect significant entropy transitions")
        self.info("  3. Classify each segment by entropy characteristics")
        self.info("  4. Identify encrypted payloads, compressed data, plaintext headers")
        self.info("")

        # Simulate segmentation based on transitions
        if transitions:
            segments = []
            segment_start = 0

            for trans in transitions:
                segments.append((segment_start, trans.offset))
                segment_start = trans.offset

            # Last segment
            segments.append((segment_start, len(test_data)))

            self.subsection("Segmentation Results")
            self.result("Segments identified", len(segments))

            for i, (start, end) in enumerate(segments):
                segment_data = test_data[start:end]
                if len(segment_data) >= 4:  # Only classify non-trivial segments
                    seg_result = classify_by_entropy(segment_data)
                    self.info(
                        f"  Segment {i + 1}: bytes {start}-{end} ({end - start} bytes) -> "
                        f"{seg_result.classification} (entropy={seg_result.entropy:.2f})"
                    )

            results["segments_identified"] = len(segments)
        else:
            self.info("No transitions detected - data appears homogeneous")
            results["segments_identified"] = 1

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate entropy analysis results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        all_valid = True

        # Validate entropy ranges
        region_entropies = results.get("region_entropies", {})

        if "encrypted" in region_entropies:
            encrypted_entropy = region_entropies["encrypted"]
            if encrypted_entropy < 7.0:
                self.error(
                    f"Encrypted data entropy too low: {encrypted_entropy:.2f} < 7.0 bits/byte"
                )
                all_valid = False
            else:
                self.success(f"Encrypted entropy validated: {encrypted_entropy:.2f} >= 7.0")

        if "plaintext" in region_entropies:
            plaintext_entropy = region_entropies["plaintext"]
            if plaintext_entropy >= 5.0:
                self.error(f"Plaintext entropy too high: {plaintext_entropy:.2f} >= 5.0 bits/byte")
                all_valid = False
            else:
                self.success(f"Plaintext entropy validated: {plaintext_entropy:.2f} < 5.0")

        # Validate entropy profile
        profile_min = results.get("profile_min", 0)
        profile_max = results.get("profile_max", 0)

        if profile_max - profile_min < 2.0:
            self.warning(
                f"Low entropy variation detected: {profile_max - profile_min:.2f} bits/byte"
            )
        else:
            self.success(f"Good entropy variation: {profile_max - profile_min:.2f} bits/byte range")

        # Validate transitions detected
        transitions_detected = results.get("transitions_detected", 0)
        if transitions_detected == 0:
            self.warning("No entropy transitions detected (expected at least 1)")
        else:
            self.success(f"Detected {transitions_detected} entropy transitions")

        # Validate classifications
        classifications = results.get("classifications", {})

        # Encrypted data should be random or compressed (both are high entropy)
        encrypted_class = classifications.get("encrypted")
        if encrypted_class not in ["random", "compressed"]:
            self.warning(
                f"Encrypted data classified as '{encrypted_class}' "
                f"(expected 'random' or 'compressed')"
            )

        if classifications.get("plaintext") not in ["text", "structured"]:
            self.warning(
                f"Plaintext classified as '{classifications.get('plaintext')}' "
                f"instead of 'text' or 'structured'"
            )

        # Overall validation
        if all_valid:
            self.success("All entropy analysis validations passed!")
        else:
            self.error("Some entropy analysis validations failed")

        return all_valid


if __name__ == "__main__":
    demo = EntropyAnalysisDemo()
    success = demo.execute()
    exit(0 if success else 1)
