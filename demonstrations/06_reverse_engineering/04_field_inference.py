"""Field Inference: Automatic field boundary and type detection

Demonstrates:
- oscura.inference.message_format.MessageFormatInferrer - Infer field structure
- oscura.inference.binary.detect_field_boundaries() - Boundary detection
- oscura.inference.binary.classify_field_type() - Type classification
- oscura.inference.binary.detect_length_field() - Length field detection
- oscura.inference.binary.detect_checksum_field() - Checksum detection
- Counter/timestamp field identification
- Length field correlation
- Checksum field detection

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/02_crc_recovery.py

Field inference is critical for understanding protocol structure. This
demonstration shows how to automatically detect field boundaries, classify
field types, and identify special fields like counters, lengths, and checksums.

This is a P0 CRITICAL feature - demonstrates field inference capability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class FieldInferenceDemo(BaseDemo):
    """Demonstrates automatic field boundary and type detection."""

    def __init__(self) -> None:
        """Initialize field inference demonstration."""
        super().__init__(
            name="field_inference",
            description="Automatic field boundary detection and type classification",
            capabilities=[
                "oscura.inference.message_format.MessageFormatInferrer",
                "oscura.inference.binary.detect_field_boundaries",
                "oscura.inference.binary.classify_field_type",
                "oscura.inference.binary.detect_length_field",
                "oscura.inference.binary.detect_checksum_field",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/02_crc_recovery.py",
            ],
        )
        self.messages: list[bytes] = []
        self.schema = None

    def generate_test_data(self) -> dict[str, Any]:
        """Generate structured messages with known field layout.

        Creates messages with:
        - Magic header (constant)
        - Sequence counter
        - Timestamp (increasing)
        - Message type (enum)
        - Payload length
        - Variable payload
        - Checksum

        Returns:
            Dictionary with message samples for field inference
        """
        self.section("Generating Structured Messages")

        MAGIC = 0xABCD
        timestamp = 1000000

        messages = []

        for seq in range(20):
            # Message type cycles through 1-4
            msg_type = (seq % 4) + 1

            # Payload size varies by type
            payload_size = msg_type * 4

            # Build message
            msg = bytearray()

            # Magic (2 bytes, constant)
            msg.extend(MAGIC.to_bytes(2, "big"))

            # Sequence (2 bytes, counter)
            msg.extend(seq.to_bytes(2, "big"))

            # Timestamp (4 bytes, increasing)
            msg.extend(timestamp.to_bytes(4, "big"))
            timestamp += 1000  # Increment by 1ms

            # Message type (1 byte, enum)
            msg.append(msg_type)

            # Payload length (1 byte)
            msg.append(payload_size)

            # Payload (variable)
            payload = bytes([np.random.randint(0, 256) for _ in range(payload_size)])
            msg.extend(payload)

            # Checksum (1 byte, XOR)
            checksum = 0
            for b in msg:
                checksum ^= b
            msg.append(checksum)

            messages.append(bytes(msg))

        self.messages = messages

        self.result("Messages generated", len(messages))
        self.result(
            "Message size range",
            f"{min(len(m) for m in messages)}-{max(len(m) for m in messages)}",
            "bytes",
        )

        # Show example message breakdown
        self.subsection("Example Message Structure")
        _example = messages[0]  # For reference
        self.info("Message 0 (ground truth):")
        self.info(f"  Bytes 0-1:   Magic = 0x{MAGIC:04X} (constant)")
        self.info("  Bytes 2-3:   Sequence = 0 (counter)")
        self.info(f"  Bytes 4-7:   Timestamp = {1000000} (increasing)")
        self.info("  Byte 8:      Type = 1 (enum)")
        self.info("  Byte 9:      Length = 4 (length field)")
        self.info("  Bytes 10-13: Payload (variable)")
        self.info("  Byte 14:     Checksum (XOR)")

        return {"messages": messages}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute field inference on structured messages."""
        results: dict[str, Any] = {}

        messages = data["messages"]

        # ===== Phase 1: Field Boundary Detection =====
        self.section("Part 1: Field Boundary Detection")

        try:
            from oscura.inference.message_format import MessageFormatInferrer

            # Use messages of same length for initial analysis
            same_length = [m for m in messages if len(m) == min(len(m) for m in messages)]

            inferrer = MessageFormatInferrer(min_samples=5)
            self.schema = inferrer.infer_format(same_length)

            self.success(f"Detected {len(self.schema.fields)} fields")
            results["fields_detected"] = len(self.schema.fields)

            self.subsection("Detected Boundaries")
            for boundary in self.schema.field_boundaries:
                self.info(f"  Byte {boundary}")

            results["boundaries"] = self.schema.field_boundaries

        except Exception as e:
            self.error(f"Field boundary detection failed: {e}")
            results["fields_detected"] = 0
            return results

        # ===== Phase 2: Field Type Classification =====
        self.section("Part 2: Field Type Classification")

        self.subsection("Field Analysis")

        constant_count = 0
        counter_count = 0
        length_count = 0
        checksum_count = 0
        data_count = 0

        for i, field in enumerate(self.schema.fields):
            self.info(f"\nField {i + 1}:")
            self.result("  Offset", field.offset)
            self.result("  Size", field.size, "bytes")
            self.result("  Type", field.field_type)
            self.result("  Entropy", f"{field.entropy:.3f}")
            self.result("  Variance", f"{field.variance:.2f}")
            self.result("  Confidence", f"{field.confidence * 100:.0f}%")

            # Count by type
            if field.field_type == "constant":
                constant_count += 1
            elif field.field_type == "counter":
                counter_count += 1
            elif field.field_type == "length":
                length_count += 1
            elif field.field_type == "checksum":
                checksum_count += 1
            elif field.field_type == "data":
                data_count += 1

            # Show sample values for interesting fields
            if field.values_seen and len(field.values_seen) > 0:
                sample_vals = field.values_seen[:3]
                self.info(f"  Sample values: {sample_vals}")

        # Summary by type
        self.subsection("Field Type Summary")
        self.result("Constant fields", constant_count)
        self.result("Counter fields", counter_count)
        self.result("Length fields", length_count)
        self.result("Checksum fields", checksum_count)
        self.result("Data fields", data_count)

        results["constant_fields"] = constant_count
        results["counter_fields"] = counter_count
        results["length_fields"] = length_count
        results["checksum_fields"] = checksum_count

        # ===== Phase 3: Special Field Detection =====
        self.section("Part 3: Special Field Detection")

        # Checksum field
        self.subsection("Checksum Field")
        if self.schema.checksum_field:
            checksum = self.schema.checksum_field
            self.success("Checksum field detected!")
            self.result("Offset", checksum.offset)
            self.result("Size", checksum.size, "bytes")
            self.result("Confidence", f"{checksum.confidence * 100:.0f}%")
            results["checksum_detected"] = True
        else:
            self.info("No checksum field detected")
            results["checksum_detected"] = False

        # Length field
        self.subsection("Length Field")
        if self.schema.length_field:
            length = self.schema.length_field
            self.success("Length field detected!")
            self.result("Offset", length.offset)
            self.result("Size", length.size, "bytes")
            self.result("Confidence", f"{length.confidence * 100:.0f}%")
            results["length_detected"] = True
        else:
            self.info("No length field detected")
            results["length_detected"] = False

        # ===== Phase 4: Field Correlation =====
        self.section("Part 4: Field Correlation Analysis")

        self.subsection("Counter Field Validation")

        # Find counter fields and verify they increment
        counter_fields = [f for f in self.schema.fields if f.field_type == "counter"]

        for field in counter_fields:
            # Extract values from all messages
            values = []
            for msg in same_length:
                value = int.from_bytes(msg[field.offset : field.offset + field.size], "big")
                values.append(value)

            # Check if monotonically increasing
            is_monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
            is_sequential = all(values[i + 1] - values[i] == 1 for i in range(len(values) - 1))

            self.info(f"Field at offset {field.offset}:")
            self.result("  Monotonic", str(is_monotonic))
            self.result("  Sequential", str(is_sequential))
            self.result("  Range", f"{min(values)}-{max(values)}")

            if is_sequential:
                self.success("  Confirmed as sequence counter!")

        # ===== Phase 5: Length Field Correlation =====
        self.subsection("Length Field Validation")

        if self.schema.length_field:
            length_field = self.schema.length_field

            # Verify length field correlates with message size
            correlations = 0
            for msg in messages:
                if len(msg) > length_field.offset + length_field.size:
                    length_value = int.from_bytes(
                        msg[length_field.offset : length_field.offset + length_field.size],
                        "big",
                    )
                    # Check if it matches payload size
                    expected_start = length_field.offset + length_field.size + 1
                    actual_payload = len(msg) - expected_start - 1  # Minus checksum

                    if length_value == actual_payload:
                        correlations += 1

            correlation_rate = correlations / len(messages) * 100
            self.result("Correlation rate", f"{correlation_rate:.0f}%")

            if correlation_rate > 90:
                self.success("Length field validated!")
                results["length_validated"] = True
            else:
                results["length_validated"] = False

        # ===== Summary =====
        self.section("Field Inference Summary")

        self.result("Total fields detected", len(self.schema.fields))
        self.result("Constant fields (magic/sync)", constant_count)
        self.result("Counter fields (sequence)", counter_count)
        self.result("Length fields", length_count)
        self.result("Checksum fields", checksum_count)
        self.result("Data/variable fields", data_count)

        results["inference_complete"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate field inference results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must detect fields
        if results.get("fields_detected", 0) < 5:
            self.error("Expected at least 5 fields detected")
            return False

        # Should detect at least one constant field (magic)
        if results.get("constant_fields", 0) == 0:
            self.warning("No constant fields detected")

        # Should detect at least one counter
        if results.get("counter_fields", 0) == 0:
            self.warning("No counter fields detected")

        # Should detect checksum
        if not results.get("checksum_detected", False):
            self.warning("Checksum not detected")

        # Should detect length field
        if not results.get("length_detected", False):
            self.warning("Length field not detected")

        # Overall success
        if not results.get("inference_complete", False):
            self.error("Field inference incomplete")
            return False

        self.success("Field inference demonstration complete!")
        return True


if __name__ == "__main__":
    demo = FieldInferenceDemo()
    success = demo.execute()
    exit(0 if success else 1)
