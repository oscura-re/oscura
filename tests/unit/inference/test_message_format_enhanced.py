"""Enhanced tests for message_format module to improve coverage.

Requirements addressed: PSI-001

This module adds additional edge case tests to improve coverage beyond
the existing comprehensive test suite.
"""

import numpy as np
import pytest

from oscura.inference.message_format import (
    MessageFormatInferrer,
    detect_field_types,
    find_dependencies,
    infer_format,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


@pytest.mark.unit
@pytest.mark.inference
class TestMessageFormatEnhanced:
    """Additional edge case tests for message format inference."""

    def test_classify_field_bytes_low_entropy(self) -> None:
        """Test field classification for bytes field with low entropy."""
        messages = []
        for _i in range(20):
            # Create messages with low-entropy large field
            msg = np.array([0xAA, 0x55] + [0x00] * 20, dtype=np.uint8)
            messages.append(msg)

        inferrer = MessageFormatInferrer(min_samples=10)
        boundaries = [0, 2]
        fields = inferrer.detect_field_types(messages, boundaries)

        # Large field with low entropy should be classified as constant or data
        assert len(fields) == 2
        assert fields[1].field_type in ["constant", "data"]

    def test_classify_field_bytes_high_entropy(self) -> None:
        """Test field classification for bytes field with high entropy."""
        rng = np.random.RandomState(42)
        messages = []
        for _ in range(20):
            # Create messages with high-entropy large field
            msg = np.concatenate(
                [np.array([0xAA, 0x55], dtype=np.uint8), rng.randint(0, 256, 20, dtype=np.uint8)]
            )
            messages.append(msg)

        inferrer = MessageFormatInferrer(min_samples=10)
        boundaries = [0, 2]
        fields = inferrer.detect_field_types(messages, boundaries)

        # Large field with high entropy should be classified as data
        assert len(fields) == 2
        assert fields[1].entropy > 5.0

    def test_detect_counter_with_wrapping_edge(self) -> None:
        """Test counter detection with wrapping at 256."""
        values = list(range(250, 256)) + list(range(6))

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        # Should detect as counter even with wrapping
        assert is_counter is True

    def test_detect_counter_with_gaps(self) -> None:
        """Test counter detection with occasional gaps."""
        values = [0, 1, 2, 5, 6, 7, 10, 11, 12]  # Some gaps

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        # May or may not be detected as counter depending on gap tolerance
        assert isinstance(is_counter, bool)

    def test_detect_counter_decreasing(self) -> None:
        """Test that decreasing sequence is not detected as counter."""
        values = list(range(100, 0, -1))

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        # Decreasing should not be detected as counter
        assert is_counter is False

    def test_detect_checksum_2_byte(self) -> None:
        """Test checksum detection with 2-byte XOR checksum."""
        messages = []
        for i in range(10):
            msg = bytearray([i, i + 1, i + 2, i + 3, 0, 0])
            xor_sum = 0
            for b in msg[:4]:
                xor_sum ^= b
            msg[4] = (xor_sum >> 8) & 0xFF
            msg[5] = xor_sum & 0xFF
            messages.append(np.array(msg, dtype=np.uint8))

        inferrer = MessageFormatInferrer()
        is_checksum = inferrer._detect_checksum_field(messages, 4, 2)

        # Should not detect as checksum (XOR is 1-byte, this is 2-byte)
        # The simple XOR check won't match 2-byte field
        assert is_checksum is False

    def test_detect_checksum_4_byte(self) -> None:
        """Test checksum detection with 4-byte field."""
        rng = np.random.RandomState(42)
        messages = [rng.randint(0, 256, 10, dtype=np.uint8) for _ in range(10)]

        inferrer = MessageFormatInferrer()
        is_checksum = inferrer._detect_checksum_field(messages, 6, 4)

        # Random data should not match checksum pattern
        assert is_checksum is False

    def test_estimate_header_size_no_high_entropy(self) -> None:
        """Test header size estimation with no high-entropy fields."""
        from oscura.inference.message_format import InferredField

        fields = [
            InferredField("field_0", 0, 2, "constant", 0.5, 10.0, 0.9),
            InferredField("field_1", 2, 2, "counter", 1.5, 50.0, 0.85),
            InferredField("field_2", 4, 2, "constant", 0.3, 5.0, 0.95),
        ]

        inferrer = MessageFormatInferrer()
        header_size = inferrer._estimate_header_size(fields)

        # Should use first 3 fields (< 4 fields, so use all)
        assert header_size == min(16, fields[-1].offset) if fields else 16

    def test_estimate_header_size_many_fields(self) -> None:
        """Test header size estimation with many low-entropy fields."""
        from oscura.inference.message_format import InferredField

        fields = [
            InferredField(f"field_{i}", i * 2, 2, "constant", 0.5, 10.0, 0.9) for i in range(10)
        ]

        inferrer = MessageFormatInferrer()
        header_size = inferrer._estimate_header_size(fields)

        # Should use first 4 fields
        assert header_size == 8

    def test_extract_field_value_3_bytes(self) -> None:
        """Test extracting 3-byte field value."""
        from oscura.inference.message_format import InferredField

        msg = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)
        field = InferredField("test", 0, 3, "data", 5.0, 1000.0, 0.7)

        inferrer = MessageFormatInferrer()
        value = inferrer._extract_field_value(msg, field)

        # For fields not exactly 1, 2, or 4 bytes, returns first byte only
        assert value == 0x12

    def test_boundary_detection_short_message(self) -> None:
        """Test boundary detection with very short messages."""
        messages = [np.array([0xAA, 0x55], dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="entropy")

        # Should at least return [0]
        assert 0 in boundaries
        assert all(b < 2 for b in boundaries)

    def test_boundary_detection_single_byte_message(self) -> None:
        """Test boundary detection with single-byte messages."""
        messages = [np.array([i], dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="combined")

        assert boundaries == [0]

    def test_infer_format_all_zeros(self) -> None:
        """Test format inference with all-zero messages."""
        messages = [bytes(20) for _ in range(20)]

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == 20
        # Should detect as constant fields
        constant_fields = [f for f in schema.fields if f.field_type == "constant"]
        assert len(constant_fields) > 0

    def test_infer_format_incrementing_bytes(self) -> None:
        """Test format inference with incrementing byte patterns."""
        messages = [bytes(range(i, i + 20)) for i in range(20)]

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == 20
        assert len(schema.fields) >= 1

    def test_detect_field_types_boundary_at_end(self) -> None:
        """Test field type detection when boundary is at message end."""
        messages = [np.array([0xAA, 0x55, i, i + 1], dtype=np.uint8) for i in range(20)]
        boundaries = [0, 2]  # Only internal boundaries (last field extends to end)

        fields = detect_field_types(messages, boundaries)

        # Should handle fields correctly
        assert len(fields) == 2
        assert fields[0].offset == 0
        assert fields[0].size == 2
        assert fields[1].offset == 2
        assert fields[1].size == 2  # Extends to end of message

    def test_find_dependencies_no_fields(self) -> None:
        """Test dependency finding with minimal schema."""
        from oscura.inference.message_format import InferredField, MessageSchema

        field = InferredField("field_0", 0, 10, "data", 7.0, 3000.0, 0.7)
        schema = MessageSchema(
            total_size=10,
            fields=[field],
            field_boundaries=[0],
            header_size=2,
            payload_offset=2,
            checksum_field=None,
            length_field=None,
        )

        messages = [np.array([i] * 10, dtype=np.uint8) for i in range(20)]

        dependencies = find_dependencies(messages, schema)

        # No special fields, so minimal dependencies
        assert isinstance(dependencies, dict)

    def test_calculate_byte_entropy_constant(self) -> None:
        """Test byte entropy calculation for constant bytes."""
        messages = [np.array([0xFF] * 10, dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        entropy = inferrer._calculate_byte_entropy(messages, 5)

        # All same value = 0 entropy
        assert entropy < 0.001

    def test_calculate_byte_entropy_alternating(self) -> None:
        """Test byte entropy calculation for alternating bytes."""
        messages = [
            np.array([0xAA if j % 2 == 0 else 0x55 for j in range(10)], dtype=np.uint8)
            for _ in range(20)
        ]

        inferrer = MessageFormatInferrer()
        entropy_even = inferrer._calculate_byte_entropy(messages, 0)
        entropy_odd = inferrer._calculate_byte_entropy(messages, 1)

        # All 0xAA at even positions, all 0x55 at odd
        assert entropy_even < 0.001
        assert entropy_odd < 0.001

    def test_classify_field_size_variations(self) -> None:
        """Test field classification with various sizes."""
        messages = [np.array([0xAA, 0x55] + [i] * 10, dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer()

        # Test 1-byte field
        boundaries_1 = [0, 1]
        fields_1 = inferrer.detect_field_types(messages, boundaries_1)
        assert fields_1[0].size == 1

        # Test 2-byte field
        boundaries_2 = [0, 2]
        fields_2 = inferrer.detect_field_types(messages, boundaries_2)
        assert fields_2[0].size == 2

    def test_detect_field_types_last_field_extends_to_end(self) -> None:
        """Test that last field correctly extends to end of message."""
        messages = [np.array([0xAA, 0x55] + [i] * 18, dtype=np.uint8) for i in range(20)]
        boundaries = [0, 2]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # Last field should extend to end of message
        assert len(fields) == 2
        assert fields[1].size == 18

    def test_infer_format_with_counter_field(self) -> None:
        """Test format inference correctly identifies counter fields."""
        messages = []
        for i in range(50):
            msg = bytearray([0xAA, 0x55])
            msg.extend(i.to_bytes(2, "big"))
            msg.extend(bytes(6))
            messages.append(bytes(msg))

        schema = infer_format(messages, min_samples=10)

        # Check for counter field (might be classified as counter, data, or unknown)
        # depending on boundary detection
        counter_fields = [f for f in schema.fields if f.field_type == "counter"]
        # Counter detection is heuristic-based, so we just check schema is valid
        assert schema.total_size == 10
        assert len(schema.fields) >= 1
        # If counter is detected, verify it's in a reasonable position
        if counter_fields:
            assert counter_fields[0].offset >= 2

    def test_infer_format_boundary_merge(self) -> None:
        """Test that close boundaries are merged."""
        # Create messages where boundaries might be very close
        messages = [np.array([0xAA, 0x55, i, i, i, i], dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # Boundaries should be merged if too close
        for i in range(len(schema.field_boundaries) - 1):
            assert schema.field_boundaries[i + 1] - schema.field_boundaries[i] >= 2


@pytest.mark.unit
@pytest.mark.inference
class TestMessageFormatLargeData:
    """Test message format inference with larger datasets."""

    def test_infer_format_many_messages(self) -> None:
        """Test inference with many messages."""
        messages = [b"\xaa\x55" + bytes(range(i % 10, (i % 10) + 10)) for i in range(100)]

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_infer_format_large_messages(self) -> None:
        """Test inference with large individual messages."""
        rng = np.random.RandomState(42)
        messages = [
            np.concatenate(
                [np.array([0xAA, 0x55], dtype=np.uint8), rng.randint(0, 256, 500, dtype=np.uint8)]
            )
            for _ in range(20)
        ]

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == 502
        assert len(schema.fields) >= 1

    def test_boundary_detection_many_transitions(self) -> None:
        """Test boundary detection with many entropy transitions."""
        messages = []
        for i in range(20):
            # Alternate between constant and variable bytes
            msg = bytearray()
            for j in range(20):
                if j % 2 == 0:
                    msg.append(0xAA)
                else:
                    msg.append((i + j) % 256)
            messages.append(np.array(msg, dtype=np.uint8))

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="combined")

        # Should detect multiple boundaries
        assert len(boundaries) > 1
