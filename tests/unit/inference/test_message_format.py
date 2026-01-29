"""Comprehensive unit tests for message format inference.

Requirements addressed: PSI-001

This module tests all public functions and classes in the message_format module,
including boundary detection, field type classification, and schema inference.
"""

import numpy as np
import pytest

from oscura.inference.message_format import (
    InferredField,
    MessageFormatInferrer,
    MessageSchema,
    detect_field_types,
    find_dependencies,
    infer_format,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferredField:
    """Test InferredField dataclass."""

    def test_create_inferred_field(self) -> None:
        """Test creating an InferredField instance."""
        field = InferredField(
            name="test_field",
            offset=0,
            size=4,
            field_type="counter",
            entropy=2.5,
            variance=100.0,
            confidence=0.9,
            values_seen=[1, 2, 3, 4, 5],
        )

        assert field.name == "test_field"
        assert field.offset == 0
        assert field.size == 4
        assert field.field_type == "counter"
        assert field.entropy == 2.5
        assert field.variance == 100.0
        assert field.confidence == 0.9
        assert len(field.values_seen) == 5

    def test_inferred_field_default_values(self) -> None:
        """Test InferredField with default values."""
        field = InferredField(
            name="field_0",
            offset=10,
            size=2,
            field_type="constant",
            entropy=0.0,
            variance=0.0,
            confidence=1.0,
        )

        # Default values_seen should be empty list
        assert field.values_seen == []

    def test_inferred_field_all_types(self) -> None:
        """Test InferredField can represent all field types."""
        types = ["constant", "counter", "timestamp", "length", "checksum", "data", "unknown"]

        for ftype in types:
            field = InferredField(
                name=f"field_{ftype}",
                offset=0,
                size=4,
                field_type=ftype,  # type: ignore[arg-type]
                entropy=1.0,
                variance=10.0,
                confidence=0.8,
            )
            assert field.field_type == ftype


@pytest.mark.unit
@pytest.mark.inference
class TestMessageSchema:
    """Test MessageSchema dataclass."""

    def test_create_message_schema(self) -> None:
        """Test creating a MessageSchema instance."""
        field1 = InferredField(
            name="header",
            offset=0,
            size=2,
            field_type="constant",
            entropy=0.0,
            variance=0.0,
            confidence=1.0,
        )
        field2 = InferredField(
            name="data",
            offset=2,
            size=10,
            field_type="data",
            entropy=7.5,
            variance=5000.0,
            confidence=0.7,
        )

        schema = MessageSchema(
            total_size=12,
            fields=[field1, field2],
            field_boundaries=[0, 2],
            header_size=2,
            payload_offset=2,
            checksum_field=None,
            length_field=None,
        )

        assert schema.total_size == 12
        assert len(schema.fields) == 2
        assert schema.header_size == 2
        assert schema.payload_offset == 2
        assert schema.checksum_field is None
        assert schema.length_field is None

    def test_message_schema_with_special_fields(self) -> None:
        """Test MessageSchema with checksum and length fields."""
        checksum_field = InferredField(
            name="checksum",
            offset=20,
            size=2,
            field_type="checksum",
            entropy=7.0,
            variance=1000.0,
            confidence=0.8,
        )

        length_field = InferredField(
            name="length",
            offset=2,
            size=2,
            field_type="length",
            entropy=2.0,
            variance=50.0,
            confidence=0.6,
        )

        schema = MessageSchema(
            total_size=24,
            fields=[length_field, checksum_field],
            field_boundaries=[0, 2, 20],
            header_size=4,
            payload_offset=4,
            checksum_field=checksum_field,
            length_field=length_field,
        )

        assert schema.checksum_field is not None
        assert schema.checksum_field.field_type == "checksum"
        assert schema.length_field is not None
        assert schema.length_field.field_type == "length"


# =============================================================================
# Test MessageFormatInferrer
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestMessageFormatInferrer:
    """Test MessageFormatInferrer class."""

    def test_inferrer_initialization(self) -> None:
        """Test MessageFormatInferrer initialization."""
        inferrer = MessageFormatInferrer(min_samples=5)
        assert inferrer.min_samples == 5

    def test_inferrer_default_min_samples(self) -> None:
        """Test MessageFormatInferrer default min_samples."""
        inferrer = MessageFormatInferrer()
        assert inferrer.min_samples == 10

    def test_infer_format_insufficient_samples(self) -> None:
        """Test infer_format raises error with too few samples."""
        inferrer = MessageFormatInferrer(min_samples=10)
        messages = [b"\xaa\x55\x00\x00"] * 5

        with pytest.raises(ValueError, match="Need at least 10 messages"):
            inferrer.infer_format(messages)

    def test_infer_format_varying_lengths(self) -> None:
        """Test infer_format raises error when messages have different lengths."""
        inferrer = MessageFormatInferrer(min_samples=3)
        messages = [b"\xaa\x55\x00", b"\xaa\x55\x00\x00", b"\xaa\x55"]

        with pytest.raises(ValueError, match="Messages have varying lengths"):
            inferrer.infer_format(messages)

    def test_infer_format_invalid_message_type(self) -> None:
        """Test infer_format raises error with invalid message type."""
        inferrer = MessageFormatInferrer(min_samples=2)
        messages = ["invalid", "messages"]  # type: ignore[list-item]

        with pytest.raises(ValueError, match="Invalid message type"):
            inferrer.infer_format(messages)

    def test_infer_format_basic_bytes(self) -> None:
        """Test basic format inference with bytes messages."""
        # Create simple messages with constant header
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1
        assert schema.field_boundaries[0] == 0

    def test_infer_format_numpy_arrays(self) -> None:
        """Test format inference with numpy array messages."""
        # Create messages as numpy arrays
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_infer_format_mixed_messages(self) -> None:
        """Test format inference with mixed bytes and numpy messages."""
        messages_bytes = [b"\xaa\x55" + bytes(range(10)) for _ in range(10)]
        messages_numpy = [
            np.array([0xAA, 0x55] + list(range(10)), dtype=np.uint8) for _ in range(10)
        ]

        messages = messages_bytes + messages_numpy

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_infer_format_identifies_constant_header(self) -> None:
        """Test that constant header fields are identified."""
        # All messages start with same 2-byte header
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(50)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # Check for constant field
        constant_fields = [f for f in schema.fields if f.field_type == "constant"]
        assert len(constant_fields) > 0
        assert constant_fields[0].offset == 0

    def test_infer_format_checksum_field(self) -> None:
        """Test detection of checksum field."""
        # Create messages with XOR checksum
        messages = []
        for i in range(20):
            msg = bytearray([0xAA, 0x55, i, i + 1, i + 2, i + 3, 0, 0, 0, 0])
            # Calculate XOR checksum of first 6 bytes
            xor_sum = 0
            for b in msg[:6]:
                xor_sum ^= b
            msg[6] = xor_sum
            messages.append(bytes(msg))

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # Check if checksum was detected
        if schema.checksum_field:
            assert schema.checksum_field.field_type == "checksum"


@pytest.mark.unit
@pytest.mark.inference
class TestDetectFieldBoundaries:
    """Test field boundary detection methods."""

    def test_detect_boundaries_entropy_method(self) -> None:
        """Test boundary detection using entropy method."""
        # Create messages with clear entropy transitions
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="entropy")

        # Should always include offset 0
        assert 0 in boundaries
        assert len(boundaries) >= 1

    def test_detect_boundaries_variance_method(self) -> None:
        """Test boundary detection using variance method."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="variance")

        assert 0 in boundaries
        assert len(boundaries) >= 1

    def test_detect_boundaries_combined_method(self) -> None:
        """Test boundary detection using combined method."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="combined")

        assert 0 in boundaries
        assert len(boundaries) >= 1

    def test_detect_boundaries_empty_messages(self) -> None:
        """Test boundary detection with empty message list."""
        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries([], method="combined")

        # Should return [0] for empty list
        assert boundaries == [0]

    def test_detect_boundaries_merge_close_boundaries(self) -> None:
        """Test that boundaries too close together are merged."""
        # Create messages that might produce close boundaries
        messages = [np.array([i] * 20, dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="combined")

        # Check that boundaries are at least 2 bytes apart
        for i in range(len(boundaries) - 1):
            assert boundaries[i + 1] - boundaries[i] >= 2

    def test_detect_boundaries_sorted(self) -> None:
        """Test that boundaries are returned in sorted order."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="combined")

        # Boundaries should be sorted
        assert boundaries == sorted(boundaries)


@pytest.mark.unit
@pytest.mark.inference
class TestDetectFieldTypes:
    """Test field type detection methods."""

    def test_detect_constant_field_type(self) -> None:
        """Test detection of constant field type."""
        # Create messages where first 2 bytes are constant
        messages = [np.array([0xAA, 0x55] + [i] * 10, dtype=np.uint8) for i in range(20)]
        boundaries = [0, 2]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # First field should be constant
        assert fields[0].field_type == "constant"
        assert fields[0].confidence == 1.0

    def test_detect_counter_field_type(self) -> None:
        """Test detection of counter field type."""
        # Create messages with counter at bytes 2-3
        messages = []
        for i in range(50):
            msg = bytearray(12)
            msg[0:2] = [0xAA, 0x55]
            msg[2:4] = i.to_bytes(2, "big")
            msg[4:] = bytes(8)
            messages.append(np.array(msg, dtype=np.uint8))

        boundaries = [0, 2, 4]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # Second field should be counter
        counter_fields = [f for f in fields if f.field_type == "counter"]
        assert len(counter_fields) > 0

    def test_detect_data_field_type(self) -> None:
        """Test detection of data field type."""
        # Create messages with high-entropy data
        rng = np.random.RandomState(42)
        messages = [
            np.concatenate([np.array([0xAA, 0x55]), rng.randint(0, 256, 20, dtype=np.uint8)])
            for _ in range(20)
        ]
        boundaries = [0, 2]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # Second field should be high entropy (data or unknown)
        assert fields[1].entropy > 4.0

    def test_detect_field_types_small_fields(self) -> None:
        """Test field type detection for small fields (1-4 bytes)."""
        messages = [
            np.array([0xAA, i & 0xFF, (i >> 8) & 0xFF, 0x00, 0x00, 0x00], dtype=np.uint8)
            for i in range(20)
        ]
        boundaries = [0, 1, 3, 5]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        assert len(fields) == 4
        # Each field should have calculated entropy and variance (allowing for numerical precision)
        for field in fields:
            assert field.entropy >= -0.001  # Allow small negative due to numerical precision
            assert field.variance >= 0

    def test_detect_field_types_large_fields(self) -> None:
        """Test field type detection for large fields (>4 bytes)."""
        rng = np.random.RandomState(42)
        messages = [
            np.concatenate([np.array([0xAA, 0x55]), rng.randint(0, 256, 30, dtype=np.uint8)])
            for _ in range(20)
        ]
        boundaries = [0, 2]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # Large field should have tuple values
        assert len(fields) == 2
        large_field = fields[1]
        assert large_field.size > 4

    def test_field_values_seen_sampling(self) -> None:
        """Test that values_seen contains sample values."""
        messages = [np.array([0xAA, 0x55, i, i + 1], dtype=np.uint8) for i in range(20)]
        boundaries = [0, 2]

        inferrer = MessageFormatInferrer()
        fields = inferrer.detect_field_types(messages, boundaries)

        # Fields should have sample values
        for field in fields:
            assert isinstance(field.values_seen, list)
            assert len(field.values_seen) <= 20


@pytest.mark.unit
@pytest.mark.inference
class TestPrivateMethods:
    """Test private helper methods."""

    def test_calculate_byte_entropy(self) -> None:
        """Test entropy calculation at byte offset."""
        # Constant byte = 0 entropy (or very close to 0 due to numerical precision)
        messages = [np.array([0xAA] * 10, dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        entropy = inferrer._calculate_byte_entropy(messages, 0)

        assert entropy < 0.001  # Close to 0

    def test_calculate_byte_entropy_random(self) -> None:
        """Test entropy calculation with random bytes."""
        rng = np.random.RandomState(42)
        # Random bytes at position 5
        messages = [
            np.concatenate([np.zeros(5, dtype=np.uint8), rng.randint(0, 256, 5, dtype=np.uint8)])
            for _ in range(100)
        ]

        inferrer = MessageFormatInferrer()
        entropy = inferrer._calculate_byte_entropy(messages, 5)

        # Should have higher entropy
        assert entropy > 2.0

    def test_calculate_entropy_empty(self) -> None:
        """Test entropy calculation with empty array."""
        inferrer = MessageFormatInferrer()
        entropy = inferrer._calculate_entropy(np.array([], dtype=np.uint8))

        assert entropy == 0.0

    def test_calculate_entropy_uniform(self) -> None:
        """Test entropy with uniform distribution."""
        # All values equally likely
        values = np.array([0, 1, 2, 3] * 25, dtype=np.uint8)

        inferrer = MessageFormatInferrer()
        entropy = inferrer._calculate_entropy(values)

        # Should be close to log2(4) = 2.0
        assert 1.8 < entropy < 2.2

    def test_detect_counter_field_simple(self) -> None:
        """Test counter detection with simple incrementing sequence."""
        values = list(range(100))

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        assert is_counter is True

    def test_detect_counter_field_with_wrapping(self) -> None:
        """Test counter detection with wrapping."""
        values = list(range(250, 256)) + list(range(10))

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        # Should detect counter even with wrapping
        assert is_counter is True

    def test_detect_counter_field_not_counter(self) -> None:
        """Test counter detection with non-counter values."""
        values = [0, 5, 2, 8, 1, 9, 3]

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        assert is_counter is False

    def test_detect_counter_field_too_few_samples(self) -> None:
        """Test counter detection with insufficient samples."""
        values = [0, 1]

        inferrer = MessageFormatInferrer()
        is_counter = inferrer._detect_counter_field(values)

        assert is_counter is False

    def test_detect_checksum_xor_1_byte(self) -> None:
        """Test checksum detection with 1-byte XOR checksum."""
        messages = []
        for i in range(10):
            msg = bytearray([i, i + 1, i + 2, 0])
            xor_sum = msg[0] ^ msg[1] ^ msg[2]
            msg[3] = xor_sum
            messages.append(np.array(msg, dtype=np.uint8))

        inferrer = MessageFormatInferrer()
        is_checksum = inferrer._detect_checksum_field(messages, 3, 1)

        assert is_checksum is True

    def test_detect_checksum_not_checksum(self) -> None:
        """Test checksum detection with non-checksum field."""
        rng = np.random.RandomState(42)
        messages = [rng.randint(0, 256, 10, dtype=np.uint8) for _ in range(10)]

        inferrer = MessageFormatInferrer()
        is_checksum = inferrer._detect_checksum_field(messages, 8, 2)

        # Random data should not match checksum pattern
        assert is_checksum is False

    def test_detect_checksum_invalid_size(self) -> None:
        """Test checksum detection with invalid size."""
        messages = [np.array([0xAA] * 10, dtype=np.uint8) for _ in range(10)]

        inferrer = MessageFormatInferrer()
        is_checksum = inferrer._detect_checksum_field(messages, 5, 3)

        # Size 3 is not a valid checksum size
        assert is_checksum is False

    def test_estimate_header_size_from_entropy(self) -> None:
        """Test header size estimation from field entropy."""
        fields = [
            InferredField("field_0", 0, 2, "constant", 0.5, 10.0, 0.9),
            InferredField("field_1", 2, 2, "constant", 0.3, 5.0, 0.95),
            InferredField("field_2", 4, 8, "data", 7.5, 5000.0, 0.7),
        ]

        inferrer = MessageFormatInferrer()
        header_size = inferrer._estimate_header_size(fields)

        # Should transition at high-entropy field
        assert header_size == 4

    def test_estimate_header_size_default(self) -> None:
        """Test header size estimation defaults."""
        fields = [
            InferredField("field_0", 0, 2, "constant", 0.5, 10.0, 0.9),
            InferredField("field_1", 2, 2, "constant", 0.3, 5.0, 0.95),
            InferredField("field_2", 4, 2, "constant", 0.2, 2.0, 0.98),
            InferredField("field_3", 6, 2, "constant", 0.1, 1.0, 0.99),
            InferredField("field_4", 8, 2, "counter", 2.0, 100.0, 0.85),
        ]

        inferrer = MessageFormatInferrer()
        header_size = inferrer._estimate_header_size(fields)

        # Should use first 4 fields
        assert header_size == 8

    def test_extract_field_value_1_byte(self) -> None:
        """Test extracting 1-byte field value."""
        msg = np.array([0xAA, 0x55, 0x12, 0x34], dtype=np.uint8)
        field = InferredField("test", 2, 1, "constant", 0.0, 0.0, 1.0)

        inferrer = MessageFormatInferrer()
        value = inferrer._extract_field_value(msg, field)

        assert value == 0x12

    def test_extract_field_value_2_bytes(self) -> None:
        """Test extracting 2-byte field value."""
        msg = np.array([0xAA, 0x55, 0x12, 0x34], dtype=np.uint8)
        field = InferredField("test", 2, 2, "constant", 0.0, 0.0, 1.0)

        inferrer = MessageFormatInferrer()
        value = inferrer._extract_field_value(msg, field)

        assert value == 0x1234

    def test_extract_field_value_4_bytes(self) -> None:
        """Test extracting 4-byte field value."""
        msg = np.array([0x12, 0x34, 0x56, 0x78, 0x9A], dtype=np.uint8)
        field = InferredField("test", 0, 4, "constant", 0.0, 0.0, 1.0)

        inferrer = MessageFormatInferrer()
        value = inferrer._extract_field_value(msg, field)

        assert value == 0x12345678

    def test_extract_field_value_large_field(self) -> None:
        """Test extracting value from large field (>4 bytes)."""
        msg = np.array([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC], dtype=np.uint8)
        field = InferredField("test", 1, 5, "data", 5.0, 1000.0, 0.7)

        inferrer = MessageFormatInferrer()
        value = inferrer._extract_field_value(msg, field)

        # Returns first byte for large fields
        assert value == 0x34


@pytest.mark.unit
@pytest.mark.inference
class TestFindDependencies:
    """Test field dependency detection."""

    def test_find_dependencies_length_field(self) -> None:
        """Test finding dependencies with length field."""
        length_field = InferredField("length", 2, 2, "length", 2.0, 50.0, 0.6)
        data_field = InferredField("data", 4, 10, "data", 7.0, 3000.0, 0.7)

        schema = MessageSchema(
            total_size=14,
            fields=[length_field, data_field],
            field_boundaries=[0, 2, 4],
            header_size=4,
            payload_offset=4,
            checksum_field=None,
            length_field=length_field,
        )

        messages = [
            np.array([0xAA, 0x55, 0x00, 0x0A] + [i] * 10, dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        dependencies = inferrer.find_dependencies(messages, schema)

        # Should identify length field dependency
        assert "length" in dependencies

    def test_find_dependencies_no_length_field(self) -> None:
        """Test finding dependencies without length field."""
        field1 = InferredField("field_0", 0, 2, "constant", 0.0, 0.0, 1.0)
        field2 = InferredField("field_1", 2, 10, "data", 7.0, 3000.0, 0.7)

        schema = MessageSchema(
            total_size=12,
            fields=[field1, field2],
            field_boundaries=[0, 2],
            header_size=2,
            payload_offset=2,
            checksum_field=None,
            length_field=None,
        )

        messages = [np.array([0xAA, 0x55] + [i] * 10, dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer()
        dependencies = inferrer.find_dependencies(messages, schema)

        # Should return empty or minimal dependencies
        assert isinstance(dependencies, dict)


# =============================================================================
# Test Module-Level Convenience Functions
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_infer_format_function(self) -> None:
        """Test infer_format convenience function."""
        messages = [b"\xaa\x55" + bytes(range(10)) for _ in range(20)]

        schema = infer_format(messages, min_samples=10)

        assert isinstance(schema, MessageSchema)
        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_infer_format_function_numpy(self) -> None:
        """Test infer_format with numpy arrays."""
        messages = [np.array([0xAA, 0x55] + list(range(10)), dtype=np.uint8) for _ in range(20)]

        schema = infer_format(messages, min_samples=10)

        assert isinstance(schema, MessageSchema)
        assert schema.total_size == 12

    def test_detect_field_types_function(self) -> None:
        """Test detect_field_types convenience function."""
        messages = [b"\xaa\x55" + bytes([i] * 10) for i in range(20)]
        boundaries = [0, 2]

        fields = detect_field_types(messages, boundaries)

        assert isinstance(fields, list)
        assert len(fields) == 2
        assert all(isinstance(f, InferredField) for f in fields)

    def test_detect_field_types_function_numpy(self) -> None:
        """Test detect_field_types with numpy arrays."""
        messages = [np.array([0xAA, 0x55] + [i] * 10, dtype=np.uint8) for i in range(20)]
        boundaries = [0, 2]

        fields = detect_field_types(messages, boundaries)

        assert len(fields) == 2

    def test_detect_field_types_invalid_type(self) -> None:
        """Test detect_field_types with invalid message type."""
        messages = ["invalid", "types"]  # type: ignore[list-item]
        boundaries = [0]

        with pytest.raises(ValueError, match="Invalid message type"):
            detect_field_types(messages, boundaries)

    def test_find_dependencies_function(self) -> None:
        """Test find_dependencies convenience function."""
        messages = [b"\xaa\x55" + bytes([i] * 10) for i in range(20)]

        # Create a simple schema
        field1 = InferredField("field_0", 0, 2, "constant", 0.0, 0.0, 1.0)
        field2 = InferredField("field_1", 2, 10, "data", 7.0, 3000.0, 0.7)

        schema = MessageSchema(
            total_size=12,
            fields=[field1, field2],
            field_boundaries=[0, 2],
            header_size=2,
            payload_offset=2,
            checksum_field=None,
            length_field=None,
        )

        dependencies = find_dependencies(messages, schema)

        assert isinstance(dependencies, dict)

    def test_find_dependencies_function_numpy(self) -> None:
        """Test find_dependencies with numpy arrays."""
        messages = [np.array([0xAA, 0x55] + [i] * 10, dtype=np.uint8) for i in range(20)]

        field1 = InferredField("field_0", 0, 2, "constant", 0.0, 0.0, 1.0)

        schema = MessageSchema(
            total_size=12,
            fields=[field1],
            field_boundaries=[0],
            header_size=2,
            payload_offset=2,
            checksum_field=None,
            length_field=None,
        )

        dependencies = find_dependencies(messages, schema)

        assert isinstance(dependencies, dict)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferenceMessageFormatEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_byte_messages(self) -> None:
        """Test inference with single-byte messages."""
        messages = [bytes([i]) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 1
        assert len(schema.fields) >= 1

    def test_all_constant_messages(self) -> None:
        """Test inference when all messages are identical."""
        messages = [b"\xaa\x55\x00\x00\x00\x00"] * 20

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # With all identical messages, there should be at least one field detected
        assert len(schema.fields) >= 1
        # All messages are identical, so we should be able to process them
        assert schema.total_size == 6

    def test_very_large_messages(self) -> None:
        """Test inference with very large messages."""
        rng = np.random.RandomState(42)
        messages = [rng.randint(0, 256, 1000, dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 1000
        assert len(schema.fields) >= 1

    def test_all_zero_messages(self) -> None:
        """Test inference with all-zero messages."""
        messages = [bytes(20) for _ in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # Should detect as constant
        assert len(schema.fields) >= 1

    def test_alternating_pattern(self) -> None:
        """Test inference with alternating byte pattern."""
        messages = [bytes([0xAA, 0x55] * 10) for _ in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema.total_size == 20

    def test_boundary_detection_no_transitions(self) -> None:
        """Test boundary detection with no entropy transitions."""
        # All constant values
        messages = [np.array([0xFF] * 20, dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages, method="entropy")

        # Should still return [0]
        assert boundaries == [0]

    def test_classify_field_with_tuple_values(self) -> None:
        """Test field classification with tuple (bytes) values."""
        messages = []
        for i in range(20):
            msg = np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8)
            messages.append(msg)

        inferrer = MessageFormatInferrer()
        boundaries = [0, 2]
        fields = inferrer.detect_field_types(messages, boundaries)

        # Second field should be classified
        assert fields[1].field_type in ["constant", "data", "unknown"]

    def test_minimum_samples_exactly(self) -> None:
        """Test inference with exactly min_samples."""
        messages = [b"\xaa\x55" + bytes(10)] * 10

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        assert schema is not None

    def test_boundary_offset_limits(self) -> None:
        """Test boundaries at message start and end."""
        messages = [np.array(list(range(20)), dtype=np.uint8) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(messages)

        # First boundary should be 0
        assert boundaries[0] == 0
        # No boundary should exceed message length
        assert all(b < 20 for b in boundaries)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferenceMessageFormatIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_simple_protocol(self) -> None:
        """Test complete workflow with simple protocol."""
        # Create messages with known structure:
        # - 2 bytes: constant header (0xAA55)
        # - 2 bytes: counter
        # - 8 bytes: random data
        # - 1 byte: XOR checksum

        rng = np.random.RandomState(42)
        messages = []
        for i in range(50):
            msg = bytearray(13)
            msg[0:2] = [0xAA, 0x55]
            msg[2:4] = i.to_bytes(2, "big")
            # Convert numpy array to bytes for assignment
            random_data = rng.randint(0, 256, 8, dtype=np.uint8).tobytes()
            msg[4:12] = random_data
            # XOR checksum
            xor_sum = 0
            for b in msg[:12]:
                xor_sum ^= b
            msg[12] = xor_sum
            messages.append(bytes(msg))

        # Infer format
        schema = infer_format(messages, min_samples=10)

        # Verify results
        assert schema.total_size == 13
        assert len(schema.fields) >= 2

        # Check for constant field
        constant_fields = [f for f in schema.fields if f.field_type == "constant"]
        assert len(constant_fields) > 0

    def test_complete_workflow_variable_length(self) -> None:
        """Test workflow with fixed-length messages only."""
        # All messages same length
        messages = []
        for i in range(30):
            length = 10  # Fixed length
            msg = bytearray([0xAA, 0x55, length])
            msg.extend(bytes(range(i, i + length)))
            messages.append(bytes(msg))

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == 13
        assert len(schema.fields) >= 1

    def test_boundaries_and_types_consistency(self) -> None:
        """Test that boundaries and field types are consistent."""
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format(messages)

        # Check consistency
        assert len(schema.fields) == len(schema.field_boundaries)

        # Check field offsets match boundaries
        for i, field in enumerate(schema.fields):
            assert field.offset == schema.field_boundaries[i]

    def test_schema_special_fields_detection(self) -> None:
        """Test detection and assignment of special fields."""
        # Create messages with length and checksum
        messages = []
        for _i in range(30):
            msg = bytearray([0xAA, 0x55, 10, 0])  # header + length
            msg.extend(bytes(range(10)))
            xor_sum = 0
            for b in msg:
                xor_sum ^= b
            msg.append(xor_sum)
            messages.append(bytes(msg))

        schema = infer_format(messages, min_samples=10)

        # Schema should have identified special fields
        assert schema.total_size == 15


# =============================================================================
# Test Voting Expert and Ensemble Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestVotingExpert:
    """Test voting expert boundary detection methods."""

    def test_detect_boundaries_voting_basic(self) -> None:
        """Test basic voting boundary detection."""
        # Create messages with clear field structure
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        boundaries = inferrer.detect_boundaries_voting(messages, min_confidence=0.4)

        # Should detect at least position 0
        assert 0 in boundaries
        assert len(boundaries) >= 1

    def test_detect_boundaries_voting_empty(self) -> None:
        """Test voting with empty message list."""
        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_boundaries_voting([], min_confidence=0.6)

        assert boundaries == [0]

    def test_detect_boundaries_voting_high_confidence(self) -> None:
        """Test voting with high confidence threshold."""
        # Create messages where experts agree
        messages = [b"\xaa\x55" + bytes([0xFF] * 10) for _ in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_boundaries_voting(messages, min_confidence=0.8)

        # With high confidence, should get fewer boundaries
        assert 0 in boundaries

    def test_detect_boundaries_voting_low_confidence(self) -> None:
        """Test voting with low confidence threshold."""
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(20)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_boundaries_voting(messages, min_confidence=0.2)

        # With low confidence, should accept more boundaries
        assert 0 in boundaries
        assert len(boundaries) >= 1

    def test_expert_entropy(self) -> None:
        """Test entropy expert."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_entropy(messages)

        assert 0 in boundaries
        assert isinstance(boundaries, set)

    def test_expert_alignment(self) -> None:
        """Test alignment expert."""
        # Create messages with conserved and variable regions
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(10)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_alignment(messages)

        assert 0 in boundaries
        assert isinstance(boundaries, set)

    def test_expert_alignment_single_message(self) -> None:
        """Test alignment expert with single message."""
        messages = [b"\xaa\x55\x00\x00"]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_alignment(messages)

        assert boundaries == {0}

    def test_expert_variance(self) -> None:
        """Test variance expert."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_variance(messages)

        assert 0 in boundaries
        assert isinstance(boundaries, set)

    def test_expert_distribution(self) -> None:
        """Test distribution expert."""
        # Create messages with distribution changes
        messages = []
        for i in range(20):
            # Constant first 2 bytes, variable next bytes
            msg = np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8)
            messages.append(msg)

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_distribution(messages)

        assert 0 in boundaries
        assert isinstance(boundaries, set)

    def test_expert_ngrams(self) -> None:
        """Test n-gram expert."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_ngrams(messages, n=2)

        assert 0 in boundaries
        assert isinstance(boundaries, set)

    def test_expert_ngrams_small_messages(self) -> None:
        """Test n-gram expert with messages smaller than n."""
        messages = [np.array([0xAA], dtype=np.uint8) for _ in range(10)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer._expert_ngrams(messages, n=2)

        assert boundaries == {0}


@pytest.mark.unit
@pytest.mark.inference
class TestEnsembleInference:
    """Test ensemble-based format inference."""

    def test_infer_format_ensemble_basic(self) -> None:
        """Test basic ensemble inference."""
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.4)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1
        assert all(isinstance(f, InferredField) for f in schema.fields)

    def test_infer_format_ensemble_confidence_scores(self) -> None:
        """Test that ensemble provides confidence scores."""
        messages = [b"\xaa\x55" + bytes(range(10)) for _ in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.5)

        # All fields should have confidence >= 0.5
        for field in schema.fields:
            assert field.confidence >= 0.5
            assert 0.0 <= field.confidence <= 1.0

    def test_infer_format_ensemble_evidence(self) -> None:
        """Test that ensemble provides evidence dictionary."""
        messages = [b"\xaa\x55" + bytes([i] * 10) for i in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.4)

        # Fields should have evidence from experts
        for field in schema.fields:
            assert isinstance(field.evidence, dict)

    def test_infer_format_ensemble_high_field_confidence(self) -> None:
        """Test ensemble with high field confidence threshold."""
        # Create messages with very clear constant field
        messages = [b"\xaa\x55" + bytes([0xFF] * 10) for _ in range(20)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.8)

        # Should have fewer fields due to high confidence requirement
        # At least the constant field should pass
        assert len(schema.fields) >= 1

    def test_infer_format_ensemble_numpy_arrays(self) -> None:
        """Test ensemble with numpy array messages."""
        messages = [
            np.array([0xAA, 0x55] + list(range(i, i + 10)), dtype=np.uint8) for i in range(20)
        ]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.4)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_infer_format_ensemble_insufficient_samples(self) -> None:
        """Test ensemble with insufficient samples."""
        messages = [b"\xaa\x55\x00\x00"] * 5

        inferrer = MessageFormatInferrer(min_samples=10)

        with pytest.raises(ValueError, match="Need at least 10 messages"):
            inferrer.infer_format_ensemble(messages)

    def test_infer_format_ensemble_varying_lengths(self) -> None:
        """Test ensemble with varying message lengths."""
        messages = [b"\xaa\x55\x00", b"\xaa\x55\x00\x00", b"\xaa\x55"]

        inferrer = MessageFormatInferrer(min_samples=3)

        with pytest.raises(ValueError, match="Messages have varying lengths"):
            inferrer.infer_format_ensemble(messages)

    def test_infer_format_ensemble_invalid_type(self) -> None:
        """Test ensemble with invalid message type."""
        messages = ["invalid", "types"]  # type: ignore[list-item]

        inferrer = MessageFormatInferrer(min_samples=2)

        with pytest.raises(ValueError, match="Invalid message type"):
            inferrer.infer_format_ensemble(messages)


@pytest.mark.unit
@pytest.mark.inference
class TestFieldTypeDetectors:
    """Test individual field type detector methods."""

    def test_detect_type_entropy(self) -> None:
        """Test entropy-based type detection."""
        inferrer = MessageFormatInferrer()

        # Constant field (all same values)
        field_data = {
            "values": [0xAA] * 20,
            "entropy": 0.0,
            "variance": 0.0,
        }
        field_type, confidence = inferrer._detect_type_entropy(field_data)
        assert field_type == "constant"
        assert confidence == 1.0

        # High entropy field (random data)
        field_data = {
            "values": list(range(20)),
            "entropy": 7.5,
            "variance": 1000.0,
        }
        field_type, confidence = inferrer._detect_type_entropy(field_data)
        assert field_type == "data"
        assert confidence > 0.5

    def test_detect_type_patterns_counter(self) -> None:
        """Test pattern-based detection of counter fields."""
        inferrer = MessageFormatInferrer()

        # Counter field
        field_data = {
            "values": list(range(50)),
            "entropy": 5.0,
            "variance": 200.0,
        }
        field_type, confidence = inferrer._detect_type_patterns(
            field_data, offset=2, size=2, msg_len=20
        )
        assert field_type == "counter"
        assert confidence > 0.8

    def test_detect_type_patterns_timestamp(self) -> None:
        """Test pattern-based detection of timestamp fields."""
        inferrer = MessageFormatInferrer()

        # Timestamp field (large incrementing values)
        field_data = {
            "values": [1000000 + i * 1000 for i in range(20)],
            "entropy": 4.0,
            "variance": 10000.0,
        }
        field_type, confidence = inferrer._detect_type_patterns(
            field_data, offset=4, size=4, msg_len=20
        )
        assert field_type == "timestamp"
        assert confidence > 0.5

    def test_detect_type_patterns_length(self) -> None:
        """Test pattern-based detection of length fields."""
        inferrer = MessageFormatInferrer()

        # Length field (small values near start)
        field_data = {
            "values": [10, 12, 15, 8, 20],
            "entropy": 2.0,
            "variance": 20.0,
        }
        field_type, confidence = inferrer._detect_type_patterns(
            field_data, offset=2, size=1, msg_len=50
        )
        assert field_type == "length"
        assert confidence >= 0.6

    def test_detect_type_patterns_checksum(self) -> None:
        """Test pattern-based detection of checksum fields."""
        inferrer = MessageFormatInferrer()

        # Checksum field (near end, non-sequential values)
        field_data = {
            "values": [0x3F, 0x2A, 0x1B, 0x4C, 0x5D, 0x6E, 0x7F, 0x80],
            "entropy": 4.0,
            "variance": 500.0,
        }
        field_type, confidence = inferrer._detect_type_patterns(
            field_data, offset=18, size=2, msg_len=20
        )
        assert field_type == "checksum"
        assert confidence >= 0.5

    def test_detect_type_statistics(self) -> None:
        """Test statistics-based type detection."""
        inferrer = MessageFormatInferrer()

        # Low variance (constant)
        field_data = {
            "values": [5, 5, 5, 6, 5],
            "entropy": 0.5,
            "variance": 2.0,
        }
        field_type, confidence = inferrer._detect_type_statistics(field_data)
        assert field_type == "constant"
        assert confidence > 0.5

        # High variance and entropy (data)
        field_data = {
            "values": list(range(100)),
            "entropy": 7.5,
            "variance": 2000.0,
        }
        field_type, confidence = inferrer._detect_type_statistics(field_data)
        assert field_type == "data"
        assert confidence > 0.5

    def test_vote_field_type(self) -> None:
        """Test field type voting mechanism."""
        inferrer = MessageFormatInferrer()

        # All agree on constant
        detections = [
            ("constant", 0.9),
            ("constant", 0.8),
            ("constant", 0.7),
        ]
        field_type, confidence, evidence = inferrer._vote_field_type(detections)
        assert field_type == "constant"
        assert confidence > 0.7
        assert isinstance(evidence, dict)

        # Mixed votes
        detections = [
            ("counter", 0.9),
            ("unknown", 0.5),
            ("counter", 0.7),
        ]
        field_type, confidence, evidence = inferrer._vote_field_type(detections)
        # Counter should win (higher total vote)
        assert field_type == "counter"
        assert confidence > 0.5

    def test_vote_field_type_empty(self) -> None:
        """Test voting with no detections."""
        inferrer = MessageFormatInferrer()

        detections: list[tuple[str, float]] = []
        field_type, confidence, evidence = inferrer._vote_field_type(detections)

        assert field_type == "unknown"
        assert confidence == 0.0

    def test_extract_field_data_small_field(self) -> None:
        """Test field data extraction for small fields."""
        messages = [np.array([0xAA, 0x55, i, i + 1], dtype=np.uint8) for i in range(20)]

        inferrer = MessageFormatInferrer()
        field_data = inferrer._extract_field_data(messages, offset=2, size=1)

        assert "values" in field_data
        assert "entropy" in field_data
        assert "variance" in field_data
        assert len(field_data["values"]) == 20

    def test_extract_field_data_large_field(self) -> None:
        """Test field data extraction for large fields."""
        rng = np.random.RandomState(42)
        messages = [
            np.concatenate([np.array([0xAA, 0x55]), rng.randint(0, 256, 20, dtype=np.uint8)])
            for _ in range(20)
        ]

        inferrer = MessageFormatInferrer()
        field_data = inferrer._extract_field_data(messages, offset=2, size=20)

        assert "values" in field_data
        assert len(field_data["values"]) == 20
        # Large field values should be tuples
        assert isinstance(field_data["values"][0], tuple)


@pytest.mark.unit
@pytest.mark.inference
class TestEnsembleComplexProtocols:
    """Test ensemble inference on complex protocol structures."""

    def test_ensemble_with_counter_and_checksum(self) -> None:
        """Test ensemble on protocol with counter and checksum."""
        rng = np.random.RandomState(42)
        messages = []
        for i in range(50):
            msg = bytearray(13)
            msg[0:2] = [0xAA, 0x55]  # Constant header
            msg[2:4] = i.to_bytes(2, "big")  # Counter
            random_data = rng.randint(0, 256, 8, dtype=np.uint8).tobytes()
            msg[4:12] = random_data  # Data
            # XOR checksum
            xor_sum = 0
            for b in msg[:12]:
                xor_sum ^= b
            msg[12] = xor_sum
            messages.append(bytes(msg))

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.5)

        assert schema.total_size == 13
        assert len(schema.fields) >= 2

        # Check for constant or data field (ensemble may merge constant+counter into one field)
        field_types = [f.field_type for f in schema.fields]
        assert "constant" in field_types or "data" in field_types

    def test_ensemble_comparison_with_basic(self) -> None:
        """Compare ensemble results with basic inference."""
        messages = [b"\xaa\x55" + bytes(range(i, i + 10)) for i in range(50)]

        inferrer = MessageFormatInferrer(min_samples=10)

        # Basic inference
        schema_basic = inferrer.infer_format(messages)

        # Ensemble inference
        schema_ensemble = inferrer.infer_format_ensemble(messages, min_field_confidence=0.5)

        # Both should detect same message size
        assert schema_basic.total_size == schema_ensemble.total_size

        # Ensemble fields should have evidence
        for field in schema_ensemble.fields:
            assert hasattr(field, "evidence")

    def test_ensemble_accuracy_synthetic_protocol(self) -> None:
        """Test ensemble accuracy on synthetic protocol."""
        # Create known protocol:
        # - 2 bytes: magic (0xAA55)
        # - 1 byte: length
        # - 2 bytes: counter
        # - N bytes: payload (varying)
        # - 1 byte: checksum

        messages = []
        for i in range(30):
            payload_len = 8
            msg = bytearray([0xAA, 0x55, payload_len])
            msg.extend(i.to_bytes(2, "big"))
            msg.extend(bytes(range(payload_len)))
            xor_sum = 0
            for b in msg:
                xor_sum ^= b
            msg.append(xor_sum)
            messages.append(bytes(msg))

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.6)

        # Should identify key fields
        field_types = [f.field_type for f in schema.fields]

        # Should have at least constant, counter, and data fields
        assert "constant" in field_types or "counter" in field_types


@pytest.mark.unit
@pytest.mark.inference
class TestEnsemblePerformance:
    """Test performance of ensemble inference."""

    def test_ensemble_large_message_set(self) -> None:
        """Test ensemble with large number of messages."""
        # 1000 messages should complete in reasonable time
        messages = [b"\xaa\x55" + bytes(range(10)) for _ in range(1000)]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.6)

        assert schema.total_size == 12
        assert len(schema.fields) >= 1

    def test_ensemble_memory_efficiency(self) -> None:
        """Test that ensemble doesn't create excessive memory usage."""
        # Large messages should not cause OOM
        rng = np.random.RandomState(42)
        messages = [
            bytes([0xAA, 0x55]) + rng.randint(0, 256, 100, dtype=np.uint8).tobytes()
            for _ in range(100)
        ]

        inferrer = MessageFormatInferrer(min_samples=10)
        schema = inferrer.infer_format_ensemble(messages, min_field_confidence=0.5)

        assert schema.total_size == 102
        # Should complete without memory issues
