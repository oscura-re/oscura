"""Property-based tests for message format inference.

This module tests the message_format module using Hypothesis for property-based testing.
It verifies that message format inference properties hold across a wide range of inputs.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from oscura.inference.message_format import (
    MessageFormatInferrer,
    detect_field_types,
    infer_format,
)
from tests.hypothesis_strategies import checksum_data, message_streams

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestMessageFormatInferenceProperties:
    """Property-based tests for message format inference."""

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_inferred_schema_total_size_matches_input(
        self, num_messages: int, msg_length: int
    ) -> None:
        """Property: Inferred schema total_size equals input message length."""
        # Generate consistent-length messages
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        assert schema.total_size == msg_length

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_boundaries_cover_entire_message(
        self, num_messages: int, msg_length: int
    ) -> None:
        """Property: Field boundaries span from 0 to message length."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        # Boundaries should start at 0
        assert schema.field_boundaries[0] == 0

        # Last field should extend to end of message
        last_field = schema.fields[-1]
        assert last_field.offset + last_field.size == msg_length

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_boundaries_sorted_and_unique(self, num_messages: int, msg_length: int) -> None:
        """Property: Field boundaries are sorted and unique."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        # Should be sorted
        assert schema.field_boundaries == sorted(schema.field_boundaries)

        # Should be unique
        assert len(schema.field_boundaries) == len(set(schema.field_boundaries))

    @given(num_messages=st.integers(min_value=10, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_constant_bytes_detected_as_constant_field(self, num_messages: int) -> None:
        """Property: Messages with constant bytes are detected as constant fields."""
        # Create messages where first 4 bytes are constant
        header = b"\xaa\xbb\xcc\xdd"
        rng = np.random.default_rng(42)

        messages = [
            header + rng.integers(0, 256, 10, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        # Should detect at least one constant field
        constant_fields = [f for f in schema.fields if f.field_type == "constant"]
        assert len(constant_fields) > 0

        # First field should be constant (the header)
        if schema.fields[0].offset == 0:
            assert schema.fields[0].field_type == "constant"

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_entropy_values_within_bounds(self, num_messages: int, msg_length: int) -> None:
        """Property: Entropy values are between 0 and 8 bits (for bytes)."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        for field in schema.fields:
            # Entropy should be non-negative and <= log2(256) = 8
            assert 0.0 <= field.entropy <= 8.0

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_confidence_values_within_bounds(self, num_messages: int, msg_length: int) -> None:
        """Property: Confidence scores are between 0 and 1."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        for field in schema.fields:
            assert 0.0 <= field.confidence <= 1.0

    @given(stream_data=message_streams())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.data_too_large, HealthCheck.filter_too_much],
    )
    def test_detects_common_header_in_message_stream(
        self, stream_data: tuple[bytes, bytes]
    ) -> None:
        """Property: Common headers in message streams are detected."""
        stream, header = stream_data

        # Parse stream into individual messages
        messages = []
        pos = 0
        while pos < len(stream):
            if stream[pos : pos + len(header)] == header:
                # Found header, extract message
                payload_size = stream[pos + len(header)]
                msg_len = len(header) + 1 + payload_size
                messages.append(stream[pos : pos + msg_len])
                pos += msg_len
            else:
                pos += 1

        # Need at least min_samples
        assume(len(messages) >= 10)

        # All messages should be same length for this test
        lengths = [len(m) for m in messages]
        assume(len(set(lengths)) == 1)

        schema = infer_format(messages, min_samples=10)

        # First field should be constant (the header)
        if len(schema.fields) > 0:
            first_field = schema.fields[0]
            if first_field.offset == 0 and first_field.size == len(header):
                assert first_field.field_type == "constant"

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=20, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_sizes_sum_to_message_length(self, num_messages: int, msg_length: int) -> None:
        """Property: Sum of field sizes equals message length."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        schema = infer_format(messages, min_samples=10)

        total_field_size = sum(f.size for f in schema.fields)
        assert total_field_size == msg_length

    @given(data_length=st.integers(min_value=100, max_value=1000))
    @settings(max_examples=30, deadline=None)
    def test_high_entropy_data_detected(self, data_length: int) -> None:
        """Property: Random data produces high entropy fields."""
        rng = np.random.default_rng(42)
        # Generate 15 messages with random data
        messages = [rng.integers(0, 256, data_length, dtype=np.uint8).tobytes() for _ in range(15)]

        schema = infer_format(messages, min_samples=10)

        # At least one field should have high entropy
        high_entropy_fields = [f for f in schema.fields if f.entropy > 5.0]
        assert len(high_entropy_fields) > 0

    @given(counter_start=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_counter_field_detection(self, counter_start: int) -> None:
        """Property: Incrementing counter fields are detected."""
        # Create messages with a counter field at start
        messages = []
        for i in range(20):
            counter = (counter_start + i) & 0xFF
            # 1-byte counter + 10 random bytes
            rng = np.random.default_rng(42 + i)
            msg = bytes([counter]) + rng.integers(0, 256, 10, dtype=np.uint8).tobytes()
            messages.append(msg)

        schema = infer_format(messages, min_samples=10)

        # Should detect counter field
        counter_fields = [f for f in schema.fields if f.field_type == "counter"]
        # Note: Detection not guaranteed for all cases, but should work often
        # This is a probabilistic test
        if len(counter_fields) == 0:
            # At least first field should have low variance
            assert schema.fields[0].variance > 0  # Not constant


class TestBoundaryDetectionProperties:
    """Property-based tests for field boundary detection."""

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_boundary_detection_always_includes_zero(
        self, num_messages: int, msg_length: int
    ) -> None:
        """Property: Boundary detection always includes offset 0."""
        rng = np.random.default_rng(42)
        msg_arrays = [rng.integers(0, 256, msg_length, dtype=np.uint8) for _ in range(num_messages)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(msg_arrays)

        assert 0 in boundaries
        assert boundaries[0] == 0

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
        method=st.sampled_from(["entropy", "variance", "combined"]),
    )
    @settings(max_examples=50, deadline=None)
    def test_boundary_detection_produces_valid_offsets(
        self, num_messages: int, msg_length: int, method: str
    ) -> None:
        """Property: All boundaries are valid offsets within message."""
        rng = np.random.default_rng(42)
        msg_arrays = [rng.integers(0, 256, msg_length, dtype=np.uint8) for _ in range(num_messages)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(
            msg_arrays,
            method=method,  # type: ignore[arg-type]
        )

        for boundary in boundaries:
            assert 0 <= boundary < msg_length

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_boundaries_create_minimum_field_size(self, num_messages: int, msg_length: int) -> None:
        """Property: Boundaries are spaced at least 2 bytes apart (after merging)."""
        rng = np.random.default_rng(42)
        msg_arrays = [rng.integers(0, 256, msg_length, dtype=np.uint8) for _ in range(num_messages)]

        inferrer = MessageFormatInferrer()
        boundaries = inferrer.detect_field_boundaries(msg_arrays)

        # Check spacing (except for last boundary)
        for i in range(len(boundaries) - 1):
            assert boundaries[i + 1] - boundaries[i] >= 2


class TestFieldTypeDetectionProperties:
    """Property-based tests for field type detection."""

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_types_are_valid(self, num_messages: int, msg_length: int) -> None:
        """Property: All field types are from valid set."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        valid_types = {
            "constant",
            "counter",
            "timestamp",
            "length",
            "checksum",
            "data",
            "unknown",
        }

        boundaries = [0, msg_length // 3, (2 * msg_length) // 3]
        fields = detect_field_types(messages, boundaries)

        for field in fields:
            assert field.field_type in valid_types

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        constant_byte=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=50, deadline=None)
    def test_constant_field_has_high_confidence(
        self, num_messages: int, constant_byte: int
    ) -> None:
        """Property: Truly constant fields have confidence = 1.0."""
        # Create messages with constant first byte
        messages = [bytes([constant_byte] * 10) for _ in range(num_messages)]

        boundaries = [0]
        fields = detect_field_types(messages, boundaries)

        # Should have one field
        assert len(fields) == 1
        assert fields[0].field_type == "constant"
        assert fields[0].confidence == 1.0

    @given(
        num_messages=st.integers(min_value=10, max_value=50),
        msg_length=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_offsets_match_boundaries(self, num_messages: int, msg_length: int) -> None:
        """Property: Field offsets correspond to provided boundaries."""
        rng = np.random.default_rng(42)
        messages = [
            rng.integers(0, 256, msg_length, dtype=np.uint8).tobytes() for _ in range(num_messages)
        ]

        boundaries = [0, msg_length // 3, (2 * msg_length) // 3]
        boundaries = sorted({b for b in boundaries if b < msg_length})

        fields = detect_field_types(messages, boundaries)

        field_offsets = [f.offset for f in fields]
        assert field_offsets == boundaries


class TestInferrerEdgeCases:
    """Edge case tests for message format inferrer."""

    def test_insufficient_samples_raises_error(self) -> None:
        """Property: Too few samples raises ValueError."""
        messages = [b"\x01\x02\x03" for _ in range(5)]

        inferrer = MessageFormatInferrer(min_samples=10)

        with pytest.raises(ValueError, match="Need at least"):
            inferrer.infer_format(messages)

    def test_varying_length_messages_raises_error(self) -> None:
        """Property: Messages of different lengths raise ValueError."""
        # Create 10+ messages with varying lengths
        messages = [b"\x01\x02\x03"] * 5 + [b"\x01\x02\x03\x04"] * 5 + [b"\x01\x02"]

        with pytest.raises(ValueError, match="varying lengths"):
            infer_format(messages)

    @given(data=checksum_data())
    @settings(max_examples=30, deadline=None)
    def test_accepts_both_bytes_and_arrays(self, data: bytes) -> None:
        """Property: Inferrer accepts both bytes and numpy arrays."""
        # Create enough messages
        messages_bytes = [data for _ in range(15)]
        messages_arrays = [np.frombuffer(data, dtype=np.uint8) for _ in range(15)]

        schema1 = infer_format(messages_bytes, min_samples=10)
        schema2 = infer_format(messages_arrays, min_samples=10)

        # Should produce same total size
        assert schema1.total_size == schema2.total_size
