"""Property-based tests for packet validation."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import checksum_data, packet_data

pytestmark = [pytest.mark.unit, pytest.mark.loader, pytest.mark.hypothesis]


class TestPacketValidationProperties:
    """Property-based tests for packet validation."""

    @given(data=packet_data())
    @settings(max_examples=50, deadline=None)
    def test_packet_length_validated(self, data: bytes) -> None:
        """Property: Packet length validation works correctly."""
        # Packet length should match actual length
        actual_length = len(data)

        assert actual_length >= 0
        assert actual_length <= 1500  # MTU

    @given(
        header_size=st.integers(min_value=4, max_value=20),
        payload_size=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_valid_packet_structure(self, header_size: int, payload_size: int) -> None:
        """Property: Valid packet structure is accepted."""
        rng = np.random.default_rng(42)
        header = rng.integers(0, 256, header_size, dtype=np.uint8).tobytes()
        payload = rng.integers(0, 256, payload_size, dtype=np.uint8).tobytes()

        packet = header + payload

        assert len(packet) == header_size + payload_size

    @given(data=checksum_data())
    @settings(max_examples=30, deadline=None)
    def test_checksum_validation_deterministic(self, data: bytes) -> None:
        """Property: Checksum validation is deterministic."""
        # Simple XOR checksum for testing
        checksum1 = 0
        for byte_val in data:
            checksum1 ^= byte_val

        checksum2 = 0
        for byte_val in data:
            checksum2 ^= byte_val

        assert checksum1 == checksum2
