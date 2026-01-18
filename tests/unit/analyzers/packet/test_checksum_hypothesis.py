"""Property-based tests for checksum algorithms."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import checksum_data

pytestmark = [pytest.mark.unit, pytest.mark.packet, pytest.mark.hypothesis]


class TestChecksumProperties:
    """Property-based tests for checksums."""

    @given(data=checksum_data())
    @settings(max_examples=100, deadline=None)
    def test_checksum_deterministic(self, data: bytes) -> None:
        """Property: Same data always produces same checksum."""
        # Simple XOR checksum
        checksum1 = 0
        for byte_val in data:
            checksum1 ^= byte_val

        checksum2 = 0
        for byte_val in data:
            checksum2 ^= byte_val

        assert checksum1 == checksum2

    @given(data=checksum_data())
    @settings(max_examples=50, deadline=None)
    def test_checksum_changes_on_modification(self, data: bytes) -> None:
        """Property: Modifying data typically changes checksum."""
        if len(data) < 2:
            pytest.skip("Data too short")

        # Original checksum
        original_checksum = 0
        for byte_val in data:
            original_checksum ^= byte_val

        # Modify one byte
        modified_data = bytearray(data)
        modified_data[0] ^= 0xFF

        # Modified checksum
        modified_checksum = 0
        for byte_val in modified_data:
            modified_checksum ^= byte_val

        # Checksums should differ
        assert original_checksum != modified_checksum

    @given(
        data=checksum_data(),
        byte_position=st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=50, deadline=None)
    def test_single_bit_change_detected(self, data: bytes, byte_position: int) -> None:
        """Property: Single bit change is detected by checksum."""
        if len(data) == 0:
            pytest.skip("Empty data")

        position = byte_position % len(data)

        # Original checksum
        original = sum(data) & 0xFFFF

        # Flip one bit
        modified_data = bytearray(data)
        modified_data[position] ^= 0x01

        # Modified checksum
        modified = sum(modified_data) & 0xFFFF

        # Should be different
        assert original != modified
