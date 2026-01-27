"""Tests for utils/bitwise.py - Bitwise operation utilities.

Tests:
- bits_to_byte: Convert up to 8 bits to byte value
- bits_to_value: Convert arbitrary number of bits to integer
- LSB first vs MSB first bit ordering
- Edge cases and error handling
"""

import pytest

from oscura.utils import bitwise


class TestBitsToByte:
    """Test bits_to_byte function."""

    def test_lsb_first(self) -> None:
        """Test LSB first bit ordering."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        result = bitwise.bits_to_byte(bits, lsb_first=True)

        # Binary: 10101010 (LSB first) = 85
        assert result == 85

    def test_msb_first(self) -> None:
        """Test MSB first bit ordering."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        result = bitwise.bits_to_byte(bits, lsb_first=False)

        # Binary: 10101010 (MSB first) = 170
        assert result == 170

    def test_all_zeros(self) -> None:
        """Test all zeros."""
        bits = [0, 0, 0, 0, 0, 0, 0, 0]
        result = bitwise.bits_to_byte(bits)

        assert result == 0

    def test_all_ones(self) -> None:
        """Test all ones."""
        bits = [1, 1, 1, 1, 1, 1, 1, 1]
        result = bitwise.bits_to_byte(bits)

        assert result == 255

    def test_fewer_than_8_bits(self) -> None:
        """Test with fewer than 8 bits."""
        bits = [1, 1, 1, 1]
        result = bitwise.bits_to_byte(bits, lsb_first=True)

        # 0b1111 = 15
        assert result == 15

    def test_more_than_8_bits(self) -> None:
        """Test with more than 8 bits (should use only first 8)."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
        result = bitwise.bits_to_byte(bits, lsb_first=True)

        # Only first 8 bits: 10101010 = 85
        assert result == 85

    def test_single_bit(self) -> None:
        """Test with single bit."""
        result1 = bitwise.bits_to_byte([0])
        result2 = bitwise.bits_to_byte([1])

        assert result1 == 0
        assert result2 == 1

    def test_empty_bits(self) -> None:
        """Test with empty bit list."""
        result = bitwise.bits_to_byte([])

        assert result == 0

    def test_invalid_bit_value(self) -> None:
        """Test error on invalid bit values."""
        with pytest.raises(ValueError, match="All bits must be 0 or 1"):
            bitwise.bits_to_byte([1, 0, 2, 1])

    def test_negative_bit_value(self) -> None:
        """Test error on negative bit values."""
        with pytest.raises(ValueError, match="All bits must be 0 or 1"):
            bitwise.bits_to_byte([1, 0, -1, 1])

    def test_known_ascii_values(self) -> None:
        """Test known ASCII values."""
        # ASCII 'A' = 65 = 0b01000001 (MSB first)
        bits_msb = [0, 1, 0, 0, 0, 0, 0, 1]
        result = bitwise.bits_to_byte(bits_msb, lsb_first=False)
        assert result == 65

        # Same as LSB first: reverse bits
        bits_lsb = [1, 0, 0, 0, 0, 0, 1, 0]
        result2 = bitwise.bits_to_byte(bits_lsb, lsb_first=True)
        assert result2 == 65


class TestBitsToValue:
    """Test bits_to_value function."""

    def test_8_bits_lsb(self) -> None:
        """Test 8 bits LSB first."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 85  # 0b01010101

    def test_8_bits_msb(self) -> None:
        """Test 8 bits MSB first."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        result = bitwise.bits_to_value(bits, lsb_first=False)

        assert result == 170  # 0b10101010

    def test_16_bits(self) -> None:
        """Test 16 bits."""
        # All ones for 16 bits
        bits = [1] * 16
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 65535  # 2^16 - 1

    def test_10_bits(self) -> None:
        """Test 10 bits."""
        bits = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 1023  # 2^10 - 1

    def test_4_bits_msb(self) -> None:
        """Test 4 bits MSB first."""
        bits = [1, 0, 1, 0]
        result = bitwise.bits_to_value(bits, lsb_first=False)

        assert result == 10  # 0b1010

    def test_4_bits_lsb(self) -> None:
        """Test 4 bits LSB first."""
        bits = [1, 0, 1, 0]
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 5  # 0b0101

    def test_single_bit(self) -> None:
        """Test single bit."""
        assert bitwise.bits_to_value([0]) == 0
        assert bitwise.bits_to_value([1]) == 1

    def test_empty_bits(self) -> None:
        """Test empty bit list."""
        result = bitwise.bits_to_value([])

        assert result == 0

    def test_all_zeros_16bit(self) -> None:
        """Test all zeros for 16 bits."""
        bits = [0] * 16
        result = bitwise.bits_to_value(bits)

        assert result == 0

    def test_alternating_pattern(self) -> None:
        """Test alternating bit pattern."""
        # 0b10101010 (8 bits MSB)
        bits = [1, 0, 1, 0, 1, 0, 1, 0]
        result_msb = bitwise.bits_to_value(bits, lsb_first=False)
        result_lsb = bitwise.bits_to_value(bits, lsb_first=True)

        assert result_msb == 170
        assert result_lsb == 85

    def test_invalid_bit_value(self) -> None:
        """Test error on invalid bit values."""
        with pytest.raises(ValueError, match="All bits must be 0 or 1"):
            bitwise.bits_to_value([1, 0, 5, 1])

    def test_float_bit_value(self) -> None:
        """Test error on float bit values."""
        with pytest.raises(ValueError):
            bitwise.bits_to_value([1, 0, 0.5, 1])  # type: ignore

    def test_large_number_of_bits(self) -> None:
        """Test with large number of bits."""
        # 32 bits all ones
        bits = [1] * 32
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 2**32 - 1

    def test_specific_value_msb(self) -> None:
        """Test specific known value MSB first."""
        # Decimal 42 = 0b00101010 (8 bits)
        bits = [0, 0, 1, 0, 1, 0, 1, 0]
        result = bitwise.bits_to_value(bits, lsb_first=False)

        assert result == 42

    def test_specific_value_lsb(self) -> None:
        """Test specific known value LSB first."""
        # Decimal 42 = 0b00101010 (8 bits)
        # Reversed for LSB: 01010100
        bits = [0, 1, 0, 1, 0, 1, 0, 0]
        result = bitwise.bits_to_value(bits, lsb_first=True)

        assert result == 42


class TestConsistency:
    """Test consistency between bits_to_byte and bits_to_value."""

    def test_8bit_consistency_lsb(self) -> None:
        """Test that both functions give same result for 8 bits LSB."""
        bits = [1, 1, 0, 0, 1, 1, 0, 0]

        byte_result = bitwise.bits_to_byte(bits, lsb_first=True)
        value_result = bitwise.bits_to_value(bits, lsb_first=True)

        assert byte_result == value_result

    def test_8bit_consistency_msb(self) -> None:
        """Test that both functions give same result for 8 bits MSB."""
        bits = [1, 0, 1, 1, 0, 1, 0, 1]

        byte_result = bitwise.bits_to_byte(bits, lsb_first=False)
        value_result = bitwise.bits_to_value(bits, lsb_first=False)

        assert byte_result == value_result

    def test_fewer_than_8bits_consistency(self) -> None:
        """Test consistency for fewer than 8 bits."""
        bits = [1, 1, 1, 0]

        byte_result = bitwise.bits_to_byte(bits, lsb_first=True)
        value_result = bitwise.bits_to_value(bits, lsb_first=True)

        assert byte_result == value_result


class TestRoundTrip:
    """Test round-trip conversions."""

    def test_byte_to_bits_to_byte(self) -> None:
        """Test converting byte to bits and back."""
        original_value = 123

        # Convert to bits (MSB first)
        bits = [(original_value >> i) & 1 for i in range(7, -1, -1)]

        # Convert back
        result = bitwise.bits_to_value(bits, lsb_first=False)

        assert result == original_value

    def test_value_to_bits_to_value_16bit(self) -> None:
        """Test converting 16-bit value to bits and back."""
        original_value = 12345

        # Convert to bits (MSB first)
        bits = [(original_value >> i) & 1 for i in range(15, -1, -1)]

        # Convert back
        result = bitwise.bits_to_value(bits, lsb_first=False)

        assert result == original_value


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_8_bits(self) -> None:
        """Test exactly 8 bits in both functions."""
        bits = [1, 0, 1, 1, 0, 0, 1, 1]

        byte_result = bitwise.bits_to_byte(bits)
        value_result = bitwise.bits_to_value(bits)

        assert byte_result == value_result
        assert byte_result == 205  # 0b11001101 LSB

    def test_power_of_two_values(self) -> None:
        """Test powers of two."""
        for i in range(8):
            # Single bit set
            bits = [0] * 8
            bits[i] = 1

            result = bitwise.bits_to_value(bits, lsb_first=True)
            assert result == 2**i

    def test_max_byte_value(self) -> None:
        """Test maximum byte value (255)."""
        bits = [1, 1, 1, 1, 1, 1, 1, 1]

        byte_result = bitwise.bits_to_byte(bits)
        value_result = bitwise.bits_to_value(bits)

        assert byte_result == 255
        assert value_result == 255
