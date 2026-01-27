"""Tests for checksum and CRC field detection and identification.

Tests the checksum/CRC detection framework (SEA-004).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistical.checksum import (
    ChecksumCandidate,
    ChecksumMatch,
    compute_checksum,
    crc8,
    crc16_ccitt,
    crc16_ibm,
    crc32,
    detect_checksum_fields,
    identify_checksum_algorithm,
    sum8,
    sum16,
    verify_checksums,
    xor_checksum,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


class TestBasicChecksums:
    """Test basic checksum algorithms."""

    def test_xor_checksum_simple(self) -> None:
        """Test XOR checksum with simple data."""
        result = xor_checksum(b"ABC")
        # A=0x41, B=0x42, C=0x43
        # 0x41 ^ 0x42 = 0x03, 0x03 ^ 0x43 = 0x40 (64)
        assert result == 64

    def test_xor_checksum_empty(self) -> None:
        """Test XOR checksum with empty data."""
        result = xor_checksum(b"")
        assert result == 0

    def test_xor_checksum_single_byte(self) -> None:
        """Test XOR checksum with single byte."""
        result = xor_checksum(b"\xff")
        assert result == 0xFF

    def test_sum8_simple(self) -> None:
        """Test 8-bit sum checksum."""
        result = sum8(b"ABC")
        # A=0x41(65), B=0x42(66), C=0x43(67) = 198
        assert result == 198

    def test_sum8_overflow(self) -> None:
        """Test 8-bit sum with overflow."""
        result = sum8(b"\xff\xff")
        # 255 + 255 = 510, modulo 256 = 254
        assert result == 254

    def test_sum8_empty(self) -> None:
        """Test 8-bit sum with empty data."""
        result = sum8(b"")
        assert result == 0

    def test_sum16_big_endian(self) -> None:
        """Test 16-bit sum with big endian."""
        result = sum16(b"ABCD", endian="big")
        # AB = 0x4142 = 16706
        # CD = 0x4344 = 17220
        # Total = 33926, modulo 65536 = 33926
        assert result == 33926

    def test_sum16_little_endian(self) -> None:
        """Test 16-bit sum with little endian."""
        result = sum16(b"ABCD", endian="little")
        # BA = 0x4241 = 16961
        # DC = 0x4443 = 17475
        # Total = 34436
        assert result == 34436

    def test_sum16_odd_length_big(self) -> None:
        """Test 16-bit sum with odd length data (big endian)."""
        result = sum16(b"ABC", endian="big")
        # AB = 0x4142 = 16706
        # C0 = 0x4300 = 17152 (padded)
        assert result == 33858

    def test_sum16_odd_length_little(self) -> None:
        """Test 16-bit sum with odd length data (little endian)."""
        result = sum16(b"ABC", endian="little")
        # BA = 0x4241 = 16961
        # 0C = 0x0043 = 67 (padded)
        assert result == 17028

    def test_sum16_empty(self) -> None:
        """Test 16-bit sum with empty data."""
        result = sum16(b"", endian="big")
        assert result == 0


class TestCRCAlgorithms:
    """Test CRC algorithms."""

    def test_crc8_default(self) -> None:
        """Test CRC-8 with default parameters."""
        result = crc8(b"123456789")
        # Standard CRC-8 test vector
        assert result == 244

    def test_crc8_custom_poly(self) -> None:
        """Test CRC-8 with custom polynomial."""
        result = crc8(b"test", poly=0x31)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_crc8_custom_init(self) -> None:
        """Test CRC-8 with custom initial value."""
        result = crc8(b"test", init=0xFF)
        assert isinstance(result, int)
        assert 0 <= result <= 255

    def test_crc8_empty(self) -> None:
        """Test CRC-8 with empty data."""
        result = crc8(b"")
        assert result == 0

    def test_crc16_ccitt_default(self) -> None:
        """Test CRC-16-CCITT with default init."""
        result = crc16_ccitt(b"123456789")
        # Standard CRC-16-CCITT test vector with init=0xFFFF
        assert result == 10673

    def test_crc16_ccitt_init_zero(self) -> None:
        """Test CRC-16-CCITT with init=0x0000."""
        result = crc16_ccitt(b"123456789", init=0x0000)
        # Different from default due to different init
        assert result != 10673
        assert 0 <= result <= 65535

    def test_crc16_ccitt_empty(self) -> None:
        """Test CRC-16-CCITT with empty data."""
        result = crc16_ccitt(b"", init=0xFFFF)
        assert result == 0xFFFF

    def test_crc16_ibm_default(self) -> None:
        """Test CRC-16-IBM with default init."""
        result = crc16_ibm(b"123456789")
        # Standard CRC-16-IBM test vector
        assert result == 47933

    def test_crc16_ibm_custom_init(self) -> None:
        """Test CRC-16-IBM with custom init."""
        result = crc16_ibm(b"test", init=0xFFFF)
        assert isinstance(result, int)
        assert 0 <= result <= 65535

    def test_crc16_ibm_empty(self) -> None:
        """Test CRC-16-IBM with empty data."""
        result = crc16_ibm(b"")
        assert result == 0

    def test_crc32_default(self) -> None:
        """Test CRC-32 with standard test vector."""
        result = crc32(b"123456789")
        # Standard CRC-32 test vector
        assert result == 3421780262

    def test_crc32_empty(self) -> None:
        """Test CRC-32 with empty data."""
        result = crc32(b"")
        # Empty data returns 0 (init value XORed with itself)
        assert result == 0

    def test_crc32_single_byte(self) -> None:
        """Test CRC-32 with single byte."""
        result = crc32(b"\x00")
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF


class TestComputeChecksum:
    """Test unified checksum computation function."""

    def test_compute_xor(self) -> None:
        """Test compute_checksum with XOR algorithm."""
        result = compute_checksum(b"ABC", "xor")
        assert result == xor_checksum(b"ABC")

    def test_compute_sum8(self) -> None:
        """Test compute_checksum with sum8 algorithm."""
        result = compute_checksum(b"ABC", "sum8")
        assert result == sum8(b"ABC")

    def test_compute_sum16_big(self) -> None:
        """Test compute_checksum with sum16_big algorithm."""
        result = compute_checksum(b"ABCD", "sum16_big")
        assert result == sum16(b"ABCD", endian="big")

    def test_compute_sum16_little(self) -> None:
        """Test compute_checksum with sum16_little algorithm."""
        result = compute_checksum(b"ABCD", "sum16_little")
        assert result == sum16(b"ABCD", endian="little")

    def test_compute_crc8(self) -> None:
        """Test compute_checksum with CRC-8 algorithm."""
        result = compute_checksum(b"123456789", "crc8")
        assert result == crc8(b"123456789")

    def test_compute_crc8_with_params(self) -> None:
        """Test compute_checksum with CRC-8 and custom parameters."""
        result = compute_checksum(b"test", "crc8", poly=0x31, init=0xFF)
        assert result == crc8(b"test", poly=0x31, init=0xFF)

    def test_compute_crc16_ccitt(self) -> None:
        """Test compute_checksum with CRC-16-CCITT algorithm."""
        result = compute_checksum(b"123456789", "crc16_ccitt")
        assert result == crc16_ccitt(b"123456789")

    def test_compute_crc16_ccitt_custom_init(self) -> None:
        """Test compute_checksum with CRC-16-CCITT and custom init."""
        result = compute_checksum(b"test", "crc16_ccitt", init=0x0000)
        assert result == crc16_ccitt(b"test", init=0x0000)

    def test_compute_crc16_ibm(self) -> None:
        """Test compute_checksum with CRC-16-IBM algorithm."""
        result = compute_checksum(b"123456789", "crc16_ibm")
        assert result == crc16_ibm(b"123456789")

    def test_compute_crc32(self) -> None:
        """Test compute_checksum with CRC-32 algorithm."""
        result = compute_checksum(b"123456789", "crc32")
        assert result == crc32(b"123456789")

    def test_compute_unknown_algorithm(self) -> None:
        """Test compute_checksum with unknown algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown checksum algorithm"):
            compute_checksum(b"test", "invalid_algo")


class TestDetectChecksumFields:
    """Test checksum field detection."""

    def test_detect_empty_messages(self) -> None:
        """Test detection with empty message list."""
        candidates = detect_checksum_fields([])
        assert candidates == []

    def test_detect_short_messages(self) -> None:
        """Test detection with messages too short."""
        candidates = detect_checksum_fields([b"A"])
        assert candidates == []

    def test_detect_xor_checksum_header(self) -> None:
        """Test detection of XOR checksum in header."""
        # Messages with XOR checksum at offset 0
        messages = [
            bytes([xor_checksum(b"DATA1")]) + b"DATA1",
            bytes([xor_checksum(b"DATA2")]) + b"DATA2",
            bytes([xor_checksum(b"DATA3")]) + b"DATA3",
        ]
        candidates = detect_checksum_fields(messages)

        # Should detect offset 0 as candidate
        assert len(candidates) > 0
        offsets = [c.offset for c in candidates]
        assert 0 in offsets

    def test_detect_xor_checksum_trailer(self) -> None:
        """Test detection of XOR checksum in trailer."""
        # Messages with XOR checksum at end
        messages = [
            b"DATA1" + bytes([xor_checksum(b"DATA1")]),
            b"DATA2" + bytes([xor_checksum(b"DATA2")]),
            b"DATA3" + bytes([xor_checksum(b"DATA3")]),
        ]
        candidates = detect_checksum_fields(messages)

        # Should detect trailer position
        assert len(candidates) > 0
        positions = [c.position for c in candidates]
        assert "trailer" in positions

    def test_detect_with_numpy_array(self) -> None:
        """Test detection with numpy array input."""
        msg1 = np.array([0x41, 0x42, 0x43], dtype=np.uint8)
        msg2 = np.array([0x44, 0x45, 0x46], dtype=np.uint8)
        candidates = detect_checksum_fields([msg1, msg2])

        # Should handle numpy arrays
        assert isinstance(candidates, list)

    def test_detect_with_candidate_offsets(self) -> None:
        """Test detection with specific candidate offsets."""
        messages = [b"\x00\x00DATA", b"\x01\x00DATA", b"\x02\x00DATA"]
        candidates = detect_checksum_fields(messages, candidate_offsets=[0, 1])

        # Should only check specified offsets
        offsets = [c.offset for c in candidates]
        for offset in offsets:
            assert offset in [0, 1]

    def test_detect_correlation_threshold(self) -> None:
        """Test that low correlation candidates are filtered out."""
        # Messages with constant field (no correlation)
        messages = [b"\x00DATA1", b"\x00DATA2", b"\x00DATA3"]
        candidates = detect_checksum_fields(messages)

        # Constant field should have low correlation and be filtered
        # Only fields that vary with content should be detected
        for candidate in candidates:
            assert candidate.correlation >= 0.3

    def test_detect_multi_size_fields(self) -> None:
        """Test detection of fields with different sizes."""
        # Create messages with 2-byte checksum
        messages = []
        for i in range(5):
            data = f"DATA{i}".encode()
            cksum = sum16(data, endian="big")
            msg = cksum.to_bytes(2, "big") + data
            messages.append(msg)

        candidates = detect_checksum_fields(messages)

        # Should detect 2-byte field
        sizes = [c.size for c in candidates]
        assert 2 in sizes


class TestIdentifyChecksumAlgorithm:
    """Test checksum algorithm identification."""

    def test_identify_empty_messages(self) -> None:
        """Test identification with empty message list."""
        result = identify_checksum_algorithm([], 0, 1)
        assert result is None

    def test_identify_xor_algorithm(self) -> None:
        """Test identification of XOR checksum."""
        # Create messages with XOR checksum
        messages = []
        for i in range(10):
            data = f"DATA{i}".encode()
            cksum = xor_checksum(data)
            msg = bytes([cksum]) + data
            messages.append(msg)

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=1)

        assert match is not None
        assert match.algorithm == "xor"
        assert match.offset == 0
        assert match.size == 1
        assert match.match_rate >= 0.8

    def test_identify_sum8_algorithm(self) -> None:
        """Test identification of sum8 checksum."""
        messages = []
        for i in range(10):
            data = f"TEST{i}".encode()
            cksum = sum8(data)
            msg = bytes([cksum]) + data
            messages.append(msg)

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=1)

        assert match is not None
        assert match.algorithm in ["xor", "sum8"]  # May match either
        assert match.match_rate >= 0.8

    def test_identify_crc16_ccitt(self) -> None:
        """Test identification of CRC-16-CCITT."""
        messages = []
        for i in range(10):
            data = f"MESSAGE{i}".encode()
            cksum = crc16_ccitt(data, init=0xFFFF)
            msg = cksum.to_bytes(2, "big") + data
            messages.append(msg)

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=2)

        assert match is not None
        assert "crc16" in match.algorithm.lower() or "sum16" in match.algorithm.lower()
        assert match.size == 2
        assert match.match_rate >= 0.8

    def test_identify_crc32(self) -> None:
        """Test identification of CRC-32."""
        messages = []
        for i in range(10):
            data = f"PACKET{i}".encode()
            cksum = crc32(data)
            msg = cksum.to_bytes(4, "big") + data
            messages.append(msg)

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=4)

        assert match is not None
        assert match.algorithm == "crc32"
        assert match.size == 4
        assert match.match_rate >= 0.8

    def test_identify_auto_size(self) -> None:
        """Test auto-detection of field size."""
        # Create messages with 2-byte checksum
        messages = []
        for i in range(10):
            data = f"DATA{i}".encode()
            cksum = sum16(data, endian="big")
            msg = cksum.to_bytes(2, "big") + data
            messages.append(msg)

        # Don't specify field_size
        match = identify_checksum_algorithm(messages, field_offset=0, field_size=None)

        if match is not None:  # May or may not identify
            assert match.size in [1, 2, 4]

    def test_identify_with_numpy_input(self) -> None:
        """Test identification with numpy array messages."""
        messages = []
        for i in range(10):
            data = f"TEST{i}".encode()
            cksum = xor_checksum(data)
            msg_bytes = bytes([cksum]) + data
            msg_array = np.frombuffer(msg_bytes, dtype=np.uint8)
            messages.append(msg_array)

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=1)

        # Should handle numpy arrays
        assert match is None or isinstance(match, ChecksumMatch)

    def test_identify_checksum_in_middle(self) -> None:
        """Test identification of checksum in middle of message."""
        messages = []
        for i in range(10):
            prefix = b"HDR"
            data = f"DATA{i}".encode()
            cksum = xor_checksum(prefix + data)
            msg = prefix + bytes([cksum]) + data
            messages.append(msg)

        match = identify_checksum_algorithm(messages, field_offset=3, field_size=1)

        if match is not None:
            assert match.offset == 3


class TestVerifyChecksums:
    """Test checksum verification."""

    def test_verify_empty_messages(self) -> None:
        """Test verification with empty message list."""
        passed, failed = verify_checksums([], "xor", 0)
        assert passed == 0
        assert failed == 0

    def test_verify_xor_all_pass(self) -> None:
        """Test verification where all checksums pass."""
        messages = []
        for i in range(10):
            data = f"DATA{i}".encode()
            cksum = xor_checksum(data)
            msg = bytes([cksum]) + data
            messages.append(msg)

        passed, failed = verify_checksums(messages, algorithm="xor", field_offset=0, scope_start=1)

        assert passed == 10
        assert failed == 0

    def test_verify_sum8_all_pass(self) -> None:
        """Test verification of sum8 checksums."""
        messages = []
        for i in range(5):
            data = f"TEST{i}".encode()
            cksum = sum8(data)
            msg = bytes([cksum]) + data
            messages.append(msg)

        passed, failed = verify_checksums(messages, algorithm="sum8", field_offset=0, scope_start=1)

        assert passed == 5
        assert failed == 0

    def test_verify_crc16_all_pass(self) -> None:
        """Test verification of CRC-16 checksums."""
        messages = []
        for i in range(5):
            data = f"MSG{i}".encode()
            cksum = crc16_ccitt(data, init=0xFFFF)
            msg = cksum.to_bytes(2, "big") + data
            messages.append(msg)

        passed, failed = verify_checksums(
            messages,
            algorithm="crc16_ccitt",
            field_offset=0,
            scope_start=2,
            init_value=0xFFFF,
        )

        assert passed == 5
        assert failed == 0

    def test_verify_mixed_results(self) -> None:
        """Test verification with some passing and some failing."""
        messages = [
            bytes([xor_checksum(b"DATA1")]) + b"DATA1",  # Good
            bytes([0xFF]) + b"DATA2",  # Bad checksum
            bytes([xor_checksum(b"DATA3")]) + b"DATA3",  # Good
        ]

        passed, failed = verify_checksums(messages, algorithm="xor", field_offset=0, scope_start=1)

        assert passed == 2
        assert failed == 1

    def test_verify_with_numpy_input(self) -> None:
        """Test verification with numpy array messages."""
        data = b"TEST"
        cksum = xor_checksum(data)
        msg_bytes = bytes([cksum]) + data
        msg_array = np.frombuffer(msg_bytes, dtype=np.uint8)

        passed, failed = verify_checksums(
            [msg_array], algorithm="xor", field_offset=0, scope_start=1
        )

        assert passed == 1
        assert failed == 0

    def test_verify_custom_scope(self) -> None:
        """Test verification with custom scope."""
        # Checksum only covers bytes 2-5
        data = b"TEST"
        cksum = xor_checksum(data[1:4])
        msg = b"H" + bytes([cksum]) + data

        passed, failed = verify_checksums(
            [msg],
            algorithm="xor",
            field_offset=1,
            scope_start=2,
            scope_end=5,
        )

        assert passed == 1
        assert failed == 0

    def test_verify_message_too_short(self) -> None:
        """Test verification with message too short."""
        messages = [b"AB"]  # Too short for 4-byte CRC32

        passed, failed = verify_checksums(
            messages, algorithm="crc32", field_offset=0, scope_start=4
        )

        assert passed == 0
        assert failed == 1


class TestDataclasses:
    """Test dataclass structures."""

    def test_checksum_candidate_creation(self) -> None:
        """Test creating ChecksumCandidate."""
        candidate = ChecksumCandidate(
            offset=0,
            size=2,
            position="header",
            correlation=0.95,
            likely_scope=(2, 20),
        )
        assert candidate.offset == 0
        assert candidate.size == 2
        assert candidate.position == "header"
        assert candidate.correlation == 0.95
        assert candidate.likely_scope == (2, 20)

    def test_checksum_match_creation(self) -> None:
        """Test creating ChecksumMatch."""
        match = ChecksumMatch(
            algorithm="crc16_ccitt",
            offset=0,
            size=2,
            scope_start=2,
            scope_end=20,
            match_rate=0.98,
            polynomial=0x1021,
            init_value=0xFFFF,
            xor_out=0x0000,
        )
        assert match.algorithm == "crc16_ccitt"
        assert match.polynomial == 0x1021
        assert match.init_value == 0xFFFF
        assert match.match_rate == 0.98

    def test_checksum_match_defaults(self) -> None:
        """Test ChecksumMatch default values."""
        match = ChecksumMatch(
            algorithm="xor",
            offset=0,
            size=1,
            scope_start=1,
            scope_end=10,
            match_rate=1.0,
        )
        assert match.polynomial is None
        assert match.init_value is None
        assert match.xor_out is None


class TestStatisticalChecksumEdgeCases:
    """Test edge cases and error handling."""

    def test_detect_with_variable_length_messages(self) -> None:
        """Test detection with variable length messages."""
        messages = [b"ABCD", b"ABCDEF", b"ABCDEFGH"]
        candidates = detect_checksum_fields(messages)

        # Should only check up to minimum length
        assert isinstance(candidates, list)

    def test_identify_with_insufficient_matches(self) -> None:
        """Test identification with low match rate."""
        # Random checksums - won't match any algorithm
        messages = [
            b"\x00DATA1",
            b"\xff DATA2",
            b"\x42DATA3",
        ]

        match = identify_checksum_algorithm(messages, field_offset=0, field_size=1)

        # Should not identify with low match rate
        if match is not None:
            assert match.match_rate < 0.8

    def test_crc_with_all_zeros(self) -> None:
        """Test CRC algorithms with all-zero data."""
        data = b"\x00\x00\x00\x00"

        result_crc8 = crc8(data)
        result_crc16 = crc16_ccitt(data)
        result_crc32 = crc32(data)

        assert isinstance(result_crc8, int)
        assert isinstance(result_crc16, int)
        assert isinstance(result_crc32, int)

    def test_crc_with_all_ones(self) -> None:
        """Test CRC algorithms with all-one data."""
        data = b"\xff\xff\xff\xff"

        result_crc8 = crc8(data)
        result_crc16 = crc16_ccitt(data)
        result_crc32 = crc32(data)

        assert isinstance(result_crc8, int)
        assert isinstance(result_crc16, int)
        assert isinstance(result_crc32, int)

    def test_large_message_performance(self) -> None:
        """Test checksum computation on large message."""
        # 1 MB message
        data = b"X" * 1_000_000

        # Should complete without timeout
        result = crc32(data)
        assert isinstance(result, int)

    def test_bytearray_input(self) -> None:
        """Test checksum functions with bytearray input."""
        data = bytearray(b"TEST")

        result_xor = xor_checksum(data)
        result_sum8 = sum8(data)
        result_crc8 = crc8(data)

        # Should work with bytearray
        assert isinstance(result_xor, int)
        assert isinstance(result_sum8, int)
        assert isinstance(result_crc8, int)


class TestIntegrationScenarios:
    """Test complete detection and identification workflows."""

    def test_full_detection_workflow_xor(self) -> None:
        """Test complete workflow: detect then identify XOR checksum."""
        # Create messages with XOR checksum
        messages = []
        for i in range(15):
            data = f"PACKET{i:03d}".encode()
            cksum = xor_checksum(data)
            msg = bytes([cksum]) + data
            messages.append(msg)

        # Step 1: Detect candidates
        candidates = detect_checksum_fields(messages)
        assert len(candidates) > 0

        # Step 2: Identify algorithm for best candidate
        best = candidates[0]
        match = identify_checksum_algorithm(
            messages, field_offset=best.offset, field_size=best.size
        )

        # Match may or may not be found depending on scope detection
        if match is not None:
            assert match.match_rate >= 0.8

            # Step 3: Verify
            passed, failed = verify_checksums(
                messages,
                algorithm=match.algorithm,
                field_offset=match.offset,
                scope_start=match.scope_start,
                scope_end=match.scope_end,
            )

            assert passed >= 12  # At least 80% should pass

    def test_full_detection_workflow_crc16(self) -> None:
        """Test complete workflow: detect then identify CRC-16 checksum."""
        # Create messages with CRC-16 checksum
        messages = []
        for i in range(15):
            prefix = b"HDR"
            data = f"DATA{i:03d}".encode()
            payload = prefix + data
            cksum = crc16_ccitt(payload, init=0xFFFF)
            msg = payload + cksum.to_bytes(2, "big")
            messages.append(msg)

        # Detect candidates
        candidates = detect_checksum_fields(messages)
        assert len(candidates) > 0

        # Find trailer candidates
        trailer_candidates = [c for c in candidates if c.position == "trailer"]
        if trailer_candidates:
            best = trailer_candidates[0]

            # Identify algorithm
            match = identify_checksum_algorithm(
                messages, field_offset=best.offset, field_size=best.size
            )

            if match is not None:
                assert match.size == 2
                assert match.match_rate >= 0.8

    def test_no_checksum_present(self) -> None:
        """Test detection when no checksum is present."""
        # Random data with no consistent checksum pattern
        messages = [
            b"RANDOM1234567890",
            b"ABCDEFGHIJKLMNOP",
            b"TESTDATA1234ABCD",
        ]

        candidates = detect_checksum_fields(messages)

        # May find some candidates but with low correlation
        for candidate in candidates:
            # High correlation would indicate false positive
            # In practice, random data should have low correlation
            assert candidate.correlation < 0.9, "Random data should not have high correlation"
            # Just verify no crashes
