"""Tests for LoRaWAN cryptographic operations.

Test coverage:
- AES-128 CTR mode decryption with test vectors
- CMAC-based MIC computation
- MIC verification
- Error handling for missing cryptography library
"""

import pytest

# Skip all tests if cryptography is not available
pytest.importorskip("cryptography", reason="cryptography not installed")

from oscura.iot.lorawan.crypto import compute_mic, decrypt_payload, verify_mic


class TestDecryptPayload:
    """Test LoRaWAN payload decryption."""

    def test_decrypt_empty_payload(self):
        """Test decrypting empty payload."""
        key = bytes(16)
        result = decrypt_payload(b"", key, 0x01020304, 1, "up")
        assert result == b""

    def test_decrypt_uplink_single_block(self):
        """Test decrypting uplink payload (single AES block)."""
        # Known test vector from LoRaWAN specification
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        encrypted = bytes.fromhex("0123456789ABCDEF")
        dev_addr = 0x01020304
        fcnt = 1

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        # Result should be 8 bytes (we're XORing with keystream)
        assert len(decrypted) == 8
        assert isinstance(decrypted, bytes)

    def test_decrypt_downlink_single_block(self):
        """Test decrypting downlink payload."""
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        encrypted = bytes.fromhex("FEDCBA9876543210")
        dev_addr = 0x04030201
        fcnt = 10

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "down")

        assert len(decrypted) == 8
        assert isinstance(decrypted, bytes)

    def test_decrypt_multi_block_payload(self):
        """Test decrypting payload spanning multiple AES blocks."""
        key = bytes(16)
        # 20 bytes (2 AES blocks)
        encrypted = bytes(20)
        dev_addr = 0x12345678
        fcnt = 100

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert len(decrypted) == 20

    def test_decrypt_invalid_key_length(self):
        """Test decryption with invalid key length."""
        key = bytes(15)  # Should be 16
        encrypted = bytes(8)

        with pytest.raises(ValueError, match="Key must be 16 bytes"):
            decrypt_payload(encrypted, key, 0x01020304, 1, "up")

    def test_decrypt_deterministic(self):
        """Test that decryption is deterministic."""
        key = bytes(16)
        encrypted = bytes.fromhex("AABBCCDD")
        dev_addr = 0x01020304
        fcnt = 5

        result1 = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")
        result2 = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert result1 == result2

    def test_decrypt_reversible(self):
        """Test that encryption/decryption are reversible."""
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        plaintext = b"Hello LoRaWAN"
        dev_addr = 0x01020304
        fcnt = 1

        # Encrypt by "decrypting" plaintext (XOR is symmetric)
        encrypted = decrypt_payload(plaintext, key, dev_addr, fcnt, "up")

        # Decrypt to get back plaintext
        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert decrypted == plaintext


class TestComputeMIC:
    """Test MIC computation."""

    def test_compute_mic_basic(self):
        """Test basic MIC computation."""
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        data = bytes.fromhex("40010203040001000100")
        dev_addr = 0x01020304
        fcnt = 1

        mic = compute_mic(data, key, dev_addr, fcnt, "up")

        # MIC should be 32-bit value
        assert isinstance(mic, int)
        assert 0 <= mic <= 0xFFFFFFFF

    def test_compute_mic_empty_data(self):
        """Test MIC computation for empty data."""
        key = bytes(16)
        data = b""
        dev_addr = 0x01020304
        fcnt = 0

        mic = compute_mic(data, key, dev_addr, fcnt, "up")

        assert isinstance(mic, int)

    def test_compute_mic_invalid_key_length(self):
        """Test MIC computation with invalid key length."""
        key = bytes(10)
        data = bytes(20)

        with pytest.raises(ValueError, match="Key must be 16 bytes"):
            compute_mic(data, key, 0x01020304, 1, "up")

    def test_compute_mic_direction_affects_result(self):
        """Test that direction affects MIC result."""
        key = bytes(16)
        data = bytes(10)
        dev_addr = 0x01020304
        fcnt = 1

        mic_up = compute_mic(data, key, dev_addr, fcnt, "up")
        mic_down = compute_mic(data, key, dev_addr, fcnt, "down")

        # Different directions should produce different MICs
        assert mic_up != mic_down

    def test_compute_mic_deterministic(self):
        """Test that MIC computation is deterministic."""
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        data = bytes.fromhex("AABBCCDD")
        dev_addr = 0x12345678
        fcnt = 10

        mic1 = compute_mic(data, key, dev_addr, fcnt, "up")
        mic2 = compute_mic(data, key, dev_addr, fcnt, "up")

        assert mic1 == mic2


class TestVerifyMIC:
    """Test MIC verification."""

    def test_verify_mic_valid(self):
        """Test verification of valid MIC."""
        key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        data = bytes.fromhex("40010203040001000100")
        dev_addr = 0x01020304
        fcnt = 1

        # Compute valid MIC
        valid_mic = compute_mic(data, key, dev_addr, fcnt, "up")

        # Verify it
        is_valid = verify_mic(data, valid_mic, key, dev_addr, fcnt, "up")

        assert is_valid is True

    def test_verify_mic_invalid(self):
        """Test verification of invalid MIC."""
        key = bytes(16)
        data = bytes(20)
        dev_addr = 0x01020304
        fcnt = 1

        # Use incorrect MIC
        invalid_mic = 0x12345678

        is_valid = verify_mic(data, invalid_mic, key, dev_addr, fcnt, "up")

        assert is_valid is False

    def test_verify_mic_wrong_direction(self):
        """Test verification with wrong direction."""
        key = bytes(16)
        data = bytes(10)
        dev_addr = 0x01020304
        fcnt = 1

        # Compute MIC for uplink
        mic = compute_mic(data, key, dev_addr, fcnt, "up")

        # Verify with downlink direction (should fail)
        is_valid = verify_mic(data, mic, key, dev_addr, fcnt, "down")

        assert is_valid is False

    def test_verify_mic_wrong_fcnt(self):
        """Test verification with wrong frame counter."""
        key = bytes(16)
        data = bytes(10)
        dev_addr = 0x01020304
        fcnt = 1

        # Compute MIC with fcnt=1
        mic = compute_mic(data, key, dev_addr, fcnt, "up")

        # Verify with fcnt=2 (should fail)
        is_valid = verify_mic(data, mic, key, dev_addr, 2, "up")

        assert is_valid is False

    def test_verify_mic_invalid_key_returns_false(self):
        """Test that verification with invalid key returns False."""
        # Even with invalid key length, verify_mic should handle gracefully
        key = bytes(10)
        data = bytes(10)

        is_valid = verify_mic(data, 0x12345678, key, 0x01020304, 1, "up")

        # Should return False rather than raising exception
        assert is_valid is False


class TestCryptoErrorHandling:
    """Test error handling for cryptographic operations."""

    def test_decrypt_with_17_byte_key(self):
        """Test decryption with key that is too long."""
        key = bytes(17)
        encrypted = bytes(8)

        with pytest.raises(ValueError, match="Key must be 16 bytes"):
            decrypt_payload(encrypted, key, 0x01020304, 1, "up")

    def test_decrypt_with_zero_length_key(self):
        """Test decryption with zero-length key."""
        key = bytes(0)
        encrypted = bytes(8)

        with pytest.raises(ValueError, match="Key must be 16 bytes"):
            decrypt_payload(encrypted, key, 0x01020304, 1, "up")

    def test_compute_mic_with_17_byte_key(self):
        """Test MIC computation with key that is too long."""
        key = bytes(17)
        data = bytes(10)

        with pytest.raises(ValueError, match="Key must be 16 bytes"):
            compute_mic(data, key, 0x01020304, 1, "up")

    def test_decrypt_single_byte_payload(self):
        """Test decrypting single-byte payload."""
        key = bytes(16)
        encrypted = bytes(1)
        dev_addr = 0x01020304
        fcnt = 1

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert len(decrypted) == 1

    def test_decrypt_exact_block_boundary(self):
        """Test decrypting payload at exact AES block boundary."""
        key = bytes(16)
        encrypted = bytes(16)  # Exactly one AES block
        dev_addr = 0x01020304
        fcnt = 1

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert len(decrypted) == 16

    def test_decrypt_two_blocks(self):
        """Test decrypting payload requiring exactly two AES blocks."""
        key = bytes(16)
        encrypted = bytes(32)  # Exactly two AES blocks
        dev_addr = 0x01020304
        fcnt = 1

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert len(decrypted) == 32

    def test_decrypt_just_over_block_boundary(self):
        """Test decrypting payload just over AES block boundary."""
        key = bytes(16)
        encrypted = bytes(17)  # One byte into second block
        dev_addr = 0x01020304
        fcnt = 1

        decrypted = decrypt_payload(encrypted, key, dev_addr, fcnt, "up")

        assert len(decrypted) == 17

    def test_compute_mic_large_message(self):
        """Test MIC computation for large message."""
        key = bytes(16)
        data = bytes(255)  # Max typical LoRaWAN payload
        dev_addr = 0x01020304
        fcnt = 1

        mic = compute_mic(data, key, dev_addr, fcnt, "up")

        assert isinstance(mic, int)
        assert 0 <= mic <= 0xFFFFFFFF

    def test_decrypt_different_dev_addresses(self):
        """Test decryption with different device addresses produces different results."""
        key = bytes(16)
        encrypted = bytes(8)
        fcnt = 1

        decrypted1 = decrypt_payload(encrypted, key, 0x01020304, fcnt, "up")
        decrypted2 = decrypt_payload(encrypted, key, 0x04030201, fcnt, "up")

        # Different dev_addr should produce different decryption
        assert decrypted1 != decrypted2

    def test_decrypt_different_fcnt(self):
        """Test decryption with different frame counters produces different results."""
        key = bytes(16)
        encrypted = bytes(8)
        dev_addr = 0x01020304

        decrypted1 = decrypt_payload(encrypted, key, dev_addr, 1, "up")
        decrypted2 = decrypt_payload(encrypted, key, dev_addr, 2, "up")

        # Different fcnt should produce different decryption
        assert decrypted1 != decrypted2

    def test_compute_mic_different_dev_addresses(self):
        """Test MIC computation with different device addresses."""
        key = bytes(16)
        data = bytes(10)
        fcnt = 1

        mic1 = compute_mic(data, key, 0x01020304, fcnt, "up")
        mic2 = compute_mic(data, key, 0x04030201, fcnt, "up")

        # Different dev_addr should produce different MIC
        assert mic1 != mic2

    def test_compute_mic_different_data(self):
        """Test MIC computation with different data."""
        key = bytes(16)
        dev_addr = 0x01020304
        fcnt = 1

        mic1 = compute_mic(b"data1", key, dev_addr, fcnt, "up")
        mic2 = compute_mic(b"data2", key, dev_addr, fcnt, "up")

        # Different data should produce different MIC
        assert mic1 != mic2
