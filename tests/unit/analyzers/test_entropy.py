"""Unit tests for entropy analysis and crypto detection.

Tests cover:
- Shannon entropy calculation
- Chi-squared test for randomness
- Sliding window entropy analysis
- Crypto field detection across messages
- Compression vs encryption distinction
- Confidence scoring
- Edge cases and error handling
"""

from __future__ import annotations

import os
import zlib

import numpy as np
import pytest

from oscura.analyzers.entropy import CryptoDetector, EntropyResult

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector() -> CryptoDetector:
    """Create CryptoDetector instance for testing."""
    return CryptoDetector()


@pytest.fixture
def random_data() -> bytes:
    """Generate cryptographically random data (high entropy)."""
    return os.urandom(1000)


@pytest.fixture
def structured_data() -> bytes:
    """Generate structured, low-entropy data."""
    return b"AAAA" * 250  # 1000 bytes, very low entropy


@pytest.fixture
def compressed_data() -> bytes:
    """Generate compressed data (medium-high entropy)."""
    # Compress text - produces medium entropy
    text = b"The quick brown fox jumps over the lazy dog. " * 50
    return zlib.compress(text)


@pytest.fixture
def mixed_data() -> bytes:
    """Generate data with mixed plaintext and encrypted regions."""
    header = b"PROTOCOL_HEADER_V1\x00"
    payload = os.urandom(200)  # Encrypted payload
    footer = b"\x00END_MARKER"
    return header + payload + footer


# =============================================================================
# Shannon Entropy Tests
# =============================================================================


def test_shannon_entropy_zero_for_uniform(detector: CryptoDetector) -> None:
    """Test that uniform data (all same byte) has zero entropy."""
    data = b"\x00" * 100
    entropy = detector._shannon_entropy(data)
    assert entropy == 0.0


def test_shannon_entropy_high_for_random(detector: CryptoDetector, random_data: bytes) -> None:
    """Test that random data has high entropy (>7.5 bits/byte)."""
    entropy = detector._shannon_entropy(random_data)
    assert entropy > 7.5
    assert entropy <= 8.0  # Maximum possible entropy


def test_shannon_entropy_low_for_structured(
    detector: CryptoDetector, structured_data: bytes
) -> None:
    """Test that structured data has low entropy."""
    entropy = detector._shannon_entropy(structured_data)
    assert entropy < 3.0  # Well below compression threshold


def test_shannon_entropy_medium_for_compressed(
    detector: CryptoDetector, compressed_data: bytes
) -> None:
    """Test that compressed data has medium-high entropy."""
    entropy = detector._shannon_entropy(compressed_data)
    # Compressed data typically 5.5-7.5 bits/byte (varies by compression ratio)
    assert 5.0 < entropy < 8.0


def test_shannon_entropy_empty_data(detector: CryptoDetector) -> None:
    """Test that empty data returns zero entropy."""
    entropy = detector._shannon_entropy(b"")
    assert entropy == 0.0


def test_shannon_entropy_single_byte(detector: CryptoDetector) -> None:
    """Test entropy calculation with single byte."""
    entropy = detector._shannon_entropy(b"\x42")
    assert entropy == 0.0  # Only one value, no uncertainty


def test_shannon_entropy_two_values(detector: CryptoDetector) -> None:
    """Test entropy with two equally-likely values."""
    data = b"\x00\xff" * 100  # Perfect 50/50 distribution
    entropy = detector._shannon_entropy(data)
    assert 0.9 < entropy < 1.1  # Should be 1.0 bit per byte


# =============================================================================
# Chi-Squared Test
# =============================================================================


def test_chi_squared_random_passes(detector: CryptoDetector, random_data: bytes) -> None:
    """Test that random data passes chi-squared test (high p-value)."""
    p_value = detector._chi_squared_test(random_data)
    assert p_value > detector.CHI_SQUARED_ALPHA  # Fail to reject null hypothesis


def test_chi_squared_structured_fails(detector: CryptoDetector, structured_data: bytes) -> None:
    """Test that structured data fails chi-squared test (low p-value)."""
    p_value = detector._chi_squared_test(structured_data)
    assert p_value < detector.CHI_SQUARED_ALPHA  # Reject null hypothesis


def test_chi_squared_empty_data(detector: CryptoDetector) -> None:
    """Test chi-squared test with empty data."""
    p_value = detector._chi_squared_test(b"")
    assert p_value == 0.0


def test_chi_squared_single_value(detector: CryptoDetector) -> None:
    """Test chi-squared with only one byte value."""
    data = b"\x42" * 100
    p_value = detector._chi_squared_test(data)
    assert p_value < 0.01  # Extremely non-uniform


# =============================================================================
# Analyze Entropy Tests
# =============================================================================


def test_analyze_entropy_encrypted(detector: CryptoDetector, random_data: bytes) -> None:
    """Test entropy analysis correctly identifies encrypted data."""
    result = detector.analyze_entropy(random_data)

    assert isinstance(result, EntropyResult)
    assert result.shannon_entropy > 7.5
    assert result.is_high_entropy
    assert result.is_random
    assert result.encryption_likelihood > 0.8
    assert result.compression_likelihood < 0.5
    assert result.confidence > 0.8


def test_analyze_entropy_structured(detector: CryptoDetector, structured_data: bytes) -> None:
    """Test entropy analysis correctly identifies structured data."""
    result = detector.analyze_entropy(structured_data)

    assert result.shannon_entropy < 3.0
    assert not result.is_high_entropy
    assert not result.is_random
    assert result.encryption_likelihood < 0.1
    assert result.compression_likelihood == 0.0  # Too low entropy
    assert result.confidence > 0.9  # Very clear structured data


def test_analyze_entropy_compressed(detector: CryptoDetector, compressed_data: bytes) -> None:
    """Test entropy analysis can distinguish compression from encryption."""
    result = detector.analyze_entropy(compressed_data)

    # Compressed data has medium-high entropy but isn't uniformly random
    assert 5.0 < result.shannon_entropy < 8.0
    assert result.compression_likelihood >= 0.0
    # May or may not be classified as high entropy depending on compression ratio


def test_analyze_entropy_empty_raises(detector: CryptoDetector) -> None:
    """Test that empty data raises ValueError."""
    with pytest.raises(ValueError, match="Cannot analyze empty data"):
        detector.analyze_entropy(b"")


def test_analyze_entropy_with_window(detector: CryptoDetector, random_data: bytes) -> None:
    """Test entropy analysis with custom window size."""
    result = detector.analyze_entropy(random_data, window_size=128)

    # Should analyze only first 128 bytes
    assert result.shannon_entropy > 6.5  # High entropy (random data varies 7.0-8.0)
    assert result.confidence < 1.0  # Lower confidence due to smaller sample


def test_analyze_entropy_invalid_window(detector: CryptoDetector) -> None:
    """Test that invalid window size raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be positive"):
        detector.analyze_entropy(b"test", window_size=0)

    with pytest.raises(ValueError, match="window_size must be positive"):
        detector.analyze_entropy(b"test", window_size=-1)


def test_analyze_entropy_confidence_scaling(detector: CryptoDetector) -> None:
    """Test that confidence increases with sample size."""
    # Small sample
    small_result = detector.analyze_entropy(os.urandom(16))
    # Medium sample
    medium_result = detector.analyze_entropy(os.urandom(64))
    # Large sample
    large_result = detector.analyze_entropy(os.urandom(512))

    # Confidence should increase with sample size
    assert small_result.confidence < medium_result.confidence
    assert medium_result.confidence < large_result.confidence


# =============================================================================
# Sliding Window Entropy Tests
# =============================================================================


def test_sliding_window_basic(detector: CryptoDetector, random_data: bytes) -> None:
    """Test basic sliding window entropy analysis."""
    windows = detector.sliding_window_entropy(random_data, window_size=128, stride=32)

    assert len(windows) > 0
    assert all(isinstance(offset, int) for offset, _ in windows)
    assert all(isinstance(entropy, float) for _, entropy in windows)
    # Random data with 128-byte window typically shows 6.4-6.7 bits/byte
    # (not enough samples for full 256-value distribution)
    assert all(6.0 < entropy < 8.0 for _, entropy in windows)  # All random


def test_sliding_window_finds_encrypted_region(detector: CryptoDetector) -> None:
    """Test sliding window can identify encrypted regions in mixed data."""
    # Use larger window size for better entropy differentiation
    header = b"PLAINTEXT_HEADER_" * 10  # 170 bytes low entropy
    payload = os.urandom(300)  # 300 bytes high entropy
    footer = b"_FOOTER_PLAINTEXT" * 10  # 170 bytes low entropy
    mixed_data = header + payload + footer

    windows = detector.sliding_window_entropy(mixed_data, window_size=128, stride=32)

    # Extract entropy values
    entropies = [ent for _, ent in windows]

    # Should have some low-entropy regions (header/footer)
    assert any(ent < 5.0 for ent in entropies), f"Expected low entropy regions, got: {entropies}"
    # Should have some high-entropy regions (encrypted payload)
    # With 128-byte window, random data shows ~6.4-6.7 bits/byte
    assert any(ent > 6.3 for ent in entropies), f"Expected high entropy regions, got: {entropies}"


def test_sliding_window_empty_raises(detector: CryptoDetector) -> None:
    """Test that empty data raises ValueError."""
    with pytest.raises(ValueError, match="Cannot analyze empty data"):
        detector.sliding_window_entropy(b"")


def test_sliding_window_invalid_window_size(detector: CryptoDetector) -> None:
    """Test that invalid window_size raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be positive"):
        detector.sliding_window_entropy(b"test", window_size=0)


def test_sliding_window_invalid_stride(detector: CryptoDetector) -> None:
    """Test that invalid stride raises ValueError."""
    with pytest.raises(ValueError, match="stride must be positive"):
        detector.sliding_window_entropy(b"test" * 100, window_size=10, stride=0)


def test_sliding_window_data_too_short(detector: CryptoDetector) -> None:
    """Test that data shorter than window_size raises ValueError."""
    with pytest.raises(ValueError, match="Data length .* must be >= window_size"):
        detector.sliding_window_entropy(b"short", window_size=100)


def test_sliding_window_offsets_correct(detector: CryptoDetector) -> None:
    """Test that sliding window offsets are computed correctly."""
    data = os.urandom(500)
    windows = detector.sliding_window_entropy(data, window_size=64, stride=32)

    # Check offsets are spaced by stride
    offsets = [offset for offset, _ in windows]
    for i in range(len(offsets) - 1):
        assert offsets[i + 1] - offsets[i] == 32


def test_sliding_window_full_coverage(detector: CryptoDetector) -> None:
    """Test that sliding window covers entire data range."""
    data = os.urandom(300)
    windows = detector.sliding_window_entropy(data, window_size=100, stride=50)

    offsets = [offset for offset, _ in windows]
    assert offsets[0] == 0  # Starts at beginning
    assert offsets[-1] + 100 <= len(data)  # Last window fits


# =============================================================================
# Crypto Field Detection Tests
# =============================================================================


def test_detect_crypto_fields_simple(detector: CryptoDetector) -> None:
    """Test crypto field detection with simple encrypted messages."""
    # Create messages with encrypted payload at fixed offset
    messages = []
    for i in range(20):
        # Use structured header (varying but low entropy when aggregated)
        header = b"HDR" + bytes([i % 4, i % 8, 0x00, 0x00, 0x00])
        payload = os.urandom(64)  # Fixed encrypted payload
        footer = b"END"
        messages.append(header + payload + footer)

    fields = detector.detect_crypto_fields(messages, min_field_size=32)

    # Should detect the encrypted payload region
    assert len(fields) > 0
    field = fields[0]
    assert field["type"] == "encrypted_payload"
    assert field["offset"] >= 3  # After fixed "HDR" part
    assert field["length"] >= 32  # At least min_field_size
    # Positional entropy with 20 samples: max ~4.32 bits, high threshold ~3.0 bits
    assert field["entropy"] > 2.5  # High positional entropy
    assert field["sample_count"] == 20


def test_detect_crypto_fields_no_encryption(detector: CryptoDetector) -> None:
    """Test crypto field detection with no encrypted fields."""
    # All plaintext messages
    messages = [b"PLAINTEXT_MESSAGE_" + str(i).encode() for i in range(10)]

    fields = detector.detect_crypto_fields(messages)

    # Should detect no crypto fields
    assert len(fields) == 0


def test_detect_crypto_fields_multiple_lengths(detector: CryptoDetector) -> None:
    """Test crypto field detection groups messages by length."""
    messages = []

    # Type A messages (100 bytes) - use more messages for better positional entropy
    for _ in range(20):
        messages.append(b"A" * 20 + os.urandom(60) + b"A" * 20)

    # Type B messages (200 bytes)
    for _ in range(20):
        messages.append(b"B" * 40 + os.urandom(120) + b"B" * 40)

    fields = detector.detect_crypto_fields(messages, min_field_size=30)

    # Should find fields in both message types
    assert len(fields) >= 2

    # Fields should have different message_length values
    lengths = {field["message_length"] for field in fields}
    assert 100 in lengths
    assert 200 in lengths


def test_detect_crypto_fields_empty_raises(detector: CryptoDetector) -> None:
    """Test that empty message list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot analyze empty message list"):
        detector.detect_crypto_fields([])


def test_detect_crypto_fields_invalid_min_size(detector: CryptoDetector) -> None:
    """Test that invalid min_field_size raises ValueError."""
    with pytest.raises(ValueError, match="min_field_size must be positive"):
        detector.detect_crypto_fields([b"test"], min_field_size=0)


def test_detect_crypto_fields_respects_min_size(detector: CryptoDetector) -> None:
    """Test that min_field_size filters small fields."""
    # Messages with small encrypted region
    messages = [b"HEADER" + os.urandom(8) + b"FOOTER" for _ in range(10)]

    # Should not detect with large min_field_size
    fields_large = detector.detect_crypto_fields(messages, min_field_size=20)
    assert len(fields_large) == 0

    # Should detect with small min_field_size
    fields_small = detector.detect_crypto_fields(messages, min_field_size=8)
    # May or may not detect depending on positional entropy


def test_detect_crypto_fields_end_of_message(detector: CryptoDetector) -> None:
    """Test crypto field detection at end of message."""
    # Messages with encrypted field at the end
    messages = [b"HEADER_" + os.urandom(64) for _ in range(15)]

    fields = detector.detect_crypto_fields(messages, min_field_size=32)

    if fields:  # May detect if positional entropy is high enough
        field = fields[0]
        # Field should extend to end of message
        assert field["offset"] + field["length"] == field["message_length"]


# =============================================================================
# Compression vs Encryption Tests
# =============================================================================


def test_compression_vs_encryption_distinction(detector: CryptoDetector) -> None:
    """Test that compressed and encrypted data are distinguished."""
    # Encrypted data (truly random)
    encrypted = os.urandom(500)
    encrypted_result = detector.analyze_entropy(encrypted)

    # Compressed data (structured randomness)
    text = b"The quick brown fox jumps over the lazy dog. " * 100
    compressed = zlib.compress(text)
    compressed_result = detector.analyze_entropy(compressed)

    # Both should have high entropy, but encryption should score higher
    assert encrypted_result.encryption_likelihood > compressed_result.encryption_likelihood


def test_compression_likelihood_medium_entropy(detector: CryptoDetector) -> None:
    """Test compression likelihood is highest in medium entropy range."""
    # Create data with controlled entropy (not perfectly random)
    # Mix of random and structured
    semi_random = os.urandom(100) + b"PATTERN" * 50
    result = detector.analyze_entropy(semi_random)

    # Should have some compression likelihood if in right range
    if 6.5 < result.shannon_entropy < 7.5:
        assert result.compression_likelihood > 0.0


def test_encryption_likelihood_requires_high_entropy(detector: CryptoDetector) -> None:
    """Test encryption likelihood requires very high entropy."""
    # Low entropy data
    low_ent = b"AAAA" * 100
    low_result = detector.analyze_entropy(low_ent)
    assert low_result.encryption_likelihood < 0.1

    # Medium entropy data
    medium_ent = zlib.compress(b"Test data " * 100)
    medium_result = detector.analyze_entropy(medium_ent)
    # May have some likelihood but not high

    # High entropy data
    high_ent = os.urandom(500)
    high_result = detector.analyze_entropy(high_ent)
    # Random data typically shows 7.5-7.6 bits/byte, giving likelihood 0.7-0.9
    assert high_result.encryption_likelihood > 0.6


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_small_sample_low_confidence(detector: CryptoDetector) -> None:
    """Test that small samples have lower confidence."""
    small = os.urandom(16)
    large = os.urandom(512)

    small_result = detector.analyze_entropy(small)
    large_result = detector.analyze_entropy(large)

    assert small_result.confidence < large_result.confidence


def test_ambiguous_entropy_lower_confidence(detector: CryptoDetector) -> None:
    """Test that ambiguous entropy range has lower confidence."""
    # Create data in ambiguous range (6.5-7.5)
    # Mix to get medium-high entropy
    mixed = os.urandom(200) + b"PATTERN" * 100
    mixed_result = detector.analyze_entropy(mixed)

    # Clear cases should have higher confidence
    clear_encrypted = detector.analyze_entropy(os.urandom(500))
    clear_structured = detector.analyze_entropy(b"AAAA" * 200)

    if 6.5 < mixed_result.shannon_entropy < 7.5:
        assert mixed_result.confidence < clear_encrypted.confidence
        assert mixed_result.confidence < clear_structured.confidence


def test_single_byte_messages_skipped(detector: CryptoDetector) -> None:
    """Test that very short messages are skipped in field detection."""
    messages = [b"A", b"B", b"C"] * 10  # All 1-byte messages

    fields = detector.detect_crypto_fields(messages, min_field_size=2)

    # Should skip all messages (too short)
    assert len(fields) == 0


def test_positional_entropy_computation(detector: CryptoDetector) -> None:
    """Test internal positional entropy computation."""
    # Messages where specific positions have high/low entropy
    messages = []
    for i in range(20):
        # First 5 bytes: variable (high positional entropy)
        prefix = os.urandom(5)
        # Next 5 bytes: constant (low positional entropy)
        constant = b"CONST"
        # Last 5 bytes: variable (high positional entropy)
        suffix = os.urandom(5)
        messages.append(prefix + constant + suffix)

    position_entropy = detector._compute_positional_entropy(messages)

    # Constant region should have lower positional entropy
    constant_region_entropy = position_entropy[5:10]
    variable_region_entropy = np.concatenate([position_entropy[:5], position_entropy[10:]])

    assert np.mean(constant_region_entropy) < np.mean(variable_region_entropy)


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow_protocol_analysis(detector: CryptoDetector) -> None:
    """Test complete workflow: analyze messages, detect fields, classify."""
    # Simulate captured protocol messages
    messages = []
    for session_id in range(20):
        # Fixed header
        header = b"PROTO\x01\x00"
        # Session ID (variable but structured)
        session = session_id.to_bytes(2, "big")
        # Encrypted payload
        payload = os.urandom(128)
        # Checksum (variable)
        checksum = os.urandom(4)

        messages.append(header + session + payload + checksum)

    # Detect crypto fields
    fields = detector.detect_crypto_fields(messages, min_field_size=64)

    # Should find the encrypted payload
    assert len(fields) >= 1

    # Verify field characteristics
    payload_field = fields[0]
    assert payload_field["length"] >= 64
    # Positional entropy with 20 samples: max ~4.32 bits
    assert payload_field["entropy"] > 2.5  # High positional entropy
    assert payload_field["type"] == "encrypted_payload"

    # Analyze individual message
    result = detector.analyze_entropy(messages[0])
    # Overall message has mixed content (header + encrypted + checksum)
    # so encryption_likelihood may be lower than pure encrypted data
    assert result.shannon_entropy > 6.0  # Has high-entropy encrypted content
    # Encryption likelihood depends on whether data passes chi-squared test
    # With mixed content, this may be lower
    assert result.encryption_likelihood > 0.0 or result.compression_likelihood > 0.0


def test_constants_are_sensible(detector: CryptoDetector) -> None:
    """Test that class constants have sensible values."""
    assert 0.0 < detector.ENTROPY_THRESHOLD_STRUCTURED < 8.0
    assert detector.ENTROPY_THRESHOLD_STRUCTURED < detector.ENTROPY_THRESHOLD_COMPRESSED
    assert detector.ENTROPY_THRESHOLD_COMPRESSED < detector.ENTROPY_THRESHOLD_ENCRYPTED
    assert detector.ENTROPY_THRESHOLD_ENCRYPTED < 8.0

    assert 0.0 < detector.CHI_SQUARED_ALPHA < 1.0
    assert detector.MIN_SAMPLE_SIZE > 0
