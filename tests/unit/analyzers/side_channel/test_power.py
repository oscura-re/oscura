"""Tests for power analysis side-channel attacks (DPA/CPA).

Tests for side-channel power analysis module.
"""

import numpy as np
import pytest

from oscura.analyzers.side_channel.power import (
    AES_SBOX,
    CPAAnalyzer,
    DPAAnalyzer,
    hamming_distance,
    hamming_weight,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Hamming weight/distance tests
# =============================================================================


def test_hamming_weight_scalar():
    """Test Hamming weight for scalar values."""
    assert hamming_weight(0x00) == 0
    assert hamming_weight(0xFF) == 8
    assert hamming_weight(0x0F) == 4
    assert hamming_weight(0xAA) == 4
    assert hamming_weight(0x55) == 4


def test_hamming_weight_array():
    """Test Hamming weight for numpy arrays."""
    values = np.array([0x00, 0xFF, 0x0F, 0xAA, 0x55], dtype=np.uint8)
    expected = np.array([0, 8, 4, 4, 4], dtype=np.int32)
    result = hamming_weight(values)
    np.testing.assert_array_equal(result, expected)


def test_hamming_distance_scalar():
    """Test Hamming distance for scalar values."""
    assert hamming_distance(0x00, 0xFF) == 8
    assert hamming_distance(0x00, 0x00) == 0
    assert hamming_distance(0x0F, 0xF0) == 8
    assert hamming_distance(0xAA, 0x55) == 8


def test_hamming_distance_array():
    """Test Hamming distance for numpy arrays."""
    val1 = np.array([0x00, 0x00, 0x0F], dtype=np.uint8)
    val2 = np.array([0xFF, 0x00, 0xF0], dtype=np.uint8)
    expected = np.array([8, 0, 8], dtype=np.int32)
    result = hamming_distance(val1, val2)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# AES S-box tests
# =============================================================================


def test_aes_sbox_known_values():
    """Test AES S-box with known values."""
    # Test some known AES S-box values
    assert AES_SBOX[0x00] == 0x63
    assert AES_SBOX[0x01] == 0x7C
    assert AES_SBOX[0xFF] == 0x16
    assert AES_SBOX[0x53] == 0xED


def test_aes_sbox_size():
    """Test AES S-box has correct size."""
    assert len(AES_SBOX) == 256
    assert AES_SBOX.dtype == np.uint8


# =============================================================================
# DPA tests
# =============================================================================


def test_dpa_initialization():
    """Test DPA analyzer initialization."""
    dpa = DPAAnalyzer(target_bit=0, byte_position=0)
    assert dpa.target_bit == 0
    assert dpa.byte_position == 0


def test_dpa_initialization_invalid_bit():
    """Test DPA with invalid target bit."""
    with pytest.raises(ValueError, match="target_bit must be 0-7"):
        DPAAnalyzer(target_bit=8)


def test_dpa_initialization_invalid_byte():
    """Test DPA with invalid byte position."""
    with pytest.raises(ValueError, match="byte_position must be 0-15"):
        DPAAnalyzer(byte_position=16)


def test_dpa_selection_function():
    """Test DPA selection function."""
    dpa = DPAAnalyzer(target_bit=0)

    # S-box(0x00 ^ 0x00) = 0x63 = 0b01100011, bit 0 = 1
    assert dpa._selection_function(0x00, 0x00) == 1

    # S-box(0x00 ^ 0xFF) = S-box(0xFF) = 0x16 = 0b00010110, bit 0 = 0
    assert dpa._selection_function(0x00, 0xFF) == 0


def test_dpa_analyze_synthetic():
    """Test DPA with synthetic traces."""
    np.random.seed(42)

    n_traces = 500
    n_samples = 100
    true_key = 0x2A

    # Generate synthetic traces with power leakage
    traces = np.random.randn(n_traces, n_samples) * 0.1
    plaintexts = np.random.randint(0, 256, n_traces, dtype=np.uint8)

    # Inject leakage at sample 50
    leak_sample = 50
    for i in range(n_traces):
        sbox_out = AES_SBOX[plaintexts[i] ^ true_key]
        bit_value = (sbox_out >> 0) & 1
        traces[i, leak_sample] += 1.0 if bit_value else -1.0

    # Run DPA attack
    dpa = DPAAnalyzer(target_bit=0, byte_position=0)
    result = dpa.analyze(traces, plaintexts)

    # Should recover correct key
    assert result.key_guess == true_key
    assert result.max_differential > 0.5
    assert result.peak_sample == leak_sample
    assert result.differential_traces.shape == (256, n_samples)
    assert len(result.key_rank) == 256


def test_dpa_analyze_2d_plaintexts():
    """Test DPA with 2D plaintext array."""
    np.random.seed(42)

    n_traces = 200
    n_samples = 50

    traces = np.random.randn(n_traces, n_samples)
    plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)

    dpa = DPAAnalyzer(target_bit=0, byte_position=0)
    result = dpa.analyze(traces, plaintexts)

    # Should complete without error
    assert result.differential_traces.shape == (256, n_samples)
    assert 0 <= result.key_guess < 256


def test_dpa_analyze_shape_mismatch():
    """Test DPA with mismatched shapes."""
    traces = np.random.randn(100, 50)
    plaintexts = np.random.randint(0, 256, 50, dtype=np.uint8)  # Wrong length

    dpa = DPAAnalyzer()
    with pytest.raises(ValueError, match="must match traces"):
        dpa.analyze(traces, plaintexts)


# =============================================================================
# CPA tests
# =============================================================================


def test_cpa_initialization():
    """Test CPA analyzer initialization."""
    cpa = CPAAnalyzer(leakage_model="hamming_weight", algorithm="aes_sbox")
    assert cpa.leakage_model == "hamming_weight"
    assert cpa.algorithm == "aes_sbox"
    assert cpa.byte_position == 0


def test_cpa_initialization_invalid_model():
    """Test CPA with invalid leakage model."""
    with pytest.raises(ValueError, match="leakage_model must be one of"):
        CPAAnalyzer(leakage_model="invalid")


def test_cpa_initialization_invalid_byte():
    """Test CPA with invalid byte position."""
    with pytest.raises(ValueError, match="byte_position must be 0-15"):
        CPAAnalyzer(byte_position=16)


def test_cpa_compute_intermediate():
    """Test CPA intermediate value computation."""
    cpa = CPAAnalyzer(algorithm="aes_sbox")

    plaintext_bytes = np.array([0x00, 0x01, 0xFF], dtype=np.uint8)
    key_guess = 0x00

    intermediates = cpa._compute_intermediate(plaintext_bytes, key_guess)

    # Should be S-box output
    expected = np.array([AES_SBOX[0x00], AES_SBOX[0x01], AES_SBOX[0xFF]])
    np.testing.assert_array_equal(intermediates, expected)


def test_cpa_analyze_synthetic():
    """Test CPA with synthetic traces."""
    np.random.seed(123)

    n_traces = 500
    n_samples = 100
    true_key = 0x42

    # Generate synthetic traces with power leakage
    plaintexts = np.random.randint(0, 256, n_traces, dtype=np.uint8)
    traces = np.random.randn(n_traces, n_samples) * 0.5

    # Inject HW leakage at sample 60
    leak_sample = 60
    for i in range(n_traces):
        sbox_out = AES_SBOX[plaintexts[i] ^ true_key]
        hw = hamming_weight(sbox_out)
        traces[i, leak_sample] += hw * 0.5

    # Run CPA attack
    cpa = CPAAnalyzer(leakage_model="hamming_weight", algorithm="aes_sbox")
    result = cpa.analyze(traces, plaintexts)

    # Should recover correct key
    assert result.key_guess == true_key
    assert result.max_correlation > 0.3  # Should have reasonable correlation
    assert result.correlations.shape == (256, n_samples)
    assert len(result.key_rank) == 256


def test_cpa_analyze_2d_plaintexts():
    """Test CPA with 2D plaintext array."""
    np.random.seed(42)

    n_traces = 200
    n_samples = 50

    traces = np.random.randn(n_traces, n_samples)
    plaintexts = np.random.randint(0, 256, (n_traces, 16), dtype=np.uint8)

    cpa = CPAAnalyzer(byte_position=1)
    result = cpa.analyze(traces, plaintexts)

    # Should complete without error
    assert result.correlations.shape == (256, n_samples)
    assert 0 <= result.key_guess < 256


def test_cpa_analyze_identity_model():
    """Test CPA with identity leakage model."""
    np.random.seed(42)

    n_traces = 200
    n_samples = 50

    traces = np.random.randn(n_traces, n_samples)
    plaintexts = np.random.randint(0, 256, n_traces, dtype=np.uint8)

    cpa = CPAAnalyzer(leakage_model="identity")
    result = cpa.analyze(traces, plaintexts)

    assert result.correlations.shape == (256, n_samples)
    assert 0 <= result.key_guess < 256


def test_cpa_analyze_shape_mismatch():
    """Test CPA with mismatched shapes."""
    traces = np.random.randn(100, 50)
    plaintexts = np.random.randint(0, 256, 50, dtype=np.uint8)

    cpa = CPAAnalyzer()
    with pytest.raises(ValueError, match="must match traces"):
        cpa.analyze(traces, plaintexts)


def test_cpa_result_attributes():
    """Test CPA result has all expected attributes."""
    np.random.seed(42)

    traces = np.random.randn(100, 50)
    plaintexts = np.random.randint(0, 256, 100, dtype=np.uint8)

    cpa = CPAAnalyzer()
    result = cpa.analyze(traces, plaintexts)

    assert hasattr(result, "key_guess")
    assert hasattr(result, "max_correlation")
    assert hasattr(result, "correlations")
    assert hasattr(result, "key_rank")
    assert hasattr(result, "peak_sample")
    assert isinstance(result.key_guess, int)
    assert isinstance(result.max_correlation, float)
    assert isinstance(result.peak_sample, int)
