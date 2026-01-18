"""Property-based tests for entropy analysis.

Tests Shannon entropy calculation, entropy profiling, and data classification.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from oscura.analyzers.statistical import (
    calculate_entropy,
    classify_by_entropy,
)
from tests.hypothesis_strategies import entropy_data

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestEntropyCalculationProperties:
    """Property-based tests for entropy calculation."""

    @given(data_length=st.integers(min_value=100, max_value=10000))
    @settings(max_examples=50, deadline=None)
    def test_entropy_bounded_0_to_8(self, data_length: int) -> None:
        """Property: Entropy is always between 0 and log2(alphabet_size)."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, data_length, dtype=np.uint8).tobytes()

        entropy = calculate_entropy(data)

        # For byte data, max entropy is log2(256) = 8 bits
        assert 0.0 <= entropy <= 8.0

    @given(length=st.integers(min_value=100, max_value=1000))
    @settings(max_examples=50, deadline=None)
    def test_constant_data_zero_entropy(self, length: int) -> None:
        """Property: Constant data has zero entropy."""
        # All same byte value
        data = bytes([42] * length)

        entropy = calculate_entropy(data)

        assert entropy == pytest.approx(0.0, abs=1e-6)

    @given(
        length=st.integers(min_value=1000, max_value=10000),
        byte_value=st.integers(min_value=0, max_value=255),
    )
    @settings(max_examples=50, deadline=None)
    def test_single_byte_entropy_zero(self, length: int, byte_value: int) -> None:
        """Property: Data with single unique byte has entropy 0."""
        data = bytes([byte_value] * length)

        entropy = calculate_entropy(data)

        assert entropy == pytest.approx(0.0, abs=1e-6)

    @given(length=st.integers(min_value=2048, max_value=10000))
    @settings(max_examples=30, deadline=None)
    def test_random_data_high_entropy(self, length: int) -> None:
        """Property: Random data has high entropy (close to 8 bits)."""
        rng = np.random.default_rng(42)
        # Use larger sample for better approximation to uniform distribution
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()

        entropy = calculate_entropy(data)

        # Random data should have entropy > 7.0 bits
        # (Perfect uniform would be 8.0, but random sampling varies)
        assert entropy > 7.0

    @given(data=entropy_data())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
    def test_entropy_non_negative(self, data: bytes) -> None:
        """Property: Entropy is never negative."""
        entropy = calculate_entropy(data)

        assert entropy >= 0.0

    @given(
        length=st.integers(min_value=100, max_value=1000),
    )
    @settings(max_examples=30, deadline=None)
    def test_entropy_deterministic(self, length: int) -> None:
        """Property: Same data always produces same entropy."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()

        entropy1 = calculate_entropy(data)
        entropy2 = calculate_entropy(data)
        entropy3 = calculate_entropy(data)

        assert entropy1 == entropy2 == entropy3

    @given(
        data=st.binary(min_size=256, max_size=1000),
    )
    @settings(max_examples=30, deadline=None)
    def test_entropy_accepts_bytes_and_arrays(self, data: bytes) -> None:
        """Property: Entropy calculation accepts both bytes and arrays."""
        data_array = np.frombuffer(data, dtype=np.uint8)

        entropy_bytes = calculate_entropy(data)
        entropy_array = calculate_entropy(data_array)

        assert entropy_bytes == pytest.approx(entropy_array, abs=1e-9)


class TestDataClassificationProperties:
    """Property-based tests for data classification based on entropy."""

    @given(length=st.integers(min_value=100, max_value=1000))
    @settings(max_examples=30, deadline=None)
    def test_constant_data_classified_correctly(self, length: int) -> None:
        """Property: Constant data is classified as 'constant'."""
        data = bytes([0xFF] * length)

        result = classify_by_entropy(data)

        assert result.classification == "constant"
        assert result.confidence >= 0.9

    @given(length=st.integers(min_value=1000, max_value=5000))
    @settings(max_examples=30, deadline=None)
    def test_random_data_classified_as_random_or_compressed(self, length: int) -> None:
        """Property: Random data classified as random or compressed."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()

        result = classify_by_entropy(data)

        # High entropy data should be random or compressed
        assert result.classification in ["random", "compressed"]

    @given(data=entropy_data())
    @settings(max_examples=50, deadline=None)
    def test_classification_confidence_bounded(self, data: bytes) -> None:
        """Property: Classification confidence is between 0 and 1."""
        result = classify_by_entropy(data)

        assert 0.0 <= result.confidence <= 1.0

    @given(data=entropy_data())
    @settings(max_examples=50, deadline=None)
    def test_classification_types_valid(self, data: bytes) -> None:
        """Property: Classification is one of valid types."""
        result = classify_by_entropy(data)

        valid_types = {"structured", "text", "compressed", "random", "constant"}
        assert result.classification in valid_types


class TestEntropyEdgeCases:
    """Edge case tests for entropy calculation."""

    def test_empty_data_raises_error(self) -> None:
        """Property: Empty data raises ValueError."""
        data = b""

        with pytest.raises(ValueError, match="Cannot calculate entropy of empty data"):
            calculate_entropy(data)

    def test_single_byte_zero_entropy(self) -> None:
        """Property: Single byte has zero entropy."""
        data = b"A"

        entropy = calculate_entropy(data)

        assert entropy == 0.0

    @given(
        num_values=st.integers(min_value=2, max_value=256),
        length=st.integers(min_value=1000, max_value=5000),
    )
    @settings(max_examples=30, deadline=None)
    def test_uniform_distribution_maximum_entropy(self, num_values: int, length: int) -> None:
        """Property: Uniform distribution maximizes entropy."""
        # Create perfectly uniform distribution
        rng = np.random.default_rng(42)
        data = rng.choice(num_values, size=length).astype(np.uint8).tobytes()

        entropy = calculate_entropy(data)

        # Expected max entropy for num_values symbols
        expected_max = np.log2(num_values)

        # Should be close to maximum (within 0.5 bits due to sampling)
        assert entropy >= expected_max - 0.5
        assert entropy <= expected_max + 0.1
