"""Property-based tests for correlation analysis."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestCorrelationProperties:
    """Property-based tests for correlation computation."""

    @given(length=st.integers(min_value=10, max_value=1000))
    @settings(max_examples=50, deadline=None)
    def test_perfect_correlation_with_self(self, length: int) -> None:
        """Property: Signal has perfect correlation with itself."""
        rng = np.random.default_rng(42)
        signal = rng.random(length)

        correlation = np.corrcoef(signal, signal)[0, 1]

        assert correlation == pytest.approx(1.0, abs=1e-10)

    @given(length=st.integers(min_value=10, max_value=1000))
    @settings(max_examples=50, deadline=None)
    def test_correlation_bounded_minus1_to_1(self, length: int) -> None:
        """Property: Correlation coefficient is between -1 and 1."""
        rng = np.random.default_rng(42)
        signal1 = rng.random(length)
        signal2 = rng.random(length)

        correlation = np.corrcoef(signal1, signal2)[0, 1]

        assert -1.0 <= correlation <= 1.0

    @given(length=st.integers(min_value=10, max_value=1000))
    @settings(max_examples=30, deadline=None)
    def test_negative_correlation_detected(self, length: int) -> None:
        """Property: Negative correlation is correctly identified."""
        rng = np.random.default_rng(42)
        signal = rng.random(length)
        # Create negatively correlated signal
        neg_signal = -signal

        correlation = np.corrcoef(signal, neg_signal)[0, 1]

        assert correlation == pytest.approx(-1.0, abs=1e-10)
