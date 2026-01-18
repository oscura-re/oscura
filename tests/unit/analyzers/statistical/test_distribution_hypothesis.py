"""Property-based tests for distribution analysis."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import distribution_samples

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestDistributionAnalysisProperties:
    """Property-based tests for distribution analysis."""

    @given(samples=distribution_samples())
    @settings(max_examples=50, deadline=None)
    def test_mean_within_data_range(self, samples: np.ndarray) -> None:
        """Property: Mean is within min-max range of data."""
        if len(samples) == 0:
            pytest.skip("Empty samples")

        mean_val = np.mean(samples)
        min_val = np.min(samples)
        max_val = np.max(samples)

        assert min_val <= mean_val <= max_val

    @given(samples=distribution_samples())
    @settings(max_examples=50, deadline=None)
    def test_standard_deviation_non_negative(self, samples: np.ndarray) -> None:
        """Property: Standard deviation is never negative."""
        if len(samples) == 0:
            pytest.skip("Empty samples")

        std_dev = np.std(samples)

        assert std_dev >= 0.0

    @given(
        num_samples=st.integers(min_value=100, max_value=1000),
        constant_value=st.floats(
            min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_constant_data_zero_variance(self, num_samples: int, constant_value: float) -> None:
        """Property: Constant data has zero variance."""
        samples = np.full(num_samples, constant_value)

        variance = np.var(samples)

        assert variance == pytest.approx(0.0, abs=1e-10)

    @given(samples=distribution_samples())
    @settings(max_examples=50, deadline=None)
    def test_variance_equals_std_squared(self, samples: np.ndarray) -> None:
        """Property: Variance equals standard deviation squared."""
        if len(samples) == 0:
            pytest.skip("Empty samples")

        variance = np.var(samples)
        std_dev = np.std(samples)

        assert variance == pytest.approx(std_dev**2, rel=1e-9)
