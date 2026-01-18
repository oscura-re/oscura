"""Property-based tests for outlier detection."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestOutlierDetectionProperties:
    """Property-based tests for outlier detection."""

    @given(
        num_samples=st.integers(min_value=100, max_value=1000),
        num_outliers=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_extreme_values_detected_as_outliers(self, num_samples: int, num_outliers: int) -> None:
        """Property: Extremely large values are detected as outliers."""
        rng = np.random.default_rng(42)
        # Normal data around mean=0, std=1
        normal_data = rng.normal(0, 1, num_samples - num_outliers)

        # Add extreme outliers (very far from mean)
        outliers = rng.uniform(100, 200, num_outliers)

        data = np.concatenate([normal_data, outliers])

        # Simple outlier detection: values > 3 std devs
        mean_val = np.mean(data)
        std_val = np.std(data)
        is_outlier = np.abs(data - mean_val) > 3 * std_val

        # Should detect at least some outliers
        assert np.sum(is_outlier) >= 1

    @given(
        num_samples=st.integers(min_value=100, max_value=1000),
        constant_value=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_no_outliers_in_constant_data(self, num_samples: int, constant_value: float) -> None:
        """Property: Constant data has no outliers."""
        data = np.full(num_samples, constant_value)

        # Z-score based outlier detection
        std_val = np.std(data)

        if std_val == 0:
            # No outliers in constant data
            num_outliers = 0
        else:
            mean_val = np.mean(data)
            is_outlier = np.abs(data - mean_val) > 3 * std_val
            num_outliers = np.sum(is_outlier)

        assert num_outliers == 0

    @given(num_samples=st.integers(min_value=50, max_value=500))
    @settings(max_examples=30, deadline=None)
    def test_gaussian_data_few_outliers(self, num_samples: int) -> None:
        """Property: Gaussian data has few outliers (< 1% for 3-sigma)."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, num_samples)

        # 3-sigma rule: ~99.7% of data within 3 std devs
        mean_val = np.mean(data)
        std_val = np.std(data)
        is_outlier = np.abs(data - mean_val) > 3 * std_val

        outlier_ratio = np.sum(is_outlier) / num_samples

        # Should be less than 1% outliers
        assert outlier_ratio < 0.01
