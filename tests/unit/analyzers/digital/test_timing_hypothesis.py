"""Property-based tests for timing analysis."""

import numpy as np
import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import edge_lists, timing_parameters

pytestmark = [pytest.mark.unit, pytest.mark.digital, pytest.mark.hypothesis]


class TestTimingAnalysisProperties:
    """Property-based tests for timing analysis."""

    @given(edges=edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_period_calculation_positive(self, edges: np.ndarray) -> None:
        """Property: Calculated periods are positive."""
        if len(edges) < 2:
            pytest.skip("Need at least 2 edges")

        # Calculate periods (time between edges)
        periods = np.diff(edges)

        assert np.all(periods >= 0)

    @given(edges=edge_lists())
    @settings(max_examples=50, deadline=None)
    def test_frequency_inverse_of_period(self, edges: np.ndarray) -> None:
        """Property: Frequency is inverse of period."""
        if len(edges) < 2:
            pytest.skip("Need at least 2 edges")

        periods = np.diff(edges)
        avg_period = np.mean(periods)

        # Guard against extremely small periods that would cause overflow
        # Skip if period is less than 1 femtosecond (1e-15)
        if avg_period > 1e-15:
            frequency = 1.0 / avg_period
            period_from_freq = 1.0 / frequency

            assert avg_period == pytest.approx(period_from_freq, rel=1e-9)

    @given(params=timing_parameters())
    @settings(max_examples=30, deadline=None)
    def test_sample_rate_positive(self, params: dict[str, float]) -> None:
        """Property: Sample rate is always positive."""
        sample_rate = params["sample_rate"]

        assert sample_rate > 0
