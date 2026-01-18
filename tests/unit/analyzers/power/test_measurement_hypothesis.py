"""Property-based tests for power analysis."""

import numpy as np
import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import power_traces

pytestmark = [pytest.mark.unit, pytest.mark.power, pytest.mark.hypothesis]


class TestPowerMeasurementProperties:
    """Property-based tests for power measurement."""

    @given(power=power_traces())
    @settings(max_examples=30, deadline=None)
    def test_power_values_non_negative(self, power: np.ndarray) -> None:
        """Property: Power measurements are non-negative."""
        assert np.all(power >= 0)

    @given(power=power_traces())
    @settings(max_examples=30, deadline=None)
    def test_average_power_non_negative(self, power: np.ndarray) -> None:
        """Property: Average power is non-negative."""
        if len(power) == 0:
            pytest.skip("Empty power trace")

        avg_power = np.mean(power)

        assert avg_power >= 0

    @given(power=power_traces())
    @settings(max_examples=30, deadline=None)
    def test_peak_power_greater_than_average(self, power: np.ndarray) -> None:
        """Property: Peak power >= average power."""
        if len(power) == 0:
            pytest.skip("Empty power trace")

        avg_power = np.mean(power)
        peak_power = np.max(power)

        # Use np.isclose to handle floating point precision issues
        assert peak_power >= avg_power or np.isclose(peak_power, avg_power, rtol=1e-9)
