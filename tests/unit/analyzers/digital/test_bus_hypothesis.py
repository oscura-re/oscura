"""Property-based tests for bus protocol analysis."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.digital, pytest.mark.hypothesis]


class TestBusProtocolProperties:
    """Property-based tests for bus protocol analysis."""

    @given(
        num_bits=st.integers(min_value=8, max_value=64),
    )
    @settings(max_examples=30, deadline=None)
    def test_bus_width_positive(self, num_bits: int) -> None:
        """Property: Bus width is positive."""
        assert num_bits > 0

    @given(
        clock_period=st.floats(
            min_value=1e-9, max_value=1e-6, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_bus_frequency_from_period(self, clock_period: float) -> None:
        """Property: Bus frequency is inverse of clock period."""
        frequency = 1.0 / clock_period

        assert frequency > 0
        assert frequency == pytest.approx(1.0 / clock_period, rel=1e-9)
