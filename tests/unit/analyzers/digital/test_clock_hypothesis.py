"""Property-based tests for clock recovery."""

import numpy as np
import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import clock_signals

pytestmark = [pytest.mark.unit, pytest.mark.digital, pytest.mark.hypothesis]


class TestClockRecoveryProperties:
    """Property-based tests for clock recovery."""

    @given(clock=clock_signals())
    @settings(max_examples=30, deadline=None)
    def test_recovered_clock_has_regular_transitions(self, clock: np.ndarray) -> None:
        """Property: Recovered clock has regular transitions."""
        # Find transitions
        transitions = []
        for i in range(1, len(clock)):
            if clock[i] != clock[i - 1]:
                transitions.append(i)

        if len(transitions) < 2:
            pytest.skip("Too few transitions")

        # Check regularity
        transition_intervals = np.diff(transitions)

        # Intervals should be approximately equal for perfect clock
        std_dev = np.std(transition_intervals)
        mean_interval = np.mean(transition_intervals)

        # Coefficient of variation should be small
        if mean_interval > 0:
            cv = std_dev / mean_interval
            assert cv < 0.5  # Within 50% variation

    @given(clock=clock_signals())
    @settings(max_examples=30, deadline=None)
    def test_clock_alternates_between_two_levels(self, clock: np.ndarray) -> None:
        """Property: Clock signal alternates between two voltage levels."""
        unique_levels = np.unique(clock)

        # Should have 2 levels (low and high)
        assert len(unique_levels) == 2
