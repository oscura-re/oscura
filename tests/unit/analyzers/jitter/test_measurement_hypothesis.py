"""Property-based tests for jitter measurement."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import jitter_samples

pytestmark = [pytest.mark.unit, pytest.mark.jitter, pytest.mark.hypothesis]


class TestJitterMeasurementProperties:
    """Property-based tests for jitter measurement."""

    @given(jitter=jitter_samples())
    @settings(max_examples=50, deadline=None)
    def test_rms_jitter_non_negative(self, jitter: np.ndarray) -> None:
        """Property: RMS jitter is never negative."""
        rms_jitter = np.sqrt(np.mean(jitter**2))

        assert rms_jitter >= 0

    @given(jitter=jitter_samples())
    @settings(max_examples=50, deadline=None)
    def test_peak_to_peak_jitter_non_negative(self, jitter: np.ndarray) -> None:
        """Property: Peak-to-peak jitter is non-negative."""
        if len(jitter) == 0:
            pytest.skip("Empty jitter samples")

        pk_pk_jitter = np.max(jitter) - np.min(jitter)

        assert pk_pk_jitter >= 0

    @given(
        num_samples=st.integers(min_value=100, max_value=1000),
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_jitter_for_perfect_clock(self, num_samples: int) -> None:
        """Property: Perfect clock has zero jitter."""
        # Perfect clock - no jitter
        jitter = np.zeros(num_samples)

        rms_jitter = np.sqrt(np.mean(jitter**2))
        pk_pk_jitter = np.max(jitter) - np.min(jitter)

        assert rms_jitter == pytest.approx(0.0, abs=1e-15)
        assert pk_pk_jitter == pytest.approx(0.0, abs=1e-15)
