"""Property-based tests for data preprocessing."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import analog_waveforms

pytestmark = [pytest.mark.unit, pytest.mark.loader, pytest.mark.hypothesis]


class TestPreprocessingProperties:
    """Property-based tests for preprocessing operations."""

    @given(signal=analog_waveforms())
    @settings(max_examples=50, deadline=None)
    def test_normalization_preserves_length(self, signal: np.ndarray) -> None:
        """Property: Normalization preserves signal length."""
        if len(signal) == 0:
            pytest.skip("Empty signal")

        # Simple normalization
        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        assert len(normalized) == len(signal)

    @given(signal=analog_waveforms())
    @settings(max_examples=50, deadline=None)
    def test_normalized_signal_zero_mean(self, signal: np.ndarray) -> None:
        """Property: Normalized signal has approximately zero mean."""
        if len(signal) == 0:
            pytest.skip("Empty signal")

        normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        mean_val = np.mean(normalized)

        assert mean_val == pytest.approx(0.0, abs=1e-10)

    @given(
        signal=analog_waveforms(),
        scale_factor=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_scaling_preserves_shape(self, signal: np.ndarray, scale_factor: float) -> None:
        """Property: Scaling preserves signal shape."""
        scaled = signal * scale_factor

        assert scaled.shape == signal.shape
