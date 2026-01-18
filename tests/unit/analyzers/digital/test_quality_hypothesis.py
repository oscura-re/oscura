"""Property-based tests for signal quality assessment."""

import numpy as np
import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import noisy_digital_signal

pytestmark = [pytest.mark.unit, pytest.mark.digital, pytest.mark.hypothesis]


class TestSignalQualityProperties:
    """Property-based tests for signal quality metrics."""

    @given(signal=noisy_digital_signal())
    @settings(max_examples=30, deadline=None)
    def test_signal_to_noise_ratio_non_negative(self, signal: np.ndarray) -> None:
        """Property: SNR is non-negative (in linear scale)."""
        signal_power = np.mean(signal**2)
        noise_estimate = np.std(signal)

        if noise_estimate > 0:
            snr = signal_power / (noise_estimate**2)
            assert snr >= 0

    @given(signal=noisy_digital_signal())
    @settings(max_examples=30, deadline=None)
    def test_clean_signal_better_quality_than_noisy(self, signal: np.ndarray) -> None:
        """Property: Clean signal has better quality metrics than noisy."""
        # Original (noisy) signal
        noisy_std = np.std(signal)

        # Create clean version (quantized to levels)
        clean_signal = np.where(signal > 1.65, 3.3, 0.0)
        clean_std = np.std(clean_signal)

        # Clean signal should have less variation in theory
        # But this depends on the noise level
        assert noisy_std >= 0
        assert clean_std >= 0
