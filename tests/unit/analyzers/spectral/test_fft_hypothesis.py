"""Property-based tests for FFT and spectral analysis."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import analog_waveforms, frequency_data

pytestmark = [pytest.mark.unit, pytest.mark.spectral, pytest.mark.hypothesis]


class TestFFTProperties:
    """Property-based tests for FFT operations."""

    @given(signal_length=st.integers(min_value=64, max_value=4096))
    @settings(max_examples=30, deadline=None)
    def test_fft_length_matches_input(self, signal_length: int) -> None:
        """Property: FFT output length matches input for real signals."""
        rng = np.random.default_rng(42)
        signal = rng.random(signal_length)

        fft_result = np.fft.rfft(signal)

        # Real FFT returns N//2 + 1 frequencies
        expected_length = signal_length // 2 + 1
        assert len(fft_result) == expected_length

    @given(signal=analog_waveforms(min_length=128, max_length=1024))
    @settings(max_examples=30, deadline=None)
    def test_fft_inverse_recovers_signal(self, signal: np.ndarray) -> None:
        """Property: Inverse FFT recovers original signal."""
        fft_result = np.fft.fft(signal)
        recovered = np.fft.ifft(fft_result)

        assert np.allclose(signal, recovered.real, atol=1e-10)

    @given(freq_data=frequency_data())
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_frequency_magnitudes_non_negative(
        self, freq_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Property: Frequency magnitudes are non-negative."""
        frequencies, magnitudes = freq_data

        assert np.all(magnitudes >= 0)
