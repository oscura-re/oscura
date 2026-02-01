"""Tests for SmartFilter."""

from __future__ import annotations

import numpy as np
import pytest

import oscura as osc
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


class TestSmartFilter:
    """Tests for the smart_filter convenience function."""

    def test_noise_filtering(self) -> None:
        """Test general noise filtering."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate

        # Clean signal with noise
        clean = np.sin(2 * np.pi * 1000 * t)
        noisy = clean + 0.3 * np.random.randn(n_samples)

        trace = WaveformTrace(
            data=noisy, metadata=TraceMetadata(sample_rate=sample_rate, channel="test")
        )

        filtered = osc.smart_filter(trace, target="noise")

        # Filtered should have lower high-frequency content
        orig_hf = np.std(np.diff(noisy))
        filt_hf = np.std(np.diff(filtered.data))
        assert filt_hf < orig_hf

    def test_60hz_hum_filtering(self) -> None:
        """Test 60 Hz hum filtering."""
        sample_rate = 10000
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate

        # Signal with 60 Hz hum
        signal_clean = np.sin(2 * np.pi * 1000 * t)
        hum = 0.2 * np.sin(2 * np.pi * 60 * t)
        noisy = signal_clean + hum

        trace = WaveformTrace(
            data=noisy, metadata=TraceMetadata(sample_rate=sample_rate, channel="test")
        )

        filtered = osc.smart_filter(trace, target="60hz_hum")

        # Check 60 Hz component is reduced
        fft_orig = np.abs(np.fft.rfft(noisy))
        fft_filt = np.abs(np.fft.rfft(filtered.data))
        freqs = np.fft.rfftfreq(n_samples, 1 / sample_rate)

        idx_60 = np.argmin(np.abs(freqs - 60))

        # 60 Hz should be significantly reduced
        assert fft_filt[idx_60] < fft_orig[idx_60] * 0.5
