"""Tests for QuickSpectral."""

from __future__ import annotations

import numpy as np
import pytest

from oscura.convenience import SpectralMetrics, quick_spectral
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit

pytestmark = pytest.mark.unit


class TestQuickSpectral:
    """Tests for the quick_spectral convenience function."""

    def test_basic_spectral_analysis(self) -> None:
        """Test basic spectral analysis."""
        # Generate clean sine wave
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 1000
        data = np.sin(2 * np.pi * fundamental * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace, fundamental=fundamental)

        assert isinstance(metrics, SpectralMetrics)
        # Clean sine should have excellent metrics
        assert metrics.thd_db < -60 or np.isnan(metrics.thd_db)  # Very low THD
        assert metrics.fundamental_freq == pytest.approx(fundamental, rel=0.1)

    def test_spectral_with_harmonics(self) -> None:
        """Test spectral analysis with harmonic distortion."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 1000

        # Add 1% THD
        data = np.sin(2 * np.pi * fundamental * t)
        data += 0.01 * np.sin(2 * np.pi * 2 * fundamental * t)  # 2nd harmonic

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace, fundamental=fundamental)

        # THD should be around -40 dB (1%)
        assert -50 < metrics.thd_db < -30

    def test_auto_fundamental_detection(self) -> None:
        """Test automatic fundamental frequency detection."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        fundamental = 2500

        data = np.sin(2 * np.pi * fundamental * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace)  # No fundamental specified

        # Should auto-detect fundamental
        assert 2000 < metrics.fundamental_freq < 3000

    def test_returns_all_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        sample_rate = 1e6
        n_samples = 10000
        t = np.arange(n_samples) / sample_rate
        data = np.sin(2 * np.pi * 1000 * t)

        trace = WaveformTrace(
            data=data, metadata=TraceMetadata(sample_rate=sample_rate, channel_name="test")
        )

        metrics = quick_spectral(trace)

        # All fields should be present
        assert hasattr(metrics, "thd_db")
        assert hasattr(metrics, "thd_percent")
        assert hasattr(metrics, "snr_db")
        assert hasattr(metrics, "sinad_db")
        assert hasattr(metrics, "enob")
        assert hasattr(metrics, "sfdr_db")
        assert hasattr(metrics, "fundamental_freq")
        assert hasattr(metrics, "fundamental_mag_db")
        assert hasattr(metrics, "noise_floor_db")
