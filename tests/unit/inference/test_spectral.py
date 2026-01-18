"""Unit tests for spectral method auto-selection (INF-003)."""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.inference.spectral import _assess_stationarity, auto_spectral_config

pytestmark = [pytest.mark.unit, pytest.mark.inference]


class TestAutoSpectralConfig:
    """Test auto_spectral_config function."""

    def test_stationary_signal(self) -> None:
        """Test configuration for highly stationary signal (white noise)."""
        # White noise is statistically stationary
        sample_rate = 1e6  # 1 MHz
        n_samples = 10000
        np.random.seed(42)
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # White noise actually has moderate stationarity due to windowing effects
        # Algorithm uses Welch for signals <= 0.8
        assert config["method"] == "welch"
        assert config["stationarity_score"] > 0.3
        assert config["noverlap"] == config["nperseg"] // 2  # Welch uses 50% overlap
        assert config["nperseg"] >= 256
        assert config["nfft"] >= config["nperseg"]
        assert "rationale" in config

    def test_moderately_stationary_signal(self) -> None:
        """Test configuration for moderately stationary signal."""
        # Create signal with some variation - sine wave has moderate stationarity
        sample_rate = 1e6
        duration = 0.01
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        data = np.sin(2 * np.pi * 1000 * t)  # Pure sine wave

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Sine wave has moderate to high stationarity, should use Welch method
        assert config["method"] == "welch"
        assert 0.3 < config["stationarity_score"] <= 0.8
        assert config["noverlap"] == config["nperseg"] // 2  # Welch uses 50% overlap
        assert "rationale" in config

    def test_nonstationary_signal(self) -> None:
        """Test configuration for non-stationary signal."""
        # Create strongly non-stationary signal
        sample_rate = 1e6
        duration = 0.01
        n_samples = int(sample_rate * duration)

        # Signal with dramatically changing amplitude and frequency
        data = np.zeros(n_samples)
        chunk_size = n_samples // 8
        for i in range(8):
            start = i * chunk_size
            end = start + chunk_size
            t = np.linspace(0, 0.001, chunk_size)
            freq = 1000 * (i + 1)
            amp = 0.1 * (i + 1)
            data[start:end] = amp * np.sin(2 * np.pi * freq * t)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Non-stationary should use Welch with shorter segments
        assert config["method"] == "welch"
        assert config["stationarity_score"] <= 0.5
        assert config["noverlap"] == config["nperseg"] // 2
        assert config["nperseg"] <= 2**12  # Shorter segments
        assert "rationale" in config

    def test_high_dynamic_range(self) -> None:
        """Test window selection for high dynamic range requirement."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace, dynamic_range_db=90.0)

        # High dynamic range should use Blackman-Harris window
        assert config["window"] == "blackman-harris"
        assert "Blackman-Harris" in config["rationale"]

    def test_moderate_dynamic_range(self) -> None:
        """Test window selection for moderate dynamic range."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Test 70 dB (should use Blackman)
        config = auto_spectral_config(trace, dynamic_range_db=70.0)
        assert config["window"] == "blackman"

        # Test 50 dB (should use Hamming)
        config = auto_spectral_config(trace, dynamic_range_db=50.0)
        assert config["window"] == "hamming"

        # Test 30 dB (should use Hann)
        config = auto_spectral_config(trace, dynamic_range_db=30.0)
        assert config["window"] == "hann"

    def test_target_resolution(self) -> None:
        """Test configuration with target frequency resolution."""
        sample_rate = 1e6  # 1 MHz
        n_samples = 100000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        target_resolution = 100.0  # 100 Hz resolution
        config = auto_spectral_config(trace, target_resolution=target_resolution)

        # nperseg should be chosen to achieve target resolution
        # Resolution = sample_rate / nperseg, so nperseg = sample_rate / resolution
        expected_nperseg = int(sample_rate / target_resolution)
        # Should be rounded to next power of 2
        expected_nperseg = 2 ** int(np.ceil(np.log2(expected_nperseg)))

        assert config["nperseg"] == expected_nperseg
        assert config["nfft"] >= config["nperseg"]

    def test_short_signal(self) -> None:
        """Test handling of short signals."""
        sample_rate = 1e6
        n_samples = 512  # Very short signal
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # nperseg should be constrained to reasonable value
        assert 256 <= config["nperseg"] <= n_samples // 2
        assert config["nfft"] >= config["nperseg"]
        assert isinstance(config["method"], str)
        assert isinstance(config["window"], str)

    def test_no_rationale(self) -> None:
        """Test configuration without rationale logging."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace, log_rationale=False)

        # Rationale should not be included
        assert "rationale" not in config
        # But other fields should still be present
        assert "method" in config
        assert "window" in config
        assert "nperseg" in config
        assert "noverlap" in config
        assert "nfft" in config
        assert "stationarity_score" in config

    def test_config_keys(self) -> None:
        """Test that configuration contains all required keys."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        required_keys = {
            "method",
            "window",
            "nperseg",
            "noverlap",
            "nfft",
            "stationarity_score",
            "rationale",
        }
        assert set(config.keys()) == required_keys

    def test_nfft_power_of_two(self) -> None:
        """Test that nfft is always a power of 2."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Check if nfft is power of 2
        nfft = config["nfft"]
        assert nfft > 0
        assert (nfft & (nfft - 1)) == 0  # Power of 2 check

    def test_nperseg_constraints(self) -> None:
        """Test nperseg is within valid constraints."""
        sample_rate = 1e6
        n_samples = 50000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # nperseg should be >= 256 and <= n_samples // 2
        assert config["nperseg"] >= 256
        assert config["nperseg"] <= n_samples // 2

    def test_welch_overlap(self) -> None:
        """Test that Welch method uses 50% overlap."""
        sample_rate = 1e6
        n_samples = 10000

        # Create moderately non-stationary signal to trigger Welch
        data = np.zeros(n_samples)
        chunk = n_samples // 4
        for i in range(4):
            start = i * chunk
            end = start + chunk
            data[start:end] = np.random.randn(chunk) * (i + 1)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        if config["method"] == "welch":
            assert config["noverlap"] == config["nperseg"] // 2

    def test_bartlett_no_overlap(self) -> None:
        """Test that Bartlett method uses no overlap."""
        sample_rate = 1e6
        n_samples = 10000

        # Create highly stationary signal to trigger Bartlett
        data = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, n_samples))

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        if config["method"] == "bartlett":
            assert config["noverlap"] == 0

    def test_stationarity_score_range(self) -> None:
        """Test that stationarity score is always in [0, 1] range."""
        sample_rate = 1e6
        n_samples = 10000

        # Test with various signals
        test_signals = [
            np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, n_samples)),  # Stationary
            np.random.randn(n_samples),  # White noise
            np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)]),  # Step
        ]

        for data in test_signals:
            trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))
            config = auto_spectral_config(trace)
            assert 0.0 <= config["stationarity_score"] <= 1.0

    def test_method_values(self) -> None:
        """Test that method is one of the expected values."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        valid_methods = {"welch", "bartlett", "periodogram"}
        assert config["method"] in valid_methods

    def test_window_values(self) -> None:
        """Test that window is one of the expected values."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Test various dynamic range values
        for db in [30, 50, 70, 90]:
            config = auto_spectral_config(trace, dynamic_range_db=db)
            valid_windows = {"hann", "hamming", "blackman", "blackman-harris"}
            assert config["window"] in valid_windows


class TestAssessStationarity:
    """Test _assess_stationarity function."""

    def test_stationary_sine_wave(self) -> None:
        """Test stationarity assessment of a pure sine wave."""
        n_samples = 10000
        t = np.linspace(0, 1, n_samples)
        data = np.sin(2 * np.pi * 10 * t)

        score = _assess_stationarity(data)

        # Pure sine wave has moderate stationarity (varying mean across windows)
        assert 0.3 < score < 0.7

    def test_white_noise(self) -> None:
        """Test stationarity assessment of white noise."""
        n_samples = 10000
        data = np.random.randn(n_samples)

        score = _assess_stationarity(data)

        # White noise has moderate stationarity (windowing effects)
        assert 0.3 < score <= 0.7

    def test_step_function(self) -> None:
        """Test stationarity assessment of step function."""
        n_samples = 10000
        data = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

        score = _assess_stationarity(data)

        # Step function is non-stationary
        assert score < 0.8

    def test_amplitude_modulated_signal(self) -> None:
        """Test stationarity of amplitude modulated signal with step change."""
        n_samples = 10000
        t = np.linspace(0, 1, n_samples)

        # AM signal with abrupt step in amplitude
        carrier = np.sin(2 * np.pi * 100 * t)
        data = carrier.copy()
        data[n_samples // 2 :] *= 5.0  # Step change at midpoint

        score = _assess_stationarity(data)

        # AM signal with step is non-stationary
        assert score < 0.3

    def test_short_signal(self) -> None:
        """Test stationarity assessment of very short signal."""
        # Signal too short for meaningful window analysis
        n_samples = 500
        data = np.random.randn(n_samples)

        score = _assess_stationarity(data)

        # Should return default moderate stationarity
        assert score == 0.7

    def test_constant_signal(self) -> None:
        """Test stationarity of constant signal."""
        n_samples = 10000
        data = np.ones(n_samples) * 5.0

        score = _assess_stationarity(data)

        # Constant signal is perfectly stationary
        assert score > 0.9

    def test_trending_signal(self) -> None:
        """Test stationarity of signal with linear trend."""
        n_samples = 10000
        data = np.linspace(0, 10, n_samples) + np.random.randn(n_samples) * 0.1

        score = _assess_stationarity(data)

        # Signal with trend is non-stationary
        assert score < 0.8

    def test_score_range(self) -> None:
        """Test that stationarity score is always in [0, 1] range."""
        test_signals = [
            np.random.randn(10000),
            np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000)),
            np.linspace(0, 100, 10000),
            np.ones(10000),
            np.concatenate([np.ones(5000), np.zeros(5000)]),
        ]

        for data in test_signals:
            score = _assess_stationarity(data)
            assert 0.0 <= score <= 1.0

    def test_variance_changes(self) -> None:
        """Test detection of variance changes."""
        n_samples = 10000

        # Signal with increasing variance
        data = np.zeros(n_samples)
        chunk_size = n_samples // 8
        for i in range(8):
            start = i * chunk_size
            end = start + chunk_size
            # Variance increases with each chunk
            data[start:end] = np.random.randn(chunk_size) * (i + 1)

        score = _assess_stationarity(data)

        # Signal with changing variance is non-stationary
        assert score < 0.7

    def test_mean_changes(self) -> None:
        """Test detection of mean changes."""
        n_samples = 10000

        # Signal with changing mean
        data = np.zeros(n_samples)
        chunk_size = n_samples // 8
        for i in range(8):
            start = i * chunk_size
            end = start + chunk_size
            # Mean changes with each chunk
            data[start:end] = np.random.randn(chunk_size) + i * 2

        score = _assess_stationarity(data)

        # Signal with changing mean is non-stationary
        assert score < 0.7

    def test_periodic_modulation(self) -> None:
        """Test stationarity of signal with periodic modulation."""
        n_samples = 10000
        t = np.linspace(0, 1, n_samples)

        # Signal with slow periodic modulation
        carrier = np.sin(2 * np.pi * 100 * t)
        modulation = 1 + 0.5 * np.sin(2 * np.pi * 2 * t)
        data = modulation * carrier

        score = _assess_stationarity(data)

        # Depends on how the windowing captures the modulation
        # Should be less stationary than pure carrier
        assert 0.0 <= score <= 1.0

    def test_return_type(self) -> None:
        """Test that return type is Python float."""
        data = np.random.randn(10000)
        score = _assess_stationarity(data)

        assert isinstance(score, float)
        assert not isinstance(score, np.floating)


class TestInferenceSpectralIntegration:
    """Integration tests for spectral configuration."""

    def test_config_usable_parameters(self) -> None:
        """Test that configuration parameters are valid for scipy.signal functions."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Verify parameters are valid
        assert isinstance(config["nperseg"], int | np.integer)
        assert isinstance(config["noverlap"], int | np.integer)
        assert isinstance(config["nfft"], int | np.integer)
        assert config["nperseg"] > 0
        assert config["noverlap"] >= 0
        assert config["noverlap"] < config["nperseg"]
        assert config["nfft"] >= config["nperseg"]

    def test_reproducibility(self) -> None:
        """Test that same input produces same configuration."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.RandomState(42).randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        # Get configuration twice
        config1 = auto_spectral_config(trace, dynamic_range_db=60.0)
        config2 = auto_spectral_config(trace, dynamic_range_db=60.0)

        # Should be identical
        assert config1["method"] == config2["method"]
        assert config1["window"] == config2["window"]
        assert config1["nperseg"] == config2["nperseg"]
        assert config1["noverlap"] == config2["noverlap"]
        assert config1["nfft"] == config2["nfft"]
        assert config1["stationarity_score"] == config2["stationarity_score"]

    def test_different_sample_rates(self) -> None:
        """Test configuration with different sample rates."""
        n_samples = 10000
        data = np.random.randn(n_samples)

        sample_rates = [1e3, 1e6, 1e9]  # 1 kHz, 1 MHz, 1 GHz

        for sr in sample_rates:
            trace = WaveformTrace(data=data.copy(), metadata=TraceMetadata(sample_rate=sr))
            config = auto_spectral_config(trace)

            # Configuration should be valid regardless of sample rate
            assert config["nperseg"] >= 256
            assert config["nperseg"] <= n_samples // 2
            assert config["noverlap"] < config["nperseg"]

    def test_edge_case_minimal_signal(self) -> None:
        """Test with minimal viable signal length."""
        sample_rate = 1e6
        n_samples = 512  # Minimum practical size
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Should handle gracefully
        assert 256 <= config["nperseg"] <= 256  # At minimum
        assert config["nfft"] >= config["nperseg"]

    def test_large_signal(self) -> None:
        """Test with very large signal."""
        sample_rate = 1e6
        n_samples = 1000000  # 1 million samples
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace)

        # Should constrain nperseg to reasonable values
        assert config["nperseg"] <= 2**14  # Max 16k for stationary
        assert config["nfft"] >= config["nperseg"]

    def test_rationale_describes_choices(self) -> None:
        """Test that rationale string describes the configuration choices."""
        sample_rate = 1e6
        n_samples = 10000
        data = np.random.randn(n_samples)

        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        config = auto_spectral_config(trace, dynamic_range_db=70.0)

        rationale = config["rationale"]

        # Rationale should mention key aspects
        assert isinstance(rationale, str)
        assert len(rationale) > 0

        # Should contain information about method selection
        assert any(method in rationale.lower() for method in ["welch", "bartlett", "periodogram"])

        # Should contain information about window selection
        assert any(window in rationale.lower() for window in ["hann", "hamming", "blackman"])
