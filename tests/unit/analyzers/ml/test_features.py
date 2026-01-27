"""Tests for ML feature extraction."""

import numpy as np
import pytest

from oscura.analyzers.ml.features import FeatureExtractor


class TestFeatureExtractor:
    """Test feature extraction for ML classification."""

    @pytest.fixture
    def extractor(self) -> FeatureExtractor:
        """Create feature extractor."""
        return FeatureExtractor()

    @pytest.fixture
    def sine_wave(self) -> tuple[np.ndarray, float]:
        """Generate 1 kHz sine wave at 10 kHz sample rate."""
        sample_rate = 10000.0
        t = np.linspace(0, 1, int(sample_rate))
        signal = np.sin(2 * np.pi * 1000 * t)
        return signal, sample_rate

    @pytest.fixture
    def square_wave(self) -> tuple[np.ndarray, float]:
        """Generate 100 Hz square wave at 10 kHz sample rate."""
        sample_rate = 10000.0
        t = np.linspace(0, 1, int(sample_rate))
        signal = np.sign(np.sin(2 * np.pi * 100 * t))
        return signal, sample_rate

    @pytest.fixture
    def noise(self) -> tuple[np.ndarray, float]:
        """Generate random noise."""
        sample_rate = 10000.0
        signal = np.random.randn(int(sample_rate))
        return signal, sample_rate

    def test_extract_all(
        self, extractor: FeatureExtractor, sine_wave: tuple[np.ndarray, float]
    ) -> None:
        """Test extracting all features."""
        signal, sample_rate = sine_wave
        features = extractor.extract_all(signal, sample_rate)

        # Should have 30+ features (statistical, spectral, temporal, entropy, shape)
        assert len(features) >= 30, f"Expected ≥30 features, got {len(features)}"

        # All values should be finite
        for name, value in features.items():
            assert np.isfinite(value), f"Feature {name} is not finite: {value}"

    def test_statistical_features(
        self, extractor: FeatureExtractor, sine_wave: tuple[np.ndarray, float]
    ) -> None:
        """Test statistical feature extraction."""
        signal, _ = sine_wave
        features = extractor.extract_statistical(signal)

        # Check expected features are present
        expected = {
            "mean",
            "std",
            "variance",
            "min",
            "max",
            "range",
            "skewness",
            "kurtosis",
            "median",
        }
        assert expected.issubset(features.keys())

        # Sine wave should have near-zero mean
        assert abs(features["mean"]) < 0.1

        # Sine wave has known variance
        expected_std = 1.0 / np.sqrt(2)
        assert abs(features["std"] - expected_std) < 0.1

    def test_spectral_features(
        self, extractor: FeatureExtractor, sine_wave: tuple[np.ndarray, float]
    ) -> None:
        """Test spectral feature extraction."""
        signal, sample_rate = sine_wave
        features = extractor.extract_spectral(signal, sample_rate)

        # Check expected features
        expected = {
            "dominant_frequency",
            "spectral_centroid",
            "bandwidth",
            "spectral_energy",
            "spectral_flatness",
            "spectral_rolloff",
            "num_spectral_peaks",
            "spectral_spread",
        }
        assert expected.issubset(features.keys())

        # Dominant frequency should be near 1000 Hz
        assert 900 < features["dominant_frequency"] < 1100

        # Spectral centroid should be reasonably close (allow wider tolerance due to FFT bins)
        assert 800 < features["spectral_centroid"] < 1500

    def test_temporal_features(
        self, extractor: FeatureExtractor, square_wave: tuple[np.ndarray, float]
    ) -> None:
        """Test temporal feature extraction."""
        signal, _ = square_wave
        features = extractor.extract_temporal(signal)

        # Check expected features
        expected = {
            "zero_crossing_rate",
            "autocorrelation",
            "peak_count",
            "peak_prominence",
            "energy",
            "rms",
            "snr_estimate",
            "crest_factor",
        }
        assert expected.issubset(features.keys())

        # Square wave has many zero crossings
        assert features["zero_crossing_rate"] > 0.01

        # Energy should be positive
        assert features["energy"] > 0

    def test_entropy_features(
        self, extractor: FeatureExtractor, noise: tuple[np.ndarray, float]
    ) -> None:
        """Test entropy feature extraction."""
        signal, _ = noise
        features = extractor.extract_entropy(signal)

        # Check expected features
        expected = {"shannon_entropy", "approximate_entropy", "sample_entropy"}
        assert expected.issubset(features.keys())

        # Random noise should have high Shannon entropy
        assert features["shannon_entropy"] > 5.0

        # Entropy values should be non-negative
        for name in expected:
            assert features[name] >= 0

    def test_shape_features(
        self, extractor: FeatureExtractor, square_wave: tuple[np.ndarray, float]
    ) -> None:
        """Test shape feature extraction."""
        signal, sample_rate = square_wave
        features = extractor.extract_shape(signal, sample_rate)

        # Check expected features
        expected = {"duty_cycle", "rise_time", "fall_time", "pulse_width", "form_factor"}
        assert expected.issubset(features.keys())

        # Square wave should have ~50% duty cycle
        assert 0.4 < features["duty_cycle"] < 0.6

        # All features should be non-negative
        for name in expected:
            assert features[name] >= 0

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    def test_constant_signal(self, extractor: FeatureExtractor) -> None:
        """Test feature extraction on constant signal."""
        signal = np.ones(1000)
        sample_rate = 10000.0

        # Should not raise errors (warnings are expected for constant signals)
        features = extractor.extract_all(signal, sample_rate)

        # Mean should be 1.0
        assert abs(features["mean"] - 1.0) < 0.01

        # Std should be near 0
        assert features["std"] < 0.01

        # Zero crossings should be 0
        assert features["zero_crossing_rate"] == 0.0

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
    def test_empty_signal(self, extractor: FeatureExtractor) -> None:
        """Test feature extraction on empty signal."""
        signal = np.array([])
        sample_rate = 10000.0

        # Should raise error or return gracefully
        with pytest.raises((ValueError, IndexError)):
            extractor.extract_all(signal, sample_rate)

    def test_single_sample(self, extractor: FeatureExtractor) -> None:
        """Test feature extraction on single sample."""
        signal = np.array([1.0])
        sample_rate = 10000.0

        # Should handle gracefully
        features = extractor.extract_all(signal, sample_rate)

        # Basic features should work
        assert features["mean"] == 1.0
        assert features["std"] == 0.0

    def test_different_signal_types(self, extractor: FeatureExtractor) -> None:
        """Test feature extraction distinguishes different signal types."""
        sample_rate = 10000.0
        t = np.linspace(0, 1, int(sample_rate))

        # Sine wave
        sine = np.sin(2 * np.pi * 1000 * t)
        sine_features = extractor.extract_all(sine, sample_rate)

        # Square wave
        square = np.sign(np.sin(2 * np.pi * 1000 * t))
        square_features = extractor.extract_all(square, sample_rate)

        # PWM (25% duty cycle)
        pwm = np.where((t % 0.001) < 0.00025, 1.0, 0.0)
        pwm_features = extractor.extract_all(pwm, sample_rate)

        # Duty cycle features
        # Sine and square both have ~50% duty cycle (time above midpoint)
        assert abs(sine_features["duty_cycle"] - 0.5) < 0.01
        assert abs(square_features["duty_cycle"] - 0.5) < 0.01
        # PWM has 25% duty cycle, should be notably different
        assert abs(pwm_features["duty_cycle"] - 0.25) < 0.01
        assert abs(square_features["duty_cycle"] - pwm_features["duty_cycle"]) > 0.2

        # Zero crossing rates should be reasonable (sine and square both cross at 1000 Hz)
        assert 0.15 < sine_features["zero_crossing_rate"] < 0.25
        assert 0.15 < square_features["zero_crossing_rate"] < 0.25

        # Spectral flatness should differ (sine is tonal, noise is flat)
        assert sine_features["spectral_flatness"] < 0.5  # Pure tone

    def test_feature_consistency(self, extractor: FeatureExtractor) -> None:
        """Test that feature extraction is deterministic."""
        signal = np.random.RandomState(42).randn(1000)
        sample_rate = 10000.0

        # Extract features twice
        features1 = extractor.extract_all(signal, sample_rate)
        features2 = extractor.extract_all(signal, sample_rate)

        # Should be identical
        for name in features1:
            assert features1[name] == features2[name]

    def test_feature_names(self, extractor: FeatureExtractor) -> None:
        """Test that feature names are consistent and well-formed."""
        signal = np.random.randn(1000)
        sample_rate = 10000.0

        features = extractor.extract_all(signal, sample_rate)

        # All feature names should be valid identifiers
        for name in features:
            assert name.isidentifier(), f"Invalid feature name: {name}"
            assert "_" in name or name.islower(), f"Feature name not snake_case: {name}"

    @pytest.mark.parametrize(
        "freq,expected_dominant",
        [
            (100, 100),
            (500, 500),
            (1000, 1000),
            (2000, 2000),
        ],
    )
    def test_frequency_detection(
        self, extractor: FeatureExtractor, freq: float, expected_dominant: float
    ) -> None:
        """Test that dominant frequency is correctly detected."""
        sample_rate = 10000.0
        t = np.linspace(0, 1, int(sample_rate))
        signal = np.sin(2 * np.pi * freq * t)

        features = extractor.extract_spectral(signal, sample_rate)

        # Allow 10% tolerance
        assert abs(features["dominant_frequency"] - expected_dominant) < 0.1 * expected_dominant
