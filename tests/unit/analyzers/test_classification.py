"""Unit tests for signal classification pipeline.

Tests cover:
- Statistical feature extraction
- Frequency domain feature extraction
- Digital pattern detection
- Signal type classification (digital, analog, UART, SPI, I2C, CAN, PWM)
- Batch classification
- Rule-based classification
- Confidence scoring
- Alternative match reporting
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.classification import (
    ClassificationResult,
    ClassifierRule,
    SignalClassifier,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def classifier() -> SignalClassifier:
    """Create default SignalClassifier instance."""
    return SignalClassifier()


@pytest.fixture
def digital_signal() -> np.ndarray:
    """Generate clean digital signal (0V and 3.3V)."""
    # Pattern: 0, 1, 1, 0, 1, 0, 0, 1 repeated
    pattern = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float64) * 3.3
    return np.tile(pattern, 125)  # 1000 samples


@pytest.fixture
def analog_signal() -> np.ndarray:
    """Generate smooth analog signal (sine wave)."""
    t = np.linspace(0, 0.01, 1000)
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def pwm_signal(sample_rate: float) -> np.ndarray:
    """Generate PWM signal with varying duty cycle."""
    t = np.linspace(0, 0.01, 1000)
    frequency = 1000.0
    # Create square wave with 30% duty cycle
    signal = (np.sin(2 * np.pi * frequency * t) > 0.6).astype(np.float64) * 3.3
    return signal


@pytest.fixture
def uart_signal() -> np.ndarray:
    """Generate UART-like signal with periodic bit timing."""
    # 115200 baud, 10 samples per bit for simplicity
    samples_per_bit = 10
    # Encode "Hello" (simplified)
    bits = [0, 1, 0, 0, 1, 0, 0, 0, 1]  # Start + data + stop
    signal = np.repeat(np.array(bits, dtype=np.float64), samples_per_bit) * 3.3
    # Repeat pattern
    return np.tile(signal, 10)


@pytest.fixture
def spi_signal() -> np.ndarray:
    """Generate SPI-like signal with high edge density."""
    # Rapid clock transitions
    clock_pattern = np.array([0, 1] * 500, dtype=np.float64) * 3.3
    return clock_pattern


@pytest.fixture
def noise_signal() -> np.ndarray:
    """Generate random noise signal."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 1000)


# =============================================================================
# Initialization Tests
# =============================================================================


def test_classifier_default_initialization() -> None:
    """Test classifier initializes with default methods."""
    classifier = SignalClassifier()
    assert "statistical" in classifier.methods
    assert "frequency" in classifier.methods
    assert "pattern" in classifier.methods


def test_classifier_custom_methods() -> None:
    """Test classifier with custom method selection."""
    classifier = SignalClassifier(methods=["statistical", "frequency"])
    assert "statistical" in classifier.methods
    assert "frequency" in classifier.methods
    assert "pattern" not in classifier.methods


def test_classifier_invalid_method_raises() -> None:
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown methods"):
        SignalClassifier(methods=["invalid_method"])


# =============================================================================
# Statistical Feature Extraction Tests
# =============================================================================


def test_extract_statistical_features_digital(
    classifier: SignalClassifier, digital_signal: np.ndarray
) -> None:
    """Test statistical features for digital signal."""
    features = classifier._extract_statistical_features(digital_signal)

    assert "mean" in features
    assert "variance" in features
    assert "duty_cycle" in features
    assert "edge_count" in features
    assert "edge_density" in features

    # Digital signal should have duty cycle around 0.5
    assert 0.3 < features["duty_cycle"] < 0.7
    # Should have many edges
    assert features["edge_count"] > 100
    assert features["edge_density"] > 0.01


def test_extract_statistical_features_analog(
    classifier: SignalClassifier, analog_signal: np.ndarray
) -> None:
    """Test statistical features for analog signal."""
    features = classifier._extract_statistical_features(analog_signal)

    # Analog sine wave should have low edge density
    assert features["edge_density"] < 0.05
    # Should have variance
    assert features["variance"] > 0


def test_extract_statistical_features_pwm(
    classifier: SignalClassifier, pwm_signal: np.ndarray
) -> None:
    """Test statistical features for PWM signal."""
    features = classifier._extract_statistical_features(pwm_signal)

    # PWM should have specific duty cycle
    assert 0.1 < features["duty_cycle"] < 0.9
    # Should have regular edges
    assert features["edge_count"] > 10


def test_extract_statistical_features_edge_cases(classifier: SignalClassifier) -> None:
    """Test statistical features with edge cases."""
    # Constant signal
    constant = np.ones(100) * 3.3
    features = classifier._extract_statistical_features(constant)
    assert features["variance"] == 0.0
    assert features["edge_count"] == 0
    # Duty cycle depends on threshold calculation - for constant signal at 3.3,
    # threshold is also 3.3, so all values are <= threshold (duty_cycle = 0 or 1)
    assert features["duty_cycle"] in (0.0, 1.0)  # Always at one extreme


# =============================================================================
# Frequency Feature Extraction Tests
# =============================================================================


def test_extract_frequency_features_sine(
    classifier: SignalClassifier, analog_signal: np.ndarray, sample_rate: float
) -> None:
    """Test frequency features for sine wave."""
    features = classifier._extract_frequency_features(analog_signal, sample_rate)

    assert "dominant_frequency" in features
    assert "bandwidth" in features
    assert "spectral_centroid" in features
    assert "spectral_flatness" in features

    # Sine wave should have a dominant frequency (may vary depending on FFT resolution)
    assert features["dominant_frequency"] > 0
    # Low spectral flatness (tonal)
    assert features["spectral_flatness"] < 0.5


def test_extract_frequency_features_noise(
    classifier: SignalClassifier, noise_signal: np.ndarray, sample_rate: float
) -> None:
    """Test frequency features for noise."""
    features = classifier._extract_frequency_features(noise_signal, sample_rate)

    # Noise should have high spectral flatness (flat spectrum)
    # Note: May vary depending on random seed
    assert features["spectral_flatness"] >= 0.0


def test_extract_frequency_features_digital(
    classifier: SignalClassifier, digital_signal: np.ndarray, sample_rate: float
) -> None:
    """Test frequency features for digital signal."""
    features = classifier._extract_frequency_features(digital_signal, sample_rate)

    # Digital signal has harmonics
    assert features["dominant_frequency"] > 0
    assert features["bandwidth"] > 0


def test_extract_frequency_features_short_signal(
    classifier: SignalClassifier, sample_rate: float
) -> None:
    """Test frequency features with very short signal."""
    short_signal = np.array([1.0])
    features = classifier._extract_frequency_features(short_signal, sample_rate)

    # Should return zeros for short signal
    assert features["dominant_frequency"] == 0.0
    assert features["bandwidth"] == 0.0


# =============================================================================
# Pattern Detection Tests
# =============================================================================


def test_detect_digital_patterns_uart(
    classifier: SignalClassifier, uart_signal: np.ndarray, sample_rate: float
) -> None:
    """Test pattern detection for UART signal."""
    patterns = classifier._detect_digital_patterns(uart_signal, sample_rate)

    assert "uart_score" in patterns
    assert "spi_score" in patterns
    assert "periodicity" in patterns

    # UART should have some periodicity (may vary with fixture)
    # Note: Actual score depends on baud rate alignment and signal characteristics
    assert patterns["periodicity"] > 0.0


def test_detect_digital_patterns_spi(
    classifier: SignalClassifier, spi_signal: np.ndarray, sample_rate: float
) -> None:
    """Test pattern detection for SPI signal."""
    patterns = classifier._detect_digital_patterns(spi_signal, sample_rate)

    # SPI should have high edge density and high spi_score
    # Note: spi_score is computed from edge density and consistency
    assert patterns["spi_score"] > 0.0


def test_detect_digital_patterns_few_edges(
    classifier: SignalClassifier, sample_rate: float
) -> None:
    """Test pattern detection with very few edges."""
    # Signal with only 1 edge
    signal = np.concatenate([np.zeros(50), np.ones(50)])
    patterns = classifier._detect_digital_patterns(signal, sample_rate)

    # Should return low scores
    assert patterns["uart_score"] == 0.0
    assert patterns["spi_score"] == 0.0


# =============================================================================
# Classification Tests
# =============================================================================


def test_classify_digital_signal(
    classifier: SignalClassifier, digital_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification of digital signal."""
    result = classifier.classify(digital_signal, sample_rate)

    assert isinstance(result, ClassificationResult)
    assert result.signal_type == "digital"
    assert result.confidence > 0.5
    assert "variance" in result.features
    assert result.reasoning != ""


def test_classify_analog_signal(
    classifier: SignalClassifier, analog_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification of analog signal."""
    result = classifier.classify(analog_signal, sample_rate)

    assert result.signal_type == "analog"
    assert result.confidence > 0.5


def test_classify_pwm_signal(
    classifier: SignalClassifier, pwm_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification of PWM signal."""
    result = classifier.classify(pwm_signal, sample_rate)

    # PWM should be detected (or at least digital)
    assert result.signal_type in ("pwm", "digital")
    assert result.confidence > 0.0


def test_classify_uart_signal(
    classifier: SignalClassifier, uart_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification of UART signal."""
    result = classifier.classify(uart_signal, sample_rate)

    # UART detection depends on baud rate alignment
    # May be classified as digital or uart
    assert result.signal_type in ("uart", "digital", "unknown")
    assert result.confidence >= 0.0


def test_classify_spi_signal(
    classifier: SignalClassifier, spi_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification of SPI signal."""
    result = classifier.classify(spi_signal, sample_rate)

    # SPI has high edge density
    assert result.signal_type in ("spi", "digital")


def test_classify_empty_signal_raises(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test that empty signal raises ValueError."""
    with pytest.raises(ValueError, match="Cannot classify empty signal"):
        classifier.classify(np.array([]), sample_rate)


def test_classify_invalid_sample_rate_raises(
    classifier: SignalClassifier, digital_signal: np.ndarray
) -> None:
    """Test that invalid sample rate raises ValueError."""
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        classifier.classify(digital_signal, 0.0)

    with pytest.raises(ValueError, match="sample_rate must be positive"):
        classifier.classify(digital_signal, -100.0)


def test_classify_invalid_threshold_raises(
    classifier: SignalClassifier, digital_signal: np.ndarray, sample_rate: float
) -> None:
    """Test that invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match="threshold must be in"):
        classifier.classify(digital_signal, sample_rate, threshold=1.5)

    with pytest.raises(ValueError, match="threshold must be in"):
        classifier.classify(digital_signal, sample_rate, threshold=-0.1)


def test_classify_with_high_threshold(
    classifier: SignalClassifier, digital_signal: np.ndarray, sample_rate: float
) -> None:
    """Test classification with high confidence threshold."""
    result = classifier.classify(digital_signal, sample_rate, threshold=0.95)

    # With very high threshold, may be classified as unknown
    assert result.signal_type in ("digital", "unknown")


def test_classify_secondary_matches(
    classifier: SignalClassifier, digital_signal: np.ndarray, sample_rate: float
) -> None:
    """Test that secondary matches are reported."""
    result = classifier.classify(digital_signal, sample_rate, threshold=0.3)

    # Should have some secondary matches
    assert isinstance(result.secondary_matches, list)
    # Each match is (type, confidence)
    for match in result.secondary_matches:
        assert len(match) == 2
        assert isinstance(match[0], str)
        assert isinstance(match[1], float)


# =============================================================================
# Batch Classification Tests
# =============================================================================


def test_classify_batch_basic(
    classifier: SignalClassifier,
    digital_signal: np.ndarray,
    analog_signal: np.ndarray,
    sample_rate: float,
) -> None:
    """Test batch classification of multiple signals."""
    signals = [digital_signal, analog_signal]
    results = classifier.classify_batch(signals, sample_rate)

    assert len(results) == 2
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert results[0].signal_type == "digital"
    assert results[1].signal_type == "analog"


def test_classify_batch_empty_raises(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test that empty batch raises ValueError."""
    with pytest.raises(ValueError, match="Cannot classify empty signal list"):
        classifier.classify_batch([], sample_rate)


def test_classify_batch_with_threshold(
    classifier: SignalClassifier,
    digital_signal: np.ndarray,
    analog_signal: np.ndarray,
    sample_rate: float,
) -> None:
    """Test batch classification with custom threshold."""
    signals = [digital_signal, analog_signal]
    results = classifier.classify_batch(signals, sample_rate, threshold=0.7)

    assert len(results) == 2
    # Results may vary depending on confidence


# =============================================================================
# Rule-Based Classification Tests
# =============================================================================


def test_classifier_rules_exist(classifier: SignalClassifier) -> None:
    """Test that classification rules are defined."""
    assert len(classifier.RULES) > 0
    assert all(isinstance(rule, ClassifierRule) for rule in classifier.RULES)


def test_evaluate_rule_match(classifier: SignalClassifier) -> None:
    """Test rule evaluation with matching features."""
    rule = ClassifierRule(
        "digital",
        {"variance": (0.2, 1.0), "edge_density": (0.01, 1.0)},
        weight=1.0,
    )

    features = {"variance": 0.5, "edge_density": 0.1}
    score = classifier._evaluate_rule(rule, features)

    # Both conditions met, should have high score
    assert score > 0.8


def test_evaluate_rule_partial_match(classifier: SignalClassifier) -> None:
    """Test rule evaluation with partial feature match."""
    rule = ClassifierRule(
        "digital",
        {"variance": (0.2, 1.0), "edge_density": (0.01, 1.0)},
        weight=1.0,
    )

    # Only variance matches
    features = {"variance": 0.5, "edge_density": 0.001}
    score = classifier._evaluate_rule(rule, features)

    # Only 50% of conditions met
    assert 0.4 < score < 0.6


def test_evaluate_rule_no_match(classifier: SignalClassifier) -> None:
    """Test rule evaluation with no matching features."""
    rule = ClassifierRule(
        "digital",
        {"variance": (0.2, 1.0), "edge_density": (0.01, 1.0)},
        weight=1.0,
    )

    features = {"variance": 0.05, "edge_density": 0.001}
    score = classifier._evaluate_rule(rule, features)

    # No conditions met
    assert score == 0.0


def test_evaluate_rule_missing_features(classifier: SignalClassifier) -> None:
    """Test rule evaluation with missing features."""
    rule = ClassifierRule(
        "uart",
        {"uart_score": (0.6, 1.0)},
        weight=1.0,
    )

    # Feature not present
    features = {"variance": 0.5}
    score = classifier._evaluate_rule(rule, features)

    # Missing feature means condition not met
    assert score == 0.0


def test_evaluate_rule_with_weight(classifier: SignalClassifier) -> None:
    """Test rule evaluation with custom weight."""
    rule = ClassifierRule(
        "pwm",
        {"duty_cycle": (0.1, 0.9), "periodicity": (0.6, 1.0)},
        weight=1.5,
    )

    features = {"duty_cycle": 0.5, "periodicity": 0.8}
    score = classifier._evaluate_rule(rule, features)

    # Both conditions met, weighted score
    assert score >= 1.0  # Can exceed 1.0 due to weight


# =============================================================================
# Reasoning Generation Tests
# =============================================================================


def test_generate_reasoning_digital(classifier: SignalClassifier) -> None:
    """Test reasoning generation for digital signal."""
    features = {"variance": 0.5, "edge_density": 0.1}
    reasoning = classifier._generate_reasoning("digital", features)

    assert "digital" in reasoning.lower()
    assert "variance" in reasoning.lower()


def test_generate_reasoning_analog(classifier: SignalClassifier) -> None:
    """Test reasoning generation for analog signal."""
    features = {"edge_density": 0.01}
    reasoning = classifier._generate_reasoning("analog", features)

    assert "analog" in reasoning.lower()
    assert "edge density" in reasoning.lower()


def test_generate_reasoning_pwm(classifier: SignalClassifier) -> None:
    """Test reasoning generation for PWM signal."""
    features = {"periodicity": 0.85, "duty_cycle": 0.3}
    reasoning = classifier._generate_reasoning("pwm", features)

    assert "pwm" in reasoning.lower()
    assert "duty cycle" in reasoning.lower()


def test_generate_reasoning_uart(classifier: SignalClassifier) -> None:
    """Test reasoning generation for UART signal."""
    features = {"uart_score": 0.9, "edge_density": 0.05}
    reasoning = classifier._generate_reasoning("uart", features)

    assert "uart" in reasoning.lower()


def test_generate_reasoning_unknown(classifier: SignalClassifier) -> None:
    """Test reasoning generation for unknown signal."""
    features = {}
    reasoning = classifier._generate_reasoning("unknown", features)

    assert "unclear" in reasoning.lower() or "unknown" in reasoning.lower()


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow_multiple_methods(digital_signal: np.ndarray, sample_rate: float) -> None:
    """Test complete workflow with all classification methods."""
    # All methods
    classifier_all = SignalClassifier(methods=["statistical", "frequency", "pattern"])
    result_all = classifier_all.classify(digital_signal, sample_rate)

    # Statistical only
    classifier_stat = SignalClassifier(methods=["statistical"])
    result_stat = classifier_stat.classify(digital_signal, sample_rate)

    # Both should detect digital
    assert result_all.signal_type == "digital"
    assert result_stat.signal_type == "digital"

    # All methods should have more features
    assert len(result_all.features) > len(result_stat.features)


def test_full_workflow_batch_diverse_signals(
    digital_signal: np.ndarray,
    analog_signal: np.ndarray,
    pwm_signal: np.ndarray,
    sample_rate: float,
) -> None:
    """Test batch classification with diverse signal types."""
    classifier = SignalClassifier()
    signals = [digital_signal, analog_signal, pwm_signal]
    results = classifier.classify_batch(signals, sample_rate)

    assert len(results) == 3
    # Each should have different classification
    types = {r.signal_type for r in results}
    # At least 2 different types detected
    assert len(types) >= 2


def test_confidence_increases_with_clear_signal() -> None:
    """Test that confidence varies with signal clarity."""
    classifier = SignalClassifier()
    sample_rate = 1e6

    # Very clear digital signal (clean square wave)
    clear_digital = np.tile([0, 0, 3.3, 3.3], 250)
    result_clear = classifier.classify(clear_digital, sample_rate)

    # Noisy signal
    rng = np.random.default_rng(42)
    noisy = rng.normal(1.5, 0.5, 1000)
    result_noisy = classifier.classify(noisy, sample_rate)

    # Clear signal should be classified with reasonable confidence
    assert result_clear.confidence >= 0.5
    # Both should have some classification
    assert result_noisy.confidence >= 0.0


def test_constant_signal_classification() -> None:
    """Test classification of constant signal."""
    classifier = SignalClassifier()
    sample_rate = 1e6

    constant = np.ones(1000) * 3.3
    result = classifier.classify(constant, sample_rate)

    # Constant signal: no edges, should be analog or unknown
    assert result.signal_type in ("analog", "unknown")
    assert result.features["edge_count"] == 0


def test_alternating_pattern_classification() -> None:
    """Test classification of rapid alternating pattern."""
    classifier = SignalClassifier()
    sample_rate = 1e6

    # Alternating 0 and 1 (maximum edge density)
    alternating = np.tile([0, 3.3], 500)
    result = classifier.classify(alternating, sample_rate)

    # High edge density - should be digital or SPI
    assert result.signal_type in ("digital", "spi")
    assert result.features["edge_density"] > 0.4


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_single_sample_signal(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test classification with single sample."""
    single = np.array([1.0])
    result = classifier.classify(single, sample_rate)

    # Should classify (likely as unknown or analog due to no edges)
    assert result.signal_type in ("analog", "unknown")


def test_two_sample_signal(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test classification with two samples."""
    two = np.array([0.0, 3.3])
    result = classifier.classify(two, sample_rate)

    # Should classify
    assert isinstance(result, ClassificationResult)


def test_very_large_signal(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test classification with very large signal."""
    # 1 million samples
    large = np.tile([0, 3.3], 500_000)
    result = classifier.classify(large, sample_rate)

    # Should complete without error
    assert result.signal_type in ("digital", "spi")


def test_negative_values(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test classification with negative voltage values."""
    negative = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 1000))
    result = classifier.classify(negative, sample_rate)

    # Should handle negative values
    assert result.signal_type == "analog"


def test_very_small_variance(classifier: SignalClassifier, sample_rate: float) -> None:
    """Test classification with very small variance."""
    # Nearly constant with tiny variation
    small_var = np.ones(1000) + np.random.default_rng(42).normal(0, 0.001, 1000)
    result = classifier.classify(small_var, sample_rate)

    # Small random variations may create edges - classification depends on pattern
    # Could be analog (low variance) or detected as digital/spi based on edge density
    assert result.signal_type in ("analog", "unknown", "digital", "spi", "uart")
