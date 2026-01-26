"""Intelligent signal classification pipeline for automatic signal type identification.

This module provides multi-method signal classification to automatically identify
signal types from waveforms without manual configuration.

Key capabilities:
- Classify signals: digital, analog, PWM, UART, SPI, I2C, CAN
- Multi-method classification: statistical, frequency domain, pattern recognition
- Confidence scoring for each classification
- Batch classification with parallel processing support
- Extensible architecture for adding new classifiers

Classification methods:
- Statistical features: mean, variance, duty cycle, edge density
- Frequency domain: FFT analysis, dominant frequencies, spectral characteristics
- Time domain patterns: Protocol-specific signature detection
- Rule-based classification: Feature threshold matching

Typical workflow:
1. Extract features from signal (statistical, frequency, pattern)
2. Apply classification rules based on feature values
3. Return best match with confidence score and alternatives

Example:
    >>> from oscura.analyzers.classification import SignalClassifier
    >>> classifier = SignalClassifier()
    >>> result = classifier.classify(signal, sample_rate=1e6)
    >>> print(f"{result.signal_type}: {result.confidence:.2f}")
    uart: 0.94
    >>> print(f"Features: {result.features}")
    Features: {'duty_cycle': 0.52, 'edge_density': 0.042, ...}

References:
    IEEE 181-2011: Transitional Waveform Definitions
    DISC-001: Automatic Signal Characterization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SignalType = Literal["digital", "analog", "pwm", "uart", "spi", "i2c", "can", "unknown"]


@dataclass
class ClassificationResult:
    """Result of signal classification.

    Contains detected signal type, confidence score, features used for classification,
    and alternative matches for ambiguous cases.

    Attributes:
        signal_type: Detected signal type (digital, analog, uart, spi, etc.)
        confidence: Confidence score (0.0-1.0), higher is more confident
        features: Dictionary of extracted features used for classification
        secondary_matches: Alternative classifications with confidence scores
        reasoning: Human-readable explanation of classification decision

    Example:
        >>> result = classifier.classify(signal, sample_rate=1e6)
        >>> if result.confidence >= 0.8:
        ...     print(f"High confidence: {result.signal_type}")
        >>> for alt_type, alt_conf in result.secondary_matches:
        ...     print(f"Alternative: {alt_type} ({alt_conf:.2f})")
    """

    signal_type: SignalType
    confidence: float
    features: dict[str, float]
    secondary_matches: list[tuple[SignalType, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ClassifierRule:
    """Rule-based classification criteria.

    Defines feature thresholds for identifying a specific signal type.

    Attributes:
        signal_type: Signal type this rule identifies
        conditions: Dict mapping feature name to (min, max) thresholds
        weight: Importance weight for this rule (default 1.0)
        required_features: Features that must be present and within range

    Example:
        >>> rule = ClassifierRule(
        ...     signal_type="digital",
        ...     conditions={"variance": (0.2, 1.0), "edge_count": (100, float('inf'))},
        ...     weight=1.0
        ... )
    """

    signal_type: SignalType
    conditions: dict[str, tuple[float, float]]
    weight: float = 1.0


class SignalClassifier:
    """Multi-method signal classifier with extensible architecture.

    Combines statistical analysis, frequency domain features, and pattern
    recognition to automatically identify signal types from waveforms.

    Classification Methods:
        - statistical: Mean, variance, duty cycle, edge statistics
        - frequency: FFT-based frequency analysis, spectral characteristics
        - pattern: Protocol-specific pattern detection (UART, SPI, etc.)

    Attributes:
        methods: List of classification methods to use
        threshold: Minimum confidence to report classification (default 0.5)

    Example:
        >>> # Default classifier uses all methods
        >>> classifier = SignalClassifier()
        >>> result = classifier.classify(signal, sample_rate=1e6)
        >>>
        >>> # Custom classifier with specific methods
        >>> classifier = SignalClassifier(methods=["statistical", "frequency"])
        >>> result = classifier.classify(signal, sample_rate=1e6, threshold=0.7)
    """

    # Classification rules for different signal types
    RULES: ClassVar[list[ClassifierRule]] = [
        # Digital signal: bimodal distribution, many edges
        ClassifierRule(
            "digital",
            {"variance": (0.15, 1.0), "edge_density": (0.01, 1.0)},
            weight=1.0,
        ),
        # Analog signal: continuous values, low edge count
        ClassifierRule(
            "analog",
            {"variance": (0.01, 0.5), "edge_density": (0.0, 0.02)},
            weight=1.0,
        ),
        # PWM signal: regular duty cycle, periodic
        ClassifierRule(
            "pwm",
            {"duty_cycle": (0.1, 0.9), "periodicity": (0.6, 1.0), "edge_density": (0.01, 0.5)},
            weight=1.2,
        ),
        # UART: specific bit timing patterns, moderate edge density
        ClassifierRule(
            "uart",
            {"uart_score": (0.6, 1.0), "edge_density": (0.01, 0.2)},
            weight=1.3,
        ),
        # SPI: clock + data patterns, high edge density
        ClassifierRule(
            "spi",
            {"spi_score": (0.6, 1.0), "edge_density": (0.1, 1.0)},
            weight=1.2,
        ),
        # I2C: ACK patterns, clock stretching
        ClassifierRule(
            "i2c",
            {"i2c_score": (0.5, 1.0)},
            weight=1.1,
        ),
        # CAN: specific encoding
        ClassifierRule(
            "can",
            {"can_score": (0.6, 1.0)},
            weight=1.1,
        ),
    ]

    def __init__(self, methods: list[str] | None = None) -> None:
        """Initialize signal classifier.

        Args:
            methods: Classification methods to use. Default uses all methods.
                Available: "statistical", "frequency", "pattern"

        Raises:
            ValueError: If unknown method is specified
        """
        available_methods = {"statistical", "frequency", "pattern"}
        if methods is None:
            self.methods = ["statistical", "frequency", "pattern"]
        else:
            # Validate methods
            unknown = set(methods) - available_methods
            if unknown:
                raise ValueError(f"Unknown methods: {unknown}. Available: {available_methods}")
            self.methods = methods

    def classify(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
        threshold: float = 0.5,
    ) -> ClassificationResult:
        """Classify single signal.

        Extracts features using configured methods and applies classification
        rules to determine signal type.

        Args:
            signal: Signal data array (voltage samples)
            sample_rate: Sample rate in Hz
            threshold: Minimum confidence for primary classification (0.0-1.0)

        Returns:
            ClassificationResult with signal type and confidence

        Raises:
            ValueError: If signal is empty or sample_rate is invalid

        Example:
            >>> result = classifier.classify(signal, sample_rate=1e6)
            >>> print(f"Type: {result.signal_type}, Confidence: {result.confidence:.2f}")
            Type: uart, Confidence: 0.94
        """
        if len(signal) == 0:
            raise ValueError("Cannot classify empty signal")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        # Extract features using configured methods
        features: dict[str, float] = {}

        if "statistical" in self.methods:
            features.update(self._extract_statistical_features(signal))

        if "frequency" in self.methods:
            features.update(self._extract_frequency_features(signal, sample_rate))

        if "pattern" in self.methods:
            features.update(self._detect_digital_patterns(signal, sample_rate))

        # Classify from features
        signal_type, confidence, alternatives = self._classify_from_features(features, threshold)

        # Generate reasoning
        reasoning = self._generate_reasoning(signal_type, features)

        return ClassificationResult(
            signal_type=signal_type,
            confidence=confidence,
            features=features,
            secondary_matches=alternatives,
            reasoning=reasoning,
        )

    def classify_batch(
        self,
        signals: list[NDArray[np.floating[Any]]],
        sample_rate: float,
        threshold: float = 0.5,
    ) -> list[ClassificationResult]:
        """Classify multiple signals.

        Classifies signals sequentially. For large batches, consider using
        multiprocessing for parallel processing.

        Args:
            signals: List of signal arrays to classify
            sample_rate: Sample rate in Hz (same for all signals)
            threshold: Minimum confidence for primary classification

        Returns:
            List of ClassificationResult objects

        Raises:
            ValueError: If signals list is empty

        Example:
            >>> results = classifier.classify_batch(signals, sample_rate=1e6)
            >>> for i, result in enumerate(results):
            ...     print(f"Signal {i}: {result.signal_type}")
        """
        if not signals:
            raise ValueError("Cannot classify empty signal list")

        return [self.classify(signal, sample_rate, threshold) for signal in signals]

    def _extract_statistical_features(
        self,
        signal: NDArray[np.floating[Any]],
    ) -> dict[str, float]:
        """Extract statistical features from signal.

        Features:
            - mean: Average voltage level
            - variance: Signal variance (normalized)
            - min/max: Voltage range
            - duty_cycle: Fraction of time signal is high (for digital)
            - edge_count: Number of transitions
            - edge_density: Edges per sample

        Args:
            signal: Signal data array

        Returns:
            Dictionary of statistical features

        Example:
            >>> features = classifier._extract_statistical_features(signal)
            >>> print(features['duty_cycle'])
            0.52
        """
        mean = float(np.mean(signal))
        variance = float(np.var(signal))
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        voltage_swing = max_val - min_val

        # Normalize variance by voltage swing to make it scale-independent
        normalized_variance = variance / (voltage_swing**2 + 1e-10)

        # Digital features: threshold at midpoint
        threshold = (max_val + min_val) / 2
        digital = (signal > threshold).astype(int)
        edges = np.diff(digital)
        edge_count = int(np.count_nonzero(edges))
        edge_density = edge_count / len(signal) if len(signal) > 0 else 0.0

        # Duty cycle (fraction of time high)
        duty_cycle = float(np.mean(digital))

        return {
            "mean": mean,
            "variance": normalized_variance,
            "min": min_val,
            "max": max_val,
            "voltage_swing": voltage_swing,
            "duty_cycle": duty_cycle,
            "edge_count": edge_count,
            "edge_density": edge_density,
        }

    def _extract_frequency_features(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> dict[str, float]:
        """Extract frequency domain features via FFT.

        Features:
            - dominant_frequency: Frequency with highest power (Hz)
            - bandwidth: Frequency range with >10% of peak power (Hz)
            - spectral_centroid: Center of mass of spectrum (Hz)
            - spectral_flatness: Ratio of geometric to arithmetic mean (0-1)

        Args:
            signal: Signal data array
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary of frequency domain features

        Example:
            >>> features = classifier._extract_frequency_features(signal, 1e6)
            >>> print(features['dominant_frequency'])
            115200.0
        """
        if len(signal) < 2:
            return {
                "dominant_frequency": 0.0,
                "bandwidth": 0.0,
                "spectral_centroid": 0.0,
                "spectral_flatness": 0.0,
            }

        # Compute FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
        magnitude = np.abs(fft)

        # Dominant frequency (skip DC component)
        if len(magnitude) > 1:
            dominant_idx = np.argmax(magnitude[1:]) + 1
            dominant_frequency = float(freqs[dominant_idx])
        else:
            dominant_frequency = 0.0

        # Bandwidth (frequencies with >10% of max power)
        max_power = np.max(magnitude)
        if max_power > 0:
            threshold_power = 0.1 * max_power
            active_freqs = freqs[magnitude > threshold_power]
            bandwidth = float(active_freqs[-1] - active_freqs[0]) if len(active_freqs) > 1 else 0.0
        else:
            bandwidth = 0.0

        # Spectral centroid (center of mass)
        if np.sum(magnitude) > 0:
            spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        else:
            spectral_centroid = 0.0

        # Spectral flatness (measure of how noise-like the spectrum is)
        # Geometric mean / arithmetic mean
        # 0 = tonal (single frequency), 1 = noise-like (flat spectrum)
        if len(magnitude) > 0 and np.all(magnitude > 0):
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-10))
        else:
            spectral_flatness = 0.0

        return {
            "dominant_frequency": dominant_frequency,
            "bandwidth": bandwidth,
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": spectral_flatness,
        }

    def _detect_digital_patterns(
        self,
        signal: NDArray[np.floating[Any]],
        sample_rate: float,
    ) -> dict[str, float]:
        """Detect protocol-specific patterns in signal.

        Computes scores for:
            - uart_score: UART bit timing alignment
            - spi_score: SPI clock consistency
            - i2c_score: I2C pattern characteristics
            - can_score: CAN encoding characteristics
            - periodicity: Signal periodicity measure

        Args:
            signal: Signal data array
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary of pattern detection scores (0.0-1.0)

        Example:
            >>> patterns = classifier._detect_digital_patterns(signal, 1e6)
            >>> print(patterns['uart_score'])
            0.85
        """
        # Threshold signal to digital
        threshold = (np.max(signal) + np.min(signal)) / 2
        digital = (signal > threshold).astype(int)
        edges = np.diff(digital)
        edge_indices = np.where(np.abs(edges) > 0)[0]

        if len(edge_indices) < 3:
            return {
                "uart_score": 0.0,
                "spi_score": 0.0,
                "i2c_score": 0.0,
                "can_score": 0.0,
                "periodicity": 0.0,
            }

        # Edge intervals
        edge_intervals = np.diff(edge_indices)
        if len(edge_intervals) == 0:
            return {
                "uart_score": 0.0,
                "spi_score": 0.0,
                "i2c_score": 0.0,
                "can_score": 0.0,
                "periodicity": 0.0,
            }

        # Periodicity score (coefficient of variation)
        mean_interval = np.mean(edge_intervals)
        std_interval = np.std(edge_intervals)
        periodicity = 1.0 - min(1.0, std_interval / (mean_interval + 1e-10))

        # UART score: check alignment with common baud rates
        uart_score = self._compute_uart_score(edge_intervals, sample_rate)

        # SPI score: high edge density + consistent timing
        edge_density = len(edge_indices) / len(signal)
        consistency = 1.0 - min(1.0, std_interval / (mean_interval + 1e-10))
        spi_score = min(1.0, edge_density * 10 * consistency)

        # I2C score: lower edge density than SPI, some irregularity (clock stretching)
        # I2C typically has burst patterns with pauses
        i2c_score = 0.5 if 0.05 < edge_density < 0.3 and periodicity < 0.9 else 0.0

        # CAN score: similar to digital but with specific encoding patterns
        # CAN uses bit stuffing - look for irregularity in bit timing
        can_score = 0.5 if 0.7 < periodicity < 0.95 and edge_density > 0.1 else 0.0

        return {
            "uart_score": uart_score,
            "spi_score": spi_score,
            "i2c_score": i2c_score,
            "can_score": can_score,
            "periodicity": periodicity,
        }

    def _compute_uart_score(
        self,
        edge_intervals: NDArray[np.integer[Any]],
        sample_rate: float,
    ) -> float:
        """Compute UART likelihood score based on baud rate alignment.

        Checks if edge intervals align with common UART baud rates.

        Args:
            edge_intervals: Array of sample counts between edges
            sample_rate: Sample rate in Hz

        Returns:
            UART score (0.0-1.0)
        """
        common_bauds = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
        baud_scores = []

        for baud in common_bauds:
            bit_period_samples = sample_rate / baud
            # Count edges that align with this baud rate (within 20% tolerance)
            tolerance = 0.2
            aligned = np.sum(
                np.abs(edge_intervals % bit_period_samples) < bit_period_samples * tolerance
            )
            score = aligned / len(edge_intervals) if len(edge_intervals) > 0 else 0.0
            baud_scores.append(score)

        return float(max(baud_scores)) if baud_scores else 0.0

    def _classify_from_features(
        self,
        features: dict[str, float],
        threshold: float,
    ) -> tuple[SignalType, float, list[tuple[SignalType, float]]]:
        """Make classification decision from extracted features.

        Applies classification rules and selects best match.

        Args:
            features: Dictionary of extracted features
            threshold: Minimum confidence for primary classification

        Returns:
            Tuple of (signal_type, confidence, alternatives)
                alternatives is list of (type, confidence) for secondary matches
        """
        # Evaluate all rules
        scores: dict[SignalType, float] = {}

        for rule in self.RULES:
            score = self._evaluate_rule(rule, features)
            scores[rule.signal_type] = score

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Best match
        best_type, best_score = sorted_scores[0]

        # If score is below threshold, mark as unknown
        if best_score < threshold:
            return "unknown", best_score, [(best_type, best_score)]

        # Collect alternatives (other types with score >= threshold * 0.6)
        alt_threshold = threshold * 0.6
        alternatives = [
            (sig_type, score) for sig_type, score in sorted_scores[1:] if score >= alt_threshold
        ]

        return best_type, best_score, alternatives

    def _evaluate_rule(
        self,
        rule: ClassifierRule,
        features: dict[str, float],
    ) -> float:
        """Evaluate classification rule against features.

        Args:
            rule: Classification rule to evaluate
            features: Extracted features

        Returns:
            Match score (0.0-1.0), weighted by rule weight
        """
        matches = 0
        total = len(rule.conditions)

        for feature_name, (min_val, max_val) in rule.conditions.items():
            if feature_name in features:
                value = features[feature_name]
                if min_val <= value <= max_val:
                    matches += 1

        # Base score is fraction of conditions met
        base_score = matches / total if total > 0 else 0.0

        # Apply rule weight
        weighted_score = base_score * rule.weight

        return min(1.0, weighted_score)

    def _generate_reasoning(
        self,
        signal_type: SignalType,
        features: dict[str, float],
    ) -> str:
        """Generate human-readable explanation of classification.

        Args:
            signal_type: Classified signal type
            features: Features used for classification

        Returns:
            Reasoning string
        """
        if signal_type == "digital":
            return (
                f"Digital signal detected: high variance ({features.get('variance', 0):.2f}), "
                f"edge density {features.get('edge_density', 0):.3f}"
            )
        elif signal_type == "analog":
            return (
                f"Analog signal detected: low edge density ({features.get('edge_density', 0):.3f}), "
                f"continuous values"
            )
        elif signal_type == "pwm":
            return (
                f"PWM signal detected: periodic pattern (periodicity {features.get('periodicity', 0):.2f}), "
                f"duty cycle {features.get('duty_cycle', 0):.2f}"
            )
        elif signal_type == "uart":
            return (
                f"UART signal detected: baud rate alignment score {features.get('uart_score', 0):.2f}, "
                f"edge density {features.get('edge_density', 0):.3f}"
            )
        elif signal_type == "spi":
            return (
                f"SPI signal detected: high edge density ({features.get('edge_density', 0):.3f}), "
                f"consistent timing"
            )
        elif signal_type == "i2c":
            return f"I2C signal detected: characteristic patterns (score {features.get('i2c_score', 0):.2f})"
        elif signal_type == "can":
            return (
                f"CAN signal detected: encoding patterns (score {features.get('can_score', 0):.2f})"
            )
        else:
            return "Signal type unclear: low confidence in all classifications"


__all__ = [
    "ClassificationResult",
    "ClassifierRule",
    "SignalClassifier",
    "SignalType",
]
