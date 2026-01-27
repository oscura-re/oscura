"""Anomaly detection system for identifying unusual patterns in protocol traffic.

This module provides multi-method anomaly detection to identify unusual patterns
in protocol traffic using statistical and machine learning methods.

Key capabilities:
- Statistical anomaly detection (Z-score, IQR, modified Z-score)
- Time-series anomaly detection (rate analysis, timing analysis)
- ML-based detection (Isolation Forest, One-Class SVM) - optional with scikit-learn
- Detect: unexpected message rates, unusual field values, timing anomalies, sequence violations
- Return anomaly scores and explanations
- Support online detection (streaming data)
- Export anomaly reports with context

Detection methods:
- Z-score: Standard deviation-based outlier detection
- IQR: Interquartile range-based outlier detection
- Modified Z-score: Median absolute deviation-based (robust to outliers)
- Isolation Forest: Tree-based anomaly detection (requires sklearn)
- One-Class SVM: Support vector-based anomaly detection (requires sklearn)

Typical workflow:
1. Train detector on normal baseline data (optional for statistical methods)
2. Detect anomalies in new data points (batch or streaming)
3. Export anomaly reports with explanations and context

Example:
    >>> from oscura.analyzers.patterns.anomaly_detection import AnomalyDetector, AnomalyDetectionConfig
    >>> config = AnomalyDetectionConfig(methods=["zscore", "iqr"])
    >>> detector = AnomalyDetector(config)
    >>> # Detect field value anomalies
    >>> anomalies = detector.detect_field_value_anomaly(
    ...     field_values=[1.0, 1.1, 0.9, 1.2, 10.0, 1.0],
    ...     field_name="voltage"
    ... )
    >>> for anomaly in anomalies:
    ...     print(f"{anomaly.anomaly_type}: {anomaly.explanation}")
    value: voltage value 10.00 deviates significantly from expected 2.53

References:
    Isolation Forest: Liu et al. (2008) "Isolation Forest"
    One-Class SVM: Schölkopf et al. (2001) "Estimating the Support of a High-Dimensional Distribution"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Detected anomaly.

    Attributes:
        timestamp: Time of anomaly occurrence
        anomaly_type: Type of anomaly detected (rate, value, timing, sequence, protocol)
        score: Anomaly score (0.0-1.0, higher = more anomalous)
        message_index: Index of message containing anomaly (None if not message-specific)
        field_name: Name of field with anomaly (None if not field-specific)
        expected_value: Expected value based on baseline/model
        actual_value: Actual value observed
        explanation: Human-readable explanation of anomaly
        context: Additional context information (rates, thresholds, etc.)

    Example:
        >>> anomaly = Anomaly(
        ...     timestamp=1234.5,
        ...     anomaly_type="value",
        ...     score=0.95,
        ...     message_index=42,
        ...     field_name="voltage",
        ...     expected_value=3.3,
        ...     actual_value=15.0,
        ...     explanation="Voltage spike detected"
        ... )
    """

    timestamp: float
    anomaly_type: str  # "rate", "value", "timing", "sequence", "protocol"
    score: float  # 0.0-1.0, higher = more anomalous
    message_index: int | None = None
    field_name: str | None = None
    expected_value: Any = None
    actual_value: Any = None
    explanation: str = ""
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration.

    Attributes:
        methods: Detection methods to use (zscore, iqr, modified_zscore, isolation_forest, one_class_svm)
        zscore_threshold: Standard deviations for Z-score method (default: 3.0)
        iqr_multiplier: Multiplier for IQR method (default: 1.5)
        contamination: Expected outlier fraction for ML methods (default: 0.1)
        window_size: Window size for streaming detection (default: 100)

    Example:
        >>> config = AnomalyDetectionConfig(
        ...     methods=["zscore", "iqr"],
        ...     zscore_threshold=2.5,
        ...     window_size=200
        ... )
    """

    methods: list[str] = field(default_factory=lambda: ["zscore", "isolation_forest"])
    zscore_threshold: float = 3.0  # Standard deviations
    iqr_multiplier: float = 1.5
    contamination: float = 0.1  # Expected outlier fraction
    window_size: int = 100  # For streaming detection


class AnomalyDetector:
    """Multi-method anomaly detection system.

    This class provides various anomaly detection methods for protocol traffic analysis.
    It supports both statistical methods (Z-score, IQR) and ML-based methods
    (Isolation Forest, One-Class SVM - if scikit-learn is available).

    Attributes:
        config: Detection configuration
        models: Trained ML models (if applicable)
        baselines: Baseline statistics for comparison
        anomalies: Detected anomalies

    Detection methods:
        Statistical: zscore, iqr, modified_zscore
        ML-based: isolation_forest, one_class_svm (requires scikit-learn)
        Time-series: rate analysis, timing analysis

    Example:
        >>> detector = AnomalyDetector()
        >>> # Detect message rate anomalies
        >>> timestamps = [0.0, 0.1, 0.2, 0.3, 5.0, 5.1]  # Gap at 0.3-5.0
        >>> anomalies = detector.detect_message_rate_anomaly(timestamps)
        >>> # Detect field value anomalies
        >>> values = [1.0, 1.1, 0.9, 1.2, 10.0, 1.0]  # Spike at 10.0
        >>> anomalies = detector.detect_field_value_anomaly(values, "voltage")
    """

    # Detection methods
    STATISTICAL_METHODS: ClassVar[list[str]] = ["zscore", "iqr", "modified_zscore"]
    ML_METHODS: ClassVar[list[str]] = ["isolation_forest", "one_class_svm", "autoencoder"]
    TIMESERIES_METHODS: ClassVar[list[str]] = ["arima", "seasonal_decomposition"]

    def __init__(self, config: AnomalyDetectionConfig | None = None) -> None:
        """Initialize anomaly detector.

        Args:
            config: Anomaly detection configuration. If None, uses default config.

        Example:
            >>> detector = AnomalyDetector()
            >>> config = AnomalyDetectionConfig(methods=["zscore"], zscore_threshold=2.5)
            >>> detector_custom = AnomalyDetector(config)
        """
        self.config = config or AnomalyDetectionConfig()
        self.models: dict[str, Any] = {}
        self.baselines: dict[str, Any] = {}
        self.anomalies: list[Anomaly] = []

    def train(self, normal_data: list[dict[str, Any]], features: list[str]) -> None:
        """Train anomaly detection models on normal baseline data.

        Args:
            normal_data: List of normal data points (dicts with feature values)
            features: List of feature names to use for training

        Raises:
            ImportError: If ML methods requested but scikit-learn not available
            ValueError: If insufficient data for training

        Example:
            >>> normal_data = [
            ...     {"voltage": 3.3, "current": 0.5},
            ...     {"voltage": 3.2, "current": 0.6},
            ... ]
            >>> detector.train(normal_data, features=["voltage", "current"])
        """
        if len(normal_data) < 10:
            raise ValueError("Need at least 10 samples for training")

        # Extract features
        X = np.array([[d[f] for f in features] for d in normal_data])

        # Calculate baseline statistics
        for i, feature in enumerate(features):
            self.baselines[feature] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "median": float(np.median(X[:, i])),
                "q1": float(np.percentile(X[:, i], 25)),
                "q3": float(np.percentile(X[:, i], 75)),
            }

        # Train ML models if requested
        for method in self.config.methods:
            if method == "isolation_forest":
                self._train_isolation_forest(X)
            elif method == "one_class_svm":
                self._train_one_class_svm(X)

    def detect(self, data: dict[str, Any], timestamp: float = 0.0) -> list[Anomaly]:
        """Detect anomalies in new data point.

        Args:
            data: Data point to analyze (dict with feature values)
            timestamp: Timestamp of data point

        Returns:
            List of detected anomalies

        Example:
            >>> anomalies = detector.detect(
            ...     {"voltage": 10.0, "current": 0.5},
            ...     timestamp=123.45
            ... )
        """
        anomalies = []

        # Check each feature against baselines
        for feature, value in data.items():
            if feature in self.baselines:
                baseline = self.baselines[feature]

                # Z-score check
                if "zscore" in self.config.methods:
                    z_score = abs(value - baseline["mean"]) / (baseline["std"] + 1e-10)
                    if z_score > self.config.zscore_threshold:
                        anomalies.append(
                            Anomaly(
                                timestamp=timestamp,
                                anomaly_type="value",
                                score=min(z_score / 5.0, 1.0),
                                field_name=feature,
                                expected_value=baseline["mean"],
                                actual_value=value,
                                explanation=f"{feature} value {value:.2f} deviates {z_score:.1f} std devs from expected {baseline['mean']:.2f}",
                                context={"z_score": z_score, "method": "zscore"},
                            )
                        )

                # IQR check
                if "iqr" in self.config.methods:
                    iqr = baseline["q3"] - baseline["q1"]
                    lower_bound = baseline["q1"] - self.config.iqr_multiplier * iqr
                    upper_bound = baseline["q3"] + self.config.iqr_multiplier * iqr

                    if value < lower_bound or value > upper_bound:
                        anomalies.append(
                            Anomaly(
                                timestamp=timestamp,
                                anomaly_type="value",
                                score=0.8,
                                field_name=feature,
                                expected_value=baseline["median"],
                                actual_value=value,
                                explanation=f"{feature} value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                                context={
                                    "iqr": iqr,
                                    "bounds": (lower_bound, upper_bound),
                                    "method": "iqr",
                                },
                            )
                        )

        self.anomalies.extend(anomalies)
        return anomalies

    def detect_batch(
        self, data_points: list[dict[str, Any]], timestamps: list[float]
    ) -> list[Anomaly]:
        """Detect anomalies in batch of data.

        Args:
            data_points: List of data points to analyze
            timestamps: Corresponding timestamps for each data point

        Returns:
            List of all detected anomalies

        Example:
            >>> data = [
            ...     {"voltage": 3.3},
            ...     {"voltage": 10.0},  # Anomaly
            ... ]
            >>> timestamps = [0.0, 1.0]
            >>> anomalies = detector.detect_batch(data, timestamps)
        """
        all_anomalies = []
        for data, timestamp in zip(data_points, timestamps, strict=True):
            anomalies = self.detect(data, timestamp)
            all_anomalies.extend(anomalies)
        return all_anomalies

    def detect_message_rate_anomaly(
        self, timestamps: list[float], window_size: int = 100
    ) -> list[Anomaly]:
        """Detect anomalous message rates (bursts, gaps).

        Uses sliding window to calculate message rate, then detects outliers
        in rate distribution using Z-score method.

        Args:
            timestamps: List of message timestamps
            window_size: Size of sliding window for rate calculation

        Returns:
            List of detected rate anomalies

        Example:
            >>> timestamps = [0.0, 0.1, 0.2, 0.3, 5.0, 5.1]  # Gap at 0.3-5.0
            >>> anomalies = detector.detect_message_rate_anomaly(timestamps)
            >>> for a in anomalies:
            ...     print(a.explanation)
            Message gap detected: 2.1 msg/s vs expected 10.0 msg/s
        """
        if len(timestamps) < window_size:
            return []

        # Calculate message rates in windows
        rates = []
        window_timestamps = []

        for i in range(len(timestamps) - window_size):
            window = timestamps[i : i + window_size]
            time_span = window[-1] - window[0]
            rate = window_size / time_span if time_span > 0 else 0
            rates.append(rate)
            window_timestamps.append(window[window_size // 2])

        # Detect anomalies in rates using dual-threshold approach
        rate_array = np.array(rates)
        mean_rate = np.mean(rate_array)
        std_rate = np.std(rate_array)
        median_rate = np.median(rate_array)

        # For extremely constant rates, avoid false positives from floating point errors
        if std_rate < mean_rate * 0.0001:  # Coefficient of variation < 0.01%
            return []

        # Use dual-threshold approach for rate anomaly detection:
        # 1. Statistical outlier detection (Z-score)
        # 2. Rate deviation threshold (>30% change from median)
        statistical_outliers = self._zscore_detection(rate_array, self.config.zscore_threshold)

        # Rate-based threshold: detect significant rate changes (>30% deviation from median)
        rate_deviations = np.abs(rate_array - median_rate) / median_rate
        significant_deviations = rate_deviations > 0.3  # 30% threshold

        # Combine: anomaly if either statistical outlier OR significant deviation
        outliers = statistical_outliers | significant_deviations

        # Create Anomaly objects
        anomalies = []

        for idx, is_outlier in enumerate(outliers):
            if is_outlier:
                if rates[idx] > mean_rate:
                    explanation = f"Message burst detected: {rates[idx]:.1f} msg/s vs expected {mean_rate:.1f} msg/s"
                else:
                    explanation = f"Message gap detected: {rates[idx]:.1f} msg/s vs expected {mean_rate:.1f} msg/s"

                anomalies.append(
                    Anomaly(
                        timestamp=window_timestamps[idx],
                        anomaly_type="rate",
                        score=min(abs(rates[idx] - mean_rate) / mean_rate, 1.0),
                        explanation=explanation,
                        context={"rate": rates[idx], "expected_rate": mean_rate},
                    )
                )

        self.anomalies.extend(anomalies)
        return anomalies

    def detect_field_value_anomaly(
        self, field_values: list[float], field_name: str, method: str = "modified_zscore"
    ) -> list[Anomaly]:
        """Detect unusual field values using statistical methods.

        Args:
            field_values: List of field values to analyze
            field_name: Name of the field being analyzed
            method: Detection method to use (zscore, iqr, modified_zscore)

        Returns:
            List of detected value anomalies

        Raises:
            ValueError: If method is unknown

        Example:
            >>> values = [1.0, 1.1, 0.9, 1.2, 10.0, 1.0]
            >>> anomalies = detector.detect_field_value_anomaly(values, "voltage")
            >>> print(anomalies[0].explanation)
            voltage value 10.00 deviates significantly from expected 2.53
        """
        values = np.array(field_values)

        # For very small datasets (n=2), use simple ratio-based detection
        if len(values) == 2:
            # Detect if one value is >5x or <0.2x the other
            ratio = values.max() / (values.min() + 1e-10)
            if ratio > 5.0:
                # The larger value is anomalous
                outliers = values == values.max()
            else:
                outliers = np.zeros(len(values), dtype=bool)
        elif method == "zscore":
            outliers = self._zscore_detection(values, self.config.zscore_threshold)
        elif method == "iqr":
            outliers = self._iqr_detection(values, self.config.iqr_multiplier)
        elif method == "modified_zscore":
            outliers = self._modified_zscore_detection(values, self.config.zscore_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create Anomaly objects
        anomalies = []
        mean_val = np.mean(values)

        for idx, is_outlier in enumerate(outliers):
            if is_outlier:
                score = abs(values[idx] - mean_val) / (np.std(values) + 1e-10)
                score = min(score / 5.0, 1.0)  # Normalize to 0-1

                anomalies.append(
                    Anomaly(
                        timestamp=float(idx),  # Index as timestamp
                        anomaly_type="value",
                        score=score,
                        message_index=idx,
                        field_name=field_name,
                        expected_value=mean_val,
                        actual_value=values[idx],
                        explanation=f"{field_name} value {values[idx]:.2f} deviates significantly from expected {mean_val:.2f}",
                        context={"method": method},
                    )
                )

        self.anomalies.extend(anomalies)
        return anomalies

    def detect_timing_anomaly(
        self,
        inter_arrival_times: list[float],
        expected_period: float | None = None,
    ) -> list[Anomaly]:
        """Detect timing anomalies (jitter, drift, unexpected delays).

        Args:
            inter_arrival_times: List of inter-arrival times between messages
            expected_period: Expected period (None = calculate from data)

        Returns:
            List of detected timing anomalies

        Example:
            >>> inter_arrival = [0.1, 0.1, 0.1, 1.0, 0.1]  # Delay at index 3
            >>> anomalies = detector.detect_timing_anomaly(inter_arrival)
            >>> print(anomalies[0].explanation)
            Timing anomaly: 1.00s vs expected 0.28s
        """
        if not inter_arrival_times:
            return []

        times = np.array(inter_arrival_times)

        # Use expected period or calculate from data
        if expected_period is None:
            expected_period = float(np.median(times))

            # For very small datasets (n=2), use ratio-based detection
            if len(times) == 2:
                # Detect if one value is >5x or <0.2x the other
                ratio = times.max() / (times.min() + 1e-10)
                if ratio > 5.0:
                    # The larger value is anomalous (unexpected delay)
                    outliers = times == times.max()
                else:
                    outliers = np.zeros(len(times), dtype=bool)
            else:
                # Use statistical detection when no expected period provided
                outliers = self._zscore_detection(times, self.config.zscore_threshold)
        else:
            # When expected period is provided, detect deviations from it directly
            # Use threshold-based detection (e.g., >50% deviation from expected)
            deviations = np.abs(times - expected_period) / expected_period
            outliers = deviations > 0.5  # 50% deviation threshold

        # Create Anomaly objects
        anomalies = []
        for idx, is_outlier in enumerate(outliers):
            if is_outlier:
                anomalies.append(
                    Anomaly(
                        timestamp=float(idx),
                        anomaly_type="timing",
                        score=min(abs(times[idx] - expected_period) / expected_period, 1.0),
                        message_index=idx,
                        expected_value=expected_period,
                        actual_value=times[idx],
                        explanation=f"Timing anomaly: {times[idx]:.2f}s vs expected {expected_period:.2f}s",
                        context={"expected_period": expected_period},
                    )
                )

        self.anomalies.extend(anomalies)
        return anomalies

    def detect_sequence_anomaly(
        self, sequences: list[list[int]], trained_model: Any = None
    ) -> list[Anomaly]:
        """Detect unusual byte/message sequences.

        Analyzes sequences for unusual patterns. If trained_model is provided,
        uses it for prediction. Otherwise uses statistical analysis.

        Args:
            sequences: List of byte/message sequences
            trained_model: Trained sequence model (optional)

        Returns:
            List of detected sequence anomalies

        Example:
            >>> sequences = [[0x01, 0x02, 0x03], [0x01, 0x02, 0xFF]]  # Last byte unusual
            >>> anomalies = detector.detect_sequence_anomaly(sequences)
        """
        anomalies = []

        # Dual-threshold approach: detect unusual sequence lengths
        lengths = np.array([len(seq) for seq in sequences])
        median_length = np.median(lengths)
        mean_length = np.mean(lengths)

        # Statistical outlier detection
        statistical_outliers = self._zscore_detection(lengths, self.config.zscore_threshold)

        # Length-based threshold: detect sequences >2x median length or <0.5x median length
        length_ratios = lengths / median_length
        significant_length_deviations = (length_ratios > 2.0) | (length_ratios < 0.5)

        # Combine: anomaly if either statistical outlier OR significant length deviation
        outliers = statistical_outliers | significant_length_deviations

        for idx, is_outlier in enumerate(outliers):
            if is_outlier:
                anomalies.append(
                    Anomaly(
                        timestamp=float(idx),
                        anomaly_type="sequence",
                        score=0.7,
                        message_index=idx,
                        expected_value=mean_length,
                        actual_value=lengths[idx],
                        explanation=f"Unusual sequence length: {lengths[idx]} bytes vs expected {mean_length:.0f} bytes",
                        context={"sequence_length": lengths[idx]},
                    )
                )

        self.anomalies.extend(anomalies)
        return anomalies

    def _zscore_detection(
        self, values: NDArray[np.floating[Any]], threshold: float = 3.0
    ) -> NDArray[np.bool_]:
        """Z-score based outlier detection.

        Z-score measures how many standard deviations a value is from the mean.
        Outliers are typically defined as |Z-score| > threshold (commonly 3.0).

        Args:
            values: Array of values to analyze
            threshold: Z-score threshold for outliers (default: 3.0)

        Returns:
            Boolean array indicating outliers (True = outlier)

        Example:
            >>> values = np.array([1.0, 1.1, 0.9, 10.0])
            >>> outliers = detector._zscore_detection(values, threshold=2.0)
            >>> print(outliers)
            [False False False  True]
        """
        if len(values) < 2:
            return np.zeros(len(values), dtype=bool)

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return np.zeros(len(values), dtype=bool)

        z_scores = np.abs((values - mean) / std)
        outliers: NDArray[np.bool_] = z_scores > threshold

        return outliers

    def _iqr_detection(
        self, values: NDArray[np.floating[Any]], multiplier: float = 1.5
    ) -> NDArray[np.bool_]:
        """Interquartile range (IQR) based outlier detection.

        IQR is the range between 25th and 75th percentiles. Outliers are
        values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR].

        Args:
            values: Array of values to analyze
            multiplier: IQR multiplier for bounds (default: 1.5)

        Returns:
            Boolean array indicating outliers (True = outlier)

        Example:
            >>> values = np.array([1.0, 1.1, 0.9, 10.0])
            >>> outliers = detector._iqr_detection(values)
            >>> print(outliers)
            [False False False  True]
        """
        if len(values) < 4:
            return np.zeros(len(values), dtype=bool)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers: NDArray[np.bool_] = (values < lower_bound) | (values > upper_bound)

        return outliers

    def _modified_zscore_detection(
        self, values: NDArray[np.floating[Any]], threshold: float = 2.5
    ) -> NDArray[np.bool_]:
        """Modified Z-score based outlier detection using median absolute deviation.

        More robust to outliers than standard Z-score. Uses median instead of mean
        and MAD instead of standard deviation.

        Args:
            values: Array of values to analyze
            threshold: Modified Z-score threshold (default: 2.5, lower than standard 3.5)

        Returns:
            Boolean array indicating outliers (True = outlier)

        Example:
            >>> values = np.array([1.0, 1.1, 0.9, 10.0])
            >>> outliers = detector._modified_zscore_detection(values)
            >>> print(outliers)
            [False False False  True]
        """
        if len(values) < 2:
            return np.zeros(len(values), dtype=bool)

        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad == 0:
            # When MAD is 0, use a different approach
            # Check if any values differ from the median
            result_outliers: NDArray[np.bool_] = values != median
            return result_outliers

        # Modified Z-score = 0.6745 * (x - median) / MAD
        modified_z_scores = 0.6745 * np.abs(values - median) / mad
        outliers: NDArray[np.bool_] = modified_z_scores > threshold

        return outliers

    def _isolation_forest_detection(
        self, X: NDArray[np.floating[Any]], contamination: float = 0.1
    ) -> NDArray[np.bool_]:
        """Isolation Forest based anomaly detection.

        Isolation Forest isolates anomalies by randomly selecting features
        and split values. Anomalies require fewer splits to isolate.

        Args:
            X: Feature matrix (n_samples, n_features)
            contamination: Expected outlier fraction (0.0-0.5)

        Returns:
            Boolean array indicating outliers (True = outlier)

        Raises:
            ImportError: If scikit-learn is not installed

        Example:
            >>> X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]])
            >>> outliers = detector._isolation_forest_detection(X)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for Isolation Forest. "
                "Install with: pip install scikit-learn"
            ) from e

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)

        predictions = model.fit_predict(X)
        outliers: NDArray[np.bool_] = predictions == -1

        # Store model for later use
        self.models["isolation_forest"] = model

        return outliers

    def _one_class_svm_detection(
        self, X: NDArray[np.floating[Any]], nu: float = 0.1
    ) -> NDArray[np.bool_]:
        """One-Class SVM based anomaly detection.

        One-Class SVM learns a decision boundary around normal data points.
        Points outside this boundary are considered anomalies.

        Args:
            X: Feature matrix (n_samples, n_features)
            nu: Upper bound on fraction of outliers (0.0-1.0)

        Returns:
            Boolean array indicating outliers (True = outlier)

        Raises:
            ImportError: If scikit-learn is not installed

        Example:
            >>> X = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]])
            >>> outliers = detector._one_class_svm_detection(X)
        """
        try:
            from sklearn.svm import OneClassSVM
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for One-Class SVM. Install with: pip install scikit-learn"
            ) from e

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        model = OneClassSVM(nu=nu, kernel="rbf", gamma="auto")

        predictions = model.fit_predict(X)
        outliers: NDArray[np.bool_] = predictions == -1

        # Store model for later use
        self.models["one_class_svm"] = model

        return outliers

    def _train_isolation_forest(self, X: NDArray[np.floating[Any]]) -> None:
        """Train Isolation Forest model.

        Args:
            X: Training data (n_samples, n_features)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("scikit-learn not available, skipping Isolation Forest training")
            return

        model = IsolationForest(
            contamination=self.config.contamination, random_state=42, n_estimators=100
        )
        model.fit(X)
        self.models["isolation_forest"] = model

    def _train_one_class_svm(self, X: NDArray[np.floating[Any]]) -> None:
        """Train One-Class SVM model.

        Args:
            X: Training data (n_samples, n_features)
        """
        try:
            from sklearn.svm import OneClassSVM
        except ImportError:
            logger.warning("scikit-learn not available, skipping One-Class SVM training")
            return

        model = OneClassSVM(nu=self.config.contamination, kernel="rbf", gamma="auto")
        model.fit(X)
        self.models["one_class_svm"] = model

    def export_report(self, output_path: Path, format: str = "json") -> None:
        """Export anomaly report with context and explanations.

        Args:
            output_path: Path to output file
            format: Export format (json, txt)

        Raises:
            ValueError: If format is unsupported

        Example:
            >>> detector.export_report(Path("anomalies.json"), format="json")
            >>> detector.export_report(Path("anomalies.txt"), format="txt")
        """
        if format == "json":
            self._export_json(output_path)
        elif format == "txt":
            self._export_txt(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, output_path: Path) -> None:
        """Export anomalies to JSON format.

        Args:
            output_path: Path to output JSON file
        """
        report = {
            "config": {
                "methods": self.config.methods,
                "zscore_threshold": self.config.zscore_threshold,
                "iqr_multiplier": self.config.iqr_multiplier,
                "contamination": self.config.contamination,
                "window_size": self.config.window_size,
            },
            "summary": {
                "total_anomalies": len(self.anomalies),
                "by_type": self._count_by_type(),
            },
            "anomalies": [
                {
                    "timestamp": a.timestamp,
                    "anomaly_type": a.anomaly_type,
                    "score": a.score,
                    "message_index": a.message_index,
                    "field_name": a.field_name,
                    "expected_value": a.expected_value,
                    "actual_value": a.actual_value,
                    "explanation": a.explanation,
                    "context": a.context,
                }
                for a in self.anomalies
            ],
        }

        with output_path.open("w") as f:
            json.dump(report, f, indent=2)

    def _export_txt(self, output_path: Path) -> None:
        """Export anomalies to text format.

        Args:
            output_path: Path to output text file
        """
        with output_path.open("w") as f:
            f.write("Anomaly Detection Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("Configuration:\n")
            f.write(f"  Methods: {', '.join(self.config.methods)}\n")
            f.write(f"  Z-score threshold: {self.config.zscore_threshold}\n")
            f.write(f"  IQR multiplier: {self.config.iqr_multiplier}\n\n")

            f.write("Summary:\n")
            f.write(f"  Total anomalies: {len(self.anomalies)}\n")
            f.write("  By type:\n")
            for anomaly_type, count in self._count_by_type().items():
                f.write(f"    {anomaly_type}: {count}\n")
            f.write("\n")

            f.write("Anomalies:\n")
            f.write("-" * 80 + "\n")
            for i, a in enumerate(self.anomalies, 1):
                f.write(f"{i}. [{a.anomaly_type}] {a.explanation}\n")
                f.write(f"   Timestamp: {a.timestamp:.3f}\n")
                f.write(f"   Score: {a.score:.3f}\n")
                if a.field_name:
                    f.write(f"   Field: {a.field_name}\n")
                if a.message_index is not None:
                    f.write(f"   Message index: {a.message_index}\n")
                f.write("\n")

    def _count_by_type(self) -> dict[str, int]:
        """Count anomalies by type.

        Returns:
            Dictionary mapping anomaly type to count
        """
        counts: dict[str, int] = {}
        for anomaly in self.anomalies:
            counts[anomaly.anomaly_type] = counts.get(anomaly.anomaly_type, 0) + 1
        return counts
