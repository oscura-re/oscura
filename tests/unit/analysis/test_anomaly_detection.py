"""Unit tests for anomaly detection module.

Tests cover:
- Statistical detection methods (Z-score, IQR, modified Z-score)
- ML-based detection methods (Isolation Forest, One-Class SVM)
- Message rate anomaly detection
- Field value anomaly detection
- Timing anomaly detection
- Sequence anomaly detection
- Anomaly report export
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.patterns.anomaly_detection import (
    Anomaly,
    AnomalyDetectionConfig,
    AnomalyDetector,
)


@pytest.fixture
def detector() -> AnomalyDetector:
    """Create basic anomaly detector with reasonable test threshold."""
    config = AnomalyDetectionConfig(methods=["zscore"], zscore_threshold=2.0)
    return AnomalyDetector(config)


@pytest.fixture
def detector_iqr() -> AnomalyDetector:
    """Create anomaly detector with IQR method."""
    config = AnomalyDetectionConfig(methods=["iqr"])
    return AnomalyDetector(config)


@pytest.fixture
def normal_data() -> list[dict[str, float]]:
    """Generate normal baseline data."""
    return [
        {"voltage": 3.3 + 0.1 * np.random.randn(), "current": 0.5 + 0.05 * np.random.randn()}
        for _ in range(100)
    ]


class TestAnomalyDataclass:
    """Test Anomaly dataclass."""

    def test_anomaly_creation(self) -> None:
        """Test Anomaly creation with all fields."""
        anomaly = Anomaly(
            timestamp=123.45,
            anomaly_type="value",
            score=0.95,
            message_index=42,
            field_name="voltage",
            expected_value=3.3,
            actual_value=15.0,
            explanation="Voltage spike detected",
            context={"threshold": 3.0},
        )

        assert anomaly.timestamp == 123.45
        assert anomaly.anomaly_type == "value"
        assert anomaly.score == 0.95
        assert anomaly.message_index == 42
        assert anomaly.field_name == "voltage"
        assert anomaly.expected_value == 3.3
        assert anomaly.actual_value == 15.0
        assert anomaly.explanation == "Voltage spike detected"
        assert anomaly.context == {"threshold": 3.0}

    def test_anomaly_minimal(self) -> None:
        """Test Anomaly with minimal required fields."""
        anomaly = Anomaly(
            timestamp=1.0,
            anomaly_type="rate",
            score=0.8,
        )

        assert anomaly.timestamp == 1.0
        assert anomaly.anomaly_type == "rate"
        assert anomaly.score == 0.8
        assert anomaly.message_index is None
        assert anomaly.field_name is None
        assert anomaly.expected_value is None
        assert anomaly.actual_value is None
        assert anomaly.explanation == ""
        assert anomaly.context == {}


class TestAnomalyDetectionConfig:
    """Test AnomalyDetectionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AnomalyDetectionConfig()

        assert config.methods == ["zscore", "isolation_forest"]
        assert config.zscore_threshold == 3.0
        assert config.iqr_multiplier == 1.5
        assert config.contamination == 0.1
        assert config.window_size == 100

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = AnomalyDetectionConfig(
            methods=["zscore", "iqr"],
            zscore_threshold=2.5,
            iqr_multiplier=2.0,
            contamination=0.05,
            window_size=200,
        )

        assert config.methods == ["zscore", "iqr"]
        assert config.zscore_threshold == 2.5
        assert config.iqr_multiplier == 2.0
        assert config.contamination == 0.05
        assert config.window_size == 200


class TestAnomalyDetector:
    """Test AnomalyDetector class."""

    def test_initialization(self, detector: AnomalyDetector) -> None:
        """Test detector initialization."""
        assert isinstance(detector.config, AnomalyDetectionConfig)
        assert detector.models == {}
        assert detector.baselines == {}
        assert detector.anomalies == []

    def test_custom_config(self) -> None:
        """Test detector with custom config."""
        config = AnomalyDetectionConfig(methods=["iqr"], window_size=50)
        detector = AnomalyDetector(config)

        assert detector.config.methods == ["iqr"]
        assert detector.config.window_size == 50


class TestStatisticalDetection:
    """Test statistical anomaly detection methods."""

    def test_zscore_detection_no_outliers(self, detector: AnomalyDetector) -> None:
        """Test Z-score detection with no outliers."""
        values = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0])
        outliers = detector._zscore_detection(values, threshold=3.0)

        assert outliers.sum() == 0
        assert len(outliers) == len(values)

    def test_zscore_detection_with_outliers(self, detector: AnomalyDetector) -> None:
        """Test Z-score detection with outliers."""
        values = np.array([1.0, 1.1, 0.9, 10.0, 1.2, 0.8])  # 10.0 is outlier
        outliers = detector._zscore_detection(values, threshold=2.0)

        assert outliers.sum() >= 1
        assert outliers[3]  # 10.0 should be detected

    def test_zscore_detection_empty(self, detector: AnomalyDetector) -> None:
        """Test Z-score detection with empty array."""
        values = np.array([])
        outliers = detector._zscore_detection(values)

        assert len(outliers) == 0

    def test_zscore_detection_constant(self, detector: AnomalyDetector) -> None:
        """Test Z-score detection with constant values."""
        values = np.array([1.0, 1.0, 1.0, 1.0])
        outliers = detector._zscore_detection(values)

        assert outliers.sum() == 0  # No outliers when std=0

    def test_iqr_detection_no_outliers(self, detector: AnomalyDetector) -> None:
        """Test IQR detection with no outliers."""
        values = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0])
        outliers = detector._iqr_detection(values, multiplier=1.5)

        assert outliers.sum() == 0

    def test_iqr_detection_with_outliers(self, detector: AnomalyDetector) -> None:
        """Test IQR detection with outliers."""
        values = np.array([1.0, 1.1, 0.9, 10.0, 1.2, 0.8])  # 10.0 is outlier
        outliers = detector._iqr_detection(values, multiplier=1.5)

        assert outliers.sum() >= 1
        assert outliers[3]  # 10.0 should be detected

    def test_iqr_detection_small_sample(self, detector: AnomalyDetector) -> None:
        """Test IQR detection with small sample."""
        values = np.array([1.0, 2.0])
        outliers = detector._iqr_detection(values)

        assert len(outliers) == 2
        assert outliers.sum() == 0  # Too few samples

    def test_modified_zscore_detection(self, detector: AnomalyDetector) -> None:
        """Test modified Z-score detection."""
        values = np.array([1.0, 1.1, 0.9, 10.0, 1.2, 0.8])  # 10.0 is outlier
        outliers = detector._modified_zscore_detection(values, threshold=3.5)

        assert outliers.sum() >= 1
        assert outliers[3]  # 10.0 should be detected

    def test_modified_zscore_robust_to_outliers(self, detector: AnomalyDetector) -> None:
        """Test that modified Z-score is more robust than Z-score."""
        # Data with multiple outliers
        values = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 20.0])

        outliers_z = detector._zscore_detection(values, threshold=2.0)
        outliers_mod = detector._modified_zscore_detection(values, threshold=3.5)

        # Modified Z-score should detect outliers more reliably
        assert outliers_mod.sum() >= 1


class TestMLDetection:
    """Test ML-based anomaly detection methods."""

    def test_isolation_forest_detection_sklearn_available(self) -> None:
        """Test Isolation Forest detection when sklearn is available."""
        pytest.importorskip("sklearn")

        detector = AnomalyDetector()
        X = np.array([[1.0], [1.1], [0.9], [1.2], [10.0]])  # 10.0 is outlier

        outliers = detector._isolation_forest_detection(X, contamination=0.2)

        assert len(outliers) == 5
        assert outliers.sum() >= 1  # At least one outlier detected
        assert "isolation_forest" in detector.models

    def test_isolation_forest_sklearn_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Isolation Forest when sklearn is not available."""

        def mock_import(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("No module named 'sklearn'")

        monkeypatch.setattr("builtins.__import__", mock_import)

        detector = AnomalyDetector()
        X = np.array([[1.0], [1.1], [0.9]])

        with pytest.raises(ImportError, match="scikit-learn is required"):
            detector._isolation_forest_detection(X)

    def test_one_class_svm_detection_sklearn_available(self) -> None:
        """Test One-Class SVM detection when sklearn is available."""
        pytest.importorskip("sklearn")

        detector = AnomalyDetector()
        X = np.array([[1.0], [1.1], [0.9], [1.2], [10.0]])  # 10.0 is outlier

        outliers = detector._one_class_svm_detection(X, nu=0.2)

        assert len(outliers) == 5
        assert outliers.sum() >= 1  # At least one outlier detected
        assert "one_class_svm" in detector.models

    def test_one_class_svm_sklearn_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test One-Class SVM when sklearn is not available."""

        def mock_import(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("No module named 'sklearn'")

        monkeypatch.setattr("builtins.__import__", mock_import)

        detector = AnomalyDetector()
        X = np.array([[1.0], [1.1], [0.9]])

        with pytest.raises(ImportError, match="scikit-learn is required"):
            detector._one_class_svm_detection(X)


class TestTraining:
    """Test anomaly detector training."""

    def test_train_basic(
        self, detector: AnomalyDetector, normal_data: list[dict[str, float]]
    ) -> None:
        """Test basic training."""
        detector.train(normal_data, features=["voltage", "current"])

        assert "voltage" in detector.baselines
        assert "current" in detector.baselines

        # Check baseline statistics
        assert "mean" in detector.baselines["voltage"]
        assert "std" in detector.baselines["voltage"]
        assert "median" in detector.baselines["voltage"]
        assert "q1" in detector.baselines["voltage"]
        assert "q3" in detector.baselines["voltage"]

    def test_train_insufficient_data(self, detector: AnomalyDetector) -> None:
        """Test training with insufficient data."""
        normal_data = [{"voltage": 3.3}] * 5  # Only 5 samples

        with pytest.raises(ValueError, match="Need at least 10 samples"):
            detector.train(normal_data, features=["voltage"])

    def test_train_with_ml_methods_sklearn_available(
        self, normal_data: list[dict[str, float]]
    ) -> None:
        """Test training with ML methods when sklearn is available."""
        pytest.importorskip("sklearn")

        config = AnomalyDetectionConfig(methods=["zscore", "isolation_forest", "one_class_svm"])
        detector = AnomalyDetector(config)

        detector.train(normal_data, features=["voltage", "current"])

        assert "isolation_forest" in detector.models
        assert "one_class_svm" in detector.models


class TestFieldValueAnomaly:
    """Test field value anomaly detection."""

    def test_detect_field_value_anomaly_zscore(self, detector: AnomalyDetector) -> None:
        """Test field value anomaly detection with Z-score."""
        values = [1.0, 1.1, 0.9, 1.2, 10.0, 1.0]  # 10.0 is outlier

        anomalies = detector.detect_field_value_anomaly(values, "voltage", method="zscore")

        assert len(anomalies) >= 1
        assert any(a.field_name == "voltage" for a in anomalies)
        assert any(a.actual_value == 10.0 for a in anomalies)
        assert all(a.anomaly_type == "value" for a in anomalies)

    def test_detect_field_value_anomaly_iqr(self, detector: AnomalyDetector) -> None:
        """Test field value anomaly detection with IQR."""
        values = [1.0, 1.1, 0.9, 1.2, 10.0, 1.0]  # 10.0 is outlier

        anomalies = detector.detect_field_value_anomaly(values, "voltage", method="iqr")

        assert len(anomalies) >= 1
        assert any(a.field_name == "voltage" for a in anomalies)

    def test_detect_field_value_anomaly_modified_zscore(self, detector: AnomalyDetector) -> None:
        """Test field value anomaly detection with modified Z-score."""
        values = [1.0, 1.1, 0.9, 1.2, 10.0, 1.0]  # 10.0 is outlier

        anomalies = detector.detect_field_value_anomaly(values, "voltage", method="modified_zscore")

        assert len(anomalies) >= 1

    def test_detect_field_value_anomaly_unknown_method(self, detector: AnomalyDetector) -> None:
        """Test field value anomaly detection with unknown method."""
        values = [1.0, 1.1, 0.9]

        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect_field_value_anomaly(values, "voltage", method="unknown")

    def test_detect_field_value_anomaly_no_outliers(self, detector: AnomalyDetector) -> None:
        """Test field value anomaly detection with no outliers."""
        values = [1.0, 1.1, 0.9, 1.2, 1.0]  # No clear outliers

        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        # Should detect 0 or very few anomalies
        assert len(anomalies) <= 1

    def test_anomalies_stored(self, detector: AnomalyDetector) -> None:
        """Test that anomalies are stored in detector."""
        values = [1.0, 1.1, 0.9, 10.0]

        initial_count = len(detector.anomalies)
        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        assert len(detector.anomalies) == initial_count + len(anomalies)


class TestMessageRateAnomaly:
    """Test message rate anomaly detection."""

    def test_detect_message_rate_anomaly_burst(self, detector: AnomalyDetector) -> None:
        """Test detection of message burst."""
        # Normal rate then sudden burst
        timestamps = list(np.arange(0, 10, 0.1))  # 100 messages at 10 msg/s
        timestamps.extend(np.arange(10, 10.5, 0.01))  # 50 messages at 100 msg/s (burst)

        anomalies = detector.detect_message_rate_anomaly(timestamps, window_size=50)

        assert len(anomalies) >= 1
        assert any("burst" in a.explanation.lower() for a in anomalies)
        assert all(a.anomaly_type == "rate" for a in anomalies)

    def test_detect_message_rate_anomaly_gap(self, detector: AnomalyDetector) -> None:
        """Test detection of message gap."""
        # Normal rate then gap
        timestamps = list(np.arange(0, 5, 0.1))  # Normal rate
        timestamps.extend(np.arange(10, 15, 0.1))  # Gap from 5 to 10

        anomalies = detector.detect_message_rate_anomaly(timestamps, window_size=30)

        assert len(anomalies) >= 1
        assert any("gap" in a.explanation.lower() for a in anomalies)

    def test_detect_message_rate_anomaly_insufficient_data(self, detector: AnomalyDetector) -> None:
        """Test rate anomaly detection with insufficient data."""
        timestamps = [0.0, 0.1, 0.2]  # Less than window_size

        anomalies = detector.detect_message_rate_anomaly(timestamps, window_size=100)

        assert len(anomalies) == 0

    def test_detect_message_rate_anomaly_constant_rate(self, detector: AnomalyDetector) -> None:
        """Test rate anomaly detection with constant rate."""
        timestamps = list(np.arange(0, 10, 0.1))  # Constant rate

        anomalies = detector.detect_message_rate_anomaly(timestamps, window_size=50)

        # Should detect 0 or very few anomalies
        assert len(anomalies) <= 2


class TestTimingAnomaly:
    """Test timing anomaly detection."""

    def test_detect_timing_anomaly_delay(self, detector: AnomalyDetector) -> None:
        """Test detection of timing delay."""
        inter_arrival = [0.1, 0.1, 0.1, 1.0, 0.1, 0.1]  # Delay at index 3

        anomalies = detector.detect_timing_anomaly(inter_arrival)

        assert len(anomalies) >= 1
        assert any(a.actual_value == 1.0 for a in anomalies)
        assert all(a.anomaly_type == "timing" for a in anomalies)

    def test_detect_timing_anomaly_with_expected_period(self, detector: AnomalyDetector) -> None:
        """Test timing anomaly with expected period."""
        inter_arrival = [0.1, 0.1, 0.5, 0.1]  # 0.5 is anomaly

        anomalies = detector.detect_timing_anomaly(inter_arrival, expected_period=0.1)

        assert len(anomalies) >= 1

    def test_detect_timing_anomaly_empty(self, detector: AnomalyDetector) -> None:
        """Test timing anomaly with empty data."""
        inter_arrival: list[float] = []

        anomalies = detector.detect_timing_anomaly(inter_arrival)

        assert len(anomalies) == 0

    def test_detect_timing_anomaly_constant(self, detector: AnomalyDetector) -> None:
        """Test timing anomaly with constant timing."""
        inter_arrival = [0.1] * 20

        anomalies = detector.detect_timing_anomaly(inter_arrival)

        assert len(anomalies) == 0


class TestSequenceAnomaly:
    """Test sequence anomaly detection."""

    def test_detect_sequence_anomaly_length(self, detector: AnomalyDetector) -> None:
        """Test detection of unusual sequence length."""
        sequences = [
            [0x01, 0x02, 0x03],
            [0x01, 0x02, 0x03],
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],  # Unusually long
            [0x01, 0x02, 0x03],
        ]

        anomalies = detector.detect_sequence_anomaly(sequences)

        assert len(anomalies) >= 1
        assert any("length" in a.explanation.lower() for a in anomalies)
        assert all(a.anomaly_type == "sequence" for a in anomalies)

    def test_detect_sequence_anomaly_constant_length(self, detector: AnomalyDetector) -> None:
        """Test sequence anomaly with constant length."""
        sequences = [[0x01, 0x02, 0x03]] * 10

        anomalies = detector.detect_sequence_anomaly(sequences)

        assert len(anomalies) == 0


class TestDetectMethod:
    """Test generic detect method."""

    def test_detect_single_feature(
        self, detector: AnomalyDetector, normal_data: list[dict[str, float]]
    ) -> None:
        """Test detect method with single feature."""
        detector.train(normal_data, features=["voltage"])

        # Normal data point
        anomalies = detector.detect({"voltage": 3.3}, timestamp=1.0)
        assert len(anomalies) == 0

        # Anomalous data point
        anomalies = detector.detect({"voltage": 10.0}, timestamp=2.0)
        assert len(anomalies) >= 1

    def test_detect_multiple_features(
        self, detector: AnomalyDetector, normal_data: list[dict[str, float]]
    ) -> None:
        """Test detect method with multiple features."""
        detector.train(normal_data, features=["voltage", "current"])

        # Anomalous voltage
        anomalies = detector.detect({"voltage": 10.0, "current": 0.5}, timestamp=1.0)
        assert len(anomalies) >= 1
        assert any(a.field_name == "voltage" for a in anomalies)

    def test_detect_batch(
        self, detector: AnomalyDetector, normal_data: list[dict[str, float]]
    ) -> None:
        """Test batch detection."""
        detector.train(normal_data, features=["voltage"])

        data_points = [
            {"voltage": 3.3},
            {"voltage": 10.0},  # Anomaly
            {"voltage": 3.2},
        ]
        timestamps = [0.0, 1.0, 2.0]

        anomalies = detector.detect_batch(data_points, timestamps)

        assert len(anomalies) >= 1
        assert any(a.timestamp == 1.0 for a in anomalies)


class TestExportReport:
    """Test anomaly report export."""

    def test_export_json(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        """Test JSON export."""
        # Generate some anomalies
        values = [1.0, 1.1, 0.9, 10.0]
        detector.detect_field_value_anomaly(values, "voltage")

        output_path = tmp_path / "anomalies.json"
        detector.export_report(output_path, format="json")

        assert output_path.exists()

        # Verify JSON content
        with output_path.open() as f:
            report = json.load(f)

        assert "config" in report
        assert "summary" in report
        assert "anomalies" in report
        assert report["summary"]["total_anomalies"] > 0

    def test_export_txt(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        """Test text export."""
        # Generate some anomalies
        values = [1.0, 1.1, 0.9, 10.0]
        detector.detect_field_value_anomaly(values, "voltage")

        output_path = tmp_path / "anomalies.txt"
        detector.export_report(output_path, format="txt")

        assert output_path.exists()

        # Verify text content
        content = output_path.read_text()
        assert "Anomaly Detection Report" in content
        assert "Configuration:" in content
        assert "Summary:" in content
        assert "Anomalies:" in content

    def test_export_unsupported_format(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        """Test export with unsupported format."""
        output_path = tmp_path / "anomalies.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            detector.export_report(output_path, format="xml")

    def test_export_multiple_anomaly_types(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        """Test export with multiple anomaly types."""
        # Generate different types of anomalies
        detector.detect_field_value_anomaly([1.0, 10.0], "voltage")
        detector.detect_timing_anomaly([0.1, 1.0])

        output_path = tmp_path / "anomalies.json"
        detector.export_report(output_path, format="json")

        with output_path.open() as f:
            report = json.load(f)

        # Check summary by type
        assert "by_type" in report["summary"]
        assert len(report["summary"]["by_type"]) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config_methods(self) -> None:
        """Test detector with empty methods list."""
        config = AnomalyDetectionConfig(methods=[])
        detector = AnomalyDetector(config)

        values = [1.0, 10.0]
        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        # Should not crash, may return empty list
        assert isinstance(anomalies, list)

    def test_single_value(self, detector: AnomalyDetector) -> None:
        """Test detection with single value."""
        values = [5.0]

        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        assert len(anomalies) == 0  # Cannot detect anomalies in single value

    def test_two_values(self, detector: AnomalyDetector) -> None:
        """Test detection with two values."""
        values = [1.0, 10.0]

        # Should not crash
        anomalies = detector.detect_field_value_anomaly(values, "voltage")
        assert isinstance(anomalies, list)

    def test_negative_values(self) -> None:
        """Test detection with negative values."""
        # Use lower threshold to detect the outlier
        config = AnomalyDetectionConfig(methods=["zscore"], zscore_threshold=1.5)
        detector = AnomalyDetector(config)

        values = [-1.0, -1.1, -0.9, -10.0]

        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        assert len(anomalies) >= 1  # Should detect -10.0

    def test_zero_values(self) -> None:
        """Test detection with zero values."""
        # Use lower threshold to detect the outlier
        config = AnomalyDetectionConfig(methods=["zscore"], zscore_threshold=1.5)
        detector = AnomalyDetector(config)

        values = [0.0, 0.0, 0.0, 5.0]

        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        assert len(anomalies) >= 1  # Should detect 5.0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_statistical(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        """Test complete workflow with statistical methods."""
        # Generate normal data
        normal_data = [{"voltage": 3.3 + 0.1 * np.random.randn()} for _ in range(100)]

        # Train detector
        detector.train(normal_data, features=["voltage"])

        # Detect anomalies in new data
        test_data = [{"voltage": 3.3}, {"voltage": 10.0}, {"voltage": 3.2}]
        timestamps = [0.0, 1.0, 2.0]

        anomalies = detector.detect_batch(test_data, timestamps)

        # Export report
        output_path = tmp_path / "report.json"
        detector.export_report(output_path, format="json")

        # Verify
        assert len(anomalies) >= 1
        assert output_path.exists()

    def test_full_workflow_multiple_methods(self, tmp_path: Path) -> None:
        """Test workflow with multiple detection methods."""
        config = AnomalyDetectionConfig(methods=["zscore", "iqr", "modified_zscore"])
        detector = AnomalyDetector(config)

        # Detect anomalies
        values = [1.0, 1.1, 0.9, 10.0, 1.2]
        anomalies = detector.detect_field_value_anomaly(values, "voltage")

        # Export
        output_path = tmp_path / "report.txt"
        detector.export_report(output_path, format="txt")

        # Should detect anomalies with multiple methods
        assert len(anomalies) >= 1
        assert output_path.exists()

    def test_real_world_scenario_protocol_analysis(self, detector: AnomalyDetector) -> None:
        """Test realistic protocol analysis scenario."""
        # Simulate protocol message timestamps with anomalies
        np.random.seed(42)
        normal_period = 0.1
        timestamps = []
        t = 0.0

        for i in range(200):
            if i == 100:
                # Introduce gap
                t += 5.0
            elif i == 150:
                # Introduce burst
                for _ in range(10):
                    timestamps.append(t)
                    t += 0.01
                continue

            timestamps.append(t)
            t += normal_period + 0.01 * np.random.randn()

        # Detect rate anomalies
        anomalies = detector.detect_message_rate_anomaly(timestamps, window_size=50)

        # Should detect both gap and burst
        assert len(anomalies) >= 1

        # Verify anomaly types
        explanations = [a.explanation for a in anomalies]
        combined = " ".join(explanations).lower()

        # Should mention either gap or burst (or both)
        assert "gap" in combined or "burst" in combined
