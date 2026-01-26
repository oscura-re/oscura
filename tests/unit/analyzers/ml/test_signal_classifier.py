"""Tests for ML-based signal classifier."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.ml import MLSignalClassifier, TrainingDataset


class TestMLSignalClassifier:
    """Test ML signal classifier."""

    @pytest.fixture
    def classifier(self) -> MLSignalClassifier:
        """Create default classifier."""
        return MLSignalClassifier(algorithm="random_forest")

    @pytest.fixture
    def synthetic_dataset(self) -> TrainingDataset:
        """Generate synthetic training dataset with distinguishable signals."""
        np.random.seed(42)
        sample_rate = 10000.0
        n_samples = 50  # Small dataset for fast tests

        signals = []
        labels = []
        sample_rates = []

        # Generate UART-like signals (serial, high zero-crossing rate)
        for _ in range(n_samples):
            t = np.linspace(0, 0.1, 1000)
            # Random bit pattern
            bits = np.random.choice([0, 1], size=100)
            uart_signal = np.repeat(bits, 10)
            signals.append(uart_signal.astype(float))
            labels.append("uart")
            sample_rates.append(sample_rate)

        # Generate SPI-like signals (clock + data, periodic)
        for _ in range(n_samples):
            t = np.linspace(0, 0.1, 1000)
            # Clock signal
            clock = np.sign(np.sin(2 * np.pi * 100 * t))
            # Data signal
            data = np.random.choice([-1, 1], size=1000)
            spi_signal = clock * 0.5 + data * 0.5
            signals.append(spi_signal)
            labels.append("spi")
            sample_rates.append(sample_rate)

        # Generate PWM-like signals (varying duty cycle)
        for _ in range(n_samples):
            t = np.linspace(0, 0.1, 1000)
            duty_cycle = 0.2 + 0.6 * np.random.random()
            pwm_signal = np.where((t % 0.01) < (0.01 * duty_cycle), 1.0, 0.0)
            signals.append(pwm_signal)
            labels.append("pwm")
            sample_rates.append(sample_rate)

        return TrainingDataset(
            signals=signals,
            labels=labels,
            sample_rates=sample_rates,
            metadata={"source": "synthetic"},
        )

    def test_init_valid_algorithm(self) -> None:
        """Test initialization with valid algorithms."""
        for algorithm in MLSignalClassifier.ALGORITHMS:
            classifier = MLSignalClassifier(algorithm=algorithm)
            assert classifier.algorithm == algorithm
            assert classifier.model is None

    def test_init_invalid_algorithm(self) -> None:
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            MLSignalClassifier(algorithm="invalid_algorithm")

    def test_training_dataset_validation(self) -> None:
        """Test training dataset validation."""
        # Mismatched lengths should raise error
        with pytest.raises(ValueError, match="length mismatch"):
            TrainingDataset(
                signals=[np.array([1, 2, 3])], labels=["uart", "spi"], sample_rates=[1000.0]
            )

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_train_random_forest(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test training with random forest algorithm."""
        pytest.importorskip("sklearn")

        metrics = classifier.train(synthetic_dataset, test_size=0.2, random_state=42)

        # Check metrics are returned
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Metrics should be reasonable (>50% for this simple dataset)
        assert metrics["accuracy"] > 0.5
        assert 0.0 <= metrics["accuracy"] <= 1.0

        # Model should be trained
        assert classifier.model is not None
        assert classifier.scaler is not None
        assert len(classifier.feature_names) > 0
        assert len(classifier.classes) == 3  # uart, spi, pwm

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_train_svm(self, synthetic_dataset: TrainingDataset) -> None:
        """Test training with SVM algorithm."""
        pytest.importorskip("sklearn")

        classifier = MLSignalClassifier(algorithm="svm")
        metrics = classifier.train(synthetic_dataset, test_size=0.2, random_state=42)

        assert metrics["accuracy"] > 0.5
        assert classifier.model is not None

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_train_neural_network(self, synthetic_dataset: TrainingDataset) -> None:
        """Test training with neural network algorithm."""
        pytest.importorskip("sklearn")

        classifier = MLSignalClassifier(algorithm="neural_network")
        metrics = classifier.train(synthetic_dataset, test_size=0.2, random_state=42)

        assert metrics["accuracy"] > 0.3  # Neural nets need more data, lower threshold
        assert classifier.model is not None

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_train_gradient_boosting(self, synthetic_dataset: TrainingDataset) -> None:
        """Test training with gradient boosting algorithm."""
        pytest.importorskip("sklearn")

        classifier = MLSignalClassifier(algorithm="gradient_boosting")
        metrics = classifier.train(synthetic_dataset, test_size=0.2, random_state=42)

        assert metrics["accuracy"] > 0.5
        assert classifier.model is not None

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_predict(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test prediction on single signal."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        # Create test signal (UART-like)
        test_signal = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 100, dtype=float)
        sample_rate = 10000.0

        result = classifier.predict(test_signal, sample_rate)

        # Check result structure
        assert result.signal_type in classifier.classes
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == len(classifier.classes)
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.01  # Probabilities sum to 1
        assert len(result.features) > 0
        assert result.model_type == "random_forest"

        # Random forest should have feature importance
        assert result.feature_importance is not None
        assert len(result.feature_importance) == len(result.features)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_predict_batch(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test batch prediction."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        # Create multiple test signals
        signals = [
            np.array([0, 0, 1, 1, 0, 1] * 100, dtype=float),
            np.array([1, 0, 1, 0, 1, 0] * 100, dtype=float),
            np.array([1, 1, 1, 0, 0, 0] * 100, dtype=float),
        ]
        sample_rate = 10000.0

        results = classifier.predict_batch(signals, sample_rate)

        # Check results
        assert len(results) == len(signals)
        for result in results:
            assert result.signal_type in classifier.classes
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_predict_without_training(self, classifier: MLSignalClassifier) -> None:
        """Test prediction fails without training."""
        signal = np.array([1, 2, 3, 4, 5], dtype=float)

        with pytest.raises(ValueError, match="Model not trained"):
            classifier.predict(signal, sample_rate=1000.0)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_save_load_model(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test model persistence."""
        pytest.importorskip("sklearn")

        # Train model
        classifier.train(synthetic_dataset, random_state=42)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            classifier.save_model(model_path)

            # Check file exists
            assert model_path.exists()

            # Load in new classifier
            new_classifier = MLSignalClassifier()
            new_classifier.load_model(model_path)

            # Check loaded state
            assert new_classifier.algorithm == classifier.algorithm
            assert new_classifier.classes == classifier.classes
            assert new_classifier.feature_names == classifier.feature_names

            # Test prediction works
            test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
            result = new_classifier.predict(test_signal, sample_rate=10000.0)
            assert result.signal_type in new_classifier.classes

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_save_without_training(self, classifier: MLSignalClassifier) -> None:
        """Test save fails without training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"

            with pytest.raises(ValueError, match="Model not trained"):
                classifier.save_model(model_path)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_load_nonexistent_file(self, classifier: MLSignalClassifier) -> None:
        """Test load fails for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            classifier.load_model(Path("/nonexistent/model.pkl"))

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_partial_fit_neural_network(self, synthetic_dataset: TrainingDataset) -> None:
        """Test incremental learning with neural network."""
        pytest.importorskip("sklearn")

        # Train initial model
        classifier = MLSignalClassifier(algorithm="neural_network")
        classifier.train(synthetic_dataset, random_state=42)

        # Add more data
        new_signals = [
            np.array([1, 0, 1, 0] * 100, dtype=float),
            np.array([0, 1, 0, 1] * 100, dtype=float),
        ]
        new_labels = ["uart", "spi"]
        sample_rate = 10000.0

        # Should not raise error
        classifier.partial_fit(new_signals, new_labels, sample_rate)

        # Verify model was updated
        assert classifier.model is not None
        # Verify can still make predictions
        prediction = classifier.predict(new_signals[0], sample_rate)
        assert prediction in ["uart", "spi", "i2c"]

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_partial_fit_unsupported_algorithm(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test partial_fit fails for unsupported algorithms."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        new_signals = [np.array([1, 0, 1, 0] * 100, dtype=float)]
        new_labels = ["uart"]

        with pytest.raises(ValueError, match="Incremental learning not supported"):
            classifier.partial_fit(new_signals, new_labels, sample_rate=10000.0)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_train_small_dataset(self, classifier: MLSignalClassifier) -> None:
        """Test training fails with too small dataset."""
        pytest.importorskip("sklearn")

        # Create tiny dataset
        small_dataset = TrainingDataset(
            signals=[np.array([1, 2, 3])], labels=["uart"], sample_rates=[1000.0]
        )

        with pytest.raises(ValueError, match="Dataset too small"):
            classifier.train(small_dataset)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_sklearn_not_installed(
        self, classifier: MLSignalClassifier, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test graceful failure when scikit-learn not installed."""

        # Mock ImportError for sklearn
        def mock_import(*args: object, **kwargs: object) -> None:
            raise ImportError("No module named 'sklearn'")

        monkeypatch.setattr("builtins.__import__", mock_import)

        dataset = TrainingDataset(
            signals=[np.random.randn(100) for _ in range(10)],
            labels=["uart"] * 10,
            sample_rates=[1000.0] * 10,
        )

        with pytest.raises(ImportError, match="scikit-learn is required"):
            classifier.train(dataset)

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_feature_importance_random_forest(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test feature importance is available for random forest."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
        result = classifier.predict(test_signal, sample_rate=10000.0)

        # Random forest should provide feature importance
        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0

        # Importance values should sum to ~1.0
        total_importance = sum(result.feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_feature_importance_svm(self, synthetic_dataset: TrainingDataset) -> None:
        """Test feature importance not available for SVM."""
        pytest.importorskip("sklearn")

        classifier = MLSignalClassifier(algorithm="svm")
        classifier.train(synthetic_dataset, random_state=42)

        test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
        result = classifier.predict(test_signal, sample_rate=10000.0)

        # SVM does not provide feature importance
        assert result.feature_importance is None

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_classification_consistency(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test that classification is deterministic."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
        sample_rate = 10000.0

        # Predict twice
        result1 = classifier.predict(test_signal, sample_rate)
        result2 = classifier.predict(test_signal, sample_rate)

        # Should be identical
        assert result1.signal_type == result2.signal_type
        assert result1.confidence == result2.confidence
        assert result1.probabilities == result2.probabilities

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_multiclass_probabilities(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test that all classes have probability predictions."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
        result = classifier.predict(test_signal, sample_rate=10000.0)

        # All trained classes should have probabilities
        for class_name in classifier.classes:
            assert class_name in result.probabilities
            assert 0.0 <= result.probabilities[class_name] <= 1.0

        # Probabilities should sum to 1
        total_prob = sum(result.probabilities.values())
        assert abs(total_prob - 1.0) < 0.01

    @pytest.mark.unit
    @pytest.mark.analyzer
    def test_confidence_equals_max_probability(
        self, classifier: MLSignalClassifier, synthetic_dataset: TrainingDataset
    ) -> None:
        """Test that confidence equals maximum probability."""
        pytest.importorskip("sklearn")

        classifier.train(synthetic_dataset, random_state=42)

        test_signal = np.array([0, 1, 0, 1] * 100, dtype=float)
        result = classifier.predict(test_signal, sample_rate=10000.0)

        max_prob = max(result.probabilities.values())
        assert abs(result.confidence - max_prob) < 0.01
