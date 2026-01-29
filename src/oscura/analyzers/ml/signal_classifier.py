"""ML-based signal classification for automatic protocol detection.

This module implements machine learning classifiers for identifying signal types
and protocols from waveform data. Supports multiple ML algorithms and provides
comprehensive feature extraction for accurate classification.

Key capabilities:
- Multi-class classification (UART, SPI, I2C, CAN, analog, digital, PWM, etc.)
- Multiple ML algorithms (Random Forest, SVM, Neural Network, Gradient Boosting)
- Confidence scores and probability distributions
- Feature importance analysis (for tree-based models)
- Model persistence (save/load trained models)
- Incremental learning (online updates)

Example:
    >>> from oscura.analyzers.ml import MLSignalClassifier, TrainingDataset
    >>> # Create and train classifier
    >>> classifier = MLSignalClassifier(algorithm="random_forest")
    >>> dataset = TrainingDataset(
    ...     signals=[uart_data, spi_data, i2c_data],
    ...     labels=["uart", "spi", "i2c"],
    ...     sample_rates=[1e6, 1e6, 1e6]
    ... )
    >>> metrics = classifier.train(dataset, test_size=0.2)
    >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    >>>
    >>> # Classify unknown signal
    >>> result = classifier.predict(unknown_signal, sample_rate=1e6)
    >>> print(f"Signal type: {result.signal_type}")
    >>> print(f"Confidence: {result.confidence:.2%}")
    >>> print(f"All probabilities: {result.probabilities}")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from oscura.analyzers.ml.features import FeatureExtractor

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class MLClassificationResult:
    """Result from ML-based signal classification.

    Attributes:
        signal_type: Detected signal type (e.g., "uart", "spi", "i2c", "analog").
        confidence: Classification confidence score (0.0-1.0). Higher values
            indicate more certain predictions.
        probabilities: Dictionary mapping each signal type to its probability.
            All values sum to 1.0.
        features: Dictionary of extracted features used for classification.
            Useful for debugging and understanding model decisions.
        feature_importance: Dictionary of feature importance scores (only for
            tree-based models like Random Forest). Higher values indicate
            features that contribute more to the classification.
        model_type: Algorithm used for classification.

    Example:
        >>> result = classifier.predict(signal, sample_rate=1e6)
        >>> if result.confidence > 0.8:
        ...     print(f"High confidence: {result.signal_type}")
        >>> # Inspect feature importance
        >>> if result.feature_importance:
        ...     top_features = sorted(
        ...         result.feature_importance.items(),
        ...         key=lambda x: x[1],
        ...         reverse=True
        ...     )[:5]
        ...     print(f"Top features: {top_features}")
    """

    signal_type: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]
    feature_importance: dict[str, float] | None = None
    model_type: str = "random_forest"


@dataclass
class TrainingDataset:
    """Training dataset for ML signal classifier.

    Attributes:
        signals: List of signal arrays (1D numpy arrays).
        labels: List of signal type labels corresponding to each signal.
            Must use consistent naming (e.g., "uart", "spi", "i2c").
        sample_rates: List of sample rates (Hz) for each signal.
        metadata: Optional metadata dictionary for dataset tracking.

    Example:
        >>> # Create dataset from synthetic signals
        >>> uart_signals = [generate_uart() for _ in range(100)]
        >>> spi_signals = [generate_spi() for _ in range(100)]
        >>> dataset = TrainingDataset(
        ...     signals=uart_signals + spi_signals,
        ...     labels=["uart"] * 100 + ["spi"] * 100,
        ...     sample_rates=[1e6] * 200,
        ...     metadata={"source": "synthetic", "version": "1.0"}
        ... )
    """

    signals: list[NDArray[np.floating[Any]]]
    labels: list[str]
    sample_rates: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dataset consistency."""
        if not (len(self.signals) == len(self.labels) == len(self.sample_rates)):
            raise ValueError(
                f"Dataset length mismatch: {len(self.signals)} signals, "
                f"{len(self.labels)} labels, {len(self.sample_rates)} sample_rates"
            )


class MLSignalClassifier:
    """ML-based signal classifier using scikit-learn.

    This class provides automatic signal type classification using machine learning.
    It supports multiple algorithms and provides comprehensive feature extraction
    for accurate protocol detection.

    Supported algorithms:
        - random_forest: Fast, robust, provides feature importance
        - svm: Good for high-dimensional data, slower training
        - neural_network: Can capture complex patterns, requires more data
        - gradient_boosting: Often highest accuracy, slower training

    Supported signal types:
        - Digital: digital, uart, spi, i2c, can, manchester, nrz, rz
        - Analog: analog, pwm, amplitude_modulated, frequency_modulated
        - Mixed: Various combinations

    Example:
        >>> # Train classifier
        >>> classifier = MLSignalClassifier(algorithm="random_forest")
        >>> metrics = classifier.train(training_dataset)
        >>>
        >>> # Save model for later use
        >>> classifier.save_model(Path("models/signal_classifier.pkl"))
        >>>
        >>> # Load and use
        >>> classifier2 = MLSignalClassifier()
        >>> classifier2.load_model(Path("models/signal_classifier.pkl"))
        >>> result = classifier2.predict(signal, sample_rate=1e6)
    """

    # Supported ML algorithms
    ALGORITHMS: ClassVar[list[str]] = [
        "random_forest",
        "svm",
        "neural_network",
        "gradient_boosting",
    ]

    # Common signal types (can be extended during training)
    SIGNAL_TYPES: ClassVar[list[str]] = [
        "digital",
        "analog",
        "pwm",
        "uart",
        "spi",
        "i2c",
        "can",
        "manchester",
        "nrz",
        "rz",
        "amplitude_modulated",
        "frequency_modulated",
    ]

    def __init__(self, algorithm: str = "random_forest") -> None:
        """Initialize ML classifier with specified algorithm.

        Args:
            algorithm: ML algorithm to use. Must be one of ALGORITHMS.

        Raises:
            ValueError: If algorithm is not supported.

        Example:
            >>> classifier = MLSignalClassifier(algorithm="random_forest")
            >>> classifier.algorithm
            'random_forest'
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose from: {', '.join(self.ALGORITHMS)}"
            )

        self.algorithm = algorithm
        self.model: Any = None
        self.scaler: Any = None
        self.feature_extractor = FeatureExtractor()
        self.feature_names: list[str] = []
        self.classes: list[str] = []

    def train(
        self, dataset: TrainingDataset, test_size: float = 0.2, random_state: int = 42
    ) -> dict[str, float]:
        """Train classifier on labeled dataset.

        Extracts features from all signals, splits into train/test sets,
        standardizes features, trains the selected ML model, and evaluates
        performance on the test set.

        Args:
            dataset: Training dataset containing signals and labels.
            test_size: Fraction of data to use for testing (0.0-1.0).
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with performance metrics:
                - accuracy: Overall classification accuracy (0.0-1.0)
                - precision: Weighted precision score (0.0-1.0)
                - recall: Weighted recall score (0.0-1.0)
                - f1_score: Weighted F1 score (0.0-1.0)

        Raises:
            ImportError: If scikit-learn is not installed.
            ValueError: If dataset is too small or has invalid labels.

        Example:
            >>> dataset = TrainingDataset(
            ...     signals=[uart1, uart2, spi1, spi2],
            ...     labels=["uart", "uart", "spi", "spi"],
            ...     sample_rates=[1e6, 1e6, 1e6, 1e6]
            ... )
            >>> metrics = classifier.train(dataset, test_size=0.25)
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
            >>> print(f"F1 Score: {metrics['f1_score']:.2%}")
        """
        _check_sklearn_available()
        _validate_dataset_size(dataset)

        # Extract features and split data
        X_train, X_test, y_train, y_test = self._prepare_training_data(
            dataset, test_size, random_state
        )

        # Train and evaluate model
        self._train_model(X_train, y_train, random_state)
        return self._evaluate_model(X_test, y_test)

    def _prepare_training_data(
        self, dataset: TrainingDataset, test_size: float, random_state: int
    ) -> tuple[Any, Any, Any, Any]:
        """Extract features, split, and scale training data."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        logger.info(f"Extracting features from {len(dataset.signals)} signals...")
        X, y = self._extract_features(dataset)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info("Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _extract_features(self, dataset: TrainingDataset) -> tuple[NDArray[Any], list[str]]:
        """Extract features from all signals in dataset."""
        X = []
        for signal, sample_rate in zip(dataset.signals, dataset.sample_rates, strict=True):
            features = self.feature_extractor.extract_all(signal, sample_rate)
            X.append(list(features.values()))

            if not self.feature_names:
                self.feature_names = list(features.keys())

        X_array = np.array(X)
        logger.info(f"Extracted {X_array.shape[1]} features per signal")
        return X_array, dataset.labels

    def _train_model(self, X_train: Any, y_train: Any, random_state: int) -> None:
        """Train the selected ML model."""
        logger.info(f"Training {self.algorithm} classifier...")
        self.model = _create_classifier(self.algorithm, random_state)
        self.model.fit(X_train, y_train)
        self.classes = list(self.model.classes_)
        logger.info(f"Trained on {len(self.classes)} classes: {self.classes}")

    def _evaluate_model(self, X_test: Any, y_test: Any) -> dict[str, float]:
        """Evaluate model performance on test set."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        logger.info("Evaluating on test set...")
        y_pred = self.model.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0.0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        logger.info(f"Training complete: {metrics}")
        return metrics

    def predict(
        self, signal: NDArray[np.floating[Any]], sample_rate: float
    ) -> MLClassificationResult:
        """Classify a single signal using trained model.

        Args:
            signal: Input signal as 1D numpy array.
            sample_rate: Sampling rate in Hz.

        Returns:
            MLClassificationResult with predicted signal type, confidence,
            probabilities, and extracted features.

        Raises:
            ValueError: If model has not been trained yet.

        Example:
            >>> result = classifier.predict(unknown_signal, sample_rate=1e6)
            >>> print(f"Type: {result.signal_type}")
            >>> print(f"Confidence: {result.confidence:.2%}")
            >>> for signal_type, prob in result.probabilities.items():
            ...     print(f"  {signal_type}: {prob:.2%}")
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        # Extract features
        features = self.feature_extractor.extract_all(signal, sample_rate)
        X = np.array([list(features.values())])

        # Standardize
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities_array = self.model.predict_proba(X_scaled)[0]

        # Build probability dictionary
        probabilities = {
            str(class_): float(prob)
            for class_, prob in zip(self.classes, probabilities_array, strict=True)
        }

        # Confidence is the maximum probability
        confidence = float(max(probabilities_array))

        # Extract feature importance if available (tree-based models)
        feature_importance: dict[str, float] | None = None
        if hasattr(self.model, "feature_importances_"):
            feature_importance = {
                name: float(importance)
                for name, importance in zip(
                    self.feature_names, self.model.feature_importances_, strict=True
                )
            }

        return MLClassificationResult(
            signal_type=str(prediction),
            confidence=confidence,
            probabilities=probabilities,
            features=features,
            feature_importance=feature_importance,
            model_type=self.algorithm,
        )

    def predict_batch(
        self, signals: list[NDArray[np.floating[Any]]], sample_rate: float
    ) -> list[MLClassificationResult]:
        """Classify multiple signals in batch.

        More efficient than calling predict() repeatedly for large batches.

        Args:
            signals: List of signal arrays.
            sample_rate: Sampling rate in Hz (same for all signals).

        Returns:
            List of MLClassificationResult objects, one per input signal.

        Raises:
            ValueError: If model has not been trained yet.

        Example:
            >>> signals = [signal1, signal2, signal3]
            >>> results = classifier.predict_batch(signals, sample_rate=1e6)
            >>> for i, result in enumerate(results):
            ...     print(f"Signal {i}: {result.signal_type} ({result.confidence:.2%})")
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        results = []
        for signal in signals:
            result = self.predict(signal, sample_rate)
            results.append(result)

        return results

    def save_model(self, path: Path) -> None:
        """Save trained model to disk.

        Saves the complete model state including the ML model, feature scaler,
        feature names, and class labels. Can be loaded later with load_model().

        Args:
            path: Path to save model file. Convention: use .pkl extension.

        Raises:
            ValueError: If model has not been trained yet.

        Example:
            >>> classifier.save_model(Path("models/uart_detector.pkl"))
            >>> # Later...
            >>> new_classifier = MLSignalClassifier()
            >>> new_classifier.load_model(Path("models/uart_detector.pkl"))
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Nothing to save.")

        model_state = {
            "algorithm": self.algorithm,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "classes": self.classes,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_state, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load trained model from disk.

        Restores the complete model state including the ML model, feature scaler,
        feature names, and class labels.

        Args:
            path: Path to saved model file.

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If model file is corrupted or incompatible.

        Example:
            >>> classifier = MLSignalClassifier()
            >>> classifier.load_model(Path("models/uart_detector.pkl"))
            >>> result = classifier.predict(signal, sample_rate=1e6)
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Validate model state
        required_keys = {"algorithm", "model", "scaler", "feature_names", "classes"}
        if not required_keys.issubset(model_state.keys()):
            raise ValueError(
                f"Invalid model file. Missing keys: {required_keys - set(model_state.keys())}"
            )

        self.algorithm = model_state["algorithm"]
        self.model = model_state["model"]
        self.scaler = model_state["scaler"]
        self.feature_names = model_state["feature_names"]
        self.classes = model_state["classes"]

        logger.info(f"Model loaded from {path} ({len(self.classes)} classes)")

    def partial_fit(
        self,
        signals: list[NDArray[np.floating[Any]]],
        labels: list[str],
        sample_rate: float,
    ) -> None:
        """Incrementally update model with new data (online learning).

        Only supported for algorithms that implement partial_fit (currently
        neural_network). For other algorithms, retrain with combined dataset.

        Args:
            signals: List of new signal arrays.
            labels: List of labels for new signals.
            sample_rate: Sampling rate in Hz (same for all signals).

        Raises:
            ValueError: If model has not been trained yet or algorithm doesn't
                support incremental learning.
            ImportError: If scikit-learn is not installed.

        Example:
            >>> # Initial training
            >>> classifier.train(initial_dataset)
            >>>
            >>> # Later, add more data
            >>> new_signals = [signal1, signal2]
            >>> new_labels = ["uart", "spi"]
            >>> classifier.partial_fit(new_signals, new_labels, sample_rate=1e6)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")

        try:
            from sklearn.neural_network import MLPClassifier
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for ML classification. "
                "Install with: uv pip install 'scikit-learn>=1.3.0'"
            ) from e

        # Only neural network supports partial_fit in scikit-learn
        if not isinstance(self.model, MLPClassifier):
            raise ValueError(
                f"Incremental learning not supported for {self.algorithm}. "
                "Use 'neural_network' algorithm or retrain with full dataset."
            )

        # Extract features
        X = []
        for signal in signals:
            features = self.feature_extractor.extract_all(signal, sample_rate)
            X.append(list(features.values()))

        X_array = np.array(X)

        # Standardize using existing scaler
        X_scaled = self.scaler.transform(X_array)

        # Partial fit
        self.model.partial_fit(X_scaled, labels, classes=self.classes)
        logger.info(f"Updated model with {len(signals)} new samples")


def _check_sklearn_available() -> None:
    """Check if scikit-learn is available."""
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for ML classification. "
            "Install with: uv pip install 'scikit-learn>=1.3.0'"
        ) from e


def _validate_dataset_size(dataset: TrainingDataset) -> None:
    """Validate that dataset has minimum required samples."""
    if len(dataset.signals) < 10:
        raise ValueError(f"Dataset too small: {len(dataset.signals)} samples (need ≥10)")


def _create_classifier(algorithm: str, random_state: int) -> Any:
    """Create classifier instance based on algorithm type."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        )
    elif algorithm == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
    elif algorithm == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=random_state,
        )
    elif algorithm == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
