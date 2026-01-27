"""Machine learning-based signal analysis module.

This module provides ML-based signal classification and analysis tools:
- Automatic protocol detection using supervised learning
- Feature extraction (statistical, spectral, temporal, entropy)
- Multiple ML algorithms (Random Forest, SVM, Neural Networks)
- Model persistence and incremental learning

Supported signal types:
- Digital protocols: UART, SPI, I2C, CAN, Manchester, NRZ, RZ
- Analog signals: PWM, AM, FM, analog baseband
- General classifications: digital, analog, mixed-signal

Example:
    >>> from oscura.analyzers.ml import MLSignalClassifier, TrainingDataset
    >>> classifier = MLSignalClassifier(algorithm="random_forest")
    >>> dataset = TrainingDataset(
    ...     signals=[uart_signal, spi_signal],
    ...     labels=["uart", "spi"],
    ...     sample_rates=[1e6, 1e6]
    ... )
    >>> metrics = classifier.train(dataset)
    >>> result = classifier.predict(unknown_signal, sample_rate=1e6)
    >>> print(f"Detected: {result.signal_type} ({result.confidence:.2%})")
"""

from oscura.analyzers.ml.features import FeatureExtractor
from oscura.analyzers.ml.signal_classifier import (
    MLClassificationResult,
    MLSignalClassifier,
    TrainingDataset,
)

__all__ = [
    "FeatureExtractor",
    "MLClassificationResult",
    "MLSignalClassifier",
    "TrainingDataset",
]
