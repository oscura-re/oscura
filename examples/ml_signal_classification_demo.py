"""ML Signal Classification Demo.

Demonstrates automatic protocol detection using machine learning with the
MLSignalClassifier. Shows training on synthetic signals and classification
of unknown signals with confidence scores.
"""

import numpy as np

from oscura.analyzers.ml import MLSignalClassifier, TrainingDataset


def generate_uart_signal(n_samples: int = 1000) -> np.ndarray:
    """Generate synthetic UART-like signal (serial, random bits)."""
    bits = np.random.choice([0, 1], size=n_samples // 10)
    return np.repeat(bits, 10).astype(float)


def generate_spi_signal(n_samples: int = 1000) -> np.ndarray:
    """Generate synthetic SPI-like signal (clock + data)."""
    t = np.linspace(0, 1, n_samples)
    clock = np.sign(np.sin(2 * np.pi * 100 * t))
    data = np.random.choice([-1, 1], size=n_samples)
    return clock * 0.5 + data * 0.5


def generate_pwm_signal(n_samples: int = 1000) -> np.ndarray:
    """Generate synthetic PWM signal (varying duty cycle)."""
    t = np.linspace(0, 1, n_samples)
    duty_cycle = 0.3
    return np.where((t % 0.01) < (0.01 * duty_cycle), 1.0, 0.0)


def main() -> None:
    """Run ML signal classification demo."""
    print("=" * 70)
    print("ML Signal Classification Demo")
    print("=" * 70)

    # Create training dataset with synthetic signals
    print("\n1. Generating training dataset...")
    n_samples_per_class = 50
    sample_rate = 10000.0

    signals = []
    labels = []
    sample_rates = []

    # Generate UART signals
    for _ in range(n_samples_per_class):
        signals.append(generate_uart_signal())
        labels.append("uart")
        sample_rates.append(sample_rate)

    # Generate SPI signals
    for _ in range(n_samples_per_class):
        signals.append(generate_spi_signal())
        labels.append("spi")
        sample_rates.append(sample_rate)

    # Generate PWM signals
    for _ in range(n_samples_per_class):
        signals.append(generate_pwm_signal())
        labels.append("pwm")
        sample_rates.append(sample_rate)

    dataset = TrainingDataset(
        signals=signals,
        labels=labels,
        sample_rates=sample_rates,
        metadata={"source": "synthetic", "demo": "ml_classification"},
    )

    print(f"   Generated {len(signals)} total signals")
    print(f"   Classes: {set(labels)}")

    # Train classifier
    print("\n2. Training Random Forest classifier...")
    classifier = MLSignalClassifier(algorithm="random_forest")

    try:
        metrics = classifier.train(dataset, test_size=0.2, random_state=42)

        print(f"   Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall:    {metrics['recall']:.2%}")
        print(f"   F1 Score:  {metrics['f1_score']:.2%}")
        print(f"   Classes:   {classifier.classes}")

        # Classify unknown signals
        print("\n3. Classifying unknown signals...")

        test_signals = [
            ("UART", generate_uart_signal()),
            ("SPI", generate_spi_signal()),
            ("PWM", generate_pwm_signal()),
        ]

        for true_type, signal in test_signals:
            result = classifier.predict(signal, sample_rate)

            print(f"\n   True type: {true_type}")
            print(f"   Predicted: {result.signal_type}")
            print(f"   Confidence: {result.confidence:.2%}")
            print("   Probabilities:")
            for signal_type, prob in sorted(
                result.probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"     {signal_type:10s}: {prob:.2%}")

            if result.feature_importance:
                print("   Top 5 features:")
                top_features = sorted(
                    result.feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:5]
                for feature, importance in top_features:
                    print(f"     {feature:25s}: {importance:.4f}")

        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)

    except ImportError:
        print("\n⚠ scikit-learn not installed. Install with:")
        print("   uv pip install 'scikit-learn>=1.3.0'")
        print("\nThis feature is optional and gracefully degraded.")


if __name__ == "__main__":
    main()
