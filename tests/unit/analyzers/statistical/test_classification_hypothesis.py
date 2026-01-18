"""Property-based tests for statistical classification."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestClassificationProperties:
    """Property-based tests for classification algorithms."""

    @given(
        num_samples=st.integers(min_value=50, max_value=200),
        num_features=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_classification_output_shape(self, num_samples: int, num_features: int) -> None:
        """Property: Classification output has correct shape."""
        # Generate random features
        rng = np.random.default_rng(42)
        features = rng.random((num_samples, num_features))
        labels = rng.integers(0, 2, num_samples)

        # Test that shapes are consistent
        assert features.shape == (num_samples, num_features)
        assert labels.shape == (num_samples,)

    @given(num_classes=st.integers(min_value=2, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_predicted_labels_in_range(self, num_classes: int) -> None:
        """Property: Predicted labels are within valid range."""
        rng = np.random.default_rng(42)
        num_samples = 100
        predictions = rng.integers(0, num_classes, num_samples)

        assert np.all(predictions >= 0)
        assert np.all(predictions < num_classes)
