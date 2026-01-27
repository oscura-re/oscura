"""Property-based tests for clustering algorithms.

Tests clustering functionality used for pattern discovery.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import clustering_data

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestClusteringProperties:
    """Property-based tests for clustering algorithms."""

    @given(data=clustering_data(min_points=10, max_points=100, dimensions=2))
    @settings(max_examples=30, deadline=None)
    def test_cluster_labels_bounded(self, data: np.ndarray) -> None:
        """Property: Cluster labels are within valid range."""
        try:
            from oscura.analyzers.patterns.clustering import (
                cluster_messages,
            )
        except ImportError:
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            pytest.skip("clustering module not available")

        n_clusters = 3
        labels = cluster_messages(data, n_clusters=n_clusters)

        # Labels should be in range [0, n_clusters)
        assert np.all(labels >= 0)
        assert np.all(labels < n_clusters)

    @given(
        data=clustering_data(min_points=20, max_points=50, dimensions=2),
        n_clusters=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_points_assigned_to_clusters(self, data: np.ndarray, n_clusters: int) -> None:
        """Property: All data points are assigned to a cluster."""
        try:
            from oscura.analyzers.patterns.clustering import (
                cluster_messages,
            )
        except ImportError:
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            pytest.skip("clustering module not available")

        # Ensure we don't ask for more clusters than points
        assume(n_clusters <= len(data))

        labels = cluster_messages(data, n_clusters=n_clusters)

        # Every point should have a label
        assert len(labels) == len(data)
        assert not np.any(np.isnan(labels))

    @given(data=clustering_data(min_points=15, max_points=50, dimensions=2))
    @settings(max_examples=30, deadline=None)
    def test_clustering_deterministic_with_seed(self, data: np.ndarray) -> None:
        """Property: Clustering is deterministic with fixed random seed."""
        try:
            from oscura.analyzers.patterns.clustering import (
                cluster_messages,
            )
        except ImportError:
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            # SKIP: Valid - Optional clustering module (scikit-learn)
            # Only skip if sklearn not installed (pip install oscura[ml])
            pytest.skip("clustering module not available")

        n_clusters = 3

        # Run clustering twice with same seed
        labels1 = cluster_messages(data, n_clusters=n_clusters, random_state=42)
        labels2 = cluster_messages(data, n_clusters=n_clusters, random_state=42)

        # Should produce identical results
        assert np.array_equal(labels1, labels2)
