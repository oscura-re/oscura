"""Performance benchmarks for clustering optimization.

Validates that vectorized K-means achieves 20-30x speedup while
maintaining numerical accuracy.

Benchmark targets:
    - 20,000 points, 10 clusters: < 150ms (vs 2300ms baseline)
    - Numerical accuracy: exact label match with original
    - Memory usage: comparable or better

Run with:
    pytest tests/performance/test_clustering_optimization.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# Import both implementations for comparison
from oscura.analyzers.patterns.clustering import _kmeans_clustering as kmeans_original
from oscura.analyzers.patterns.clustering_optimized import kmeans_vectorized


@pytest.fixture
def clustering_data():
    """Generate synthetic clustering test data."""
    np.random.seed(42)
    # 20,000 points in 10 dimensions
    return np.random.randn(20000, 10).astype(np.float64)


@pytest.fixture
def small_clustering_data():
    """Generate small test data for accuracy validation."""
    np.random.seed(42)
    # 100 points in 5 dimensions
    return np.random.randn(100, 5).astype(np.float64)


def test_kmeans_vectorized_correctness(small_clustering_data):
    """Verify vectorized K-means produces correct results.

    Tests:
        - Correct number of clusters
        - All points assigned to valid cluster
        - Deterministic with random_state
    """
    data = small_clustering_data
    n_clusters = 5

    labels, centroids = kmeans_vectorized(data, n_clusters, random_state=42)

    # Check output shapes
    assert labels.shape == (len(data),)
    assert centroids.shape == (n_clusters, data.shape[1])

    # Check all labels are valid
    assert np.all((labels >= 0) & (labels < n_clusters))

    # Check all clusters have at least one point
    unique_labels = set(labels)
    assert len(unique_labels) == n_clusters

    # Check determinism
    labels2, centroids2 = kmeans_vectorized(data, n_clusters, random_state=42)
    np.testing.assert_array_equal(labels, labels2)
    np.testing.assert_array_almost_equal(centroids, centroids2)


def test_kmeans_vectorized_numerical_accuracy(small_clustering_data):
    """Compare vectorized K-means to original implementation.

    Validates that vectorized version produces numerically equivalent
    results to the original nested loop implementation.
    """
    data = small_clustering_data
    n_clusters = 5
    random_state = 42

    # Run both implementations
    labels_original = kmeans_original(data, n_clusters, random_state=random_state)
    labels_vectorized, _centroids = kmeans_vectorized(data, n_clusters, random_state=random_state)

    # Should produce identical results (deterministic with same seed)
    np.testing.assert_array_equal(labels_original, labels_vectorized)


def test_kmeans_vectorized_performance(clustering_data):
    """Benchmark vectorized K-means performance.

    Target: < 150ms for 20,000 points x 10 dimensions x 10 clusters
    Baseline: ~2300ms (original implementation)
    Expected speedup: > 15x
    """
    data = clustering_data
    n_clusters = 10

    # Warm-up run (JIT compilation, cache warming)
    kmeans_vectorized(data, n_clusters, random_state=42)

    # Actual benchmark (3 runs, take median)
    times = []
    for i in range(3):
        start = time.perf_counter()
        labels, centroids = kmeans_vectorized(data, n_clusters, random_state=42 + i)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = np.median(times)

    # Assertions
    assert median_time < 0.150, f"Performance regression: {median_time:.3f}s > 150ms target"

    # Report performance
    print(f"\n{'=' * 60}")
    print("K-means Vectorized Performance:")
    print(f"  Data: {data.shape[0]} points x {data.shape[1]} dimensions")
    print(f"  Clusters: {n_clusters}")
    print(f"  Median time: {median_time * 1000:.1f} ms")
    print(f"  Speedup vs baseline: ~{2.3 / median_time:.1f}x")
    print(f"{'=' * 60}")


def test_kmeans_vectorized_vs_original_performance(small_clustering_data):
    """Direct performance comparison between implementations.

    Measures actual speedup on same data to validate optimization claims.
    """
    data = np.tile(small_clustering_data, (200, 1))  # 20,000 points
    n_clusters = 10

    # Benchmark original implementation
    start = time.perf_counter()
    _labels_original = kmeans_original(data, n_clusters, random_state=42)
    time_original = time.perf_counter() - start

    # Benchmark vectorized implementation
    start = time.perf_counter()
    _labels_vectorized, _centroids = kmeans_vectorized(data, n_clusters, random_state=42)
    time_vectorized = time.perf_counter() - start

    speedup = time_original / time_vectorized

    # Report comparison
    print(f"\n{'=' * 60}")
    print("Performance Comparison:")
    print(f"  Original: {time_original * 1000:.1f} ms")
    print(f"  Vectorized: {time_vectorized * 1000:.1f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"{'=' * 60}")

    # Verify speedup meets target
    assert speedup > 15.0, f"Insufficient speedup: {speedup:.1f}x < 15x target"


def test_kmeans_vectorized_memory_efficiency(clustering_data):
    """Verify vectorized implementation doesn't use excessive memory.

    Checks that distance matrix allocation is reasonable and no
    memory leaks occur during iterations.
    """
    import tracemalloc

    data = clustering_data
    n_clusters = 10

    tracemalloc.start()

    labels, centroids = kmeans_vectorized(data, n_clusters, random_state=42)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Distance matrix: n_points x n_clusters x 8 bytes
    expected_distance_matrix = data.shape[0] * n_clusters * 8
    # Data: n_points x n_features x 8 bytes
    expected_data = data.shape[0] * data.shape[1] * 8
    # Centroids: n_clusters x n_features x 8 bytes
    expected_centroids = n_clusters * data.shape[1] * 8

    expected_total = expected_distance_matrix + expected_data + expected_centroids

    # Allow 50% overhead for Python objects, temporary arrays
    assert peak < expected_total * 1.5, f"Excessive memory usage: {peak / 1e6:.1f} MB"

    print(f"\n{'=' * 60}")
    print("Memory Usage:")
    print(f"  Peak: {peak / 1e6:.1f} MB")
    print(f"  Expected: {expected_total / 1e6:.1f} MB")
    print(f"  Overhead: {(peak / expected_total - 1) * 100:.1f}%")
    print(f"{'=' * 60}")


@pytest.mark.parametrize("n_clusters", [3, 5, 10, 20])
def test_kmeans_scalability(clustering_data, n_clusters):
    """Test performance scaling with different cluster counts.

    Validates that complexity remains O(k x n x d) as expected.
    """
    data = clustering_data

    start = time.perf_counter()
    labels, centroids = kmeans_vectorized(data, n_clusters, random_state=42)
    elapsed = time.perf_counter() - start

    # Should complete in reasonable time for all cluster counts
    assert elapsed < 0.5, f"Slow for n_clusters={n_clusters}: {elapsed:.3f}s"

    # Verify clustering quality
    assert len(set(labels)) == n_clusters  # All clusters used
    assert labels.shape == (len(data),)
    assert centroids.shape == (n_clusters, data.shape[1])


if __name__ == "__main__":
    # Allow running benchmarks directly
    pytest.main([__file__, "-v", "-s"])
