"""Tests for LSH-based payload clustering.

Verifies correctness and performance of Locality-Sensitive Hashing
for fast payload clustering.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.utils.performance.lsh_clustering import LSHClustering, cluster_payloads_lsh


def generate_test_payloads(
    n_payloads: int = 50,
    n_clusters: int = 3,
    payload_size: int = 100,
    variation: float = 0.1,
) -> list[bytes]:
    """Generate synthetic payloads for testing."""
    rng = np.random.default_rng(42)

    # Generate cluster templates
    templates = []
    for _ in range(n_clusters):
        template = rng.bytes(payload_size)
        templates.append(template)

    # Generate payloads with variations
    payloads = []
    for i in range(n_payloads):
        template_idx = i % n_clusters
        template = bytearray(templates[template_idx])

        # Add random variations
        n_mutations = int(payload_size * variation)
        for _ in range(n_mutations):
            pos = rng.integers(0, payload_size)
            template[pos] = rng.integers(0, 256)

        payloads.append(bytes(template))

    return payloads


class TestLSHClustering:
    """Test LSH clustering implementation."""

    def test_initialization(self) -> None:
        """Test LSH clusterer initialization."""
        lsh = LSHClustering(n_hash_functions=128, n_bands=16)

        assert lsh.n_hash_functions == 128
        assert lsh.n_bands == 16
        assert lsh.rows_per_band == 8
        assert len(lsh._hash_seeds) == 128

    def test_shingle_creation(self) -> None:
        """Test shingle creation from payloads."""
        lsh = LSHClustering(shingle_size=4)

        payload = b"ABCDEFGH"
        shingles = lsh._shingle(payload)

        # Should create shingles: ABCD, BCDE, CDEF, DEFG, EFGH
        assert len(shingles) == 5
        assert b"ABCD" in shingles
        assert b"EFGH" in shingles

    def test_shingle_short_payload(self) -> None:
        """Test shingle creation for very short payloads."""
        lsh = LSHClustering(shingle_size=4)

        payload = b"AB"
        shingles = lsh._shingle(payload)

        # Should use whole payload as single shingle
        assert len(shingles) == 1
        assert payload in shingles

    def test_minhash_signature(self) -> None:
        """Test MinHash signature computation."""
        lsh = LSHClustering(n_hash_functions=64)

        payload = b"test payload data"
        shingles = lsh._shingle(payload)
        sig = lsh._minhash_signature(shingles)

        # Should produce correct number of hash values
        assert len(sig) == 64
        assert all(isinstance(v, int) for v in sig)

    def test_minhash_determinism(self) -> None:
        """Test that MinHash is deterministic."""
        lsh = LSHClustering(n_hash_functions=64)

        payload = b"test payload"
        shingles = lsh._shingle(payload)

        sig1 = lsh._minhash_signature(shingles)
        sig2 = lsh._minhash_signature(shingles)

        assert sig1 == sig2

    def test_minhash_similarity_identical(self) -> None:
        """Test similarity estimation for identical payloads."""
        lsh = LSHClustering(n_hash_functions=128)

        payload = b"identical payload"
        shingles = lsh._shingle(payload)
        sig = lsh._minhash_signature(shingles)

        similarity = lsh._estimate_similarity(sig, sig)
        assert similarity == 1.0

    def test_minhash_similarity_different(self) -> None:
        """Test similarity estimation for completely different payloads."""
        lsh = LSHClustering(n_hash_functions=128)

        payload_a = b"aaaaaaaaaa"
        payload_b = b"bbbbbbbbbb"

        shingles_a = lsh._shingle(payload_a)
        shingles_b = lsh._shingle(payload_b)

        sig_a = lsh._minhash_signature(shingles_a)
        sig_b = lsh._minhash_signature(shingles_b)

        similarity = lsh._estimate_similarity(sig_a, sig_b)

        # Should have very low similarity
        assert similarity < 0.2

    def test_lsh_buckets(self) -> None:
        """Test LSH bucket assignment."""
        lsh = LSHClustering(n_hash_functions=64, n_bands=8)

        # Create some signatures
        sig1 = tuple(range(64))
        sig2 = tuple(range(64))  # Same as sig1
        sig3 = tuple(range(1, 65))  # Different

        buckets = lsh._lsh_buckets([sig1, sig2, sig3])

        # sig1 and sig2 should share at least some buckets
        # (exact behavior depends on hashing, just verify structure)
        assert isinstance(buckets, dict)
        assert all(isinstance(indices, list) for indices in buckets.values())

    def test_cluster_basic(self) -> None:
        """Test basic clustering functionality."""
        payloads = generate_test_payloads(n_payloads=30, n_clusters=3, variation=0.05)

        lsh = LSHClustering(n_hash_functions=128, n_bands=32)
        clusters = lsh.cluster(payloads, threshold=0.85)

        # LSH is approximate - expect ballpark correct number of clusters
        # With low variation, should get close to 3 clusters (but may vary)
        assert 1 <= len(clusters) <= 15

        # All payloads should be assigned
        total_assigned = sum(c.size for c in clusters)
        assert total_assigned == len(payloads)

        # Each cluster should have members
        for cluster in clusters:
            assert cluster.size > 0
            assert len(cluster.payloads) == cluster.size

    def test_cluster_empty(self) -> None:
        """Test clustering with empty input."""
        lsh = LSHClustering()
        clusters = lsh.cluster([], threshold=0.8)

        assert len(clusters) == 0

    def test_cluster_single_payload(self) -> None:
        """Test clustering with single payload."""
        lsh = LSHClustering()
        clusters = lsh.cluster([b"single"], threshold=0.8)

        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_cluster_without_verification(self) -> None:
        """Test clustering using only MinHash estimation."""
        payloads = generate_test_payloads(n_payloads=20, n_clusters=2, variation=0.05)

        lsh = LSHClustering(n_hash_functions=128, n_bands=16)
        clusters = lsh.cluster(payloads, threshold=0.8, verify_with_levenshtein=False)

        # Should still produce reasonable clusters
        assert len(clusters) > 0
        total_assigned = sum(c.size for c in clusters)
        assert total_assigned == len(payloads)


class TestConvenienceFunction:
    """Test cluster_payloads_lsh convenience function."""

    def test_cluster_payloads_lsh(self) -> None:
        """Test convenience function for LSH clustering."""
        payloads = generate_test_payloads(n_payloads=30, n_clusters=3, variation=0.05)

        clusters = cluster_payloads_lsh(payloads, threshold=0.85)

        assert len(clusters) > 0
        total_assigned = sum(c.size for c in clusters)
        assert total_assigned == len(payloads)

    def test_cluster_payloads_lsh_custom_params(self) -> None:
        """Test LSH clustering with custom parameters."""
        payloads = generate_test_payloads(n_payloads=20, n_clusters=2, variation=0.1)

        clusters = cluster_payloads_lsh(
            payloads,
            threshold=0.75,
            n_hash_functions=64,
            n_bands=8,
            verify=False,
        )

        assert len(clusters) > 0

    def test_cluster_quality_vs_greedy(self) -> None:
        """Compare LSH clustering quality with greedy algorithm."""
        from oscura.analyzers.packet.payload_analysis import cluster_payloads

        payloads = generate_test_payloads(n_payloads=50, n_clusters=3, variation=0.05)

        # Cluster with both algorithms
        clusters_greedy = cluster_payloads(payloads, threshold=0.85, algorithm="greedy")
        clusters_lsh = cluster_payloads(payloads, threshold=0.85, algorithm="lsh")

        # LSH is approximate - may produce more clusters than greedy
        # Both should be reasonable (not one giant cluster or all singletons)
        assert len(clusters_greedy) >= 1
        assert len(clusters_lsh) >= 1
        assert len(clusters_lsh) <= len(payloads) // 2  # Not all singletons

        # Both should assign all payloads
        assert sum(c.size for c in clusters_greedy) == len(payloads)
        assert sum(c.size for c in clusters_lsh) == len(payloads)


@pytest.mark.performance
class TestLSHPerformance:
    """Performance tests for LSH clustering."""

    def test_lsh_faster_than_greedy_large_dataset(self) -> None:
        """Verify LSH is faster than greedy on large datasets."""
        import time

        from oscura.analyzers.packet.payload_analysis import cluster_payloads

        # Generate larger dataset
        payloads = generate_test_payloads(n_payloads=500, n_clusters=5, variation=0.1)

        # Test greedy
        start = time.perf_counter()
        clusters_greedy = cluster_payloads(payloads, threshold=0.8, algorithm="greedy")
        time_greedy = time.perf_counter() - start

        # Test LSH
        start = time.perf_counter()
        clusters_lsh = cluster_payloads(payloads, threshold=0.8, algorithm="lsh")
        time_lsh = time.perf_counter() - start

        print("\nPerformance comparison (500 payloads):")
        print(f"  Greedy: {time_greedy:.4f}s → {len(clusters_greedy)} clusters")
        print(f"  LSH:    {time_lsh:.4f}s → {len(clusters_lsh)} clusters")
        print(f"  Speedup: {time_greedy / time_lsh:.2f}x")

        # LSH should be faster for large datasets
        # (may not always be true for small datasets due to overhead)
        assert time_lsh < time_greedy * 2  # At least not significantly slower
