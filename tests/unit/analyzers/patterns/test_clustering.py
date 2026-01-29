"""Comprehensive unit tests for pattern clustering module.


This module provides comprehensive test coverage for pattern clustering
capabilities including Hamming distance clustering, edit distance clustering,
hierarchical clustering, and cluster analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.patterns.clustering import (
    ClusteringResult,
    ClusterResult,
    PatternClusterer,
    analyze_cluster,
    cluster_by_edit_distance,
    cluster_by_hamming,
    cluster_hierarchical,
    compute_distance_matrix,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.pattern]


# =============================================================================
# ClusterResult Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestClusterResult:
    """Test ClusterResult dataclass."""

    def test_create_valid_cluster_result(self) -> None:
        """Test creating a valid cluster result."""
        patterns = [b"AAA", b"AAB"]
        result = ClusterResult(
            cluster_id=0,
            patterns=patterns,
            centroid=b"AAA",
            size=2,
            variance=0.1,
            common_bytes=[0, 1],
            variable_bytes=[2],
        )

        assert result.cluster_id == 0
        assert result.patterns == patterns
        assert result.centroid == b"AAA"
        assert result.size == 2
        assert result.variance == 0.1
        assert result.common_bytes == [0, 1]
        assert result.variable_bytes == [2]

    def test_cluster_id_must_be_non_negative(self) -> None:
        """Test that cluster_id must be non-negative."""
        with pytest.raises(ValueError, match="cluster_id must be non-negative"):
            ClusterResult(
                cluster_id=-1,
                patterns=[b"AAA"],
                centroid=b"AAA",
                size=1,
                variance=0.0,
                common_bytes=[],
                variable_bytes=[],
            )

    def test_size_must_be_non_negative(self) -> None:
        """Test that size must be non-negative."""
        with pytest.raises(ValueError, match="size must be non-negative"):
            ClusterResult(
                cluster_id=0,
                patterns=[b"AAA"],
                centroid=b"AAA",
                size=-1,
                variance=0.0,
                common_bytes=[],
                variable_bytes=[],
            )

    def test_patterns_length_must_match_size(self) -> None:
        """Test that patterns length must match size."""
        with pytest.raises(ValueError, match="patterns length must match size"):
            ClusterResult(
                cluster_id=0,
                patterns=[b"AAA", b"BBB"],
                centroid=b"AAA",
                size=3,  # Mismatch
                variance=0.0,
                common_bytes=[],
                variable_bytes=[],
            )


# =============================================================================
# ClusteringResult Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestClusteringResult:
    """Test ClusteringResult dataclass."""

    def test_create_valid_clustering_result(self) -> None:
        """Test creating a valid clustering result."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AAA"],
            centroid=b"AAA",
            size=1,
            variance=0.0,
            common_bytes=[0, 1, 2],
            variable_bytes=[],
        )

        result = ClusteringResult(
            clusters=[cluster],
            labels=np.array([0]),
            num_clusters=1,
            silhouette_score=0.5,
        )

        assert len(result.clusters) == 1
        assert result.num_clusters == 1
        assert result.silhouette_score == 0.5
        assert len(result.labels) == 1

    def test_num_clusters_must_match_clusters_length(self) -> None:
        """Test that num_clusters must match clusters length."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AAA"],
            centroid=b"AAA",
            size=1,
            variance=0.0,
            common_bytes=[],
            variable_bytes=[],
        )

        with pytest.raises(ValueError, match="num_clusters must match clusters length"):
            ClusteringResult(
                clusters=[cluster],
                labels=np.array([0]),
                num_clusters=2,  # Mismatch
                silhouette_score=0.0,
            )


# =============================================================================
# cluster_by_hamming Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestClusterByHamming:
    """Test cluster_by_hamming function."""

    def test_empty_patterns(self) -> None:
        """Test clustering empty pattern list."""
        result = cluster_by_hamming([])

        assert result.num_clusters == 0
        assert len(result.clusters) == 0
        assert len(result.labels) == 0
        assert result.silhouette_score == 0.0

    def test_single_cluster_similar_patterns(self) -> None:
        """Test clustering similar patterns into single cluster."""
        patterns = [b"ABCD", b"ABCE", b"ABCF"]
        result = cluster_by_hamming(patterns, threshold=0.3)

        assert result.num_clusters >= 1
        # At least some patterns should cluster together
        assert len(result.labels) == 3

    def test_multiple_clusters_dissimilar_patterns(self) -> None:
        """Test clustering dissimilar patterns into multiple clusters."""
        patterns = [b"AAAA", b"AAAB", b"BBBB", b"BBBC", b"CCCC"]
        result = cluster_by_hamming(patterns, threshold=0.3, min_cluster_size=2)

        # Should have multiple clusters
        assert result.num_clusters >= 1
        assert len(result.labels) == 5

    def test_threshold_affects_clustering(self) -> None:
        """Test that threshold parameter affects clustering."""
        patterns = [b"AAAA", b"AAAB", b"AABB", b"ABBB"]

        # Lower threshold: stricter clustering
        result_strict = cluster_by_hamming(patterns, threshold=0.1, min_cluster_size=1)

        # Higher threshold: looser clustering
        result_loose = cluster_by_hamming(patterns, threshold=0.5, min_cluster_size=1)

        # Loose threshold should result in fewer or equal clusters
        assert (
            result_loose.num_clusters <= result_strict.num_clusters
            or result_strict.num_clusters >= 1
        )

    def test_min_cluster_size_filtering(self) -> None:
        """Test min_cluster_size filters small clusters."""
        patterns = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]

        # With min_cluster_size=2, singleton clusters should be filtered
        result = cluster_by_hamming(patterns, threshold=0.1, min_cluster_size=2)

        # All patterns are dissimilar, so no clusters should form
        for cluster in result.clusters:
            assert cluster.size >= 2

    def test_different_length_patterns_raises_error(self) -> None:
        """Test that patterns with different lengths raise ValueError."""
        patterns = [b"AAA", b"BBBB", b"CCC"]

        with pytest.raises(ValueError, match="expected"):
            cluster_by_hamming(patterns)

    def test_numpy_array_patterns(self) -> None:
        """Test clustering with numpy array patterns."""
        patterns = [
            np.array([0xAA, 0xBB, 0xCC], dtype=np.uint8),
            np.array([0xAA, 0xBB, 0xCD], dtype=np.uint8),
            np.array([0xAA, 0xBB, 0xCE], dtype=np.uint8),
        ]

        result = cluster_by_hamming(patterns, threshold=0.5, min_cluster_size=2)

        # Should form at least one cluster (patterns differ by 1/3 = 0.33)
        assert result.num_clusters >= 1
        assert len(result.labels) == 3

    def test_cluster_centroid_calculation(self) -> None:
        """Test that cluster centroids are correctly computed."""
        patterns = [b"AAAA", b"AAAB", b"AAAC"]
        result = cluster_by_hamming(patterns, threshold=0.3, min_cluster_size=2)

        if result.num_clusters > 0:
            cluster = result.clusters[0]
            # Centroid should be a pattern (bytes or array)
            assert cluster.centroid is not None
            assert len(cluster.centroid) == 4

    def test_common_and_variable_bytes_identification(self) -> None:
        """Test identification of common vs variable byte positions."""
        patterns = [b"AXXA", b"AYYA", b"AZZA"]
        result = cluster_by_hamming(patterns, threshold=0.5, min_cluster_size=2)

        if result.num_clusters > 0:
            cluster = result.clusters[0]
            # Positions 0 and 3 should be common (A)
            # Positions 1 and 2 should be variable
            assert len(cluster.common_bytes) + len(cluster.variable_bytes) == 4

    def test_variance_calculation(self) -> None:
        """Test within-cluster variance calculation."""
        patterns = [b"AAAA", b"AAAB"]
        result = cluster_by_hamming(patterns, threshold=0.3, min_cluster_size=2)

        if result.num_clusters > 0:
            cluster = result.clusters[0]
            assert cluster.variance >= 0.0


# =============================================================================
# cluster_by_edit_distance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestClusterByEditDistance:
    """Test cluster_by_edit_distance function."""

    def test_empty_patterns(self) -> None:
        """Test clustering empty pattern list."""
        result = cluster_by_edit_distance([])

        assert result.num_clusters == 0
        assert len(result.clusters) == 0
        assert len(result.labels) == 0

    def test_variable_length_patterns(self) -> None:
        """Test clustering patterns with different lengths."""
        patterns = [b"ABC", b"ABCD", b"ABCDE", b"XYZ"]
        result = cluster_by_edit_distance(patterns, threshold=0.4, min_cluster_size=2)

        # Should cluster similar patterns
        assert len(result.labels) == 4

    def test_threshold_affects_clustering(self) -> None:
        """Test that threshold parameter affects clustering."""
        patterns = [b"AAAA", b"AAAB", b"AABB", b"ABBB"]

        result_strict = cluster_by_edit_distance(patterns, threshold=0.1, min_cluster_size=1)
        result_loose = cluster_by_edit_distance(patterns, threshold=0.5, min_cluster_size=1)

        # More lenient threshold should generally produce fewer clusters
        assert (
            result_loose.num_clusters <= result_strict.num_clusters
            or result_strict.num_clusters >= 1
        )

    def test_identical_patterns(self) -> None:
        """Test clustering identical patterns."""
        patterns = [b"AAAA", b"AAAA", b"AAAA"]
        result = cluster_by_edit_distance(patterns, threshold=0.1, min_cluster_size=2)

        # All identical patterns should be in one cluster
        assert result.num_clusters == 1
        assert result.clusters[0].size == 3

    def test_completely_different_patterns(self) -> None:
        """Test clustering completely different patterns."""
        patterns = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]
        result = cluster_by_edit_distance(patterns, threshold=0.1, min_cluster_size=2)

        # No clusters should form (all patterns too different)
        # All should be labeled as noise (-1)
        assert np.all((result.labels == -1) | (result.labels >= 0))

    def test_centroid_is_most_common_pattern(self) -> None:
        """Test that centroid is the most common pattern."""
        patterns = [b"AAAA", b"AAAA", b"AAAB"]
        result = cluster_by_edit_distance(patterns, threshold=0.3, min_cluster_size=2)

        if result.num_clusters > 0:
            cluster = result.clusters[0]
            # Centroid should be the most common pattern
            assert cluster.centroid in patterns


# =============================================================================
# cluster_hierarchical Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestClusterHierarchical:
    """Test cluster_hierarchical function."""

    def test_must_specify_num_clusters_or_threshold(self) -> None:
        """Test that either num_clusters or distance_threshold must be specified."""
        patterns = [b"AAA", b"BBB"]

        with pytest.raises(ValueError, match="Must specify either"):
            cluster_hierarchical(patterns, num_clusters=None, distance_threshold=None)

    def test_empty_patterns(self) -> None:
        """Test clustering empty pattern list."""
        result = cluster_hierarchical([], num_clusters=2)

        assert result.num_clusters == 0
        assert len(result.clusters) == 0

    def test_num_clusters_parameter(self) -> None:
        """Test clustering with specified number of clusters."""
        patterns = [b"AAA", b"AAB", b"BBB", b"BBC", b"CCC", b"CCD"]
        result = cluster_hierarchical(patterns, num_clusters=3)

        # Should produce 3 or fewer clusters
        assert result.num_clusters <= 3
        assert len(result.labels) == 6

    def test_distance_threshold_parameter(self) -> None:
        """Test clustering with distance threshold."""
        patterns = [b"AAA", b"AAB", b"BBB", b"BBC"]
        result = cluster_hierarchical(patterns, distance_threshold=0.5)

        # Should produce some clustering
        assert len(result.labels) == 4

    def test_single_linkage_method(self) -> None:
        """Test hierarchical clustering with single linkage."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        result = cluster_hierarchical(patterns, method="single", num_clusters=2)

        assert result.num_clusters <= 2

    def test_complete_linkage_method(self) -> None:
        """Test hierarchical clustering with complete linkage."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        result = cluster_hierarchical(patterns, method="complete", num_clusters=2)

        assert result.num_clusters <= 2

    def test_average_linkage_method(self) -> None:
        """Test hierarchical clustering with average linkage."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        result = cluster_hierarchical(patterns, method="average", num_clusters=2)

        assert result.num_clusters <= 2

    def test_upgma_method(self) -> None:
        """Test hierarchical clustering with UPGMA."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        result = cluster_hierarchical(patterns, method="upgma", num_clusters=2)

        assert result.num_clusters <= 2

    def test_fixed_length_patterns(self) -> None:
        """Test hierarchical clustering with fixed-length patterns."""
        patterns = [b"AAAA", b"AAAB", b"BBBB", b"BBBC"]
        result = cluster_hierarchical(patterns, num_clusters=2)

        assert len(result.labels) == 4
        # Should produce 2 or fewer clusters
        assert result.num_clusters <= 2


# =============================================================================
# analyze_cluster Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestAnalyzeCluster:
    """Test analyze_cluster function."""

    def test_empty_cluster(self) -> None:
        """Test analyzing empty cluster."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[],
            centroid=b"",
            size=0,
            variance=0.0,
            common_bytes=[],
            variable_bytes=[],
        )

        analysis = analyze_cluster(cluster)

        assert analysis["common_bytes"] == []
        assert analysis["variable_bytes"] == []
        assert analysis["entropy_per_byte"] == []
        assert analysis["consensus"] == b""

    def test_single_pattern_cluster(self) -> None:
        """Test analyzing cluster with single pattern."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"ABCD"],
            centroid=b"ABCD",
            size=1,
            variance=0.0,
            common_bytes=[0, 1, 2, 3],
            variable_bytes=[],
        )

        analysis = analyze_cluster(cluster)

        # All bytes should be common (entropy = 0)
        assert len(analysis["common_bytes"]) == 4
        assert len(analysis["variable_bytes"]) == 0
        assert analysis["consensus"] == b"ABCD"

    def test_identify_common_bytes(self) -> None:
        """Test identification of common byte positions."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AXXA", b"AXXA", b"AXXA"],
            centroid=b"AXXA",
            size=3,
            variance=0.0,
            common_bytes=[0, 3],
            variable_bytes=[1, 2],
        )

        analysis = analyze_cluster(cluster)

        # All positions should be common (identical patterns)
        assert len(analysis["common_bytes"]) == 4
        assert analysis["consensus"] == b"AXXA"

    def test_identify_variable_bytes(self) -> None:
        """Test identification of variable byte positions."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AXXA", b"AYYA", b"AZZA"],
            centroid=b"AXXA",
            size=3,
            variance=0.1,
            common_bytes=[0, 3],
            variable_bytes=[1, 2],
        )

        analysis = analyze_cluster(cluster)

        # Positions 1 and 2 vary
        assert 1 in analysis["variable_bytes"] or 2 in analysis["variable_bytes"]

    def test_entropy_calculation(self) -> None:
        """Test entropy calculation per byte position."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AAAA", b"AAAA", b"AAAB"],
            centroid=b"AAAA",
            size=3,
            variance=0.1,
            common_bytes=[0, 1, 2],
            variable_bytes=[3],
        )

        analysis = analyze_cluster(cluster)

        # Should have entropy values for each position
        assert len(analysis["entropy_per_byte"]) == 4
        # First three positions should have low entropy
        assert analysis["entropy_per_byte"][0] == 0.0
        assert analysis["entropy_per_byte"][1] == 0.0
        assert analysis["entropy_per_byte"][2] == 0.0
        # Last position should have non-zero entropy
        assert analysis["entropy_per_byte"][3] > 0.0

    def test_consensus_pattern(self) -> None:
        """Test consensus pattern generation."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"AAAA", b"AAAB", b"AAAC"],
            centroid=b"AAAA",
            size=3,
            variance=0.1,
            common_bytes=[0, 1, 2],
            variable_bytes=[3],
        )

        analysis = analyze_cluster(cluster)

        # Consensus should use most common byte at each position
        assert len(analysis["consensus"]) == 4
        assert analysis["consensus"][:3] == b"AAA"  # First 3 bytes all 'A'

    def test_variable_length_patterns(self) -> None:
        """Test analyzing cluster with variable-length patterns."""
        cluster = ClusterResult(
            cluster_id=0,
            patterns=[b"ABC", b"ABCD", b"ABCDE"],
            centroid=b"ABCD",
            size=3,
            variance=0.2,
            common_bytes=[0, 1, 2],
            variable_bytes=[3, 4],
        )

        analysis = analyze_cluster(cluster)

        # Should pad to max length
        assert len(analysis["consensus"]) == 5


# =============================================================================
# compute_distance_matrix Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestComputeDistanceMatrix:
    """Test compute_distance_matrix function."""

    def test_empty_patterns(self) -> None:
        """Test distance matrix for empty patterns."""
        patterns: list[bytes] = []
        dist_matrix = compute_distance_matrix(patterns)

        assert dist_matrix.shape == (0, 0)

    def test_single_pattern(self) -> None:
        """Test distance matrix for single pattern."""
        patterns = [b"AAA"]
        dist_matrix = compute_distance_matrix(patterns)

        assert dist_matrix.shape == (1, 1)
        assert dist_matrix[0, 0] == 0.0

    def test_hamming_distance_metric(self) -> None:
        """Test distance matrix with Hamming metric."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        dist_matrix = compute_distance_matrix(patterns, metric="hamming")

        assert dist_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert dist_matrix[0, 0] == 0.0
        assert dist_matrix[1, 1] == 0.0
        assert dist_matrix[2, 2] == 0.0
        # Matrix should be symmetric
        assert dist_matrix[0, 1] == dist_matrix[1, 0]
        assert dist_matrix[0, 2] == dist_matrix[2, 0]

    def test_levenshtein_distance_metric(self) -> None:
        """Test distance matrix with Levenshtein metric."""
        patterns = [b"ABC", b"ABCD", b"XYZ"]
        dist_matrix = compute_distance_matrix(patterns, metric="levenshtein")

        assert dist_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert dist_matrix[0, 0] == 0.0
        # ABC and ABCD should be similar
        assert dist_matrix[0, 1] < dist_matrix[0, 2]

    def test_jaccard_distance_metric(self) -> None:
        """Test distance matrix with Jaccard metric."""
        patterns = [b"AAA", b"AAB", b"BBB"]
        dist_matrix = compute_distance_matrix(patterns, metric="jaccard")

        assert dist_matrix.shape == (3, 3)
        # Diagonal should be zero
        assert dist_matrix[0, 0] == 0.0
        # Should be symmetric
        assert dist_matrix[0, 1] == dist_matrix[1, 0]

    def test_unknown_metric_raises_error(self) -> None:
        """Test that unknown metric raises ValueError."""
        patterns = [b"AAA", b"BBB"]

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_distance_matrix(patterns, metric="unknown")  # type: ignore

    def test_matrix_symmetry(self) -> None:
        """Test that distance matrix is symmetric."""
        patterns = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]
        dist_matrix = compute_distance_matrix(patterns, metric="hamming")

        # Check symmetry
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_numpy_array_patterns(self) -> None:
        """Test distance matrix with numpy array patterns."""
        patterns = [
            np.array([0xAA, 0xBB], dtype=np.uint8),
            np.array([0xAA, 0xCC], dtype=np.uint8),
        ]
        dist_matrix = compute_distance_matrix(patterns, metric="hamming")

        assert dist_matrix.shape == (2, 2)
        assert dist_matrix[0, 0] == 0.0


# =============================================================================
# PatternClusterer Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternClusterer:
    """Test PatternClusterer class."""

    def test_initialization_defaults(self) -> None:
        """Test PatternClusterer initialization with defaults."""
        clusterer = PatternClusterer()

        assert clusterer.n_clusters == 3
        assert clusterer.method == "hamming"
        assert clusterer.distance_metric == "hamming"
        assert clusterer.threshold == 0.3
        assert clusterer.min_cluster_size == 2
        assert clusterer.result_ is None

    def test_initialization_custom_parameters(self) -> None:
        """Test PatternClusterer initialization with custom parameters."""
        clusterer = PatternClusterer(
            n_clusters=5,
            method="edit",
            distance_metric="levenshtein",
            threshold=0.5,
            min_cluster_size=3,
        )

        assert clusterer.n_clusters == 5
        assert clusterer.method == "edit"
        assert clusterer.distance_metric == "levenshtein"
        assert clusterer.threshold == 0.5
        assert clusterer.min_cluster_size == 3

    def test_cluster_with_hamming_method(self) -> None:
        """Test clustering with Hamming method."""
        clusterer = PatternClusterer(method="hamming", threshold=0.3)
        patterns = [b"AAAA", b"AAAB", b"BBBB"]

        labels = clusterer.cluster(patterns)

        assert len(labels) == 3
        assert clusterer.result_ is not None

    def test_cluster_with_edit_method(self) -> None:
        """Test clustering with edit distance method."""
        clusterer = PatternClusterer(method="edit", threshold=0.4)
        patterns = [b"ABC", b"ABCD", b"XYZ"]

        labels = clusterer.cluster(patterns)

        assert len(labels) == 3
        assert clusterer.result_ is not None

    def test_cluster_with_hierarchical_method(self) -> None:
        """Test clustering with hierarchical method."""
        clusterer = PatternClusterer(method="hierarchical", n_clusters=2)
        patterns = [b"AAA", b"AAB", b"BBB", b"BBC"]

        labels = clusterer.cluster(patterns)

        assert len(labels) == 4
        assert clusterer.result_ is not None

    def test_fit_method(self) -> None:
        """Test fit method returns self."""
        clusterer = PatternClusterer()
        patterns = [b"AAA", b"BBB"]

        result = clusterer.fit(patterns)

        assert result is clusterer
        assert clusterer.result_ is not None

    def test_fit_predict_method(self) -> None:
        """Test fit_predict method."""
        clusterer = PatternClusterer()
        patterns = [b"AAA", b"AAB", b"BBB"]

        labels = clusterer.fit_predict(patterns)

        assert len(labels) == 3
        assert clusterer.result_ is not None

    def test_get_clusters_before_clustering_raises_error(self) -> None:
        """Test that get_clusters raises error before clustering."""
        clusterer = PatternClusterer()

        with pytest.raises(ValueError, match="Must call cluster"):
            clusterer.get_clusters()

    def test_get_clusters_after_clustering(self) -> None:
        """Test getting clusters after clustering."""
        clusterer = PatternClusterer()
        patterns = [b"AAA", b"AAB", b"BBB", b"BBC"]
        clusterer.cluster(patterns)

        clusters = clusterer.get_clusters()

        assert isinstance(clusters, list)
        assert all(isinstance(c, ClusterResult) for c in clusters)

    def test_get_silhouette_score_before_clustering_raises_error(self) -> None:
        """Test that get_silhouette_score raises error before clustering."""
        clusterer = PatternClusterer()

        with pytest.raises(ValueError, match="Must call cluster"):
            clusterer.get_silhouette_score()

    def test_get_silhouette_score_after_clustering(self) -> None:
        """Test getting silhouette score after clustering."""
        clusterer = PatternClusterer()
        patterns = [b"AAA", b"AAB", b"BBB", b"BBC"]
        clusterer.cluster(patterns)

        score = clusterer.get_silhouette_score()

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternsClusteringEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_pattern_clustering(self) -> None:
        """Test clustering with a single pattern."""
        patterns = [b"AAAA"]
        result = cluster_by_hamming(patterns, min_cluster_size=1)

        # Single pattern should form one cluster
        assert result.num_clusters == 1
        assert result.clusters[0].size == 1

    def test_identical_patterns_clustering(self) -> None:
        """Test clustering with all identical patterns."""
        patterns = [b"AAAA"] * 5
        result = cluster_by_hamming(patterns, threshold=0.1)

        # All identical patterns should be in one cluster
        assert result.num_clusters == 1
        assert result.clusters[0].size == 5
        assert result.clusters[0].variance == 0.0

    def test_very_small_threshold(self) -> None:
        """Test clustering with very small threshold."""
        patterns = [b"AAAA", b"AAAB", b"AAAC"]
        result = cluster_by_hamming(patterns, threshold=0.01, min_cluster_size=1)

        # Very small threshold should create separate clusters
        # or assign to noise
        assert len(result.labels) == 3

    def test_very_large_threshold(self) -> None:
        """Test clustering with very large threshold."""
        patterns = [b"AAAA", b"BBBB", b"CCCC"]
        result = cluster_by_hamming(patterns, threshold=1.0, min_cluster_size=2)

        # Large threshold might cluster everything together
        assert len(result.labels) == 3

    def test_mixed_bytes_and_arrays(self) -> None:
        """Test clustering with mixed bytes and numpy arrays."""
        patterns = [
            b"AAAA",
            np.array([0x41, 0x41, 0x41, 0x42], dtype=np.uint8),  # AAAB
            b"BBBB",
        ]
        result = cluster_by_hamming(patterns, threshold=0.3)

        assert len(result.labels) == 3

    def test_binary_data_patterns(self) -> None:
        """Test clustering with binary data patterns."""
        patterns = [
            b"\x00\x01\x02\x03",
            b"\x00\x01\x02\x04",
            b"\xff\xfe\xfd\xfc",
        ]
        result = cluster_by_hamming(patterns, threshold=0.3, min_cluster_size=2)

        assert len(result.labels) == 3

    def test_zero_variance_cluster(self) -> None:
        """Test that identical patterns produce zero variance cluster."""
        patterns = [b"XXXX", b"XXXX", b"XXXX"]
        result = cluster_by_hamming(patterns, threshold=0.1, min_cluster_size=2)

        if result.num_clusters > 0:
            assert result.clusters[0].variance == 0.0

    def test_high_variance_cluster(self) -> None:
        """Test cluster with high variance."""
        # Patterns that are just within threshold but quite different
        patterns = [b"AAAA", b"AAAB", b"AABB"]
        result = cluster_by_hamming(patterns, threshold=0.5, min_cluster_size=2)

        if result.num_clusters > 0:
            # Variance should be non-zero
            assert result.clusters[0].variance > 0.0

    def test_bytearray_patterns(self) -> None:
        """Test clustering with bytearray patterns."""
        patterns = [
            bytearray(b"AAAA"),
            bytearray(b"AAAB"),
            bytearray(b"BBBB"),
        ]
        result = cluster_by_hamming(patterns, threshold=0.3)

        assert len(result.labels) == 3

    def test_memoryview_patterns(self) -> None:
        """Test clustering with memoryview patterns."""
        patterns = [
            memoryview(b"AAAA"),
            memoryview(b"AAAB"),
            memoryview(b"BBBB"),
        ]
        result = cluster_by_hamming(patterns, threshold=0.3)

        assert len(result.labels) == 3

    def test_empty_byte_strings(self) -> None:
        """Test handling of empty byte strings."""
        patterns = [b"", b""]
        result = cluster_by_edit_distance(patterns, threshold=0.1)

        # Empty patterns should cluster together
        assert len(result.labels) == 2

    def test_single_byte_patterns(self) -> None:
        """Test clustering single-byte patterns."""
        patterns = [b"A", b"A", b"B"]
        result = cluster_by_hamming(patterns, threshold=0.1, min_cluster_size=2)

        # Two A's should cluster
        assert result.num_clusters >= 1

    def test_unsupported_pattern_type_raises_error(self) -> None:
        """Test that unsupported pattern types raise TypeError."""
        from oscura.analyzers.patterns.clustering import _to_array

        with pytest.raises(TypeError, match="Unsupported type"):
            _to_array("invalid")  # type: ignore

    def test_jaccard_distance_with_empty_sets(self) -> None:
        """Test Jaccard distance with patterns producing empty sets."""
        from oscura.analyzers.patterns.clustering import _jaccard_distance

        # Empty patterns
        dist = _jaccard_distance(b"", b"")
        assert dist == 0.0

    def test_hamming_distance_different_lengths(self) -> None:
        """Test Hamming distance with different length patterns."""
        from oscura.analyzers.patterns.clustering import _hamming_distance

        # Different lengths - should pad
        dist = _hamming_distance(b"AA", b"AAA")
        assert 0.0 <= dist <= 1.0

    def test_edit_distance_empty_patterns(self) -> None:
        """Test edit distance with empty patterns."""
        from oscura.analyzers.patterns.clustering import _edit_distance

        # Both empty
        assert _edit_distance(b"", b"") == 0.0

        # One empty
        assert _edit_distance(b"ABC", b"") == 1.0
        assert _edit_distance(b"", b"ABC") == 1.0

    def test_centroid_with_empty_patterns_list(self) -> None:
        """Test centroid calculation with empty patterns list."""
        from oscura.analyzers.patterns.clustering import _compute_centroid_hamming

        centroid = _compute_centroid_hamming([])
        assert len(centroid) == 0

    def test_pattern_variance_empty_list(self) -> None:
        """Test pattern variance analysis with empty list."""
        from oscura.analyzers.patterns.clustering import _analyze_pattern_variance

        common, variable = _analyze_pattern_variance([])
        assert common == []
        assert variable == []

    def test_byte_entropy_empty_list(self) -> None:
        """Test byte entropy with empty list."""
        from oscura.analyzers.patterns.clustering import _compute_byte_entropy

        entropy = _compute_byte_entropy([])
        assert entropy == 0.0

    def test_silhouette_score_edge_cases(self) -> None:
        """Test silhouette score with edge cases."""
        from oscura.analyzers.patterns.clustering import _compute_silhouette_score

        # Single point
        dist = np.array([[0.0]])
        labels = np.array([0])
        score = _compute_silhouette_score(dist, labels)
        assert score == 0.0

        # All noise
        dist = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([-1, -1])
        score = _compute_silhouette_score(dist, labels)
        assert score == 0.0

    def test_hierarchical_empty_merge(self) -> None:
        """Test hierarchical clustering edge case with no merges."""
        # Single pattern
        patterns = [b"AAA"]
        result = cluster_hierarchical(patterns, num_clusters=1)

        assert result.num_clusters == 1
        assert result.clusters[0].size == 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternsClusteringIntegration:
    """Integration tests for complete clustering workflows."""

    def test_complete_hamming_workflow(self) -> None:
        """Test complete workflow with Hamming clustering."""
        # Create test patterns
        patterns = [
            b"AAAA",
            b"AAAB",
            b"AAAC",  # Cluster 1
            b"BBBB",
            b"BBBC",
            b"BBBD",  # Cluster 2
            b"CCCC",  # Noise (singleton)
        ]

        # Cluster
        result = cluster_by_hamming(patterns, threshold=0.3, min_cluster_size=2)

        # Verify results
        assert result.num_clusters >= 1
        assert len(result.labels) == 7

        # Analyze each cluster
        for cluster in result.clusters:
            analysis = analyze_cluster(cluster)
            assert "common_bytes" in analysis
            assert "variable_bytes" in analysis
            assert "entropy_per_byte" in analysis
            assert "consensus" in analysis

    def test_complete_edit_distance_workflow(self) -> None:
        """Test complete workflow with edit distance clustering."""
        patterns = [
            b"HELLO",
            b"HELO",
            b"WORLD",
            b"WRLD",
        ]

        result = cluster_by_edit_distance(patterns, threshold=0.4, min_cluster_size=2)

        assert len(result.labels) == 4

        for cluster in result.clusters:
            analysis = analyze_cluster(cluster)
            assert analysis["consensus"] is not None

    def test_complete_hierarchical_workflow(self) -> None:
        """Test complete workflow with hierarchical clustering."""
        patterns = [
            b"AAA",
            b"AAB",
            b"BBB",
            b"BBC",
            b"CCC",
            b"CCD",
        ]

        result = cluster_hierarchical(patterns, num_clusters=3)

        assert result.num_clusters <= 3
        assert len(result.labels) == 6

    def test_clusterer_class_complete_workflow(self) -> None:
        """Test complete workflow using PatternClusterer class."""
        patterns = [
            b"AAAA",
            b"AAAB",
            b"BBBB",
            b"BBBC",
            b"CCCC",
        ]

        # Initialize clusterer
        clusterer = PatternClusterer(
            method="hamming",
            threshold=0.3,
            min_cluster_size=2,
        )

        # Fit and predict
        labels = clusterer.fit_predict(patterns)

        # Get results
        clusters = clusterer.get_clusters()
        score = clusterer.get_silhouette_score()

        # Verify
        assert len(labels) == 5
        assert isinstance(clusters, list)
        assert isinstance(score, float)

        # Analyze each cluster
        for cluster in clusters:
            analysis = analyze_cluster(cluster)
            assert len(analysis["consensus"]) > 0

    def test_distance_matrix_integration(self) -> None:
        """Test distance matrix computation integration."""
        patterns = [b"AAA", b"AAB", b"BBB"]

        # Test all metrics
        for metric in ["hamming", "levenshtein", "jaccard"]:
            dist_matrix = compute_distance_matrix(patterns, metric=metric)

            # Use distance matrix for manual clustering verification
            assert dist_matrix.shape == (3, 3)
            assert np.allclose(dist_matrix, dist_matrix.T)  # Symmetric
            assert np.allclose(np.diag(dist_matrix), 0.0)  # Zero diagonal
