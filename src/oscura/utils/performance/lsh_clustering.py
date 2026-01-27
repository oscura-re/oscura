"""Locality-Sensitive Hashing for fast payload clustering.

This module implements LSH-based clustering to reduce payload clustering
from O(n²) to O(n log n) complexity. Uses length-based bucketing and
sampling for byte sequences.

Strategy:
1. Bucket payloads by length (exact edit distance requires similar lengths)
2. Within each bucket, use sample-based hashing for candidate generation
3. Verify candidates with full comparison

References:
    - Indyk, P. & Motwani, R. (1998). "Approximate Nearest Neighbors"
    - Leskovec et al. (2014). "Mining of Massive Datasets"
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oscura.analyzers.packet.payload_analysis import PayloadCluster

__all__ = [
    "LSHClustering",
    "cluster_payloads_lsh",
]


class LSHClustering:
    """Locality-Sensitive Hashing for fast payload clustering.

    Uses MinHash with shingles to quickly identify candidate similar payloads,
    reducing comparisons from O(n²) to O(n log n).

    Example:
        >>> lsh = LSHClustering(n_hash_functions=128, n_bands=16)
        >>> clusters = lsh.cluster(payloads, threshold=0.8)
        >>> print(f"Found {len(clusters)} clusters")
    """

    def __init__(
        self,
        n_hash_functions: int = 128,
        n_bands: int = 8,
        shingle_size: int = 4,
        use_byte_shingles: bool = True,
    ) -> None:
        """Initialize LSH clustering.

        Args:
            n_hash_functions: Number of hash functions for MinHash (more = better accuracy).
            n_bands: Number of bands for LSH banding (fewer = more candidates, better recall).
            shingle_size: Size of shingles in bytes.
            use_byte_shingles: Use byte-level shingles instead of bit-level.
        """
        self.n_hash_functions = n_hash_functions
        self.n_bands = n_bands
        self.rows_per_band = n_hash_functions // n_bands
        self.shingle_size = shingle_size
        self.use_byte_shingles = use_byte_shingles

        # Pre-generate hash seeds for MinHash
        self._hash_seeds = self._generate_hash_seeds(n_hash_functions)

    def _generate_hash_seeds(self, n: int) -> list[int]:
        """Generate deterministic hash seeds.

        Args:
            n: Number of seeds to generate.

        Returns:
            List of integer seeds.
        """
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        return [int(rng.integers(1, 2**31 - 1)) for _ in range(n)]

    def _shingle(self, payload: bytes) -> set[bytes]:
        """Convert payload to set of shingles.

        Args:
            payload: Payload bytes to shingle.

        Returns:
            Set of shingles (byte sequences).
        """
        if len(payload) < self.shingle_size:
            # For very short payloads, use the whole payload
            return {payload}

        shingles = set()
        for i in range(len(payload) - self.shingle_size + 1):
            shingle = payload[i : i + self.shingle_size]
            shingles.add(shingle)

        return shingles

    def _minhash_signature(self, shingles: set[bytes]) -> tuple[int, ...]:
        """Compute MinHash signature for a set of shingles.

        Args:
            shingles: Set of shingles from payload.

        Returns:
            Tuple of MinHash signature values.
        """
        if not shingles:
            # Empty payload
            return tuple([0] * self.n_hash_functions)

        signature = []

        for seed in self._hash_seeds:
            min_hash = 2**32 - 1  # Max 32-bit value

            for shingle in shingles:
                # Hash shingle with this seed
                h = hashlib.sha256(shingle + seed.to_bytes(4, "big")).digest()
                hash_val = int.from_bytes(h[:4], "big")
                min_hash = min(min_hash, hash_val)

            signature.append(min_hash)

        return tuple(signature)

    def _lsh_buckets(self, signatures: list[tuple[int, ...]]) -> dict[int, list[int]]:
        """Assign signatures to LSH buckets using banding.

        Args:
            signatures: List of MinHash signatures.

        Returns:
            Dictionary mapping bucket IDs to payload indices.
        """
        buckets: dict[int, list[int]] = defaultdict(list)

        for idx, sig in enumerate(signatures):
            # Split signature into bands
            for band_idx in range(self.n_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band = sig[start:end]

                # Hash the band to get bucket ID
                band_bytes = b"".join(v.to_bytes(4, "big") for v in band)
                bucket_id = int.from_bytes(
                    hashlib.sha256(band_bytes + band_idx.to_bytes(4, "big")).digest()[:4],
                    "big",
                )

                buckets[bucket_id].append(idx)

        return buckets

    def _estimate_similarity(self, sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
        """Estimate Jaccard similarity from MinHash signatures.

        Args:
            sig_a: MinHash signature of first payload.
            sig_b: MinHash signature of second payload.

        Returns:
            Estimated Jaccard similarity (0-1).
        """
        matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
        return matches / len(sig_a)

    def cluster(
        self,
        payloads: Sequence[bytes],
        threshold: float = 0.8,
        verify_with_levenshtein: bool = True,
    ) -> list[PayloadCluster]:
        """Cluster payloads using LSH.

        Args:
            payloads: List of payloads to cluster.
            threshold: Similarity threshold for clustering.
            verify_with_levenshtein: Verify LSH candidates with full comparison.

        Returns:
            List of PayloadCluster objects.
        """

        if not payloads:
            return []

        # Compute signatures and find candidate pairs
        signatures = self._compute_all_signatures(payloads)
        candidate_pairs = self._find_candidate_pairs(signatures)

        # Build similarity graph
        graph = self._build_similarity_graph(
            payloads, signatures, candidate_pairs, threshold, verify_with_levenshtein
        )

        # Find connected components
        return self._extract_clusters(payloads, graph)

    def _compute_all_signatures(self, payloads: Sequence[bytes]) -> list[tuple[int, ...]]:
        """Compute MinHash signatures for all payloads."""
        signatures = []
        for payload in payloads:
            shingles = self._shingle(payload)
            sig = self._minhash_signature(shingles)
            signatures.append(sig)
        return signatures

    def _find_candidate_pairs(self, signatures: list[tuple[int, ...]]) -> set[tuple[int, int]]:
        """Find candidate pairs from LSH buckets."""
        buckets = self._lsh_buckets(signatures)
        candidate_pairs: set[tuple[int, int]] = set()

        for bucket_indices in buckets.values():
            for i in range(len(bucket_indices)):
                for j in range(i + 1, len(bucket_indices)):
                    idx_a = bucket_indices[i]
                    idx_b = bucket_indices[j]
                    pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                    candidate_pairs.add(pair)

        return candidate_pairs

    def _build_similarity_graph(
        self,
        payloads: Sequence[bytes],
        signatures: list[tuple[int, ...]],
        candidate_pairs: set[tuple[int, int]],
        threshold: float,
        verify: bool,
    ) -> dict[int, set[int]]:
        """Build adjacency graph from verified candidate pairs."""
        from oscura.analyzers.packet.payload_analysis import compute_similarity

        graph: dict[int, set[int]] = {i: set() for i in range(len(payloads))}

        for i, j in candidate_pairs:
            if verify:
                similarity = compute_similarity(payloads[i], payloads[j])
            else:
                similarity = self._estimate_similarity(signatures[i], signatures[j])

            if similarity >= threshold:
                graph[i].add(j)
                graph[j].add(i)

        return graph

    def _extract_clusters(
        self, payloads: Sequence[bytes], graph: dict[int, set[int]]
    ) -> list[PayloadCluster]:
        """Extract clusters from similarity graph using BFS."""

        from oscura.analyzers.packet.payload_analysis import PayloadCluster

        clusters: list[PayloadCluster] = []
        visited: set[int] = set()

        for start in range(len(payloads)):
            if start in visited:
                continue

            component = self._bfs_component(graph, start, visited)

            cluster_payloads = [payloads[i] for i in component]
            clusters.append(
                PayloadCluster(
                    cluster_id=len(clusters),
                    payloads=cluster_payloads,
                    indices=component,
                    representative=payloads[component[0]],
                    size=len(component),
                )
            )

        return clusters

    def _bfs_component(
        self, graph: dict[int, set[int]], start: int, visited: set[int]
    ) -> list[int]:
        """Find connected component using BFS."""
        from collections import deque

        component = []
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component


def cluster_payloads_lsh(
    payloads: Sequence[bytes],
    threshold: float = 0.8,
    n_hash_functions: int = 128,
    n_bands: int = 8,  # Fewer bands = more candidates = better recall
    verify: bool = True,
) -> list[PayloadCluster]:
    """Cluster payloads using LSH for O(n log n) performance.

    Args:
        payloads: List of payloads to cluster.
        threshold: Similarity threshold for clustering.
        n_hash_functions: Number of hash functions for MinHash.
        n_bands: Number of bands for LSH banding (fewer = more lenient).
        verify: Verify LSH candidates with Levenshtein distance.

    Returns:
        List of PayloadCluster objects.

    Example:
        >>> clusters = cluster_payloads_lsh(payloads, threshold=0.85)
        >>> print(f"Found {len(clusters)} clusters in O(n log n) time")
    """
    lsh = LSHClustering(
        n_hash_functions=n_hash_functions,
        n_bands=n_bands,
    )
    return lsh.cluster(payloads, threshold=threshold, verify_with_levenshtein=verify)
