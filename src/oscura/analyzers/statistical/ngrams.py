"""N-gram frequency analysis for protocol fingerprinting.


This module provides tools for analyzing n-gram (byte sequence) frequencies
in binary data, useful for pattern identification, data characterization,
and protocol fingerprinting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for input data
DataType = Union[bytes, bytearray, "NDArray[Any]"]


@dataclass
class NgramProfile:
    """N-gram frequency profile.

    Attributes:
        n: N-gram size (number of bytes)
        frequencies: Dictionary mapping n-grams to their counts
        total_ngrams: Total number of n-grams extracted
        unique_ngrams: Number of unique n-grams found
        top_k: List of top n-grams with (ngram, count, frequency)
        entropy: Shannon entropy of n-gram distribution
    """

    n: int
    frequencies: dict[bytes, int]
    total_ngrams: int
    unique_ngrams: int
    top_k: list[tuple[bytes, int, float]]  # (ngram, count, frequency)
    entropy: float


@dataclass
class NgramComparison:
    """Comparison of two n-gram profiles.

    Attributes:
        similarity: Jaccard similarity coefficient (0-1, 1 = identical)
        cosine_similarity: Cosine similarity of frequency vectors (0-1)
        chi_square: Chi-square distance between distributions
        common_ngrams: Number of n-grams present in both profiles
        unique_to_a: Number of n-grams unique to first profile
        unique_to_b: Number of n-grams unique to second profile
    """

    similarity: float
    cosine_similarity: float
    chi_square: float
    common_ngrams: int
    unique_to_a: int
    unique_to_b: int


def ngram_frequency(data: DataType, n: int = 2, overlap: bool = True) -> NgramProfile:
    """Compute n-gram frequencies.

    : N-gram Frequency Analysis

    Extracts all n-grams from the data and computes their frequency distribution.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        n: N-gram size in bytes (default: 2 for bigrams)
        overlap: If True, use overlapping n-grams; if False, non-overlapping (default: True)

    Returns:
        NgramProfile containing frequency statistics

    Raises:
        ValueError: If n < 1

    Example:
        >>> profile = ngram_frequency(b'ABCABC', n=2)
        >>> profile.n
        2
        >>> profile.total_ngrams
        5
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if n < 1:
        raise ValueError(f"N-gram size must be >= 1, got {n}")

    # Handle data shorter than n - return empty profile
    if len(data) < n:
        return NgramProfile(
            n=n,
            frequencies={},
            total_ngrams=0,
            unique_ngrams=0,
            top_k=[],
            entropy=0.0,
        )

    # Extract n-grams
    step = 1 if overlap else n
    ngrams = []

    for i in range(0, len(data) - n + 1, step):
        ngrams.append(bytes(data[i : i + n]))

    # Count frequencies
    freq_counter = Counter(ngrams)
    total = len(ngrams)
    unique = len(freq_counter)

    # Get top k n-grams (sorted by count, then by bytes for consistency)
    top_k = [
        (ngram, count, count / total)
        for ngram, count in sorted(
            freq_counter.items(),
            key=lambda x: (-x[1], x[0]),  # Sort by count desc, then ngram asc
        )[:100]  # Limit to top 100
    ]

    # Calculate n-gram entropy
    entropy_val = 0.0
    for count in freq_counter.values():
        prob = count / total
        entropy_val -= prob * np.log2(prob)

    return NgramProfile(
        n=n,
        frequencies=dict(freq_counter),
        total_ngrams=total,
        unique_ngrams=unique,
        top_k=top_k,
        entropy=float(entropy_val),
    )


def ngram_entropy(data: DataType, n: int = 2) -> float:
    """Calculate entropy over n-gram distribution.

    : N-gram Frequency Analysis

    Computes Shannon entropy of the n-gram distribution, which measures
    the predictability of n-gram sequences.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        n: N-gram size in bytes (default: 2)

    Returns:
        N-gram entropy in bits (0.0 if data is too short)

    Example:
        >>> entropy = ngram_entropy(b'AAAA', n=2)
        >>> entropy
        0.0
    """
    profile = ngram_frequency(data, n=n, overlap=True)
    return profile.entropy


def compare_ngram_profiles(data_a: DataType, data_b: DataType, n: int = 2) -> NgramComparison:
    """Compare n-gram profiles between two datasets.

    : N-gram Frequency Analysis

    Computes multiple similarity metrics between two n-gram distributions
    for protocol fingerprinting and classification.

    Args:
        data_a: First dataset
        data_b: Second dataset
        n: N-gram size in bytes (default: 2)

    Returns:
        NgramComparison with similarity metrics

    Example:
        >>> comp = compare_ngram_profiles(b'ABCABC', b'ABCABC', n=2)
        >>> comp.similarity
        1.0
    """
    # Generate profiles
    profile_a = ngram_frequency(data_a, n=n, overlap=True)
    profile_b = ngram_frequency(data_b, n=n, overlap=True)

    freq_a = profile_a.frequencies
    freq_b = profile_b.frequencies

    # Get all unique n-grams
    all_ngrams = set(freq_a.keys()) | set(freq_b.keys())
    common = set(freq_a.keys()) & set(freq_b.keys())

    # Jaccard similarity (set-based)
    jaccard = len(common) / len(all_ngrams) if all_ngrams else 1.0

    # Cosine similarity (frequency-based)
    if all_ngrams:
        vec_a = np.array([freq_a.get(ng, 0) for ng in all_ngrams])
        vec_b = np.array([freq_b.get(ng, 0) for ng in all_ngrams])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a > 0 and norm_b > 0:
            cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        else:
            cosine_sim = 1.0 if norm_a == norm_b else 0.0
    else:
        cosine_sim = 1.0

    # Chi-square distance
    chi_square_val = 0.0
    if all_ngrams:
        total_a = profile_a.total_ngrams
        total_b = profile_b.total_ngrams

        for ngram in all_ngrams:
            freq_a_norm = freq_a.get(ngram, 0) / total_a if total_a > 0 else 0
            freq_b_norm = freq_b.get(ngram, 0) / total_b if total_b > 0 else 0
            expected = (freq_a_norm + freq_b_norm) / 2

            if expected > 0:
                chi_square_val += (
                    (freq_a_norm - expected) ** 2 + (freq_b_norm - expected) ** 2
                ) / expected

    return NgramComparison(
        similarity=float(jaccard),
        cosine_similarity=float(cosine_sim),
        chi_square=float(chi_square_val),
        common_ngrams=len(common),
        unique_to_a=len(freq_a) - len(common),
        unique_to_b=len(freq_b) - len(common),
    )


def find_unusual_ngrams(
    data: DataType, baseline: NgramProfile | None = None, n: int = 2, z_threshold: float = 3.0
) -> list[tuple[bytes, float]]:
    """Find unusually frequent or rare n-grams.

    : N-gram Frequency Analysis

    Identifies n-grams with frequencies that deviate significantly from
    expected (baseline) distributions using z-score analysis.

    Args:
        data: Input data to analyze
        baseline: Baseline n-gram profile for comparison (None = use uniform)
        n: N-gram size in bytes (default: 2)
        z_threshold: Z-score threshold for unusual classification (default: 3.0)

    Returns:
        List of (ngram, z_score) tuples for unusual n-grams, sorted by |z_score|

    Raises:
        ValueError: If baseline n-gram size doesn't match requested size.

    Example:
        >>> unusual = find_unusual_ngrams(b'AAABBBCCC', n=1)
        >>> len(unusual) >= 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    # Generate profile for current data
    profile = ngram_frequency(data, n=n, overlap=True)

    if profile.total_ngrams == 0:
        return []

    if baseline is None:
        # Use uniform distribution as baseline
        expected_count = profile.total_ngrams / (256**n)
        baseline_freqs = {}  # Empty means all n-grams have same expected frequency
    else:
        if baseline.n != n:
            raise ValueError(f"Baseline n-gram size ({baseline.n}) != requested size ({n})")
        # Normalize baseline frequencies
        baseline_freqs = {
            ng: count / baseline.total_ngrams * profile.total_ngrams
            for ng, count in baseline.frequencies.items()
        }
        expected_count = profile.total_ngrams / (256**n)

    # Calculate z-scores
    unusual = []

    for ngram, observed in profile.frequencies.items():
        expected = baseline_freqs.get(ngram, expected_count)

        # Use Poisson approximation for count data
        if expected > 0:
            # Z-score = (observed - expected) / sqrt(expected)
            z_score = (observed - expected) / np.sqrt(expected)

            if abs(z_score) >= z_threshold:
                unusual.append((ngram, float(z_score)))

    # Sort by absolute z-score descending
    unusual.sort(key=lambda x: abs(x[1]), reverse=True)

    return unusual


def ngram_heatmap(data: DataType, n: int = 2) -> NDArray[np.float64]:
    """Generate n-gram co-occurrence heatmap.

    : N-gram Frequency Analysis

    Creates a heatmap matrix showing n-gram frequencies. For bigrams (n=2),
    this produces a 256x256 matrix where entry [i,j] is the count of bigram
    (byte_i, byte_j).

    Args:
        data: Input data as bytes, bytearray, or numpy array
        n: N-gram size in bytes (must be 2 for heatmap visualization)

    Returns:
        Numpy array of shape (256, 256) for bigrams, normalized to [0, 1]

    Raises:
        ValueError: If n != 2 (only bigrams supported for heatmap)

    Example:
        >>> heatmap = ngram_heatmap(b'ABAB', n=2)
        >>> heatmap.shape
        (256, 256)
    """
    if n != 2:
        raise ValueError(f"Heatmap only supported for bigrams (n=2), got n={n}")

    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    # Initialize 256x256 matrix
    heatmap = np.zeros((256, 256), dtype=np.float64)

    # Count bigrams
    for i in range(len(data) - 1):
        byte1 = data[i]
        byte2 = data[i + 1]
        heatmap[byte1, byte2] += 1

    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap


def extract_ngrams(data: DataType, n: int = 2, overlap: bool = True) -> list[bytes]:
    """Extract n-grams from data as a list.

    : N-gram Extraction

    Extracts all n-grams from the input data and returns them as a list,
    with or without overlapping.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        n: N-gram size in bytes (default: 2)
        overlap: If True, use overlapping n-grams; if False, non-overlapping (default: True)

    Returns:
        List of bytes objects, each representing an n-gram

    Raises:
        ValueError: If n < 1

    Example:
        >>> ngrams = extract_ngrams(b'ABCABC', n=2)
        >>> len(ngrams)
        5
        >>> ngrams[0]
        b'AB'
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if n < 1:
        raise ValueError(f"N-gram size must be >= 1, got {n}")

    # Extract n-grams
    step = 1 if overlap else n
    ngrams: list[bytes] = []

    for i in range(0, len(data) - n + 1, step):
        ngrams.append(bytes(data[i : i + n]))

    return ngrams


def ngram_frequencies(data: DataType, n: int = 2, overlap: bool = True) -> dict[bytes, int]:
    """Get n-gram frequency counts as a dictionary.

    : N-gram Frequency Analysis

    Extracts n-grams and counts their frequencies, returning a dictionary
    mapping each n-gram to its count.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        n: N-gram size in bytes (default: 2)
        overlap: If True, use overlapping n-grams; if False, non-overlapping (default: True)

    Returns:
        Dictionary mapping n-grams (bytes) to their counts (int)

    Raises:
        ValueError: If n < 1

    Example:
        >>> freqs = ngram_frequencies(b'ABCABC', n=2)
        >>> freqs[b'AB']
        2
        >>> freqs[b'BC']
        2
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if n < 1:
        raise ValueError(f"N-gram size must be >= 1, got {n}")

    # Extract and count
    ngrams = extract_ngrams(data, n=n, overlap=overlap)
    return dict(Counter(ngrams))


def find_common_ngrams(data_a: DataType, data_b: DataType, n: int = 2) -> set[bytes]:
    """Find n-grams that appear in both datasets.

    : N-gram Analysis

    Identifies n-grams that occur in both input datasets.

    Args:
        data_a: First dataset
        data_b: Second dataset
        n: N-gram size in bytes (default: 2)

    Returns:
        Set of n-grams present in both datasets

    Example:
        >>> common = find_common_ngrams(b'ABCABC', b'BCABC', n=2)
        >>> b'BC' in common
        True
    """
    freqs_a = ngram_frequencies(data_a, n=n, overlap=True)
    freqs_b = ngram_frequencies(data_b, n=n, overlap=True)

    return set(freqs_a.keys()) & set(freqs_b.keys())


class NGramAnalyzer:
    """Object-oriented wrapper for n-gram analysis.

    Provides a class-based interface for n-gram analysis operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> analyzer = NGramAnalyzer(n=2)
        >>> frequencies = analyzer.analyze(b'ABCABC')
        >>> frequencies[b'AB']  # Direct dict access
        2
    """

    def __init__(self, n: int = 2, overlap: bool = True):
        """Initialize n-gram analyzer.

        Args:
            n: N-gram size in bytes.
            overlap: Whether to use overlapping n-grams.
        """
        self.n = n
        self.overlap = overlap
        self._last_profile: NgramProfile | None = None

    def analyze(self, data: DataType) -> dict[bytes, int]:
        """Analyze n-gram frequencies in data.

        Returns a dictionary mapping n-grams to counts for direct access.
        Returns empty dict if data is shorter than n.

        Args:
            data: Input data as bytes, bytearray, or numpy array.

        Returns:
            Dictionary mapping n-grams (bytes) to their counts (int).

        Example:
            >>> analyzer = NGramAnalyzer(n=2)
            >>> frequencies = analyzer.analyze(b'ABCABC')
            >>> frequencies[b'AB']
            2
        """
        self._last_profile = ngram_frequency(data, n=self.n, overlap=self.overlap)
        return self._last_profile.frequencies

    def analyze_profile(self, data: DataType) -> NgramProfile:
        """Analyze n-gram frequencies and return full profile.

        Args:
            data: Input data as bytes, bytearray, or numpy array.

        Returns:
            NgramProfile with full frequency statistics.

        Example:
            >>> analyzer = NGramAnalyzer(n=2)
            >>> profile = analyzer.analyze_profile(b'ABCDEF')
            >>> profile.n == 2
            True
        """
        self._last_profile = ngram_frequency(data, n=self.n, overlap=self.overlap)
        return self._last_profile

    def get_distribution(self, frequencies: dict[bytes, int]) -> dict[bytes, float]:
        """Convert frequency counts to normalized distribution.

        Args:
            frequencies: Dictionary mapping n-grams to counts.

        Returns:
            Dictionary mapping n-grams to normalized frequencies (0-1).

        Example:
            >>> analyzer = NGramAnalyzer(n=2)
            >>> freqs = analyzer.analyze(b'ABAB')
            >>> dist = analyzer.get_distribution(freqs)
            >>> sum(dist.values())  # Should sum to 1.0
            1.0
        """
        total = sum(frequencies.values())
        if total == 0:
            return {}
        return {ngram: count / total for ngram, count in frequencies.items()}

    def entropy(self, data: DataType) -> float:
        """Calculate n-gram entropy.

        Args:
            data: Input data.

        Returns:
            Entropy in bits.
        """
        return ngram_entropy(data, n=self.n)

    def compare(self, data_a: DataType, data_b: DataType) -> NgramComparison:
        """Compare n-gram profiles of two datasets.

        Args:
            data_a: First dataset.
            data_b: Second dataset.

        Returns:
            NgramComparison with similarity metrics.
        """
        return compare_ngram_profiles(data_a, data_b, n=self.n)

    def find_unusual(
        self, data: DataType, baseline: NgramProfile | None = None, z_threshold: float = 3.0
    ) -> list[tuple[bytes, float]]:
        """Find unusual n-grams in data.

        Args:
            data: Input data.
            baseline: Baseline profile for comparison.
            z_threshold: Z-score threshold.

        Returns:
            List of (ngram, z_score) tuples.
        """
        return find_unusual_ngrams(data, baseline=baseline, n=self.n, z_threshold=z_threshold)

    def heatmap(self, data: DataType) -> NDArray[np.float64]:
        """Generate bigram heatmap.

        Args:
            data: Input data.

        Returns:
            256x256 heatmap array.
        """
        return ngram_heatmap(data, n=self.n)


__all__ = [
    "DataType",
    "NGramAnalyzer",
    "NgramComparison",
    "NgramProfile",
    "compare_ngram_profiles",
    "extract_ngrams",
    "find_common_ngrams",
    "find_unusual_ngrams",
    "ngram_entropy",
    "ngram_frequencies",
    "ngram_frequency",
    "ngram_heatmap",
]
