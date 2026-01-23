"""Automatic signature and delimiter discovery.

This module implements algorithms for automatically discovering candidate
signatures, headers, and delimiters in binary data through statistical analysis.


Author: Oscura Development Team
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CandidateSignature:
    """A candidate signature/header pattern.

    Attributes:
        pattern: The signature byte pattern
        length: Length of signature in bytes
        occurrences: Number of occurrences in data
        positions: Start positions of each occurrence
        interval_mean: Mean interval between occurrences (samples)
        interval_std: Standard deviation of intervals (consistency measure)
        entropy: Pattern entropy (low = more structured)
        score: Overall distinctiveness score (0-1, higher = better)
    """

    pattern: bytes
    length: int
    occurrences: int
    positions: list[int]
    interval_mean: float
    interval_std: float
    entropy: float
    score: float

    def __post_init__(self) -> None:
        """Validate candidate signature."""
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.occurrences < 0:
            raise ValueError("occurrences must be non-negative")
        if len(self.pattern) != self.length:
            raise ValueError("pattern length must match length field")
        if self.score < 0 or self.score > 1:
            raise ValueError("score must be in range [0, 1]")


class SignatureDiscovery:
    """Automatic signature and header discovery.

    : Automatic Signature Discovery

    This class analyzes binary data to automatically identify candidate
    signatures, headers, and delimiters based on statistical patterns.
    """

    def __init__(self, min_length: int = 4, max_length: int = 16, min_occurrences: int = 2):
        """Initialize signature discovery.

        Args:
            min_length: Minimum signature length in bytes
            max_length: Maximum signature length in bytes
            min_occurrences: Minimum number of times a pattern must occur to be considered

        Raises:
            ValueError: If min_length or max_length are invalid
        """
        if min_length < 1:
            raise ValueError("min_length must be at least 1")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        if min_occurrences < 1:
            raise ValueError("min_occurrences must be at least 1")

        self.min_length = min_length
        self.max_length = max_length
        self.min_occurrences = min_occurrences

    def discover_signatures(
        self, data: bytes | NDArray[np.uint8] | list[bytes]
    ) -> list[CandidateSignature]:
        """Discover candidate signatures in data.

        : General signature discovery

        Finds byte patterns that appear regularly throughout the data,
        suggesting they may be headers, sync markers, or delimiters.

        Args:
            data: Input binary data or list of messages

        Returns:
            List of CandidateSignature sorted by score (best first)

        Examples:
            >>> data = b"\\xAA\\x55DATA" * 100
            >>> discovery = SignatureDiscovery(min_length=2, max_length=8)
            >>> sigs = discovery.discover_signatures(data)
            >>> assert any(s.pattern == b"\\xAA\\x55" for s in sigs)
        """
        # Handle list of messages
        if isinstance(data, list):
            # Concatenate all messages for analysis
            data_bytes = b"".join(_to_bytes(msg) for msg in data)
        else:
            data_bytes = _to_bytes(data)
        n = len(data_bytes)

        if n < self.min_length:
            return []

        # Find all repeating patterns
        pattern_dict = defaultdict(list)

        for length in range(self.min_length, min(self.max_length + 1, n + 1)):
            for i in range(n - length + 1):
                pattern = data_bytes[i : i + length]
                pattern_dict[pattern].append(i)

        # Analyze each pattern
        candidates = []
        for pattern, positions in pattern_dict.items():
            # Filter by min_occurrences
            if len(positions) < self.min_occurrences:
                continue

            # Calculate statistics
            intervals = np.diff(positions)
            interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
            interval_std = float(np.std(intervals)) if len(intervals) > 0 else 0.0

            # Calculate pattern entropy
            entropy = _calculate_entropy(pattern)

            # Calculate distinctiveness score
            score = self._calculate_score(
                pattern=pattern,
                occurrences=len(positions),
                interval_mean=interval_mean,
                interval_std=interval_std,
                entropy=entropy,
                data_length=n,
            )

            candidates.append(
                CandidateSignature(
                    pattern=pattern,
                    length=len(pattern),
                    occurrences=len(positions),
                    positions=sorted(positions),
                    interval_mean=interval_mean,
                    interval_std=interval_std,
                    entropy=entropy,
                    score=score,
                )
            )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates

    def find_header_candidates(
        self, data: bytes | NDArray[np.uint8], max_candidates: int = 20
    ) -> list[CandidateSignature]:
        """Find patterns likely to be message headers.

        : Header candidate detection

        Headers typically:
        - Have low entropy (structured, not random)
        - Appear at regular intervals
        - Are relatively short (2-16 bytes)
        - May contain magic bytes or sync markers

        Args:
            data: Input binary data
            max_candidates: Maximum number of candidates to return

        Returns:
            List of CandidateSignature sorted by likelihood (best first)
        """
        # Discover all signatures
        candidates = self.discover_signatures(data)

        # Filter and rank for header characteristics
        header_candidates = []
        for sig in candidates:
            # Headers should have low entropy (structured)
            if sig.entropy > 6.0:
                continue

            # Headers should be reasonably frequent
            if sig.occurrences < 3:
                continue

            # Prefer regular intervals (low std deviation)
            regularity = 1.0 / (1.0 + sig.interval_std / max(sig.interval_mean, 1.0))

            # Combine score with header-specific features
            header_score = sig.score * 0.6 + (1.0 - sig.entropy / 8.0) * 0.2 + regularity * 0.2

            header_candidates.append(
                CandidateSignature(
                    pattern=sig.pattern,
                    length=sig.length,
                    occurrences=sig.occurrences,
                    positions=sig.positions,
                    interval_mean=sig.interval_mean,
                    interval_std=sig.interval_std,
                    entropy=sig.entropy,
                    score=header_score,
                )
            )

        # Sort by header score
        header_candidates.sort(key=lambda x: x.score, reverse=True)

        return header_candidates[:max_candidates]

    def find_delimiter_candidates(
        self, data: bytes | NDArray[np.uint8]
    ) -> list[CandidateSignature]:
        """Find patterns likely to be message delimiters.

        : Delimiter candidate detection

        Delimiters typically:
        - Are short (1-4 bytes)
        - Have very low entropy (often single byte like \\n, \\0, etc.)
        - Appear frequently
        - May have variable intervals

        Args:
            data: Input binary data

        Returns:
            List of CandidateSignature sorted by likelihood (best first)
        """
        data_bytes = _to_bytes(data)
        n = len(data_bytes)

        if n < 2:
            return []

        # Focus on short patterns (typical delimiters)
        delimiter_candidates = []
        max_delim_length = min(4, self.max_length)

        for length in range(1, max_delim_length + 1):
            pattern_positions = defaultdict(list)

            for i in range(n - length + 1):
                pattern = data_bytes[i : i + length]
                pattern_positions[pattern].append(i)

            for pattern, positions in pattern_positions.items():
                if len(positions) < 5:  # Delimiters should be frequent
                    continue

                # Calculate statistics
                intervals = np.diff(positions)
                interval_mean = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
                interval_std = float(np.std(intervals)) if len(intervals) > 0 else 0.0
                entropy = _calculate_entropy(pattern)

                # Delimiters should have very low entropy
                if entropy > 3.0:
                    continue

                # Calculate delimiter score
                # High frequency + low entropy + short length = good delimiter
                frequency_score = min(len(positions) / (n / 100.0), 1.0)
                entropy_score = 1.0 - entropy / 8.0
                length_score = 1.0 - (length - 1) / max_delim_length

                delimiter_score = frequency_score * 0.5 + entropy_score * 0.3 + length_score * 0.2

                delimiter_candidates.append(
                    CandidateSignature(
                        pattern=pattern,
                        length=length,
                        occurrences=len(positions),
                        positions=sorted(positions),
                        interval_mean=interval_mean,
                        interval_std=interval_std,
                        entropy=entropy,
                        score=delimiter_score,
                    )
                )

        # Sort by delimiter score
        delimiter_candidates.sort(key=lambda x: x.score, reverse=True)

        return delimiter_candidates[:20]  # Top 20 delimiter candidates

    def rank_signatures(self, candidates: list[CandidateSignature]) -> list[CandidateSignature]:
        """Rank signatures by distinctiveness.

        : Signature ranking

        Re-ranks candidates considering:
        - Frequency vs. expected random occurrence
        - Regularity of appearance
        - Entropy characteristics
        - Pattern uniqueness

        Args:
            candidates: List of candidate signatures

        Returns:
            Re-ranked list of CandidateSignature
        """
        if not candidates:
            return []

        # Re-calculate scores with more sophisticated ranking
        ranked = []
        for sig in candidates:
            # Regularity measure
            if sig.interval_mean > 0:
                regularity = 1.0 / (1.0 + sig.interval_std / sig.interval_mean)
            else:
                regularity = 0.0

            # Entropy score (prefer low entropy for signatures)
            entropy_score = max(0.0, 1.0 - sig.entropy / 8.0)

            # Frequency score (normalized)
            frequency_score = min(sig.occurrences / 100.0, 1.0)

            # Combined score
            new_score = regularity * 0.4 + entropy_score * 0.3 + frequency_score * 0.3

            ranked.append(
                CandidateSignature(
                    pattern=sig.pattern,
                    length=sig.length,
                    occurrences=sig.occurrences,
                    positions=sig.positions,
                    interval_mean=sig.interval_mean,
                    interval_std=sig.interval_std,
                    entropy=sig.entropy,
                    score=new_score,
                )
            )

        # Sort by new score
        ranked.sort(key=lambda x: x.score, reverse=True)

        return ranked

    def _calculate_score(
        self,
        pattern: bytes,
        occurrences: int,
        interval_mean: float,
        interval_std: float,
        entropy: float,
        data_length: int,
    ) -> float:
        """Calculate distinctiveness score for a pattern.

        Args:
            pattern: The byte pattern
            occurrences: Number of occurrences
            interval_mean: Mean interval between occurrences
            interval_std: Standard deviation of intervals
            entropy: Pattern entropy
            data_length: Total data length

        Returns:
            Score in range [0, 1], higher is more distinctive
        """
        # Frequency score (normalized)
        frequency_score = min(occurrences / 50.0, 1.0)

        # Regularity score (prefer consistent intervals)
        if interval_mean > 0:
            regularity_score = 1.0 / (1.0 + interval_std / interval_mean)
        else:
            regularity_score = 0.0

        # Entropy score (prefer structured patterns, not random)
        entropy_score = max(0.0, 1.0 - entropy / 8.0)

        # Length score (prefer medium-length patterns)
        optimal_length = 4.0
        length_score = 1.0 - abs(len(pattern) - optimal_length) / 8.0
        length_score = max(0.0, length_score)

        # Combine scores
        score = (
            frequency_score * 0.3
            + regularity_score * 0.4
            + entropy_score * 0.2
            + length_score * 0.1
        )

        return min(1.0, max(0.0, score))


# Convenience functions


def discover_signatures(
    data: bytes | NDArray[np.uint8] | list[bytes],
    min_length: int = 4,
    max_length: int = 16,
    min_occurrences: int = 2,
) -> list[CandidateSignature]:
    """Convenience function for signature discovery.

    : Signature discovery API

    Args:
        data: Input binary data or list of messages
        min_length: Minimum signature length
        max_length: Maximum signature length
        min_occurrences: Minimum number of times a pattern must occur

    Returns:
        List of CandidateSignature sorted by score

    Examples:
        >>> data = b"\\xFF\\xFF" + b"DATA" * 50
        >>> signatures = discover_signatures(data, min_length=2)
    """
    discovery = SignatureDiscovery(min_length, max_length, min_occurrences)
    return discovery.discover_signatures(data)


def find_header_candidates(data: bytes | NDArray[np.uint8]) -> list[CandidateSignature]:
    """Find header candidates.

    : Header discovery API

    Args:
        data: Input binary data

    Returns:
        List of header candidates

    Examples:
        >>> data = b"HDR" + b"payload" * 20
        >>> headers = find_header_candidates(data)
    """
    discovery = SignatureDiscovery(min_length=2, max_length=16)
    return discovery.find_header_candidates(data)


def find_delimiter_candidates(data: bytes | NDArray[np.uint8]) -> list[CandidateSignature]:
    """Find delimiter candidates.

    : Delimiter discovery API

    Args:
        data: Input binary data

    Returns:
        List of delimiter candidates

    Examples:
        >>> data = b"field1,field2,field3"
        >>> delimiters = find_delimiter_candidates(data)
        >>> assert any(d.pattern == b"," for d in delimiters)
    """
    discovery = SignatureDiscovery(min_length=1, max_length=4)
    return discovery.find_delimiter_candidates(data)


# Helper functions


def _to_bytes(data: bytes | NDArray[np.uint8] | memoryview | bytearray) -> bytes:
    """Convert input data to bytes.

    Args:
        data: Input data (bytes, bytearray, memoryview, or numpy array)

    Returns:
        Bytes representation

    Raises:
        TypeError: If data type is not supported
    """
    if isinstance(data, bytes):
        return data
    elif isinstance(data, bytearray | memoryview):
        return bytes(data)
    elif isinstance(data, np.ndarray):
        return data.astype(np.uint8).tobytes()
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte sequence.

    : Entropy calculation

    Args:
        data: Byte sequence

    Returns:
        Entropy in bits (0-8 for byte data)

    Examples:
        >>> _calculate_entropy(b"\\x00" * 100)  # All same byte
        0.0
        >>> entropy = _calculate_entropy(bytes(range(256)))  # Uniform distribution
        >>> assert entropy > 7.9  # Close to 8.0
    """
    if len(data) == 0:
        return 0.0

    # Count byte frequencies
    byte_counts = Counter(data)
    n = len(data)

    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            prob = count / n
            entropy -= prob * math.log2(prob)

    return entropy
