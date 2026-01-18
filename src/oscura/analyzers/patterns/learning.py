"""Pattern learning and automatic discovery from binary data.

    - RE-PAT-004: Pattern Learning and Discovery

This module provides machine learning inspired approaches for discovering
patterns in binary data without prior knowledge, including entropy-based
segmentation, frequency analysis, and structural inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearnedPattern:
    """A pattern discovered through learning.

    Implements RE-PAT-004: Learned pattern representation.

    Attributes:
        pattern: The pattern bytes.
        frequency: Number of occurrences.
        confidence: Confidence score (0-1).
        positions: List of positions where found.
        context_before: Common bytes appearing before pattern.
        context_after: Common bytes appearing after pattern.
        is_structural: Whether pattern appears to be structural.
        is_delimiter: Whether pattern appears to be a delimiter.
    """

    pattern: bytes
    frequency: int
    confidence: float
    positions: list[int] = field(default_factory=list)
    context_before: bytes = b""
    context_after: bytes = b""
    is_structural: bool = False
    is_delimiter: bool = False


@dataclass
class StructureHypothesis:
    """Hypothesis about data structure.

    Implements RE-PAT-004: Structure hypothesis.

    Attributes:
        field_boundaries: Detected field boundaries.
        field_types: Inferred field types.
        header_size: Estimated header size.
        record_size: Estimated record size (if fixed).
        delimiters: Detected delimiters.
        confidence: Overall confidence.
    """

    field_boundaries: list[int]
    field_types: list[str]
    header_size: int
    record_size: int | None
    delimiters: list[bytes]
    confidence: float


@dataclass
class NgramModel:
    """N-gram language model for binary data.

    Implements RE-PAT-004: N-gram modeling.

    Attributes:
        n: N-gram size.
        counts: N-gram frequency counts.
        total: Total n-grams observed.
        vocabulary_size: Number of unique n-grams.
    """

    n: int
    counts: dict[bytes, int] = field(default_factory=dict)
    total: int = 0
    vocabulary_size: int = 0


class PatternLearner:
    """Learn patterns from binary data samples.

    Implements RE-PAT-004: Pattern Learning and Discovery.

    Uses entropy analysis, n-gram frequency, and positional statistics
    to discover recurring patterns without prior knowledge.

    Example:
        >>> learner = PatternLearner()
        >>> learner.add_sample(data1)
        >>> learner.add_sample(data2)
        >>> patterns = learner.learn_patterns()
    """

    def __init__(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: int = 16,
        min_frequency: int = 3,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize pattern learner.

        Args:
            min_pattern_length: Minimum pattern length to consider.
            max_pattern_length: Maximum pattern length to consider.
            min_frequency: Minimum occurrences to consider pattern.
            min_confidence: Minimum confidence threshold.
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence

        self._samples: list[bytes] = []
        self._ngram_models: dict[int, NgramModel] = {}
        self._position_stats: dict[bytes, list[int]] = defaultdict(list)

    def add_sample(self, data: bytes) -> None:
        """Add a data sample for learning.

        Args:
            data: Binary data sample.
        """
        self._samples.append(data)

    def add_samples(self, samples: Sequence[bytes]) -> None:
        """Add multiple data samples.

        Args:
            samples: List of binary data samples.
        """
        self._samples.extend(samples)

    def learn_patterns(self, top_k: int = 20) -> list[LearnedPattern]:
        """Learn patterns from accumulated samples.

        Implements RE-PAT-004: Pattern discovery workflow.

        Args:
            top_k: Maximum number of patterns to return.

        Returns:
            List of discovered patterns, sorted by confidence.
        """
        if not self._samples:
            return []

        # Build n-gram models
        self._build_ngram_models()

        # Find candidate patterns
        candidates = self._find_candidates()

        # Score and filter patterns
        scored = self._score_patterns(candidates)

        # Sort by confidence and return top K
        scored.sort(key=lambda p: -p.confidence)
        return scored[:top_k]

    def learn_structure(self) -> StructureHypothesis:
        """Learn structural patterns from samples.

        Implements RE-PAT-004: Structure inference.

        Returns:
            StructureHypothesis about data organization.
        """
        if not self._samples:
            return StructureHypothesis(
                field_boundaries=[],
                field_types=[],
                header_size=0,
                record_size=None,
                delimiters=[],
                confidence=0.0,
            )

        # Analyze entropy profile for field boundaries
        boundaries = self._detect_field_boundaries()

        # Infer field types
        field_types = self._infer_field_types(boundaries)

        # Estimate header size
        header_size = self._estimate_header_size(boundaries)

        # Check for fixed record size
        record_size = self._detect_record_size()

        # Find delimiters
        delimiters = self._find_delimiters()

        # Calculate confidence
        confidence = self._calculate_structure_confidence(
            boundaries, field_types, record_size, delimiters
        )

        return StructureHypothesis(
            field_boundaries=boundaries,
            field_types=field_types,
            header_size=header_size,
            record_size=record_size,
            delimiters=delimiters,
            confidence=confidence,
        )

    def predict_next_bytes(
        self, context: bytes, n_predictions: int = 5
    ) -> list[tuple[bytes, float]]:
        """Predict likely next bytes given context.

        Implements RE-PAT-004: Byte prediction using n-gram models.

        Args:
            context: Context bytes.
            n_predictions: Number of predictions to return.

        Returns:
            List of (next_byte, probability) tuples.
        """
        predictions = []

        # Use largest n-gram model that fits context
        for n in range(min(len(context) + 1, self.max_pattern_length), 0, -1):
            if n not in self._ngram_models:
                continue

            model = self._ngram_models[n]
            prefix = context[-(n - 1) :] if n > 1 else b""

            # Find matching prefixes
            matching = {}
            for ngram, count in model.counts.items():
                if ngram[:-1] == prefix:
                    matching[ngram[-1:]] = count

            if matching:
                total = sum(matching.values())
                for byte_val, count in matching.items():
                    prob = count / total
                    predictions.append((byte_val, prob))
                break

        # Sort by probability
        predictions.sort(key=lambda x: -x[1])
        return predictions[:n_predictions]

    def build_ngram_model(self, n: int) -> NgramModel:
        """Build n-gram model from samples.

        Args:
            n: N-gram size.

        Returns:
            NgramModel with frequency statistics.
        """
        model = NgramModel(n=n)

        for sample in self._samples:
            for i in range(len(sample) - n + 1):
                ngram = sample[i : i + n]
                if ngram not in model.counts:
                    model.counts[ngram] = 0
                    model.vocabulary_size += 1
                model.counts[ngram] += 1
                model.total += 1

        self._ngram_models[n] = model
        return model

    def _build_ngram_models(self) -> None:
        """Build n-gram models for all sizes."""
        for n in range(self.min_pattern_length, self.max_pattern_length + 1):
            self.build_ngram_model(n)

    def _find_candidates(self) -> dict[bytes, int]:
        """Find candidate patterns based on frequency.

        Returns:
            Dictionary mapping patterns to frequencies.
        """
        candidates = {}

        for n in range(self.min_pattern_length, self.max_pattern_length + 1):
            if n not in self._ngram_models:
                continue

            model = self._ngram_models[n]
            for pattern, count in model.counts.items():
                if count >= self.min_frequency:
                    candidates[pattern] = count

        return candidates

    def _score_patterns(self, candidates: dict[bytes, int]) -> list[LearnedPattern]:
        """Score candidate patterns.

        Args:
            candidates: Dictionary of pattern -> frequency.

        Returns:
            List of scored LearnedPattern objects.
        """
        patterns = []

        for pattern, frequency in candidates.items():
            # Find all positions across samples
            positions = []
            for sample_idx, sample in enumerate(self._samples):
                start = 0
                while True:
                    pos = sample.find(pattern, start)
                    if pos == -1:
                        break
                    positions.append((sample_idx, pos))
                    start = pos + 1

            # Calculate confidence based on distribution
            confidence = self._calculate_pattern_confidence(pattern, positions)

            if confidence < self.min_confidence:
                continue

            # Get context
            context_before, context_after = self._get_context(pattern, positions)

            # Check if structural
            is_structural = self._is_structural(pattern, positions)

            # Check if delimiter
            is_delimiter = self._is_delimiter(pattern, positions)

            patterns.append(
                LearnedPattern(
                    pattern=pattern,
                    frequency=frequency,
                    confidence=confidence,
                    positions=[p for _, p in positions],
                    context_before=context_before,
                    context_after=context_after,
                    is_structural=is_structural,
                    is_delimiter=is_delimiter,
                )
            )

        return patterns

    def _calculate_pattern_confidence(
        self, pattern: bytes, positions: list[tuple[int, int]]
    ) -> float:
        """Calculate confidence score for pattern.

        Args:
            pattern: The pattern.
            positions: List of (sample_idx, position) tuples.

        Returns:
            Confidence score (0-1).
        """
        if not positions:
            return 0.0

        # Factor 1: Frequency across samples
        samples_with_pattern = len({p[0] for p in positions})
        sample_coverage = samples_with_pattern / len(self._samples)

        # Factor 2: Positional consistency
        position_offsets = [p[1] for p in positions]
        if len(position_offsets) > 1:
            variance = float(np.var(position_offsets))
            max_pos = max(max(len(s) for s in self._samples), 1)
            position_consistency = 1.0 / (1.0 + variance / (max_pos**2))
        else:
            position_consistency = 0.5

        # Factor 3: Pattern complexity (non-trivial patterns)
        unique_bytes = len(set(pattern))
        complexity = unique_bytes / len(pattern) if pattern else 0

        # Combined score
        confidence = 0.4 * sample_coverage + 0.3 * position_consistency + 0.3 * complexity

        return float(min(1.0, confidence))

    def _get_context(self, pattern: bytes, positions: list[tuple[int, int]]) -> tuple[bytes, bytes]:
        """Get common context before and after pattern.

        Args:
            pattern: The pattern.
            positions: List of (sample_idx, position) tuples.

        Returns:
            Tuple of (context_before, context_after).
        """
        before_bytes = []
        after_bytes = []

        context_len = min(4, self.min_pattern_length)

        for sample_idx, pos in positions[:100]:  # Limit samples
            sample = self._samples[sample_idx]

            # Bytes before
            if pos >= context_len:
                before_bytes.append(sample[pos - context_len : pos])

            # Bytes after
            end_pos = pos + len(pattern)
            if end_pos + context_len <= len(sample):
                after_bytes.append(sample[end_pos : end_pos + context_len])

        # Find most common
        context_before = b""
        context_after = b""

        if before_bytes:
            counter = Counter(before_bytes)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:
                context_before = most_common[0][0]

        if after_bytes:
            counter = Counter(after_bytes)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:
                context_after = most_common[0][0]

        return context_before, context_after

    def _is_structural(self, pattern: bytes, positions: list[tuple[int, int]]) -> bool:
        """Check if pattern appears structural.

        Args:
            pattern: The pattern.
            positions: List of positions.

        Returns:
            True if pattern appears structural.
        """
        if not positions:
            return False

        # Structural patterns tend to appear at consistent offsets
        offsets = [p[1] for p in positions]
        if len(set(offsets)) == 1:
            return True

        # Or at regular intervals
        if len(offsets) > 2:
            diffs = [
                offsets[i + 1] - offsets[i]
                for i in range(len(offsets) - 1)
                if offsets[i + 1] > offsets[i]
            ]
            if diffs and len(set(diffs)) == 1:
                return True

        return False

    def _is_delimiter(self, pattern: bytes, positions: list[tuple[int, int]]) -> bool:
        """Check if pattern appears to be a delimiter.

        Args:
            pattern: The pattern.
            positions: List of positions.

        Returns:
            True if pattern appears to be a delimiter.
        """
        # Delimiters often have regular spacing
        if not positions:
            return False

        # Group by sample
        by_sample = defaultdict(list)
        for sample_idx, pos in positions:
            by_sample[sample_idx].append(pos)

        regular_count = 0
        for sample_positions in by_sample.values():
            if len(sample_positions) >= 3:
                diffs = [
                    sample_positions[i + 1] - sample_positions[i]
                    for i in range(len(sample_positions) - 1)
                ]
                # Check for regular intervals
                if len(set(diffs)) == 1 or (diffs and max(diffs) - min(diffs) < 4):
                    regular_count += 1

        return regular_count >= len(by_sample) * 0.5

    def _detect_field_boundaries(self) -> list[int]:
        """Detect field boundaries using entropy transitions."""
        if not self._samples:
            return []

        # Use first sample or combined samples
        combined = b"".join(self._samples[:10])

        from oscura.analyzers.statistical.entropy import detect_entropy_transitions

        try:
            transitions = detect_entropy_transitions(combined, window=64, threshold=0.8, min_gap=4)
            return [t.offset for t in transitions]
        except ValueError:
            return []

    def _infer_field_types(self, boundaries: list[int]) -> list[str]:
        """Infer field types based on content patterns.

        Args:
            boundaries: Field boundary offsets.

        Returns:
            List of inferred field types.
        """
        if not boundaries or not self._samples:
            return []

        field_types = []
        sample = self._samples[0]
        boundaries = [0] + boundaries + [len(sample)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = min(boundaries[i + 1], len(sample))
            field_data = sample[start:end]

            field_type = self._classify_field(field_data)
            field_types.append(field_type)

        return field_types

    def _classify_field(self, data: bytes) -> str:
        """Classify a field based on its content.

        Args:
            data: Field data.

        Returns:
            Field type string.
        """
        if not data:
            return "empty"

        # Check for constant
        if len(set(data)) == 1:
            return "constant"

        # Check for counter (monotonic)
        if len(data) <= 4:
            values = list(data)
            if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
                return "counter"

        # Check for printable text
        printable = sum(1 for b in data if 32 <= b <= 126)
        if printable / len(data) > 0.8:
            return "text"

        # Check for high entropy (random/encrypted)
        from oscura.analyzers.statistical.entropy import shannon_entropy

        entropy = shannon_entropy(data)
        if entropy > 7.0:
            return "random"
        elif entropy > 5.0:
            return "binary"

        return "structured"

    def _estimate_header_size(self, boundaries: list[int]) -> int:
        """Estimate header size from boundaries.

        Args:
            boundaries: Field boundary offsets.

        Returns:
            Estimated header size.
        """
        if not boundaries:
            return 0

        # Header typically ends at first high-entropy transition
        for b in boundaries:
            if b > 0:
                return b

        return boundaries[0] if boundaries else 0

    def _detect_record_size(self) -> int | None:
        """Detect fixed record size if present.

        Returns:
            Record size or None if variable.
        """
        if len(self._samples) < 2:
            return None

        # Check if all samples have same length
        lengths = [len(s) for s in self._samples]
        if len(set(lengths)) == 1:
            return lengths[0]

        # Check for GCD of lengths (might indicate record size)
        from functools import reduce
        from math import gcd

        if all(length > 0 for length in lengths):
            common_div = reduce(gcd, lengths)
            if common_div > 1 and common_div != min(lengths):
                return common_div

        return None

    def _find_delimiters(self) -> list[bytes]:
        """Find delimiter patterns.

        Returns:
            List of likely delimiter bytes.
        """
        patterns = self.learn_patterns(top_k=50)
        return [p.pattern for p in patterns if p.is_delimiter][:5]

    def _calculate_structure_confidence(
        self,
        boundaries: list[int],
        field_types: list[str],
        record_size: int | None,
        delimiters: list[bytes],
    ) -> float:
        """Calculate confidence in structure hypothesis.

        Args:
            boundaries: Detected boundaries.
            field_types: Inferred types.
            record_size: Detected record size.
            delimiters: Found delimiters.

        Returns:
            Confidence score (0-1).
        """
        score = 0.0

        # Having boundaries adds confidence
        if boundaries:
            score += 0.3

        # Having non-unknown field types adds confidence
        known_types = sum(1 for t in field_types if t != "structured")
        if field_types:
            score += 0.2 * (known_types / len(field_types))

        # Fixed record size adds confidence
        if record_size is not None:
            score += 0.2

        # Delimiters add confidence
        if delimiters:
            score += 0.2

        # Multiple samples add confidence
        if len(self._samples) > 5:
            score += 0.1

        return min(1.0, score)


def learn_patterns_from_data(
    data: bytes | Sequence[bytes],
    min_length: int = 2,
    max_length: int = 16,
    min_frequency: int = 3,
    top_k: int = 20,
) -> list[LearnedPattern]:
    """Learn patterns from binary data.

    Implements RE-PAT-004: Pattern Learning and Discovery.

    Args:
        data: Single data sample or list of samples.
        min_length: Minimum pattern length.
        max_length: Maximum pattern length.
        min_frequency: Minimum occurrences.
        top_k: Number of patterns to return.

    Returns:
        List of discovered patterns.

    Example:
        >>> patterns = learn_patterns_from_data(binary_data)
        >>> for p in patterns:
        ...     print(f"Pattern: {p.pattern.hex()}, freq: {p.frequency}")
    """
    learner = PatternLearner(
        min_pattern_length=min_length,
        max_pattern_length=max_length,
        min_frequency=min_frequency,
    )

    if isinstance(data, bytes):
        learner.add_sample(data)
    else:
        learner.add_samples(data)

    return learner.learn_patterns(top_k=top_k)


def infer_structure(samples: Sequence[bytes]) -> StructureHypothesis:
    """Infer data structure from samples.

    Implements RE-PAT-004: Structure inference.

    Args:
        samples: List of binary data samples.

    Returns:
        StructureHypothesis about data organization.

    Example:
        >>> hypothesis = infer_structure(packet_samples)
        >>> print(f"Header size: {hypothesis.header_size}")
    """
    learner = PatternLearner()
    learner.add_samples(samples)
    return learner.learn_structure()


def find_recurring_structures(
    data: bytes,
    min_size: int = 8,
    max_size: int = 256,
) -> list[tuple[int, int, float]]:
    """Find recurring fixed-size structures in data.

    Implements RE-PAT-004: Structure detection.

    Args:
        data: Binary data.
        min_size: Minimum structure size.
        max_size: Maximum structure size.

    Returns:
        List of (size, offset, confidence) tuples for detected structures.
    """
    results = []

    for size in range(min_size, min(max_size, len(data) // 2) + 1):
        # Check if data divides evenly
        if len(data) % size != 0:
            continue

        num_records = len(data) // size
        if num_records < 2:
            continue

        # Compare records for similarity
        records = [data[i * size : (i + 1) * size] for i in range(num_records)]

        # Calculate similarity between consecutive records
        similarities = []
        for i in range(len(records) - 1):
            matching = sum(a == b for a, b in zip(records[i], records[i + 1], strict=True))
            similarities.append(matching / size)

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            if avg_similarity > 0.3:  # Some structural similarity
                results.append((size, 0, avg_similarity))

    # Sort by confidence
    results.sort(key=lambda x: -x[2])
    return results[:5]


__all__ = [
    # Data classes
    "LearnedPattern",
    "NgramModel",
    # Classes
    "PatternLearner",
    "StructureHypothesis",
    "find_recurring_structures",
    "infer_structure",
    # Functions
    "learn_patterns_from_data",
]
