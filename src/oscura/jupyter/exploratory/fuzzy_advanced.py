"""Advanced fuzzy matching for binary pattern analysis.

This module provides advanced fuzzy matching capabilities including
pattern variant characterization, consensus finding, and multiple
binary sequence alignment.


Example:
    >>> from oscura.jupyter.exploratory.fuzzy_advanced import (
    ...     characterize_variants,
    ...     align_sequences,
    ... )
    >>> patterns = [b'\\x12\\x34\\x56', b'\\x12\\x35\\x56', b'\\x12\\x34\\x57']
    >>> result = characterize_variants(patterns)
    >>> print(f"Consensus: {result.consensus.hex()}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "AlignedSequence",
    "AlignmentResult",
    "PositionAnalysis",
    "VariantCharacterization",
    "VariationType",
    "align_sequences",
    "align_two_sequences",
    "characterize_variants",
    "compute_conservation_scores",
]


# =============================================================================
# =============================================================================


class VariationType(Enum):
    """Classification of position variation.

    References:
        FUZZY-004: Binary Pattern Variant Characterization and Consensus
    """

    CONSTANT = "constant"  # Entropy < 0.5 bits, confidence >= 0.95
    LOW_VARIATION = "low_variation"  # Entropy 0.5-2.0 bits
    HIGH_VARIATION = "high_variation"  # Entropy 2.0-6.0 bits
    RANDOM = "random"  # Entropy > 6.0 bits (likely random or encrypted)


@dataclass
class PositionAnalysis:
    """Analysis of a single byte position.

    Attributes:
        position: Byte position index
        consensus_byte: Most common byte at this position
        consensus_confidence: Frequency of consensus byte
        entropy: Shannon entropy in bits
        variation_type: Classification of variation
        value_distribution: Distribution of byte values
        is_error: True if variation likely from errors

    References:
        FUZZY-004: Binary Pattern Variant Characterization and Consensus
    """

    position: int
    consensus_byte: int
    consensus_confidence: float
    entropy: float
    variation_type: VariationType
    value_distribution: dict[int, int]
    is_error: bool = False


@dataclass
class VariantCharacterization:
    """Result of variant characterization.

    Attributes:
        consensus: Consensus pattern (most common byte per position)
        positions: Per-position analysis
        constant_positions: Indices of constant positions
        variable_positions: Indices of variable positions
        suggested_boundaries: Suggested field boundaries
        pattern_count: Number of patterns analyzed
        min_length: Minimum pattern length

    References:
        FUZZY-004: Binary Pattern Variant Characterization and Consensus
    """

    consensus: bytes
    positions: list[PositionAnalysis]
    constant_positions: list[int]
    variable_positions: list[int]
    suggested_boundaries: list[int]
    pattern_count: int
    min_length: int


def _compute_entropy(values: list[int]) -> float:
    """Compute Shannon entropy of byte values.

    Args:
        values: List of byte values

    Returns:
        Entropy in bits
    """
    if not values:
        return 0.0

    # Count frequencies
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    total = len(values)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    return entropy


def _classify_variation(entropy: float, confidence: float) -> VariationType:
    """Classify variation type based on entropy and confidence.

    Args:
        entropy: Shannon entropy in bits
        confidence: Consensus confidence

    Returns:
        Variation type classification
    """
    if entropy < 0.5 and confidence >= 0.95:
        return VariationType.CONSTANT
    elif entropy < 2.0:
        return VariationType.LOW_VARIATION
    elif entropy < 6.0:
        return VariationType.HIGH_VARIATION
    else:
        return VariationType.RANDOM


def _detect_error_variation(
    values: list[int],
    consensus: int,
    confidence: float,
) -> bool:
    """Detect if variation is likely from errors vs intentional.

    Args:
        values: List of byte values at position
        consensus: Consensus byte
        confidence: Consensus confidence

    Returns:
        True if variation appears to be from errors
    """
    if confidence >= 0.99:
        # Very rare variations are likely errors
        return True

    # Check if variations are single-bit flips from consensus
    variations = [v for v in set(values) if v != consensus]
    if not variations:
        return False

    single_bit_flips = 0
    for v in variations:
        diff = v ^ consensus
        # Check if exactly one bit different
        if diff != 0 and (diff & (diff - 1)) == 0:
            single_bit_flips += 1

    # If most variations are single-bit flips, likely errors
    return single_bit_flips >= len(variations) * 0.8


def characterize_variants(
    patterns: Sequence[bytes | bytearray],
    min_confidence: float = 0.95,
) -> VariantCharacterization:
    """Characterize variants in a collection of binary patterns.

    Analyzes a collection of similar patterns to find consensus sequence,
    variable positions, and variant frequencies.

    Args:
        patterns: Collection of binary patterns
        min_confidence: Minimum confidence for constant classification

    Returns:
        VariantCharacterization with analysis results

    Example:
        >>> patterns = [b'\\x12\\x34\\x56', b'\\x12\\x35\\x56']
        >>> result = characterize_variants(patterns)

    References:
        FUZZY-004: Binary Pattern Variant Characterization and Consensus
    """
    if not patterns:
        return _create_empty_characterization()

    pattern_count = len(patterns)
    min_length = min(len(p) for p in patterns)

    positions, consensus_bytes, constant_positions, variable_positions = _analyze_positions(
        patterns, min_length
    )

    boundaries = _suggest_field_boundaries(positions)

    return VariantCharacterization(
        consensus=bytes(consensus_bytes),
        positions=positions,
        constant_positions=constant_positions,
        variable_positions=variable_positions,
        suggested_boundaries=boundaries,
        pattern_count=pattern_count,
        min_length=min_length,
    )


def _create_empty_characterization() -> VariantCharacterization:
    """Create empty characterization result.

    Returns:
        Empty VariantCharacterization.
    """
    return VariantCharacterization(
        consensus=b"",
        positions=[],
        constant_positions=[],
        variable_positions=[],
        suggested_boundaries=[],
        pattern_count=0,
        min_length=0,
    )


def _analyze_positions(
    patterns: Sequence[bytes | bytearray], min_length: int
) -> tuple[list[PositionAnalysis], list[int], list[int], list[int]]:
    """Analyze each position across all patterns.

    Args:
        patterns: Collection of patterns.
        min_length: Minimum pattern length.

    Returns:
        Tuple of (positions, consensus_bytes, constant_positions, variable_positions).
    """
    positions: list[PositionAnalysis] = []
    consensus_bytes: list[int] = []
    constant_positions: list[int] = []
    variable_positions: list[int] = []

    for pos in range(min_length):
        values = [p[pos] for p in patterns if pos < len(p)]
        distribution = _build_distribution(values)

        consensus_byte = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        consensus_count = distribution[consensus_byte]
        consensus_confidence = consensus_count / len(values)

        entropy = _compute_entropy(values)
        variation_type = _classify_variation(entropy, consensus_confidence)
        is_error = _detect_error_variation(values, consensus_byte, consensus_confidence)

        analysis = PositionAnalysis(
            position=pos,
            consensus_byte=consensus_byte,
            consensus_confidence=consensus_confidence,
            entropy=entropy,
            variation_type=variation_type,
            value_distribution=distribution,
            is_error=is_error,
        )

        positions.append(analysis)
        consensus_bytes.append(consensus_byte)

        if variation_type == VariationType.CONSTANT:
            constant_positions.append(pos)
        else:
            variable_positions.append(pos)

    return positions, consensus_bytes, constant_positions, variable_positions


def _build_distribution(values: list[int]) -> dict[int, int]:
    """Build frequency distribution of values.

    Args:
        values: List of byte values.

    Returns:
        Dictionary mapping value to count.
    """
    distribution: dict[int, int] = {}
    for v in values:
        distribution[v] = distribution.get(v, 0) + 1
    return distribution


def _suggest_field_boundaries(positions: list[PositionAnalysis]) -> list[int]:
    """Suggest field boundaries from position analysis.

    Args:
        positions: List of position analyses.

    Returns:
        List of boundary positions.
    """
    boundaries: list[int] = []
    prev_is_constant = None

    for pos, analysis in enumerate(positions):
        is_constant = analysis.variation_type == VariationType.CONSTANT
        if prev_is_constant is not None and is_constant != prev_is_constant:
            boundaries.append(pos)
        prev_is_constant = is_constant

    return boundaries


# =============================================================================
# =============================================================================


@dataclass
class AlignedSequence:
    """A sequence with alignment information.

    Attributes:
        original: Original sequence
        aligned: Aligned sequence with gaps
        gaps: Gap positions
        score: Alignment score

    References:
        FUZZY-005: Multiple Binary Sequence Alignment (MSA)
    """

    original: bytes
    aligned: bytes
    gaps: list[int]
    score: float


@dataclass
class AlignmentResult:
    """Result of sequence alignment.

    Attributes:
        sequences: Aligned sequences
        conservation_scores: Per-position conservation scores
        conserved_regions: Indices of highly conserved regions
        gap_positions: Common gap positions
        alignment_score: Overall alignment score

    References:
        FUZZY-005: Multiple Binary Sequence Alignment (MSA)
    """

    sequences: list[AlignedSequence]
    conservation_scores: list[float]
    conserved_regions: list[tuple[int, int]]
    gap_positions: list[int]
    alignment_score: float


# Gap representation
GAP_BYTE = 0xFF  # Using 0xFF as gap marker


def _needleman_wunsch(
    seq1: bytes,
    seq2: bytes,
    match_bonus: int = 1,
    mismatch_penalty: int = -1,
    gap_open: int = -2,
    gap_extend: int = -1,
) -> tuple[bytes, bytes, float]:
    """Needleman-Wunsch global alignment algorithm.

    Args:
        seq1: First sequence
        seq2: Second sequence
        match_bonus: Score for match
        mismatch_penalty: Score for mismatch
        gap_open: Gap opening penalty
        gap_extend: Gap extension penalty

    Returns:
        (aligned_seq1, aligned_seq2, score)
    """
    # Initialize matrices
    DIAG, UP, LEFT = 0, 1, 2
    score, traceback = _initialize_alignment_matrices(seq1, seq2, gap_open, gap_extend, UP, LEFT)

    # Fill scoring matrix
    _fill_alignment_matrix(
        seq1,
        seq2,
        score,
        traceback,
        match_bonus,
        mismatch_penalty,
        gap_open,
        gap_extend,
        DIAG,
        UP,
        LEFT,
    )

    # Traceback to build alignment
    aligned1, aligned2 = _traceback_alignment(
        seq1, seq2, traceback, len(seq1), len(seq2), DIAG, UP, LEFT
    )

    return bytes(aligned1), bytes(aligned2), float(score[len(seq1), len(seq2)])


def _initialize_alignment_matrices(
    seq1: bytes,
    seq2: bytes,
    gap_open: int,
    gap_extend: int,
    UP: int,
    LEFT: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Initialize score and traceback matrices."""
    m, n = len(seq1), len(seq2)
    score = np.zeros((m + 1, n + 1), dtype=np.int32)
    traceback = np.zeros((m + 1, n + 1), dtype=np.int8)

    # Initialize first row and column with gap penalties
    for i in range(1, m + 1):
        score[i, 0] = gap_open + (i - 1) * gap_extend
        traceback[i, 0] = UP

    for j in range(1, n + 1):
        score[0, j] = gap_open + (j - 1) * gap_extend
        traceback[0, j] = LEFT

    return score, traceback


def _fill_alignment_matrix(
    seq1: bytes,
    seq2: bytes,
    score: np.ndarray[Any, Any],
    traceback: np.ndarray[Any, Any],
    match_bonus: int,
    mismatch_penalty: int,
    gap_open: int,
    gap_extend: int,
    DIAG: int,
    UP: int,
    LEFT: int,
) -> None:
    """Fill alignment matrix using dynamic programming."""
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            # Calculate scores for three possible moves
            diag_score = score[i - 1, j - 1] + (
                match_bonus if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
            )
            up_score = score[i - 1, j] + (gap_extend if traceback[i - 1, j] == UP else gap_open)
            left_score = score[i, j - 1] + (gap_extend if traceback[i, j - 1] == LEFT else gap_open)

            # Choose best score and direction
            best = max(diag_score, up_score, left_score)
            score[i, j] = best

            if best == diag_score:
                traceback[i, j] = DIAG
            elif best == up_score:
                traceback[i, j] = UP
            else:
                traceback[i, j] = LEFT


def _traceback_alignment(
    seq1: bytes,
    seq2: bytes,
    traceback: np.ndarray[Any, Any],
    i: int,
    j: int,
    DIAG: int,
    UP: int,
    LEFT: int,
) -> tuple[list[int], list[int]]:
    """Traceback through matrix to build final alignment."""
    aligned1: list[int] = []
    aligned2: list[int] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i, j] == DIAG:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and traceback[i, j] == UP:
            aligned1.append(seq1[i - 1])
            aligned2.append(GAP_BYTE)
            i -= 1
        else:
            aligned1.append(GAP_BYTE)
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return aligned1, aligned2


def _smith_waterman(
    seq1: bytes,
    seq2: bytes,
    match_bonus: int = 1,
    mismatch_penalty: int = -1,
    gap_penalty: int = -2,
) -> tuple[bytes, bytes, float, int, int]:
    """Smith-Waterman local alignment algorithm.

    Args:
        seq1: First sequence
        seq2: Second sequence
        match_bonus: Score for match
        mismatch_penalty: Score for mismatch
        gap_penalty: Gap penalty

    Returns:
        (aligned_seq1, aligned_seq2, score, start1, start2)
    """
    m, n = len(seq1), len(seq2)

    # Initialize score matrix
    score = np.zeros((m + 1, n + 1), dtype=np.int32)
    traceback = np.zeros((m + 1, n + 1), dtype=np.int8)

    max_score = 0
    max_i, max_j = 0, 0

    # Direction constants
    DIAG, UP, LEFT, STOP = 0, 1, 2, 3

    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Match/mismatch
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score[i - 1, j - 1] + match_bonus
            else:
                diag_score = score[i - 1, j - 1] + mismatch_penalty

            up_score = score[i - 1, j] + gap_penalty
            left_score = score[i, j - 1] + gap_penalty

            # Local alignment can restart (score 0)
            best = max(0, diag_score, up_score, left_score)
            score[i, j] = best

            if best == 0:
                traceback[i, j] = STOP
            elif best == diag_score:
                traceback[i, j] = DIAG
            elif best == up_score:
                traceback[i, j] = UP
            else:
                traceback[i, j] = LEFT

            if best > max_score:
                max_score = best
                max_i, max_j = i, j

    # Traceback from max score
    aligned1: list[int] = []
    aligned2: list[int] = []
    i, j = max_i, max_j

    while i > 0 and j > 0 and traceback[i, j] != STOP:
        if traceback[i, j] == DIAG:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif traceback[i, j] == UP:
            aligned1.append(seq1[i - 1])
            aligned2.append(GAP_BYTE)
            i -= 1
        else:
            aligned1.append(GAP_BYTE)
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()

    start1 = i
    start2 = j

    return bytes(aligned1), bytes(aligned2), float(max_score), start1, start2


def align_two_sequences(
    seq1: bytes,
    seq2: bytes,
    method: str = "global",
    gap_open: int = -2,
    gap_extend: int = -1,
    mismatch_penalty: int = -1,
    match_bonus: int = 1,
) -> tuple[bytes, bytes, float]:
    """Align two binary sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence
        method: 'global' (Needleman-Wunsch) or 'local' (Smith-Waterman)
        gap_open: Gap opening penalty
        gap_extend: Gap extension penalty
        mismatch_penalty: Mismatch penalty
        match_bonus: Match bonus

    Returns:
        (aligned_seq1, aligned_seq2, score)

    Raises:
        ValueError: If method is unknown.

    Example:
        >>> seq1 = b'\\x12\\x34\\x56\\x78'
        >>> seq2 = b'\\x12\\x34\\x78'
        >>> aligned1, aligned2, score = align_two_sequences(seq1, seq2)

    References:
        FUZZY-005: Multiple Binary Sequence Alignment (MSA)
    """
    if method == "global":
        return _needleman_wunsch(seq1, seq2, match_bonus, mismatch_penalty, gap_open, gap_extend)
    elif method == "local":
        aligned1, aligned2, score, _, _ = _smith_waterman(
            seq1, seq2, match_bonus, mismatch_penalty, gap_open
        )
        return aligned1, aligned2, score
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def align_sequences(
    sequences: Sequence[bytes],
    method: str = "progressive",
    gap_open: int = -2,
    gap_extend: int = -1,
) -> AlignmentResult:
    """Align multiple binary sequences.

    Performs multiple sequence alignment (MSA) on a collection of
    binary sequences.

    Args:
        sequences: Sequences to align
        method: 'progressive' or 'iterative'
        gap_open: Gap opening penalty
        gap_extend: Gap extension penalty

    Returns:
        AlignmentResult with aligned sequences

    Raises:
        ValueError: If method is unknown.

    Example:
        >>> seqs = [b'\\x12\\x34\\x56', b'\\x12\\x56', b'\\x12\\x34\\x78']
        >>> result = align_sequences(seqs)
        >>> for seq in result.sequences:
        ...     print(seq.aligned.hex())

    References:
        FUZZY-005: Multiple Binary Sequence Alignment (MSA)
    """
    # Validate method parameter first
    if method not in ("progressive", "iterative"):
        raise ValueError(f"Unknown alignment method: {method}")

    if not sequences:
        return AlignmentResult(
            sequences=[],
            conservation_scores=[],
            conserved_regions=[],
            gap_positions=[],
            alignment_score=0.0,
        )

    if len(sequences) == 1:
        return AlignmentResult(
            sequences=[
                AlignedSequence(
                    original=sequences[0],
                    aligned=sequences[0],
                    gaps=[],
                    score=0.0,
                )
            ],
            conservation_scores=[1.0] * len(sequences[0]),
            conserved_regions=[(0, len(sequences[0]) - 1)] if sequences[0] else [],
            gap_positions=[],
            alignment_score=0.0,
        )

    if method == "progressive":
        return _progressive_alignment(sequences, gap_open, gap_extend)
    elif method == "iterative":
        # Iterative refinement (simplified)
        result = _progressive_alignment(sequences, gap_open, gap_extend)
        # Could add refinement passes here
        return result
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _progressive_alignment(
    sequences: Sequence[bytes],
    gap_open: int,
    gap_extend: int,
) -> AlignmentResult:
    """Progressive multiple sequence alignment."""
    # Align sequences progressively
    aligned_seqs, total_score = _align_progressively(sequences, gap_open, gap_extend)

    # Build result sequences
    result_seqs = _build_result_sequences(sequences, aligned_seqs)

    # Analyze alignment
    alignment_length = len(aligned_seqs[0]) if aligned_seqs else 0
    conservation = compute_conservation_scores(aligned_seqs)
    conserved_regions = _find_conserved_regions(conservation)
    gap_positions = _find_gap_positions(aligned_seqs, alignment_length)

    return AlignmentResult(
        sequences=result_seqs,
        conservation_scores=conservation,
        conserved_regions=conserved_regions,
        gap_positions=gap_positions,
        alignment_score=total_score,
    )


def _align_progressively(
    sequences: Sequence[bytes],
    gap_open: int,
    gap_extend: int,
) -> tuple[list[bytes], float]:
    """Align sequences progressively.

    Args:
        sequences: Input sequences.
        gap_open: Gap opening penalty.
        gap_extend: Gap extension penalty.

    Returns:
        Tuple of (aligned sequences, total score).
    """
    aligned_seqs: list[bytes] = [sequences[0]]
    total_score = 0.0

    for seq in sequences[1:]:
        # Align to reference
        ref_aligned, seq_aligned, score = _needleman_wunsch(
            aligned_seqs[0],
            seq,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )

        # Update existing alignments
        new_aligned = _update_alignments(aligned_seqs, ref_aligned)
        aligned_seqs = new_aligned
        aligned_seqs.append(seq_aligned)
        total_score += score

    return aligned_seqs, total_score


def _update_alignments(aligned_seqs: list[bytes], ref_aligned: bytes) -> list[bytes]:
    """Update existing alignments to match reference.

    Args:
        aligned_seqs: Current aligned sequences.
        ref_aligned: New reference alignment.

    Returns:
        Updated aligned sequences.
    """
    new_aligned: list[bytes] = []
    for prev in aligned_seqs:
        new_prev: list[int] = []
        prev_idx = 0
        for byte in ref_aligned:
            if byte == GAP_BYTE:
                new_prev.append(GAP_BYTE)
            elif prev_idx < len(prev):
                new_prev.append(prev[prev_idx])
                prev_idx += 1
        new_aligned.append(bytes(new_prev))
    return new_aligned


def _build_result_sequences(
    sequences: Sequence[bytes],
    aligned_seqs: list[bytes],
) -> list[AlignedSequence]:
    """Build result sequences from alignment.

    Args:
        sequences: Original sequences.
        aligned_seqs: Aligned sequences.

    Returns:
        List of AlignedSequence objects.
    """
    result_seqs: list[AlignedSequence] = []
    for orig, aligned in zip(sequences, aligned_seqs, strict=False):
        gaps = [i for i, b in enumerate(aligned) if b == GAP_BYTE]
        result_seqs.append(
            AlignedSequence(
                original=orig,
                aligned=aligned,
                gaps=gaps,
                score=0.0,
            )
        )
    return result_seqs


def _find_conserved_regions(conservation: list[float]) -> list[tuple[int, int]]:
    """Find conserved regions in alignment.

    Args:
        conservation: Conservation scores.

    Returns:
        List of (start, end) tuples.
    """
    conserved_regions: list[tuple[int, int]] = []
    in_region = False
    region_start = 0

    for i, score in enumerate(conservation):
        if score >= 0.8 and not in_region:
            in_region = True
            region_start = i
        elif score < 0.8 and in_region:
            in_region = False
            conserved_regions.append((region_start, i - 1))

    if in_region:
        conserved_regions.append((region_start, len(conservation) - 1))

    return conserved_regions


def _find_gap_positions(aligned_seqs: list[bytes], alignment_length: int) -> list[int]:
    """Find positions with common gaps.

    Args:
        aligned_seqs: Aligned sequences.
        alignment_length: Length of alignment.

    Returns:
        List of gap positions.
    """
    gap_positions: list[int] = []
    for pos in range(alignment_length):
        gap_count = sum(1 for seq in aligned_seqs if seq[pos] == GAP_BYTE)
        if gap_count > len(aligned_seqs) // 2:
            gap_positions.append(pos)
    return gap_positions


def compute_conservation_scores(
    aligned_sequences: list[bytes],
) -> list[float]:
    """Compute per-position conservation scores.

    Conservation score at position i = frequency of most common byte
    (1.0 = fully conserved, <0.5 = poorly conserved).

    Args:
        aligned_sequences: Aligned sequences (same length)

    Returns:
        List of conservation scores

    References:
        FUZZY-005: Multiple Binary Sequence Alignment (MSA)
    """
    if not aligned_sequences:
        return []

    alignment_length = len(aligned_sequences[0])
    len(aligned_sequences)
    scores: list[float] = []

    for pos in range(alignment_length):
        # Count byte frequencies (excluding gaps)
        counts: dict[int, int] = {}
        non_gap_count = 0

        for seq in aligned_sequences:
            byte = seq[pos]
            if byte != GAP_BYTE:
                counts[byte] = counts.get(byte, 0) + 1
                non_gap_count += 1

        if non_gap_count == 0:
            scores.append(0.0)
        else:
            max_count = max(counts.values())
            scores.append(max_count / non_gap_count)

    return scores
