"""Sequence alignment algorithms for binary message comparison.

Requirements addressed: PSI-003

This module applies sequence alignment algorithms to compare binary messages
for identifying common structures and variations.

Key capabilities:
- Needleman-Wunsch for global alignment
- Smith-Waterman for local alignment
- Multiple sequence alignment
- Conserved/variable region detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class AlignmentResult:
    """Result of sequence alignment.

    : Alignment result representation.

    Attributes:
        aligned_a: Aligned sequence A (with gaps as -1)
        aligned_b: Aligned sequence B (with gaps as -1)
        score: Alignment score
        similarity: Similarity ratio (0-1)
        identity: Fraction of identical positions
        gaps: Number of gap positions
        conserved_regions: List of (start, end) tuples for conserved regions
        variable_regions: List of (start, end) tuples for variable regions
    """

    aligned_a: bytes | list[int]  # Aligned sequence A (with gaps as -1)
    aligned_b: bytes | list[int]  # Aligned sequence B (with gaps as -1)
    score: float
    similarity: float  # 0-1
    identity: float  # Fraction of identical positions
    gaps: int  # Number of gap positions
    conserved_regions: list[tuple[int, int]]  # (start, end) of conserved regions
    variable_regions: list[tuple[int, int]]  # (start, end) of variable regions


def align_global(
    seq_a: bytes | NDArray[Any],
    seq_b: bytes | NDArray[Any],
    gap_penalty: float = -1.0,
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
) -> AlignmentResult:
    """Global alignment using Needleman-Wunsch algorithm.

    : Needleman-Wunsch global alignment (O(mn) complexity).

    Args:
        seq_a: First sequence (bytes or array)
        seq_b: Second sequence (bytes or array)
        gap_penalty: Penalty for gaps
        match_score: Score for matching positions
        mismatch_penalty: Penalty for mismatches

    Returns:
        AlignmentResult with aligned sequences and statistics
    """
    # Convert to arrays
    if isinstance(seq_a, bytes):
        arr_a = np.frombuffer(seq_a, dtype=np.uint8)
    else:
        arr_a = np.array(seq_a, dtype=np.uint8)

    if isinstance(seq_b, bytes):
        arr_b = np.frombuffer(seq_b, dtype=np.uint8)
    else:
        arr_b = np.array(seq_b, dtype=np.uint8)

    n, m = len(arr_a), len(arr_b)

    # Initialize scoring matrix and traceback matrix
    score_matrix = np.zeros((n + 1, m + 1), dtype=np.float32)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Initialize first row and column with gap penalties
    for i in range(1, n + 1):
        score_matrix[i, 0] = i * gap_penalty
        traceback[i, 0] = 1  # Up (gap in seq_b)

    for j in range(1, m + 1):
        score_matrix[0, j] = j * gap_penalty
        traceback[0, j] = 2  # Left (gap in seq_a)

    # Fill the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Match/mismatch
            if arr_a[i - 1] == arr_b[j - 1]:
                diag_score = score_matrix[i - 1, j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1, j - 1] + mismatch_penalty

            # Gap in seq_b (up)
            up_score = score_matrix[i - 1, j] + gap_penalty

            # Gap in seq_a (left)
            left_score = score_matrix[i, j - 1] + gap_penalty

            # Choose best
            max_score = max(diag_score, up_score, left_score)
            score_matrix[i, j] = max_score

            if max_score == diag_score:
                traceback[i, j] = 0  # Diagonal
            elif max_score == up_score:
                traceback[i, j] = 1  # Up
            else:
                traceback[i, j] = 2  # Left

    # Traceback to get alignment
    aligned_a, aligned_b = _traceback_alignment(traceback, arr_a, arr_b, n, m)

    # Calculate statistics
    final_score = float(score_matrix[n, m])
    similarity = compute_similarity(aligned_a, aligned_b)
    identity, gaps = _calculate_alignment_stats(aligned_a, aligned_b)

    # Find conserved and variable regions
    conserved = _find_conserved_simple(aligned_a, aligned_b)
    variable = _find_variable_simple(aligned_a, aligned_b)

    return AlignmentResult(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        score=final_score,
        similarity=similarity,
        identity=identity,
        gaps=gaps,
        conserved_regions=conserved,
        variable_regions=variable,
    )


def align_local(
    seq_a: bytes | NDArray[Any],
    seq_b: bytes | NDArray[Any],
    gap_penalty: float = -1.0,
    match_score: float = 2.0,
    mismatch_penalty: float = -1.0,
) -> AlignmentResult:
    """Local alignment using Smith-Waterman algorithm.

    : Smith-Waterman local alignment (O(mn) complexity).

    Args:
        seq_a: First sequence
        seq_b: Second sequence
        gap_penalty: Penalty for gaps
        match_score: Score for matches
        mismatch_penalty: Penalty for mismatches

    Returns:
        AlignmentResult with best local alignment
    """
    arr_a, arr_b = _convert_to_arrays(seq_a, seq_b)
    score_matrix, traceback, max_score, max_pos = _build_sw_matrix(
        arr_a, arr_b, gap_penalty, match_score, mismatch_penalty
    )
    aligned_a, aligned_b = _traceback_local(traceback, arr_a, arr_b, max_pos)
    similarity, identity, gaps = _compute_local_stats(aligned_a, aligned_b)
    conserved = _find_conserved_simple(aligned_a, aligned_b)
    variable = _find_variable_simple(aligned_a, aligned_b)

    return AlignmentResult(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        score=float(max_score),
        similarity=similarity,
        identity=identity,
        gaps=gaps,
        conserved_regions=conserved,
        variable_regions=variable,
    )


def align_multiple(
    sequences: list[bytes | NDArray[Any]],
    method: Literal["progressive", "iterative"] = "progressive",
) -> list[list[int]]:
    """Multiple sequence alignment.

    : Progressive MSA using guide tree and pairwise alignment.

    Args:
        sequences: List of sequences (bytes or arrays)
        method: Alignment method ('progressive' or 'iterative')

    Returns:
        List of aligned sequences (as lists with -1 for gaps)
    """
    if len(sequences) == 0:
        return []
    if len(sequences) == 1:
        # Convert to list
        if isinstance(sequences[0], bytes):
            return [list(np.frombuffer(sequences[0], dtype=np.uint8))]
        else:
            return [list(sequences[0])]

    # Progressive alignment
    if method == "progressive":
        # Start with first two sequences
        result = align_global(sequences[0], sequences[1])
        # Convert to list[int] if needed
        aligned_a_list = (
            list(result.aligned_a) if isinstance(result.aligned_a, bytes) else result.aligned_a
        )
        aligned_b_list = (
            list(result.aligned_b) if isinstance(result.aligned_b, bytes) else result.aligned_b
        )
        aligned: list[list[int]] = [aligned_a_list, aligned_b_list]

        # Add remaining sequences one by one
        for seq in sequences[2:]:
            # Align seq to consensus of current alignment
            consensus_seq = _compute_consensus(aligned)
            consensus_bytes = bytes([v if v != -1 else 0 for v in consensus_seq])
            result = align_global(consensus_bytes, seq)

            # Insert gaps in existing alignments
            new_aligned: list[list[int]] = []
            result_a_list = (
                list(result.aligned_a) if isinstance(result.aligned_a, bytes) else result.aligned_a
            )
            for existing in aligned:
                new_seq = _insert_gaps_from_alignment(existing, result_a_list)
                new_aligned.append(new_seq)

            # Add new sequence
            result_b_list = (
                list(result.aligned_b) if isinstance(result.aligned_b, bytes) else result.aligned_b
            )
            new_aligned.append(result_b_list)
            aligned = new_aligned

        return aligned
    else:
        # Iterative not implemented, fall back to progressive
        return align_multiple(sequences, method="progressive")


def compute_similarity(aligned_a: bytes | list[int], aligned_b: bytes | list[int]) -> float:
    """Compute similarity between aligned sequences.

    : Similarity calculation.

    Args:
        aligned_a: First aligned sequence
        aligned_b: Second aligned sequence

    Returns:
        Similarity ratio (0-1)

    Raises:
        ValueError: If aligned sequences have different lengths.
    """
    if len(aligned_a) != len(aligned_b):
        raise ValueError("Aligned sequences must have same length")

    if len(aligned_a) == 0:
        return 0.0

    matches = 0
    total = 0

    for a, b in zip(aligned_a, aligned_b, strict=True):
        # Skip double gaps
        if a == -1 and b == -1:
            continue

        total += 1
        if a == b and a != -1:
            matches += 1

    if total == 0:
        return 0.0

    return matches / total


def find_conserved_regions(
    aligned_sequences: list[list[int]], min_conservation: float = 0.9, min_length: int = 4
) -> list[tuple[int, int]]:
    """Find highly conserved regions in aligned sequences.

    : Conserved region detection.

    Args:
        aligned_sequences: List of aligned sequences
        min_conservation: Minimum conservation ratio (0-1)
        min_length: Minimum region length

    Returns:
        List of (start, end) tuples for conserved regions
    """
    if not aligned_sequences:
        return []

    length = len(aligned_sequences[0])
    _num_seqs = len(aligned_sequences)

    # Calculate conservation at each position
    conservation = []
    for pos in range(length):
        values = [seq[pos] for seq in aligned_sequences if pos < len(seq)]

        # Skip gaps
        non_gap_values = [v for v in values if v != -1]

        if len(non_gap_values) == 0:
            conservation.append(0.0)
            continue

        # Count most common value
        from collections import Counter

        counts = Counter(non_gap_values)
        most_common_count = counts.most_common(1)[0][1]

        cons = most_common_count / len(non_gap_values)
        conservation.append(cons)

    # Find regions above threshold
    regions = []
    start = None

    for i, cons in enumerate(conservation):
        if cons >= min_conservation:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_length:
                    regions.append((start, i))
                start = None

    # Handle region at end
    if start is not None and length - start >= min_length:
        regions.append((start, length))

    return regions


def find_variable_regions(
    aligned_sequences: list[list[int]], max_conservation: float = 0.5, min_length: int = 2
) -> list[tuple[int, int]]:
    """Find highly variable regions in aligned sequences.

    : Variable region detection.

    Args:
        aligned_sequences: List of aligned sequences
        max_conservation: Maximum conservation ratio (0-1)
        min_length: Minimum region length

    Returns:
        List of (start, end) tuples for variable regions
    """
    if not aligned_sequences:
        return []

    length = len(aligned_sequences[0])

    # Calculate conservation at each position
    conservation = []
    for pos in range(length):
        values = [seq[pos] for seq in aligned_sequences if pos < len(seq)]

        # Skip gaps
        non_gap_values = [v for v in values if v != -1]

        if len(non_gap_values) == 0:
            conservation.append(1.0)  # All gaps = conserved
            continue

        # Count most common value
        from collections import Counter

        counts = Counter(non_gap_values)
        most_common_count = counts.most_common(1)[0][1]

        cons = most_common_count / len(non_gap_values)
        conservation.append(cons)

    # Find regions below threshold
    regions = []
    start = None

    for i, cons in enumerate(conservation):
        if cons <= max_conservation:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_length:
                    regions.append((start, i))
                start = None

    # Handle region at end
    if start is not None and length - start >= min_length:
        regions.append((start, length))

    return regions


def _convert_to_arrays(
    seq_a: bytes | NDArray[Any], seq_b: bytes | NDArray[Any]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Convert input sequences to numpy arrays.

    Args:
        seq_a: First sequence (bytes or array).
        seq_b: Second sequence (bytes or array).

    Returns:
        Tuple of (arr_a, arr_b) as uint8 arrays.
    """
    if isinstance(seq_a, bytes):
        arr_a = np.frombuffer(seq_a, dtype=np.uint8)
    else:
        arr_a = np.array(seq_a, dtype=np.uint8)

    if isinstance(seq_b, bytes):
        arr_b = np.frombuffer(seq_b, dtype=np.uint8)
    else:
        arr_b = np.array(seq_b, dtype=np.uint8)

    return arr_a, arr_b


def _build_sw_matrix(
    arr_a: NDArray[np.uint8],
    arr_b: NDArray[np.uint8],
    gap_penalty: float,
    match_score: float,
    mismatch_penalty: float,
) -> tuple[NDArray[np.float32], NDArray[np.int8], float, tuple[int, int]]:
    """Build Smith-Waterman scoring and traceback matrices.

    Args:
        arr_a: First sequence array.
        arr_b: Second sequence array.
        gap_penalty: Penalty for gaps.
        match_score: Score for matches.
        mismatch_penalty: Penalty for mismatches.

    Returns:
        (score_matrix, traceback, max_score, max_position).
    """
    n, m = len(arr_a), len(arr_b)
    score_matrix = np.zeros((n + 1, m + 1), dtype=np.float32)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)
    max_score = 0.0
    max_i, max_j = 0, 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Match/mismatch
            if arr_a[i - 1] == arr_b[j - 1]:
                diag_score = score_matrix[i - 1, j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1, j - 1] + mismatch_penalty

            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty

            # Smith-Waterman: can start fresh (score = 0)
            cell_score = max(0.0, diag_score, up_score, left_score)
            score_matrix[i, j] = cell_score

            if cell_score == 0:
                traceback[i, j] = -1  # Stop
            elif cell_score == diag_score:
                traceback[i, j] = 0  # Diagonal
            elif cell_score == up_score:
                traceback[i, j] = 1  # Up
            else:
                traceback[i, j] = 2  # Left

            if cell_score > max_score:
                max_score = cell_score
                max_i, max_j = i, j

    return score_matrix, traceback, max_score, (max_i, max_j)


def _traceback_local(
    traceback: NDArray[np.int8],
    arr_a: NDArray[np.uint8],
    arr_b: NDArray[np.uint8],
    max_pos: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """Perform traceback from max position for local alignment.

    Args:
        traceback: Traceback matrix.
        arr_a: First sequence array.
        arr_b: Second sequence array.
        max_pos: (i, j) position of maximum score.

    Returns:
        Tuple of (aligned_a, aligned_b) with -1 for gaps.
    """
    aligned_a = []
    aligned_b = []

    i, j = max_pos
    while i > 0 and j > 0 and traceback[i, j] != -1:
        if traceback[i, j] == 0:  # Diagonal
            aligned_a.append(int(arr_a[i - 1]))
            aligned_b.append(int(arr_b[j - 1]))
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:  # Up
            aligned_a.append(int(arr_a[i - 1]))
            aligned_b.append(-1)  # Gap
            i -= 1
        else:  # Left
            aligned_a.append(-1)  # Gap
            aligned_b.append(int(arr_b[j - 1]))
            j -= 1

    return list(reversed(aligned_a)), list(reversed(aligned_b))


def _compute_local_stats(aligned_a: list[int], aligned_b: list[int]) -> tuple[float, float, int]:
    """Compute statistics for local alignment.

    Args:
        aligned_a: First aligned sequence.
        aligned_b: Second aligned sequence.

    Returns:
        (similarity, identity, gaps).
    """
    if len(aligned_a) == 0:
        return 0.0, 0.0, 0

    similarity = compute_similarity(aligned_a, aligned_b)
    identity = sum(
        1 for a, b in zip(aligned_a, aligned_b, strict=True) if a == b and a != -1
    ) / len(aligned_a)
    gaps = sum(1 for a, b in zip(aligned_a, aligned_b, strict=True) if a == -1 or b == -1)

    return similarity, identity, gaps


def _traceback_alignment(
    traceback: NDArray[np.int8],
    arr_a: NDArray[np.uint8],
    arr_b: NDArray[np.uint8],
    n: int,
    m: int,
) -> tuple[list[int], list[int]]:
    """Perform traceback to extract aligned sequences.

    Args:
        traceback: Traceback matrix (0=diagonal, 1=up, 2=left).
        arr_a: First sequence array.
        arr_b: Second sequence array.
        n: Length of first sequence.
        m: Length of second sequence.

    Returns:
        Tuple of (aligned_a, aligned_b) with -1 for gaps.
    """
    aligned_a = []
    aligned_b = []

    i, j = n, m
    while i > 0 or j > 0:
        if traceback[i, j] == 0:  # Diagonal (match/mismatch)
            aligned_a.append(int(arr_a[i - 1]))
            aligned_b.append(int(arr_b[j - 1]))
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:  # Up (gap in seq_b)
            aligned_a.append(int(arr_a[i - 1]))
            aligned_b.append(-1)  # Gap
            i -= 1
        else:  # Left (gap in seq_a)
            aligned_a.append(-1)  # Gap
            aligned_b.append(int(arr_b[j - 1]))
            j -= 1

    # Reverse (we traced backwards)
    return list(reversed(aligned_a)), list(reversed(aligned_b))


def _calculate_alignment_stats(
    aligned_a: list[int],
    aligned_b: list[int],
) -> tuple[float, int]:
    """Calculate identity and gap statistics for alignment.

    Args:
        aligned_a: First aligned sequence (-1 for gaps).
        aligned_b: Second aligned sequence (-1 for gaps).

    Returns:
        Tuple of (identity, gaps).
    """
    if len(aligned_a) == 0:
        return 0.0, 0

    identity = sum(
        1 for a, b in zip(aligned_a, aligned_b, strict=True) if a == b and a != -1
    ) / len(aligned_a)
    gaps = sum(1 for a, b in zip(aligned_a, aligned_b, strict=True) if a == -1 or b == -1)

    return identity, gaps


def _find_conserved_simple(aligned_a: list[int], aligned_b: list[int]) -> list[tuple[int, int]]:
    """Find conserved regions in pairwise alignment.

    Args:
        aligned_a: First aligned sequence
        aligned_b: Second aligned sequence

    Returns:
        List of (start, end) tuples
    """
    regions = []
    start = None

    for i, (a, b) in enumerate(zip(aligned_a, aligned_b, strict=True)):
        if a == b and a != -1:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= 4:  # Min length 4
                    regions.append((start, i))
                start = None

    # Handle region at end
    if start is not None and len(aligned_a) - start >= 4:
        regions.append((start, len(aligned_a)))

    return regions


def _find_variable_simple(aligned_a: list[int], aligned_b: list[int]) -> list[tuple[int, int]]:
    """Find variable regions in pairwise alignment.

    Args:
        aligned_a: First aligned sequence
        aligned_b: Second aligned sequence

    Returns:
        List of (start, end) tuples
    """
    regions = []
    start = None

    for i, (a, b) in enumerate(zip(aligned_a, aligned_b, strict=True)):
        if a != b:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= 2:  # Min length 2
                    regions.append((start, i))
                start = None

    # Handle region at end
    if start is not None and len(aligned_a) - start >= 2:
        regions.append((start, len(aligned_a)))

    return regions


def _compute_consensus(aligned_sequences: list[list[int]]) -> list[int]:
    """Compute consensus sequence from multiple aligned sequences.

    Args:
        aligned_sequences: List of aligned sequences

    Returns:
        Consensus sequence
    """
    if not aligned_sequences:
        return []

    length = max(len(seq) for seq in aligned_sequences)
    consensus = []

    for pos in range(length):
        values = [seq[pos] for seq in aligned_sequences if pos < len(seq)]

        # Skip gaps when computing consensus
        non_gap_values = [v for v in values if v != -1]

        if non_gap_values:
            # Most common value
            from collections import Counter

            counts = Counter(non_gap_values)
            consensus_val = counts.most_common(1)[0][0]
            consensus.append(consensus_val)
        else:
            # All gaps
            consensus.append(-1)

    return consensus


def _insert_gaps_from_alignment(sequence: list[int], alignment_template: list[int]) -> list[int]:
    """Insert gaps into sequence based on alignment template.

    Args:
        sequence: Original sequence
        alignment_template: Template showing where gaps should be

    Returns:
        Sequence with gaps inserted
    """
    result = []
    seq_idx = 0

    for template_val in alignment_template:
        if template_val == -1:
            # Gap in template, insert gap
            result.append(-1)
        else:
            # Non-gap, copy from sequence
            if seq_idx < len(sequence):
                result.append(sequence[seq_idx])
                seq_idx += 1
            else:
                result.append(-1)

    return result
