"""Pattern search in digital traces.


This module provides efficient bit pattern matching in digital signals
with wildcard support via mask parameter.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray


def find_pattern(
    trace: NDArray[np.float64] | NDArray[np.uint8],
    pattern: int | NDArray[np.uint8],
    mask: int | NDArray[np.uint8] | None = None,
    *,
    threshold: float | None = None,
    min_spacing: int = 1,
) -> list[tuple[int, NDArray[np.uint8]]]:
    """Find occurrences of bit patterns in digital traces.

    : Pattern search with wildcard support via mask.
    Works on both raw analog traces (with threshold) and decoded digital data.

    Args:
        trace: Input trace array. If analog (float), threshold is required.
            If already digital (uint8), threshold is ignored.
        pattern: Bit pattern to search for. Can be:
            - Integer: e.g., 0b10101010 (8-bit pattern)
            - Array: sequence of bytes to match
        mask: Optional mask for wildcard matching. Bits set to 0 in mask
            are "don't care" positions. Can be:
            - Integer: e.g., 0xFF (all bits matter)
            - Array: per-byte masks
            If None, all bits must match (equivalent to all 1s).
        threshold: Threshold for converting analog to digital (required if
            trace is analog). Typically mid-level of logic family.
        min_spacing: Minimum samples between detected patterns to avoid
            overlapping matches (default: 1)

    Returns:
        List of (index, match) tuples where:
        - index: Starting sample index of the pattern
        - match: The actual matched bit sequence as uint8 array

    Raises:
        ValueError: If analog trace provided without threshold
        ValueError: If pattern is empty

    Examples:
        >>> # Find 0xAA pattern in analog trace
        >>> import numpy as np
        >>> trace = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        >>> matches = find_pattern(trace, 0b10101010, threshold=0.5)
        >>> print(f"Found {len(matches)} matches")

        >>> # Wildcard search: find 0b1010xxxx (x = don't care)
        >>> pattern = 0b10100000
        >>> mask = 0b11110000  # Only upper 4 bits matter
        >>> matches = find_pattern(trace, pattern, mask, threshold=0.5)

        >>> # Search in already-decoded digital data
        >>> digital = np.array([0xAA, 0x55, 0xAA, 0x00], dtype=np.uint8)
        >>> matches = find_pattern(digital, 0xAA)

    Notes:
        - For analog traces, values >= threshold are interpreted as '1'
        - Mask bits: 1 = must match, 0 = don't care
        - Overlapping patterns can be filtered with min_spacing > 1
        - Returns empty list if no matches found

    References:
        SRCH-001: Pattern Search
    """
    if trace.size == 0:
        return []

    # Phase 1: Input normalization
    pattern_arr = _normalize_pattern(pattern)
    mask_arr = _normalize_mask(mask, pattern_arr)

    # Phase 2: Convert trace to digital format
    digital_packed = _convert_trace_to_digital(trace, threshold)

    if digital_packed.size < pattern_arr.size:
        return []

    # Phase 3: Sliding window search with mask
    return _sliding_window_search(digital_packed, pattern_arr, mask_arr, min_spacing)


def _normalize_pattern(pattern: int | NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Normalize pattern input to numpy array.

    Args:
        pattern: Pattern as integer or array.

    Returns:
        Pattern as uint8 numpy array.

    Raises:
        ValueError: If pattern is negative or empty.

    Example:
        >>> _normalize_pattern(0xAA)
        array([170], dtype=uint8)
        >>> _normalize_pattern(0x1234)
        array([18, 52], dtype=uint8)
    """
    if isinstance(pattern, int):
        if pattern < 0:
            raise ValueError("Pattern must be non-negative")
        # Convert to byte array (variable length based on value)
        pattern_bytes = []
        if pattern == 0:
            pattern_bytes = [0]
        else:
            temp = pattern
            while temp > 0:
                pattern_bytes.insert(0, temp & 0xFF)
                temp >>= 8
        pattern_arr = np.array(pattern_bytes, dtype=np.uint8)
    else:
        pattern_arr = np.asarray(pattern, dtype=np.uint8)

    if pattern_arr.size == 0:
        raise ValueError("Pattern cannot be empty")

    return pattern_arr


def _normalize_mask(
    mask: int | NDArray[np.uint8] | None, pattern_arr: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """Normalize mask input to numpy array matching pattern length.

    Args:
        mask: Mask as integer, array, or None.
        pattern_arr: Pattern array to match length to.

    Returns:
        Mask as uint8 numpy array with same length as pattern.

    Raises:
        ValueError: If mask and pattern have different lengths.

    Example:
        >>> pattern = np.array([0xAA, 0x55], dtype=np.uint8)
        >>> _normalize_mask(0xFF, pattern)
        array([255, 255], dtype=uint8)
        >>> _normalize_mask(None, pattern)
        array([255, 255], dtype=uint8)
    """
    if mask is not None:
        if isinstance(mask, int):
            mask_bytes: list[int] = []
            temp = mask
            # Match pattern length
            for _ in range(len(pattern_arr)):
                mask_bytes.insert(0, temp & 0xFF)
                temp >>= 8
            mask_arr = np.array(mask_bytes, dtype=np.uint8)
        else:
            mask_arr = np.asarray(mask, dtype=np.uint8)

        # Ensure mask and pattern have same length
        if mask_arr.size != pattern_arr.size:
            raise ValueError("Mask and pattern must have same length")
    else:
        # Default: all bits matter
        mask_arr = np.full(pattern_arr.size, 0xFF, dtype=np.uint8)

    return mask_arr


def _convert_trace_to_digital(
    trace: NDArray[np.float64] | NDArray[np.uint8], threshold: float | None
) -> NDArray[np.uint8]:
    """Convert trace to digital packed format.

    If trace is already digital (uint8), returns as-is.
    If trace is analog (float), converts using threshold and packs bits.

    Args:
        trace: Input trace (analog or digital).
        threshold: Threshold for analog-to-digital conversion.

    Returns:
        Digital trace as packed uint8 array.

    Raises:
        ValueError: If analog trace provided without threshold.

    Example:
        >>> analog = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        >>> _convert_trace_to_digital(analog, 0.5)
        array([170], dtype=uint8)  # 0b10101010 = 0xAA
    """
    if trace.dtype != np.uint8:
        if threshold is None:
            raise ValueError(
                "Threshold required for analog trace conversion. "
                "Provide threshold parameter or pre-convert to digital."
            )
        # Simple threshold conversion: >= threshold is 1
        digital = (trace >= threshold).astype(np.uint8)
        # Pack bits into bytes (8 samples per byte)
        # Pad to multiple of 8
        n_pad = (8 - len(digital) % 8) % 8
        if n_pad:
            digital = np.pad(digital, (0, n_pad), constant_values=0)
        # Pack bits
        digital_packed: NDArray[np.uint8] = np.packbits(digital, bitorder="big")
    else:
        digital_packed = cast("NDArray[np.uint8]", trace)

    return digital_packed


def _sliding_window_search(
    digital_packed: NDArray[np.uint8],
    pattern_arr: NDArray[np.uint8],
    mask_arr: NDArray[np.uint8],
    min_spacing: int,
) -> list[tuple[int, NDArray[np.uint8]]]:
    """Perform sliding window pattern matching with mask.

    Args:
        digital_packed: Digital trace data.
        pattern_arr: Pattern to search for.
        mask_arr: Mask for wildcard matching.
        min_spacing: Minimum spacing between matches.

    Returns:
        List of (index, matched_data) tuples.

    Example:
        >>> data = np.array([0xAA, 0x55, 0xAA], dtype=np.uint8)
        >>> pattern = np.array([0xAA], dtype=np.uint8)
        >>> mask = np.array([0xFF], dtype=np.uint8)
        >>> _sliding_window_search(data, pattern, mask, 1)
        [(0, array([170], dtype=uint8)), (2, array([170], dtype=uint8))]
    """
    matches: list[tuple[int, NDArray[np.uint8]]] = []
    i = 0

    while i <= len(digital_packed) - len(pattern_arr):
        window = digital_packed[i : i + len(pattern_arr)]

        # Apply mask and compare
        if _matches_pattern(window, pattern_arr, mask_arr):
            # NECESSARY COPY: window is a numpy view that gets reused.
            # Without .copy(), all matches would reference same memory location.
            # Could optimize: store indices + lazy evaluation instead of copies.
            matches.append((i, window.copy()))
            # Skip ahead by min_spacing to avoid overlapping matches
            i += max(1, min_spacing)
        else:
            i += 1

    return matches


def _matches_pattern(
    window: NDArray[np.uint8], pattern: NDArray[np.uint8], mask: NDArray[np.uint8]
) -> bool:
    """Check if window matches pattern with mask.

    Args:
        window: Data window to check.
        pattern: Pattern to match.
        mask: Mask for wildcard positions.

    Returns:
        True if window matches pattern under mask.

    Example:
        >>> window = np.array([0xA5], dtype=np.uint8)
        >>> pattern = np.array([0xA0], dtype=np.uint8)
        >>> mask = np.array([0xF0], dtype=np.uint8)
        >>> _matches_pattern(window, pattern, mask)
        True
    """
    masked_window = window & mask
    masked_pattern = pattern & mask
    return bool(np.array_equal(masked_window, masked_pattern))
