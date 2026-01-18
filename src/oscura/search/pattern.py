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

    # Convert pattern to array if integer
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

    # Convert mask to array if integer
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

    # Convert analog trace to digital if needed
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

    if digital_packed.size < pattern_arr.size:
        return []

    # Sliding window pattern matching with mask
    matches: list[tuple[int, NDArray[np.uint8]]] = []
    i = 0

    while i <= len(digital_packed) - len(pattern_arr):
        window = digital_packed[i : i + len(pattern_arr)]

        # Apply mask and compare
        masked_window = window & mask_arr
        masked_pattern = pattern_arr & mask_arr

        if np.array_equal(masked_window, masked_pattern):
            matches.append((i, window.copy()))
            # Skip ahead by min_spacing to avoid overlapping matches
            i += max(1, min_spacing)
        else:
            i += 1

    return matches
