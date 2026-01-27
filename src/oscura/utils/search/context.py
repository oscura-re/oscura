"""Context extraction around points of interest.


This module provides efficient extraction of signal context around
events, maintaining original time references for debugging workflows.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def extract_context(
    trace: NDArray[np.float64],
    index: int | list[int] | NDArray[np.int_],
    *,
    before: int = 100,
    after: int = 100,
    sample_rate: float | None = None,
    include_metadata: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract signal context around a point of interest.

    : Context extraction with time reference preservation.
    Supports batch extraction for multiple indices and optional protocol data.

    Args:
        trace: Input signal trace
        index: Sample index or list of indices to extract context around.
            Can be int, list of ints, or numpy array.
        before: Number of samples to include before index (default: 100)
        after: Number of samples to include after index (default: 100)
        sample_rate: Optional sample rate in Hz for time calculations
        include_metadata: Include metadata dict with context info (default: True)

    Returns:
        If index is scalar: Single context dictionary
        If index is list/array: List of context dictionaries

        Each context dictionary contains:
        - data: Extracted sub-trace array
        - start_index: Starting index in original trace
        - end_index: Ending index in original trace
        - center_index: Center index (original query index)
        - time_reference: Time offset if sample_rate provided
        - length: Number of samples in context

    Raises:
        ValueError: If index is out of bounds
        ValueError: If before or after are negative

    Examples:
        >>> # Extract context around a glitch
        >>> trace = np.random.randn(1000)
        >>> glitch_index = 500
        >>> context = extract_context(
        ...     trace,
        ...     glitch_index,
        ...     before=50,
        ...     after=50,
        ...     sample_rate=1e6
        ... )
        >>> print(f"Context length: {len(context['data'])}")
        >>> print(f"Time reference: {context['time_reference']*1e6:.2f} µs")

        >>> # Batch extraction for multiple events
        >>> event_indices = [100, 200, 300]
        >>> contexts = extract_context(
        ...     trace,
        ...     event_indices,
        ...     before=25,
        ...     after=25
        ... )
        >>> print(f"Extracted {len(contexts)} contexts")

    Notes:
        - Handles edge cases at trace boundaries automatically
        - Context may be shorter than before+after at boundaries
        - Time reference is relative to start of extracted context
        - Original trace is not modified

    References:
        SRCH-003: Context Extraction
    """
    # Phase 1: Input validation
    _validate_context_params(before, after, trace)

    # Phase 2: Normalize indices
    indices, return_single = _normalize_indices(index, trace)

    # Phase 3: Extract contexts
    contexts = [
        _extract_single_context(trace, idx, before, after, sample_rate, include_metadata)
        for idx in indices
    ]

    # Return single context or list
    return contexts[0] if return_single else contexts


def _validate_context_params(before: int, after: int, trace: NDArray[np.float64]) -> None:
    """Validate context extraction parameters.

    Args:
        before: Samples before index.
        after: Samples after index.
        trace: Input trace.

    Raises:
        ValueError: If parameters are invalid.

    Example:
        >>> trace = np.array([1.0, 2.0, 3.0])
        >>> _validate_context_params(10, 10, trace)
        >>> _validate_context_params(-1, 10, trace)
        Traceback (most recent call last):
        ValueError: before and after must be non-negative
    """
    if before < 0 or after < 0:
        raise ValueError("before and after must be non-negative")

    if trace.size == 0:
        raise ValueError("Trace cannot be empty")


def _normalize_indices(
    index: int | list[int] | NDArray[np.int_], trace: NDArray[np.float64]
) -> tuple[list[int], bool]:
    """Normalize index input to list of integers.

    Args:
        index: Input index (int, list, or array).
        trace: Trace to validate against.

    Returns:
        Tuple of (normalized_indices, return_single_flag).

    Raises:
        ValueError: If any index is out of bounds.

    Example:
        >>> trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> _normalize_indices(2, trace)
        ([2], True)
        >>> _normalize_indices([1, 3], trace)
        ([1, 3], False)
    """
    # Handle single index vs multiple indices
    if isinstance(index, int | np.integer):
        indices = [int(index)]
        return_single = True
    else:
        indices = [int(i) for i in index]
        return_single = False

    # Validate indices
    for idx in indices:
        if idx < 0 or idx >= len(trace):
            raise ValueError(f"Index {idx} out of bounds for trace of length {len(trace)}")

    return indices, return_single


def _extract_single_context(
    trace: NDArray[np.float64],
    idx: int,
    before: int,
    after: int,
    sample_rate: float | None,
    include_metadata: bool,
) -> dict[str, Any]:
    """Extract context for a single index.

    Args:
        trace: Input signal trace.
        idx: Center index to extract around.
        before: Samples before index.
        after: Samples after index.
        sample_rate: Optional sample rate.
        include_metadata: Include metadata dict.

    Returns:
        Context dictionary with extracted data and metadata.

    Example:
        >>> trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> ctx = _extract_single_context(trace, 2, 1, 1, None, False)
        >>> ctx['data']
        array([2., 3., 4.])
    """
    # Calculate window bounds with boundary handling
    start_idx, end_idx = _calculate_window_bounds(idx, before, after, len(trace))

    # Extract data
    data = trace[start_idx:end_idx].copy()

    # Build context dictionary
    context: dict[str, Any] = {
        "data": data,
        "start_index": start_idx,
        "end_index": end_idx,
        "center_index": idx,
        "length": len(data),
    }

    # Add time information if requested
    if sample_rate is not None:
        _add_time_information(context, start_idx, sample_rate, len(data))

    # Add metadata if requested
    if include_metadata:
        _add_boundary_metadata(context, idx, start_idx, end_idx, len(trace))

    return context


def _calculate_window_bounds(idx: int, before: int, after: int, trace_len: int) -> tuple[int, int]:
    """Calculate window boundaries with edge handling.

    Args:
        idx: Center index.
        before: Samples before center.
        after: Samples after center.
        trace_len: Length of trace.

    Returns:
        Tuple of (start_idx, end_idx).

    Example:
        >>> _calculate_window_bounds(50, 10, 10, 100)
        (40, 61)
        >>> _calculate_window_bounds(5, 10, 10, 100)
        (0, 16)
    """
    start_idx = max(0, idx - before)
    end_idx = min(trace_len, idx + after + 1)
    return start_idx, end_idx


def _add_time_information(
    context: dict[str, Any], start_idx: int, sample_rate: float, data_len: int
) -> None:
    """Add time reference information to context.

    Args:
        context: Context dictionary to update.
        start_idx: Start index in original trace.
        sample_rate: Sample rate in Hz.
        data_len: Length of extracted data.

    Example:
        >>> ctx = {}
        >>> _add_time_information(ctx, 100, 1e6, 50)
        >>> ctx['time_reference']
        0.0001
        >>> ctx['sample_rate']
        1000000.0
    """
    time_offset = start_idx / sample_rate
    context["time_reference"] = time_offset
    context["sample_rate"] = sample_rate

    # Time array for the context
    dt = 1.0 / sample_rate
    context["time_array"] = np.arange(data_len) * dt + time_offset


def _add_boundary_metadata(
    context: dict[str, Any], idx: int, start_idx: int, end_idx: int, trace_len: int
) -> None:
    """Add boundary metadata to context.

    Args:
        context: Context dictionary to update.
        idx: Center index.
        start_idx: Window start index.
        end_idx: Window end index.
        trace_len: Total trace length.

    Example:
        >>> ctx = {}
        >>> _add_boundary_metadata(ctx, 5, 0, 11, 100)
        >>> ctx['metadata']['at_start_boundary']
        True
        >>> ctx['metadata']['samples_before']
        5
    """
    context["metadata"] = {
        "samples_before": idx - start_idx,
        "samples_after": end_idx - idx - 1,
        "at_start_boundary": start_idx == 0,
        "at_end_boundary": end_idx == trace_len,
    }
