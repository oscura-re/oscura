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
    if before < 0 or after < 0:
        raise ValueError("before and after must be non-negative")

    if trace.size == 0:
        raise ValueError("Trace cannot be empty")

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

    # Extract contexts
    contexts = []

    for idx in indices:
        # Calculate window bounds with boundary handling
        start_idx = max(0, idx - before)
        end_idx = min(len(trace), idx + after + 1)

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

        # Add time reference if sample rate provided
        if sample_rate is not None:
            time_offset = start_idx / sample_rate
            context["time_reference"] = time_offset
            context["sample_rate"] = sample_rate

            # Time array for the context
            dt = 1.0 / sample_rate
            context["time_array"] = np.arange(len(data)) * dt + time_offset

        if include_metadata:
            context["metadata"] = {
                "samples_before": idx - start_idx,
                "samples_after": end_idx - idx - 1,
                "at_start_boundary": start_idx == 0,
                "at_end_boundary": end_idx == len(trace),
            }

        contexts.append(context)

    # Return single context or list
    if return_single:
        return contexts[0]
    else:
        return contexts
