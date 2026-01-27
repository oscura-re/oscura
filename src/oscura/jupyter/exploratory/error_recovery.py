"""Error recovery and graceful degradation for signal analysis.

This module provides error recovery mechanisms for handling corrupted,
noisy, or incomplete signal data.


Example:
    >>> from oscura.jupyter.exploratory.error_recovery import recover_corrupted_data
    >>> recovered, stats = recover_corrupted_data(trace)
    >>> print(f"Recovered {stats.recovered_samples} samples")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from oscura.core.types import WaveformTrace

T = TypeVar("T")

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class RecoveryStats:
    """Statistics from data recovery.

    Attributes:
        total_samples: Total samples in original data.
        corrupted_samples: Number of detected corrupted samples.
        recovered_samples: Number of successfully recovered samples.
        unrecoverable_samples: Number that could not be recovered.
        recovery_method: Method used for recovery.
        confidence: Confidence in recovered data.
    """

    total_samples: int
    corrupted_samples: int
    recovered_samples: int
    unrecoverable_samples: int
    recovery_method: str
    confidence: float


def recover_corrupted_data(
    trace: WaveformTrace,
    *,
    corruption_threshold: float = 3.0,
    recovery_method: str = "interpolate",
    max_gap_samples: int = 100,
) -> tuple[WaveformTrace, RecoveryStats]:
    """Recover corrupted data.

    Detects and attempts to recover corrupted samples using
    interpolation or other techniques.

    Args:
        trace: Trace with potentially corrupted data.
        corruption_threshold: Threshold for detecting corruption (in std devs).
        recovery_method: 'interpolate', 'median', or 'zero'.
        max_gap_samples: Maximum gap that can be recovered.

    Returns:
        Tuple of (recovered_trace, recovery_stats).

    Example:
        >>> recovered, stats = recover_corrupted_data(trace)
        >>> print(f"Recovered {stats.recovered_samples} samples")
        >>> print(f"Confidence: {stats.confidence:.1%}")

    References:
        ERROR-001: Error Recovery from Corrupted Data
    """
    data = trace.data.copy()
    n = len(data)

    # Detect corrupted samples
    corrupted_mask, corrupted_indices = _detect_corrupted_samples(data, corruption_threshold)
    n_corrupted = len(corrupted_indices)

    if n_corrupted == 0:
        return trace, RecoveryStats(
            total_samples=n,
            corrupted_samples=0,
            recovered_samples=0,
            unrecoverable_samples=0,
            recovery_method="none",
            confidence=1.0,
        )

    # Group corrupted samples into contiguous regions (gaps)
    gaps = _find_corruption_gaps(corrupted_indices)

    # Attempt recovery for each gap
    recovered, unrecoverable = _recover_gaps(
        data, gaps, corrupted_mask, n, recovery_method, max_gap_samples
    )

    # Create recovered trace
    recovered_trace = WaveformTrace(
        data=data,
        metadata=trace.metadata,
    )

    # Calculate confidence
    recovery_ratio = recovered / max(n_corrupted, 1)
    gap_sizes = [end - start + 1 for start, end in gaps]
    avg_gap_size = np.mean(gap_sizes) if gap_sizes else 0
    confidence = recovery_ratio * (1 - avg_gap_size / max_gap_samples)

    confidence_clamped: float = float(max(0.0, min(1.0, confidence)))
    return recovered_trace, RecoveryStats(
        total_samples=n,
        corrupted_samples=n_corrupted,
        recovered_samples=recovered,
        unrecoverable_samples=unrecoverable,
        recovery_method=recovery_method,
        confidence=confidence_clamped,
    )


def _detect_corrupted_samples(
    data: NDArray[np.float64],
    corruption_threshold: float,
) -> tuple[NDArray[np.bool_], NDArray[np.int_]]:
    """Detect corrupted samples using statistical outlier detection.

    Args:
        data: Waveform data array.
        corruption_threshold: Threshold in standard deviations.

    Returns:
        Tuple of (corrupted_mask, corrupted_indices).
    """
    # Filter out nan/inf for initial statistics calculation
    valid_mask = np.isfinite(data)
    valid_data = data[valid_mask] if np.any(valid_mask) else data

    median = np.median(valid_data) if len(valid_data) > 0 else 0.0
    mad = np.median(np.abs(valid_data - median)) if len(valid_data) > 0 else 0.0

    if mad < 1e-10:
        mad = np.std(valid_data) if len(valid_data) > 0 else 1.0

    # Z-score based on MAD (Median Absolute Deviation)
    z_scores = np.abs(data - median) / (1.4826 * mad + 1e-10)

    # Find corrupted samples (statistical outliers or invalid values)
    corrupted_mask = z_scores > corruption_threshold
    corrupted_mask |= np.isnan(data)
    corrupted_mask |= np.isinf(data)

    corrupted_indices = np.where(corrupted_mask)[0]

    return corrupted_mask, corrupted_indices


def _find_corruption_gaps(
    corrupted_indices: NDArray[np.int_],
) -> list[tuple[int, int]]:
    """Group corrupted indices into contiguous regions (gaps).

    Args:
        corrupted_indices: Indices of corrupted samples.

    Returns:
        List of (start, end) tuples for each gap.
    """
    if len(corrupted_indices) == 0:
        return []

    gaps = []
    gap_start = corrupted_indices[0]
    gap_end = corrupted_indices[0]

    for idx in corrupted_indices[1:]:
        if idx == gap_end + 1:
            gap_end = idx
        else:
            gaps.append((gap_start, gap_end))
            gap_start = idx
            gap_end = idx

    gaps.append((gap_start, gap_end))
    return gaps


def _recover_gaps(
    data: NDArray[np.float64],
    gaps: list[tuple[int, int]],
    corrupted_mask: NDArray[np.bool_],
    n: int,
    recovery_method: str,
    max_gap_samples: int,
) -> tuple[int, int]:
    """Attempt recovery for each corruption gap.

    Args:
        data: Waveform data array (modified in place).
        gaps: List of (start, end) tuples for gaps.
        corrupted_mask: Boolean mask of corrupted samples.
        n: Total number of samples.
        recovery_method: Recovery method ('interpolate', 'median', 'zero').
        max_gap_samples: Maximum recoverable gap size.

    Returns:
        Tuple of (recovered_count, unrecoverable_count).
    """
    recovered = 0
    unrecoverable = 0

    for start, end in gaps:
        gap_length = end - start + 1

        if gap_length > max_gap_samples:
            unrecoverable += gap_length
            continue

        if recovery_method == "interpolate":
            recovered += _recover_gap_interpolate(data, start, end, gap_length, n)
        elif recovery_method == "median":
            result = _recover_gap_median(data, corrupted_mask, start, end, gap_length, n)
            if result:
                recovered += gap_length
            else:
                unrecoverable += gap_length
        elif recovery_method == "zero":
            data[start : end + 1] = 0
            recovered += gap_length
        else:
            unrecoverable += gap_length

    return recovered, unrecoverable


def _recover_gap_interpolate(
    data: NDArray[np.float64],
    start: int,
    end: int,
    gap_length: int,
    n: int,
) -> int:
    """Recover gap using linear interpolation.

    Args:
        data: Waveform data array (modified in place).
        start: Gap start index.
        end: Gap end index.
        gap_length: Gap length.
        n: Total number of samples.

    Returns:
        Number of samples recovered.
    """
    left_idx = max(0, start - 1)
    right_idx = min(n - 1, end + 1)

    if left_idx < start and right_idx > end:
        # Can interpolate between valid samples
        left_val = data[left_idx]
        right_val = data[right_idx]
        for i, idx in enumerate(range(start, end + 1)):
            t = (i + 1) / (gap_length + 1)
            data[idx] = left_val * (1 - t) + right_val * t
    else:
        # Edge case - use nearest valid value
        if left_idx >= start:
            data[start : end + 1] = data[right_idx]
        else:
            data[start : end + 1] = data[left_idx]

    return gap_length


def _recover_gap_median(
    data: NDArray[np.float64],
    corrupted_mask: NDArray[np.bool_],
    start: int,
    end: int,
    gap_length: int,
    n: int,
) -> bool:
    """Recover gap using local median value.

    Args:
        data: Waveform data array (modified in place).
        corrupted_mask: Boolean mask of corrupted samples.
        start: Gap start index.
        end: Gap end index.
        gap_length: Gap length.
        n: Total number of samples.

    Returns:
        True if recovery succeeded, False otherwise.
    """
    window_start = max(0, start - 50)
    window_end = min(n, end + 50)
    window_data = data[window_start:window_end]
    valid_data = window_data[~corrupted_mask[window_start:window_end]]

    if len(valid_data) > 0:
        fill_value = np.median(valid_data)
        data[start : end + 1] = fill_value
        return True
    else:
        return False


@dataclass
class GracefulDegradationResult:
    """Result of gracefully degraded analysis.

    Attributes:
        result: Analysis result (may be partial).
        quality_level: 'full', 'degraded', or 'minimal'.
        available_features: Features that could be computed.
        missing_features: Features that failed.
        warnings: List of warnings about degradation.
    """

    result: dict[str, Any]
    quality_level: str
    available_features: list[str]
    missing_features: list[str]
    warnings: list[str]


def graceful_degradation(
    analysis_func: Callable[..., dict[str, Any]],
    trace: WaveformTrace,
    *,
    required_features: list[str] | None = None,
    optional_features: list[str] | None = None,
    **kwargs: Any,
) -> GracefulDegradationResult:
    """Execute analysis with graceful degradation.

    Attempts to provide partial results when full analysis fails.

    Args:
        analysis_func: Analysis function to call.
        trace: Trace to analyze.
        required_features: Features that must succeed.
        optional_features: Features that can fail.
        **kwargs: Additional arguments to analysis function.

    Returns:
        GracefulDegradationResult with partial or full results.

    Example:
        >>> result = graceful_degradation(analyze_signal, trace)
        >>> print(f"Quality: {result.quality_level}")
        >>> print(f"Available: {result.available_features}")

    References:
        ERROR-002: Graceful Degradation
    """
    if required_features is None:
        required_features = []
    if optional_features is None:
        optional_features = []

    result: dict[str, Any] = {}
    available = []
    missing = []
    warnings = []

    # Try full analysis first
    try:
        result = analysis_func(trace, **kwargs)
        available = list(result.keys())
        quality_level = "full"

    except Exception as e:
        logger.warning("Full analysis failed: %s", e, exc_info=True)
        warnings.append(f"Full analysis failed: {e!s}")

        # Try reduced analysis
        for feature in required_features + optional_features:
            try:
                # Attempt to compute individual feature
                if hasattr(trace, feature):
                    result[feature] = getattr(trace, feature)
                    available.append(feature)
                else:
                    missing.append(feature)
            except Exception as fe:
                logger.debug("Feature %s failed: %s", feature, fe, exc_info=True)
                missing.append(feature)
                if feature in required_features:
                    warnings.append(f"Required feature {feature} failed: {fe!s}")

        # Determine quality level
        if all(f in available for f in required_features):
            quality_level = "degraded"
        elif len(available) > 0:
            quality_level = "minimal"
        else:
            quality_level = "failed"
            warnings.append("Analysis completely failed")

    return GracefulDegradationResult(
        result=result,
        quality_level=quality_level,
        available_features=available,
        missing_features=missing,
        warnings=warnings,
    )


@dataclass
class PartialDecodeResult:
    """Result of partial protocol decode.

    Attributes:
        complete_packets: Successfully decoded packets.
        partial_packets: Partially decoded packets.
        error_regions: Regions that could not be decoded.
        decode_rate: Percentage of signal successfully decoded.
        confidence: Confidence in decoded data.
    """

    complete_packets: list[dict[str, Any]]
    partial_packets: list[dict[str, Any]]
    error_regions: list[dict[str, Any]]
    decode_rate: float
    confidence: float


def partial_decode(
    trace: WaveformTrace,
    decode_func: Callable[[WaveformTrace], list[dict[str, Any]]],
    *,
    segment_size: int = 10000,
    min_valid_ratio: float = 0.5,
) -> PartialDecodeResult:
    """Decode protocol with partial result support.

    Continues decoding after errors to capture as much data as possible.

    Args:
        trace: Trace to decode.
        decode_func: Protocol decode function.
        segment_size: Size of segments to try independently.
        min_valid_ratio: Minimum valid ratio to accept segment.

    Returns:
        PartialDecodeResult with all decoded data.

    Example:
        >>> result = partial_decode(trace, uart_decode)
        >>> print(f"Decoded {len(result.complete_packets)} complete packets")
        >>> print(f"Decode rate: {result.decode_rate:.1%}")

    References:
        ERROR-003: Partial Decode Support
    """
    n = len(trace.data)

    # Try full decode first
    full_decode_result = _try_full_decode(trace, decode_func, n)
    if full_decode_result is not None:
        return full_decode_result

    # Fall back to segment decode
    complete_packets, partial_packets, error_regions, decoded_samples, total_samples = (
        _segment_decode(trace, decode_func, segment_size, min_valid_ratio, n)
    )

    # Calculate statistics
    decode_rate = decoded_samples / max(total_samples, 1)
    error_ratio = len(error_regions) / max((n // segment_size), 1)
    confidence = decode_rate * (1 - error_ratio)

    return PartialDecodeResult(
        complete_packets=complete_packets,
        partial_packets=partial_packets,
        error_regions=error_regions,
        decode_rate=decode_rate,
        confidence=confidence,
    )


def _try_full_decode(
    trace: WaveformTrace,
    decode_func: Callable[[WaveformTrace], list[dict[str, Any]]],
    n: int,
) -> PartialDecodeResult | None:
    """Try to decode entire trace at once.

    Args:
        trace: Trace to decode.
        decode_func: Protocol decode function.
        n: Number of samples.

    Returns:
        PartialDecodeResult if successful, None if failed.
    """
    try:
        full_result = decode_func(trace)
        if full_result:
            return PartialDecodeResult(
                complete_packets=full_result,
                partial_packets=[],
                error_regions=[],
                decode_rate=1.0,
                confidence=1.0,
            )
    except Exception as e:
        logger.info("Full decode failed, falling back to segment decode: %s", e)

    return None


def _segment_decode(
    trace: WaveformTrace,
    decode_func: Callable[[WaveformTrace], list[dict[str, Any]]],
    segment_size: int,
    min_valid_ratio: float,
    n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Decode trace segment by segment.

    Args:
        trace: Trace to decode.
        decode_func: Protocol decode function.
        segment_size: Size of segments to try independently.
        min_valid_ratio: Minimum valid ratio to accept segment.
        n: Number of samples.

    Returns:
        Tuple of (complete_packets, partial_packets, error_regions, decoded_samples, total_samples).
    """
    complete_packets: list[dict[str, Any]] = []
    partial_packets: list[dict[str, Any]] = []
    error_regions: list[dict[str, Any]] = []
    total_samples = 0
    decoded_samples = 0

    for start in range(0, n, segment_size):
        end = min(start + segment_size, n)
        segment_data = trace.data[start:end]

        segment_trace = WaveformTrace(data=segment_data, metadata=trace.metadata)
        total_samples += len(segment_data)

        try:
            segment_result = decode_func(segment_trace)

            if segment_result:
                _adjust_packet_timestamps(segment_result, start, trace.metadata.sample_rate)

                # Classify segment as complete or partial
                valid_ratio = len(segment_result) / max(len(segment_data) / 100, 1)
                if valid_ratio >= min_valid_ratio:
                    complete_packets.extend(segment_result)
                    decoded_samples += len(segment_data)
                else:
                    partial_packets.extend(segment_result)
                    decoded_samples += len(segment_data) // 2

        except Exception as e:
            logger.debug("Segment decode failed at sample %d: %s", start, e)
            error_regions.append({"start_sample": start, "end_sample": end, "error": str(e)})

    return complete_packets, partial_packets, error_regions, decoded_samples, total_samples


def _adjust_packet_timestamps(
    packets: list[dict[str, Any]],
    start_sample: int,
    sample_rate: float,
) -> None:
    """Adjust packet timestamps to account for segment offset.

    Args:
        packets: List of decoded packets to adjust.
        start_sample: Start sample of this segment.
        sample_rate: Sample rate in Hz.
    """
    for packet in packets:
        if "timestamp" in packet:
            packet["timestamp"] += start_sample / sample_rate
        if "sample" in packet:
            packet["sample"] += start_sample


@dataclass
class ErrorContext:
    """Preserved error context for debugging.

    Attributes:
        error_type: Type of error that occurred.
        error_message: Error message.
        location: Where in the signal the error occurred.
        context_before: Signal context before error.
        context_after: Signal context after error.
        parameters: Parameters at time of error.
        suggestions: Suggestions for fixing the error.
    """

    error_type: str
    error_message: str
    location: int | None
    context_before: NDArray[np.float64] | None
    context_after: NDArray[np.float64] | None
    parameters: dict[str, Any]
    suggestions: list[str]

    @classmethod
    def capture(
        cls,
        exception: Exception,
        trace: WaveformTrace | None = None,
        location: int | None = None,
        context_samples: int = 100,
        parameters: dict[str, Any] | None = None,
    ) -> ErrorContext:
        """Capture error context from exception.

        Args:
            exception: The exception that occurred.
            trace: Signal trace (for context extraction).
            location: Sample index where error occurred.
            context_samples: Number of context samples to capture.
            parameters: Analysis parameters at time of error.

        Returns:
            ErrorContext with all available information.
        """
        context_before = None
        context_after = None

        if trace is not None and location is not None:
            data = trace.data
            n = len(data)

            if location >= 0 and location < n:
                start = max(0, location - context_samples)
                end = min(n, location + context_samples)
                context_before = data[start:location]
                context_after = data[location:end]

        # Generate suggestions based on error type
        suggestions = []
        error_str = str(exception)

        if "insufficient" in error_str.lower():
            suggestions.append("Try providing more data samples")
            suggestions.append("Check if trace is complete")

        if "threshold" in error_str.lower():
            suggestions.append("Try adjusting threshold parameter")
            suggestions.append("Check signal levels are as expected")

        if "timeout" in error_str.lower():
            suggestions.append("Increase timeout parameter")
            suggestions.append("Process in smaller chunks")

        if "memory" in error_str.lower():
            suggestions.append("Use chunked processing")
            suggestions.append("Reduce analysis window size")

        if not suggestions:
            suggestions.append("Check input data format")
            suggestions.append("Verify analysis parameters")

        return cls(
            error_type=type(exception).__name__,
            error_message=str(exception),
            location=location,
            context_before=context_before,
            context_after=context_after,
            parameters=parameters or {},
            suggestions=suggestions,
        )


@dataclass
class RetryResult:
    """Result of retry with parameter adjustment.

    Attributes:
        success: True if retry succeeded.
        result: Analysis result (if successful).
        attempts: Number of attempts made.
        final_parameters: Parameters that worked.
        adjustments_made: List of adjustments made.
    """

    success: bool
    result: Any
    attempts: int
    final_parameters: dict[str, Any]
    adjustments_made: list[str]


def retry_with_adjustment(
    func: Callable[..., T],
    trace: WaveformTrace,
    initial_params: dict[str, Any],
    *,
    max_retries: int = 3,
    adjustment_rules: dict[str, Callable[[Any, int], Any]] | None = None,
) -> RetryResult:
    """Retry analysis with automatic parameter adjustment.

    Adjusts parameters and retries when analysis fails.

    Args:
        func: Analysis function to retry.
        trace: Trace to analyze.
        initial_params: Initial parameters.
        max_retries: Maximum retry attempts.
        adjustment_rules: Rules for adjusting parameters.

    Returns:
        RetryResult with outcome of retries.

    Example:
        >>> rules = {
        ...     'threshold': lambda v, n: v * 0.9,  # Reduce by 10% each retry
        ...     'window_size': lambda v, n: v * 2,  # Double each retry
        ... }
        >>> result = retry_with_adjustment(analyze, trace, params, adjustment_rules=rules)
        >>> if result.success:
        ...     print(f"Succeeded after {result.attempts} attempts")

    References:
        ERROR-005: Automatic Retry with Parameter Adjustment
    """
    if adjustment_rules is None:
        # Default adjustment rules
        adjustment_rules = {
            "threshold": lambda v, n: v * (0.9**n),
            "tolerance": lambda v, n: v * (1.2**n),
            "window_size": lambda v, n: int(v * (1.5**n)),
            "min_samples": lambda v, n: max(1, int(v * (0.8**n))),
        }

    params = initial_params.copy()
    adjustments_made = []  # type: ignore[var-annotated]

    for attempt in range(max_retries + 1):
        try:
            result = func(trace, **params)
            return RetryResult(
                success=True,
                result=result,
                attempts=attempt + 1,
                final_parameters=params,
                adjustments_made=adjustments_made,
            )

        except Exception as e:
            logger.debug("Retry attempt %d failed: %s", attempt + 1, e)
            if attempt >= max_retries:
                logger.warning("Max retries (%d) reached, giving up: %s", max_retries, e)
                break

            # Adjust parameters for next attempt
            for param_name, adjust_func in adjustment_rules.items():
                if param_name in params:
                    old_val = params[param_name]
                    new_val = adjust_func(old_val, attempt + 1)
                    params[param_name] = new_val
                    adjustments_made.append(
                        f"Attempt {attempt + 1}: {param_name} {old_val} -> {new_val}"
                    )

    return RetryResult(
        success=False,
        result=None,
        attempts=max_retries + 1,
        final_parameters=params,
        adjustments_made=adjustments_made,
    )


__all__ = [
    "ErrorContext",
    "GracefulDegradationResult",
    "PartialDecodeResult",
    "RecoveryStats",
    "RetryResult",
    "graceful_degradation",
    "partial_decode",
    "recover_corrupted_data",
    "retry_with_adjustment",
]
