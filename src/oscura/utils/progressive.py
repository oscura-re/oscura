"""Progressive resolution analysis for memory-constrained scenarios.

This module provides multi-pass analysis capabilities: preview first,
then zoom into regions of interest for detailed analysis.


Example:
    >>> from oscura.utils.progressive import create_preview, analyze_roi
    >>> preview = create_preview(trace, downsample_factor=10)
    >>> # User inspects preview, selects ROI
    >>> roi_result = analyze_roi(trace, start_time=0.001, end_time=0.002)

References:
    Multi-resolution analysis techniques
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class PreviewResult:
    """Result of preview analysis.


    Attributes:
        downsampled_data: Downsampled waveform data.
        downsample_factor: Downsampling factor applied.
        original_length: Length of original signal.
        preview_length: Length of preview signal.
        sample_rate: Sample rate of preview (original / factor).
        time_vector: Time axis for preview.
        basic_stats: Basic statistics from preview.
    """

    downsampled_data: NDArray[np.float64]
    downsample_factor: int
    original_length: int
    preview_length: int
    sample_rate: float
    time_vector: NDArray[np.float64]
    basic_stats: dict[str, float]


@dataclass
class ROISelection:
    """Region of interest selection.


    Attributes:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        start_index: Start sample index in original signal.
        end_index: End sample index in original signal.
        duration: Duration in seconds.
        num_samples: Number of samples in ROI.
    """

    start_time: float
    end_time: float
    start_index: int
    end_index: int
    duration: float
    num_samples: int


def create_preview(
    trace: WaveformTrace,
    *,
    downsample_factor: int | None = None,
    max_samples: int = 10_000,
    apply_antialiasing: bool = True,
) -> PreviewResult:
    """Create downsampled preview of waveform for quick inspection.


    Args:
        trace: Input waveform trace.
        downsample_factor: Downsampling factor (auto-computed if None).
        max_samples: Target maximum samples in preview.
        apply_antialiasing: Apply anti-aliasing lowpass filter before decimation.

    Returns:
        PreviewResult with downsampled data and metadata.

    Example:
        >>> preview = create_preview(large_trace, downsample_factor=10)
        >>> print(f"Preview: {preview.preview_length} samples (factor {preview.downsample_factor}x)")
        >>> # Inspect preview.basic_stats
    """
    from scipy import signal as sp_signal

    data = trace.data
    original_length = len(data)
    sample_rate = trace.metadata.sample_rate

    # Auto-compute downsample factor
    if downsample_factor is None:
        downsample_factor = max(1, original_length // max_samples)
        # Round to nearest power of 2 for efficiency
        downsample_factor = 2 ** int(np.ceil(np.log2(downsample_factor)))
        downsample_factor = max(1, downsample_factor)

    # Apply anti-aliasing filter if requested
    if apply_antialiasing and downsample_factor > 1:
        # Lowpass filter at Nyquist frequency of downsampled rate
        nyquist_freq = (sample_rate / downsample_factor) / 2
        sos = sp_signal.butter(8, nyquist_freq, btype="low", fs=sample_rate, output="sos")
        filtered = sp_signal.sosfilt(sos, data)
        downsampled = filtered[::downsample_factor]
    else:
        # Simple decimation without filtering
        downsampled = data[::downsample_factor]

    preview_length = len(downsampled)
    preview_sample_rate = sample_rate / downsample_factor

    # Create time vector
    time_vector = np.arange(preview_length) / preview_sample_rate

    # Compute basic statistics
    basic_stats = {
        "mean": float(np.mean(downsampled)),
        "std": float(np.std(downsampled)),
        "min": float(np.min(downsampled)),
        "max": float(np.max(downsampled)),
        "rms": float(np.sqrt(np.mean(downsampled**2))),
        "peak_to_peak": float(np.ptp(downsampled)),
    }

    return PreviewResult(
        downsampled_data=downsampled,
        downsample_factor=downsample_factor,
        original_length=original_length,
        preview_length=preview_length,
        sample_rate=preview_sample_rate,
        time_vector=time_vector,
        basic_stats=basic_stats,
    )


def select_roi(
    trace: WaveformTrace,
    start_time: float,
    end_time: float,
) -> ROISelection:
    """Create ROI selection from time range.


    Args:
        trace: Input waveform trace.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        ROISelection with sample indices and metadata.

    Raises:
        ValueError: If time range is invalid.

    Example:
        >>> roi = select_roi(trace, start_time=0.001, end_time=0.002)
        >>> print(f"ROI: {roi.num_samples} samples ({roi.duration*1e6:.1f} µs)")
    """
    sample_rate = trace.metadata.sample_rate
    total_length = len(trace.data)
    total_duration = total_length / sample_rate

    # Validate time range
    if start_time < 0 or end_time > total_duration:
        raise ValueError(
            f"Time range [{start_time}, {end_time}] outside signal duration [0, {total_duration}]"
        )
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be < end_time ({end_time})")

    # Convert to sample indices
    start_index = int(start_time * sample_rate)
    end_index = int(end_time * sample_rate)

    # Clamp to valid range
    start_index = max(0, min(start_index, total_length - 1))
    end_index = max(start_index + 1, min(end_index, total_length))

    duration = end_time - start_time
    num_samples = end_index - start_index

    return ROISelection(
        start_time=start_time,
        end_time=end_time,
        start_index=start_index,
        end_index=end_index,
        duration=duration,
        num_samples=num_samples,
    )


def analyze_roi(
    trace: WaveformTrace,
    roi: ROISelection,
    *,
    analysis_func: Callable[[WaveformTrace], Any],
    **analysis_kwargs: Any,
) -> Any:
    """Analyze region of interest with high resolution.


    Args:
        trace: Input waveform trace.
        roi: ROI selection.
        analysis_func: Analysis function to apply to ROI.
        **analysis_kwargs: Additional arguments for analysis function.

    Returns:
        Result of analysis function on ROI.

    Example:
        >>> from oscura.analyzers.waveform.spectral import fft
        >>> roi = select_roi(trace, 0.001, 0.002)
        >>> freq, mag = analyze_roi(trace, roi, analysis_func=fft, window='hann')
    """
    from oscura.core.types import TraceMetadata, WaveformTrace

    # Extract ROI data
    roi_data = trace.data[roi.start_index : roi.end_index]

    # Create new trace for ROI with only standard metadata fields
    roi_trace = WaveformTrace(
        data=roi_data,
        metadata=TraceMetadata(
            sample_rate=trace.metadata.sample_rate,
            vertical_scale=trace.metadata.vertical_scale,
            vertical_offset=trace.metadata.vertical_offset,
            acquisition_time=trace.metadata.acquisition_time,
            trigger_info=trace.metadata.trigger_info,
            source_file=trace.metadata.source_file,
            channel_name=getattr(trace.metadata, "channel_name", None),
        ),
    )

    # Apply analysis function
    return analysis_func(roi_trace, **analysis_kwargs)


def progressive_analysis(
    trace: WaveformTrace,
    *,
    analysis_func: Callable[[WaveformTrace], Any],
    downsample_factor: int = 10,
    roi_selector: Callable[[PreviewResult], ROISelection] | None = None,
    **analysis_kwargs: Any,
) -> tuple[PreviewResult, Any]:
    """Perform progressive multi-pass analysis.


    Workflow:
    1. Create downsampled preview
    2. User/algorithm selects ROI from preview
    3. Perform high-resolution analysis on ROI only

    Args:
        trace: Input waveform trace.
        analysis_func: Analysis function to apply.
        downsample_factor: Downsampling factor for preview.
        roi_selector: Function to select ROI from preview (if None, analyzes full trace).
        **analysis_kwargs: Additional arguments for analysis function.

    Returns:
        Tuple of (preview_result, analysis_result).

    Example:
        >>> def select_peak_region(preview):
        ...     # Find region with highest amplitude
        ...     peak_idx = np.argmax(np.abs(preview.downsampled_data))
        ...     start_time = max(0, (peak_idx - 500) / preview.sample_rate)
        ...     end_time = min(preview.preview_length / preview.sample_rate,
        ...                    (peak_idx + 500) / preview.sample_rate)
        ...     return select_roi(trace, start_time, end_time)
        >>>
        >>> from oscura.analyzers.waveform.spectral import fft
        >>> preview, result = progressive_analysis(
        ...     trace,
        ...     analysis_func=fft,
        ...     downsample_factor=10,
        ...     roi_selector=select_peak_region
        ... )
    """
    # Pass 1: Create preview
    preview = create_preview(trace, downsample_factor=downsample_factor)

    # Pass 2: Select ROI
    if roi_selector is not None:
        roi = roi_selector(preview)
        # Pass 3: Analyze ROI
        result = analyze_roi(trace, roi, analysis_func=analysis_func, **analysis_kwargs)
    else:
        # No ROI selection, analyze full trace
        result = analysis_func(trace, **analysis_kwargs)

    return preview, result


def estimate_optimal_preview_factor(
    trace_length: int,
    *,
    target_memory: int = 100_000_000,  # 100 MB
    bytes_per_sample: int = 8,
) -> int:
    """Estimate optimal downsampling factor for preview.

    Args:
        trace_length: Number of samples in original trace.
        target_memory: Target memory for preview (bytes).
        bytes_per_sample: Bytes per sample (8 for float64).

    Returns:
        Recommended downsampling factor.

    Example:
        >>> factor = estimate_optimal_preview_factor(1_000_000_000)  # 1B samples
        >>> print(f"Downsample by {factor}x for preview")
    """
    # Calculate required factor to fit in target memory
    current_memory = trace_length * bytes_per_sample
    factor = max(1, int(np.ceil(current_memory / target_memory)))

    # Round to power of 2
    factor = 2 ** int(np.ceil(np.log2(factor)))

    return factor  # type: ignore[no-any-return]


__all__ = [
    "PreviewResult",
    "ROISelection",
    "analyze_roi",
    "create_preview",
    "estimate_optimal_preview_factor",
    "progressive_analysis",
    "select_roi",
]
