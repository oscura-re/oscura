"""Visualization optimization functions for automatic plot parameter selection.

This module provides intelligent optimization algorithms for plot parameters
including axis ranges, decimation, grid spacing, dynamic range, and
interesting region detection.


Example:
    >>> from oscura.visualization.optimization import calculate_optimal_y_range
    >>> y_min, y_max = calculate_optimal_y_range(signal_data)
    >>> ax.set_ylim(y_min, y_max)

References:
    - Wilkinson's tick placement algorithm (1999)
    - LTTB (Largest Triangle Three Buckets) decimation
    - Percentile-based outlier detection
    - Edge detection using Sobel operators
    - Statistical outlier detection using MAD (Median Absolute Deviation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal as sp_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray


def calculate_optimal_y_range(
    data: NDArray[np.float64],
    *,
    outlier_threshold: float = 3.0,
    margin_percent: float = 5.0,
    symmetric: bool = False,
    clip_warning_threshold: float = 0.01,
) -> tuple[float, float]:
    """Calculate optimal Y-axis range with outlier exclusion.

    Uses percentile-based outlier detection and smart margins to ensure
    data visibility without wasted space. Detects clipping when too many
    samples are excluded.

    Args:
        data: Signal data array.
        outlier_threshold: Number of standard deviations for outlier exclusion (default 3.0).
        margin_percent: Margin as percentage of data range (default 5%, auto-adjusts).
        symmetric: If True, use symmetric range ±max for bipolar signals.
        clip_warning_threshold: Warn if this fraction of samples are clipped (default 1%).

    Returns:
        Tuple of (y_min, y_max) for axis limits.

    Raises:
        ValueError: If data is empty or all NaN.

    Example:
        >>> signal = np.random.randn(1000)
        >>> y_min, y_max = calculate_optimal_y_range(signal, symmetric=True)
        >>> # For bipolar signal: y_min ≈ -y_max

    References:
        VIS-013: Auto Y-Axis Range Optimization
    """
    clean_data = _validate_and_clean_data(data)
    filtered_data = _filter_outliers(clean_data, outlier_threshold)
    _check_clipping(clean_data, filtered_data, clip_warning_threshold)

    # Calculate data range
    data_min, data_max = float(np.min(filtered_data)), float(np.max(filtered_data))
    margin = _select_smart_margin(len(filtered_data), margin_percent)

    # Apply range mode
    if symmetric:
        return _symmetric_range(data_min, data_max, margin)
    else:
        return _asymmetric_range(data_min, data_max, margin)


def _validate_and_clean_data(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate data and remove NaN values."""
    if len(data) == 0:
        raise ValueError("Data array is empty")

    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        raise ValueError("Data contains only NaN values")

    return clean_data


def _filter_outliers(data: NDArray[np.float64], outlier_threshold: float) -> NDArray[np.float64]:
    """Filter outliers using robust MAD-based z-scores.

    Falls back to standard deviation when MAD = 0 (highly concentrated data).
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    robust_std = 1.4826 * mad  # MAD to std conversion

    if robust_std > 0:
        z_scores = np.abs(data - median) / robust_std
        filtered: NDArray[np.float64] = data[z_scores <= outlier_threshold]
        return filtered

    # Fallback to standard deviation when MAD = 0
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        z_scores = np.abs(data - mean) / std
        filtered = data[z_scores <= outlier_threshold]
        return filtered

    return data


def _check_clipping(
    clean_data: NDArray[np.float64],
    filtered_data: NDArray[np.float64],
    clip_warning_threshold: float,
) -> None:
    """Check and warn if too many samples are clipped."""
    clipped_fraction = 1.0 - (len(filtered_data) / len(clean_data))
    if clipped_fraction > clip_warning_threshold:
        import warnings

        warnings.warn(
            f"Clipping detected: {clipped_fraction * 100:.1f}% of samples "
            f"excluded by range limits (threshold: {clip_warning_threshold * 100:.1f}%)",
            UserWarning,
            stacklevel=2,
        )


def _select_smart_margin(n_samples: int, margin_percent: float) -> float:
    """Select margin based on data density.

    Only applies smart margin when using default value (5.0%).
    Otherwise respects user's explicit margin_percent.
    """
    # Always respect explicit user values (non-default)
    if margin_percent != 5.0:
        return margin_percent / 100.0

    # Apply smart margin only for default value
    if n_samples > 10000:
        return 0.02  # Dense data: smaller margin
    elif n_samples < 100:
        return 0.10  # Sparse data: larger margin
    return margin_percent / 100.0


def _symmetric_range(data_min: float, data_max: float, margin: float) -> tuple[float, float]:
    """Calculate symmetric range for bipolar signals."""
    max_abs = max(abs(data_min), abs(data_max))

    # Handle constant data
    if max_abs == 0:
        return (-0.5, 0.5)  # Default range for constant zero

    margin_value = max_abs * margin
    return (-(max_abs + margin_value), max_abs + margin_value)


def _asymmetric_range(data_min: float, data_max: float, margin: float) -> tuple[float, float]:
    """Calculate asymmetric range."""
    data_range = data_max - data_min

    # Handle constant data (range = 0)
    if data_range == 0:
        # Add fixed margin for constant data
        default_margin = 0.5 if data_min == 0 else abs(data_min) * 0.1
        return (data_min - default_margin, data_max + default_margin)

    margin_value = data_range * margin
    return (data_min - margin_value, data_max + margin_value)


def calculate_optimal_x_window(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    *,
    target_features: int = 5,
    samples_per_pixel: float = 2.0,
    screen_width: int = 1000,
    activity_threshold: float = 0.1,
) -> tuple[float, float]:
    """Calculate optimal X-axis time window with activity detection.

    Intelligently determines time window based on signal activity and features.
    Detects regions with significant activity and zooms to show N complete features.

    Args:
        time: Time axis array in seconds.
        data: Signal data array.
        target_features: Number of complete features to display (default 5-10).
        samples_per_pixel: Threshold for decimation (default >2 samples/pixel).
        screen_width: Screen width in pixels for decimation calculation.
        activity_threshold: Relative threshold for activity detection (0-1).

    Returns:
        Tuple of (t_start, t_end) for time window in seconds.

    Raises:
        ValueError: If arrays are empty or mismatched.

    Example:
        >>> time = np.linspace(0, 1e-3, 10000)
        >>> signal = np.sin(2 * np.pi * 1000 * time)
        >>> t_start, t_end = calculate_optimal_x_window(time, signal, target_features=5)

    References:
        VIS-014: Adaptive X-Axis Time Window
    """
    if len(time) == 0 or len(data) == 0:
        raise ValueError("Time or data array is empty")

    if len(time) != len(data):
        raise ValueError(f"Time and data arrays must match: {len(time)} vs {len(data)}")

    # Detect signal activity using RMS windowing
    window_size = max(10, len(data) // 100)
    rms = np.sqrt(np.convolve(data**2, np.ones(window_size) / window_size, mode="same"))

    # Find activity threshold
    rms_threshold = activity_threshold * np.max(rms)
    active_regions = rms > rms_threshold

    if not np.any(active_regions):
        # No significant activity, return padded full range
        time_range = time[-1] - time[0]
        padding = time_range * 0.05  # 5% padding on each side
        return (float(time[0] - padding), float(time[-1] + padding))

    # Find first active region
    active_indices = np.where(active_regions)[0]
    first_active = active_indices[0]

    # Detect features using autocorrelation for periodic signals
    # Use a subset for efficiency
    subset_size = min(5000, len(data) - first_active)
    subset_start = first_active
    subset_end = subset_start + subset_size

    if subset_end > len(data):
        subset_end = len(data)
        subset_start = max(0, subset_end - subset_size)

    subset = data[subset_start:subset_end]

    # Try to detect periodicity
    if len(subset) > 20:
        # Use zero-crossing to detect period
        mean_val = np.mean(subset)
        crossings = np.where(np.diff(np.sign(subset - mean_val)))[0]

        if len(crossings) >= 4:
            # Estimate period from crossings (two crossings per cycle)
            # crossings[::2] already gives full periods (every other crossing)
            periods = np.diff(crossings[::2])
            if len(periods) > 0:
                median_period = np.median(periods)
                samples_per_feature = int(median_period)  # Already full cycle from [::2]

                # Calculate window to show target_features
                total_samples = samples_per_feature * target_features

                # Respect decimation constraint
                max_window_samples = int(screen_width * samples_per_pixel)
                total_samples = min(total_samples, max_window_samples)

                window_start = first_active
                window_end = min(window_start + total_samples, len(time) - 1)

                return (float(time[window_start]), float(time[window_end]))

    # Fallback: zoom to respect decimation threshold
    # Limit window to screen_width * samples_per_pixel samples
    max_window_samples = int(screen_width * samples_per_pixel)
    active_duration = len(active_indices)
    zoom_samples = min(active_duration, max_window_samples)
    window_end = min(first_active + zoom_samples, len(time) - 1)

    return (float(time[first_active]), float(time[window_end]))


def calculate_grid_spacing(
    axis_min: float,
    axis_max: float,
    *,
    target_major_ticks: int = 7,
    log_scale: bool = False,
    time_axis: bool = False,
) -> tuple[float, float]:
    """Calculate optimal grid spacing using nice numbers.

    Implements Wilkinson's tick placement algorithm to generate
    aesthetically pleasing major and minor grid line spacing.

    Args:
        axis_min: Minimum axis value.
        axis_max: Maximum axis value.
        target_major_ticks: Target number of major gridlines (default 5-10).
        log_scale: Use logarithmic spacing for log-scale axes.
        time_axis: Use time-unit alignment (ms, μs, ns).

    Returns:
        Tuple of (major_spacing, minor_spacing).

    Raises:
        ValueError: If axis_max <= axis_min.

    Example:
        >>> major, minor = calculate_grid_spacing(0, 100, target_major_ticks=5)
        >>> # Returns nice numbers like (20.0, 4.0)

    References:
        VIS-019: Grid Auto-Spacing
        Wilkinson (1999): The Grammar of Graphics
    """
    if axis_max <= axis_min:
        raise ValueError(f"Invalid axis range: [{axis_min}, {axis_max}]")

    if log_scale:
        # Logarithmic spacing: major grids at decade boundaries
        log_min = np.log10(max(axis_min, 1e-100))
        log_max = np.log10(axis_max)
        n_decades = log_max - log_min

        if n_decades < 1:
            # Less than one decade: use linear spacing
            major_spacing = _calculate_nice_number((axis_max - axis_min) / target_major_ticks)
            minor_spacing = major_spacing / 5
        else:
            # Major grids at decades, minors at 2, 5
            major_spacing = 10.0 ** np.ceil(n_decades / target_major_ticks)
            minor_spacing = major_spacing / 5

        return (float(major_spacing), float(minor_spacing))

    # Linear spacing with nice numbers
    axis_range = axis_max - axis_min
    rough_spacing = axis_range / target_major_ticks

    # Find nice number for major spacing
    major_spacing = _calculate_nice_number(rough_spacing)

    # Minor spacing: 1/5 or 1/2 of major
    # Use 1/5 for spacings ending in 5 or 10, 1/2 otherwise
    if major_spacing % 5 == 0 or major_spacing % 10 == 0:
        minor_spacing = major_spacing / 5
    else:
        minor_spacing = major_spacing / 2

    # Time axis alignment
    if time_axis:
        # Align to natural time units
        time_units = [
            1e-9,
            2e-9,
            5e-9,  # ns
            1e-6,
            2e-6,
            5e-6,  # μs
            1e-3,
            2e-3,
            5e-3,  # ms
            1.0,
            2.0,
            5.0,
        ]  # s

        # Find closest time unit
        closest_idx = np.argmin(np.abs(np.array(time_units) - major_spacing))
        major_spacing = time_units[closest_idx]
        minor_spacing = major_spacing / 5

    # Check if grid would be too dense
    actual_major_ticks = axis_range / major_spacing
    if actual_major_ticks > 15:
        # Disable minor grids (set equal to major)
        minor_spacing = major_spacing

    return (float(major_spacing), float(minor_spacing))


def _calculate_nice_number(value: float) -> float:
    """Calculate nice number using powers of 10 × (1, 2, 5).  # noqa: RUF002

    Args:
        value: Input value.

    Returns:
        Nice number (1, 2, or 5 × 10^n).  # noqa: RUF002
    """
    if value <= 0:
        return 1.0

    # Find exponent
    exponent = np.floor(np.log10(value))
    fraction = value / (10**exponent)

    # Round to nice fraction (1, 2, 5)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.5:
        nice_fraction = 2.0
    elif fraction < 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0

    return nice_fraction * (10**exponent)  # type: ignore[no-any-return]


def optimize_db_range(
    spectrum: NDArray[np.float64],
    *,
    noise_floor_percentile: float = 5.0,
    peak_threshold_db: float = 10.0,
    margin_db: float = 10.0,
    max_dynamic_range_db: float = 100.0,
) -> tuple[float, float]:
    """Optimize dB range for spectrum plots with noise floor detection.

    Automatically detects noise floor and calculates optimal dynamic range
    for maximum information visibility in frequency-domain plots.

    Args:
        spectrum: Spectrum magnitude array (linear or dB).
        noise_floor_percentile: Percentile for noise floor estimation (default 5%).
        peak_threshold_db: Threshold above noise floor for peak detection (default 10 dB).
        margin_db: Margin below noise floor (default 10 dB).
        max_dynamic_range_db: Maximum dynamic range to display (default 100 dB).

    Returns:
        Tuple of (db_min, db_max) for spectrum plot limits.

    Raises:
        ValueError: If spectrum is empty or all zero.

    Example:
        >>> spectrum_db = 20 * np.log10(np.abs(fft_result))
        >>> db_min, db_max = optimize_db_range(spectrum_db)
        >>> ax.set_ylim(db_min, db_max)

    References:
        VIS-022: Spectrum dB Range Optimization
    """
    if len(spectrum) == 0:
        raise ValueError("Spectrum array is empty")

    # Convert to dB if needed (check if values are in linear scale)
    if np.max(spectrum) > 100:
        # Likely linear, convert to dB
        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-100))
    else:
        # Assume already in dB
        spectrum_db = spectrum

    # Detect noise floor using percentile method
    noise_floor = np.percentile(spectrum_db, noise_floor_percentile)

    # Find signal peaks using scipy
    peak_indices, peak_properties = sp_signal.find_peaks(
        spectrum_db,
        height=noise_floor + peak_threshold_db,
        prominence=peak_threshold_db / 2,
    )

    if len(peak_indices) > 0:
        peak_max = np.max(peak_properties["peak_heights"])
    else:
        # No peaks detected, use maximum value
        peak_max = np.max(spectrum_db)

    # Calculate dB range
    db_max = float(peak_max)
    db_min = float(noise_floor - margin_db)

    # Apply dynamic range compression if too wide
    dynamic_range = db_max - db_min
    if dynamic_range > max_dynamic_range_db:
        db_min = db_max - max_dynamic_range_db

    return (db_min, db_max)


def decimate_for_display(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    *,
    max_points: int = 2000,
    method: Literal["lttb", "minmax", "uniform"] = "lttb",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decimate signal data for display using LTTB or min-max envelope.

    Intelligently reduces number of points while preserving visual appearance
    and important features like edges and peaks.

    Args:
        time: Time axis array.
        data: Signal data array.
        max_points: Maximum number of points to retain.
        method: Decimation method ("lttb", "minmax", "uniform").

    Returns:
        Tuple of (decimated_time, decimated_data).

    Raises:
        ValueError: If arrays are empty or method is invalid.

    Example:
        >>> time_dec, data_dec = decimate_for_display(time, data, max_points=1000)
        >>> # Reduced from 100k to 1k points while preserving shape

    References:
        VIS-014: Adaptive X-Axis Time Window
        LTTB: Largest Triangle Three Buckets algorithm
    """
    if len(time) == 0 or len(data) == 0:
        raise ValueError("Time or data array is empty")

    if len(time) != len(data):
        raise ValueError(f"Time and data arrays must match: {len(time)} vs {len(data)}")

    # Don't decimate if already below threshold
    if len(data) <= max_points:
        return (time, data)

    # Never decimate very small signals
    if len(data) < 10:
        return (time, data)

    if method == "uniform":
        # Simple uniform stride decimation
        stride = len(data) // max_points
        indices = np.arange(0, len(data), stride)
        return (time[indices], data[indices])

    elif method == "minmax":
        # Min-max envelope: preserve peaks and valleys
        return _decimate_minmax(time, data, max_points)

    elif method == "lttb":
        # Largest Triangle Three Buckets
        return _decimate_lttb(time, data, max_points)

    else:
        raise ValueError(f"Invalid decimation method: {method}")


def _decimate_minmax(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    max_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decimate using min-max envelope to preserve peaks/valleys.

    Args:
        time: Time array.
        data: Signal data array.
        max_points: Maximum number of points to retain.

    Returns:
        Tuple of (decimated_time, decimated_data).
    """
    # Calculate bucket size
    bucket_size = len(data) // (max_points // 2)

    decimated_time = []
    decimated_data = []

    for i in range(0, len(data), bucket_size):
        bucket = data[i : i + bucket_size]
        time_bucket = time[i : i + bucket_size]

        if len(bucket) > 0:
            # Add min and max from each bucket
            min_idx = np.argmin(bucket)
            max_idx = np.argmax(bucket)

            # Add in chronological order
            if min_idx < max_idx:
                decimated_time.extend([time_bucket[min_idx], time_bucket[max_idx]])
                decimated_data.extend([bucket[min_idx], bucket[max_idx]])
            else:
                decimated_time.extend([time_bucket[max_idx], time_bucket[min_idx]])
                decimated_data.extend([bucket[max_idx], bucket[min_idx]])

    return (np.array(decimated_time), np.array(decimated_data))


def _decimate_lttb(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    max_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decimate using Largest Triangle Three Buckets algorithm.

    Preserves visual appearance by selecting points that maximize
    the area of triangles formed with neighboring buckets.

    Args:
        time: Time array.
        data: Signal data array.
        max_points: Maximum number of points to retain.

    Returns:
        Tuple of (decimated_time, decimated_data).
    """
    if len(data) <= max_points:
        return (time, data)

    # Always include first and last points
    sampled_time = [time[0]]
    sampled_data = [data[0]]

    # Calculate bucket size
    bucket_size = (len(data) - 2) / (max_points - 2)

    # Previous selected point
    prev_idx = 0

    for i in range(max_points - 2):
        # Calculate average point of next bucket (for triangle area calculation)
        avg_range_start = int((i + 1) * bucket_size) + 1
        avg_range_end = int((i + 2) * bucket_size) + 1
        avg_range_end = min(avg_range_end, len(data))

        if avg_range_start < avg_range_end:
            avg_time = np.mean(time[avg_range_start:avg_range_end])
            avg_data = np.mean(data[avg_range_start:avg_range_end])
        else:
            avg_time = time[-1]
            avg_data = data[-1]

        # Current bucket range
        range_start = int(i * bucket_size) + 1
        range_end = int((i + 1) * bucket_size) + 1
        range_end = min(range_end, len(data) - 1)

        # Find point in current bucket that forms largest triangle
        prev_time = time[prev_idx]
        prev_data = data[prev_idx]

        max_area = -1.0
        max_idx = range_start

        for idx in range(range_start, range_end):
            # Calculate triangle area
            area = abs(
                (prev_time - avg_time) * (data[idx] - prev_data)
                - (prev_time - time[idx]) * (avg_data - prev_data)
            )

            if area > max_area:
                max_area = area
                max_idx = idx

        sampled_time.append(time[max_idx])
        sampled_data.append(data[max_idx])
        prev_idx = max_idx

    # Always include last point
    sampled_time.append(time[-1])
    sampled_data.append(data[-1])

    return (np.array(sampled_time), np.array(sampled_data))


@dataclass
class InterestingRegion:
    """Represents an interesting region in a signal.

    Attributes:
        start_idx: Starting sample index
        end_idx: Ending sample index
        start_time: Starting time in seconds
        end_time: Ending time in seconds
        type: Type of interesting feature
        significance: Significance score (0-1, higher is more significant)
        metadata: Additional metadata about the region
    """

    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    type: Literal["edge", "glitch", "anomaly", "pattern_change"]
    significance: float
    metadata: dict  # type: ignore[type-arg]


def detect_interesting_regions(
    signal: NDArray[np.float64],
    sample_rate: float,
    *,
    edge_threshold: float | None = None,
    glitch_sigma: float = 3.0,
    anomaly_threshold: float = 3.0,
    min_region_samples: int = 1,
    max_regions: int = 10,
) -> list[InterestingRegion]:
    """Detect interesting regions in a signal for automatic zoom/focus.

    : Automatically detect and zoom to interesting signal
    regions such as edges, glitches, anomalies, or pattern changes.

    Args:
        signal: Input signal array
        sample_rate: Sample rate in Hz
        edge_threshold: Edge detection threshold (default: auto from signal stddev)
        glitch_sigma: Sigma threshold for glitch detection (default: 3.0)
        anomaly_threshold: Threshold for anomaly detection in sigma (default: 3.0)
        min_region_samples: Minimum samples per region (default: 1)
        max_regions: Maximum number of regions to return (default: 10)

    Returns:
        List of InterestingRegion objects, sorted by significance (descending)

    Raises:
        ValueError: If signal is empty or sample_rate is invalid

    Example:
        >>> signal = np.sin(2*np.pi*1000*t) + 0.1*np.random.randn(len(t))
        >>> regions = detect_interesting_regions(signal, 1e6)
        >>> print(f"Found {len(regions)} interesting regions")

    References:
        VIS-020: Zoom to Interesting Regions
        Edge detection: Sobel operator on signal derivative
        Glitch detection: MAD-based outlier detection
    """
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")

    regions: list[InterestingRegion] = []

    # 1. Edge detection using first derivative
    edges = _detect_edges(signal, sample_rate, edge_threshold)
    regions.extend(edges)

    # 2. Glitch detection using statistical outliers
    glitches = _detect_glitches(signal, sample_rate, glitch_sigma)
    regions.extend(glitches)

    # 3. Anomaly detection using MAD
    anomalies = _detect_anomalies(signal, sample_rate, anomaly_threshold)
    regions.extend(anomalies)

    # 4. Pattern change detection (simplified using variance changes)
    pattern_changes = _detect_pattern_changes(signal, sample_rate)
    regions.extend(pattern_changes)

    # Filter out regions that are too small
    regions = [r for r in regions if (r.end_idx - r.start_idx) >= min_region_samples]

    # Sort by significance (descending)
    regions.sort(key=lambda r: r.significance, reverse=True)

    # Return top N regions
    return regions[:max_regions]


def _detect_edges(
    signal: NDArray[np.float64],
    sample_rate: float,
    threshold: float | None,
) -> list[InterestingRegion]:
    """Detect edge transitions using first derivative.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold: Edge threshold (auto if None)

    Returns:
        List of edge regions
    """
    # Calculate first derivative (gradient)
    gradient = np.gradient(signal)

    # Auto threshold based on signal statistics
    if threshold is None:
        threshold = np.std(gradient) * 2.0

    # Find where gradient exceeds threshold
    edge_mask = np.abs(gradient) > threshold

    # Find continuous edge regions
    regions: list[InterestingRegion] = []
    in_edge = False
    start_idx = 0

    for i, is_edge in enumerate(edge_mask):
        if is_edge and not in_edge:
            # Start of edge
            start_idx = i
            in_edge = True
        elif not is_edge and in_edge:
            # End of edge
            end_idx = i

            # Calculate significance based on gradient magnitude
            edge_gradient = gradient[start_idx:end_idx]
            significance = min(1.0, np.max(np.abs(edge_gradient)) / (threshold * 5))

            time_base = 1.0 / sample_rate
            regions.append(
                InterestingRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=start_idx * time_base,
                    end_time=end_idx * time_base,
                    type="edge",
                    significance=significance,
                    metadata={
                        "max_gradient": float(np.max(np.abs(edge_gradient))),
                        "threshold": threshold,
                    },
                )
            )
            in_edge = False

    # Handle edge at end of signal
    if in_edge:
        end_idx = len(signal)
        edge_gradient = gradient[start_idx:end_idx]
        significance = min(1.0, np.max(np.abs(edge_gradient)) / (threshold * 5))
        time_base = 1.0 / sample_rate
        regions.append(
            InterestingRegion(
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_idx * time_base,
                end_time=end_idx * time_base,
                type="edge",
                significance=significance,
                metadata={
                    "max_gradient": float(np.max(np.abs(edge_gradient))),
                    "threshold": threshold,
                },
            )
        )

    return regions


def _detect_glitches(
    signal: NDArray[np.float64],
    sample_rate: float,
    sigma_threshold: float,
) -> list[InterestingRegion]:
    """Detect isolated spikes (glitches) using z-score.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        sigma_threshold: Sigma threshold for outlier detection

    Returns:
        List of glitch regions
    """
    # Calculate z-scores
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return []

    z_scores = np.abs((signal - mean) / std)

    # Find outliers
    outlier_mask = z_scores > sigma_threshold

    # Find isolated glitches (single sample or very short bursts)
    regions: list[InterestingRegion] = []
    time_base = 1.0 / sample_rate

    i = 0
    while i < len(outlier_mask):
        if outlier_mask[i]:
            # Start of potential glitch
            start_idx = i

            # Find end of glitch (max 5 samples to be considered a glitch)
            while i < len(outlier_mask) and outlier_mask[i] and (i - start_idx) < 5:
                i += 1

            end_idx = i

            # Calculate significance based on z-score
            glitch_z_scores = z_scores[start_idx:end_idx]
            significance = min(1.0, np.max(glitch_z_scores) / (sigma_threshold * 3))

            regions.append(
                InterestingRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=start_idx * time_base,
                    end_time=end_idx * time_base,
                    type="glitch",
                    significance=significance,
                    metadata={
                        "max_z_score": float(np.max(glitch_z_scores)),
                        "threshold_sigma": sigma_threshold,
                    },
                )
            )
        else:
            i += 1

    return regions


def _detect_anomalies(
    signal: NDArray[np.float64],
    sample_rate: float,
    threshold_sigma: float,
) -> list[InterestingRegion]:
    """Detect anomalies using MAD (Median Absolute Deviation).

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        threshold_sigma: Sigma threshold for MAD

    Returns:
        List of anomaly regions
    """
    # Calculate MAD
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))

    if mad == 0:
        return []

    # Modified z-score using MAD (more robust than standard z-score)
    modified_z_scores = 0.6745 * (signal - median) / mad

    # Find anomalies
    anomaly_mask = np.abs(modified_z_scores) > threshold_sigma

    # Find continuous anomaly regions
    regions: list[InterestingRegion] = []
    in_anomaly = False
    start_idx = 0
    time_base = 1.0 / sample_rate

    for i, is_anomaly in enumerate(anomaly_mask):
        if is_anomaly and not in_anomaly:
            start_idx = i
            in_anomaly = True
        elif not is_anomaly and in_anomaly:
            end_idx = i

            # Calculate significance
            anomaly_scores = modified_z_scores[start_idx:end_idx]
            significance = min(1.0, np.max(np.abs(anomaly_scores)) / (threshold_sigma * 3))

            regions.append(
                InterestingRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=start_idx * time_base,
                    end_time=end_idx * time_base,
                    type="anomaly",
                    significance=significance,
                    metadata={
                        "max_mad_score": float(np.max(np.abs(anomaly_scores))),
                        "threshold_sigma": threshold_sigma,
                    },
                )
            )
            in_anomaly = False

    # Handle anomaly at end
    if in_anomaly:
        end_idx = len(signal)
        anomaly_scores = modified_z_scores[start_idx:end_idx]
        significance = min(1.0, np.max(np.abs(anomaly_scores)) / (threshold_sigma * 3))
        regions.append(
            InterestingRegion(
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=start_idx * time_base,
                end_time=end_idx * time_base,
                type="anomaly",
                significance=significance,
                metadata={
                    "max_mad_score": float(np.max(np.abs(anomaly_scores))),
                    "threshold_sigma": threshold_sigma,
                },
            )
        )

    return regions


def _detect_pattern_changes(
    signal: NDArray[np.float64],
    sample_rate: float,
) -> list[InterestingRegion]:
    """Detect pattern changes using windowed variance analysis.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz

    Returns:
        List of pattern change regions
    """
    # Use sliding window to detect variance changes
    window_size = min(100, len(signal) // 10)

    if window_size < 10:
        return []

    # Calculate windowed variance
    variances = np.zeros(len(signal) - window_size + 1)
    for i in range(len(variances)):
        variances[i] = np.var(signal[i : i + window_size])

    # Detect changes in variance
    if len(variances) < 2:
        return []

    variance_gradient = np.gradient(variances)
    threshold = np.std(variance_gradient) * 2.0

    change_mask = np.abs(variance_gradient) > threshold

    # Find change regions
    regions: list[InterestingRegion] = []
    time_base = 1.0 / sample_rate

    for i in range(len(change_mask)):
        if change_mask[i]:
            start_idx = i
            end_idx = min(i + window_size, len(signal))

            # Calculate significance
            significance = min(1.0, np.abs(variance_gradient[i]) / (threshold * 5))

            regions.append(
                InterestingRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=start_idx * time_base,
                    end_time=end_idx * time_base,
                    type="pattern_change",
                    significance=significance,
                    metadata={
                        "variance_change": float(variance_gradient[i]),
                        "threshold": threshold,
                    },
                )
            )

    return regions


__all__ = [
    "InterestingRegion",
    "calculate_grid_spacing",
    "calculate_optimal_x_window",
    "calculate_optimal_y_range",
    "decimate_for_display",
    "detect_interesting_regions",
    "optimize_db_range",
]
