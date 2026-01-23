"""Eye diagram metrics and measurements.

This module provides measurements on eye diagrams including height,
width, Q-factor, and crossing percentage.


Example:
    >>> from oscura.analyzers.eye.metrics import eye_height, eye_width, q_factor
    >>> height = eye_height(eye_diagram)
    >>> width = eye_width(eye_diagram)
    >>> q = q_factor(eye_diagram)

References:
    IEEE 802.3: Ethernet Physical Layer Specifications
    OIF CEI: Common Electrical I/O
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.analyzers.eye.diagram import EyeDiagram


@dataclass
class EyeMetrics:
    """Complete eye diagram measurement results.

    Attributes:
        height: Eye height in volts.
        height_at_ber: Eye height at specified BER.
        width: Eye width in UI.
        width_at_ber: Eye width at specified BER.
        q_factor: Signal quality factor.
        crossing_percent: Crossing percentage (ideal = 50%).
        mean_high: Mean logic high level.
        mean_low: Mean logic low level.
        sigma_high: Standard deviation of high level.
        sigma_low: Standard deviation of low level.
        snr: Signal-to-noise ratio in dB.
        ber_estimate: Estimated BER from Q-factor.
    """

    height: float
    height_at_ber: float | None
    width: float
    width_at_ber: float | None
    q_factor: float
    crossing_percent: float
    mean_high: float
    mean_low: float
    sigma_high: float
    sigma_low: float
    snr: float
    ber_estimate: float


def eye_height(
    eye: EyeDiagram,
    *,
    position: float = 0.5,
    ber: float | None = None,
) -> float:
    """Measure vertical eye opening (eye height).

    Measures the vertical distance between logic levels at the
    specified horizontal position within the unit interval.

    Args:
        eye: Eye diagram data.
        position: Horizontal position in UI (0.0 to 1.0, default 0.5 = center).
        ber: If specified, calculate height at this BER using Gaussian extrapolation.

    Returns:
        Eye height in volts (or input units).

    Example:
        >>> height = eye_height(eye)
        >>> print(f"Eye height: {height * 1e3:.2f} mV")

    References:
        IEEE 802.3 Clause 68: 10GBASE-T PHY
    """
    # Get samples at specified position
    samples_per_ui = eye.samples_per_ui
    position_idx = int(position * samples_per_ui) % len(eye.time_axis)

    # Use global threshold to separate high from low logic levels
    all_data = eye.data.flatten()
    low_level = np.percentile(all_data, 10)
    high_level = np.percentile(all_data, 90)
    threshold = (low_level + high_level) / 2

    # Extract voltage values at this position from all traces
    voltages = eye.data[:, position_idx]
    high_voltages = voltages[voltages > threshold]
    low_voltages = voltages[voltages <= threshold]

    # If no eye opening at this position, search for a better position
    if len(high_voltages) == 0 or len(low_voltages) == 0:
        # Search all positions for one with both high and low samples
        for idx in range(len(eye.time_axis)):
            v = eye.data[:, idx]
            h_v = v[v > threshold]
            l_v = v[v <= threshold]
            if len(h_v) > 0 and len(l_v) > 0:
                # Found a position with eye opening
                high_voltages = h_v
                low_voltages = l_v
                break
        else:
            # No position found with eye opening
            return np.nan

    if ber is None:
        # Simple min-max eye height
        min_high = np.min(high_voltages)
        max_low = np.max(low_voltages)
        return max(0.0, min_high - max_low)  # type: ignore[no-any-return]

    else:
        # BER-extrapolated eye height
        mu_high = np.mean(high_voltages)
        mu_low = np.mean(low_voltages)
        sigma_high = np.std(high_voltages)
        sigma_low = np.std(low_voltages)

        if sigma_high <= 0 or sigma_low <= 0:
            return mu_high - mu_low  # type: ignore[no-any-return]

        # Q-factor for BER
        q = np.sqrt(2) * special.erfcinv(2 * ber)

        # Eye height at BER = (mu_high - q*sigma_high) - (mu_low + q*sigma_low)
        height = (mu_high - q * sigma_high) - (mu_low + q * sigma_low)

        return max(0.0, height)  # type: ignore[no-any-return]


def eye_width(
    eye: EyeDiagram,
    *,
    level: float = 0.5,
    ber: float | None = None,
) -> float:
    """Measure horizontal eye opening (eye width).

    Measures the horizontal opening at the decision threshold level.

    Args:
        eye: Eye diagram data.
        level: Vertical level as fraction (0.0 = low, 1.0 = high, default 0.5).
        ber: If specified, calculate width at this BER.

    Returns:
        Eye width in UI (0.0 to 1.0).

    Example:
        >>> width = eye_width(eye)
        >>> print(f"Eye width: {width:.3f} UI")

    References:
        IEEE 802.3 Clause 68
    """
    data = eye.data

    # Calculate global threshold to separate logic levels
    all_data = data.flatten()
    low_level = np.percentile(all_data, 10)
    high_level = np.percentile(all_data, 90)
    global_threshold = (low_level + high_level) / 2

    # For a 2-UI eye, we need to find the eye opening across all time positions
    # We look for the widest region where all traces are separated
    samples_per_ui = eye.samples_per_ui

    # Calculate separation at each time point
    separations = []
    time_indices = []

    for i in range(len(eye.time_axis)):
        voltages = data[:, i]
        high_v = voltages[voltages > global_threshold]
        low_v = voltages[voltages <= global_threshold]

        if len(high_v) > 0 and len(low_v) > 0:
            # Measure separation between distributions
            separation = np.min(high_v) - np.max(low_v)
            if separation > 0:
                separations.append(separation)
                time_indices.append(i)

    if len(separations) == 0:
        return np.nan

    # Find contiguous region with good separation
    if len(time_indices) < 2:
        return float(len(time_indices)) / samples_per_ui

    # Find the widest contiguous region
    diffs = np.diff(time_indices)
    gaps = np.where(diffs > 1)[0]

    if len(gaps) == 0:
        # All contiguous
        width_samples = len(time_indices)
    else:
        # Find longest contiguous segment
        segments = []
        start = 0
        for gap in gaps:
            segments.append(gap + 1 - start)
            start = gap + 1
        segments.append(len(time_indices) - start)
        width_samples = max(segments)

    # Width in UI (can be > 1.0 for 2-UI eyes)
    width_ui = width_samples / samples_per_ui
    # Clamp to 1.0 for single UI measurement
    width_ui = min(1.0, width_ui)

    # Apply BER margin if requested
    if ber is not None and width_ui > 0:
        q = np.sqrt(2) * special.erfcinv(2 * ber)
        # Reduce width by jitter margin (rough approximation)
        jitter_reduction = 0.1 * q / 7.0  # Scale by Q/7 (Q=7 is ~1e-12 BER)
        width_ui = max(0.0, width_ui - jitter_reduction)

    return max(0.0, min(1.0, width_ui))


def q_factor(eye: EyeDiagram, *, position: float = 0.5) -> float:
    """Calculate Q-factor from eye diagram.

    Q-factor measures signal quality:
    Q = (mu_high - mu_low) / (sigma_high + sigma_low)

    Higher Q indicates cleaner eye with better BER margin.

    Args:
        eye: Eye diagram data.
        position: Horizontal position in UI for measurement.

    Returns:
        Q-factor value.

    Example:
        >>> q = q_factor(eye)
        >>> print(f"Q-factor: {q:.2f}")

    References:
        IEEE 802.3 Clause 52
    """
    samples_per_ui = eye.samples_per_ui
    position_idx = int(position * samples_per_ui) % len(eye.time_axis)

    # Use global threshold to separate high from low logic levels
    all_data = eye.data.flatten()
    low_level = np.percentile(all_data, 10)
    high_level = np.percentile(all_data, 90)
    threshold = (low_level + high_level) / 2

    voltages = eye.data[:, position_idx]
    high_voltages = voltages[voltages > threshold]
    low_voltages = voltages[voltages <= threshold]

    # If no eye opening at this position, search for a better position
    if len(high_voltages) < 2 or len(low_voltages) < 2:
        # Search all positions for one with both high and low samples
        for idx in range(len(eye.time_axis)):
            v = eye.data[:, idx]
            h_v = v[v > threshold]
            l_v = v[v <= threshold]
            if len(h_v) >= 2 and len(l_v) >= 2:
                # Found a position with eye opening
                high_voltages = h_v
                low_voltages = l_v
                break
        else:
            # No position found with eye opening
            return np.nan

    mu_high = np.mean(high_voltages)
    mu_low = np.mean(low_voltages)
    sigma_high = np.std(high_voltages)
    sigma_low = np.std(low_voltages)

    denominator = sigma_high + sigma_low

    if denominator <= 0:
        return np.inf if mu_high > mu_low else np.nan

    q = (mu_high - mu_low) / denominator

    return q  # type: ignore[no-any-return]


def crossing_percentage(eye: EyeDiagram) -> float:
    """Measure eye crossing percentage.

    The crossing percentage indicates where the eye crosses
    vertically. Ideal is 50% (equal rise/fall times).
    Deviation indicates duty cycle distortion.

    Args:
        eye: Eye diagram data.

    Returns:
        Crossing percentage (0.0 to 100.0).

    Example:
        >>> xing = crossing_percentage(eye)
        >>> print(f"Crossing: {xing:.1f}%")

    References:
        OIF CEI 3.0 Section 5.3
    """
    data = eye.data
    samples_per_ui = eye.samples_per_ui

    # Find voltage range
    all_low = np.percentile(data, 5)
    all_high = np.percentile(data, 95)
    amplitude = all_high - all_low

    if amplitude <= 0:
        return np.nan

    # Find crossing points (where traces cross the center time)
    # Look at the rising and falling edges
    center_idx = samples_per_ui // 2

    # Extract crossing voltages (at or near 0.5 UI and 1.5 UI)
    crossing_voltages = []

    for trace in data:
        # Find zero-crossings in derivative (transitions)
        diff = np.diff(trace)

        # Find rising crossings
        rising_mask = (diff[:-1] > 0) & (diff[1:] > 0)
        rising_idx = np.where(rising_mask)[0]

        for idx in rising_idx:
            if abs(idx - center_idx) < samples_per_ui // 4:
                crossing_voltages.append(trace[idx])

        # Find falling crossings
        falling_mask = (diff[:-1] < 0) & (diff[1:] < 0)
        falling_idx = np.where(falling_mask)[0]

        for idx in falling_idx:
            if abs(idx - center_idx) < samples_per_ui // 4:
                crossing_voltages.append(trace[idx])

    if len(crossing_voltages) < 2:
        # Fall back to simple median crossing level
        np.percentile(data, 50)
        return 50.0

    crossing_voltage = np.mean(crossing_voltages)

    # Calculate crossing percentage
    crossing_percent = (crossing_voltage - all_low) / amplitude * 100

    return crossing_percent  # type: ignore[no-any-return]


def eye_contour(
    eye: EyeDiagram,
    ber_levels: list[float] | None = None,
) -> dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Generate eye contour polygons at various BER levels.

    Creates nested contours showing the eye opening at different
    BER levels, useful for margin analysis.

    Args:
        eye: Eye diagram data.
        ber_levels: List of BER levels (default: [1e-3, 1e-6, 1e-9, 1e-12]).

    Returns:
        Dictionary mapping BER to (time_ui, voltage) contour arrays.

    Example:
        >>> contours = eye_contour(eye)
        >>> for ber, (t, v) in contours.items():
        ...     print(f"BER {ber:.0e}: {len(t)} points")

    References:
        OIF CEI: Eye Contour Methodology
    """
    if ber_levels is None:
        ber_levels = [1e-3, 1e-6, 1e-9, 1e-12]

    contours: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    # Use global threshold to separate high from low logic levels
    # Use mean of 10th and 90th percentiles to handle skewed distributions
    all_data = eye.data.flatten()
    low_level = np.percentile(all_data, 10)
    high_level = np.percentile(all_data, 90)
    global_threshold = (low_level + high_level) / 2

    for ber in ber_levels:
        # Q-factor for this BER
        q = np.sqrt(2) * special.erfcinv(2 * ber)

        upper_times = []
        upper_voltages = []
        lower_times = []
        lower_voltages = []

        # Calculate eye opening at each time position across all UIs
        for i in range(len(eye.time_axis)):
            t_ui = eye.time_axis[i]
            voltages = eye.data[:, i]

            # Use global threshold
            high_v = voltages[voltages > global_threshold]
            low_v = voltages[voltages <= global_threshold]

            # Need reasonable number of both high and low samples
            if len(high_v) < 2 or len(low_v) < 2:
                continue

            # Skip if distribution is too skewed (likely transition region)
            total = len(high_v) + len(low_v)
            if len(high_v) < total * 0.2 or len(low_v) < total * 0.2:
                continue

            mu_high = np.mean(high_v)
            sigma_high = np.std(high_v)
            mu_low = np.mean(low_v)
            sigma_low = np.std(low_v)

            # Upper contour: mu_high - q * sigma_high
            upper = mu_high - q * sigma_high

            # Lower contour: mu_low + q * sigma_low
            lower = mu_low + q * sigma_low

            if upper > lower:
                upper_times.append(t_ui)
                upper_voltages.append(upper)
                lower_times.append(t_ui)
                lower_voltages.append(lower)

        if len(upper_times) > 0:
            # Create closed contour: upper trace forward, lower trace backward
            contour_times = np.concatenate([np.array(upper_times), np.array(lower_times[::-1])])
            contour_voltages = np.concatenate(
                [np.array(upper_voltages), np.array(lower_voltages[::-1])]
            )

            contours[ber] = (contour_times, contour_voltages)

    return contours


def measure_eye(
    eye: EyeDiagram,
    *,
    ber: float = 1e-12,
) -> EyeMetrics:
    """Compute comprehensive eye diagram measurements.

    Args:
        eye: Eye diagram data.
        ber: BER level for extrapolated measurements.

    Returns:
        EyeMetrics with all measurements.

    Example:
        >>> metrics = measure_eye(eye)
        >>> print(f"Height: {metrics.height * 1e3:.2f} mV")
        >>> print(f"Width: {metrics.width:.3f} UI")
        >>> print(f"Q-factor: {metrics.q_factor:.2f}")
    """
    # Get samples at center
    samples_per_ui = eye.samples_per_ui
    center_idx = samples_per_ui // 2
    center_voltages = eye.data[:, center_idx]

    # Use global threshold to separate logic levels
    all_data = eye.data.flatten()
    low_level = np.percentile(all_data, 10)
    high_level = np.percentile(all_data, 90)
    threshold = (low_level + high_level) / 2

    high_v = center_voltages[center_voltages > threshold]
    low_v = center_voltages[center_voltages <= threshold]

    if len(high_v) < 2:
        high_v = center_voltages[center_voltages >= np.percentile(center_voltages, 75)]
    if len(low_v) < 2:
        low_v = center_voltages[center_voltages <= np.percentile(center_voltages, 25)]

    mean_high = float(np.mean(high_v)) if len(high_v) > 0 else np.nan
    mean_low = float(np.mean(low_v)) if len(low_v) > 0 else np.nan
    sigma_high = float(np.std(high_v)) if len(high_v) > 0 else np.nan
    sigma_low = float(np.std(low_v)) if len(low_v) > 0 else np.nan

    # Calculate metrics
    height = eye_height(eye)
    height_at_ber = eye_height(eye, ber=ber)
    width = eye_width(eye)
    width_at_ber = eye_width(eye, ber=ber)
    q = q_factor(eye)
    xing = crossing_percentage(eye)

    # SNR
    amplitude = mean_high - mean_low
    noise_rms = np.sqrt((sigma_high**2 + sigma_low**2) / 2)
    if noise_rms > 0 and amplitude > 0:
        snr = 20 * np.log10(amplitude / noise_rms)
    else:
        snr = np.inf if amplitude > 0 else np.nan

    # BER estimate from Q-factor
    ber_estimate = 0.5 * special.erfc(q / np.sqrt(2)) if q > 0 and np.isfinite(q) else 0.5

    return EyeMetrics(
        height=height,
        height_at_ber=height_at_ber,
        width=width,
        width_at_ber=width_at_ber,
        q_factor=q,
        crossing_percent=xing,
        mean_high=mean_high,
        mean_low=mean_low,
        sigma_high=sigma_high,
        sigma_low=sigma_low,
        snr=snr,
        ber_estimate=ber_estimate,
    )


__all__ = [
    "EyeMetrics",
    "crossing_percentage",
    "eye_contour",
    "eye_height",
    "eye_width",
    "measure_eye",
    "q_factor",
]
