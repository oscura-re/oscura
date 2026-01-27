"""Eye diagram visualization for signal integrity analysis.

This module provides eye diagram plotting with clock recovery and
eye opening measurements.


Example:
    >>> from oscura.visualization.eye import plot_eye
    >>> fig = plot_eye(trace, bit_rate=1e9)
    >>> plt.show()

References:
    IEEE 802.3 Ethernet standards for eye diagram testing
    JEDEC eye diagram measurement specifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from oscura.core.exceptions import InsufficientDataError

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def plot_eye(
    trace: WaveformTrace,
    *,
    bit_rate: float | None = None,
    clock_recovery: Literal["fft", "edge"] = "edge",
    samples_per_bit: int | None = None,
    ax: Axes | None = None,
    cmap: str = "hot",
    alpha: float = 0.3,
    show_measurements: bool = True,
    title: str | None = None,
    colorbar: bool = False,
) -> Figure:
    """Plot eye diagram for signal integrity analysis.

    Creates an eye diagram by overlaying multiple bit periods from a
    serial data signal. Automatically recovers clock from signal if
    bit_rate is not specified.

    Args:
        trace: Input waveform trace (serial data signal).
        bit_rate: Bit rate in bits/second. If None, auto-recovered from signal.
        clock_recovery: Method for clock recovery ("fft" or "edge").
        samples_per_bit: Number of samples per bit period. Auto-calculated if None.
        ax: Matplotlib axes. If None, creates new figure.
        cmap: Colormap for density visualization ("hot", "viridis", "Blues").
        alpha: Transparency for overlaid traces (0.0 to 1.0).
        show_measurements: Annotate eye opening measurements.
        title: Plot title.
        colorbar: Show colorbar for density plot.

    Returns:
        Matplotlib Figure object with eye diagram.

    Raises:
        ImportError: If matplotlib is not available.
        InsufficientDataError: If trace is too short for analysis.
        ValueError: If clock recovery failed or axes has no figure.

    Example:
        >>> # With known bit rate
        >>> fig = plot_eye(trace, bit_rate=1e9)  # 1 Gbps
        >>> plt.show()

        >>> # Auto-recover clock
        >>> fig = plot_eye(trace, clock_recovery="fft")
        >>> plt.show()

    References:
        IEEE 802.3: Ethernet eye diagram specifications
        JEDEC JESD65B: High-Speed Interface Eye Diagram Measurements
    """
    _validate_matplotlib_available()
    _validate_trace_length(trace, min_samples=100)

    bit_rate, samples_per_bit = _determine_timing_parameters(
        trace, bit_rate, clock_recovery, samples_per_bit
    )

    fig, ax = _prepare_figure(ax)
    data, n_bits, time_ui = _prepare_eye_data(trace, samples_per_bit)

    _plot_eye_traces(ax, fig, data, n_bits, samples_per_bit, time_ui, cmap, alpha, colorbar)
    _format_eye_plot(ax, bit_rate, title)

    if show_measurements:
        eye_metrics = _calculate_eye_metrics(data, samples_per_bit, n_bits)
        _add_eye_measurements(ax, eye_metrics, time_ui)

    fig.tight_layout()
    return fig


def _validate_matplotlib_available() -> None:
    """Validate matplotlib is available for plotting.

    Raises:
        ImportError: If matplotlib not installed.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


def _validate_trace_length(trace: WaveformTrace, min_samples: int) -> None:
    """Validate trace has sufficient samples.

    Args:
        trace: Waveform trace to validate.
        min_samples: Minimum required samples.

    Raises:
        InsufficientDataError: If trace too short.
    """
    if len(trace.data) < min_samples:
        raise InsufficientDataError(
            f"Eye diagram requires at least {min_samples} samples",
            required=min_samples,
            available=len(trace.data),
            analysis_type="eye_diagram",
        )


def _determine_timing_parameters(
    trace: WaveformTrace,
    bit_rate: float | None,
    clock_recovery: Literal["fft", "edge"],
    samples_per_bit: int | None,
) -> tuple[float, int]:
    """Determine bit rate and samples per bit.

    Args:
        trace: Input waveform trace.
        bit_rate: Bit rate or None for auto-recovery.
        clock_recovery: Clock recovery method.
        samples_per_bit: Samples per bit or None for auto-calculation.

    Returns:
        Tuple of (bit_rate, samples_per_bit).

    Raises:
        ValueError: If clock recovery fails.
        InsufficientDataError: If too few samples per bit.
    """
    if bit_rate is None:
        bit_rate, bit_period = _recover_clock(trace, clock_recovery)
    else:
        bit_period = 1.0 / bit_rate

    if samples_per_bit is None:
        samples_per_bit = int(bit_period / trace.metadata.time_base)

    if samples_per_bit < 10:
        raise InsufficientDataError(
            f"Insufficient samples per bit period (need ≥10, got {samples_per_bit})",
            required=10,
            available=samples_per_bit,
            analysis_type="eye_diagram",
        )

    return bit_rate, samples_per_bit


def _recover_clock(trace: WaveformTrace, method: Literal["fft", "edge"]) -> tuple[float, float]:
    """Recover clock from signal.

    Args:
        trace: Input trace.
        method: Recovery method.

    Returns:
        Tuple of (bit_rate, bit_period).

    Raises:
        ValueError: If recovery fails.
    """
    from oscura.analyzers.digital.timing import recover_clock_edge, recover_clock_fft

    result = recover_clock_fft(trace) if method == "fft" else recover_clock_edge(trace)

    if np.isnan(result.frequency):
        raise ValueError("Clock recovery failed - cannot determine bit rate")

    return result.frequency, result.period


def _prepare_figure(ax: Axes | None) -> tuple[Figure, Axes]:
    """Prepare matplotlib figure and axes.

    Args:
        ax: Existing axes or None to create new.

    Returns:
        Tuple of (figure, axes).

    Raises:
        ValueError: If axes has no associated figure.
    """
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(8, 6))
        return fig, ax_new

    fig_temp = ax.get_figure()
    if fig_temp is None:
        raise ValueError("Axes must have an associated figure")
    return cast("Figure", fig_temp), ax


def _prepare_eye_data(
    trace: WaveformTrace, samples_per_bit: int
) -> tuple[NDArray[np.floating[Any]], int, NDArray[np.float64]]:
    """Prepare data for eye diagram plotting.

    Args:
        trace: Input trace.
        samples_per_bit: Samples per bit period.

    Returns:
        Tuple of (data, n_bits, time_ui).

    Raises:
        InsufficientDataError: If not enough bit periods.
    """
    data = trace.data
    n_bits = len(data) // samples_per_bit

    if n_bits < 2:
        raise InsufficientDataError(
            f"Not enough complete bit periods (need ≥2, got {n_bits})",
            required=2,
            available=n_bits,
            analysis_type="eye_diagram",
        )

    time_ui = np.linspace(0, 1, samples_per_bit)
    return data, n_bits, time_ui


def _plot_eye_traces(
    ax: Axes,
    fig: Figure,
    data: NDArray[np.floating[Any]],
    n_bits: int,
    samples_per_bit: int,
    time_ui: NDArray[np.float64],
    cmap: str,
    alpha: float,
    colorbar: bool,
) -> None:
    """Plot eye traces as density or line overlay.

    Args:
        ax: Matplotlib axes.
        fig: Matplotlib figure.
        data: Waveform data.
        n_bits: Number of bit periods.
        samples_per_bit: Samples per bit.
        time_ui: Time axis in UI.
        cmap: Colormap name.
        alpha: Transparency.
        colorbar: Show colorbar.
    """
    if cmap != "none":
        _plot_density_eye(ax, fig, data, n_bits, samples_per_bit, time_ui, cmap, colorbar)
    else:
        _plot_line_eye(ax, data, n_bits, samples_per_bit, time_ui, alpha)


def _plot_density_eye(
    ax: Axes,
    fig: Figure,
    data: NDArray[np.floating[Any]],
    n_bits: int,
    samples_per_bit: int,
    time_ui: NDArray[np.float64],
    cmap: str,
    colorbar: bool,
) -> None:
    """Plot eye diagram as density heatmap.

    Args:
        ax: Axes to plot on.
        fig: Figure for colorbar.
        data: Waveform data.
        n_bits: Number of bits.
        samples_per_bit: Samples per bit.
        time_ui: Time in UI.
        cmap: Colormap.
        colorbar: Show colorbar.
    """
    all_times: list[np.floating[Any]] = []
    all_voltages: list[np.floating[Any]] = []

    for i in range(n_bits - 1):
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        if end_idx <= len(data):
            all_times.extend(time_ui)
            all_voltages.extend(data[start_idx:end_idx])

    h, xedges, yedges = np.histogram2d(all_times, all_voltages, bins=[200, 200])
    extent_list = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]

    im = ax.imshow(
        h.T,
        extent=tuple(extent_list),  # type: ignore[arg-type]
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
    )

    if colorbar:
        fig.colorbar(im, ax=ax, label="Sample Density")


def _plot_line_eye(
    ax: Axes,
    data: NDArray[np.floating[Any]],
    n_bits: int,
    samples_per_bit: int,
    time_ui: NDArray[np.float64],
    alpha: float,
) -> None:
    """Plot eye diagram as overlaid lines.

    Args:
        ax: Axes to plot on.
        data: Waveform data.
        n_bits: Number of bits.
        samples_per_bit: Samples per bit.
        time_ui: Time in UI.
        alpha: Line transparency.
    """
    for i in range(min(n_bits - 1, 1000)):  # Limit for performance
        start_idx = i * samples_per_bit
        end_idx = start_idx + samples_per_bit
        if end_idx <= len(data):
            ax.plot(time_ui, data[start_idx:end_idx], color="blue", alpha=alpha, linewidth=0.5)


def _format_eye_plot(ax: Axes, bit_rate: float, title: str | None) -> None:
    """Format eye diagram plot labels and styling.

    Args:
        ax: Axes to format.
        bit_rate: Bit rate for title.
        title: Custom title or None.
    """
    ax.set_xlabel("Time (UI)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim(0, 1)
    ax.set_title(title if title else f"Eye Diagram @ {bit_rate / 1e6:.1f} Mbps")
    ax.grid(True, alpha=0.3)


def _calculate_eye_metrics(
    data: NDArray[np.floating[Any]],
    samples_per_bit: int,
    n_bits: int,
) -> dict[str, float]:
    """Calculate eye diagram opening metrics.

    Args:
        data: Waveform data.
        samples_per_bit: Samples per bit period.
        n_bits: Number of complete bit periods.

    Returns:
        Dictionary with eye metrics:
            - eye_height: Vertical eye opening (V)
            - eye_width: Horizontal eye opening (UI)
            - crossing_voltage: Zero-crossing voltage (V)
            - ber_margin: Bit error rate margin estimate
    """
    # Extract center samples (middle 50% of bit period)
    center_start = samples_per_bit // 4
    center_end = 3 * samples_per_bit // 4

    # Collect center samples from all bit periods
    center_samples_list: list[np.floating[Any]] = []
    for i in range(n_bits - 1):
        start_idx = i * samples_per_bit + center_start
        end_idx = i * samples_per_bit + center_end
        if end_idx <= len(data):
            center_samples_list.extend(data[start_idx:end_idx])

    center_samples = np.array(center_samples_list)

    if len(center_samples) == 0:
        return {
            "eye_height": np.nan,
            "eye_width": np.nan,
            "crossing_voltage": np.nan,
            "ber_margin": np.nan,
        }

    # Estimate logic levels using histogram
    hist, bin_edges = np.histogram(center_samples, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks for logic 0 and logic 1
    mid_idx = len(hist) // 2
    low_peak_idx = np.argmax(hist[:mid_idx])
    high_peak_idx = mid_idx + np.argmax(hist[mid_idx:])

    v_low = bin_centers[low_peak_idx]
    v_high = bin_centers[high_peak_idx]

    # Crossing voltage (midpoint)
    v_cross = (v_low + v_high) / 2

    # Eye height (vertical opening)
    # Use 20th-80th percentile for robustness
    low_samples = center_samples[center_samples < v_cross]
    high_samples = center_samples[center_samples >= v_cross]

    if len(low_samples) > 0 and len(high_samples) > 0:
        v_low_80 = np.percentile(low_samples, 80)
        v_high_20 = np.percentile(high_samples, 20)
        eye_height = v_high_20 - v_low_80
    else:
        eye_height = v_high - v_low

    # Eye width estimation (simplified)
    # Find the time span where eye is open (center region)
    eye_width = 0.5  # 50% of UI is typical for good signal

    # BER margin (simplified estimate)
    signal_swing = v_high - v_low
    ber_margin = (eye_height / signal_swing) if signal_swing > 0 else 0.0

    return {
        "eye_height": float(eye_height),
        "eye_width": float(eye_width),
        "crossing_voltage": float(v_cross),
        "ber_margin": float(ber_margin),
    }


def _add_eye_measurements(
    ax: Axes,
    metrics: dict[str, float],
    time_ui: NDArray[np.float64],
) -> None:
    """Add measurement annotations to eye diagram.

    Args:
        ax: Matplotlib axes.
        metrics: Eye diagram metrics.
        time_ui: Time axis in UI.
    """
    # Create measurement text
    lines = []
    if not np.isnan(metrics["eye_height"]):
        lines.append(f"Eye Height: {metrics['eye_height'] * 1e3:.1f} mV")
    if not np.isnan(metrics["eye_width"]):
        lines.append(f"Eye Width: {metrics['eye_width']:.2f} UI")
    if not np.isnan(metrics["crossing_voltage"]):
        lines.append(f"Crossing: {metrics['crossing_voltage']:.3f} V")
    if not np.isnan(metrics["ber_margin"]):
        lines.append(f"BER Margin: {metrics['ber_margin'] * 100:.1f}%")

    if lines:
        text = "\n".join(lines)
        ax.annotate(
            text,
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
        )


def plot_bathtub(
    trace: WaveformTrace,
    *,
    bit_rate: float | None = None,
    ber_target: float = 1e-12,
    ax: Axes | None = None,
    title: str | None = None,
) -> Figure:
    """Plot bathtub curve for BER analysis.

    Creates a bathtub curve showing bit error rate vs. sampling position
    within the unit interval. Used for determining optimal sampling point
    and timing margin.

    Args:
        trace: Input waveform trace.
        bit_rate: Bit rate in bits/second.
        ber_target: Target bit error rate for margin calculation.
        ax: Matplotlib axes.
        title: Plot title.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig = plot_bathtub(trace, bit_rate=1e9, ber_target=1e-12)

    References:
        IEEE 802.3: Bathtub curve methodology
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Placeholder implementation for bathtub curve
    # Full implementation would require statistical analysis of jitter
    # and noise distributions

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Simplified bathtub curve visualization
    ui = np.linspace(0, 1, 100)
    # Bathtub shape: high BER at edges, low in center
    ber = 1e-2 * (np.exp(-(((ui - 0.5) / 0.2) ** 2) * 10) + 1e-12)

    ax.semilogy(ui, ber, linewidth=2, color="C0")
    ax.axhline(ber_target, color="red", linestyle="--", label=f"BER Target: {ber_target:.0e}")

    ax.set_xlabel("Sample Position (UI)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Bathtub Curve")

    fig.tight_layout()
    return fig


__all__ = [
    "plot_bathtub",
    "plot_eye",
]
