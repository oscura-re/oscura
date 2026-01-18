"""Interactive visualization features.

This module provides interactive plotting capabilities including zoom,
pan, cursors, and specialized plot types.


Example:
    >>> from oscura.visualization.interactive import (
    ...     plot_with_cursors, plot_phase, plot_bode,
    ...     plot_waterfall, plot_histogram
    ... )
    >>> fig, ax = plot_with_cursors(trace)
    >>> plot_bode(frequencies, magnitude, phase)

References:
    matplotlib interactive features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

from oscura.core.types import WaveformTrace

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Cursor, MultiCursor, SpanSelector  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class CursorMeasurement:
    """Measurement result from cursors.

    Attributes:
        x1: First cursor X position.
        x2: Second cursor X position.
        y1: First cursor Y position.
        y2: Second cursor Y position.
        delta_x: X difference (x2 - x1).
        delta_y: Y difference (y2 - y1).
        frequency: 1/delta_x if delta_x > 0.
        slope: delta_y/delta_x if delta_x != 0.

    References:
        VIS-008
    """

    x1: float
    x2: float
    y1: float
    y2: float
    delta_x: float
    delta_y: float
    frequency: float | None = None
    slope: float | None = None


@dataclass
class ZoomState:
    """Current zoom/pan state.

    Attributes:
        xlim: Current X-axis limits.
        ylim: Current Y-axis limits.
        history: Stack of previous zoom states.
        home_xlim: Original X-axis limits.
        home_ylim: Original Y-axis limits.

    References:
        VIS-007
    """

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    history: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    home_xlim: tuple[float, float] | None = None
    home_ylim: tuple[float, float] | None = None


def enable_zoom_pan(
    ax: Axes,
    *,
    enable_zoom: bool = True,
    enable_pan: bool = True,
    zoom_factor: float = 1.5,
) -> ZoomState:
    """Enable interactive zoom and pan on an axes.

    Adds scroll wheel zoom and click-drag pan functionality.

    Args:
        ax: Matplotlib axes to enable zoom/pan on.
        enable_zoom: Enable scroll wheel zoom.
        enable_pan: Enable click-drag pan.
        zoom_factor: Zoom factor per scroll step.

    Returns:
        ZoomState object tracking zoom history.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(trace.time_vector, trace.data)
        >>> state = enable_zoom_pan(ax)

    References:
        VIS-007
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    # Store initial state
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    state = ZoomState(
        xlim=xlim,
        ylim=ylim,
        home_xlim=xlim,
        home_ylim=ylim,
    )

    def on_scroll(event):  # type: ignore[no-untyped-def]
        if event.inaxes != ax:
            return

        # Get mouse position
        x_data = event.xdata
        y_data = event.ydata

        if x_data is None or y_data is None:
            return

        # Determine zoom direction
        if event.button == "up":
            factor = 1 / zoom_factor
        elif event.button == "down":
            factor = zoom_factor
        else:
            return

        # Save current state
        state.history.append((state.xlim, state.ylim))

        # Calculate new limits centered on mouse position
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * factor

        rel_x = (x_data - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (y_data - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = (
            x_data - new_width * rel_x,
            x_data + new_width * (1 - rel_x),
        )
        new_ylim = (
            y_data - new_height * rel_y,
            y_data + new_height * (1 - rel_y),
        )

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        state.xlim = new_xlim
        state.ylim = new_ylim

        ax.figure.canvas.draw_idle()

    if enable_zoom:
        ax.figure.canvas.mpl_connect("scroll_event", on_scroll)

    # Pan state
    pan_active = [False]
    pan_start: list[float | None] = [None, None]

    def on_press(event):  # type: ignore[no-untyped-def]
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click
            pan_active[0] = True
            pan_start[0] = event.xdata
            pan_start[1] = event.ydata

    def on_release(event: MouseEvent) -> None:
        pan_active[0] = False

    def on_motion(event: MouseEvent) -> None:
        if not pan_active[0]:
            return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if pan_start[0] is None or pan_start[1] is None:
            return

        dx = pan_start[0] - event.xdata
        dy = pan_start[1] - event.ydata

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        new_xlim = (cur_xlim[0] + dx, cur_xlim[1] + dx)
        new_ylim = (cur_ylim[0] + dy, cur_ylim[1] + dy)

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        state.xlim = new_xlim
        state.ylim = new_ylim

        ax.figure.canvas.draw_idle()

    if enable_pan:
        ax.figure.canvas.mpl_connect("button_press_event", on_press)
        ax.figure.canvas.mpl_connect("button_release_event", on_release)  # type: ignore[arg-type]
        ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)  # type: ignore[arg-type]

    return state


def plot_with_cursors(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    sample_rate: float | None = None,
    cursor_type: Literal["vertical", "horizontal", "cross"] = "cross",
    ax: Axes | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes, Cursor]:
    """Plot waveform with interactive measurement cursors.

    Args:
        trace: Input trace or numpy array.
        sample_rate: Sample rate (required for arrays).
        cursor_type: Type of cursor lines.
        ax: Existing axes to plot on.
        **plot_kwargs: Additional arguments to plot().

    Returns:
        Tuple of (figure, axes, cursor widget).

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig, ax, cursor = plot_with_cursors(trace)
        >>> plt.show()

    References:
        VIS-008
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    # Get data and time vector
    if isinstance(trace, WaveformTrace):
        data = trace.data
        time = trace.time_vector
    else:
        data = np.asarray(trace)
        if sample_rate is None:
            sample_rate = 1.0
        time = np.arange(len(data)) / sample_rate

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig_temp = ax.figure
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Plot data
    ax.plot(time, data, **plot_kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    # Create cursor
    if cursor_type == "vertical":
        cursor = Cursor(ax, useblit=True, color="red", linewidth=1, vertOn=True, horizOn=False)
    elif cursor_type == "horizontal":
        cursor = Cursor(ax, useblit=True, color="red", linewidth=1, vertOn=False, horizOn=True)
    else:  # cross
        cursor = Cursor(ax, useblit=True, color="red", linewidth=1)

    return fig, ax, cursor


def add_measurement_cursors(
    ax: Axes,
    *,
    color: str = "red",
    linestyle: str = "--",
) -> dict:  # type: ignore[type-arg]
    """Add dual measurement cursors to an axes.

    Click and drag to define measurement region. Returns measurement
    data in the callback.

    Args:
        ax: Axes to add cursors to.
        color: Cursor line color.
        linestyle: Cursor line style.

    Returns:
        Dictionary with cursor state and get_measurement() function.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> cursors = add_measurement_cursors(ax)
        >>> measurement = cursors['get_measurement']()
        >>> print(f"Delta X: {measurement.delta_x}")

    References:
        VIS-008
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    state: dict[str, float | None | Any] = {
        "x1": None,
        "x2": None,
        "y1": None,
        "y2": None,
        "line1": None,
        "line2": None,
    }

    def onselect(xmin: float, xmax: float) -> None:
        state["x1"] = xmin
        state["x2"] = xmax

        # Get Y values at cursor positions
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            # Type narrowing: these return ArrayLike from Line2D
            xdata_arr = np.asarray(xdata)
            ydata_arr = np.asarray(ydata)
            if len(xdata_arr) > 0:
                # Interpolate Y at cursor positions
                y1_interp: float = float(np.interp(xmin, xdata_arr, ydata_arr))
                y2_interp: float = float(np.interp(xmax, xdata_arr, ydata_arr))
                state["y1"] = y1_interp
                state["y2"] = y2_interp
                break

    span = SpanSelector(
        ax,
        onselect,
        "horizontal",
        useblit=True,
        props={"alpha": 0.3, "facecolor": color},
        interactive=True,
    )

    def get_measurement() -> CursorMeasurement | None:
        x1 = state["x1"]
        x2 = state["x2"]
        y1 = state["y1"]
        y2 = state["y2"]

        if (
            x1 is None
            or x2 is None
            or not isinstance(x1, int | float)
            or not isinstance(x2, int | float)
        ):
            return None

        delta_x = x2 - x1
        y1_val = float(y1) if y1 is not None else 0.0
        y2_val = float(y2) if y2 is not None else 0.0
        delta_y = y2_val - y1_val

        return CursorMeasurement(
            x1=x1,
            x2=x2,
            y1=y1_val,
            y2=y2_val,
            delta_x=delta_x,
            delta_y=delta_y,
            frequency=1 / delta_x if delta_x > 0 else None,
            slope=delta_y / delta_x if delta_x != 0 else None,
        )

    return {
        "span": span,
        "state": state,
        "get_measurement": get_measurement,
    }


def plot_phase(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]] | None = None,
    *,
    delay: int = 1,
    delay_samples: int | None = None,
    ax: Axes | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create phase plot (X-Y plot) of two signals.

    Plots trace1 on X-axis vs trace2 on Y-axis, useful for
    visualizing phase relationships and Lissajous figures.
    If trace2 is not provided, creates a self-phase plot using
    time-delayed version of trace1.

    Args:
        trace1: Signal for X-axis.
        trace2: Signal for Y-axis. If None, uses delayed trace1.
        delay: Sample delay for self-phase plot (when trace2=None).
        delay_samples: Alias for delay parameter.
        ax: Existing axes to plot on.
        **plot_kwargs: Additional arguments to plot().

    Returns:
        Tuple of (figure, axes).

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig, ax = plot_phase(signal_x, signal_y)
        >>> plt.show()
        >>> # Self-phase plot
        >>> fig, ax = plot_phase(signal, delay_samples=10)

    References:
        VIS-009
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    # Handle delay_samples alias
    if delay_samples is not None:
        delay = delay_samples

    # Get data
    data1 = trace1.data if isinstance(trace1, WaveformTrace) else np.asarray(trace1)

    # If trace2 not provided, create self-phase plot with delay
    if trace2 is None:
        data2 = np.roll(data1, -delay)
    else:
        data2 = trace2.data if isinstance(trace2, WaveformTrace) else np.asarray(trace2)

    # Ensure same length
    n = min(len(data1), len(data2))
    data1 = data1[:n]
    data2 = data2[:n]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig_temp = ax.figure
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Plot
    defaults: dict[str, Any] = {"alpha": 0.5, "marker": ".", "linestyle": "-", "markersize": 2}
    defaults.update(plot_kwargs)
    ax.plot(data1, data2, **defaults)

    # Equal aspect ratio for proper phase visualization
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Signal 1")
    ax.set_ylabel("Signal 2")
    ax.set_title("Phase Plot (X-Y)")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_bode(
    frequencies: NDArray[np.floating[Any]],
    magnitude: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
    phase: NDArray[np.floating[Any]] | None = None,
    *,
    magnitude_db: bool = True,
    phase_degrees: bool = True,
    fig: Figure | None = None,
    **plot_kwargs: Any,
) -> Figure:
    """Create Bode plot with magnitude and phase.

    Standard frequency response visualization with logarithmic
    frequency axis.

    Args:
        frequencies: Frequency array in Hz.
        magnitude: Magnitude array (linear or dB), or complex transfer function H(s).
            If complex, magnitude and phase are extracted automatically.
        phase: Phase array in radians (optional). Ignored if magnitude is complex.
        magnitude_db: If True, magnitude is already in dB. Ignored if complex input.
        phase_degrees: If True, convert phase to degrees.
        fig: Existing figure to plot on.
        **plot_kwargs: Additional arguments to plot().

    Returns:
        Matplotlib Figure object with magnitude and optionally phase axes.

    Raises:
        ImportError: If matplotlib is not available.

    Example:
        >>> # With complex transfer function
        >>> H = 1 / (1 + 1j * freqs / 1000)
        >>> fig = plot_bode(freqs, H)
        >>> ax_mag, ax_phase = fig.axes[:2]  # Access axes from figure
        >>> plt.show()

    References:
        VIS-010
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    frequencies = np.asarray(frequencies)
    magnitude = np.asarray(magnitude)

    # Handle complex transfer function input
    if np.iscomplexobj(magnitude):
        # Extract phase from complex input
        phase = np.angle(magnitude)
        # Convert to magnitude in dB
        with np.errstate(divide="ignore"):
            magnitude = 20 * np.log10(np.abs(magnitude))
            magnitude = np.nan_to_num(magnitude, neginf=-200)
    elif not magnitude_db:
        # Convert magnitude to dB if needed
        with np.errstate(divide="ignore"):
            magnitude = 20 * np.log10(np.abs(magnitude))
            magnitude = np.nan_to_num(magnitude, neginf=-200)

    # Create figure
    if phase is not None:
        if fig is None:
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        else:
            axes = fig.subplots(2, 1, sharex=True)
            ax_mag, ax_phase = axes
    else:
        if fig is None:
            fig, ax_mag = plt.subplots(figsize=(10, 5))
        else:
            ax_mag = fig.subplots()
        ax_phase = None

    # Plot magnitude
    ax_mag.semilogx(frequencies, magnitude, **plot_kwargs)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title("Bode Plot")

    # Plot phase if provided
    if phase is not None and ax_phase is not None:
        phase = np.asarray(phase)
        if phase_degrees:
            phase = np.degrees(phase)
            ylabel = "Phase (degrees)"
        else:
            ylabel = "Phase (radians)"

        ax_phase.semilogx(frequencies, phase, **plot_kwargs)
        ax_phase.set_ylabel(ylabel)
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.grid(True, which="both", alpha=0.3)
    else:
        ax_mag.set_xlabel("Frequency (Hz)")

    fig.tight_layout()

    return fig


def plot_waterfall(
    data: NDArray[np.floating[Any]],
    *,
    time_axis: NDArray[np.floating[Any]] | None = None,
    freq_axis: NDArray[np.floating[Any]] | None = None,
    sample_rate: float = 1.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create 3D waterfall plot (spectrogram with depth).

    Shows spectrum evolution over time as stacked frequency slices.

    Args:
        data: Input signal array (1D) or pre-computed spectrogram (2D).
            If 2D, treated as (n_traces, n_points) spectrogram data.
        time_axis: Time axis for signal.
        freq_axis: Frequency axis (if pre-computed).
        sample_rate: Sample rate in Hz.
        nperseg: Segment length for FFT.
        noverlap: Overlap between segments.
        cmap: Colormap for amplitude coloring.
        ax: Existing 3D axes to plot on.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (figure, axes).

    Raises:
        ImportError: If matplotlib is not available.
        TypeError: If axes is not a 3D axes.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig, ax = plot_waterfall(signal, sample_rate=1e6)
        >>> plt.show()
        >>> # With 2D precomputed data
        >>> fig, ax = plot_waterfall(spectrogram_data)

    References:
        VIS-011
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    data = np.asarray(data)

    # Check if data is 2D (precomputed spectrogram)
    if data.ndim == 2:
        # Treat as precomputed spectrogram (n_traces, n_points)
        Sxx_db = data
        n_traces, n_points = data.shape
        frequencies = freq_axis if freq_axis is not None else np.arange(n_points)
        times = time_axis if time_axis is not None else np.arange(n_traces)
    elif freq_axis is not None:
        # 1D data with explicit freq_axis means precomputed
        Sxx_db = data
        frequencies = freq_axis
        times = (
            time_axis
            if time_axis is not None
            else np.arange(Sxx_db.shape[1] if Sxx_db.ndim > 1 else 1)
        )
    else:
        # Compute spectrogram from 1D signal
        if noverlap is None:
            noverlap = nperseg // 2

        frequencies, times, Sxx = scipy_signal.spectrogram(
            data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        times = time_axis if time_axis is not None else np.arange(Sxx_db.shape[1])

    # Create 3D figure
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig_temp = ax.figure
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Create meshgrid
    T, F = np.meshgrid(times, frequencies)

    # Ensure Sxx_db matches meshgrid shape (n_frequencies, n_times)
    if Sxx_db.shape != T.shape:
        if Sxx_db.T.shape == T.shape:
            Sxx_db = Sxx_db.T
        # If still mismatched, the data dimensions may be incompatible
        # but we'll let plot_surface raise a more informative error

    # Plot surface
    # Type checking: ax must be a 3D axes at this point
    if not hasattr(ax, "plot_surface"):
        raise TypeError("Axes must be a 3D axes for waterfall plot")
    surf = ax.plot_surface(  # type: ignore[attr-defined,union-attr]
        T,
        F,
        Sxx_db,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel("Power (dB)")  # type: ignore[attr-defined]
    ax.set_title("Waterfall Plot (Spectrogram)")

    fig.colorbar(surf, ax=ax, label="Power (dB)", shrink=0.5)

    return fig, ax


def plot_histogram(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    bins: int | str | NDArray[np.floating[Any]] = "auto",
    density: bool = True,
    show_stats: bool = True,
    show_kde: bool = False,
    ax: Axes | None = None,
    save_path: str | None = None,
    show: bool = True,
    **hist_kwargs: Any,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """Create histogram plot of signal amplitude distribution.

    Optionally overlays kernel density estimate and statistics.

    Args:
        trace: Input trace or numpy array.
        bins: Number of bins or binning strategy.
        density: If True, normalize to probability density.
        show_stats: Show mean and standard deviation lines.
        show_kde: Overlay kernel density estimate.
        ax: Existing axes to plot on.
        save_path: Path to save figure. If None, figure is not saved.
        show: If True, display the figure. If False, close it.
        **hist_kwargs: Additional arguments to hist().

    Returns:
        Tuple of (Figure, Axes, statistics dict).

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If axes has no associated figure.

    Example:
        >>> fig = plot_histogram(trace, bins=50, show_kde=True)
        >>> # With save
        >>> fig = plot_histogram(trace, save_path="hist.png", show=False)

    References:
        VIS-012
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for interactive visualization")

    # Get data
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig_temp = ax.figure
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Calculate statistics
    mean = float(np.mean(data))
    std = float(np.std(data))
    median = float(np.median(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    stats = {
        "mean": mean,
        "std": std,
        "median": median,
        "min": min_val,
        "max": max_val,
        "count": len(data),
    }

    # Plot histogram
    defaults: dict[str, Any] = {"alpha": 0.7, "edgecolor": "black", "linewidth": 0.5}
    defaults.update(hist_kwargs)
    _counts, bin_edges, _patches = ax.hist(data, bins=bins, density=density, **defaults)  # type: ignore[arg-type]

    stats["bins"] = len(bin_edges) - 1

    # Show statistics lines
    if show_stats:
        ax.get_ylim()
        ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.3g}")
        ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.5, label="Mean - Std")
        ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.5, label="Mean + Std")

    # Show KDE
    if show_kde:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(data)
        x_kde = np.linspace(min_val, max_val, 200)
        y_kde = kde(x_kde)

        if density:
            ax.plot(x_kde, y_kde, "r-", linewidth=2, label="KDE")
        else:
            # Scale KDE to histogram
            bin_width = bin_edges[1] - bin_edges[0]
            ax.plot(x_kde, y_kde * len(data) * bin_width, "r-", linewidth=2, label="KDE")

    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Amplitude Distribution")
    # Only show legend if there are labeled artists
    if show_stats or show_kde:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show or close
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, stats


__all__ = [
    "CursorMeasurement",
    "ZoomState",
    "add_measurement_cursors",
    "enable_zoom_pan",
    "plot_bode",
    "plot_histogram",
    "plot_phase",
    "plot_waterfall",
    "plot_with_cursors",
]
