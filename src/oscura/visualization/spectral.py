"""Spectral visualization functions.

This module provides spectrum and spectrogram plots for
frequency-domain analysis visualization.


Example:
    >>> from oscura.visualization.spectral import plot_spectrum, plot_spectrogram
    >>> plot_spectrum(trace)
    >>> plot_spectrogram(trace)

References:
    matplotlib best practices for scientific visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def _get_fft_data(
    trace: WaveformTrace,
    fft_result: tuple[Any, Any] | None,
    window: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get FFT data, either from cache or by computing.

    Args:
        trace: Waveform trace.
        fft_result: Pre-computed FFT result.
        window: Window function name.

    Returns:
        Tuple of (frequencies, magnitudes_db).
    """
    if fft_result is not None:
        return fft_result

    from oscura.analyzers.waveform.spectral import fft

    return fft(trace, window=window)  # type: ignore[return-value]


def _scale_frequencies(
    freq: NDArray[np.float64], freq_unit: str
) -> tuple[NDArray[np.float64], float, str]:
    """Scale frequencies to appropriate unit.

    Args:
        freq: Frequency array in Hz.
        freq_unit: Requested unit or "auto".

    Returns:
        Tuple of (scaled_frequencies, divisor, unit_name).
    """
    if freq_unit == "auto":
        max_freq = freq[-1]
        freq_unit = _auto_select_freq_unit(max_freq)

    freq_divisors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    divisor = freq_divisors.get(freq_unit, 1.0)
    return freq / divisor, divisor, freq_unit


def _set_auto_ylimits(ax: Axes, mag_db: NDArray[np.float64]) -> None:
    """Set reasonable y-axis limits based on data.

    Args:
        ax: Matplotlib axes.
        mag_db: Magnitude data in dB.
    """
    valid_db = mag_db[np.isfinite(mag_db)]
    if len(valid_db) == 0:
        return

    y_max = np.max(valid_db)
    y_min = max(np.min(valid_db), y_max - 120)  # Limit dynamic range
    ax.set_ylim(y_min, y_max + 5)


def _apply_axis_limits(
    ax: Axes,
    divisor: float,
    freq_range: tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> None:
    """Apply custom axis limits if specified.

    Args:
        ax: Matplotlib axes.
        divisor: Frequency divisor for unit conversion.
        freq_range: Frequency range in Hz (will be converted to display units).
        xlim: X-axis limits in display units.
        ylim: Y-axis limits.
    """
    if freq_range is not None and len(freq_range) == 2:
        # freq_range is in Hz, convert to display units
        freq_min = freq_range[0] / divisor
        freq_max = freq_range[1] / divisor

        # For log scale, ensure minimum is positive (avoid 0 on log axis)
        if ax.get_xscale() == "log" and freq_min <= 0:
            freq_min = freq_max / 1000  # Use a small positive value

        ax.set_xlim(freq_min, freq_max)
    elif xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)


def _prepare_spectrum_data(
    trace: WaveformTrace,
    fft_result: tuple[Any, Any] | None,
    window: str,
    db_ref: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare spectrum data with FFT and dB scaling.

    Args:
        trace: Waveform trace to analyze.
        fft_result: Pre-computed FFT result or None.
        window: Window function name.
        db_ref: Reference for dB scaling or None.

    Returns:
        Tuple of (frequencies, magnitudes_db).
    """
    freq, mag_db = _get_fft_data(trace, fft_result, window)

    # Adjust dB reference if specified
    if db_ref is not None:
        mag_db = mag_db - db_ref

    return freq, mag_db


def _render_spectrum_plot(
    ax: Axes,
    freq_scaled: NDArray[np.float64],
    mag_db: NDArray[np.float64],
    freq_unit: str,
    color: str,
    title: str | None,
    log_scale: bool,
    show_grid: bool,
) -> None:
    """Render spectrum plot on axes.

    Args:
        ax: Matplotlib axes to plot on.
        freq_scaled: Scaled frequency array.
        mag_db: Magnitude array in dB.
        freq_unit: Frequency unit string.
        color: Line color.
        title: Plot title.
        log_scale: Use logarithmic frequency scale.
        show_grid: Show grid lines.
    """
    ax.plot(freq_scaled, mag_db, color=color, linewidth=0.8)
    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale("log" if log_scale else "linear")
    ax.set_title(title if title else "Magnitude Spectrum")

    if show_grid:
        ax.grid(True, alpha=0.3, which="both")


def plot_spectrum(
    trace: WaveformTrace,
    *,
    ax: Axes | None = None,
    freq_unit: str = "auto",
    db_ref: float | None = None,
    freq_range: tuple[float, float] | None = None,
    show_grid: bool = True,
    color: str = "C0",
    title: str | None = None,
    window: str = "hann",
    show: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fft_result: tuple[Any, Any] | None = None,
    log_scale: bool = True,
) -> Figure:
    """Plot magnitude spectrum.

    Args:
        trace: Waveform trace to analyze.
        ax: Matplotlib axes. If None, creates new figure.
        freq_unit: Frequency unit ("Hz", "kHz", "MHz", "auto").
        db_ref: Reference for dB scaling. If None, uses max value.
        freq_range: Frequency range (min, max) in Hz to display.
        show_grid: Show grid lines.
        color: Line color.
        title: Plot title.
        window: Window function for FFT.
        show: If True, call plt.show() to display the plot.
        save_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches. Only used if ax is None.
        xlim: X-axis limits (min, max) in selected frequency units.
        ylim: Y-axis limits (min, max) in dB.
        fft_result: Pre-computed FFT result (frequencies, magnitudes). If None, computes FFT.
        log_scale: Use logarithmic scale for frequency axis (default True).

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If axes must have an associated figure.

    Example:
        >>> import oscura as osc
        >>> trace = osc.load("signal.wfm")
        >>> fig = osc.plot_spectrum(trace, freq_unit="MHz", log_scale=True)

        >>> # With pre-computed FFT
        >>> freq, mag = osc.fft(trace)
        >>> fig = osc.plot_spectrum(trace, fft_result=(freq, mag), show=False)
        >>> fig.savefig("spectrum.png")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Figure/axes creation
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Data preparation
    freq, mag_db = _prepare_spectrum_data(trace, fft_result, window, db_ref)

    # Unit/scale selection
    freq_scaled, divisor, freq_unit = _scale_frequencies(freq, freq_unit)

    # Plotting/rendering
    _render_spectrum_plot(ax, freq_scaled, mag_db, freq_unit, color, title, log_scale, show_grid)

    # Set limits
    _set_auto_ylimits(ax, mag_db)
    _apply_axis_limits(ax, divisor, freq_range, xlim, ylim)

    # Layout/formatting
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _auto_select_time_unit(max_time: float) -> str:
    """Select appropriate time unit based on maximum time value.

    Args:
        max_time: Maximum time value in seconds.

    Returns:
        Time unit string ("s", "ms", "us", or "ns").
    """
    if max_time < 1e-6:
        return "ns"
    elif max_time < 1e-3:
        return "us"
    elif max_time < 1:
        return "ms"
    else:
        return "s"


def _auto_select_freq_unit(max_freq: float) -> str:
    """Select appropriate frequency unit based on maximum frequency.

    Args:
        max_freq: Maximum frequency in Hz.

    Returns:
        Frequency unit string ("Hz", "kHz", "MHz", or "GHz").
    """
    if max_freq >= 1e9:
        return "GHz"
    elif max_freq >= 1e6:
        return "MHz"
    elif max_freq >= 1e3:
        return "kHz"
    else:
        return "Hz"


def _get_unit_multipliers() -> tuple[dict[str, float], dict[str, float]]:
    """Get time and frequency unit multipliers.

    Returns:
        Tuple of (time_multipliers, freq_divisors).
    """
    time_mult = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    freq_div = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    return time_mult, freq_div


def _auto_color_limits(
    data: NDArray[np.float64], vmin: float | None, vmax: float | None
) -> tuple[float | None, float | None]:
    """Automatically determine color limits for spectrogram.

    Args:
        data: Spectrogram data in dB.
        vmin: Minimum dB value (if None, auto-computed).
        vmax: Maximum dB value (if None, auto-computed).

    Returns:
        Tuple of (vmin, vmax).
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax

    valid_db = data[np.isfinite(data)]
    if len(valid_db) == 0:
        return vmin, vmax

    if vmax is None:
        vmax = np.max(valid_db)
    if vmin is None:
        vmin = max(np.min(valid_db), vmax - 80)

    return vmin, vmax


def _compute_spectrogram_data(
    trace: WaveformTrace,
    window: str,
    nperseg: int | None,
    nfft: int | None,
    overlap: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute spectrogram data using STFT.

    Args:
        trace: Waveform trace to analyze.
        window: Window function name.
        nperseg: Segment length for STFT.
        nfft: FFT length (overrides nperseg if specified).
        overlap: Overlap fraction (0.0 to 1.0).

    Returns:
        Tuple of (times, frequencies, Sxx_db).
    """
    from oscura.analyzers.waveform.spectral import spectrogram

    if nfft is not None:
        nperseg = nfft
    noverlap = int(nperseg * overlap) if overlap is not None and nperseg is not None else None

    return spectrogram(trace, window=window, nperseg=nperseg, noverlap=noverlap)


def _scale_spectrogram_axes(
    times: NDArray[np.float64],
    freq: NDArray[np.float64],
    time_unit: str,
    freq_unit: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, str]:
    """Scale time and frequency axes to appropriate units.

    Args:
        times: Time array in seconds.
        freq: Frequency array in Hz.
        time_unit: Time unit ("auto" or specific).
        freq_unit: Frequency unit ("auto" or specific).

    Returns:
        Tuple of (times_scaled, freq_scaled, time_unit, freq_unit).
    """
    if time_unit == "auto":
        max_time = times[-1] if len(times) > 0 else 0
        time_unit = _auto_select_time_unit(max_time)

    if freq_unit == "auto":
        max_freq = freq[-1] if len(freq) > 0 else 0
        freq_unit = _auto_select_freq_unit(max_freq)

    time_multipliers, freq_divisors = _get_unit_multipliers()
    times_scaled = times * time_multipliers.get(time_unit, 1.0)
    freq_scaled = freq / freq_divisors.get(freq_unit, 1.0)

    return times_scaled, freq_scaled, time_unit, freq_unit


def _render_spectrogram_plot(
    ax: Axes,
    times_scaled: NDArray[np.float64],
    freq_scaled: NDArray[np.float64],
    Sxx_db: NDArray[np.float64],
    time_unit: str,
    freq_unit: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
) -> None:
    """Render spectrogram plot on axes.

    Args:
        ax: Matplotlib axes to plot on.
        times_scaled: Scaled time array.
        freq_scaled: Scaled frequency array.
        Sxx_db: Spectrogram data in dB.
        time_unit: Time unit string.
        freq_unit: Frequency unit string.
        cmap: Colormap name.
        vmin: Minimum dB value for color scaling.
        vmax: Maximum dB value for color scaling.
        title: Plot title.
    """
    # Auto color limits
    vmin, vmax = _auto_color_limits(Sxx_db, vmin, vmax)

    # Plot
    pcm = ax.pcolormesh(
        times_scaled,
        freq_scaled,
        Sxx_db,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Frequency ({freq_unit})")
    ax.set_title(title if title else "Spectrogram")

    # Colorbar
    fig = ax.get_figure()
    if fig is not None:
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Magnitude (dB)")


def plot_spectrogram(
    trace: WaveformTrace,
    *,
    ax: Axes | None = None,
    time_unit: str = "auto",
    freq_unit: str = "auto",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    window: str = "hann",
    nperseg: int | None = None,
    nfft: int | None = None,
    overlap: float | None = None,
) -> Figure:
    """Plot spectrogram (time-frequency representation).

    Args:
        trace: Waveform trace to analyze.
        ax: Matplotlib axes. If None, creates new figure.
        time_unit: Time unit ("s", "ms", "us", "auto").
        freq_unit: Frequency unit ("Hz", "kHz", "MHz", "auto").
        cmap: Colormap name.
        vmin: Minimum dB value for color scaling.
        vmax: Maximum dB value for color scaling.
        title: Plot title.
        window: Window function.
        nperseg: Segment length for STFT.
        nfft: FFT length. If specified, overrides nperseg.
        overlap: Overlap fraction (0.0 to 1.0). Default is 0.5 (50%).

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If axes must have an associated figure.

    Example:
        >>> fig = plot_spectrogram(trace)
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Figure/axes creation
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Data preparation/computation
    times, freq, Sxx_db = _compute_spectrogram_data(trace, window, nperseg, nfft, overlap)

    # Unit/scale selection
    times_scaled, freq_scaled, time_unit, freq_unit = _scale_spectrogram_axes(
        times, freq, time_unit, freq_unit
    )

    # Plotting/rendering
    _render_spectrogram_plot(
        ax, times_scaled, freq_scaled, Sxx_db, time_unit, freq_unit, cmap, vmin, vmax, title
    )

    # Layout/formatting
    fig.tight_layout()
    return fig


def plot_psd(
    trace: WaveformTrace,
    *,
    ax: Axes | None = None,
    freq_unit: str = "auto",
    show_grid: bool = True,
    color: str = "C0",
    title: str | None = None,
    window: str = "hann",
    log_scale: bool = True,
) -> Figure:
    """Plot Power Spectral Density.

    Args:
        trace: Waveform trace to analyze.
        ax: Matplotlib axes.
        freq_unit: Frequency unit.
        show_grid: Show grid lines.
        color: Line color.
        title: Plot title.
        window: Window function.
        log_scale: Use logarithmic scale for frequency axis (default True).

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If axes must have an associated figure.

    Example:
        >>> fig = plot_psd(trace)
        >>> plt.show()
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    from oscura.analyzers.waveform.spectral import psd

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Compute PSD
    freq, psd_db = psd(trace, window=window)

    # Auto-select frequency unit
    if freq_unit == "auto":
        max_freq = freq[-1]
        if max_freq >= 1e9:
            freq_unit = "GHz"
        elif max_freq >= 1e6:
            freq_unit = "MHz"
        elif max_freq >= 1e3:
            freq_unit = "kHz"
        else:
            freq_unit = "Hz"

    freq_divisors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    divisor = freq_divisors.get(freq_unit, 1.0)
    freq_scaled = freq / divisor

    # Plot
    ax.plot(freq_scaled, psd_db, color=color, linewidth=0.8)

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_xscale("log" if log_scale else "linear")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Power Spectral Density")

    if show_grid:
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def plot_fft(
    trace: WaveformTrace,
    *,
    ax: Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    title: str | None = None,
    xlabel: str = "Frequency",
    ylabel: str = "Magnitude (dB)",
    figsize: tuple[float, float] = (10, 6),
    freq_unit: str = "auto",
    log_scale: bool = True,
    show_grid: bool = True,
    color: str = "C0",
    window: str = "hann",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Plot FFT magnitude spectrum.

    Computes and plots the FFT magnitude spectrum of a waveform trace.
    This is a convenience function that combines FFT computation and visualization.

    Args:
        trace: Waveform trace to analyze and plot.
        ax: Matplotlib axes. If None, creates new figure.
        show: If True, call plt.show() to display the plot.
        save_path: Path to save the figure. If None, figure is not saved.
        title: Plot title. If None, uses default "FFT Magnitude Spectrum".
        xlabel: X-axis label (appended with frequency unit).
        ylabel: Y-axis label.
        figsize: Figure size (width, height) in inches. Only used if ax is None.
        freq_unit: Frequency unit ("Hz", "kHz", "MHz", "GHz", "auto").
        log_scale: Use logarithmic scale for frequency axis.
        show_grid: Show grid lines.
        color: Line color.
        window: Window function for FFT computation.
        xlim: X-axis limits (min, max) in selected frequency units.
        ylim: Y-axis limits (min, max) in dB.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If axes must have an associated figure.

    Example:
        >>> import oscura as osc
        >>> trace = osc.load("signal.wfm")
        >>> fig = osc.plot_fft(trace, freq_unit="MHz", show=False)
        >>> fig.savefig("spectrum.png")

        >>> # With custom styling
        >>> fig = osc.plot_fft(trace,
        ...                   title="Signal FFT",
        ...                   log_scale=True,
        ...                   xlim=(1e3, 1e6),
        ...                   ylim=(-100, 0))

    References:
        IEEE 1241-2010: Standard for Terminology and Test Methods for
        Analog-to-Digital Converters
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Setup figure and axes
    fig, ax = _setup_plot_figure(ax, figsize)

    # Plot spectrum using main plotting function
    plot_spectrum(
        trace,
        ax=ax,
        freq_unit=freq_unit,
        show_grid=show_grid,
        color=color,
        title=title if title else "FFT Magnitude Spectrum",
        window=window,
        log_scale=log_scale,
    )

    # Apply custom labels and limits
    _apply_custom_labels(ax, xlabel, ylabel)
    _apply_axis_limits_simple(ax, xlim, ylim)

    # Output handling
    _handle_plot_output(fig, save_path, show)

    return fig


def _setup_plot_figure(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Setup figure and axes for plotting.

    Args:
        ax: Existing axes or None.
        figsize: Figure size if creating new.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        return plt.subplots(figsize=figsize)

    fig_temp = ax.get_figure()
    if fig_temp is None:
        raise ValueError("Axes must have an associated figure")
    return cast("Figure", fig_temp), ax


def _apply_custom_labels(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Apply custom labels to plot axes.

    Args:
        ax: Matplotlib axes.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    if xlabel != "Frequency":
        current_label = ax.get_xlabel()
        if "(" in current_label and ")" in current_label:
            unit = current_label[current_label.find("(") : current_label.find(")") + 1]
            ax.set_xlabel(f"{xlabel} {unit}")
        else:
            ax.set_xlabel(xlabel)

    if ylabel != "Magnitude (dB)":
        ax.set_ylabel(ylabel)


def _apply_axis_limits_simple(
    ax: Axes, xlim: tuple[float, float] | None, ylim: tuple[float, float] | None
) -> None:
    """Apply axis limits if specified.

    Args:
        ax: Matplotlib axes.
        xlim: X-axis limits.
        ylim: Y-axis limits.
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _handle_plot_output(fig: Figure, save_path: str | None, show: bool) -> None:
    """Handle plot output (save and/or show).

    Args:
        fig: Matplotlib figure.
        save_path: Path to save figure.
        show: Whether to display plot.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def _create_harmonic_labels(
    n_harmonics: int,
    fundamental_freq: float | None,
) -> list[str]:
    """Create x-axis labels for harmonics.

    Args:
        n_harmonics: Number of harmonics.
        fundamental_freq: Fundamental frequency in Hz or None.

    Returns:
        List of label strings.
    """
    if fundamental_freq is not None:
        labels = [
            f"H{i + 1}\n({(i + 1) * fundamental_freq / 1e3:.1f} kHz)"
            if fundamental_freq >= 1000
            else f"H{i + 1}\n({(i + 1) * fundamental_freq:.0f} Hz)"
            for i in range(n_harmonics)
        ]
        labels[0] = (
            f"Fund\n({fundamental_freq / 1e3:.1f} kHz)"
            if fundamental_freq >= 1000
            else f"Fund\n({fundamental_freq:.0f} Hz)"
        )
    else:
        labels = [f"H{i + 1}" for i in range(n_harmonics)]
        labels[0] = "Fund"

    return labels


def _assign_harmonic_colors(
    harmonic_magnitudes: NDArray[np.floating[Any]],
) -> list[str]:
    """Assign colors to harmonics based on magnitude.

    Args:
        harmonic_magnitudes: Array of harmonic magnitudes in dB.

    Returns:
        List of color strings.
    """
    colors = []
    for i, mag in enumerate(harmonic_magnitudes):
        if i == 0:
            colors.append("#3498DB")  # Blue for fundamental
        elif mag > -30:
            colors.append("#E74C3C")  # Red for significant harmonics
        elif mag > -50:
            colors.append("#F39C12")  # Orange for moderate
        else:
            colors.append("#95A5A6")  # Gray for low

    return colors


def _add_thd_annotation(
    ax: Axes,
    thd_value: float | None,
    show_thd: bool,
) -> None:
    """Add THD annotation to plot.

    Args:
        ax: Matplotlib axes to annotate.
        thd_value: THD value in dB or %.
        show_thd: Show annotation flag.
    """
    if show_thd and thd_value is not None:
        if thd_value > 0:
            thd_text = f"THD: {thd_value:.2f}%"
        else:
            thd_text = f"THD: {thd_value:.1f} dB"

        ax.text(
            0.98,
            0.98,
            thd_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.9},
        )


def plot_thd_bars(
    harmonic_magnitudes: NDArray[np.floating[Any]],
    *,
    fundamental_freq: float | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    thd_value: float | None = None,
    show_thd: bool = True,
    reference_db: float = 0.0,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot THD harmonic bar chart.

    Creates a bar chart showing harmonic content relative to the fundamental,
    useful for visualizing Total Harmonic Distortion analysis results.

    Args:
        harmonic_magnitudes: Array of harmonic magnitudes in dB (relative to fundamental).
            Index 0 = fundamental (0 dB), Index 1 = 2nd harmonic, etc.
        fundamental_freq: Fundamental frequency in Hz (for x-axis labels).
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size in inches.
        title: Plot title.
        thd_value: Pre-calculated THD value in dB or % to display.
        show_thd: Show THD annotation on plot.
        reference_db: Reference level for the fundamental (default 0 dB).
        show: Display plot interactively.
        save_path: Save plot to file.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> # Harmonic magnitudes relative to fundamental (in dB)
        >>> harmonics = np.array([0, -40, -60, -55, -70, -65])  # Fund, H2, H3, H4, H5, H6
        >>> fig = plot_thd_bars(harmonics, fundamental_freq=1000, thd_value=-38.5)

    References:
        IEEE 1241-2010: ADC Testing Standards
        IEC 61000-4-7: Harmonics measurement
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Figure/axes creation
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Data preparation
    n_harmonics = len(harmonic_magnitudes)
    x_pos = np.arange(n_harmonics)
    labels = _create_harmonic_labels(n_harmonics, fundamental_freq)
    colors = _assign_harmonic_colors(harmonic_magnitudes)

    # Plotting/rendering
    ax.bar(
        x_pos, harmonic_magnitudes - reference_db, color=colors, edgecolor="black", linewidth=0.5
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Annotation/labeling
    _add_thd_annotation(ax, thd_value, show_thd)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Harmonic", fontsize=11)
    ax.set_ylabel("Magnitude (dB rel. to fundamental)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Layout/formatting
    min_mag = min(harmonic_magnitudes) - reference_db
    ax.set_ylim(min(min_mag - 10, -80), 10)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Harmonic Distortion Analysis", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_quality_summary(
    metrics: dict[str, float],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    show_specs: dict[str, float] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot ADC/signal quality summary with metrics.

    Creates a summary panel showing SNR, SINAD, THD, ENOB, and SFDR
    with optional pass/fail indication against specifications.

    Args:
        metrics: Dictionary with keys like "snr", "sinad", "thd", "enob", "sfdr".
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        show_specs: Dictionary of specification values for pass/fail.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> metrics = {"snr": 72.5, "sinad": 70.2, "thd": -65.3, "enob": 11.2, "sfdr": 75.8}
        >>> specs = {"snr": 70.0, "enob": 10.0}
        >>> fig = plot_quality_summary(metrics, show_specs=specs)

    References:
        IEEE 1241-2010: ADC Testing Standards
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Setup figure and axes
    fig, ax = _setup_quality_plot_axes(ax, figsize)

    # Define metric metadata
    metric_info = _get_metric_info()

    # Filter to available metrics
    available_metrics = [(k, v) for k, v in metrics.items() if k in metric_info]

    if len(available_metrics) == 0:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Plot metrics with pass/fail coloring
    _plot_quality_bars(ax, available_metrics, metric_info, show_specs)

    # Add value labels and spec markers
    _add_quality_labels(ax, available_metrics, metric_info)
    _add_spec_markers(ax, available_metrics, show_specs)

    # Configure axes
    _configure_quality_axes(ax, available_metrics, metric_info, title)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _setup_quality_plot_axes(
    ax: Axes | None,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    """Setup figure and axes for quality summary plot.

    Args:
        ax: Existing axes or None.
        figsize: Figure size if creating new figure.

    Returns:
        Tuple of (Figure, Axes).

    Raises:
        ValueError: If provided axes has no associated figure.
    """
    if ax is None:
        fig, ax_obj = plt.subplots(figsize=figsize)
        return fig, ax_obj
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        return cast("Figure", fig_temp), ax


def _get_metric_info() -> dict[str, dict[str, Any]]:
    """Get metric display information.

    Returns:
        Dictionary mapping metric keys to display info (name, unit, higher_better).
    """
    return {
        "snr": {"name": "SNR", "unit": "dB", "higher_better": True},
        "sinad": {"name": "SINAD", "unit": "dB", "higher_better": True},
        "thd": {"name": "THD", "unit": "dB", "higher_better": False},
        "enob": {"name": "ENOB", "unit": "bits", "higher_better": True},
        "sfdr": {"name": "SFDR", "unit": "dBc", "higher_better": True},
    }


def _plot_quality_bars(
    ax: Axes,
    available_metrics: list[tuple[str, float]],
    metric_info: dict[str, dict[str, Any]],
    show_specs: dict[str, float] | None,
) -> None:
    """Plot horizontal bars for quality metrics.

    Args:
        ax: Matplotlib axes.
        available_metrics: List of (key, value) tuples for available metrics.
        metric_info: Metric display information.
        show_specs: Specification values for pass/fail coloring.
    """
    n_metrics = len(available_metrics)
    y_pos = np.arange(n_metrics)
    values = [v for _, v in available_metrics]

    # Determine colors based on pass/fail
    colors = _determine_bar_colors(available_metrics, metric_info, show_specs)

    # Plot horizontal bars
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)


def _determine_bar_colors(
    available_metrics: list[tuple[str, float]],
    metric_info: dict[str, dict[str, Any]],
    show_specs: dict[str, float] | None,
) -> list[str]:
    """Determine bar colors based on pass/fail status.

    Args:
        available_metrics: List of (key, value) tuples.
        metric_info: Metric display information.
        show_specs: Specification values.

    Returns:
        List of color strings (hex codes).
    """
    colors = []
    for key, value in available_metrics:
        if show_specs and key in show_specs:
            spec = show_specs[key]
            info = metric_info[key]
            if info["higher_better"]:
                passed = value >= spec
            else:
                # For THD, more negative is better
                passed = value <= spec
            colors.append("#27AE60" if passed else "#E74C3C")
        else:
            colors.append("#3498DB")
    return colors


def _add_quality_labels(
    ax: Axes,
    available_metrics: list[tuple[str, float]],
    metric_info: dict[str, dict[str, Any]],
) -> None:
    """Add value labels to quality metric bars.

    Args:
        ax: Matplotlib axes.
        available_metrics: List of (key, value) tuples.
        metric_info: Metric display information.
    """
    for i, (key, value) in enumerate(available_metrics):
        unit = metric_info[key]["unit"]
        label_text = f"{value:.1f} {unit}"
        ax.text(
            value + 2 if value >= 0 else value - 2,
            i,
            label_text,
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
        )


def _add_spec_markers(
    ax: Axes,
    available_metrics: list[tuple[str, float]],
    show_specs: dict[str, float] | None,
) -> None:
    """Add specification markers to quality plot.

    Args:
        ax: Matplotlib axes.
        available_metrics: List of (key, value) tuples.
        show_specs: Specification values.
    """
    if not show_specs:
        return

    for i, (key, _) in enumerate(available_metrics):
        if key in show_specs:
            spec = show_specs[key]
            ax.plot(spec, i, "k|", markersize=20, markeredgewidth=2)
            ax.text(spec, i + 0.3, f"Spec: {spec}", fontsize=8, ha="center")


def _configure_quality_axes(
    ax: Axes,
    available_metrics: list[tuple[str, float]],
    metric_info: dict[str, dict[str, Any]],
    title: str | None,
) -> None:
    """Configure axes for quality summary plot.

    Args:
        ax: Matplotlib axes.
        available_metrics: List of (key, value) tuples.
        metric_info: Metric display information.
        title: Plot title.
    """
    n_metrics = len(available_metrics)
    y_pos = np.arange(n_metrics)
    names = [metric_info[k]["name"] for k, _ in available_metrics]

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(name) for name in names], fontsize=11)
    ax.set_xlabel("Value", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Signal Quality Summary (IEEE 1241-2010)", fontsize=12, fontweight="bold")


__all__ = [
    "plot_fft",
    "plot_psd",
    "plot_spectrogram",
    "plot_spectrum",
]
