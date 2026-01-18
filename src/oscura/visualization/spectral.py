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
from typing import TYPE_CHECKING, Any, Literal, cast

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
    xscale: Literal["linear", "log"] = "log",
    show: bool = True,
    save_path: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fft_result: tuple[Any, Any] | None = None,
    log_scale: bool = True,
    db_scale: bool | None = None,
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
        xscale: X-axis scale ("linear" or "log"). Deprecated, use log_scale instead.
        show: If True, call plt.show() to display the plot.
        save_path: Path to save the figure. If None, figure is not saved.
        figsize: Figure size (width, height) in inches. Only used if ax is None.
        xlim: X-axis limits (min, max) in selected frequency units.
        ylim: Y-axis limits (min, max) in dB.
        fft_result: Pre-computed FFT result (frequencies, magnitudes). If None, computes FFT.
        log_scale: Use logarithmic scale for frequency axis (default True).
        db_scale: Deprecated alias for log_scale. If provided, overrides log_scale.

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

    # Handle deprecated db_scale parameter
    if db_scale is not None:
        log_scale = db_scale

    from oscura.analyzers.waveform.spectral import fft

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Compute FFT if not provided
    if fft_result is not None:
        freq, mag_db = fft_result
    else:
        freq, mag_db = fft(trace, window=window)  # type: ignore[misc]

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

    # Adjust dB reference if specified
    if db_ref is not None:
        mag_db = mag_db - db_ref

    # Plot
    ax.plot(freq_scaled, mag_db, color=color, linewidth=0.8)

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("Magnitude (dB)")

    # Use log_scale parameter, fall back to xscale for backward compatibility
    # Note: xscale is Literal["linear", "log"] so can never be "log" at this point
    ax.set_xscale("log" if log_scale else "linear")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Magnitude Spectrum")

    if show_grid:
        ax.grid(True, alpha=0.3, which="both")

    # Set reasonable y-limits
    valid_db = mag_db[np.isfinite(mag_db)]
    if len(valid_db) > 0:
        y_max = np.max(valid_db)
        y_min = max(np.min(valid_db), y_max - 120)  # Limit dynamic range
        ax.set_ylim(y_min, y_max + 5)

    # Apply custom limits if specified
    if freq_range is not None:
        ax.set_xlim(freq_range[0] / divisor, freq_range[1] / divisor)
    elif xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


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

    from oscura.analyzers.waveform.spectral import spectrogram

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Handle nfft as alias for nperseg
    if nfft is not None:
        nperseg = nfft

    # Compute spectrogram with optional overlap
    noverlap = None
    if overlap is not None and nperseg is not None:
        noverlap = int(nperseg * overlap)
    times, freq, Sxx_db = spectrogram(trace, window=window, nperseg=nperseg, noverlap=noverlap)

    # Auto-select units
    if time_unit == "auto":
        max_time = times[-1] if len(times) > 0 else 0
        if max_time < 1e-6:
            time_unit = "ns"
        elif max_time < 1e-3:
            time_unit = "us"
        elif max_time < 1:
            time_unit = "ms"
        else:
            time_unit = "s"

    if freq_unit == "auto":
        max_freq = freq[-1] if len(freq) > 0 else 0
        if max_freq >= 1e9:
            freq_unit = "GHz"
        elif max_freq >= 1e6:
            freq_unit = "MHz"
        elif max_freq >= 1e3:
            freq_unit = "kHz"
        else:
            freq_unit = "Hz"

    time_multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    freq_divisors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

    time_mult = time_multipliers.get(time_unit, 1.0)
    freq_div = freq_divisors.get(freq_unit, 1.0)

    times_scaled = times * time_mult
    freq_scaled = freq / freq_div

    # Auto color limits
    if vmin is None or vmax is None:
        valid_db = Sxx_db[np.isfinite(Sxx_db)]
        if len(valid_db) > 0:
            if vmax is None:
                vmax = np.max(valid_db)
            if vmin is None:
                vmin = max(np.min(valid_db), vmax - 80)

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

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Spectrogram")

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Magnitude (dB)")

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
    xscale: Literal["linear", "log"] = "log",
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
        xscale: X-axis scale.

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
    ax.set_xscale(xscale)

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

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Use plot_spectrum to do the actual plotting
    xscale_value: Literal["linear", "log"] = "log" if log_scale else "linear"
    plot_spectrum(
        trace,
        ax=ax,
        freq_unit=freq_unit,
        show_grid=show_grid,
        color=color,
        title=title if title else "FFT Magnitude Spectrum",
        window=window,
        xscale=xscale_value,
    )

    # Apply custom labels if different from defaults
    if xlabel != "Frequency":
        # Get current label to preserve unit
        current_label = ax.get_xlabel()
        if "(" in current_label and ")" in current_label:
            unit = current_label[current_label.find("(") : current_label.find(")") + 1]
            ax.set_xlabel(f"{xlabel} {unit}")
        else:
            ax.set_xlabel(xlabel)

    if ylabel != "Magnitude (dB)":
        ax.set_ylabel(ylabel)

    # Apply custom limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    n_harmonics = len(harmonic_magnitudes)

    # Create x-positions for harmonics
    x_pos = np.arange(n_harmonics)

    # Create labels
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

    # Color code: fundamental in blue, harmonics in orange/red based on magnitude
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

    # Plot bars
    ax.bar(
        x_pos, harmonic_magnitudes - reference_db, color=colors, edgecolor="black", linewidth=0.5
    )

    # Reference line at fundamental level
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # THD annotation
    if show_thd and thd_value is not None:
        # Position in upper right
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

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Harmonic", fontsize=11)
    ax.set_ylabel("Magnitude (dB rel. to fundamental)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Y-axis limits
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Define metric display info
    metric_info = {
        "snr": {"name": "SNR", "unit": "dB", "higher_better": True},
        "sinad": {"name": "SINAD", "unit": "dB", "higher_better": True},
        "thd": {
            "name": "THD",
            "unit": "dB",
            "higher_better": False,
        },  # Lower (more negative) is better
        "enob": {"name": "ENOB", "unit": "bits", "higher_better": True},
        "sfdr": {"name": "SFDR", "unit": "dBc", "higher_better": True},
    }

    # Filter to available metrics
    available_metrics = [(k, v) for k, v in metrics.items() if k in metric_info]
    n_metrics = len(available_metrics)

    if n_metrics == 0:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Create horizontal bar chart
    y_pos = np.arange(n_metrics)
    values = [v for _, v in available_metrics]
    names = [metric_info[k]["name"] for k, _ in available_metrics]

    # Determine colors based on pass/fail
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

    # Plot horizontal bars
    ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
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

    # Add spec markers
    if show_specs:
        for i, (key, _) in enumerate(available_metrics):
            if key in show_specs:
                spec = show_specs[key]
                ax.plot(spec, i, "k|", markersize=20, markeredgewidth=2)
                ax.text(spec, i + 0.3, f"Spec: {spec}", fontsize=8, ha="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(name) for name in names], fontsize=11)
    ax.set_xlabel("Value", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Signal Quality Summary (IEEE 1241-2010)", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


__all__ = [
    "plot_fft",
    "plot_psd",
    "plot_spectrogram",
    "plot_spectrum",
]
