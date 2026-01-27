"""Extended Power Analysis Visualization Functions.

This module provides visualization functions for power conversion analysis
including efficiency curves, ripple analysis, loss breakdown, and
multi-channel power waveforms.

Example:
    >>> from oscura.visualization.power_extended import (
    ...     plot_efficiency_curve, plot_ripple_waveform, plot_loss_breakdown
    ... )
    >>> fig = plot_efficiency_curve(load_currents, efficiencies)
    >>> fig = plot_ripple_waveform(voltage_trace, ripple_trace)

References:
    - Power supply measurement best practices
    - DC-DC converter efficiency testing
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


__all__ = [
    "plot_efficiency_curve",
    "plot_loss_breakdown",
    "plot_power_waveforms",
    "plot_ripple_waveform",
]


def _normalize_efficiency_values(
    efficiency_values: NDArray[np.floating[Any]],
    efficiency_sets: list[NDArray[np.floating[Any]]] | None,
) -> tuple[NDArray[np.floating[Any]], list[NDArray[np.floating[Any]]] | None]:
    """Normalize efficiency values to percentage (0-100).

    Args:
        efficiency_values: Efficiency values (0-100 or 0-1).
        efficiency_sets: List of efficiency arrays or None.

    Returns:
        Tuple of (normalized_efficiency, normalized_sets).
    """
    if np.max(efficiency_values) <= 1.0:
        efficiency_values = efficiency_values * 100
        if efficiency_sets is not None:
            efficiency_sets = [e * 100 for e in efficiency_sets]

    return efficiency_values, efficiency_sets


def _plot_multi_efficiency_curves(
    ax: Axes,
    load_values: NDArray[np.floating[Any]],
    v_in_values: list[float],
    efficiency_sets: list[NDArray[np.floating[Any]]],
    show_peak: bool,
) -> None:
    """Plot multiple efficiency curves for different input voltages.

    Args:
        ax: Matplotlib axes to plot on.
        load_values: Load current or power array.
        v_in_values: List of input voltages.
        efficiency_sets: List of efficiency arrays.
        show_peak: Show peak efficiency markers.
    """
    colors = ["#3498DB", "#E74C3C", "#27AE60", "#9B59B6", "#F39C12"]

    for i, (v_in, eff) in enumerate(zip(v_in_values, efficiency_sets, strict=False)):
        color = colors[i % len(colors)]
        ax.plot(load_values, eff, "-", linewidth=2, color=color, label=f"Vin = {v_in}V")

        if show_peak:
            peak_idx = np.argmax(eff)
            ax.plot(load_values[peak_idx], eff[peak_idx], "o", color=color, markersize=8)


def _plot_single_efficiency_curve(
    ax: Axes,
    load_values: NDArray[np.floating[Any]],
    efficiency_values: NDArray[np.floating[Any]],
    load_unit: str,
    show_peak: bool,
) -> None:
    """Plot single efficiency curve with peak annotation.

    Args:
        ax: Matplotlib axes to plot on.
        load_values: Load current or power array.
        efficiency_values: Efficiency values in %.
        load_unit: Load axis unit.
        show_peak: Show peak efficiency annotation.
    """
    ax.plot(load_values, efficiency_values, "-", linewidth=2.5, color="#3498DB", label="Efficiency")

    if show_peak:
        peak_idx = np.argmax(efficiency_values)
        peak_load = load_values[peak_idx]
        peak_eff = efficiency_values[peak_idx]
        ax.plot(peak_load, peak_eff, "o", color="#E74C3C", markersize=10, zorder=5)
        ax.annotate(
            f"Peak: {peak_eff:.1f}%\n@ {peak_load:.2f} {load_unit}",
            xy=(peak_load, peak_eff),
            xytext=(15, -15),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.2"},
        )


def _format_efficiency_plot(
    ax: Axes,
    load_values: NDArray[np.floating[Any]],
    efficiency_values: NDArray[np.floating[Any]],
    efficiency_sets: list[NDArray[np.floating[Any]]] | None,
    load_unit: str,
    target_efficiency: float | None,
    title: str | None,
) -> None:
    """Format efficiency plot axes and labels.

    Args:
        ax: Matplotlib axes to format.
        load_values: Load current or power array.
        efficiency_values: Efficiency values in %.
        efficiency_sets: List of efficiency arrays or None.
        load_unit: Load axis unit.
        target_efficiency: Target efficiency line.
        title: Plot title.
    """
    # Target efficiency line
    if target_efficiency is not None:
        ax.axhline(
            target_efficiency,
            color="#E74C3C",
            linestyle="--",
            linewidth=1.5,
            label=f"Target: {target_efficiency}%",
        )

    # Fill area under curve
    ax.fill_between(
        load_values,
        0,
        efficiency_values if efficiency_sets is None else efficiency_sets[0],
        alpha=0.1,
        color="#3498DB",
    )

    # Labels and formatting
    ax.set_xlabel(f"Load ({load_unit})", fontsize=11)
    ax.set_ylabel("Efficiency (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(load_values[0], load_values[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Converter Efficiency vs Load", fontsize=12, fontweight="bold")


def plot_efficiency_curve(
    load_values: NDArray[np.floating[Any]],
    efficiency_values: NDArray[np.floating[Any]],
    *,
    v_in_values: list[float] | None = None,
    efficiency_sets: list[NDArray[np.floating[Any]]] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    title: str | None = None,
    load_unit: str = "A",
    target_efficiency: float | None = None,
    show_peak: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot efficiency vs load curve for power converters.

    Creates an efficiency plot showing converter efficiency as a function
    of load current or power, with optional multiple input voltage curves.

    Args:
        load_values: Load current or power array.
        efficiency_values: Efficiency values (0-100 or 0-1).
        v_in_values: List of input voltages for multi-curve plot.
        efficiency_sets: List of efficiency arrays for each v_in.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        load_unit: Load axis unit ("A", "W", "%").
        target_efficiency: Target efficiency line.
        show_peak: Annotate peak efficiency point.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> load = np.linspace(0.1, 5, 50)  # 0.1A to 5A
        >>> eff = 90 - 5 * np.exp(-load)  # Example efficiency curve
        >>> fig = plot_efficiency_curve(load, eff, target_efficiency=85)
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

    # Data preparation/validation
    efficiency_values, efficiency_sets = _normalize_efficiency_values(
        efficiency_values, efficiency_sets
    )

    # Plotting/rendering
    if v_in_values is not None and efficiency_sets is not None:
        _plot_multi_efficiency_curves(ax, load_values, v_in_values, efficiency_sets, show_peak)
    else:
        _plot_single_efficiency_curve(ax, load_values, efficiency_values, load_unit, show_peak)

    # Annotation/labeling and layout/formatting
    _format_efficiency_plot(
        ax, load_values, efficiency_values, efficiency_sets, load_unit, target_efficiency, title
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _determine_time_scale(time: NDArray[np.floating[Any]], time_unit: str) -> tuple[str, float]:
    """Determine time axis scale and multiplier.

    Args:
        time: Time array in seconds.
        time_unit: Requested time unit ("auto" or specific).

    Returns:
        Tuple of (unit_name, multiplier).
    """
    if time_unit != "auto":
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(time_unit, 1.0)
        return time_unit, time_mult

    max_time = np.max(time)
    if max_time < 1e-6:
        return "us", 1e6
    elif max_time < 1e-3:
        return "ms", 1e3
    else:
        return "s", 1.0


def _plot_voltage_current_panel(
    ax: Axes,
    time_scaled: NDArray[np.floating[Any]],
    voltage: NDArray[np.floating[Any]],
    current: NDArray[np.floating[Any]] | None,
    v_label: str,
    i_label: str,
    v_color: str,
    i_color: str,
    panel_title: str,
) -> None:
    """Plot voltage and current on dual-axis panel.

    Args:
        ax: Matplotlib axes for voltage.
        time_scaled: Scaled time array.
        voltage: Voltage waveform.
        current: Current waveform (optional).
        v_label: Voltage axis label.
        i_label: Current axis label.
        v_color: Voltage plot color.
        i_color: Current plot color.
        panel_title: Panel title.
    """
    ax.plot(time_scaled, voltage, v_color, linewidth=1.5)
    ax.set_ylabel(v_label, color=v_color, fontsize=10)
    ax.tick_params(axis="y", labelcolor=v_color)
    ax.grid(True, alpha=0.3)

    if current is not None:
        ax2 = ax.twinx()
        ax2.plot(time_scaled, current, i_color, linewidth=1.5)
        ax2.set_ylabel(i_label, color=i_color, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=i_color)

    ax.set_title(panel_title, fontsize=10, fontweight="bold", loc="left")


def _plot_power_panel(
    ax: Axes,
    time_scaled: NDArray[np.floating[Any]],
    v_in: NDArray[np.floating[Any]] | None,
    i_in: NDArray[np.floating[Any]] | None,
    v_out: NDArray[np.floating[Any]] | None,
    i_out: NDArray[np.floating[Any]] | None,
) -> None:
    """Plot instantaneous power panel.

    Args:
        ax: Matplotlib axes.
        time_scaled: Scaled time array.
        v_in: Input voltage (optional).
        i_in: Input current (optional).
        v_out: Output voltage (optional).
        i_out: Output current (optional).
    """
    if v_in is not None and i_in is not None:
        p_in = v_in * i_in
        ax.plot(
            time_scaled,
            p_in,
            "#3498DB",
            linewidth=1.5,
            label=f"P_in (avg: {np.mean(p_in):.2f}W)",
        )

    if v_out is not None and i_out is not None:
        p_out = v_out * i_out
        ax.plot(
            time_scaled,
            p_out,
            "#27AE60",
            linewidth=1.5,
            label=f"P_out (avg: {np.mean(p_out):.2f}W)",
        )

    ax.set_ylabel("Power (W)", fontsize=10)
    ax.set_title("Instantaneous Power", fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _setup_power_waveform_figure(
    figsize: tuple[float, float],
    v_in: NDArray[np.floating[Any]] | None,
    v_out: NDArray[np.floating[Any]] | None,
    show_power: bool,
) -> tuple[Figure, list[Axes]]:
    """Setup figure and axes for power waveform plot.

    Args:
        figsize: Figure size.
        v_in: Input voltage (optional).
        v_out: Output voltage (optional).
        show_power: Show power panel.

    Returns:
        Tuple of (figure, axes_list).
    """
    n_plots = sum(
        [
            v_in is not None,
            v_out is not None,
            show_power and (v_in is not None or v_out is not None),
        ]
    )
    if n_plots == 0:
        raise ValueError("At least one voltage waveform must be provided")

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    return fig, axes


def _plot_power_waveform_panels(
    axes: list[Axes],
    time_scaled: NDArray[np.floating[Any]],
    v_in: NDArray[np.floating[Any]] | None,
    i_in: NDArray[np.floating[Any]] | None,
    v_out: NDArray[np.floating[Any]] | None,
    i_out: NDArray[np.floating[Any]] | None,
    show_power: bool,
) -> None:
    """Plot all voltage/current panels.

    Args:
        axes: List of axes to plot on.
        time_scaled: Scaled time array.
        v_in: Input voltage (optional).
        i_in: Input current (optional).
        v_out: Output voltage (optional).
        i_out: Output current (optional).
        show_power: Show power panel.
    """
    ax_idx = 0

    if v_in is not None:
        _plot_voltage_current_panel(
            axes[ax_idx],
            time_scaled,
            v_in,
            i_in,
            "V_in (V)",
            "I_in (A)",
            "#3498DB",
            "#E74C3C",
            "Input",
        )
        ax_idx += 1

    if v_out is not None:
        _plot_voltage_current_panel(
            axes[ax_idx],
            time_scaled,
            v_out,
            i_out,
            "V_out (V)",
            "I_out (A)",
            "#27AE60",
            "#9B59B6",
            "Output",
        )
        ax_idx += 1

    if show_power:
        _plot_power_panel(axes[ax_idx], time_scaled, v_in, i_in, v_out, i_out)


def _finalize_power_waveform_plot(
    fig: Figure,
    axes: list[Axes],
    time_unit: str,
    title: str | None,
    save_path: str | Path | None,
    show: bool,
) -> None:
    """Finalize power waveform plot formatting and save.

    Args:
        fig: Matplotlib figure.
        axes: List of axes.
        time_unit: Time axis unit.
        title: Plot title.
        save_path: Save path.
        show: Display plot.
    """
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)
    fig.suptitle(title if title else "Power Converter Waveforms", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def plot_power_waveforms(
    time: NDArray[np.floating[Any]],
    *,
    v_in: NDArray[np.floating[Any]] | None = None,
    i_in: NDArray[np.floating[Any]] | None = None,
    v_out: NDArray[np.floating[Any]] | None = None,
    i_out: NDArray[np.floating[Any]] | None = None,
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
    time_unit: str = "auto",
    show_power: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot multi-channel power waveforms with optional power calculation.

    Creates a multi-panel plot showing input/output voltage and current
    waveforms with optional instantaneous power overlay.

    Args:
        time: Time array in seconds.
        v_in: Input voltage waveform.
        i_in: Input current waveform.
        v_out: Output voltage waveform.
        i_out: Output current waveform.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time axis unit.
        show_power: Calculate and show instantaneous power.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Setup: determine layout and prepare axes
    fig, axes = _setup_power_waveform_figure(figsize, v_in, v_out, show_power)
    time_unit, time_mult = _determine_time_scale(time, time_unit)
    time_scaled = time * time_mult

    # Processing: plot data panels
    _plot_power_waveform_panels(axes, time_scaled, v_in, i_in, v_out, i_out, show_power)

    # Formatting: finalize and save
    _finalize_power_waveform_plot(fig, axes, time_unit, title, save_path, show)

    return fig


def _determine_time_unit_and_multiplier(
    time: NDArray[np.floating[Any]], time_unit: str
) -> tuple[str, float]:
    """Determine time unit and multiplier for time axis scaling.

    Args:
        time: Time array in seconds.
        time_unit: Requested time unit ("auto" or specific unit).

    Returns:
        Tuple of (time_unit, time_multiplier).
    """
    if time_unit == "auto":
        max_time = np.max(time)
        if max_time < 1e-6:
            return "us", 1e6
        elif max_time < 1e-3:
            return "ms", 1e3
        else:
            return "s", 1.0
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(time_unit, 1.0)
        return time_unit, time_mult


def _calculate_ripple_metrics(
    voltage: NDArray[np.floating[Any]],
) -> tuple[float, NDArray[np.floating[Any]], float, float]:
    """Calculate DC level and AC ripple metrics.

    Args:
        voltage: Voltage waveform array.

    Returns:
        Tuple of (dc_level, ac_ripple, ripple_pp, ripple_rms).
    """
    dc_level = float(np.mean(voltage))
    ac_ripple = voltage - dc_level
    ripple_pp = float(np.ptp(ac_ripple))
    ripple_rms = float(np.std(ac_ripple))
    return dc_level, ac_ripple, ripple_pp, ripple_rms


def _plot_dc_coupled_waveform(
    ax: Axes,
    time_scaled: NDArray[np.floating[Any]],
    voltage: NDArray[np.floating[Any]],
    dc_level: float,
) -> None:
    """Plot DC-coupled waveform with DC level indicator.

    Args:
        ax: Matplotlib axes to plot on.
        time_scaled: Scaled time array.
        voltage: Voltage waveform.
        dc_level: DC level value.
    """
    ax.plot(time_scaled, voltage, "#3498DB", linewidth=1)
    ax.axhline(
        dc_level, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"DC: {dc_level:.3f}V"
    )
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.set_title("DC-Coupled Waveform", fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_ac_ripple_waveform(
    ax: Axes,
    time_scaled: NDArray[np.floating[Any]],
    ac_ripple: NDArray[np.floating[Any]],
    ripple_pp: float,
    ripple_rms: float,
) -> None:
    """Plot AC-coupled ripple waveform with peak-to-peak annotation.

    Args:
        ax: Matplotlib axes to plot on.
        time_scaled: Scaled time array.
        ac_ripple: AC ripple waveform.
        ripple_pp: Peak-to-peak ripple voltage.
        ripple_rms: RMS ripple voltage.
    """
    ax.plot(time_scaled, ac_ripple * 1e3, "#27AE60", linewidth=1)  # Convert to mV
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)

    # Mark peak-to-peak
    max_idx = int(np.argmax(ac_ripple))
    min_idx = int(np.argmin(ac_ripple))
    ax.annotate(
        "",
        xy=(time_scaled[max_idx], ac_ripple[max_idx] * 1e3),
        xytext=(time_scaled[min_idx], ac_ripple[min_idx] * 1e3),
        arrowprops={"arrowstyle": "<->", "color": "#E74C3C", "lw": 1.5},
    )

    ax.set_ylabel("Ripple (mV)", fontsize=10)
    ax.set_title(
        f"AC Ripple (pk-pk: {ripple_pp * 1e3:.2f}mV, RMS: {ripple_rms * 1e3:.2f}mV)",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    ax.grid(True, alpha=0.3)


def _plot_ripple_spectrum(
    ax: Axes,
    ac_ripple: NDArray[np.floating[Any]],
    sample_rate: float,
) -> None:
    """Plot ripple frequency spectrum.

    Args:
        ax: Matplotlib axes to plot on.
        ac_ripple: AC ripple waveform.
        sample_rate: Sample rate in Hz.
    """
    n_fft = len(ac_ripple)
    freq = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(ac_ripple)) / n_fft * 2
    fft_db = 20 * np.log10(fft_mag + 1e-12)

    # Find dominant ripple frequency
    peak_idx = int(np.argmax(fft_mag[1:])) + 1  # Skip DC
    peak_freq = freq[peak_idx]

    # Plot in kHz
    freq_khz = freq / 1e3
    ax.plot(freq_khz, fft_db, "#9B59B6", linewidth=1)
    ax.plot(
        freq_khz[peak_idx],
        fft_db[peak_idx],
        "ro",
        markersize=8,
        label=f"Peak: {peak_freq / 1e3:.1f}kHz",
    )

    ax.set_ylabel("Magnitude (dB)", fontsize=10)
    ax.set_xlabel("Frequency (kHz)", fontsize=10)
    ax.set_title("Ripple Spectrum", fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(0, min(freq_khz[-1], sample_rate / 2e3))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _estimate_sample_rate(time: NDArray[np.floating[Any]]) -> float:
    """Estimate sample rate from time array.

    Args:
        time: Time array in seconds.

    Returns:
        Estimated sample rate in Hz.
    """
    if len(time) > 1:
        return float(1 / (time[1] - time[0]))
    return 1e6  # Default 1 MHz


def plot_ripple_waveform(
    time: NDArray[np.floating[Any]],
    voltage: NDArray[np.floating[Any]],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    time_unit: str = "auto",
    show_dc: bool = True,
    show_ac: bool = True,
    show_spectrum: bool = True,
    sample_rate: float | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot ripple waveform with DC, AC, and spectral analysis.

    Creates a multi-panel view showing DC-coupled waveform, AC-coupled
    ripple, and optionally the ripple frequency spectrum.

    Args:
        time: Time array in seconds.
        voltage: Voltage waveform.
        ax: Matplotlib axes (creates multi-panel if None).
        figsize: Figure size.
        title: Plot title.
        time_unit: Time axis unit ("auto", "s", "ms", "us", "ns").
        show_dc: Show DC-coupled waveform.
        show_ac: Show AC-coupled ripple.
        show_spectrum: Show ripple spectrum.
        sample_rate: Sample rate for FFT (estimated if None).
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> time = np.linspace(0, 1e-3, 1000)  # 1ms capture
        >>> voltage = 5.0 + 0.01 * np.sin(2 * np.pi * 100e3 * time)  # 5V + 10mV ripple
        >>> fig = plot_ripple_waveform(time, voltage, show_spectrum=True)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    n_plots = sum([show_dc, show_ac, show_spectrum])
    if n_plots == 0:
        raise ValueError("At least one display option must be True")

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Determine time scaling
    time_unit, time_mult = _determine_time_unit_and_multiplier(time, time_unit)
    time_scaled = time * time_mult

    # Calculate ripple metrics
    dc_level, ac_ripple, ripple_pp, ripple_rms = _calculate_ripple_metrics(voltage)

    ax_idx = 0

    # Plot DC-coupled waveform
    if show_dc:
        _plot_dc_coupled_waveform(axes[ax_idx], time_scaled, voltage, dc_level)
        ax_idx += 1

    # Plot AC-coupled ripple
    if show_ac:
        _plot_ac_ripple_waveform(axes[ax_idx], time_scaled, ac_ripple, ripple_pp, ripple_rms)
        ax_idx += 1

    # Plot ripple spectrum
    if show_spectrum:
        sr = sample_rate if sample_rate is not None else _estimate_sample_rate(time)
        _plot_ripple_spectrum(axes[ax_idx], ac_ripple, sr)
    else:
        axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)

    # Finalize figure
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Ripple Analysis", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _create_loss_autopct_formatter(
    show_watts: bool, total_loss: float
) -> str | Callable[[float], str]:
    """Create autopct formatter for pie chart labels.

    Args:
        show_watts: Whether to show watt values.
        total_loss: Total loss in watts.

    Returns:
        Format string or callable for autopct.
    """
    if show_watts:

        def autopct_func(pct: float) -> str:
            watts = pct / 100 * total_loss
            return f"{pct:.1f}%\n({watts * 1e3:.1f}mW)"

        return autopct_func
    return "%1.1f%%"


def _create_loss_pie_chart(
    ax: Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    autopct_val: str | Callable[[float], str],
) -> tuple[Any, ...]:
    """Create pie chart with loss breakdown.

    Args:
        ax: Matplotlib axes.
        labels: Loss type labels.
        values: Loss values.
        colors: Color palette.
        autopct_val: Autopct formatter.

    Returns:
        Pie chart result tuple.
    """
    return ax.pie(
        values,
        labels=labels,
        autopct=autopct_val,
        colors=colors[: len(labels)],
        startangle=90,
        explode=[0.02] * len(labels),
        shadow=True,
    )


def _format_loss_pie_chart(
    ax: Axes, pie_result: tuple[Any, ...], total_loss: float, title: str | None
) -> None:
    """Format pie chart styling and annotations.

    Args:
        ax: Matplotlib axes.
        pie_result: Result from ax.pie.
        total_loss: Total loss value.
        title: Chart title.
    """
    # Style autotexts if available
    if len(pie_result) >= 3:
        autotexts = pie_result[2]
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight("bold")

    # Add total loss annotation
    ax.text(
        0,
        -1.3,
        f"Total Loss: {total_loss * 1e3:.1f}mW ({total_loss:.3f}W)",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_aspect("equal")
    ax.set_title(title if title else "Power Loss Breakdown", fontsize=12, fontweight="bold", pad=20)


def plot_loss_breakdown(
    loss_values: dict[str, float],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    show_watts: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot power loss breakdown as pie chart.

    Creates a pie chart showing the contribution of each loss mechanism
    (switching, conduction, magnetic, etc.) to total power dissipation.

    Args:
        loss_values: Dictionary mapping loss type to value in Watts.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        show_watts: Show watt values on slices.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> losses = {
        ...     "Switching": 0.5,
        ...     "Conduction": 0.3,
        ...     "Magnetic": 0.15,
        ...     "Gate Drive": 0.05
        ... }
        >>> fig = plot_loss_breakdown(losses)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Setup: create figure and extract data
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    labels = list(loss_values.keys())
    values = list(loss_values.values())
    total_loss = sum(values)
    colors = [
        "#3498DB",
        "#E74C3C",
        "#27AE60",
        "#9B59B6",
        "#F39C12",
        "#1ABC9C",
        "#E67E22",
        "#95A5A6",
    ]

    # Processing: create pie chart
    autopct_val = _create_loss_autopct_formatter(show_watts, total_loss)
    pie_result = _create_loss_pie_chart(ax, labels, values, colors, autopct_val)

    # Result building: format and finalize
    _format_loss_pie_chart(ax, pie_result, total_loss, title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
