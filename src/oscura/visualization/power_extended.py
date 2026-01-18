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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Normalize efficiency to percentage if needed
    if np.max(efficiency_values) <= 1.0:
        efficiency_values = efficiency_values * 100
        if efficiency_sets is not None:
            efficiency_sets = [e * 100 for e in efficiency_sets]

    # Color palette for multiple curves
    colors = ["#3498DB", "#E74C3C", "#27AE60", "#9B59B6", "#F39C12"]

    if v_in_values is not None and efficiency_sets is not None:
        # Multiple input voltage curves
        for i, (v_in, eff) in enumerate(zip(v_in_values, efficiency_sets, strict=False)):
            color = colors[i % len(colors)]
            ax.plot(load_values, eff, "-", linewidth=2, color=color, label=f"Vin = {v_in}V")

            if show_peak:
                peak_idx = np.argmax(eff)
                ax.plot(load_values[peak_idx], eff[peak_idx], "o", color=color, markersize=8)
    else:
        # Single curve
        ax.plot(
            load_values, efficiency_values, "-", linewidth=2.5, color="#3498DB", label="Efficiency"
        )

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

    # Labels
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

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


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

    # Determine number of subplots needed
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

    # Time unit conversion
    if time_unit == "auto":
        max_time = np.max(time)
        if max_time < 1e-6:
            time_unit = "us"
            time_mult = 1e6
        elif max_time < 1e-3:
            time_unit = "ms"
            time_mult = 1e3
        else:
            time_unit = "s"
            time_mult = 1.0
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(time_unit, 1.0)

    time_scaled = time * time_mult

    ax_idx = 0

    # Input voltage/current panel
    if v_in is not None:
        ax = axes[ax_idx]
        ax.plot(time_scaled, v_in, "#3498DB", linewidth=1.5, label="V_in")
        ax.set_ylabel("V_in (V)", color="#3498DB", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#3498DB")
        ax.grid(True, alpha=0.3)

        if i_in is not None:
            ax2 = ax.twinx()
            ax2.plot(time_scaled, i_in, "#E74C3C", linewidth=1.5, label="I_in")
            ax2.set_ylabel("I_in (A)", color="#E74C3C", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="#E74C3C")

        ax.set_title("Input", fontsize=10, fontweight="bold", loc="left")
        ax_idx += 1

    # Output voltage/current panel
    if v_out is not None:
        ax = axes[ax_idx]
        ax.plot(time_scaled, v_out, "#27AE60", linewidth=1.5, label="V_out")
        ax.set_ylabel("V_out (V)", color="#27AE60", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#27AE60")
        ax.grid(True, alpha=0.3)

        if i_out is not None:
            ax2 = ax.twinx()
            ax2.plot(time_scaled, i_out, "#9B59B6", linewidth=1.5, label="I_out")
            ax2.set_ylabel("I_out (A)", color="#9B59B6", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="#9B59B6")

        ax.set_title("Output", fontsize=10, fontweight="bold", loc="left")
        ax_idx += 1

    # Power panel
    if show_power:
        ax = axes[ax_idx]

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

    # X-axis label on bottom
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Power Converter Waveforms", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


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
        time_unit: Time axis unit.
        show_dc: Show DC-coupled waveform.
        show_ac: Show AC-coupled ripple.
        show_spectrum: Show ripple spectrum.
        sample_rate: Sample rate for FFT (required if show_spectrum=True).
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    n_plots = sum([show_dc, show_ac, show_spectrum])
    if n_plots == 0:
        raise ValueError("At least one display option must be True")

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Time unit conversion
    if time_unit == "auto":
        max_time = np.max(time)
        if max_time < 1e-6:
            time_unit = "us"
            time_mult = 1e6
        elif max_time < 1e-3:
            time_unit = "ms"
            time_mult = 1e3
        else:
            time_unit = "s"
            time_mult = 1.0
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}.get(time_unit, 1.0)

    time_scaled = time * time_mult

    # Calculate DC level and ripple
    dc_level = np.mean(voltage)
    ac_ripple = voltage - dc_level
    ripple_pp = np.ptp(ac_ripple)
    ripple_rms = np.std(ac_ripple)

    ax_idx = 0

    # DC-coupled view
    if show_dc:
        ax = axes[ax_idx]
        ax.plot(time_scaled, voltage, "#3498DB", linewidth=1)
        ax.axhline(
            dc_level, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"DC: {dc_level:.3f}V"
        )
        ax.set_ylabel("Voltage (V)", fontsize=10)
        ax.set_title("DC-Coupled Waveform", fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # AC-coupled (ripple only) view
    if show_ac:
        ax = axes[ax_idx]
        ax.plot(time_scaled, ac_ripple * 1e3, "#27AE60", linewidth=1)  # Convert to mV
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)

        # Mark peak-to-peak
        max_idx = np.argmax(ac_ripple)
        min_idx = np.argmin(ac_ripple)
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
        ax_idx += 1

    # Spectrum view
    if show_spectrum:
        ax = axes[ax_idx]

        if sample_rate is None:
            # Estimate from time array
            sample_rate = 1 / (time[1] - time[0]) if len(time) > 1 else 1e6

        n_fft = len(voltage)
        freq = np.fft.rfftfreq(n_fft, 1 / sample_rate)
        fft_mag = np.abs(np.fft.rfft(ac_ripple)) / n_fft * 2
        fft_db = 20 * np.log10(fft_mag + 1e-12)

        # Find dominant ripple frequency
        peak_idx = np.argmax(fft_mag[1:]) + 1  # Skip DC
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
    else:
        axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)

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

    # Color palette
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

    # Format labels with percentages and watts
    autopct_val: str | Callable[[float], str]
    if show_watts:

        def autopct_func(pct: float) -> str:
            watts = pct / 100 * total_loss
            return f"{pct:.1f}%\n({watts * 1e3:.1f}mW)"

        autopct_val = autopct_func
    else:
        autopct_val = "%1.1f%%"

    pie_result = ax.pie(
        values,
        labels=labels,
        autopct=autopct_val,  # type: ignore[arg-type]
        colors=colors[: len(labels)],
        startangle=90,
        explode=[0.02] * len(labels),
        shadow=True,
    )
    # ax.pie returns (wedges, texts, autotexts) when autopct is provided
    # Unpack with length check for type safety
    if len(pie_result) >= 3:
        _wedges = pie_result[0]
        _texts = pie_result[1]
        autotexts = pie_result[2]
    else:
        autotexts = []

    # Style autotexts
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

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    else:
        ax.set_title("Power Loss Breakdown", fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
