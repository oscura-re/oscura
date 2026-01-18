"""Signal Integrity Visualization Functions.

This module provides visualization functions for signal integrity analysis
including TDR impedance plots, S-parameter displays, setup/hold timing
diagrams, and eye diagram enhancements.

Example:
    >>> from oscura.visualization.signal_integrity import plot_tdr, plot_sparams
    >>> fig = plot_tdr(impedance_profile, distance_axis)
    >>> fig = plot_sparams(frequencies, s11, s21)

References:
    - IEEE 370-2020: Electrical Characterization of Printed Circuit Board
    - TDR impedance measurement best practices
"""

from __future__ import annotations

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
    "plot_setup_hold_timing",
    "plot_sparams_magnitude",
    "plot_sparams_phase",
    "plot_tdr",
    "plot_timing_margin",
]


def plot_tdr(
    impedance: NDArray[np.floating[Any]],
    distance: NDArray[np.floating[Any]],
    *,
    z0: float = 50.0,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    distance_unit: str = "auto",
    show_reference: bool = True,
    show_discontinuities: bool = True,
    discontinuity_threshold: float = 5.0,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot TDR impedance profile vs distance.

    Creates a Time Domain Reflectometry impedance plot showing impedance
    as a function of distance along a transmission line, with annotations
    for discontinuities and reference impedance.

    Args:
        impedance: Impedance values in Ohms.
        distance: Distance values (in meters).
        z0: Reference impedance (Ohms) for the reference line.
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size in inches (only if ax is None).
        title: Plot title.
        distance_unit: Distance unit ("m", "cm", "mm", "auto").
        show_reference: Show reference impedance line at z0.
        show_discontinuities: Annotate significant discontinuities.
        discontinuity_threshold: Impedance change threshold (Ohms) for marking.
        show: Display plot interactively.
        save_path: Save plot to file.

    Returns:
        Matplotlib Figure object.

    Raises:
        ImportError: If matplotlib is not available.
        ValueError: If input arrays have different lengths.

    Example:
        >>> z_profile = np.array([50, 50, 75, 75, 50, 50])
        >>> dist = np.linspace(0, 0.5, 6)  # 0 to 50 cm
        >>> fig = plot_tdr(z_profile, dist, z0=50, show=False)
        >>> fig.savefig("tdr_impedance.png")
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    if len(impedance) != len(distance):
        raise ValueError(
            f"impedance and distance must have same length "
            f"(got {len(impedance)} and {len(distance)})"
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)

    # Convert distance units
    if distance_unit == "auto":
        max_dist = np.max(distance)
        if max_dist < 0.01:
            distance_unit = "mm"
            distance_mult = 1000.0
        elif max_dist < 1.0:
            distance_unit = "cm"
            distance_mult = 100.0
        else:
            distance_unit = "m"
            distance_mult = 1.0
    else:
        distance_mult = {"m": 1.0, "cm": 100.0, "mm": 1000.0}.get(distance_unit, 1.0)

    dist_scaled = distance * distance_mult

    # Clip impedance for display (handle inf values)
    impedance_display = np.clip(impedance, 0, 500)

    # Plot impedance profile
    ax.plot(dist_scaled, impedance_display, "b-", linewidth=2, label="Impedance")

    # Fill regions based on impedance deviation from z0
    for i in range(len(dist_scaled) - 1):
        z = impedance_display[i]
        if z > z0 + discontinuity_threshold:
            color = "#FFA500"  # Orange for high-Z
            alpha = 0.3
        elif z < z0 - discontinuity_threshold:
            color = "#1E90FF"  # Blue for low-Z
            alpha = 0.3
        else:
            color = "#90EE90"  # Light green for matched
            alpha = 0.2

        ax.fill_between(
            [dist_scaled[i], dist_scaled[i + 1]],
            [z0, z0],
            [z, impedance_display[i + 1]],
            color=color,
            alpha=alpha,
        )

    # Reference line
    if show_reference:
        ax.axhline(z0, color="gray", linestyle="--", linewidth=1.5, label=f"Z0 = {z0} Ω")

    # Find and annotate discontinuities
    if show_discontinuities:
        # Find significant changes
        z_diff = np.abs(np.diff(impedance_display))
        discontinuities = np.where(z_diff > discontinuity_threshold)[0]

        for idx in discontinuities:
            z_before = impedance_display[idx]
            z_after = impedance_display[idx + 1]
            d = dist_scaled[idx]

            # Determine discontinuity type
            if z_after > z_before + discontinuity_threshold:
                disc_type = "High-Z"
                color = "orange"
            elif z_after < z_before - discontinuity_threshold:
                disc_type = "Low-Z"
                color = "blue"
            else:
                continue

            # Add marker
            ax.plot(d, z_after, "o", color=color, markersize=8)

            # Add annotation
            z_str = f"{z_after:.0f}" if z_after < 500 else "Open"
            ax.annotate(
                f"{disc_type}\n{z_str} Ω",
                xy=(d, z_after),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

    # Labels and formatting
    ax.set_xlabel(f"Distance ({distance_unit})", fontsize=11)
    ax.set_ylabel("Impedance (Ω)", fontsize=11)
    ax.set_xlim(0, dist_scaled[-1])

    # Set y-axis limits with padding
    y_min = max(0, np.min(impedance_display) - 10)
    y_max = min(200, np.max(impedance_display) + 10)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("TDR Impedance Profile", fontsize=12, fontweight="bold")

    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_sparams_magnitude(
    frequencies: NDArray[np.floating[Any]],
    s11: NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]] | None = None,
    s21: NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]] | None = None,
    s12: NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]] | None = None,
    s22: NDArray[np.complexfloating[Any, Any]] | NDArray[np.floating[Any]] | None = None,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    freq_unit: str = "auto",
    show_markers: bool = True,
    db_3_marker: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot S-parameter magnitude vs frequency.

    Creates a frequency response plot showing S-parameter magnitudes
    in dB with optional -3dB marker for bandwidth measurement.

    Args:
        frequencies: Frequency array in Hz.
        s11: S11 (input reflection) - complex or dB values.
        s21: S21 (forward transmission) - complex or dB values.
        s12: S12 (reverse transmission) - complex or dB values.
        s22: S22 (output reflection) - complex or dB values.
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size in inches.
        title: Plot title.
        freq_unit: Frequency unit ("Hz", "kHz", "MHz", "GHz", "auto").
        show_markers: Show markers at key frequencies.
        db_3_marker: Show -3dB bandwidth marker for S21.
        show: Display plot interactively.
        save_path: Save plot to file.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> freq = np.linspace(1e6, 1e9, 1000)
        >>> s21 = 1 / (1 + 1j * freq / 100e6)  # Low-pass response
        >>> fig = plot_sparams_magnitude(freq, s21=s21)
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

    # Select frequency unit
    if freq_unit == "auto":
        max_freq = np.max(frequencies)
        if max_freq >= 1e9:
            freq_unit = "GHz"
            freq_div = 1e9
        elif max_freq >= 1e6:
            freq_unit = "MHz"
            freq_div = 1e6
        elif max_freq >= 1e3:
            freq_unit = "kHz"
            freq_div = 1e3
        else:
            freq_unit = "Hz"
            freq_div = 1.0
    else:
        freq_div = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(freq_unit, 1.0)

    freq_scaled = frequencies / freq_div

    def to_db(s: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """Convert S-parameter to dB."""
        if np.iscomplexobj(s):
            result: NDArray[np.floating[Any]] = 20 * np.log10(np.abs(s) + 1e-12)
            return result
        return np.asarray(s, dtype=np.float64)

    # Color scheme
    colors = {"S11": "#E74C3C", "S21": "#3498DB", "S12": "#2ECC71", "S22": "#9B59B6"}
    linestyles = {"S11": "-", "S21": "-", "S12": "--", "S22": "--"}

    params = [("S11", s11), ("S21", s21), ("S12", s12), ("S22", s22)]

    for name, s_param in params:
        if s_param is not None:
            s_db = to_db(s_param)
            ax.semilogx(
                freq_scaled,
                s_db,
                color=colors[name],
                linestyle=linestyles[name],
                linewidth=2,
                label=name,
            )

            # -3dB marker for S21
            if name == "S21" and db_3_marker:
                max_db = np.max(s_db)
                db_3_level = max_db - 3

                # Find -3dB crossover
                crossings = np.where(np.diff(np.sign(s_db - db_3_level)))[0]
                if len(crossings) > 0:
                    f_3db = float(freq_scaled[crossings[0]])
                    db_3_level_float = float(db_3_level)
                    ax.axhline(
                        db_3_level_float, color="gray", linestyle=":", alpha=0.7, linewidth=1
                    )
                    ax.axvline(f_3db, color="gray", linestyle=":", alpha=0.7, linewidth=1)
                    ax.plot(f_3db, db_3_level_float, "ko", markersize=6)
                    ax.annotate(
                        f"-3dB: {f_3db:.2f} {freq_unit}",
                        xy=(f_3db, db_3_level_float),
                        xytext=(10, -15),
                        textcoords="offset points",
                        fontsize=9,
                        ha="left",
                    )

    # Labels and formatting
    ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=11)
    ax.set_ylabel("Magnitude (dB)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("S-Parameter Magnitude Response", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_sparams_phase(
    frequencies: NDArray[np.floating[Any]],
    s11: NDArray[np.complexfloating[Any, Any]] | None = None,
    s21: NDArray[np.complexfloating[Any, Any]] | None = None,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    freq_unit: str = "auto",
    unwrap: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot S-parameter phase vs frequency.

    Args:
        frequencies: Frequency array in Hz.
        s11: S11 complex values.
        s21: S21 complex values.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        freq_unit: Frequency unit.
        unwrap: Unwrap phase discontinuities.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
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

    # Select frequency unit
    if freq_unit == "auto":
        max_freq = np.max(frequencies)
        if max_freq >= 1e9:
            freq_unit = "GHz"
            freq_div = 1e9
        elif max_freq >= 1e6:
            freq_unit = "MHz"
            freq_div = 1e6
        else:
            freq_unit = "kHz"
            freq_div = 1e3
    else:
        freq_div = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}.get(freq_unit, 1.0)

    freq_scaled = frequencies / freq_div

    colors = {"S11": "#E74C3C", "S21": "#3498DB"}

    for name, s_param in [("S11", s11), ("S21", s21)]:
        if s_param is not None:
            phase = np.angle(s_param, deg=True)
            if unwrap:
                phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

            ax.semilogx(freq_scaled, phase, color=colors[name], linewidth=2, label=name)

    ax.set_xlabel(f"Frequency ({freq_unit})", fontsize=11)
    ax.set_ylabel("Phase (degrees)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("S-Parameter Phase Response", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_setup_hold_timing(
    clock_edges: NDArray[np.floating[Any]],
    data_edges: NDArray[np.floating[Any]],
    setup_time: float,
    hold_time: float,
    *,
    clock_data: NDArray[np.floating[Any]] | None = None,
    data_data: NDArray[np.floating[Any]] | None = None,
    time_axis: NDArray[np.floating[Any]] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (14, 8),
    title: str | None = None,
    time_unit: str = "auto",
    show_margins: bool = True,
    setup_spec: float | None = None,
    hold_spec: float | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot setup/hold timing diagram with annotations.

    Creates a timing diagram showing clock and data relationships
    with setup and hold time annotations and optional pass/fail
    indication against specifications.

    Args:
        clock_edges: Array of clock edge times (rising edges).
        data_edges: Array of data transition times.
        setup_time: Measured setup time (seconds).
        hold_time: Measured hold time (seconds).
        clock_data: Optional clock waveform for display.
        data_data: Optional data waveform for display.
        time_axis: Time axis for waveforms.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time unit ("s", "ms", "us", "ns", "ps", "auto").
        show_margins: Show setup/hold timing arrows.
        setup_spec: Setup time specification for pass/fail.
        hold_spec: Hold time specification for pass/fail.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> clk_edges = np.array([0, 10e-9, 20e-9])
        >>> data_edges = np.array([8e-9, 18e-9])
        >>> fig = plot_setup_hold_timing(
        ...     clk_edges, data_edges,
        ...     setup_time=2e-9, hold_time=1e-9,
        ...     setup_spec=1e-9, hold_spec=0.5e-9
        ... )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    # Create figure with multiple rows
    if ax is not None:
        fig_temp = ax.get_figure()
        if fig_temp is None:
            raise ValueError("Axes must have an associated figure")
        fig = cast("Figure", fig_temp)
        axes = [ax]
        n_rows = 1
    else:
        n_rows = 3 if clock_data is not None else 1
        fig, axes = plt.subplots(
            n_rows, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1] * n_rows}
        )
        if n_rows == 1:
            axes = [axes]

    # Select time unit
    if time_unit == "auto":
        max_time = max(np.max(clock_edges), np.max(data_edges))
        if max_time < 1e-9:
            time_unit = "ps"
            time_mult = 1e12
        elif max_time < 1e-6:
            time_unit = "ns"
            time_mult = 1e9
        elif max_time < 1e-3:
            time_unit = "us"
            time_mult = 1e6
        else:
            time_unit = "ms"
            time_mult = 1e3
    else:
        time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(time_unit, 1e9)

    setup_scaled = setup_time * time_mult
    hold_scaled = hold_time * time_mult

    # If waveforms provided, plot them
    if clock_data is not None and data_data is not None and time_axis is not None:
        time_scaled = time_axis * time_mult

        # Clock waveform
        ax_clk = axes[0]
        ax_clk.step(time_scaled, clock_data, where="post", color="#3498DB", linewidth=2)
        ax_clk.set_ylabel(
            "CLK", rotation=0, ha="right", va="center", fontsize=11, fontweight="bold"
        )
        ax_clk.set_ylim(-0.2, 1.3)
        ax_clk.set_yticks([0, 1])
        ax_clk.grid(True, axis="x", alpha=0.3)

        # Data waveform
        ax_data = axes[1]
        ax_data.step(time_scaled, data_data, where="post", color="#E74C3C", linewidth=2)
        ax_data.set_ylabel(
            "DATA", rotation=0, ha="right", va="center", fontsize=11, fontweight="bold"
        )
        ax_data.set_ylim(-0.2, 1.3)
        ax_data.set_yticks([0, 1])
        ax_data.grid(True, axis="x", alpha=0.3)

        ax_timing = axes[2] if len(axes) > 2 else axes[-1]
    else:
        ax_timing = axes[0]

    # Timing annotation panel
    ax_timing.set_ylim(0, 1)
    ax_timing.set_xlim(0, max(clock_edges[-1], data_edges[-1]) * time_mult * 1.1)
    ax_timing.axis("off")

    # Draw timing arrows for first clock edge
    if len(clock_edges) > 0 and len(data_edges) > 0:
        clk_edge = clock_edges[0] * time_mult

        # Find nearest data edge before clock
        data_before = data_edges[data_edges < clock_edges[0]]
        if len(data_before) > 0:
            data_edge = data_before[-1] * time_mult

            # Setup time arrow (data_edge to clk_edge)
            if show_margins:
                y_setup = 0.7
                ax_timing.annotate(
                    "",
                    xy=(clk_edge, y_setup),
                    xytext=(data_edge, y_setup),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": "#27AE60",
                        "lw": 2,
                    },
                )
                ax_timing.text(
                    (data_edge + clk_edge) / 2,
                    y_setup + 0.1,
                    f"Setup: {setup_scaled:.2f} {time_unit}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#27AE60",
                )

        # Find nearest data edge after clock
        data_after = data_edges[data_edges > clock_edges[0]]
        if len(data_after) > 0:
            data_edge_after = data_after[0] * time_mult

            # Hold time arrow (clk_edge to data_edge_after)
            if show_margins:
                y_hold = 0.3
                ax_timing.annotate(
                    "",
                    xy=(data_edge_after, y_hold),
                    xytext=(clk_edge, y_hold),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": "#E67E22",
                        "lw": 2,
                    },
                )
                ax_timing.text(
                    (clk_edge + data_edge_after) / 2,
                    y_hold + 0.1,
                    f"Hold: {hold_scaled:.2f} {time_unit}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#E67E22",
                )

    # Add pass/fail status
    status_y = 0.9
    if setup_spec is not None:
        setup_pass = setup_time >= setup_spec
        status = "PASS" if setup_pass else "FAIL"
        color = "#27AE60" if setup_pass else "#E74C3C"
        ax_timing.text(
            0.02,
            status_y,
            f"Setup: {status} (spec: {setup_spec * time_mult:.2f} {time_unit})",
            transform=ax_timing.transAxes,
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        status_y -= 0.15

    if hold_spec is not None:
        hold_pass = hold_time >= hold_spec
        status = "PASS" if hold_pass else "FAIL"
        color = "#27AE60" if hold_pass else "#E74C3C"
        ax_timing.text(
            0.02,
            status_y,
            f"Hold: {status} (spec: {hold_spec * time_mult:.2f} {time_unit})",
            transform=ax_timing.transAxes,
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    # Set x-label on bottom axes
    axes[-1].set_xlabel(f"Time ({time_unit})", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    else:
        fig.suptitle("Setup/Hold Timing Analysis", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_timing_margin(
    setup_times: NDArray[np.floating[Any]],
    hold_times: NDArray[np.floating[Any]],
    *,
    setup_spec: float | None = None,
    hold_spec: float | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    time_unit: str = "ns",
    show: bool = True,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot setup vs hold timing margin scatter plot.

    Creates a scatter plot showing the distribution of setup and hold
    times with specification regions marked.

    Args:
        setup_times: Array of setup time measurements.
        hold_times: Array of hold time measurements.
        setup_spec: Setup time specification.
        hold_spec: Hold time specification.
        ax: Matplotlib axes.
        figsize: Figure size.
        title: Plot title.
        time_unit: Time unit for display.
        show: Display plot.
        save_path: Save path.

    Returns:
        Matplotlib Figure object.
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

    time_mult = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}.get(time_unit, 1e9)

    setup_scaled = setup_times * time_mult
    hold_scaled = hold_times * time_mult

    # Scatter plot
    ax.scatter(setup_scaled, hold_scaled, c="#3498DB", alpha=0.6, s=50)

    # Add specification lines if provided
    if setup_spec is not None:
        spec_scaled = setup_spec * time_mult
        ax.axvline(
            spec_scaled,
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            label=f"Setup Spec ({spec_scaled:.2f} {time_unit})",
        )

    if hold_spec is not None:
        spec_scaled = hold_spec * time_mult
        ax.axhline(
            spec_scaled,
            color="#E67E22",
            linestyle="--",
            linewidth=2,
            label=f"Hold Spec ({spec_scaled:.2f} {time_unit})",
        )

    # Mark pass/fail regions
    if setup_spec is not None and hold_spec is not None:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        # Pass region (upper right)
        ax.fill_between(
            [setup_spec * time_mult, x_lim[1]],
            [hold_spec * time_mult, hold_spec * time_mult],
            [y_lim[1], y_lim[1]],
            color="#27AE60",
            alpha=0.1,
            label="Pass Region",
        )

    ax.set_xlabel(f"Setup Time ({time_unit})", fontsize=11)
    ax.set_ylabel(f"Hold Time ({time_unit})", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Setup/Hold Timing Margin", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
