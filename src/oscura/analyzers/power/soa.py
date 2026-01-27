"""Safe Operating Area (SOA) analysis for Oscura.

Provides SOA checking and visualization for power semiconductor devices.


Example:
    >>> from oscura.analyzers.power.soa import soa_analysis, SOALimit
    >>> limits = [
    ...     SOALimit(v_max=100, i_max=50, pulse_width=1e-6),
    ...     SOALimit(v_max=80, i_max=100, pulse_width=10e-6),
    ... ]
    >>> result = soa_analysis(v_trace, i_trace, limits)
    >>> print(f"SOA violations: {len(result['violations'])}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from oscura.core.types import WaveformTrace


@dataclass
class SOALimit:
    """Defines a point on the SOA boundary.

    The SOA is typically defined by a piecewise linear boundary in
    log-log V-I space, with different limits for different pulse widths.

    Attributes:
        v_max: Maximum voltage at this limit point (Volts).
        i_max: Maximum current at this limit point (Amps).
        pulse_width: Pulse duration for this limit (seconds).
                    Use np.inf for DC limit.
        name: Optional name for this limit point.
    """

    v_max: float
    i_max: float
    pulse_width: float = np.inf
    name: str = ""


@dataclass
class SOAViolation:
    """Information about an SOA violation.

    Attributes:
        timestamp: Time of violation (seconds).
        sample_index: Sample index of violation.
        voltage: Voltage at violation point.
        current: Current at violation point.
        limit: The SOA limit that was violated.
        margin: How far inside (negative) or outside (positive) the limit.
    """

    timestamp: float
    sample_index: int
    voltage: float
    current: float
    limit: SOALimit
    margin: float


def soa_analysis(
    voltage: WaveformTrace,
    current: WaveformTrace,
    limits: list[SOALimit],
    *,
    pulse_width: float | None = None,
) -> dict[str, Any]:
    """Analyze voltage-current trajectory against SOA limits.

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.
        limits: List of SOA limit points.
        pulse_width: Pulse width to use for limit selection.
                    If None, uses DC limits.

    Returns:
        Dictionary with:
        - passed: True if no violations
        - violations: List of SOAViolation objects
        - v_trajectory: Voltage values
        - i_trajectory: Current values
        - min_margin: Minimum margin to SOA boundary (negative = inside)
        - applicable_limits: Limits used for this pulse width

    Example:
        >>> result = soa_analysis(v_ds, i_d, soa_limits)
        >>> if not result['passed']:
        ...     print(f"SOA violated at {result['violations'][0].timestamp}s")
    """
    v_data = voltage.data
    i_data = current.data

    # Ensure same length
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]
    sample_period = voltage.metadata.time_base

    # Select applicable limits based on pulse width
    if pulse_width is None:
        pulse_width = np.inf

    applicable_limits = [l for l in limits if l.pulse_width >= pulse_width]
    if not applicable_limits:
        applicable_limits = limits  # Use all if none match

    # Build SOA boundary (interpolate between limit points)
    # Sort by voltage
    sorted_limits = sorted(applicable_limits, key=lambda l: l.v_max)

    violations: list[SOAViolation] = []
    margins: list[float] = []

    for i in range(len(v_data)):
        v = abs(v_data[i])
        current_i = abs(i_data[i])

        # Find applicable limit (linear interpolation in log-log space)
        i_limit = _interpolate_soa_limit(v, sorted_limits)

        margin = i_limit - current_i
        margins.append(margin)

        if current_i > i_limit:
            # Find which specific limit was violated
            for limit in sorted_limits:
                if v <= limit.v_max and current_i > limit.i_max:
                    violations.append(
                        SOAViolation(
                            timestamp=i * sample_period,
                            sample_index=i,
                            voltage=float(v_data[i]),
                            current=float(i_data[i]),
                            limit=limit,
                            margin=-margin,
                        )
                    )
                    break

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "v_trajectory": v_data,
        "i_trajectory": i_data,
        "min_margin": float(np.min(margins)) if margins else 0.0,
        "applicable_limits": applicable_limits,
    }


def _interpolate_soa_limit(voltage: float, limits: list[SOALimit]) -> float:
    """Interpolate SOA current limit at given voltage.

    Uses log-log interpolation between limit points.

    Args:
        voltage: Voltage at which to interpolate current limit
        limits: List of SOALimit points defining the boundary

    Returns:
        Interpolated current limit in Amps
    """
    if len(limits) == 0:
        return np.inf

    if len(limits) == 1:
        if voltage <= limits[0].v_max:
            return limits[0].i_max
        return 0.0

    # Find bracketing points
    for i in range(len(limits) - 1):
        if limits[i].v_max <= voltage <= limits[i + 1].v_max:
            # Log-log interpolation
            v1, v2 = limits[i].v_max, limits[i + 1].v_max
            i1, i2 = limits[i].i_max, limits[i + 1].i_max

            if v1 <= 0 or v2 <= 0 or i1 <= 0 or i2 <= 0:
                # Linear interpolation fallback
                t = (voltage - v1) / (v2 - v1)
                return i1 + t * (i2 - i1)

            # Log-log
            log_v = np.log10(voltage)
            log_v1, log_v2 = np.log10(v1), np.log10(v2)
            log_i1, log_i2 = np.log10(i1), np.log10(i2)

            t = (log_v - log_v1) / (log_v2 - log_v1)
            log_i = log_i1 + t * (log_i2 - log_i1)
            return 10**log_i  # type: ignore[no-any-return]

    # Beyond limits
    if voltage < limits[0].v_max:
        return limits[0].i_max
    return 0.0


def check_soa_violations(
    voltage: WaveformTrace,
    current: WaveformTrace,
    limits: list[SOALimit],
) -> list[SOAViolation]:
    """Check for SOA violations and return list of violations.

    Convenience function that just returns violations.

    Args:
        voltage: Voltage waveform.
        current: Current waveform.
        limits: SOA limits.

    Returns:
        List of SOA violations (empty if all within limits).
    """
    result = soa_analysis(voltage, current, limits)
    return result["violations"]  # type: ignore[no-any-return]


def plot_soa(
    voltage: WaveformTrace,
    current: WaveformTrace,
    limits: list[SOALimit],
    *,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
    show_violations: bool = True,
) -> Figure:
    """Plot SOA diagram with trajectory and limits.

    Args:
        voltage: Voltage waveform.
        current: Current waveform.
        limits: SOA limits to plot.
        figsize: Figure size in inches.
        title: Plot title.
        show_violations: If True, highlight violations.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> fig = plot_soa(v_ds, i_d, soa_limits)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    result = soa_analysis(voltage, current, limits)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot SOA boundary
    sorted_limits = sorted(limits, key=lambda l: l.v_max)
    v_boundary = [l.v_max for l in sorted_limits]
    i_boundary = [l.i_max for l in sorted_limits]

    # Add corner points for closed boundary
    v_boundary = [0, *v_boundary, v_boundary[-1], 0]
    i_boundary = [i_boundary[0], *i_boundary, 0, 0]

    ax.fill(v_boundary, i_boundary, alpha=0.2, color="green", label="Safe Operating Area")
    ax.plot(v_boundary, i_boundary, "g-", linewidth=2)

    # Plot trajectory
    v_traj = np.abs(result["v_trajectory"])
    i_traj = np.abs(result["i_trajectory"])
    ax.plot(v_traj, i_traj, "b-", linewidth=1, alpha=0.7, label="Operating trajectory")

    # Highlight violations
    if show_violations and result["violations"]:
        v_viol = [v.voltage for v in result["violations"]]
        i_viol = [v.current for v in result["violations"]]
        ax.scatter(
            np.abs(v_viol),
            np.abs(i_viol),
            c="red",
            s=50,
            marker="x",
            label=f"Violations ({len(result['violations'])})",
        )

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        ax.set_title(title)
    else:
        status = "PASS" if result["passed"] else "FAIL"
        ax.set_title(f"Safe Operating Area Analysis - {status}")

    plt.tight_layout()
    return fig


def create_mosfet_soa(
    v_ds_max: float,
    i_d_max: float,
    p_d_max: float,
    *,
    pulse_limits: dict[float, float] | None = None,
) -> list[SOALimit]:
    """Create SOA limits for a MOSFET.

    Generates typical SOA boundary from datasheet parameters.

    Args:
        v_ds_max: Maximum drain-source voltage.
        i_d_max: Maximum continuous drain current.
        p_d_max: Maximum power dissipation.
        pulse_limits: Optional dict of {pulse_width: i_max} for pulsed limits.

    Returns:
        List of SOALimit objects defining the boundary.

    Example:
        >>> limits = create_mosfet_soa(v_ds_max=100, i_d_max=50, p_d_max=150)
    """
    limits = []

    # Current limit (horizontal line at I_max)
    limits.append(SOALimit(v_max=1.0, i_max=i_d_max, name="I_max"))

    # Power limit (hyperbola P = V * I)
    # Find intersection with I_max line
    v_at_imax = p_d_max / i_d_max
    if v_at_imax < v_ds_max:
        limits.append(SOALimit(v_max=v_at_imax, i_max=i_d_max, name="P_max_start"))

        # Add points along power hyperbola
        for v in np.geomspace(v_at_imax, v_ds_max * 0.9, 5)[1:]:
            i = p_d_max / v
            limits.append(SOALimit(v_max=v, i_max=i, name="P_max"))

    # Voltage limit
    limits.append(SOALimit(v_max=v_ds_max, i_max=0.1, name="V_max"))

    # Add pulsed limits if provided
    if pulse_limits:
        for pw, i_max in pulse_limits.items():
            limits.append(
                SOALimit(
                    v_max=v_ds_max,
                    i_max=i_max,
                    pulse_width=pw,
                    name=f"Pulse_{pw * 1e6:.0f}us",
                )
            )

    return limits


__all__ = [
    "SOALimit",
    "SOAViolation",
    "check_soa_violations",
    "create_mosfet_soa",
    "plot_soa",
    "soa_analysis",
]
