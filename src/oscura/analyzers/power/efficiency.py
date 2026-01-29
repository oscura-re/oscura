"""Power efficiency calculations for Oscura.

Provides efficiency calculations for power converters and systems.


Example:
    >>> from oscura.analyzers.power.efficiency import efficiency
    >>> eta = efficiency(v_in, i_in, v_out, i_out)
    >>> print(f"Efficiency: {eta*100:.1f}%")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.power.basic import average_power, instantaneous_power

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def efficiency(
    v_in: WaveformTrace,
    i_in: WaveformTrace,
    v_out: WaveformTrace,
    i_out: WaveformTrace,
) -> float:
    """Calculate power conversion efficiency.

    eta = P_out / P_in * 100%

    Args:
        v_in: Input voltage trace.
        i_in: Input current trace.
        v_out: Output voltage trace.
        i_out: Output current trace.

    Returns:
        Efficiency as a ratio (0 to 1).

    Example:
        >>> eta = efficiency(v_in, i_in, v_out, i_out)
        >>> print(f"Efficiency: {eta*100:.1f}%")
    """
    p_in = average_power(voltage=v_in, current=i_in)
    p_out = average_power(voltage=v_out, current=i_out)

    if p_in <= 0:
        return 0.0

    return p_out / p_in


def power_conversion_efficiency(
    p_in: float,
    p_out: float,
) -> float:
    """Calculate efficiency from power values.

    Args:
        p_in: Input power in Watts.
        p_out: Output power in Watts.

    Returns:
        Efficiency as a ratio (0 to 1).

    Example:
        >>> eta = power_conversion_efficiency(p_in=100, p_out=90)
        >>> print(f"Efficiency: {eta*100:.1f}%")
    """
    if p_in <= 0:
        return 0.0
    return p_out / p_in


def multi_output_efficiency(
    v_in: WaveformTrace,
    i_in: WaveformTrace,
    outputs: list[tuple[WaveformTrace, WaveformTrace]],
) -> dict[str, float]:
    """Calculate efficiency for multi-output power supply.

    Args:
        v_in: Input voltage trace.
        i_in: Input current trace.
        outputs: List of (v_out, i_out) trace tuples for each output.

    Returns:
        Dictionary with:
        - total_efficiency: Overall efficiency
        - output_N_efficiency: Per-output efficiency (contribution)
        - output_N_power: Per-output power
        - total_output_power: Sum of all output powers
        - input_power: Input power
        - losses: Power losses (P_in - P_out_total)

    Example:
        >>> outputs = [(v1, i1), (v2, i2), (v3, i3)]
        >>> result = multi_output_efficiency(v_in, i_in, outputs)
        >>> print(f"Total efficiency: {result['total_efficiency']*100:.1f}%")
    """
    p_in = average_power(voltage=v_in, current=i_in)

    result = {
        "input_power": p_in,
    }

    total_output = 0.0
    for idx, (v_out, i_out) in enumerate(outputs):
        p_out = average_power(voltage=v_out, current=i_out)
        result[f"output_{idx + 1}_power"] = p_out
        result[f"output_{idx + 1}_efficiency"] = p_out / p_in if p_in > 0 else 0.0
        total_output += p_out

    result["total_output_power"] = total_output
    result["total_efficiency"] = total_output / p_in if p_in > 0 else 0.0
    result["losses"] = p_in - total_output

    return result


def efficiency_vs_load(
    v_in: WaveformTrace,
    i_in: WaveformTrace,
    v_out: WaveformTrace,
    i_out: WaveformTrace,
    *,
    n_points: int = 100,
) -> dict[str, NDArray[np.float64]]:
    """Calculate efficiency across the load range.

    Segments the waveforms and calculates efficiency at each load level.

    Args:
        v_in: Input voltage trace.
        i_in: Input current trace.
        v_out: Output voltage trace.
        i_out: Output current trace.
        n_points: Number of load points to evaluate.

    Returns:
        Dictionary with:
        - load_percent: Load levels as percentage of max
        - efficiency: Efficiency at each load level
        - output_power: Output power at each load level
        - input_power: Input power at each load level

    Example:
        >>> result = efficiency_vs_load(v_in, i_in, v_out, i_out)
        >>> plt.plot(result['load_percent'], result['efficiency'] * 100)
    """
    # Calculate instantaneous power
    p_out_trace = instantaneous_power(v_out, i_out)
    p_in_trace = instantaneous_power(v_in, i_in)

    p_out_data = p_out_trace.data
    p_in_data = p_in_trace.data[: len(p_out_data)]

    # Sort by output power to get load curve
    sort_idx = np.argsort(p_out_data)
    p_out_sorted = p_out_data[sort_idx]
    p_in_sorted = p_in_data[sort_idx]

    # Divide into bins
    bin_size = len(p_out_sorted) // n_points
    bin_size = max(bin_size, 1)

    load_pct = []
    efficiency_vals = []
    p_out_vals = []
    p_in_vals = []

    max_p_out = np.max(p_out_data)

    for i in range(0, len(p_out_sorted), bin_size):
        bin_p_out = np.mean(p_out_sorted[i : i + bin_size])
        bin_p_in = np.mean(p_in_sorted[i : i + bin_size])

        load_pct.append(bin_p_out / max_p_out * 100 if max_p_out > 0 else 0)
        p_out_vals.append(bin_p_out)
        p_in_vals.append(bin_p_in)
        efficiency_vals.append(bin_p_out / bin_p_in if bin_p_in > 0 else 0)

    return {
        "load_percent": np.array(load_pct),
        "efficiency": np.array(efficiency_vals),
        "output_power": np.array(p_out_vals),
        "input_power": np.array(p_in_vals),
    }


def loss_breakdown(
    v_in: WaveformTrace,
    i_in: WaveformTrace,
    v_out: WaveformTrace,
    i_out: WaveformTrace,
    *,
    switching_loss: float = 0.0,
    conduction_loss: float = 0.0,
    magnetic_loss: float = 0.0,
    gate_drive_loss: float = 0.0,
) -> dict[str, float]:
    """Break down power losses by category.

    Args:
        v_in: Input voltage trace.
        i_in: Input current trace.
        v_out: Output voltage trace.
        i_out: Output current trace.
        switching_loss: Known switching losses in Watts.
        conduction_loss: Known conduction losses in Watts.
        magnetic_loss: Known magnetic (core/copper) losses in Watts.
        gate_drive_loss: Known gate drive losses in Watts.

    Returns:
        Dictionary with loss breakdown.

    Example:
        >>> result = loss_breakdown(v_in, i_in, v_out, i_out,
        ...     switching_loss=2.0, conduction_loss=1.5, magnetic_loss=0.5)
        >>> print(f"Other losses: {result['other_loss']:.2f} W")
    """
    p_in = average_power(voltage=v_in, current=i_in)
    p_out = average_power(voltage=v_out, current=i_out)
    total_loss = p_in - p_out

    known_losses = switching_loss + conduction_loss + magnetic_loss + gate_drive_loss
    other_loss = total_loss - known_losses

    eta = p_out / p_in if p_in > 0 else 0.0

    return {
        "input_power": p_in,
        "output_power": p_out,
        "efficiency": eta,
        "total_loss": total_loss,
        "switching_loss": switching_loss,
        "conduction_loss": conduction_loss,
        "magnetic_loss": magnetic_loss,
        "gate_drive_loss": gate_drive_loss,
        "other_loss": max(0, other_loss),  # Clamp to non-negative
        "switching_loss_percent": switching_loss / total_loss * 100 if total_loss > 0 else 0,
        "conduction_loss_percent": conduction_loss / total_loss * 100 if total_loss > 0 else 0,
        "magnetic_loss_percent": magnetic_loss / total_loss * 100 if total_loss > 0 else 0,
    }


def thermal_efficiency(
    p_in: float,
    p_out: float,
    ambient_temp: float,
    case_temp: float,
    thermal_resistance: float,
) -> dict[str, float]:
    """Calculate thermal-related efficiency metrics.

    Args:
        p_in: Input power in Watts.
        p_out: Output power in Watts.
        ambient_temp: Ambient temperature in Celsius.
        case_temp: Case/junction temperature in Celsius.
        thermal_resistance: Thermal resistance (Rth_j-a) in C/W.

    Returns:
        Dictionary with thermal analysis.

    Example:
        >>> result = thermal_efficiency(100, 90, 25, 65, 2.5)
        >>> print(f"Estimated losses: {result['estimated_losses']:.1f} W")
    """
    losses = p_in - p_out
    eta = p_out / p_in if p_in > 0 else 0.0

    # Estimate losses from thermal measurement
    thermal_losses = (case_temp - ambient_temp) / thermal_resistance

    return {
        "efficiency": eta,
        "electrical_losses": losses,
        "thermal_estimated_losses": thermal_losses,
        "temperature_rise": case_temp - ambient_temp,
        "loss_discrepancy": abs(losses - thermal_losses),
    }


__all__ = [
    "efficiency",
    "efficiency_vs_load",
    "loss_breakdown",
    "multi_output_efficiency",
    "power_conversion_efficiency",
    "thermal_efficiency",
]
