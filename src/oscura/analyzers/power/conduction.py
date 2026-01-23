"""Conduction loss analysis for Oscura.

Provides conduction loss calculations for power semiconductor devices
during their on-state.


Example:
    >>> from oscura.analyzers.power.conduction import conduction_loss
    >>> p_cond = conduction_loss(v_on_trace, i_d_trace, duty_cycle=0.5)
    >>> print(f"Conduction loss: {p_cond:.2f} W")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def conduction_loss(
    voltage: WaveformTrace,
    current: WaveformTrace,
    duty_cycle: float | None = None,
) -> float:
    """Calculate conduction loss during on-state.

    P_cond = V_on * I_on * D (steady state)
    or
    P_cond = mean(V(t) * I(t)) over on-state periods

    Args:
        voltage: On-state voltage trace (V_ds(on) or V_ce(sat)).
        current: On-state current trace.
        duty_cycle: Duty cycle (0 to 1). If None, calculates from waveforms.

    Returns:
        Conduction loss in Watts.

    Example:
        >>> p_cond = conduction_loss(v_on, i_d, duty_cycle=0.5)
        >>> print(f"Conduction loss: {p_cond:.2f} W")

    References:
        Infineon Application Note AN-9010
    """
    v_data = voltage.data
    i_data = current.data

    # Ensure same length
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    if duty_cycle is not None:
        # Use average values and duty cycle
        v_on = float(np.mean(v_data))
        i_on = float(np.mean(i_data))
        return v_on * i_on * duty_cycle
    else:
        # Calculate instantaneous power and average
        power = v_data * i_data
        return float(np.mean(power))


def on_resistance(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    min_current: float | None = None,
    min_current_fraction: float = 0.1,
) -> float:
    """Calculate on-state resistance (R_ds(on) or R_ce(sat)).

    R_on = V_on / I_on

    Args:
        voltage: On-state voltage trace.
        current: On-state current trace.
        min_current: Minimum current threshold (to avoid division by small values).
            If None, uses min_current_fraction * peak current.
        min_current_fraction: Fraction of peak current to use as minimum threshold
            when min_current is None (default: 0.1 = 10% of peak).

    Returns:
        On-state resistance in Ohms.

    Example:
        >>> r_on = on_resistance(v_ds, i_d)
        >>> print(f"R_ds(on): {r_on*1e3:.2f} mOhm")
        >>> # With tighter threshold (5% of peak):
        >>> r_on = on_resistance(v_ds, i_d, min_current_fraction=0.05)
    """
    v_data = voltage.data
    i_data = current.data

    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    # Filter by minimum current
    if min_current is None:
        min_current = min_current_fraction * np.max(np.abs(i_data))

    mask = np.abs(i_data) >= min_current
    if not np.any(mask):
        return np.nan

    v_on = v_data[mask]
    i_on = i_data[mask]

    # Linear fit to get resistance
    # V = I * R -> slope is R
    coeffs = np.polyfit(i_on, v_on, 1)
    return float(coeffs[0])


def forward_voltage(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    current_threshold: float | None = None,
    current_threshold_fraction: float = 0.1,
    threshold_window: float = 0.1,
) -> float:
    """Calculate forward voltage drop (for diodes or IGBT V_ce(sat)).

    Extracts the voltage at a reference current level.

    Args:
        voltage: Forward voltage trace.
        current: Forward current trace.
        current_threshold: Current level at which to measure Vf.
            If None, uses current_threshold_fraction * peak current.
        current_threshold_fraction: Fraction of peak current to use as threshold
            when current_threshold is None (default: 0.1 = 10% of peak).
        threshold_window: Tolerance window around the current threshold,
            as a fraction of the threshold (default: 0.1 = +/-10% window).
            Samples within this window are averaged to compute Vf.

    Returns:
        Forward voltage in Volts.

    Example:
        >>> vf = forward_voltage(v_f, i_f)
        >>> print(f"Forward voltage: {vf:.2f} V")
        >>> # With tighter window for more precise measurement:
        >>> vf = forward_voltage(v_f, i_f, threshold_window=0.05)
    """
    v_data = voltage.data
    i_data = current.data

    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    if current_threshold is None:
        current_threshold = current_threshold_fraction * np.max(np.abs(i_data))

    # Find samples near the current threshold
    near_threshold = np.abs(i_data - current_threshold) < threshold_window * current_threshold
    if not np.any(near_threshold):
        # Interpolate
        idx = np.argmin(np.abs(i_data - current_threshold))
        return float(v_data[idx])

    return float(np.mean(v_data[near_threshold]))


def duty_cycle_weighted_loss(
    losses: list[tuple[float, float]],
) -> float:
    """Calculate total loss from multiple operating points.

    Useful for variable duty cycle or multi-mode operation.

    Args:
        losses: List of (loss_watts, duty_cycle) tuples.

    Returns:
        Total weighted average loss in Watts.

    Raises:
        AnalysisError: If total duty cycle exceeds 1.0

    Example:
        >>> # 10W at 30% duty, 5W at 50% duty
        >>> total = duty_cycle_weighted_loss([(10, 0.3), (5, 0.5)])
    """
    total = 0.0
    total_duty = 0.0

    for loss, duty in losses:
        total += loss * duty
        total_duty += duty

    if total_duty > 1.0:
        raise AnalysisError(f"Total duty cycle exceeds 1.0: {total_duty}")

    return total


def temperature_derating(
    r_on_25c: float,
    temperature: float,
    temp_coefficient: float = 0.004,
) -> float:
    """Calculate temperature-derated on-resistance.

    R_on(T) = R_on(25C) * (1 + alpha * (T - 25))

    Args:
        r_on_25c: On-resistance at 25C in Ohms.
        temperature: Operating temperature in Celsius.
        temp_coefficient: Temperature coefficient (default 0.4%/C for Si MOSFET).

    Returns:
        Derated on-resistance in Ohms.

    Example:
        >>> r_on_100c = temperature_derating(0.010, 100)  # 10mOhm at 25C
        >>> print(f"R_on at 100C: {r_on_100c*1e3:.2f} mOhm")
    """
    return r_on_25c * (1 + temp_coefficient * (temperature - 25))


def mosfet_conduction_loss(
    i_rms: float,
    r_ds_on: float,
    temperature: float = 25.0,
    temp_coefficient: float = 0.004,
) -> float:
    """Calculate MOSFET conduction loss.

    P_cond = I_rms^2 * R_ds(on)

    Args:
        i_rms: RMS drain current in Amps.
        r_ds_on: On-state resistance at 25C in Ohms.
        temperature: Junction temperature in Celsius.
        temp_coefficient: Temperature coefficient.

    Returns:
        Conduction loss in Watts.

    Example:
        >>> p = mosfet_conduction_loss(i_rms=10, r_ds_on=0.010, temperature=100)
    """
    r_on = temperature_derating(r_ds_on, temperature, temp_coefficient)
    return i_rms**2 * r_on


def diode_conduction_loss(
    i_avg: float,
    i_rms: float,
    v_f: float,
    r_d: float = 0.0,
) -> float:
    """Calculate diode conduction loss.

    P_cond = V_f * I_avg + r_d * I_rms^2

    Args:
        i_avg: Average forward current in Amps.
        i_rms: RMS forward current in Amps.
        v_f: Forward voltage drop in Volts.
        r_d: Dynamic resistance in Ohms.

    Returns:
        Conduction loss in Watts.

    Example:
        >>> p = diode_conduction_loss(i_avg=5, i_rms=7, v_f=0.7, r_d=0.01)
    """
    return v_f * i_avg + r_d * i_rms**2


def igbt_conduction_loss(
    i_c: float,
    v_ce_sat: float,
    r_c: float = 0.0,
) -> float:
    """Calculate IGBT conduction loss.

    P_cond = V_ce(sat) * I_c + r_c * I_c^2

    Args:
        i_c: Collector current in Amps.
        v_ce_sat: Collector-emitter saturation voltage in Volts.
        r_c: Collector resistance in Ohms.

    Returns:
        Conduction loss in Watts.

    Example:
        >>> p = igbt_conduction_loss(i_c=50, v_ce_sat=2.0, r_c=0.01)
    """
    return v_ce_sat * i_c + r_c * i_c**2


__all__ = [
    "conduction_loss",
    "diode_conduction_loss",
    "duty_cycle_weighted_loss",
    "forward_voltage",
    "igbt_conduction_loss",
    "mosfet_conduction_loss",
    "on_resistance",
    "temperature_derating",
]
