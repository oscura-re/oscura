"""Switching loss analysis for Oscura.

Provides switching loss calculations for power semiconductor devices
including MOSFETs, IGBTs, and diodes.


Example:
    >>> from oscura.analyzers.power.switching import switching_loss
    >>> losses = switching_loss(v_ds_trace, i_d_trace)
    >>> print(f"Turn-on: {losses['e_on']*1e6:.2f} uJ")
    >>> print(f"Turn-off: {losses['e_off']*1e6:.2f} uJ")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.analyzers.power.basic import instantaneous_power

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class SwitchingEvent:
    """Information about a switching transition.

    Attributes:
        start_time: Start time of transition (seconds).
        end_time: End time of transition (seconds).
        duration: Transition duration (seconds).
        energy: Energy dissipated during transition (Joules).
        peak_power: Peak power during transition (Watts).
        event_type: "turn_on" or "turn_off".
    """

    start_time: float
    end_time: float
    duration: float
    energy: float
    peak_power: float
    event_type: Literal["turn_on", "turn_off"]


def switching_loss(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    v_threshold: float | None = None,
    i_threshold: float | None = None,
) -> dict[str, Any]:
    """Calculate switching losses for a power device.

    Analyzes voltage and current waveforms to find switching transitions
    and calculate turn-on and turn-off energy losses.

    Args:
        voltage: Drain-source (or collector-emitter) voltage trace.
        current: Drain (or collector) current trace.
        v_threshold: Voltage threshold for on/off detection.
                    If None, uses 10% of peak voltage.
        i_threshold: Current threshold for on/off detection.
                    If None, uses 10% of peak current.

    Returns:
        Dictionary with:
        - e_on: Turn-on energy per event (Joules)
        - e_off: Turn-off energy per event (Joules)
        - e_total: Total switching energy per cycle (Joules)
        - p_sw: Switching power at estimated frequency (Watts)
        - events: List of SwitchingEvent objects
        - n_turn_on: Number of turn-on events
        - n_turn_off: Number of turn-off events

    Example:
        >>> losses = switching_loss(v_ds, i_d)
        >>> print(f"E_on: {losses['e_on']*1e6:.2f} uJ")
        >>> print(f"E_off: {losses['e_off']*1e6:.2f} uJ")
        >>> print(f"Switching power @ 100kHz: {losses['p_sw']*100e3:.2f} W")

    References:
        Infineon Application Note AN-9010
    """
    # Calculate instantaneous power
    power = instantaneous_power(voltage, current)

    # Ensure i_data matches v_data length (handle mismatched array sizes)
    min_len = min(len(voltage.data), len(current.data))
    v_data = voltage.data[:min_len]
    i_data = current.data[:min_len]
    p_data = power.data[:min_len]
    sample_period = power.metadata.time_base

    # Auto-detect thresholds if not provided
    if v_threshold is None:
        v_threshold = 0.1 * float(np.max(np.abs(v_data)))
    if i_threshold is None:
        i_threshold = 0.1 * float(np.max(np.abs(i_data)))

    # Add hysteresis to prevent false transitions due to ringing (Schmitt trigger)
    # Use 20% hysteresis band around thresholds
    hysteresis_factor = 0.2
    v_threshold_high = v_threshold * (1 + hysteresis_factor)
    v_threshold_low = v_threshold * (1 - hysteresis_factor)
    i_threshold_high = i_threshold * (1 + hysteresis_factor)
    i_threshold_low = i_threshold * (1 - hysteresis_factor)

    # Find switching events
    events: list[SwitchingEvent] = []

    # Determine device state at each sample with hysteresis
    # ON: low voltage, high current
    # OFF: high voltage, low current
    # Use hysteresis to avoid rapid state changes due to noise/ringing
    device_state = np.zeros(min_len, dtype=int)  # 0=unknown, 1=on, 2=off
    current_state = 0  # Start in unknown state

    for i in range(min_len):
        v = v_data[i]
        i_val = i_data[i]

        # Determine next state based on current state and measurements
        if current_state == 1:  # Currently ON
            # Stay ON unless voltage goes high (with hysteresis)
            if v > v_threshold_high:
                current_state = 2  # Transition to OFF
        elif current_state == 2:  # Currently OFF
            # Stay OFF unless voltage goes low (with hysteresis)
            if v < v_threshold_low and i_val > i_threshold_low:
                current_state = 1  # Transition to ON
        else:  # Unknown state - determine initial state
            if v < v_threshold_low and i_val > i_threshold_high:
                current_state = 1  # ON
            elif v > v_threshold_high and i_val < i_threshold_low:
                current_state = 2  # OFF

        device_state[i] = current_state

    device_on = device_state == 1
    device_off = device_state == 2

    # Find transitions
    i = 0
    while i < len(device_on) - 1:
        # Look for turn-on: device was off, now turning on
        if device_off[i] and not device_off[i + 1]:
            # Find end of transition (device fully on)
            start_idx = i
            end_idx = start_idx + 1
            while end_idx < len(device_on) and not device_on[end_idx]:
                end_idx += 1

            if end_idx < len(device_on):
                # Calculate transition energy (scipy for stable API)
                from scipy.integrate import trapezoid

                transition_power = p_data[start_idx : end_idx + 1]
                e = float(trapezoid(transition_power, dx=sample_period))
                peak_p = float(np.max(transition_power))

                events.append(
                    SwitchingEvent(
                        start_time=start_idx * sample_period,
                        end_time=end_idx * sample_period,
                        duration=(end_idx - start_idx) * sample_period,
                        energy=e,
                        peak_power=peak_p,
                        event_type="turn_on",
                    )
                )
                i = end_idx
                continue

        # Look for turn-off: device was on, now turning off
        if device_on[i] and not device_on[i + 1]:
            start_idx = i
            end_idx = start_idx + 1
            while end_idx < len(device_off) and not device_off[end_idx]:
                end_idx += 1

            if end_idx < len(device_off):
                from scipy.integrate import trapezoid

                transition_power = p_data[start_idx : end_idx + 1]
                e = float(trapezoid(transition_power, dx=sample_period))
                peak_p = float(np.max(transition_power))

                events.append(
                    SwitchingEvent(
                        start_time=start_idx * sample_period,
                        end_time=end_idx * sample_period,
                        duration=(end_idx - start_idx) * sample_period,
                        energy=e,
                        peak_power=peak_p,
                        event_type="turn_off",
                    )
                )
                i = end_idx
                continue

        i += 1

    # Calculate average energies
    turn_on_events = [e for e in events if e.event_type == "turn_on"]
    turn_off_events = [e for e in events if e.event_type == "turn_off"]

    e_on = float(np.mean([e.energy for e in turn_on_events])) if turn_on_events else 0.0
    e_off = float(np.mean([e.energy for e in turn_off_events])) if turn_off_events else 0.0
    e_total = e_on + e_off

    # Estimate switching frequency from event spacing
    if len(events) >= 2:
        event_times = [e.start_time for e in events]
        avg_period = float(np.mean(np.diff(event_times))) * 2  # Full cycle
        f_sw = 1.0 / avg_period if avg_period > 0 else 0.0
    else:
        f_sw = 0.0

    return {
        "e_on": e_on,
        "e_off": e_off,
        "e_total": e_total,
        "f_sw": f_sw,
        "p_sw": e_total * f_sw,  # Switching power at this frequency
        "events": events,
        "n_turn_on": len(turn_on_events),
        "n_turn_off": len(turn_off_events),
    }


def switching_energy(
    voltage: WaveformTrace,
    current: WaveformTrace,
    start_time: float,
    end_time: float,
) -> float:
    """Calculate switching energy over a specific time window.

    E = integral(V(t) * I(t) dt) from start_time to end_time

    Args:
        voltage: Voltage trace.
        current: Current trace.
        start_time: Start of integration window (seconds).
        end_time: End of integration window (seconds).

    Returns:
        Switching energy in Joules.

    Example:
        >>> e = switching_energy(v_ds, i_d, start_time=1e-6, end_time=1.5e-6)
        >>> print(f"Switching energy: {e*1e9:.2f} nJ")
    """
    power = instantaneous_power(voltage, current)
    sample_period = power.metadata.time_base
    time_vector = np.arange(len(power.data)) * sample_period

    # Select time window
    mask = (time_vector >= start_time) & (time_vector <= end_time)
    window_power = power.data[mask]

    # Use scipy for stable API across NumPy versions
    from scipy.integrate import trapezoid

    return float(trapezoid(window_power, dx=sample_period))


def turn_on_loss(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    v_threshold: float | None = None,
    i_threshold: float | None = None,
) -> float:
    """Calculate average turn-on energy loss.

    Convenience function that returns just the turn-on energy.

    Args:
        voltage: Drain-source voltage trace.
        current: Drain current trace.
        v_threshold: Voltage threshold for on/off detection.
        i_threshold: Current threshold for on/off detection.

    Returns:
        Average turn-on energy in Joules.
    """
    result = switching_loss(voltage, current, v_threshold=v_threshold, i_threshold=i_threshold)
    return float(result["e_on"])


def turn_off_loss(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    v_threshold: float | None = None,
    i_threshold: float | None = None,
) -> float:
    """Calculate average turn-off energy loss.

    Args:
        voltage: Drain-source voltage trace.
        current: Drain current trace.
        v_threshold: Voltage threshold for on/off detection.
        i_threshold: Current threshold for on/off detection.

    Returns:
        Average turn-off energy in Joules.
    """
    result = switching_loss(voltage, current, v_threshold=v_threshold, i_threshold=i_threshold)
    return float(result["e_off"])


def total_switching_loss(
    voltage: WaveformTrace,
    current: WaveformTrace,
    frequency: float,
    *,
    v_threshold: float | None = None,
    i_threshold: float | None = None,
) -> float:
    """Calculate total switching power loss at given frequency.

    P_sw = (E_on + E_off) * f_sw

    Args:
        voltage: Drain-source voltage trace.
        current: Drain current trace.
        frequency: Switching frequency in Hz.
        v_threshold: Voltage threshold for on/off detection.
        i_threshold: Current threshold for on/off detection.

    Returns:
        Switching power loss in Watts.

    Example:
        >>> p_sw = total_switching_loss(v_ds, i_d, frequency=100e3)
        >>> print(f"Switching loss at 100kHz: {p_sw:.2f} W")
    """
    result = switching_loss(voltage, current, v_threshold=v_threshold, i_threshold=i_threshold)
    return float(result["e_total"]) * frequency


def switching_frequency(
    voltage: WaveformTrace,
    *,
    threshold: float | None = None,
) -> float:
    """Estimate switching frequency from voltage waveform.

    Args:
        voltage: Drain-source voltage trace.
        threshold: Voltage threshold for edge detection.

    Returns:
        Estimated switching frequency in Hz.

    Example:
        >>> f_sw = switching_frequency(v_ds)
        >>> print(f"Switching frequency: {f_sw/1e3:.1f} kHz")
    """
    data = voltage.data
    sample_rate = voltage.metadata.sample_rate

    if threshold is None:
        threshold = float((np.max(data) + np.min(data)) / 2)

    # Find rising edges
    below = data < threshold
    above = data >= threshold
    rising = np.where(below[:-1] & above[1:])[0]

    if len(rising) < 2:
        return 0.0

    # Calculate average period
    periods = np.diff(rising) / sample_rate
    avg_period = float(np.mean(periods))

    return 1.0 / avg_period if avg_period > 0 else 0.0


def switching_times(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    v_threshold: float | None = None,
    i_threshold: float | None = None,
) -> dict[str, float]:
    """Measure switching times (tr, tf, ton, toff).

    Args:
        voltage: Drain-source voltage trace.
        current: Drain current trace.
        v_threshold: Voltage threshold (10%-90% levels if None).
        i_threshold: Current threshold (10%-90% levels if None).

    Returns:
        Dictionary with:
        - tr: Rise time (10%-90%)
        - tf: Fall time (90%-10%)
        - t_on: Turn-on delay time
        - t_off: Turn-off delay time
    """
    # Ensure arrays match in length (handle mismatched array sizes)
    min_len = min(len(voltage.data), len(current.data))
    v_data = voltage.data[:min_len]
    i_data = current.data[:min_len]
    sample_period = voltage.metadata.time_base

    v_min, v_max = float(np.min(v_data)), float(np.max(v_data))
    i_min, i_max = float(np.min(i_data)), float(np.max(i_data))

    v_10 = v_min + 0.1 * (v_max - v_min)
    v_90 = v_min + 0.9 * (v_max - v_min)
    i_10 = i_min + 0.1 * (i_max - i_min)
    i_90 = i_min + 0.9 * (i_max - i_min)

    # Find voltage transitions
    def find_transition_time(
        data: NDArray[np.floating[Any]], low: float, high: float, rising: bool
    ) -> float:
        if rising:
            below_low = data < low
            start_idx_arr = np.where(below_low[:-1] & ~below_low[1:])[0]
            if len(start_idx_arr) == 0:
                return float(np.nan)
            start_idx = int(start_idx_arr[0])
            remaining = data[start_idx:]
            above_mask = remaining > high
            if not np.any(above_mask):
                return float(np.nan)
            end_offset = int(np.argmax(above_mask))
            return float(end_offset) * sample_period
        else:
            above_high = data > high
            start_idx_arr = np.where(above_high[:-1] & ~above_high[1:])[0]
            if len(start_idx_arr) == 0:
                return float(np.nan)
            start_idx = int(start_idx_arr[0])
            remaining = data[start_idx:]
            below_mask = remaining < low
            if not np.any(below_mask):
                return float(np.nan)
            end_offset = int(np.argmax(below_mask))
            return float(end_offset) * sample_period

    # Voltage fall time (turn-on)
    tf_v = find_transition_time(v_data, v_10, v_90, rising=False)
    # Voltage rise time (turn-off)
    tr_v = find_transition_time(v_data, v_10, v_90, rising=True)
    # Current rise time (turn-on)
    tr_i = find_transition_time(i_data, i_10, i_90, rising=True)
    # Current fall time (turn-off)
    tf_i = find_transition_time(i_data, i_10, i_90, rising=False)

    return {
        "tr": tr_v,  # Voltage rise time (turn-off)
        "tf": tf_v,  # Voltage fall time (turn-on)
        "tr_current": tr_i,  # Current rise time (turn-on)
        "tf_current": tf_i,  # Current fall time (turn-off)
    }


__all__ = [
    "SwitchingEvent",
    "switching_energy",
    "switching_frequency",
    "switching_loss",
    "switching_times",
    "total_switching_loss",
    "turn_off_loss",
    "turn_on_loss",
]
