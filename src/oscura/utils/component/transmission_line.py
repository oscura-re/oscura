"""Transmission line analysis for Oscura.

This module provides transmission line characterization including
characteristic impedance, propagation delay, and velocity factor.


Example:
    >>> from oscura.utils.component import transmission_line_analysis
    >>> result = transmission_line_analysis(tdr_trace)

References:
    IPC-TM-650 2.5.5.7: Characteristic Impedance of Lines on PCBs
    IEEE 370-2020: Electrical Characterization of Interconnects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as sp_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class TransmissionLineResult:
    """Transmission line characterization result.

    Attributes:
        z0: Characteristic impedance in ohms.
        propagation_delay: Propagation delay in seconds.
        velocity_factor: Velocity factor (0-1).
        velocity: Propagation velocity in m/s.
        length: Estimated line length in meters.
        loss: Estimated loss in dB (if available).
        return_loss: Return loss in dB (if available).
        insertion_loss: Insertion loss in dB (if available).
        statistics: Additional measurements.
    """

    z0: float
    propagation_delay: float
    velocity_factor: float
    velocity: float
    length: float
    loss: float | None = None
    return_loss: float | None = None
    insertion_loss: float | None = None
    statistics: dict = field(default_factory=dict)  # type: ignore[type-arg]


def transmission_line_analysis(
    trace: WaveformTrace,
    *,
    z0_source: float = 50.0,
    line_length: float | None = None,
    dielectric_constant: float | None = None,
) -> TransmissionLineResult:
    """Analyze transmission line from TDR measurement.

    Characterizes a transmission line by extracting characteristic
    impedance, propagation delay, and loss parameters.

    Args:
        trace: TDR reflection waveform.
        z0_source: Source impedance (default 50 ohms).
        line_length: Known line length in meters (improves accuracy).
        dielectric_constant: Known dielectric constant (improves velocity).

    Returns:
        TransmissionLineResult with line parameters.

    Example:
        >>> result = transmission_line_analysis(tdr_trace, line_length=0.1)
        >>> print(f"Z0 = {result.z0:.1f} ohms, delay = {result.propagation_delay*1e9:.2f} ns")
    """
    from oscura.utils.component.impedance import extract_impedance

    # Speed of light
    c = 299792458.0

    # Extract impedance profile
    z0, profile = extract_impedance(trace, z0_source=z0_source)

    # Estimate propagation delay from reflection
    data = trace.data.astype(np.float64)
    sample_rate = trace.metadata.sample_rate

    # Find incident edge and first reflection
    incident_time, reflection_time = _find_reflection_times(data, sample_rate)
    round_trip_time = reflection_time - incident_time

    # Propagation delay is half the round-trip time
    propagation_delay = round_trip_time / 2

    # Calculate velocity
    if line_length is not None:
        # Use known length
        velocity = line_length / propagation_delay if propagation_delay > 0 else c * 0.66
        velocity_factor = velocity / c
    elif dielectric_constant is not None:
        # Calculate from dielectric constant
        velocity_factor = 1 / np.sqrt(dielectric_constant)
        velocity = c * velocity_factor
        line_length = velocity * propagation_delay
    else:
        # Estimate from typical FR4
        velocity_factor = 0.66
        velocity = c * velocity_factor
        line_length = velocity * propagation_delay

    # Estimate loss from reflection amplitude decay
    loss = _estimate_loss(data, sample_rate, propagation_delay)

    # Estimate return loss
    return_loss = _calculate_return_loss(z0, z0_source)

    return TransmissionLineResult(
        z0=z0,
        propagation_delay=propagation_delay,
        velocity_factor=velocity_factor,
        velocity=velocity,
        length=line_length,
        loss=loss,
        return_loss=return_loss,
        statistics={
            "incident_time": incident_time,
            "reflection_time": reflection_time,
            "round_trip_time": round_trip_time,
            "z0_std": profile.statistics.get("z0_std", 0),
        },
    )


def characteristic_impedance(
    trace: WaveformTrace,
    *,
    z0_source: float = 50.0,
    start_time: float | None = None,
    end_time: float | None = None,
) -> float:
    """Extract characteristic impedance from TDR measurement.

    Calculates the characteristic impedance from a stable region
    of the TDR waveform.

    Args:
        trace: TDR reflection waveform.
        z0_source: Source impedance.
        start_time: Start of analysis window (seconds).
        end_time: End of analysis window (seconds).

    Returns:
        Characteristic impedance in ohms.

    Example:
        >>> z0 = characteristic_impedance(tdr_trace)
        >>> print(f"Z0 = {z0:.1f} ohms")
    """
    from oscura.utils.component.impedance import extract_impedance

    z0, _ = extract_impedance(
        trace,
        z0_source=z0_source,
        start_time=start_time,
        end_time=end_time,
    )
    return z0


def propagation_delay(
    trace: WaveformTrace,
    *,
    threshold: float = 0.5,
) -> float:
    """Measure propagation delay from TDR waveform.

    Calculates the one-way propagation delay from the incident edge
    to the first reflection.

    Args:
        trace: TDR reflection waveform.
        threshold: Threshold level for edge detection (normalized).

    Returns:
        Propagation delay in seconds.

    Example:
        >>> delay = propagation_delay(tdr_trace)
        >>> print(f"Delay = {delay * 1e9:.2f} ns")
    """
    data = trace.data.astype(np.float64)
    sample_rate = trace.metadata.sample_rate

    incident_time, reflection_time = _find_reflection_times(data, sample_rate, threshold)

    return (reflection_time - incident_time) / 2


def velocity_factor(
    trace: WaveformTrace,
    line_length: float,
) -> float:
    """Calculate velocity factor from TDR and known length.

    Determines the propagation velocity as a fraction of the
    speed of light.

    Args:
        trace: TDR reflection waveform.
        line_length: Known line length in meters.

    Returns:
        Velocity factor (0 to 1).

    Example:
        >>> vf = velocity_factor(tdr_trace, line_length=0.1)
        >>> print(f"Velocity factor = {vf:.2f}")
    """
    c = 299792458.0
    delay = propagation_delay(trace)

    if delay > 0:
        velocity = line_length / delay
        return float(min(1.0, velocity / c))
    return 0.66  # Default for FR4


def _find_reflection_times(
    data: NDArray[np.float64],
    sample_rate: float,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Find incident and reflection edge times."""
    # Normalize data
    data_norm = (data - np.min(data)) / (np.ptp(data) + 1e-10)

    # Calculate derivative to find edges
    derivative = np.abs(np.diff(data_norm))

    # Find peaks in derivative
    peaks, _ = sp_signal.find_peaks(derivative, height=0.1 * np.max(derivative))

    if len(peaks) < 2:
        # Fallback: use threshold crossing
        above_thresh = data_norm > threshold
        crossings = np.where(np.diff(above_thresh.astype(int)))[0]

        if len(crossings) >= 2:
            incident_idx = crossings[0]
            reflection_idx = crossings[1]
        else:
            # Can't find edges
            incident_idx = 0
            reflection_idx = len(data) // 2
    else:
        incident_idx = peaks[0]
        reflection_idx = peaks[1]

    incident_time = incident_idx / sample_rate
    reflection_time = reflection_idx / sample_rate

    return incident_time, reflection_time


def _estimate_loss(
    data: NDArray[np.float64],
    sample_rate: float,
    delay: float,
) -> float | None:
    """Estimate transmission line loss from reflection amplitudes."""
    # Find incident and reflected amplitudes
    incident_region = data[: int(delay * sample_rate * 0.5)]
    if len(incident_region) == 0:
        return None

    incident_amp = np.max(incident_region) - np.min(incident_region)

    # Find reflected amplitude (after round-trip)
    reflection_start = int(delay * 2 * sample_rate)
    if reflection_start >= len(data):
        return None

    reflected_region = data[reflection_start:]
    if len(reflected_region) == 0:
        return None

    reflected_amp = np.max(reflected_region) - np.min(reflected_region)

    if incident_amp > 0:
        # Loss in dB = 20 * log10(reflected / incident)
        # But this is round-trip, so divide by 2 for one-way
        ratio = reflected_amp / incident_amp
        if ratio > 0 and ratio < 1:
            return float(-20 * np.log10(ratio) / 2)

    return None


def _calculate_return_loss(z0: float, z0_source: float) -> float:
    """Calculate return loss from impedance mismatch."""
    if z0 + z0_source > 0:
        rho = abs((z0 - z0_source) / (z0 + z0_source))
        if rho > 0:
            return float(-20 * np.log10(rho))
        return float("inf")  # Perfect match
    return 0.0
