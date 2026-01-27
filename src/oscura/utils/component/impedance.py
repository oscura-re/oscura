"""TDR impedance extraction for Oscura.

This module provides impedance extraction from Time Domain Reflectometry
(TDR) measurements, including impedance profiling and discontinuity analysis.


Example:
    >>> from oscura.utils.component import extract_impedance
    >>> z0, z_profile = extract_impedance(tdr_trace)

References:
    IPC-TM-650 2.5.5.7: Characteristic Impedance of Lines on PCBs
    IEEE 370-2020: Electrical Characterization of Interconnects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal as sp_signal

from oscura.core.exceptions import InsufficientDataError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class ImpedanceProfile:
    """Impedance profile from TDR measurement.

    Attributes:
        distance: Distance axis in meters.
        time: Time axis in seconds.
        impedance: Impedance values in ohms.
        z0_source: Source impedance (reference).
        velocity: Propagation velocity used (m/s).
        statistics: Additional statistics.
    """

    distance: NDArray[np.float64]
    time: NDArray[np.float64]
    impedance: NDArray[np.float64]
    z0_source: float
    velocity: float
    statistics: dict = field(default_factory=dict)  # type: ignore[type-arg]

    @property
    def mean_impedance(self) -> float:
        """Mean impedance value."""
        return float(np.mean(self.impedance))

    @property
    def max_impedance(self) -> float:
        """Maximum impedance value."""
        return float(np.max(self.impedance))

    @property
    def min_impedance(self) -> float:
        """Minimum impedance value."""
        return float(np.min(self.impedance))


@dataclass
class Discontinuity:
    """A detected impedance discontinuity.

    Attributes:
        position: Position in meters.
        time: Time position in seconds.
        impedance_before: Impedance before discontinuity.
        impedance_after: Impedance after discontinuity.
        magnitude: Magnitude of change (ohms).
        reflection_coeff: Reflection coefficient (rho).
        discontinuity_type: Type of discontinuity.
    """

    position: float
    time: float
    impedance_before: float
    impedance_after: float
    magnitude: float
    reflection_coeff: float
    discontinuity_type: Literal["capacitive", "inductive", "resistive", "unknown"]


def extract_impedance(
    trace: WaveformTrace,
    *,
    z0_source: float = 50.0,
    velocity: float | None = None,
    velocity_factor: float = 0.66,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[float, ImpedanceProfile]:
    """Extract impedance profile from TDR waveform.

    Calculates the impedance profile from a TDR reflection measurement
    using the relationship between incident and reflected waves.

    Args:
        trace: TDR reflection waveform.
        z0_source: Source/reference impedance (default 50 ohms).
        velocity: Propagation velocity in m/s. If None, calculated from
            velocity_factor.
        velocity_factor: Fraction of speed of light (default 0.66 for FR4).
        start_time: Start time for analysis window (seconds).
        end_time: End time for analysis window (seconds).

    Returns:
        Tuple of (characteristic_impedance, impedance_profile).

    Raises:
        InsufficientDataError: If trace has fewer than 10 samples.

    Example:
        >>> z0, profile = extract_impedance(tdr_trace, z0_source=50)
        >>> print(f"Z0 = {z0:.1f} ohms")

    References:
        IPC-TM-650 2.5.5.7
    """
    data, sample_rate = _prepare_tdr_data(trace)
    velocity_val = _compute_velocity(velocity, velocity_factor)
    time_axis, distance_axis = _create_axes(data, sample_rate, velocity_val)
    start_idx, end_idx = _compute_analysis_window(len(data), sample_rate, start_time, end_time)
    impedance = _compute_impedance_profile(data, z0_source)
    z0, stats = _extract_impedance_statistics(impedance, start_idx, end_idx, distance_axis)
    profile = ImpedanceProfile(
        distance=distance_axis,
        time=time_axis,
        impedance=impedance,
        z0_source=z0_source,
        velocity=velocity_val,
        statistics=stats,
    )
    return z0, profile


def _prepare_tdr_data(trace: WaveformTrace) -> tuple[NDArray[np.float64], float]:
    """Prepare TDR data for analysis.

    Args:
        trace: TDR reflection waveform.

    Returns:
        Tuple of (data array, sample rate).

    Raises:
        InsufficientDataError: If trace has fewer than 10 samples.
    """
    data = trace.data.astype(np.float64)
    if len(data) < 10:
        raise InsufficientDataError(
            "TDR analysis requires at least 10 samples",
            required=10,
            available=len(data),
            analysis_type="tdr_impedance",
        )
    return data, trace.metadata.sample_rate


def _compute_velocity(velocity: float | None, velocity_factor: float) -> float:
    """Compute propagation velocity.

    Args:
        velocity: Explicit velocity or None.
        velocity_factor: Fraction of speed of light.

    Returns:
        Propagation velocity in m/s.
    """
    if velocity is not None:
        return velocity
    c = 299792458.0  # Speed of light in m/s
    return c * velocity_factor


def _create_axes(
    data: NDArray[np.float64], sample_rate: float, velocity: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create time and distance axes for TDR.

    Args:
        data: TDR data array.
        sample_rate: Sample rate in Hz.
        velocity: Propagation velocity in m/s.

    Returns:
        Tuple of (time_axis, distance_axis).
    """
    dt = 1.0 / sample_rate
    time_axis = np.arange(len(data)) * dt
    distance_axis = velocity * time_axis / 2.0  # Round trip
    return time_axis, distance_axis


def _compute_analysis_window(
    data_len: int, sample_rate: float, start_time: float | None, end_time: float | None
) -> tuple[int, int]:
    """Compute analysis window indices.

    Args:
        data_len: Length of data array.
        sample_rate: Sample rate in Hz.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Tuple of (start_idx, end_idx).
    """
    start_idx = 0 if start_time is None else int(start_time * sample_rate)
    end_idx = data_len if end_time is None else int(end_time * sample_rate)
    start_idx = max(0, min(start_idx, data_len - 1))
    end_idx = max(start_idx + 1, min(end_idx, data_len))
    return start_idx, end_idx


def _compute_impedance_profile(data: NDArray[np.float64], z0_source: float) -> NDArray[np.float64]:
    """Compute impedance profile from TDR data.

    Args:
        data: TDR voltage data.
        z0_source: Source impedance.

    Returns:
        Impedance profile array.
    """
    incident_level = _find_incident_level(data)
    rho = (data / incident_level) - 1.0 if incident_level > 0 else data - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        impedance = z0_source * (1 + rho) / (1 - rho)
        return np.clip(impedance, 1.0, 10000.0)


def _extract_impedance_statistics(
    impedance: NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    distance_axis: NDArray[np.float64],
) -> tuple[float, dict[str, float]]:
    """Extract impedance statistics from stable region.

    Args:
        impedance: Impedance profile.
        start_idx: Start index of analysis window.
        end_idx: End index of analysis window.
        distance_axis: Distance array.

    Returns:
        Tuple of (characteristic impedance, statistics dict).
    """
    stable_region = impedance[start_idx:end_idx]
    z0 = float(np.median(stable_region))
    stats = {
        "z0_measured": z0,
        "z0_std": float(np.std(stable_region)),
        "z0_min": float(np.min(stable_region)),
        "z0_max": float(np.max(stable_region)),
        "analysis_start_m": float(distance_axis[start_idx]),
        "analysis_end_m": float(distance_axis[end_idx - 1]),
    }
    return z0, stats


def impedance_profile(
    trace: WaveformTrace,
    *,
    z0_source: float = 50.0,
    velocity_factor: float = 0.66,
    smooth_window: int = 0,
) -> ImpedanceProfile:
    """Get impedance profile from TDR waveform.

    Convenience function that returns just the impedance profile.

    Args:
        trace: TDR reflection waveform.
        z0_source: Source/reference impedance.
        velocity_factor: Fraction of speed of light.
        smooth_window: Smoothing window size (0 = no smoothing).

    Returns:
        ImpedanceProfile object.
    """
    _, profile = extract_impedance(
        trace,
        z0_source=z0_source,
        velocity_factor=velocity_factor,
    )

    if smooth_window > 0:
        # Apply smoothing
        kernel = np.ones(smooth_window) / smooth_window
        profile.impedance = np.convolve(profile.impedance, kernel, mode="same")

    return profile


def discontinuity_analysis(
    trace: WaveformTrace,
    *,
    z0_source: float = 50.0,
    velocity_factor: float = 0.66,
    threshold: float = 5.0,
    min_separation: float = 1e-12,
) -> list[Discontinuity]:
    """Analyze impedance discontinuities in TDR waveform.

    Detects and characterizes impedance discontinuities along a
    transmission line from TDR measurements.

    Args:
        trace: TDR reflection waveform.
        z0_source: Source/reference impedance.
        velocity_factor: Fraction of speed of light.
        threshold: Minimum impedance change to detect (ohms).
        min_separation: Minimum time between discontinuities (seconds).

    Returns:
        List of detected Discontinuity objects.

    Example:
        >>> disconts = discontinuity_analysis(tdr_trace)
        >>> for d in disconts:
        ...     print(f"{d.position*1000:.1f}mm: {d.magnitude:.1f} ohms")
    """
    # Get impedance profile
    _, profile = extract_impedance(
        trace,
        z0_source=z0_source,
        velocity_factor=velocity_factor,
    )

    impedance = profile.impedance
    time_axis = profile.time
    distance_axis = profile.distance

    # Find discontinuities using derivative
    derivative = np.abs(np.diff(impedance))

    # Smooth derivative
    if len(derivative) > 5:
        kernel = np.ones(5) / 5
        derivative = np.convolve(derivative, kernel, mode="same")

    # Find peaks in derivative (discontinuities)
    sample_rate = trace.metadata.sample_rate
    min_samples = int(min_separation * sample_rate)

    peaks, _properties = sp_signal.find_peaks(
        derivative,
        height=threshold,
        distance=max(1, min_samples),
    )

    # Analyze each discontinuity
    discontinuities = []
    for peak_idx in peaks:
        if peak_idx < 1 or peak_idx >= len(impedance) - 1:
            continue

        z_before = float(np.mean(impedance[max(0, peak_idx - 5) : peak_idx]))
        z_after = float(np.mean(impedance[peak_idx + 1 : min(len(impedance), peak_idx + 6)]))

        magnitude = z_after - z_before
        position = float(distance_axis[peak_idx])
        time_pos = float(time_axis[peak_idx])

        # Calculate reflection coefficient
        rho = (z_after - z_before) / (z_after + z_before) if z_before + z_after > 0 else 0.0

        # Determine discontinuity type
        if magnitude > 0:
            # Increasing impedance
            if abs(magnitude) > 20:
                disc_type: Literal["capacitive", "inductive", "resistive", "unknown"] = "inductive"
            else:
                disc_type = "resistive"
        # Decreasing impedance
        elif abs(magnitude) > 20:
            disc_type = "capacitive"
        else:
            disc_type = "resistive"

        discontinuities.append(
            Discontinuity(
                position=position,
                time=time_pos,
                impedance_before=z_before,
                impedance_after=z_after,
                magnitude=magnitude,
                reflection_coeff=float(rho),
                discontinuity_type=disc_type,
            )
        )

    return discontinuities


def _find_incident_level(data: NDArray[np.float64]) -> float:
    """Find the incident step level in TDR data.

    Looks for the stable level after the initial edge and before
    any reflections return.

    Args:
        data: TDR waveform data array.

    Returns:
        Median voltage level in the incident region.
    """
    if len(data) < 10:
        return float(np.max(data))

    # Look at first 10-20% of data for incident level
    search_end = len(data) // 5
    search_start = len(data) // 20

    if search_end <= search_start:
        return float(np.max(data[:search_end]))

    # Find stable region using variance
    stable_data = data[search_start:search_end]
    return float(np.median(stable_data))
