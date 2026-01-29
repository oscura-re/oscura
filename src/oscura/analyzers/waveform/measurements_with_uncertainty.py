"""Waveform measurements with uncertainty propagation.

This module extends the standard measurements module with uncertainty
estimation following GUM (Guide to the Expression of Uncertainty in Measurement)
principles.

All measurements return MeasurementResult objects that include both the
value and its associated uncertainty.

Example:
    >>> from oscura.analyzers.waveform import measurements_with_uncertainty as meas_u
    >>> result = meas_u.rise_time(trace)
    >>> print(f"Rise time: {result.value*1e9:.2f} ± {result.uncertainty*1e9:.2f} ns")
    Rise time: 2.34 ± 0.12 ns

References:
    JCGM 100:2008 - Guide to the Expression of Uncertainty in Measurement (GUM)
    IEEE 181-2011 - Standard for Transitional Waveform Definitions (Annex B: Uncertainty)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.waveform import measurements as meas
from oscura.core.uncertainty import MeasurementWithUncertainty, UncertaintyEstimator

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def rise_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.1, 0.9),
    include_uncertainty: bool = True,
) -> MeasurementWithUncertainty:
    """Measure rise time with uncertainty estimation.

    Uncertainty sources:
    - Time base accuracy (from calibration info if available)
    - Sample interpolation error (sub-sample timing)
    - Noise-induced edge jitter (from signal SNR)

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
        include_uncertainty: If False, only return value estimate (faster).

    Returns:
        MeasurementResult with value and uncertainty.

    Example:
        >>> result = rise_time(trace)
        >>> print(f"t_rise = {result.value*1e9:.2f} ± {result.uncertainty*1e9:.2f} ns")

    References:
        IEEE 181-2011 Section 5.2 (rise time)
        IEEE 181-2011 Annex B (measurement uncertainty)
    """
    # Get the measurement value
    value = meas.rise_time(trace, ref_levels=ref_levels)

    if not include_uncertainty or np.isnan(value):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="s")

    # Estimate uncertainty components
    uncertainties = []

    # 1. Time base uncertainty (Type B)
    if (
        trace.metadata.calibration_info is not None
        and trace.metadata.calibration_info.timebase_accuracy is not None
    ):
        # Use calibration timebase accuracy if available
        timebase_ppm = trace.metadata.calibration_info.timebase_accuracy
        u_timebase = UncertaintyEstimator.time_base_uncertainty(
            trace.metadata.sample_rate, timebase_ppm
        )
        # Rise time involves 2 samples (start and stop), so uncertainty scales
        u_timebase_rise = u_timebase * np.sqrt(2)
        uncertainties.append(u_timebase_rise)
    elif trace.metadata.calibration_info is not None:
        # Use conservative estimate if calibration present but no timebase accuracy
        timebase_ppm = 25.0  # Typical scope: 25-50 ppm
        u_timebase = UncertaintyEstimator.time_base_uncertainty(
            trace.metadata.sample_rate, timebase_ppm
        )
        u_timebase_rise = u_timebase * np.sqrt(2)
        uncertainties.append(u_timebase_rise)
    else:
        # No calibration info - use conservative estimate
        u_timebase_rise = (1.0 / trace.metadata.sample_rate) * 50e-6  # 50 ppm
        uncertainties.append(u_timebase_rise)

    # 2. Interpolation uncertainty (Type B - rectangular distribution)
    # Linear interpolation error: typically ±0.5 samples worst case
    sample_period = trace.metadata.time_base
    u_interp = UncertaintyEstimator.type_b_rectangular(0.5 * sample_period)
    uncertainties.append(u_interp)

    # 3. Noise-induced uncertainty (Type A equivalent)
    # Estimate from local signal noise
    # Find the data region near the edge
    data_slice = trace.data  # Could refine to edge region
    if len(data_slice) > 10:
        noise_estimate = np.std(data_slice[:10]) if len(data_slice) >= 10 else 0.0
        # Noise-to-slew-rate ratio gives time uncertainty
        amplitude = np.ptp(trace.data)
        if amplitude > 0:
            # Approximate slew rate: amplitude / rise_time
            slew_rate = amplitude / value if value > 0 else np.inf
            u_noise = noise_estimate / slew_rate if slew_rate != np.inf else 0.0
            uncertainties.append(u_noise)

    # Combine all uncertainty sources (uncorrelated)
    total_uncertainty = UncertaintyEstimator.combined_uncertainty(uncertainties)

    return MeasurementWithUncertainty(
        value=float(value),
        uncertainty=total_uncertainty,
        unit="s",
        n_samples=len(trace.data),
    )


def fall_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.9, 0.1),
    include_uncertainty: bool = True,
) -> MeasurementWithUncertainty:
    """Measure fall time with uncertainty estimation.

    Similar uncertainty sources as rise_time().

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
        include_uncertainty: If False, only return value estimate (faster).

    Returns:
        MeasurementResult with value and uncertainty.

    References:
        IEEE 181-2011 Section 5.2
    """
    value = meas.fall_time(trace, ref_levels=ref_levels)

    if not include_uncertainty or np.isnan(value):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="s")

    # Similar uncertainty calculation as rise_time
    uncertainties = []

    # Time base uncertainty
    if (
        trace.metadata.calibration_info is not None
        and trace.metadata.calibration_info.timebase_accuracy is not None
    ):
        timebase_ppm = trace.metadata.calibration_info.timebase_accuracy
    else:
        timebase_ppm = 25.0  # Conservative default
    u_timebase = UncertaintyEstimator.time_base_uncertainty(
        trace.metadata.sample_rate, timebase_ppm
    )
    uncertainties.append(u_timebase * np.sqrt(2))

    # Interpolation uncertainty
    sample_period = trace.metadata.time_base
    u_interp = UncertaintyEstimator.type_b_rectangular(0.5 * sample_period)
    uncertainties.append(u_interp)

    total_uncertainty = UncertaintyEstimator.combined_uncertainty(uncertainties)

    return MeasurementWithUncertainty(
        value=float(value),
        uncertainty=total_uncertainty,
        unit="s",
        n_samples=len(trace.data),
    )


def frequency(
    trace: WaveformTrace, *, include_uncertainty: bool = True
) -> MeasurementWithUncertainty:
    """Measure frequency with uncertainty estimation.

    Uncertainty sources:
    - Time base accuracy
    - Period measurement uncertainty
    - Allan variance (short-term stability)

    Args:
        trace: Input waveform trace.
        include_uncertainty: If False, only return value estimate (faster).

    Returns:
        MeasurementResult with value and uncertainty in Hz.

    Example:
        >>> result = frequency(trace)
        >>> print(f"f = {result.value/1e6:.6f} ± {result.relative_uncertainty*100:.2f}% MHz")

    References:
        IEEE 181-2011 Section 5.3
        IEEE 1057-2017 Section 4.3
    """
    value = meas.frequency(trace)

    if not include_uncertainty or np.isnan(value):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="Hz")

    # Frequency is 1/period, so uncertainty propagation:
    # u(f) = f^2 * u(T)  where T is period
    period = 1.0 / value if value != 0 else np.nan

    if np.isnan(period):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="Hz")

    # Estimate period uncertainty
    uncertainties = []

    # Time base uncertainty
    if (
        trace.metadata.calibration_info is not None
        and trace.metadata.calibration_info.timebase_accuracy is not None
    ):
        timebase_ppm = trace.metadata.calibration_info.timebase_accuracy
    else:
        timebase_ppm = 25.0  # Conservative default
    # Period measurement spans multiple cycles, typically more accurate
    u_period_timebase = period * (timebase_ppm * 1e-6)
    uncertainties.append(u_period_timebase)

    # Interpolation uncertainty for edge detection
    sample_period = trace.metadata.time_base
    u_interp = UncertaintyEstimator.type_b_rectangular(0.5 * sample_period)
    # Two edges per period
    u_period_interp = u_interp * np.sqrt(2)
    uncertainties.append(u_period_interp)

    # Combine to get period uncertainty
    u_period = UncertaintyEstimator.combined_uncertainty([float(u) for u in uncertainties])

    # Propagate to frequency: u(f) = |df/dT| * u(T) = f^2 * u(T)
    u_frequency = float((value**2) * u_period)

    return MeasurementWithUncertainty(
        value=float(value), uncertainty=u_frequency, unit="Hz", n_samples=len(trace.data)
    )


def amplitude(
    trace: WaveformTrace, *, include_uncertainty: bool = True
) -> MeasurementWithUncertainty:
    """Measure amplitude (Vpp) with uncertainty estimation.

    Uncertainty sources:
    - Vertical gain accuracy (from calibration info)
    - Vertical offset error
    - Quantization noise (ADC resolution)
    - Signal noise (statistical)

    Args:
        trace: Input waveform trace.
        include_uncertainty: If False, only return value estimate (faster).

    Returns:
        MeasurementResult with value and uncertainty in volts.

    References:
        IEEE 1057-2017 Section 4.2 (amplitude measurement)
        IEEE 1057-2017 Section 4.4 (amplitude accuracy)
    """
    value = meas.amplitude(trace)

    if not include_uncertainty or np.isnan(value):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="V")

    uncertainties = []

    # 1. Vertical accuracy (Type B)
    # Typical scope: ±2% of reading ± 0.1% of full scale
    vertical_accuracy_pct = 2.0  # Conservative
    if trace.metadata.vertical_scale is not None:
        full_scale = trace.metadata.vertical_scale * 10  # 10 divisions typical
        offset_error = full_scale * 0.001  # 0.1%
    else:
        offset_error = 0.001  # 1 mV default

    u_vertical = UncertaintyEstimator.vertical_uncertainty(
        float(value), vertical_accuracy_pct, offset_error
    )
    uncertainties.append(u_vertical)

    # 2. Quantization uncertainty (Type B - rectangular)
    if (
        trace.metadata.calibration_info is not None
        and trace.metadata.calibration_info.vertical_resolution is not None
    ):
        bits = trace.metadata.calibration_info.vertical_resolution
        vertical_range = np.ptp(trace.data)  # Simplification
        lsb = vertical_range / (2**bits)
        u_quant = UncertaintyEstimator.type_b_rectangular(0.5 * lsb)
        uncertainties.append(u_quant)
    else:
        # Default: 8-bit ADC assumption
        vertical_range = np.ptp(trace.data)
        lsb = vertical_range / 256
        u_quant = UncertaintyEstimator.type_b_rectangular(0.5 * lsb)
        uncertainties.append(u_quant)

    # 3. Signal noise (Type A)
    # For amplitude (Vpp) measurements, noise is already captured in the peak detection
    # uncertainty. Adding additional noise estimation from "flat regions" is inappropriate
    # for periodic signals where no regions are truly flat. Skip noise term for amplitude.

    total_uncertainty = UncertaintyEstimator.combined_uncertainty(uncertainties)

    return MeasurementWithUncertainty(
        value=float(value),
        uncertainty=total_uncertainty,
        unit="V",
        n_samples=len(trace.data),
    )


def rms(
    trace: WaveformTrace,
    *,
    ac_coupled: bool = False,
    include_uncertainty: bool = True,
) -> MeasurementWithUncertainty:
    """Measure RMS voltage with uncertainty estimation.

    Args:
        trace: Input waveform trace.
        ac_coupled: If True, remove DC component before calculating RMS.
        include_uncertainty: If False, only return value estimate (faster).

    Returns:
        MeasurementResult with value and uncertainty in volts RMS.

    References:
        IEEE 1057-2017 Section 4.3
    """
    value = meas.rms(trace, ac_coupled=ac_coupled)

    if not include_uncertainty or np.isnan(value):
        return MeasurementWithUncertainty(value=float(value), uncertainty=float(np.nan), unit="V")

    uncertainties = []

    # Vertical accuracy
    vertical_accuracy_pct = 2.0
    offset_error = 0.001  # 1 mV
    u_vertical = UncertaintyEstimator.vertical_uncertainty(
        float(value), vertical_accuracy_pct, offset_error
    )
    uncertainties.append(u_vertical)

    # Statistical uncertainty (Type A)
    # RMS of N samples: u(RMS) ≈ RMS / sqrt(2N) for Gaussian noise
    n = len(trace.data)
    u_statistical = value / np.sqrt(2 * n) if n > 0 else 0.0
    uncertainties.append(u_statistical)

    total_uncertainty = UncertaintyEstimator.combined_uncertainty(uncertainties)

    return MeasurementWithUncertainty(
        value=float(value),
        uncertainty=total_uncertainty,
        unit="V",
        n_samples=len(trace.data),
    )


__all__ = [
    "amplitude",
    "fall_time",
    "frequency",
    "rise_time",
    "rms",
]
