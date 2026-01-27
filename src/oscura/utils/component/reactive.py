"""Reactive component extraction for Oscura.

This module provides capacitance and inductance measurement from
waveform data, including parasitic extraction.


Example:
    >>> from oscura.utils.component import measure_capacitance, measure_inductance
    >>> C = measure_capacitance(voltage_trace, current_trace)
    >>> L = measure_inductance(voltage_trace, current_trace)

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from oscura.core.exceptions import AnalysisError, InsufficientDataError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class CapacitanceMeasurement:
    """Result of capacitance measurement.

    Attributes:
        capacitance: Measured capacitance in Farads.
        esr: Equivalent Series Resistance in ohms.
        method: Measurement method used.
        confidence: Confidence in measurement (0-1).
        statistics: Additional measurement statistics.
    """

    capacitance: float
    esr: float = 0.0
    method: str = ""
    confidence: float = 1.0
    statistics: dict = field(default_factory=dict)  # type: ignore[type-arg]


@dataclass
class InductanceMeasurement:
    """Result of inductance measurement.

    Attributes:
        inductance: Measured inductance in Henrys.
        dcr: DC Resistance in ohms.
        q_factor: Quality factor at measurement frequency.
        method: Measurement method used.
        confidence: Confidence in measurement (0-1).
        statistics: Additional measurement statistics.
    """

    inductance: float
    dcr: float = 0.0
    q_factor: float | None = None
    method: str = ""
    confidence: float = 1.0
    statistics: dict = field(default_factory=dict)  # type: ignore[type-arg]


@dataclass
class ParasiticExtraction:
    """Result of parasitic parameter extraction.

    Attributes:
        capacitance: Parasitic capacitance in Farads.
        inductance: Parasitic inductance in Henrys.
        resistance: Parasitic resistance in ohms.
        model_type: Equivalent circuit model type.
        resonant_freq: Self-resonant frequency (if applicable).
        fit_quality: Quality of model fit (R-squared).
    """

    capacitance: float
    inductance: float
    resistance: float
    model_type: Literal["series_RLC", "parallel_RLC", "pi", "tee"]
    resonant_freq: float | None = None
    fit_quality: float = 0.0


def measure_capacitance(
    voltage_trace: WaveformTrace,
    current_trace: WaveformTrace | None = None,
    *,
    method: Literal["charge", "slope", "frequency"] = "charge",
    resistance: float | None = None,
) -> CapacitanceMeasurement:
    """Measure capacitance from voltage/current waveforms.

    Calculates capacitance using the relationship C = Q/V or C = I/(dV/dt).

    Args:
        voltage_trace: Voltage waveform across capacitor.
        current_trace: Current waveform through capacitor (optional for
            some methods).
        method: Measurement method:
            - "charge": C = integral(I*dt) / delta_V
            - "slope": C = I / (dV/dt)
            - "frequency": Extract from RC time constant
        resistance: Known resistance for frequency method.

    Returns:
        CapacitanceMeasurement with capacitance value.

    Raises:
        AnalysisError: If measurement conditions are not met.
        InsufficientDataError: If insufficient samples are provided.

    Example:
        >>> C = measure_capacitance(voltage, current)
        >>> print(f"C = {C.capacitance * 1e12:.1f} pF")

    References:
        COMP-002
    """
    voltage = voltage_trace.data.astype(np.float64)
    sample_rate = voltage_trace.metadata.sample_rate
    dt = 1.0 / sample_rate

    if len(voltage) < 10:
        raise InsufficientDataError(
            "Capacitance measurement requires at least 10 samples",
            required=10,
            available=len(voltage),
            analysis_type="capacitance",
        )

    if method == "charge" and current_trace is not None:
        return _measure_capacitance_charge(
            voltage, current_trace.data.astype(np.float64), dt, sample_rate
        )
    elif method == "slope" and current_trace is not None:
        return _measure_capacitance_slope(voltage, current_trace.data.astype(np.float64), dt)
    elif method == "frequency":
        return _measure_capacitance_frequency(voltage, sample_rate, resistance)
    else:
        raise AnalysisError(f"Method '{method}' requires current_trace or resistance parameter")


def _measure_capacitance_charge(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    dt: float,
    sample_rate: float,
) -> CapacitanceMeasurement:
    """Measure capacitance using charge integration method."""
    min_len = min(len(voltage), len(current))
    voltage, current = voltage[:min_len], current[:min_len]

    charge = np.cumsum(current) * dt
    delta_v = np.max(voltage) - np.min(voltage)

    if delta_v <= 1e-10:
        raise AnalysisError("Voltage change too small for capacitance measurement")

    delta_q = np.max(charge) - np.min(charge)
    capacitance = delta_q / delta_v
    esr = _estimate_esr(voltage, current, sample_rate)

    return CapacitanceMeasurement(
        capacitance=float(abs(capacitance)),
        esr=esr,
        method="charge_integration",
        confidence=0.9,
        statistics={"delta_v": delta_v, "delta_q": delta_q, "num_samples": min_len},
    )


def _measure_capacitance_slope(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    dt: float,
) -> CapacitanceMeasurement:
    """Measure capacitance using slope method."""
    min_len = min(len(voltage), len(current))
    voltage, current = voltage[:min_len], current[:min_len]

    dv_dt = np.diff(voltage) / dt
    significant_mask = np.abs(dv_dt) > np.max(np.abs(dv_dt)) * 0.1

    if np.sum(significant_mask) < 5:
        raise AnalysisError("Insufficient voltage slope for capacitance measurement")

    capacitance_values = current[:-1][significant_mask] / dv_dt[significant_mask]
    capacitance = float(np.median(np.abs(capacitance_values)))

    return CapacitanceMeasurement(
        capacitance=capacitance,
        method="slope",
        confidence=0.85,
        statistics={
            "num_valid_points": int(np.sum(significant_mask)),
            "capacitance_std": float(np.std(np.abs(capacitance_values))),
        },
    )


def _measure_capacitance_frequency(
    voltage: NDArray[np.float64],
    sample_rate: float,
    resistance: float | None,
) -> CapacitanceMeasurement:
    """Measure capacitance using time constant method."""
    if resistance is None:
        raise AnalysisError("Resistance value required for frequency method")

    tau = _extract_time_constant(voltage, sample_rate)
    capacitance = tau / resistance

    return CapacitanceMeasurement(
        capacitance=float(capacitance),
        method="time_constant",
        confidence=0.8,
        statistics={"time_constant": tau, "resistance": resistance},
    )


def measure_inductance(
    voltage_trace: WaveformTrace,
    current_trace: WaveformTrace | None = None,
    *,
    method: Literal["flux", "slope", "frequency"] = "slope",
    resistance: float | None = None,
) -> InductanceMeasurement:
    """Measure inductance from voltage/current waveforms.

    Calculates inductance using the relationship V = L * dI/dt.

    Args:
        voltage_trace: Voltage waveform across inductor.
        current_trace: Current waveform through inductor (optional for
            some methods).
        method: Measurement method:
            - "flux": L = integral(V*dt) / delta_I
            - "slope": L = V / (dI/dt)
            - "frequency": Extract from RL time constant
        resistance: Known resistance for frequency method.

    Returns:
        InductanceMeasurement with inductance value.

    Raises:
        AnalysisError: If measurement conditions are not met.
        InsufficientDataError: If insufficient samples are provided.

    Example:
        >>> L = measure_inductance(voltage, current)
        >>> print(f"L = {L.inductance * 1e6:.1f} uH")

    References:
        COMP-003
    """
    voltage = voltage_trace.data.astype(np.float64)
    sample_rate = voltage_trace.metadata.sample_rate
    dt = 1.0 / sample_rate

    if len(voltage) < 10:
        raise InsufficientDataError(
            "Inductance measurement requires at least 10 samples",
            required=10,
            available=len(voltage),
            analysis_type="inductance",
        )

    if method == "flux" and current_trace is not None:
        return _measure_inductance_flux(voltage, current_trace.data.astype(np.float64), dt)
    elif method == "slope" and current_trace is not None:
        return _measure_inductance_slope(voltage, current_trace.data.astype(np.float64), dt)
    elif method == "frequency":
        return _measure_inductance_frequency(voltage, sample_rate, resistance)
    else:
        raise AnalysisError(f"Method '{method}' requires current_trace or resistance parameter")


def _measure_inductance_flux(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    dt: float,
) -> InductanceMeasurement:
    """Measure inductance using flux integration method."""
    min_len = min(len(voltage), len(current))
    voltage, current = voltage[:min_len], current[:min_len]

    flux = np.cumsum(voltage) * dt
    delta_i = np.max(current) - np.min(current)

    if delta_i <= 1e-10:
        raise AnalysisError("Current change too small for inductance measurement")

    delta_flux = np.max(flux) - np.min(flux)
    inductance = delta_flux / delta_i
    dcr = _estimate_dcr(voltage, current)

    return InductanceMeasurement(
        inductance=float(abs(inductance)),
        dcr=dcr,
        method="flux_integration",
        confidence=0.9,
        statistics={"delta_i": delta_i, "delta_flux": delta_flux, "num_samples": min_len},
    )


def _measure_inductance_slope(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    dt: float,
) -> InductanceMeasurement:
    """Measure inductance using slope method."""
    min_len = min(len(voltage), len(current))
    voltage, current = voltage[:min_len], current[:min_len]

    di_dt = np.diff(current) / dt
    significant_mask = np.abs(di_dt) > np.max(np.abs(di_dt)) * 0.1

    if np.sum(significant_mask) < 5:
        raise AnalysisError("Insufficient current slope for inductance measurement")

    inductance_values = voltage[:-1][significant_mask] / di_dt[significant_mask]
    inductance = float(np.median(np.abs(inductance_values)))

    return InductanceMeasurement(
        inductance=inductance,
        method="slope",
        confidence=0.85,
        statistics={
            "num_valid_points": int(np.sum(significant_mask)),
            "inductance_std": float(np.std(np.abs(inductance_values))),
        },
    )


def _measure_inductance_frequency(
    voltage: NDArray[np.float64],
    sample_rate: float,
    resistance: float | None,
) -> InductanceMeasurement:
    """Measure inductance using time constant method."""
    if resistance is None:
        raise AnalysisError("Resistance value required for frequency method")

    tau = _extract_time_constant(voltage, sample_rate)
    inductance = tau * resistance

    return InductanceMeasurement(
        inductance=float(inductance),
        method="time_constant",
        confidence=0.8,
        statistics={"time_constant": tau, "resistance": resistance},
    )


def extract_parasitics(
    voltage_trace: WaveformTrace,
    current_trace: WaveformTrace,
    *,
    model: Literal["series_RLC", "parallel_RLC"] = "series_RLC",
    frequency_range: tuple[float, float] | None = None,
) -> ParasiticExtraction:
    """Extract parasitic R, L, C parameters from impedance measurement.

    Fits an equivalent circuit model to measured voltage/current data
    to extract parasitic component values.

    Args:
        voltage_trace: Voltage waveform.
        current_trace: Current waveform.
        model: Equivalent circuit model type.
        frequency_range: Frequency range for analysis (Hz).

    Returns:
        ParasiticExtraction with R, L, C values.

    Raises:
        AnalysisError: If measurement conditions are not met.
        InsufficientDataError: If insufficient samples are provided.

    Example:
        >>> params = extract_parasitics(voltage, current)
        >>> print(f"C = {params.capacitance*1e12:.1f}pF, L = {params.inductance*1e9:.1f}nH")

    References:
        COMP-004
    """
    voltage = voltage_trace.data.astype(np.float64)
    current = current_trace.data.astype(np.float64)
    sample_rate = voltage_trace.metadata.sample_rate

    min_len = min(len(voltage), len(current))
    voltage = voltage[:min_len]
    current = current[:min_len]

    if min_len < 100:
        raise InsufficientDataError(
            "Parasitic extraction requires at least 100 samples",
            required=100,
            available=min_len,
            analysis_type="parasitic_extraction",
        )

    # Compute impedance in frequency domain
    from scipy.fft import fft, fftfreq

    V_fft = fft(voltage)
    I_fft = fft(current)

    freqs = fftfreq(min_len, 1 / sample_rate)

    # Use only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    Z_fft = V_fft[pos_mask] / (I_fft[pos_mask] + 1e-20)

    # Apply frequency range filter
    if frequency_range is not None:
        freq_mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
        freqs = freqs[freq_mask]
        Z_fft = Z_fft[freq_mask]

    if len(freqs) < 10:
        raise AnalysisError("Insufficient frequency points for parasitic extraction")

    # Fit RLC model to impedance data
    if model == "series_RLC":
        R, L, C = _fit_series_rlc(freqs, Z_fft)
    else:  # parallel_RLC
        R, L, C = _fit_parallel_rlc(freqs, Z_fft)

    # Calculate resonant frequency
    f_res = 1 / (2 * np.pi * np.sqrt(L * C)) if L > 0 and C > 0 else None

    # Calculate fit quality
    Z_model = _calculate_rlc_impedance(freqs, R, L, C, model)
    ss_res = np.sum(np.abs(Z_fft - Z_model) ** 2)
    ss_tot = np.sum(np.abs(Z_fft - np.mean(Z_fft)) ** 2)
    r_squared = 1 - ss_res / (ss_tot + 1e-20)

    return ParasiticExtraction(
        capacitance=float(C),
        inductance=float(L),
        resistance=float(R),
        model_type=model,
        resonant_freq=f_res,
        fit_quality=float(max(0, r_squared)),
    )


def _extract_time_constant(data: NDArray[np.float64], sample_rate: float) -> float:
    """Extract time constant from step response."""
    # Normalize data
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min < 1e-10:
        return 1 / sample_rate  # Return minimum time constant

    normalized = (data - data_min) / (data_max - data_min)

    # Find 63.2% point (time constant)
    target = 0.632
    idx_raw = np.argmax(normalized >= target)
    idx = int(idx_raw)
    if idx == 0:
        idx = int(len(data) // 2)

    tau = float(idx) / sample_rate
    min_tau = 1 / sample_rate
    return float(tau if tau > min_tau else min_tau)


def _estimate_esr(
    voltage: NDArray[np.float64],
    current: NDArray[np.float64],
    sample_rate: float,
) -> float:
    """Estimate ESR from voltage/current phase relationship."""
    # Use correlation to estimate resistive component
    # ESR causes in-phase voltage drop
    correlation = np.correlate(voltage - np.mean(voltage), current - np.mean(current))
    if np.std(current) > 1e-10:
        esr = np.max(correlation) / (len(current) * np.var(current))
        return float(abs(esr))
    return 0.0


def _estimate_dcr(voltage: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    """Estimate DC resistance from steady-state V/I."""
    # Use last 10% of data as steady-state
    n = len(voltage)
    start = int(0.9 * n)
    v_ss = np.mean(voltage[start:])
    i_ss = np.mean(current[start:])
    if abs(i_ss) > 1e-10:
        return float(v_ss / i_ss)
    return 0.0


def _fit_series_rlc(
    freqs: NDArray[np.float64], Z: NDArray[np.complex128]
) -> tuple[float, float, float]:
    """Fit series RLC model to impedance data."""
    omega = 2 * np.pi * freqs

    # Initial estimates
    R_init = float(np.real(np.mean(Z)))
    # From imaginary part at low/high frequency
    L_init = float(np.abs(np.imag(Z[-1])) / omega[-1]) if len(omega) > 0 else 1e-9
    C_init = 1e-12

    def model(omega: NDArray[np.float64], R: float, L: float, C: float) -> NDArray[np.complex128]:
        return R + 1j * (omega * L - 1 / (omega * C + 1e-20))

    try:
        # Fit real and imaginary parts separately
        np.real(Z)
        np.imag(Z)

        # Simple optimization
        from scipy.optimize import minimize

        def objective(params: NDArray[np.float64]) -> np.floating[Any]:
            R, L, C = params
            Z_model = model(omega, R, L, C)
            return float(np.sum(np.abs(Z - Z_model) ** 2))  # type: ignore[return-value]

        result = minimize(
            objective,
            [R_init, L_init, C_init],
            bounds=[(1e-6, 1e6), (1e-15, 1e-3), (1e-15, 1e-3)],
            method="L-BFGS-B",
        )
        return tuple(result.x)
    except Exception:
        return (R_init, L_init, C_init)


def _fit_parallel_rlc(
    freqs: NDArray[np.float64], Z: NDArray[np.complex128]
) -> tuple[float, float, float]:
    """Fit parallel RLC model to impedance data."""
    # Convert to admittance
    Y = 1 / (Z + 1e-20)
    omega = 2 * np.pi * freqs

    # Initial estimates
    G_init = float(np.real(np.mean(Y)))
    R_init = 1 / G_init if G_init > 1e-10 else 1e3
    C_init = float(np.abs(np.imag(Y[-1])) / omega[-1]) if len(omega) > 0 else 1e-12
    L_init = 1e-9

    try:
        from scipy.optimize import minimize

        def objective(params: NDArray[np.float64]) -> np.floating[Any]:
            R, L, C = params
            Y_model = 1 / R + 1j * omega * C + 1 / (1j * omega * L + 1e-20)
            Z_model = 1 / (Y_model + 1e-20)
            return float(np.sum(np.abs(Z - Z_model) ** 2))  # type: ignore[return-value]

        result = minimize(
            objective,
            [R_init, L_init, C_init],
            bounds=[(1e-6, 1e9), (1e-15, 1e-3), (1e-15, 1e-3)],
            method="L-BFGS-B",
        )
        return tuple(result.x)
    except Exception:
        return (R_init, L_init, C_init)


def _calculate_rlc_impedance(
    freqs: NDArray[np.float64],
    R: float,
    L: float,
    C: float,
    model: str,
) -> NDArray[np.complex128]:
    """Calculate impedance of RLC circuit."""
    omega = 2 * np.pi * freqs

    if model == "series_RLC":
        Z: NDArray[np.complex128] = R + 1j * (omega * L - 1 / (omega * C + 1e-20))
        return Z
    else:  # parallel_RLC
        Y = 1 / R + 1j * omega * C + 1 / (1j * omega * L + 1e-20)
        return cast("NDArray[np.complex128]", 1 / (Y + 1e-20))
