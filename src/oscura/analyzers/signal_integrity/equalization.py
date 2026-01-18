"""Equalization algorithms for signal integrity.

This module provides FFE, DFE, and CTLE equalization to
compensate for channel loss and ISI.


Example:
    >>> from oscura.analyzers.signal_integrity.equalization import ffe_equalize
    >>> equalized = ffe_equalize(trace, taps=[-0.1, 1.0, -0.1])

References:
    IEEE 802.3: Ethernet PHY Equalization Requirements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class FFEResult:
    """Result of FFE equalization.

    Attributes:
        equalized_data: Equalized waveform data.
        taps: FFE tap coefficients used.
        n_precursor: Number of precursor taps.
        n_postcursor: Number of postcursor taps.
        mse: Mean squared error (if optimized).
    """

    equalized_data: NDArray[np.float64]
    taps: NDArray[np.float64]
    n_precursor: int
    n_postcursor: int
    mse: float | None = None


@dataclass
class DFEResult:
    """Result of DFE equalization.

    Attributes:
        equalized_data: Equalized waveform data.
        taps: DFE tap coefficients.
        decisions: Decoded bit decisions.
        n_taps: Number of DFE taps.
        error_count: Number of decision errors (if reference known).
    """

    equalized_data: NDArray[np.float64]
    taps: NDArray[np.float64]
    decisions: NDArray[np.int_] | None
    n_taps: int
    error_count: int | None = None


@dataclass
class CTLEResult:
    """Result of CTLE equalization.

    Attributes:
        equalized_data: Equalized waveform data.
        dc_gain: DC gain in dB.
        ac_gain: AC (peaking) gain in dB.
        pole_frequency: Pole frequency in Hz.
        zero_frequency: Zero frequency in Hz.
        boost: High-frequency boost in dB.
    """

    equalized_data: NDArray[np.float64]
    dc_gain: float
    ac_gain: float
    pole_frequency: float
    zero_frequency: float | None
    boost: float


def ffe_equalize(
    trace: WaveformTrace,
    taps: list[float] | NDArray[np.float64],
    *,
    samples_per_symbol: int | None = None,
) -> FFEResult:
    """Apply Feed-Forward Equalization to waveform.

    FFE uses a linear FIR filter to compensate for channel ISI.
    The main cursor (largest tap) should be 1.0 for unity gain.

    Args:
        trace: Input waveform trace.
        taps: FFE tap coefficients (main cursor should be 1.0).
        samples_per_symbol: Samples per UI (auto-detected if None).

    Returns:
        FFEResult with equalized data.

    Example:
        >>> result = ffe_equalize(trace, taps=[-0.1, 1.0, -0.1])
        >>> # 3-tap equalizer: 1 precursor, 1 main, 1 postcursor

    References:
        IEEE 802.3 Clause 93
    """
    taps = np.array(taps, dtype=np.float64)

    # Find main cursor position
    main_idx = int(np.argmax(np.abs(taps)))
    n_precursor = main_idx
    n_postcursor = len(taps) - main_idx - 1

    # Apply FIR filter
    data = trace.data
    equalized = np.convolve(data, taps, mode="same")

    return FFEResult(
        equalized_data=equalized,
        taps=taps,
        n_precursor=n_precursor,
        n_postcursor=n_postcursor,
    )


def optimize_ffe(
    trace: WaveformTrace,
    n_taps: int = 5,
    *,
    n_precursor: int = 1,
    samples_per_symbol: int | None = None,
    target: NDArray[np.float64] | None = None,
) -> FFEResult:
    """Find optimal FFE tap coefficients.

    Uses least-squares optimization to find taps that minimize
    ISI and maximize eye opening.

    Args:
        trace: Input waveform trace.
        n_taps: Total number of FFE taps.
        n_precursor: Number of precursor taps.
        samples_per_symbol: Samples per UI.
        target: Target (ideal) waveform for optimization.

    Returns:
        FFEResult with optimized taps.

    Example:
        >>> result = optimize_ffe(trace, n_taps=5, n_precursor=1)
        >>> print(f"Optimal taps: {result.taps}")
    """
    data = trace.data
    len(data)

    if target is None:
        # Create target from sliced data (simplified)
        # Use a decision slicer approach
        threshold = np.median(data)
        target = np.where(data > threshold, 1.0, -1.0)

    def objective(taps):  # type: ignore[no-untyped-def]
        """Minimize MSE between equalized and target."""
        equalized = np.convolve(data, taps, mode="same")
        mse = np.mean((equalized - target) ** 2)
        return mse

    # Initial guess: main cursor at 1.0, others small
    n_postcursor = n_taps - n_precursor - 1
    x0 = np.zeros(n_taps)
    x0[n_precursor] = 1.0

    # Constraints: limit tap magnitude
    bounds = [(-2.0, 2.0)] * n_taps
    bounds[n_precursor] = (0.5, 1.5)  # Main cursor near 1.0

    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100},
    )

    optimal_taps = result.x

    # Normalize so main cursor is 1.0
    main_val = optimal_taps[n_precursor]
    if abs(main_val) > 1e-6:
        optimal_taps = optimal_taps / main_val

    equalized = np.convolve(data, optimal_taps, mode="same")
    mse = float(np.mean((equalized - target) ** 2))

    return FFEResult(
        equalized_data=equalized,
        taps=optimal_taps,
        n_precursor=n_precursor,
        n_postcursor=n_postcursor,
        mse=mse,
    )


def dfe_equalize(
    trace: WaveformTrace,
    taps: list[float] | NDArray[np.float64],
    *,
    threshold: float | None = None,
    samples_per_symbol: int = 1,
) -> DFEResult:
    """Apply Decision Feedback Equalization.

    DFE cancels post-cursor ISI using feedback from previous
    bit decisions. Unlike FFE, DFE does not amplify noise.

    Args:
        trace: Input waveform trace.
        taps: DFE tap coefficients for post-cursor cancellation.
        threshold: Decision threshold (auto-detected if None).
        samples_per_symbol: Samples per UI (default 1 for symbol-rate).

    Returns:
        DFEResult with equalized data and decisions.

    Example:
        >>> result = dfe_equalize(trace, taps=[0.2, 0.1])
        >>> # 2-tap DFE canceling h1 and h2

    References:
        IEEE 802.3 Clause 93
    """
    taps = np.array(taps, dtype=np.float64)
    n_taps = len(taps)
    data = trace.data
    n = len(data)

    # Auto-detect threshold
    if threshold is None:
        threshold = float(np.median(data))

    # Output arrays
    equalized = np.zeros(n, dtype=np.float64)
    decisions = np.zeros(n // samples_per_symbol, dtype=np.int_)

    # Previous decisions buffer (for feedback)
    prev_decisions = np.zeros(n_taps, dtype=np.float64)

    # Process symbol-by-symbol
    decision_idx = 0

    for i in range(0, n, samples_per_symbol):
        # Get input sample
        input_val = data[i]

        # Subtract DFE feedback
        dfe_correction = np.dot(taps, prev_decisions)
        corrected = input_val - dfe_correction

        # Make decision
        decision = 1.0 if corrected > threshold else -1.0

        # Store
        equalized[i : i + samples_per_symbol] = corrected
        if decision_idx < len(decisions):
            decisions[decision_idx] = int((decision + 1) / 2)  # 0 or 1
            decision_idx += 1

        # Shift feedback register
        prev_decisions = np.roll(prev_decisions, 1)
        prev_decisions[0] = decision

    return DFEResult(
        equalized_data=equalized,
        taps=taps,
        decisions=decisions[:decision_idx],
        n_taps=n_taps,
    )


def ctle_equalize(
    trace: WaveformTrace,
    dc_gain: float = 0.0,
    ac_gain: float = 6.0,
    pole_frequency: float = 5e9,
    *,
    zero_frequency: float | None = None,
) -> CTLEResult:
    """Apply Continuous Time Linear Equalization.

    CTLE provides high-frequency boost to compensate for
    channel loss. It uses an analog-style transfer function.

    Args:
        trace: Input waveform trace.
        dc_gain: DC gain in dB.
        ac_gain: AC (peaking) gain in dB.
        pole_frequency: Pole frequency in Hz.
        zero_frequency: Zero frequency in Hz (computed if None).

    Returns:
        CTLEResult with equalized data.

    Example:
        >>> result = ctle_equalize(trace, ac_gain=6, pole_frequency=5e9)
        >>> # 6 dB of high-frequency boost

    References:
        IEEE 802.3 Clause 93
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate
    len(data)

    # Convert gains from dB to linear
    dc_linear = 10 ** (dc_gain / 20)
    ac_linear = 10 ** (ac_gain / 20)

    # Calculate zero frequency to achieve desired boost
    if zero_frequency is None:
        # Zero at lower frequency than pole to create peaking
        zero_frequency = pole_frequency / (ac_linear / dc_linear)

    # Compute boost
    boost = ac_gain - dc_gain

    # Create CTLE transfer function
    # H(s) = (1 + s/wz) / (1 + s/wp) * gain
    wz = 2 * np.pi * zero_frequency
    wp = 2 * np.pi * pole_frequency

    # Convert to digital filter using bilinear transform
    b_analog = [1 / wz, 1]  # numerator coefficients
    a_analog = [1 / wp, 1]  # denominator coefficients

    # Bilinear transform
    b_digital, a_digital = scipy_signal.bilinear(b_analog, a_analog, fs=sample_rate)

    # Scale for DC gain
    b_digital = b_digital * dc_linear

    # Apply filter
    equalized = scipy_signal.lfilter(b_digital, a_digital, data)

    return CTLEResult(
        equalized_data=equalized.astype(np.float64),
        dc_gain=dc_gain,
        ac_gain=ac_gain,
        pole_frequency=pole_frequency,
        zero_frequency=zero_frequency,
        boost=boost,
    )


__all__ = [
    "CTLEResult",
    "DFEResult",
    "FFEResult",
    "ctle_equalize",
    "dfe_equalize",
    "ffe_equalize",
    "optimize_ffe",
]
