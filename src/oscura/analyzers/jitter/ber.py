"""BER-related jitter analysis functions.

This module provides total jitter at BER calculations, bathtub curve
generation, and eye opening measurements using the dual-Dirac model.


Example:
    >>> from oscura.analyzers.jitter.ber import tj_at_ber, bathtub_curve
    >>> tj = tj_at_ber(rj_rms=1e-12, dj_pp=10e-12, ber=1e-12)
    >>> positions, ber_values = bathtub_curve(tie_data, unit_interval=1e-9)

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class BathtubCurveResult:
    """Result of bathtub curve generation.

    Attributes:
        positions: Sampling positions in UI (0.0 to 1.0).
        ber_left: BER values for left side of eye.
        ber_right: BER values for right side of eye.
        ber_total: Combined BER (left + right).
        eye_opening: Eye opening in UI at specified target BER.
        target_ber: Target BER for eye opening calculation.
        unit_interval: Unit interval in seconds.
    """

    positions: NDArray[np.float64]
    ber_left: NDArray[np.float64]
    ber_right: NDArray[np.float64]
    ber_total: NDArray[np.float64]
    eye_opening: float
    target_ber: float
    unit_interval: float


def q_factor_from_ber(ber: float) -> float:
    """Calculate Q-factor from target BER.

    The Q-factor is the number of standard deviations from the mean
    for a Gaussian distribution to achieve the target BER.

    Args:
        ber: Bit error rate (e.g., 1e-12).

    Returns:
        Q-factor value.

    Example:
        >>> q = q_factor_from_ber(1e-12)
        >>> print(f"Q(1e-12) = {q:.3f}")  # ~7.034

    References:
        IEEE 2414-2020: Q = sqrt(2) * erfc_inv(2 * BER)
    """
    if ber <= 0 or ber >= 0.5:
        return np.nan

    # BER = 0.5 * erfc(Q / sqrt(2))
    # erfc(Q / sqrt(2)) = 2 * BER
    # Q / sqrt(2) = erfc_inv(2 * BER)
    # Q = sqrt(2) * erfc_inv(2 * BER)

    q = np.sqrt(2) * special.erfcinv(2 * ber)
    return float(q)


def ber_from_q_factor(q: float) -> float:
    """Calculate BER from Q-factor.

    Args:
        q: Q-factor value.

    Returns:
        Bit error rate.

    Example:
        >>> ber = ber_from_q_factor(7.034)
        >>> print(f"BER = {ber:.2e}")  # ~1e-12
    """
    if q <= 0:
        return 0.5

    # BER = 0.5 * erfc(Q / sqrt(2))
    ber = 0.5 * special.erfc(q / np.sqrt(2))
    return float(ber)


def tj_at_ber(
    rj_rms: float,
    dj_pp: float,
    ber: float = 1e-12,
) -> float:
    """Calculate total jitter at specified BER using dual-Dirac model.

    The dual-Dirac model combines random and deterministic jitter:
    TJ(BER) = 2 * Q(BER) * RJ_rms + DJ_pp

    Common Q values:
    - Q(1e-12) = 7.034
    - Q(1e-15) = 7.941

    Args:
        rj_rms: Random jitter RMS in seconds.
        dj_pp: Deterministic jitter peak-to-peak in seconds.
        ber: Target bit error rate (default 1e-12).

    Returns:
        Total jitter in seconds at specified BER.

    Raises:
        ValueError: If rj_rms is negative.

    Example:
        >>> tj = tj_at_ber(rj_rms=1e-12, dj_pp=10e-12, ber=1e-12)
        >>> print(f"TJ@1e-12: {tj * 1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 6.6
    """
    if rj_rms < 0:
        raise ValueError("RJ must be non-negative")

    if dj_pp < 0:
        raise ValueError("DJ must be non-negative")

    q = q_factor_from_ber(ber)

    if np.isnan(q):
        return np.nan

    # TJ = 2 * Q * RJ_rms + DJ_pp
    tj = 2 * q * rj_rms + dj_pp

    return tj


def bathtub_curve(
    tie_data: NDArray[np.float64],
    unit_interval: float,
    *,
    n_points: int = 100,
    target_ber: float = 1e-12,
    rj_rms: float | None = None,
    dj_delta: float | None = None,
) -> BathtubCurveResult:
    """Generate bathtub curve showing BER vs. sampling position.

    The bathtub curve shows how bit error rate varies across the
    unit interval, with low BER in the center (eye opening) and
    high BER near the edges.

    Args:
        tie_data: Time Interval Error data in seconds.
        unit_interval: Unit interval (bit period) in seconds.
        n_points: Number of points in the curve.
        target_ber: Target BER for eye opening calculation.
        rj_rms: Pre-computed RJ (extracted from data if None).
        dj_delta: Pre-computed DJ delta (extracted from data if None).

    Returns:
        BathtubCurveResult with position and BER arrays.

    Example:
        >>> result = bathtub_curve(tie_data, unit_interval=1e-9)
        >>> print(f"Eye opening: {result.eye_opening:.3f} UI at BER=1e-12")

    References:
        IEEE 2414-2020 Section 6.7
    """

    valid_data = tie_data[~np.isnan(tie_data)]

    # Extract jitter components if not provided
    if rj_rms is None or dj_delta is None:
        rj_rms, dj_delta = _extract_jitter_components(valid_data)

    # Convert to UI and generate positions
    sigma_ui, delta_ui = rj_rms / unit_interval, dj_delta / unit_interval
    positions = np.linspace(0, 1, n_points)

    # Calculate BER arrays using dual-Dirac model
    ber_left, ber_right = _compute_ber_arrays(positions, sigma_ui, delta_ui)

    # Combine and clip BER values
    ber_total = np.clip(ber_left + ber_right, 1e-18, 0.5)
    ber_left = np.clip(ber_left, 1e-18, 0.5)
    ber_right = np.clip(ber_right, 1e-18, 0.5)

    eye_opening = _calculate_eye_opening(positions, ber_total, target_ber)

    return BathtubCurveResult(
        positions=positions,
        ber_left=ber_left,
        ber_right=ber_right,
        ber_total=ber_total,
        eye_opening=eye_opening,
        target_ber=target_ber,
        unit_interval=unit_interval,
    )


def _extract_jitter_components(valid_data: NDArray[np.float64]) -> tuple[float, float]:
    """Extract RJ and DJ components from TIE data."""
    from oscura.analyzers.jitter.decomposition import extract_dj, extract_rj

    try:
        rj_result = extract_rj(valid_data, min_samples=100)
        rj_rms = rj_result.rj_rms
    except Exception:
        rj_rms_raw = np.std(valid_data)
        rj_rms = float(rj_rms_raw)

    try:
        dj_result = extract_dj(valid_data, min_samples=100)
        dj_delta = dj_result.dj_delta
    except Exception:
        dj_delta = 0.0

    return rj_rms, dj_delta


def _compute_ber_arrays(
    positions: NDArray[np.float64], sigma_ui: float, delta_ui: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute BER arrays for left and right edges using dual-Dirac model."""
    n_points = len(positions)
    ber_left, ber_right = np.zeros(n_points), np.zeros(n_points)

    for i, pos in enumerate(positions):
        if sigma_ui > 0:
            # Q-function for Gaussian random jitter
            q_left = (pos - delta_ui) / sigma_ui
            ber_left[i] = 0.5 * special.erfc(q_left / np.sqrt(2))

            q_right = (1 - pos - delta_ui) / sigma_ui
            ber_right[i] = 0.5 * special.erfc(q_right / np.sqrt(2))
        else:
            # No random jitter - step function
            ber_left[i] = 0.5 if pos <= delta_ui else 0
            ber_right[i] = 0.5 if pos >= (1 - delta_ui) else 0

    return ber_left, ber_right


def _calculate_eye_opening(
    positions: NDArray[np.float64],
    ber: NDArray[np.float64],
    target_ber: float,
) -> float:
    """Calculate eye opening at target BER.

    Args:
        positions: Sampling positions in UI.
        ber: BER values at each position.
        target_ber: Target BER for eye opening.

    Returns:
        Eye opening in UI.
    """
    # Find positions where BER <= target_ber
    valid_positions = positions[ber <= target_ber]

    if len(valid_positions) == 0:
        return 0.0

    # Eye opening is the range of valid positions
    eye_opening = float(np.max(valid_positions) - np.min(valid_positions))

    return eye_opening


def eye_opening_at_ber(
    rj_rms: float,
    dj_pp: float,
    unit_interval: float,
    target_ber: float = 1e-12,
) -> float:
    """Calculate horizontal eye opening at target BER.

    Uses the dual-Dirac model to calculate the eye opening width
    at a specified BER level.

    Args:
        rj_rms: Random jitter RMS in seconds.
        dj_pp: Deterministic jitter peak-to-peak in seconds.
        unit_interval: Unit interval in seconds.
        target_ber: Target BER level.

    Returns:
        Eye opening in UI (0.0 to 1.0).

    Example:
        >>> opening = eye_opening_at_ber(1e-12, 10e-12, 100e-12, 1e-12)
        >>> print(f"Eye opening: {opening:.3f} UI")
    """
    # Total jitter at BER
    tj = tj_at_ber(rj_rms, dj_pp, target_ber)

    # Eye opening = UI - TJ
    opening_seconds = unit_interval - tj

    if opening_seconds <= 0:
        return 0.0

    # Convert to UI
    opening_ui = opening_seconds / unit_interval

    return max(0.0, min(1.0, opening_ui))


__all__ = [
    "BathtubCurveResult",
    "bathtub_curve",
    "ber_from_q_factor",
    "eye_opening_at_ber",
    "q_factor_from_ber",
    "tj_at_ber",
]
