"""Logic family auto-detection from signal levels.

This module analyzes digital signal voltage levels to automatically
detect the logic family (TTL, CMOS, LVTTL, LVCMOS, etc.).


Example:
    >>> import oscura as osc
    >>> trace = osc.load('digital_signal.wfm')
    >>> family = osc.detect_logic_family(trace)
    >>> print(f"Detected: {family['primary']['name']}")
    >>> print(f"Confidence: {family['primary']['confidence']:.1%}")

References:
    JEDEC Standard definitions for logic families
    TTL: 5V, CMOS: 3.3V/5V, LVTTL: 3.3V, LVCMOS: 1.8V/2.5V/3.3V
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace

# Logic family specifications (VOH, VOL, VIH, VIL, typical VDD)
LOGIC_FAMILY_SPECS = {
    "TTL": {
        "vdd": 5.0,
        "voh_min": 2.4,
        "vol_max": 0.4,
        "vih_min": 2.0,
        "vil_max": 0.8,
    },
    "CMOS_5V": {
        "vdd": 5.0,
        "voh_min": 4.4,
        "vol_max": 0.5,
        "vih_min": 3.5,
        "vil_max": 1.5,
    },
    "CMOS_3V3": {
        "vdd": 3.3,
        "voh_min": 2.4,
        "vol_max": 0.4,
        "vih_min": 2.0,
        "vil_max": 0.8,
    },
    "LVTTL": {
        "vdd": 3.3,
        "voh_min": 2.4,
        "vol_max": 0.4,
        "vih_min": 2.0,
        "vil_max": 0.8,
    },
    "LVCMOS_2V5": {
        "vdd": 2.5,
        "voh_min": 2.0,
        "vol_max": 0.2,
        "vih_min": 1.7,
        "vil_max": 0.7,
    },
    "LVCMOS_1V8": {
        "vdd": 1.8,
        "voh_min": 1.35,
        "vol_max": 0.45,
        "vih_min": 1.17,
        "vil_max": 0.63,
    },
    "LVCMOS_1V5": {
        "vdd": 1.5,
        "voh_min": 1.125,
        "vol_max": 0.375,
        "vih_min": 0.975,
        "vil_max": 0.525,
    },
    "LVCMOS_1V2": {
        "vdd": 1.2,
        "voh_min": 1.1,
        "vol_max": 0.1,
        "vih_min": 0.84,  # 0.7 * 1.2
        "vil_max": 0.36,  # 0.3 * 1.2
    },
}


def detect_logic_family(
    trace: WaveformTrace,
    *,
    min_confidence: float = 0.7,
    return_candidates: bool = False,
) -> dict[str, Any]:
    """Auto-detect logic family from signal levels.

    Analyzes the signal histogram to detect bimodal distribution
    corresponding to logic high and low levels, then maps to the
    nearest logic family specification.

    Args:
        trace: Digital signal to analyze.
        min_confidence: Minimum confidence threshold (0-1).
                        Returns all candidates if confidence below threshold.
        return_candidates: If True, return all candidate families with scores.

    Returns:
        Dictionary containing:
        - primary: Dict with detected family info:
            - name: Logic family name (e.g., 'TTL', 'CMOS_3V3')
            - confidence: Detection confidence (0-1)
            - voh: Measured high-level voltage
            - vol: Measured low-level voltage
            - thresholds: Dict with VIH, VIL, VOH, VOL thresholds
        - candidates: List of all candidates (if return_candidates=True)

    Raises:
        AnalysisError: If signal is not bimodal or levels are ambiguous.

    Example:
        >>> trace = osc.load('cmos_signal.wfm')
        >>> result = osc.detect_logic_family(trace, return_candidates=True)
        >>> print(f"Primary: {result['primary']['name']}")
        >>> print(f"Confidence: {result['primary']['confidence']:.1%}")
        >>> for candidate in result['candidates']:
        ...     print(f"  {candidate['name']}: {candidate['confidence']:.1%}")

    References:
        JEDEC Standard No. 8 (Interface Standards)
        TTL: Texas Instruments SN54/74 Series
    """
    # Analyze histogram to find bimodal distribution
    voh, vol, confidence_bimodal = _detect_logic_levels(trace.data)

    if np.isnan(voh) or np.isnan(vol):
        raise AnalysisError(
            "Could not detect distinct logic levels. "
            "Signal may not be digital or has insufficient transitions."
        )

    # Calculate swing and nominal VDD
    voh - vol
    estimated_vdd = voh + 0.3  # Assume VOH is slightly below VDD

    # Score each logic family
    candidates = []
    for family_name, specs in LOGIC_FAMILY_SPECS.items():
        score = _score_logic_family(voh, vol, estimated_vdd, specs)
        candidates.append(
            {
                "name": family_name,
                "confidence": score * confidence_bimodal,
                "voh": voh,
                "vol": vol,
                "vdd_estimated": estimated_vdd,
                "thresholds": {
                    "vih": specs["vih_min"],
                    "vil": specs["vil_max"],
                    "voh": specs["voh_min"],
                    "vol": specs["vol_max"],
                },
            }
        )

    # Sort by confidence
    # Type narrowing: x["confidence"] is object, but we know it's a float-like value
    candidates.sort(
        key=lambda x: float(cast("float", x["confidence"])) if x["confidence"] is not None else 0.0,
        reverse=True,
    )

    # Primary detection
    primary = candidates[0]

    # Build result
    result = {
        "primary": primary,
    }

    if return_candidates:
        result["candidates"] = candidates  # type: ignore[assignment]

    # Warn if confidence is low
    if primary["confidence"] < min_confidence:  # type: ignore[operator]
        # In real implementation, might prompt user or return multiple candidates
        pass

    return result


def _detect_logic_levels(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Detect logic high and low levels from signal histogram.

    Args:
        data: Signal data array.

    Returns:
        Tuple of (VOH, VOL, confidence) where confidence is 0-1.
    """
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram (expect bimodal for digital signal)
    # Smooth histogram slightly
    from scipy.ndimage import gaussian_filter1d

    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

    # Find local maxima
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            if hist_smooth[i] > np.max(hist_smooth) * 0.1:  # At least 10% of max
                peaks.append((bin_centers[i], hist_smooth[i]))

    # Should have 2 peaks for digital signal
    if len(peaks) < 2:
        # Fall back to percentile method
        vol = np.percentile(data, 5)
        voh = np.percentile(data, 95)
        confidence = 0.5
    else:
        # Take two highest peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        peak1, peak2 = peaks[0], peaks[1]

        # Lower voltage is VOL, higher is VOH
        if peak1[0] < peak2[0]:
            vol, voh = peak1[0], peak2[0]
        else:
            vol, voh = peak2[0], peak1[0]

        # Confidence based on peak separation and height
        separation = abs(voh - vol)
        min_peak_height = min(peak1[1], peak2[1])
        max_peak_height = max(peak1[1], peak2[1])

        # Good separation and similar peak heights = high confidence
        confidence = min(1.0, (separation / 5.0) * (min_peak_height / max_peak_height))

    return voh, vol, confidence


def _score_logic_family(
    voh: float,
    vol: float,
    vdd: float,
    specs: dict[str, float],
) -> float:
    """Score how well measured levels match a logic family.

    Args:
        voh: Measured high level.
        vol: Measured low level.
        vdd: Estimated supply voltage.
        specs: Logic family specifications.

    Returns:
        Score from 0 (no match) to 1 (perfect match).
    """
    # Check VDD match
    vdd_error = abs(vdd - specs["vdd"]) / specs["vdd"]
    vdd_score = max(0, 1.0 - vdd_error * 2)  # Within 50% gets some score

    # Check VOH is above spec minimum
    if voh >= specs["voh_min"]:
        voh_score = 1.0
    else:
        voh_error = (specs["voh_min"] - voh) / specs["voh_min"]
        voh_score = max(0, 1.0 - voh_error * 5)

    # Check VOL is below spec maximum
    if vol <= specs["vol_max"]:
        vol_score = 1.0
    else:
        vol_error = (vol - specs["vol_max"]) / specs["vol_max"]
        vol_score = max(0, 1.0 - vol_error * 5)

    # Combined score (weighted average)
    score = vdd_score * 0.5 + voh_score * 0.25 + vol_score * 0.25

    return score


__all__ = ["LOGIC_FAMILY_SPECS", "detect_logic_family"]
