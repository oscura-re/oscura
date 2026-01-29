"""Legacy system signal analysis.

This module provides analysis tools for legacy RTL/TTL systems with
mixed logic families and multi-voltage domains.


Example:
    >>> from oscura.jupyter.exploratory.legacy import detect_logic_families_multi_channel
    >>> families = detect_logic_families_multi_channel(channels)
    >>> for ch, result in families.items():
    ...     print(f"Channel {ch}: {result['family']} (confidence={result['confidence']:.2f})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace

# Logic family specifications per IEEE/JEDEC standards
LOGIC_FAMILY_SPECS: dict[str, dict[str, float | None]] = {
    "TTL": {
        "vil_max": 0.8,
        "vih_min": 2.0,
        "vol_max": 0.4,
        "voh_min": 2.4,
        "vcc": 5.0,
    },
    "CMOS_5V": {
        "vil_max": 1.5,
        "vih_min": 3.5,
        "vol_max": 0.5,
        "voh_min": 4.5,
        "vcc": 5.0,
    },
    "LVTTL": {
        "vil_max": 0.8,
        "vih_min": 2.0,
        "vol_max": 0.4,
        "voh_min": 2.4,
        "vcc": 3.3,
    },
    "LVCMOS_3V3": {
        "vil_max": 0.8,
        "vih_min": 2.0,
        "vol_max": 0.4,
        "voh_min": 2.4,
        "vcc": 3.3,
    },
    "LVCMOS_2V5": {
        "vil_max": 0.7,
        "vih_min": 1.7,
        "vol_max": 0.4,
        "voh_min": 2.0,
        "vcc": 2.5,
    },
    "LVCMOS_1V8": {
        "vil_max": 0.35 * 1.8,
        "vih_min": 0.65 * 1.8,
        "vol_max": 0.4,
        "voh_min": 1.4,
        "vcc": 1.8,
    },
    "ECL": {
        "vil_max": -1.475,
        "vih_min": -1.105,
        "vol_max": -1.65,
        "voh_min": -0.98,
        "vcc": -5.2,
    },
    "PECL": {
        "vil_max": 3.4,
        "vih_min": 4.0,
        "vol_max": 3.2,
        "voh_min": 4.4,
        "vcc": 5.0,
    },
    "OPEN_COLLECTOR": {
        "vil_max": 0.8,
        "vih_min": 2.0,
        "vol_max": 0.4,
        "voh_min": None,  # Depends on pullup
        "vcc": 5.0,
    },
}


@dataclass
class LogicFamilyResult:
    """Result of logic family detection.

    Attributes:
        family: Detected logic family name.
        confidence: Confidence score (0.0 to 1.0).
        v_low: Measured low voltage level.
        v_high: Measured high voltage level.
        alternatives: List of alternative candidates with confidence.
        degradation_warning: Optional warning about signal degradation.
        deviation_pct: Deviation from spec as percentage.
    """

    family: str
    confidence: float
    v_low: float
    v_high: float
    alternatives: list[tuple[str, float]]
    degradation_warning: str | None = None
    deviation_pct: float = 0.0


def detect_logic_families_multi_channel(
    channels: list[WaveformTrace] | dict[int, WaveformTrace],
    *,
    confidence_thresholds: dict[str, float] | None = None,
    warn_on_degradation: bool = True,
    voltage_tolerance: float = 0.20,
    min_edges_for_detection: int = 10,
) -> dict[int, LogicFamilyResult]:
    """Detect logic family for each channel independently.

    Analyzes voltage distribution per channel and maps to logic family specs.

    Args:
        channels: List or dict of WaveformTrace objects.
        confidence_thresholds: Thresholds for high/medium confidence.
                              Default: {'high': 0.9, 'medium': 0.7}
        warn_on_degradation: If True, warn on degraded signals.
        voltage_tolerance: Tolerance for spec matching (default 20%).
        min_edges_for_detection: Minimum edges required per channel.

    Returns:
        Dictionary mapping channel ID to LogicFamilyResult.

    Example:
        >>> channels = [trace.get_channel(i) for i in range(8)]
        >>> families = detect_logic_families_multi_channel(channels)
        >>> for ch_id, result in families.items():
        ...     print(f"Channel {ch_id}: {result.family} (confidence={result.confidence:.2f})")

    References:
        LEGACY-001: Multi-Channel Logic Family Auto-Detection
        IEEE 1164: Standard for Logic Families
        JEDEC: Logic Family Specifications
    """
    if confidence_thresholds is None:
        confidence_thresholds = {"high": 0.9, "medium": 0.7}

    # Convert list to dict if needed
    channel_dict = dict(enumerate(channels)) if isinstance(channels, list) else channels

    results = {}
    for ch_id, trace in channel_dict.items():
        voltage_levels = _extract_voltage_levels(trace.data)
        candidates = _score_all_logic_families(voltage_levels, voltage_tolerance)
        result = _build_logic_family_result(
            voltage_levels, candidates, min_edges_for_detection, warn_on_degradation
        )
        results[ch_id] = result

    return results


def _extract_voltage_levels(data: NDArray[np.float64]) -> dict[str, float]:
    """Extract voltage levels from channel data.

    Args:
        data: Channel voltage data.

    Returns:
        Dict with v_low, v_high, threshold, edges.
    """
    v_low = float(np.percentile(data, 10))
    v_high = float(np.percentile(data, 90))
    threshold = (v_low + v_high) / 2
    edges = int(np.sum(np.abs(np.diff(data > threshold))))

    return {"v_low": v_low, "v_high": v_high, "threshold": threshold, "edges": edges}


def _score_all_logic_families(
    voltage_levels: dict[str, float], tolerance: float
) -> list[tuple[str, float]]:
    """Score all logic families against measured voltage levels.

    Args:
        voltage_levels: Dict with v_low and v_high.
        tolerance: Voltage tolerance for matching.

    Returns:
        List of (family_name, score) tuples sorted by score.
    """
    candidates = []
    for family_name, specs in LOGIC_FAMILY_SPECS.items():
        score = _score_logic_family(
            voltage_levels["v_low"], voltage_levels["v_high"], specs, tolerance
        )
        if score > 0:
            candidates.append((family_name, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _build_logic_family_result(
    voltage_levels: dict[str, float],
    candidates: list[tuple[str, float]],
    min_edges: int,
    warn_on_degradation: bool,
) -> LogicFamilyResult:
    """Build LogicFamilyResult from voltage data and candidates.

    Args:
        voltage_levels: Dict with v_low, v_high, edges.
        candidates: Sorted list of (family, score) tuples.
        min_edges: Minimum edges for reliable detection.
        warn_on_degradation: Check for signal degradation.

    Returns:
        LogicFamilyResult with classification.
    """
    v_low, v_high, edges = (
        voltage_levels["v_low"],
        voltage_levels["v_high"],
        voltage_levels["edges"],
    )

    if not candidates:
        return LogicFamilyResult(
            family="UNKNOWN",
            confidence=0.0,
            v_low=v_low,
            v_high=v_high,
            alternatives=[],
            degradation_warning="No matching logic family found",
        )

    best_family, best_score = candidates[0]
    confidence = min(1.0, best_score)
    if edges < min_edges:
        confidence *= 0.5

    alternatives = [(n, s) for n, s in candidates[1:4] if best_score - s < 0.2]
    degradation_warning, deviation_pct = _check_degradation(
        best_family, v_high, warn_on_degradation
    )

    return LogicFamilyResult(
        family=best_family,
        confidence=confidence,
        v_low=v_low,
        v_high=v_high,
        alternatives=alternatives,
        degradation_warning=degradation_warning,
        deviation_pct=deviation_pct,
    )


def _check_degradation(family: str, v_high: float, check: bool) -> tuple[str | None, float]:
    """Check for signal degradation against spec.

    Args:
        family: Logic family name.
        v_high: Measured high voltage.
        check: Whether to perform check.

    Returns:
        Tuple of (warning_message, deviation_percent).
    """
    if not check:
        return None, 0.0

    specs = LOGIC_FAMILY_SPECS[family]
    expected_voh = specs["voh_min"]

    if expected_voh is not None and v_high < expected_voh:
        deviation_pct = 100 * (expected_voh - v_high) / expected_voh
        if deviation_pct > 10:
            return f"V_high below spec (expected >= {expected_voh:.3f}V)", deviation_pct

    return None, 0.0


def _score_logic_family(
    v_low: float,
    v_high: float,
    specs: dict[str, float | None],
    tolerance: float,
) -> float:
    """Score how well voltage levels match a logic family.

    Args:
        v_low: Measured low voltage.
        v_high: Measured high voltage.
        specs: Logic family specifications.
        tolerance: Tolerance for matching.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 1.0

    # Check VOL (output low)
    vol_max = specs["vol_max"]
    if vol_max is not None:
        if v_low <= vol_max:
            score *= 1.0  # Exact match
        elif v_low <= vol_max * (1 + tolerance):
            score *= 0.85  # Within tolerance
        else:
            score *= 0.0  # Outside tolerance

    # Check VOH (output high)
    voh_min = specs["voh_min"]
    if voh_min is not None:
        if v_high >= voh_min:
            score *= 1.0
        elif v_high >= voh_min * (1 - tolerance):
            score *= 0.85
        else:
            score *= 0.0

    return score


@dataclass
class CrossCorrelationResult:
    """Result of multi-reference cross-correlation.

    Attributes:
        correlation: Pearson correlation coefficient.
        confidence: Overall confidence in result.
        ref_offset_mv: Reference voltage offset in mV.
        offset_uncertainty_mv: Uncertainty in offset measurement.
        lag_samples: Time lag in samples.
        lag_ns: Time lag in nanoseconds.
        drift_detected: True if reference drift detected.
        drift_rate: Drift rate in V/ms if detected.
        normalized_signal1: Normalized first signal.
        normalized_signal2: Normalized second signal.
    """

    correlation: float
    confidence: float
    ref_offset_mv: float
    offset_uncertainty_mv: float
    lag_samples: int
    lag_ns: float
    drift_detected: bool = False
    drift_rate: float | None = None
    normalized_signal1: NDArray[np.float64] | None = None
    normalized_signal2: NDArray[np.float64] | None = None


def _prepare_signals_for_correlation(
    data1: NDArray[np.float64], data2: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Prepare signals for correlation by normalizing and aligning.

    Args:
        data1: First signal data.
        data2: Second signal data.

    Returns:
        Tuple of (norm1, norm2_corrected, min_len).
    """
    norm1 = _normalize_to_logic_levels(data1)
    norm2 = _normalize_to_logic_levels(data2)
    dc_offset = np.mean(norm1) - np.mean(norm2)
    norm2_corrected = norm2 + dc_offset
    min_len = min(len(norm1), len(norm2_corrected))
    return norm1[:min_len], norm2_corrected[:min_len], min_len


def _compute_correlation_and_lag(
    norm1: NDArray[np.float64], norm2: NDArray[np.float64]
) -> tuple[float, int]:
    """Compute correlation coefficient and lag.

    Args:
        norm1: Normalized first signal.
        norm2: Normalized second signal.

    Returns:
        Tuple of (correlation, lag_samples).
    """
    correlation = np.corrcoef(norm1, norm2)[0, 1]
    xcorr = np.correlate(norm1 - np.mean(norm1), norm2 - np.mean(norm2), mode="full")
    lag_samples = xcorr.argmax() - (len(norm1) - 1)
    return float(correlation), int(lag_samples)


def _compute_reference_offset(
    data1: NDArray[np.float64], data2: NDArray[np.float64], correlation: float
) -> tuple[float, float]:
    """Compute reference voltage offset and confidence.

    Args:
        data1: First signal data.
        data2: Second signal data.
        correlation: Correlation coefficient.

    Returns:
        Tuple of (ref_offset_mv, confidence).
    """
    v1_min: float = float(np.min(data1))
    v2_min: float = float(np.min(data2))
    ref_offset_mv: float = (v2_min - v1_min) * 1000
    confidence: float = abs(correlation) * (1 - min(abs(ref_offset_mv) / 1000, 1.0))
    return float(ref_offset_mv), float(confidence)


def _detect_reference_drift(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
    drift_window_ms: float,
    min_len: int,
) -> tuple[bool, float | None]:
    """Detect time-varying reference drift.

    Args:
        data1: First signal data.
        data2: Second signal data.
        sample_rate: Sample rate in Hz.
        drift_window_ms: Window size in ms.
        min_len: Minimum length of signals.

    Returns:
        Tuple of (drift_detected, drift_rate).
    """
    window_samples = int(drift_window_ms * 1e-3 * sample_rate)
    n_windows = min_len // window_samples

    if n_windows < 2:
        return False, None

    offsets = [
        np.mean(data1[i * window_samples : (i + 1) * window_samples])
        - np.mean(data2[i * window_samples : (i + 1) * window_samples])
        for i in range(n_windows)
    ]

    offset_change = abs(offsets[-1] - offsets[0])
    drift_rate_val = offset_change / (n_windows * drift_window_ms)

    if drift_rate_val > 0.1:
        return True, float(drift_rate_val)

    return False, None


def cross_correlate_multi_reference(
    signal1: WaveformTrace,
    signal2: WaveformTrace,
    *,
    detect_drift: bool = False,
    drift_window_ms: float = 10.0,
) -> CrossCorrelationResult:
    """Correlate signals with different voltage references.

    Normalizes signals to [0, 1] using per-signal logic levels before
    computing correlation, enabling comparison of signals with different
    ground references.

    Args:
        signal1: First signal trace.
        signal2: Second signal trace.
        detect_drift: If True, detect time-varying reference drift.
        drift_window_ms: Window size for drift detection in ms.

    Returns:
        CrossCorrelationResult with correlation and offset information.

    Example:
        >>> ttl = trace.get_channel(0)   # 5V TTL
        >>> cmos = trace.get_channel(1)  # 3.3V CMOS
        >>> result = cross_correlate_multi_reference(ttl, cmos)
        >>> print(f"Correlation: {result.correlation:.3f}")
        >>> print(f"Reference offset: {result.ref_offset_mv:.1f} mV")

    References:
        LEGACY-002: Multi-Reference Voltage Signal Correlation
    """
    # Setup: normalize and align signals
    norm1, norm2_corrected, min_len = _prepare_signals_for_correlation(signal1.data, signal2.data)

    # Processing: compute correlation
    correlation, lag_samples = _compute_correlation_and_lag(norm1, norm2_corrected)
    ref_offset_mv, confidence = _compute_reference_offset(signal1.data, signal2.data, correlation)

    # Formatting: detect drift if requested
    drift_result = (
        _detect_reference_drift(
            signal1.data, signal2.data, signal1.metadata.sample_rate, drift_window_ms, min_len
        )
        if detect_drift
        else (False, None)
    )

    lag_ns = lag_samples / signal1.metadata.sample_rate * 1e9

    return CrossCorrelationResult(
        correlation=float(correlation),
        confidence=float(confidence),
        ref_offset_mv=float(ref_offset_mv),
        offset_uncertainty_mv=float(abs(ref_offset_mv) * 0.1),
        lag_samples=int(lag_samples),
        lag_ns=float(lag_ns),
        drift_detected=drift_result[0],
        drift_rate=drift_result[1],
        normalized_signal1=norm1,
        normalized_signal2=norm2_corrected,
    )


def _normalize_to_logic_levels(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize signal to [0, 1] based on logic levels.

    Args:
        data: Signal data.

    Returns:
        Normalized signal.
    """
    v_min = float(np.percentile(data, 5))
    v_max = float(np.percentile(data, 95))
    v_range = v_max - v_min

    if v_range < 1e-6:
        return np.zeros_like(data)

    return (data - v_min) / v_range


@dataclass
class SignalQualityResult:
    """Result of signal quality assessment.

    Attributes:
        status: 'OK', 'WARNING', or 'CRITICAL'.
        violation_count: Number of spec violations.
        total_samples: Total samples analyzed.
        min_margin_mv: Minimum margin to spec in mV.
        violations: List of violation details.
        vil_violations: Count of VIL violations.
        vih_violations: Count of VIH violations.
        vol_violations: Count of VOL violations.
        voh_violations: Count of VOH violations.
        failure_diagnosis: Suggested failure mode.
        time_to_failure_s: Estimated time to failure.
        drift_rate_mv_per_s: Voltage drift rate.
    """

    status: Literal["OK", "WARNING", "CRITICAL"]
    violation_count: int
    total_samples: int
    min_margin_mv: float
    violations: list[dict[str, Any]]
    vil_violations: int = 0
    vih_violations: int = 0
    vol_violations: int = 0
    voh_violations: int = 0
    vil_rate: float = 0.0
    vih_rate: float = 0.0
    vol_rate: float = 0.0
    voh_rate: float = 0.0
    failure_diagnosis: str | None = None
    time_to_failure_s: float | None = None
    drift_rate_mv_per_s: float | None = None


def assess_signal_quality(
    signal: WaveformTrace,
    logic_family: str,
    *,
    check_aging: bool = False,
    time_window_s: float = 1.0,
) -> SignalQualityResult:
    """Assess signal quality against logic family specs.

    Checks voltage compliance with specifications and detects degraded
    signal levels that may indicate aging or failing components.

    Args:
        signal: Signal trace to assess.
        logic_family: Logic family name (e.g., 'TTL', 'CMOS_5V').
        check_aging: If True, analyze for aging/degradation.
        time_window_s: Window for drift analysis.

    Returns:
        SignalQualityResult with compliance status and violations.

    Example:
        >>> result = assess_signal_quality(signal, logic_family='TTL')
        >>> print(f"Status: {result.status}")
        >>> print(f"Violations: {result.violation_count}")

    References:
        LEGACY-003: Logic Level Compliance Checking
        JEDEC Standard No. 8C
    """
    if logic_family not in LOGIC_FAMILY_SPECS:
        logic_family = "TTL"  # Default fallback

    specs_dict: dict[str, float | None] = dict(LOGIC_FAMILY_SPECS[logic_family])
    data = signal.data
    sample_rate = signal.metadata.sample_rate

    # Classify samples and check violations
    vil_max = specs_dict["vil_max"]
    vih_min = specs_dict["vih_min"]
    assert vil_max is not None and vih_min is not None
    threshold = (vil_max + vih_min) / 2
    is_high = data > threshold
    violations, voh_violations, vol_violations = _check_voltage_violations(
        data, is_high, specs_dict, sample_rate
    )

    # Calculate margins and status
    high_samples = data[is_high]
    low_samples = data[~is_high]
    min_margin_mv = _calculate_voltage_margins(high_samples, low_samples, specs_dict)
    status = _determine_quality_status(min_margin_mv)

    # Calculate violation rates
    n_high = len(high_samples)
    n_low = len(low_samples)
    voh_rate = voh_violations / n_high if n_high > 0 else 0.0
    vol_rate = vol_violations / n_low if n_low > 0 else 0.0

    # Aging analysis
    aging_result = None
    if check_aging and len(data) > 1000:
        aging_result = _analyze_aging(
            data,
            high_samples,
            specs_dict,
            sample_rate,
            time_window_s,
            voh_violations,
            vol_violations,
        )

    return SignalQualityResult(
        status=status,
        violation_count=voh_violations + vol_violations,
        total_samples=len(data),
        min_margin_mv=min_margin_mv,
        violations=violations,
        voh_violations=voh_violations,
        vol_violations=vol_violations,
        voh_rate=voh_rate,
        vol_rate=vol_rate,
        failure_diagnosis=aging_result.get("diagnosis") if aging_result else None,
        time_to_failure_s=aging_result.get("time_to_failure") if aging_result else None,
        drift_rate_mv_per_s=aging_result.get("drift_rate") if aging_result else None,
    )


def _check_voltage_violations(
    data: NDArray[np.float64],
    is_high: NDArray[np.bool_],
    specs: dict[str, float | None],
    sample_rate: float,
) -> tuple[list[dict[str, Any]], int, int]:
    """Check for VOH and VOL violations.

    Args:
        data: Signal data.
        is_high: Boolean mask for high samples.
        specs: Logic family specifications.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (violations list, voh_violation_count, vol_violation_count).
    """
    high_samples = data[is_high]
    low_samples = data[~is_high]

    voh_min = specs["voh_min"]
    vol_max = specs["vol_max"]

    violations = []
    voh_violations = 0
    vol_violations = 0

    # Check VOH violations
    if voh_min is not None and len(high_samples) > 0:
        voh_mask = high_samples < voh_min
        voh_violations = int(np.sum(voh_mask))
        if voh_violations > 0:
            violation_indices = np.where(is_high)[0][voh_mask]
            for idx in violation_indices[:10]:
                violations.append(
                    {
                        "timestamp_us": idx / sample_rate * 1e6,
                        "type": "VOH",
                        "voltage": data[idx],
                        "spec_limit": voh_min,
                    }
                )

    # Check VOL violations
    if vol_max is not None and len(low_samples) > 0:
        vol_mask = low_samples > vol_max
        vol_violations = int(np.sum(vol_mask))
        if vol_violations > 0:
            violation_indices = np.where(~is_high)[0][vol_mask]
            for idx in violation_indices[:10]:
                violations.append(
                    {
                        "timestamp_us": idx / sample_rate * 1e6,
                        "type": "VOL",
                        "voltage": data[idx],
                        "spec_limit": vol_max,
                    }
                )

    return violations, voh_violations, vol_violations


def _calculate_voltage_margins(
    high_samples: NDArray[np.float64],
    low_samples: NDArray[np.float64],
    specs: dict[str, float | None],
) -> float:
    """Calculate minimum voltage margin to spec.

    Args:
        high_samples: High-level samples.
        low_samples: Low-level samples.
        specs: Logic family specifications.

    Returns:
        Minimum margin in mV.
    """
    voh_min = specs["voh_min"]
    vol_max = specs["vol_max"]

    margins: list[float] = []
    if len(high_samples) > 0 and voh_min is not None:
        margins.extend((high_samples - voh_min) * 1000)
    if len(low_samples) > 0 and vol_max is not None:
        margins.extend((vol_max - low_samples) * 1000)

    return min(margins) if margins else 0.0


def _determine_quality_status(min_margin_mv: float) -> Literal["OK", "WARNING", "CRITICAL"]:
    """Determine quality status from voltage margin.

    Args:
        min_margin_mv: Minimum voltage margin in mV.

    Returns:
        Quality status.
    """
    if min_margin_mv < 100:
        return "CRITICAL"
    elif min_margin_mv < 200:
        return "WARNING"
    else:
        return "OK"


def _analyze_aging(
    data: NDArray[np.float64],
    high_samples: NDArray[np.float64],
    specs: dict[str, float | None],
    sample_rate: float,
    time_window_s: float,
    voh_violations: int,
    vol_violations: int,
) -> dict[str, Any] | None:
    """Analyze signal for aging and degradation.

    Args:
        data: Full signal data.
        high_samples: High-level samples.
        specs: Logic family specifications.
        sample_rate: Sample rate in Hz.
        time_window_s: Window size for drift analysis.
        voh_violations: Count of VOH violations.
        vol_violations: Count of VOL violations.

    Returns:
        Dict with diagnosis, time_to_failure, and drift_rate, or None.
    """
    window_samples = int(time_window_s * sample_rate)
    n_windows = len(data) // window_samples

    if n_windows < 2:
        return None

    window_means = [
        np.mean(data[i * window_samples : (i + 1) * window_samples]) for i in range(n_windows)
    ]

    drift = window_means[-1] - window_means[0]
    drift_rate_mv_per_s = drift * 1000 / (n_windows * time_window_s)

    if abs(drift_rate_mv_per_s) <= 0.1:
        return None

    # Estimate time to failure
    time_to_failure = None
    voh_min = specs["voh_min"]
    if voh_min is not None and drift_rate_mv_per_s < 0:
        current_margin = np.mean(high_samples) - voh_min
        if current_margin > 0:
            time_to_failure = current_margin * 1000 / abs(drift_rate_mv_per_s)

    # Diagnose failure mode
    if voh_violations > vol_violations:
        diagnosis = "Degraded output driver (weak high)"
    elif vol_violations > voh_violations:
        diagnosis = "Degraded output driver (weak low)"
    else:
        diagnosis = "General signal degradation"

    return {
        "diagnosis": diagnosis,
        "time_to_failure": time_to_failure,
        "drift_rate": drift_rate_mv_per_s,
    }


@dataclass
class TestPointCharacterization:
    """Characterization of a single test point.

    Attributes:
        channel_id: Channel identifier.
        v_low: Low voltage level.
        v_high: High voltage level.
        v_swing: Voltage swing.
        logic_family: Detected logic family.
        confidence: Detection confidence.
        is_digital: True if signal appears digital.
        is_clock: True if signal appears to be a clock.
        frequency: Estimated frequency if periodic.
    """

    channel_id: int
    v_low: float
    v_high: float
    v_swing: float
    logic_family: str
    confidence: float
    is_digital: bool
    is_clock: bool
    frequency: float | None


def characterize_test_points(
    channels: list[WaveformTrace] | dict[int, WaveformTrace],
    *,
    sample_rate: float | None = None,
) -> dict[int, TestPointCharacterization]:
    """Batch characterize multiple test points.

    Analyzes 8-16 test points to build a voltage level map of an
    unknown board.

    Args:
        channels: List or dict of WaveformTrace objects.
        sample_rate: Sample rate in Hz (uses metadata if not specified).

    Returns:
        Dictionary mapping channel ID to TestPointCharacterization.

    Example:
        >>> channels = [trace.get_channel(i) for i in range(8)]
        >>> chars = characterize_test_points(channels)
        >>> for ch_id, char in chars.items():
        ...     print(f"CH{ch_id}: {char.logic_family} ({char.v_low:.2f}V - {char.v_high:.2f}V)")

    References:
        LEGACY-004: Multi-Channel Voltage Characterization
    """
    if isinstance(channels, list):
        channels = dict(enumerate(channels))

    # First detect logic families
    families = detect_logic_families_multi_channel(channels)

    results = {}

    for ch_id, trace in channels.items():
        data = trace.data
        sr = sample_rate or trace.metadata.sample_rate

        # Voltage statistics
        v_low = float(np.percentile(data, 10))
        v_high = float(np.percentile(data, 90))
        v_swing = v_high - v_low

        # Get logic family result
        family_result = families.get(
            ch_id,
            LogicFamilyResult(
                family="UNKNOWN",
                confidence=0.0,
                v_low=v_low,
                v_high=v_high,
                alternatives=[],
            ),
        )

        # Determine if digital (bimodal distribution)
        is_digital = v_swing > 0.5 and _is_bimodal(data)

        # Check for clock signal
        is_clock = False
        frequency = None

        if is_digital and sr is not None:
            # Check for periodic signal via FFT
            from scipy import signal as sp_signal

            f, psd = sp_signal.welch(data, fs=sr, nperseg=min(1024, len(data)))
            peak_idx = np.argmax(psd[1:]) + 1  # Skip DC
            if psd[peak_idx] > 10 * np.mean(psd):  # Strong peak
                frequency = f[peak_idx]
                # Check duty cycle for clock
                threshold = (v_low + v_high) / 2
                high_ratio = np.mean(data > threshold)
                if 0.4 <= high_ratio <= 0.6:
                    is_clock = True

        results[ch_id] = TestPointCharacterization(
            channel_id=ch_id,
            v_low=v_low,
            v_high=v_high,
            v_swing=v_swing,
            logic_family=family_result.family,
            confidence=family_result.confidence,
            is_digital=is_digital,
            is_clock=is_clock,
            frequency=frequency,
        )

    return results


def _is_bimodal(data: NDArray[np.float64], bins: int = 50) -> bool:
    """Check if data has bimodal distribution.

    Args:
        data: Signal data.
        bins: Number of histogram bins.

    Returns:
        True if distribution appears bimodal (digital signal).
        False for analog signals (sine waves have many peaks).
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram
    peaks = _find_histogram_peaks(hist, centers)

    # Too many peaks suggests analog signal
    if len(peaks) >= 4:
        return False

    # Check if distribution is bimodal
    if len(peaks) == 2 or len(peaks) == 3:
        return _is_bimodal_distribution(data, peaks)

    return False


def _find_histogram_peaks(
    hist: NDArray[np.int64], centers: NDArray[np.float64]
) -> list[tuple[int, int, float]]:
    """Find peaks in histogram.

    Args:
        hist: Histogram counts.
        centers: Bin centers.

    Returns:
        List of (index, count, center) tuples.
    """
    threshold = 0.1 * np.max(hist)
    peaks = []

    # Check first bin (only needs to be > right neighbor)
    if len(hist) > 1 and hist[0] > hist[1] and hist[0] > threshold:
        peaks.append((0, int(hist[0]), float(centers[0])))

    # Check middle bins (need to be > both neighbors)
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > threshold:
            peaks.append((i, int(hist[i]), float(centers[i])))

    # Check last bin (only needs to be > left neighbor)
    if len(hist) > 1 and hist[-1] > hist[-2] and hist[-1] > threshold:
        peaks.append((len(hist) - 1, int(hist[-1]), float(centers[-1])))

    return peaks


def _is_bimodal_distribution(
    data: NDArray[np.float64], peaks: list[tuple[int, int, float]]
) -> bool:
    """Check if peaks indicate bimodal distribution.

    Args:
        data: Signal data.
        peaks: List of (index, count, center) tuples.

    Returns:
        True if distribution is bimodal.
    """
    # Sort peaks by count (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Check if peaks are well-separated
    v_min, v_max = np.min(data), np.max(data)
    v_range = v_max - v_min
    if v_range == 0:
        return False

    # Normalize peak positions
    peak_positions = [(p[2] - v_min) / v_range for p in peaks[:2]]

    # Digital signals have one peak < 0.4 and one peak > 0.6
    has_low_peak = any(p < 0.4 for p in peak_positions)
    has_high_peak = any(p > 0.6 for p in peak_positions)

    # Second peak should be significant
    return has_low_peak and has_high_peak and peaks[1][1] > 0.3 * peaks[0][1]


__all__ = [
    "LOGIC_FAMILY_SPECS",
    "CrossCorrelationResult",
    "LogicFamilyResult",
    "SignalQualityResult",
    "TestPointCharacterization",
    "assess_signal_quality",
    "characterize_test_points",
    "cross_correlate_multi_reference",
    "detect_logic_families_multi_channel",
]
