"""Data quality assessment for signal analysis.

This module assesses whether captured data is sufficient and of adequate
quality for meaningful analysis.


Example:
    >>> from oscura.discovery import assess_data_quality
    >>> quality = assess_data_quality(trace)
    >>> print(f"Status: {quality.status}")
    >>> for metric in quality.metrics:
    ...     print(f"{metric.name}: {metric.status}")

References:
    IEEE 1241-2010: ADC Terminology and Test Methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.analyzers.statistics.basic import basic_stats
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

QualityStatus = Literal["PASS", "WARNING", "FAIL"]
AnalysisScenario = Literal["protocol_decode", "timing_analysis", "fft", "eye_diagram", "general"]


@dataclass
class QualityMetric:
    """Individual quality metric result.

    Attributes:
        name: Metric name (e.g., "Sample Rate", "Resolution").
        status: Quality status (PASS, WARNING, FAIL).
        passed: Whether metric passes minimum requirements.
        current_value: Measured value.
        required_value: Required value for this scenario.
        unit: Unit of measurement.
        margin_percent: Margin relative to requirement (positive = good).
        explanation: Plain-language explanation if failed.
        recommendation: Actionable recommendation to fix issue.

    Example:
        >>> metric = QualityMetric(
        ...     name="Sample Rate",
        ...     status="WARNING",
        ...     passed=False,
        ...     current_value=50.0,
        ...     required_value=100.0,
        ...     unit="MS/s"
        ... )
    """

    name: str
    status: QualityStatus
    passed: bool
    current_value: float
    required_value: float
    unit: str
    margin_percent: float = 0.0
    explanation: str = ""
    recommendation: str = ""


@dataclass
class DataQuality:
    """Overall data quality assessment result.

    Attributes:
        status: Overall quality status (PASS, WARNING, FAIL).
        confidence: Assessment confidence (0.0-1.0).
        metrics: List of individual quality metrics.
        improvement_suggestions: Suggested improvements if quality is poor.

    Example:
        >>> quality = assess_data_quality(trace)
        >>> if quality.status != "PASS":
        ...     print("Quality issues detected:")
        ...     for metric in quality.metrics:
        ...         if not metric.passed:
        ...             print(f"  - {metric.name}: {metric.explanation}")
    """

    status: QualityStatus
    confidence: float
    metrics: list[QualityMetric] = field(default_factory=list)
    improvement_suggestions: list[dict[str, str]] = field(default_factory=list)


def _extract_trace_data(
    trace: WaveformTrace | DigitalTrace,
) -> tuple[NDArray[np.floating[Any]], float, bool]:
    """Extract data array, sample rate, and type from trace.

    Args:
        trace: Input trace.

    Returns:
        Tuple of (data, sample_rate, is_analog).

    Raises:
        ValueError: If trace is empty.
    """
    if len(trace) == 0:
        raise ValueError("Cannot assess quality of empty trace")

    if isinstance(trace, WaveformTrace):
        return trace.data, trace.metadata.sample_rate, True
    else:
        return trace.data.astype(np.float64), trace.metadata.sample_rate, False


def _prepare_quality_assessment(
    data: NDArray[np.floating[Any]], protocol_params: dict[str, Any] | None
) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Prepare data for quality assessment.

    Args:
        data: Signal data array.
        protocol_params: Optional protocol parameters.

    Returns:
        Tuple of (stats, voltage_swing, protocol_params).
    """
    stats = basic_stats(data)
    voltage_swing = stats["max"] - stats["min"]
    return stats, voltage_swing, protocol_params or {}


def _assess_all_quality_metrics(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
    is_analog: bool,
    scenario: AnalysisScenario,
    protocol_params: dict[str, Any],
) -> list[QualityMetric]:
    """Assess all quality metrics.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.
        is_analog: Whether signal is analog.
        scenario: Analysis scenario.
        protocol_params: Protocol parameters.

    Returns:
        List of quality metrics.
    """
    return [
        _assess_sample_rate(sample_rate, data, stats, scenario, protocol_params),
        _assess_resolution(data, voltage_swing, stats, is_analog, scenario),
        _assess_duration(len(data), sample_rate, data, scenario, protocol_params),
        _assess_noise(data, voltage_swing, stats, scenario),
    ]


def _determine_overall_quality_status(
    metrics: list[QualityMetric], strict_mode: bool
) -> QualityStatus:
    """Determine overall quality status from individual metrics.

    Args:
        metrics: List of quality metrics.
        strict_mode: Whether to fail on warnings.

    Returns:
        Overall quality status.
    """
    failed = [m for m in metrics if m.status == "FAIL"]
    warnings = [m for m in metrics if m.status == "WARNING"]

    if failed or (strict_mode and warnings):
        return "FAIL"
    elif warnings:
        return "WARNING"
    else:
        return "PASS"


def _calculate_quality_confidence(metrics: list[QualityMetric]) -> float:
    """Calculate quality confidence score.

    Args:
        metrics: List of quality metrics.

    Returns:
        Confidence score (0.5-1.0).
    """
    passed_count = sum(1 for m in metrics if m.passed)
    return round(0.5 + (passed_count / len(metrics)) * 0.5, 2)


def _generate_improvement_suggestions(metrics: list[QualityMetric]) -> list[dict[str, str]]:
    """Generate improvement suggestions from failed metrics.

    Args:
        metrics: List of quality metrics.

    Returns:
        List of suggestion dictionaries.
    """
    suggestions = []
    for metric in metrics:
        if not metric.passed and metric.recommendation:
            suggestions.append(
                {
                    "action": metric.recommendation,
                    "expected_benefit": f"Improves {metric.name.lower()} to required level",
                    "difficulty_level": "Easy"
                    if "setting" in metric.recommendation.lower()
                    else "Medium",
                }
            )
    return suggestions


def assess_data_quality(
    trace: WaveformTrace | DigitalTrace,
    *,
    scenario: AnalysisScenario = "general",
    protocol_params: dict[str, Any] | None = None,
    strict_mode: bool = False,
) -> DataQuality:
    """Assess whether captured data is adequate for analysis.

    Evaluates sample rate, resolution, duration, and noise level against
    scenario-specific requirements.

    Args:
        trace: Input waveform or digital trace.
        scenario: Analysis scenario for scenario-specific thresholds.
        protocol_params: Protocol-specific parameters (e.g., clock frequency).
        strict_mode: If True, fail on any warnings.

    Returns:
        DataQuality assessment with overall status and individual metrics.

    Raises:
        ValueError: If trace is empty or invalid.

    Example:
        >>> quality = assess_data_quality(trace, scenario='protocol_decode')
        >>> print(f"Overall: {quality.status} (confidence: {quality.confidence:.2f})")
        >>> for metric in quality.metrics:
        ...     if metric.status != 'PASS':
        ...         print(f"Issue: {metric.name} - {metric.explanation}")
        ...         print(f"Fix: {metric.recommendation}")

    References:
        DISC-009: Data Quality Assessment
    """
    # Setup: extract and prepare signal data
    data, sample_rate, is_analog = _extract_trace_data(trace)
    stats, voltage_swing, protocol_params = _prepare_quality_assessment(data, protocol_params)

    # Processing: assess individual metrics
    metrics = _assess_all_quality_metrics(
        data, sample_rate, voltage_swing, stats, is_analog, scenario, protocol_params
    )

    # Formatting: determine overall status and generate report
    overall_status = _determine_overall_quality_status(metrics, strict_mode)
    confidence = _calculate_quality_confidence(metrics)
    suggestions = _generate_improvement_suggestions(metrics)

    return DataQuality(
        status=overall_status,
        confidence=confidence,
        metrics=metrics,
        improvement_suggestions=suggestions,
    )


def _assess_sample_rate(
    sample_rate: float,
    data: NDArray[np.floating[Any]],
    stats: dict[str, float],
    scenario: AnalysisScenario,
    protocol_params: dict[str, Any],
) -> QualityMetric:
    """Assess sample rate adequacy.

    Args:
        sample_rate: Sample rate in Hz.
        data: Signal data array.
        stats: Basic statistics.
        scenario: Analysis scenario.
        protocol_params: Protocol-specific parameters.

    Returns:
        QualityMetric for sample rate.
    """
    # Estimate signal frequency
    mean_val = stats["mean"]
    crossings = np.where(np.diff(np.sign(data - mean_val)) != 0)[0]

    if len(crossings) >= 2:
        avg_half_period = np.mean(np.diff(crossings))
        signal_freq = sample_rate / (avg_half_period * 2) if avg_half_period > 0 else 0
    else:
        signal_freq = 0

    # Determine required sample rate based on scenario
    if scenario == "protocol_decode":
        # Need 10x the bit rate
        if "clock_freq_mhz" in protocol_params:
            clock_freq = protocol_params["clock_freq_mhz"] * 1e6
            required_rate = clock_freq * 10
        elif signal_freq > 0:
            required_rate = signal_freq * 10
        else:
            required_rate = 10e6  # Default 10 MS/s minimum
    elif scenario == "timing_analysis":
        # Need 100x the edge rate
        required_rate = signal_freq * 100 if signal_freq > 0 else 100e6
    elif scenario == "fft":
        # Nyquist + 20%
        required_rate = signal_freq * 2.4 if signal_freq > 0 else 10e6
    elif scenario == "eye_diagram":
        # Need high oversampling
        required_rate = signal_freq * 50 if signal_freq > 0 else 100e6
    else:  # general
        # At least 10x signal frequency
        required_rate = signal_freq * 10 if signal_freq > 0 else 10e6

    # Calculate margin
    margin_percent = ((sample_rate - required_rate) / required_rate) * 100

    # Determine status
    if margin_percent >= 0:
        status: QualityStatus = "PASS"
        passed = True
        explanation = ""
        recommendation = ""
    elif margin_percent >= -20:
        status = "WARNING"
        passed = False
        explanation = f"Sample rate is {abs(margin_percent):.0f}% below recommended"
        recommendation = (
            f"Increase sample rate to {required_rate / 1e6:.0f} MS/s "
            f"(currently {sample_rate / 1e6:.0f} MS/s)"
        )
    else:
        status = "FAIL"
        passed = False
        explanation = f"Sample rate is critically low ({abs(margin_percent):.0f}% below required)"
        recommendation = f"Increase sample rate to at least {required_rate / 1e6:.0f} MS/s"

    return QualityMetric(
        name="Sample Rate",
        status=status,
        passed=passed,
        current_value=sample_rate / 1e6,
        required_value=required_rate / 1e6,
        unit="MS/s",
        margin_percent=margin_percent,
        explanation=explanation,
        recommendation=recommendation,
    )


def _assess_resolution(
    data: NDArray[np.floating[Any]],
    voltage_swing: float,
    stats: dict[str, float],
    is_analog: bool,
    scenario: AnalysisScenario,
) -> QualityMetric:
    """Assess vertical resolution adequacy.

    Args:
        data: Signal data array.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.
        is_analog: Whether signal is analog.
        scenario: Analysis scenario.

    Returns:
        QualityMetric for resolution.
    """
    # Estimate effective number of bits (ENOB)
    if voltage_swing > 0:
        # Approximate ENOB from noise level
        noise_rms = stats["std"]
        snr_linear = (voltage_swing / 2) / (noise_rms + 1e-12)
        snr_db = 20 * np.log10(snr_linear) if snr_linear > 0 else 0
    else:
        # No voltage swing - cannot assess SNR meaningfully
        # Treat as infinite SNR (perfect) since there's no dynamic range
        snr_db = 100.0

    # Determine required resolution
    if scenario in ("protocol_decode", "timing_analysis"):
        required_snr = 20.0  # dB
    elif scenario in ("fft", "eye_diagram"):
        required_snr = 40.0  # dB
    else:
        required_snr = 20.0  # dB

    # Use SNR for assessment
    current_snr = snr_db
    margin_percent = ((current_snr - required_snr) / required_snr) * 100

    # Determine status
    if current_snr >= required_snr:
        status: QualityStatus = "PASS"
        passed = True
        explanation = ""
        recommendation = ""
    elif current_snr >= required_snr * 0.8:
        status = "WARNING"
        passed = False
        explanation = f"SNR is {abs(margin_percent):.0f}% below recommended ({current_snr:.1f} dB)"
        recommendation = "Reduce noise sources or increase signal amplitude"
    else:
        status = "FAIL"
        passed = False
        explanation = f"SNR is critically low ({current_snr:.1f} dB, need {required_snr:.0f} dB)"
        recommendation = "Significantly improve signal quality or use higher resolution capture"

    return QualityMetric(
        name="Resolution",
        status=status,
        passed=passed,
        current_value=current_snr,
        required_value=required_snr,
        unit="dB SNR",
        margin_percent=margin_percent,
        explanation=explanation,
        recommendation=recommendation,
    )


def _assess_duration(
    n_samples: int,
    sample_rate: float,
    data: NDArray[np.floating[Any]],
    scenario: AnalysisScenario,
    protocol_params: dict[str, Any],
) -> QualityMetric:
    """Assess capture duration adequacy.

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.
        data: Signal data array.
        scenario: Analysis scenario.
        protocol_params: Protocol-specific parameters.

    Returns:
        QualityMetric for duration.
    """
    duration_sec = n_samples / sample_rate

    # Estimate signal period
    mean_val = np.mean(data)
    crossings = np.where(np.diff(np.sign(data - mean_val)) != 0)[0]

    if len(crossings) >= 2:
        avg_half_period = np.mean(np.diff(crossings))
        signal_period = (avg_half_period * 2) / sample_rate
        num_periods = duration_sec / signal_period if signal_period > 0 else 0
    else:
        num_periods = 0
        signal_period = duration_sec / 10  # Assume at least 10 periods

    # Determine required duration
    if scenario in {"protocol_decode", "timing_analysis"}:
        required_periods = 100
    elif scenario == "fft":
        required_periods = 10  # Need enough for frequency resolution
    elif scenario == "eye_diagram":
        required_periods = 1000  # Need many UIs
    else:
        required_periods = 100

    required_duration = required_periods * signal_period
    margin_percent = (
        ((duration_sec - required_duration) / required_duration) * 100
        if required_duration > 0
        else 100
    )

    # Determine status
    if num_periods >= required_periods or margin_percent >= 0:
        status: QualityStatus = "PASS"
        passed = True
        explanation = ""
        recommendation = ""
    elif num_periods >= required_periods * 0.5:
        status = "WARNING"
        passed = False
        explanation = (
            f"Captured only {num_periods:.0f} signal periods, "
            f"recommended minimum is {required_periods}"
        )
        recommendation = (
            f"Increase capture duration to at least {required_duration * 1e3:.1f} ms "
            f"(currently {duration_sec * 1e3:.1f} ms)"
        )
    else:
        status = "FAIL"
        passed = False
        explanation = f"Capture duration is critically short ({num_periods:.0f} periods)"
        recommendation = f"Increase capture duration to at least {required_duration * 1e3:.1f} ms"

    return QualityMetric(
        name="Duration",
        status=status,
        passed=passed,
        current_value=duration_sec * 1e3,
        required_value=required_duration * 1e3,
        unit="ms",
        margin_percent=margin_percent,
        explanation=explanation,
        recommendation=recommendation,
    )


def _assess_noise(
    data: NDArray[np.floating[Any]],
    voltage_swing: float,
    stats: dict[str, float],
    scenario: AnalysisScenario,
) -> QualityMetric:
    """Assess noise level.

    Args:
        data: Signal data array.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.
        scenario: Analysis scenario.

    Returns:
        QualityMetric for noise level.
    """
    if voltage_swing == 0:
        # No signal swing, can't assess noise
        return QualityMetric(
            name="Noise Level",
            status="PASS",
            passed=True,
            current_value=0.0,
            required_value=0.0,
            unit="% of swing",
            margin_percent=100.0,
        )

    # Noise RMS as percentage of swing
    noise_rms = stats["std"]
    noise_percent = (noise_rms / voltage_swing) * 100

    # Determine acceptable noise level
    if scenario in ("protocol_decode", "timing_analysis"):
        max_noise_percent = 10.0
    elif scenario in ("fft", "eye_diagram"):
        max_noise_percent = 5.0
    else:
        max_noise_percent = 10.0

    margin_percent = ((max_noise_percent - noise_percent) / max_noise_percent) * 100

    # Determine status
    if noise_percent <= max_noise_percent:
        status: QualityStatus = "PASS"
        passed = True
        explanation = ""
        recommendation = ""
    elif noise_percent <= max_noise_percent * 1.5:
        status = "WARNING"
        passed = False
        explanation = (
            f"Noise level is {noise_percent:.1f}% of signal swing "
            f"(max recommended: {max_noise_percent:.0f}%)"
        )
        recommendation = "Reduce noise sources, check grounding, or use averaging"
    else:
        status = "FAIL"
        passed = False
        explanation = f"Noise level is critically high ({noise_percent:.1f}% of swing)"
        recommendation = (
            "Significantly reduce noise through better probing, shielding, or bandwidth limiting"
        )

    return QualityMetric(
        name="Noise Level",
        status=status,
        passed=passed,
        current_value=noise_percent,
        required_value=max_noise_percent,
        unit="% of swing",
        margin_percent=margin_percent,
        explanation=explanation,
        recommendation=recommendation,
    )


__all__ = [
    "AnalysisScenario",
    "DataQuality",
    "QualityMetric",
    "QualityStatus",
    "assess_data_quality",
]
