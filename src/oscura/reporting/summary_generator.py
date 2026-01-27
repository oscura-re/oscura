"""Natural language summary generation for signal analysis.

This module generates human-readable descriptions of measurements and analysis
results that avoid jargon and explain findings in accessible language.


Example:
    >>> from oscura.reporting import generate_summary
    >>> trace = load("capture.wfm")
    >>> summary = generate_summary(trace)
    >>> print(summary.text)

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


@dataclass
class Finding:
    """Individual analysis finding.

    Attributes:
        title: Short title for the finding.
        description: Plain language description.
        confidence: Confidence score (0.0-1.0).
        severity: Severity level (INFO, WARNING, CRITICAL).
    """

    title: str
    description: str
    confidence: float = 1.0
    severity: str = "INFO"


@dataclass
class Summary:
    """Natural language summary of signal analysis.

    Attributes:
        text: Complete summary text (2-3 sentences, 100-200 words).
        overview: High-level overview sentence.
        findings: List of key findings (minimum 3).
        recommendations: Actionable insights and next steps.
        word_count: Number of words in summary text.
        grade_level: Flesch-Kincaid grade level.
    """

    text: str
    overview: str
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    word_count: int = 0
    grade_level: float = 0.0


def _estimate_grade_level(text: str) -> float:
    """Estimate Flesch-Kincaid grade level.

    Simple approximation based on sentence and word length.

    Args:
        text: Text to analyze.

    Returns:
        Estimated grade level.
    """
    # Split into sentences (simple split on period)
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0

    # Split into words
    words = text.split()
    if not words:
        return 0.0

    # Count syllables (approximation: count vowel groups)
    total_syllables = 0
    for word in words:
        word_lower = word.lower()
        syllable_count = 0
        previous_was_vowel = False

        for char in word_lower:
            is_vowel = char in "aeiouy"
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Minimum 1 syllable per word
        total_syllables += max(1, syllable_count)

    # Flesch-Kincaid formula
    avg_words_per_sentence = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)

    grade_level = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59

    return max(0.0, grade_level)


def _characterize_signal_type(trace: WaveformTrace) -> tuple[str, float]:
    """Characterize basic signal type.

    Simple heuristic-based signal type detection.

    Args:
        trace: Waveform to analyze.

    Returns:
        Tuple of (signal_type, confidence).
    """
    data = trace.data.astype(np.float64)

    # Check if digital (only 2-3 distinct levels)
    unique_values = len(np.unique(np.round(data, decimals=2)))
    value_range = np.ptp(data)

    if unique_values <= 3 and value_range > 0.1:
        # Likely digital
        return "digital", 0.85
    elif value_range < 0.01:
        # Constant signal
        return "DC level", 0.90
    else:
        # Analog
        # Check for periodicity
        if len(data) > 100:
            autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            autocorr = autocorr / autocorr[0]

            # Look for peaks in autocorrelation
            if len(autocorr) > 10 and np.max(autocorr[10:]) > 0.5:
                return "periodic analog", 0.75
        return "analog", 0.70


def _assess_quality(trace: WaveformTrace) -> tuple[str, list[str]]:
    """Assess signal quality.

    Args:
        trace: Waveform to analyze.

    Returns:
        Tuple of (quality_level, issues).
    """
    data = trace.data.astype(np.float64)
    issues = []

    # Check for sufficient data
    if len(data) < 100:
        issues.append("Very short capture (less than 100 samples)")

    # Check noise level (standard deviation relative to range)
    data_range = np.ptp(data)
    if data_range > 0:
        noise_ratio = np.std(data) / data_range
        if noise_ratio > 0.2:
            issues.append("High noise level detected")

    # Check for clipping
    if len(data) > 0:
        data_min = np.min(data)
        data_max = np.max(data)

        # Check if many samples at min/max (possible clipping)
        at_min = np.sum(data == data_min)
        at_max = np.sum(data == data_max)

        if at_min > len(data) * 0.05:
            issues.append("Possible clipping at minimum level")
        if at_max > len(data) * 0.05:
            issues.append("Possible clipping at maximum level")

    # Determine quality level
    if not issues:
        quality = "excellent"
    elif len(issues) == 1:
        quality = "good"
    elif len(issues) == 2:
        quality = "fair"
    else:
        quality = "poor"

    return quality, issues


def _format_frequency(freq_hz: float) -> str:
    """Format frequency in human-readable form.

    Args:
        freq_hz: Frequency in Hz.

    Returns:
        Formatted string.
    """
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.1f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.1f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.1f} kHz"
    else:
        return f"{freq_hz:.1f} Hz"


def _build_findings(
    signal_type: str,
    type_confidence: float,
    quality_level: str,
    quality_issues: list[str],
    trace: WaveformTrace,
) -> list[Finding]:
    """Build findings list from analysis results."""
    findings = []

    findings.append(
        Finding(
            title="Signal Type",
            description=f"Identified as {signal_type}",
            confidence=type_confidence,
            severity="INFO",
        )
    )

    quality_desc = f"Signal quality is {quality_level}"
    if quality_issues:
        quality_desc += f" with {len(quality_issues)} issue(s) noted"

    findings.append(
        Finding(
            title="Signal Quality",
            description=quality_desc,
            confidence=0.85,
            severity="WARNING" if quality_issues else "INFO",
        )
    )

    v_min = float(np.min(trace.data))
    v_max = float(np.max(trace.data))
    v_range = v_max - v_min

    findings.append(
        Finding(
            title="Voltage Range",
            description=f"Signal ranges from {v_min:.3f}V to {v_max:.3f}V (swing: {v_range:.3f}V)",
            confidence=1.0,
            severity="INFO",
        )
    )

    return findings


def _build_recommendations(signal_type: str, quality_issues: list[str]) -> list[str]:
    """Build recommendations from analysis results."""
    recommendations = []

    if "very short" in str(quality_issues).lower():
        recommendations.append("Capture a longer duration to enable more detailed analysis")

    if "noise" in str(quality_issues).lower():
        recommendations.append(
            "Check signal integrity and consider using better probes or shielding"
        )

    if "clipping" in str(quality_issues).lower():
        recommendations.append("Adjust voltage range to prevent signal clipping and data loss")

    if signal_type == "digital" and not recommendations:
        recommendations.append("Signal appears clean and suitable for digital protocol analysis")
    elif signal_type in ["analog", "periodic analog"] and not recommendations:
        recommendations.append("Consider spectral analysis to identify frequency components")

    return recommendations


def _build_summary_text(
    overview: str,
    findings: list[Finding],
    recommendations: list[str],
    include_sections: list[str],
    max_words: int,
) -> str:
    """Build complete summary text from components."""
    summary_parts = []

    if "overview" in include_sections:
        summary_parts.append(overview)

    if "findings" in include_sections and findings:
        key_findings = findings[:3]
        findings_text = " ".join(
            [f"{finding.title}: {finding.description}." for finding in key_findings]
        )
        summary_parts.append(findings_text)

    if "recommendations" in include_sections and recommendations:
        rec_text = "Recommended next steps: " + "; ".join(recommendations[:2]) + "."
        summary_parts.append(rec_text)

    full_text = " ".join(summary_parts)

    words = full_text.split()
    if len(words) > max_words:
        words = words[:max_words]
        full_text = " ".join(words) + "..."

    return full_text


def generate_summary(
    trace: WaveformTrace,
    *,
    context: dict[str, Any] | None = None,
    detail_level: str = "summary",
    max_words: int = 200,
    include_sections: list[str] | None = None,
) -> Summary:
    """Generate natural language summary of signal analysis.

    Creates a plain-English description of the signal and analysis results,
    avoiding technical jargon and explaining findings in accessible terms.

    Args:
        trace: Waveform to summarize.
        context: Optional analysis context (characterization, anomalies, etc.).
        detail_level: Summary detail level ("summary", "intermediate", "expert").
        max_words: Maximum word count for summary text.
        include_sections: Sections to include (default: all).

    Returns:
        Summary object with natural language description.

    Example:
        >>> trace = load("uart_signal.wfm")
        >>> summary = generate_summary(trace)
        >>> print(summary.text)
        This is a digital signal with two voltage levels...

    References:
        DISC-003: Natural Language Summaries
    """
    context = context or {}
    include_sections = include_sections or ["overview", "findings", "recommendations"]

    signal_type, type_confidence = _characterize_signal_type(trace)
    quality_level, quality_issues = _assess_quality(trace)

    sample_rate = trace.metadata.sample_rate
    duration_ms = len(trace.data) / sample_rate * 1000
    overview = f"This is a {signal_type} signal captured at {_format_frequency(sample_rate)} sample rate for {duration_ms:.1f} milliseconds."

    findings = _build_findings(signal_type, type_confidence, quality_level, quality_issues, trace)
    recommendations = _build_recommendations(signal_type, quality_issues)

    full_text = _build_summary_text(
        overview, findings, recommendations, include_sections, max_words
    )
    word_count = len(full_text.split())
    grade_level = _estimate_grade_level(full_text)

    return Summary(
        text=full_text,
        overview=overview,
        findings=findings,
        recommendations=recommendations,
        word_count=word_count,
        grade_level=grade_level,
    )


__all__ = [
    "Finding",
    "Summary",
    "generate_summary",
]
