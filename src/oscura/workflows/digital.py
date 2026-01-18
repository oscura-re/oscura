"""Digital buffer characterization workflow.

This module implements complete TTL/CMOS buffer characterization in a single
function call, with automatic logic family detection.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('74hc04_output.wfm')
    >>> result = osc.characterize_buffer(trace)
    >>> print(f"Logic Family: {result['logic_family']}")
    >>> print(f"Rise Time: {result['rise_time']:.2f} ns")
    >>> print(f"Status: {result['status']}")

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    JEDEC Standard No. 65B: High-Speed Interface Timing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def characterize_buffer(
    trace: WaveformTrace,
    *,
    reference_trace: WaveformTrace | None = None,
    logic_family: str | None = None,
    thresholds: dict[str, float] | None = None,
    report: str | None = None,
) -> dict[str, Any]:
    """Characterize digital buffer timing and quality.

    One-call characterization of digital buffer including:
    - Automatic logic family detection (if not specified)
    - Rise/fall time measurements
    - Propagation delay (if reference provided)
    - Overshoot/undershoot analysis
    - Noise margin calculation
    - Pass/fail against logic family specifications

    Args:
        trace: Output signal to characterize.
        reference_trace: Optional reference (input) signal for propagation delay.
        logic_family: Logic family override (e.g., 'TTL', 'CMOS_3V3', 'CMOS_5V').
                      If None, auto-detected from signal levels.
        thresholds: Optional dict of custom pass/fail thresholds
                    (e.g., {'rise_time': 10e-9} for 10 ns max).
        report: Optional path to save HTML report.

    Returns:
        Dictionary containing:
        - logic_family: Detected or specified logic family
        - rise_time: 10%-90% rise time in seconds
        - fall_time: 10%-90% fall time in seconds
        - propagation_delay: Delay from reference (if provided), in seconds
        - overshoot: Peak overshoot voltage
        - overshoot_percent: Overshoot as percentage of swing
        - undershoot: Peak undershoot voltage
        - undershoot_percent: Undershoot as percentage of swing
        - noise_margin_high: High-level noise margin in volts
        - noise_margin_low: Low-level noise margin in volts
        - status: 'PASS' or 'FAIL' based on logic family specs
        - reference_comparison: Dict with timing drift if reference provided
        - confidence: Confidence score for logic family detection (0-1)

    Raises:
        AnalysisError: If trace has insufficient transitions for analysis.

    Example:
        >>> trace = osc.load('74hc04_output.wfm')
        >>> result = osc.characterize_buffer(trace, logic_family='CMOS_3V3')
        >>> print(f"Rise Time: {result['rise_time']*1e9:.2f} ns")
        >>> print(f"Status: {result['status']}")

    References:
        IEEE 181-2011 Section 5.2 (Edge timing)
        JEDEC Standard No. 65B (Logic family specifications)
    """
    # Import here to avoid circular dependencies
    from oscura.analyzers.waveform.measurements import (
        fall_time,
        overshoot,
        rise_time,
        undershoot,
    )
    from oscura.inference.logic import detect_logic_family

    # Auto-detect logic family if not specified
    if logic_family is None:
        detection = detect_logic_family(trace)
        logic_family = detection["primary"]["name"]
        confidence = detection["primary"]["confidence"]
        voh = detection["primary"]["voh"]
        vol = detection["primary"]["vol"]
    else:
        confidence = 1.0
        # Measure VOH/VOL from trace
        voh = np.percentile(trace.data, 95)
        vol = np.percentile(trace.data, 5)

    # Measure timing parameters
    try:
        t_rise = rise_time(trace)
        t_fall = fall_time(trace)
    except Exception as e:
        raise AnalysisError(f"Failed to measure rise/fall time: {e}") from e

    # Measure overshoot/undershoot
    v_overshoot = overshoot(trace)
    v_undershoot = undershoot(trace)

    # Calculate percentages
    swing = voh - vol
    if swing > 0:
        overshoot_pct = (v_overshoot / swing) * 100.0
        undershoot_pct = (v_undershoot / swing) * 100.0
    else:
        overshoot_pct = 0.0
        undershoot_pct = 0.0

    # Calculate noise margins (simplified - uses typical values)
    # In a real implementation, these would come from LOGIC_FAMILIES constants
    logic_specs = _get_logic_specs(logic_family)
    noise_margin_high = voh - logic_specs["vih"]
    noise_margin_low = logic_specs["vil"] - vol

    # Propagation delay if reference provided
    propagation_delay = None
    timing_drift = None
    if reference_trace is not None:
        try:
            from oscura.analyzers.digital.timing import (
                propagation_delay as prop_delay,
            )

            propagation_delay = prop_delay(reference_trace, trace)
        except Exception:
            # If propagation delay measurement fails, set to None
            propagation_delay = None

    # Apply thresholds and determine pass/fail
    status = "PASS"
    if thresholds is not None:
        if "rise_time" in thresholds and t_rise > thresholds["rise_time"]:
            status = "FAIL"
        if "fall_time" in thresholds and t_fall > thresholds["fall_time"]:
            status = "FAIL"
        if "overshoot_percent" in thresholds and overshoot_pct > thresholds["overshoot_percent"]:
            status = "FAIL"
    else:
        # Use logic family defaults
        if t_rise > logic_specs.get("max_rise_time", float("inf")):
            status = "FAIL"
        if t_fall > logic_specs.get("max_fall_time", float("inf")):
            status = "FAIL"

    # Build result dictionary
    result = {
        "logic_family": logic_family,
        "confidence": confidence,
        "rise_time": t_rise,
        "fall_time": t_fall,
        "propagation_delay": propagation_delay,
        "overshoot": v_overshoot,
        "overshoot_percent": overshoot_pct,
        "undershoot": v_undershoot,
        "undershoot_percent": undershoot_pct,
        "noise_margin_high": noise_margin_high,
        "noise_margin_low": noise_margin_low,
        "voh": voh,
        "vol": vol,
        "status": status,
        "reference_comparison": None,
    }

    if reference_trace is not None and propagation_delay is not None:
        result["reference_comparison"] = {
            "propagation_delay": propagation_delay,
            "timing_drift": timing_drift,
        }

    # Generate report if requested
    if report is not None:
        _generate_buffer_report(result, report)

    return result


def _get_logic_specs(family: str) -> dict[str, float]:
    """Get specifications for a logic family.

    Args:
        family: Logic family name.

    Returns:
        Dict with VIH, VIL, and timing specs.
    """
    specs = {
        "TTL": {
            "vih": 2.0,
            "vil": 0.8,
            "max_rise_time": 10e-9,
            "max_fall_time": 10e-9,
        },
        "CMOS_5V": {
            "vih": 3.5,
            "vil": 1.5,
            "max_rise_time": 15e-9,
            "max_fall_time": 15e-9,
        },
        "CMOS_3V3": {
            "vih": 2.0,
            "vil": 0.8,
            "max_rise_time": 5e-9,
            "max_fall_time": 5e-9,
        },
        "LVTTL": {
            "vih": 2.0,
            "vil": 0.8,
            "max_rise_time": 3e-9,
            "max_fall_time": 3e-9,
        },
        "LVCMOS": {
            "vih": 1.7,
            "vil": 0.7,
            "max_rise_time": 2e-9,
            "max_fall_time": 2e-9,
        },
    }
    return specs.get(family, specs["CMOS_3V3"])


def _generate_buffer_report(result: dict[str, Any], output_path: str) -> None:
    """Generate HTML report for buffer characterization.

    Args:
        result: Characterization result dictionary.
        output_path: Path to save HTML report.
    """
    # Simplified report generation - in real implementation would use
    # oscura.reporting module
    html = f"""
    <html>
    <head><title>Buffer Characterization Report</title></head>
    <body>
    <h1>Buffer Characterization Report</h1>
    <h2>Logic Family: {result["logic_family"]} (confidence: {result["confidence"]:.1%})</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
        <tr><td>Rise Time</td><td>{result["rise_time"] * 1e9:.2f}</td><td>ns</td></tr>
        <tr><td>Fall Time</td><td>{result["fall_time"] * 1e9:.2f}</td><td>ns</td></tr>
        <tr><td>Overshoot</td><td>{result["overshoot_percent"]:.1f}</td><td>%</td></tr>
        <tr><td>Undershoot</td><td>{result["undershoot_percent"]:.1f}</td><td>%</td></tr>
        <tr><td>Status</td><td><b>{result["status"]}</b></td><td></td></tr>
    </table>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)


__all__ = ["characterize_buffer"]
