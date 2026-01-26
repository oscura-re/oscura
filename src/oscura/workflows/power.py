"""Power analysis workflow.

This module implements comprehensive power consumption analysis from
voltage and current traces.


Example:
    >>> import oscura as osc
    >>> voltage = osc.load('vdd.wfm')
    >>> current = osc.load('idd.wfm')
    >>> result = osc.power_analysis(voltage, current)
    >>> print(f"Average Power: {result['average_power']*1e3:.2f} mW")

References:
    IEC 61000: Electromagnetic compatibility
    IEEE 1241-2010: ADC terminology and test methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def power_analysis(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    input_voltage: WaveformTrace | None = None,
    input_current: WaveformTrace | None = None,
    report: str | None = None,
) -> dict[str, Any]:
    """Comprehensive power consumption analysis.

    Analyzes power consumption from voltage and current measurements.

    Args:
        voltage: Output voltage trace.
        current: Output current trace.
        input_voltage: Optional input voltage for efficiency calculation.
        input_current: Optional input current for efficiency calculation.
        report: Optional path to save HTML report.

    Returns:
        Dictionary with power_trace, average_power, output_power_avg, output_power_rms,
        peak_power, min_power, energy, duration, and optionally efficiency, power_loss, input_power_avg.

    Raises:
        AnalysisError: If traces have incompatible sample rates.

    Example:
        >>> result = osc.power_analysis(v_trace, i_trace)
        >>> print(f"Average: {result['average_power']*1e3:.2f} mW")

    References:
        IEC 61000-4-7, IEEE 1459-2010
    """
    from oscura.analyzers.power.basic import instantaneous_power, power_statistics

    _validate_sample_rates(voltage, current)
    power_trace = instantaneous_power(voltage, current)
    stats = power_statistics(power_trace)

    result = _build_power_result(power_trace, stats)

    if input_voltage is not None and input_current is not None:
        result.update(_calculate_efficiency(input_voltage, input_current, stats["average"]))

    if report is not None:
        _generate_power_report(result, report)

    return result


def _validate_sample_rates(voltage: WaveformTrace, current: WaveformTrace) -> None:
    """Validate that traces have matching sample rates."""
    if voltage.metadata.sample_rate != current.metadata.sample_rate:
        raise AnalysisError(
            f"Sample rate mismatch: {voltage.metadata.sample_rate} vs {current.metadata.sample_rate}"
        )


def _build_power_result(power_trace: WaveformTrace, stats: dict[str, Any]) -> dict[str, Any]:
    """Build power analysis result dictionary."""
    return {
        "power_trace": power_trace,
        "average_power": stats["average"],
        "output_power_avg": stats["average"],
        "output_power_rms": stats["rms"],
        "peak_power": stats["peak"],
        "min_power": stats.get("min", np.min(power_trace.data)),
        "energy": stats["energy"],
        "duration": stats["duration"],
    }


def _calculate_efficiency(
    input_voltage: WaveformTrace, input_current: WaveformTrace, output_power_avg: float
) -> dict[str, float]:
    """Calculate power efficiency metrics."""
    from oscura.analyzers.power.basic import instantaneous_power, power_statistics

    input_power_trace = instantaneous_power(input_voltage, input_current)
    input_stats = power_statistics(input_power_trace)
    input_power_avg = input_stats["average"]

    if input_power_avg > 0:
        efficiency = (output_power_avg / input_power_avg) * 100.0
        power_loss = input_power_avg - output_power_avg
    else:
        efficiency = power_loss = 0.0

    return {"efficiency": efficiency, "power_loss": power_loss, "input_power_avg": input_power_avg}


def _generate_power_report(result: dict[str, Any], output_path: str) -> None:
    """Generate HTML report for power analysis.

    Args:
        result: Power analysis result dictionary.
        output_path: Path to save HTML report.
    """
    html = f"""
    <html>
    <head><title>Power Analysis Report</title></head>
    <body>
    <h1>Power Analysis Report</h1>
    <h2>Power Statistics</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Units</th></tr>
        <tr><td>Average Power</td><td>{result["average_power"] * 1e3:.3f}</td><td>mW</td></tr>
        <tr><td>RMS Power</td><td>{result["output_power_rms"] * 1e3:.3f}</td><td>mW</td></tr>
        <tr><td>Peak Power</td><td>{result["peak_power"] * 1e3:.3f}</td><td>mW</td></tr>
        <tr><td>Total Energy</td><td>{result["energy"] * 1e6:.3f}</td><td>µJ</td></tr>
        <tr><td>Duration</td><td>{result["duration"] * 1e3:.3f}</td><td>ms</td></tr>
    """
    if "efficiency" in result:
        html += f"""
        <tr><td>Efficiency</td><td>{result["efficiency"]:.1f}</td><td>%</td></tr>
        <tr><td>Input Power</td><td>{result["input_power_avg"] * 1e3:.3f}</td><td>mW</td></tr>
        <tr><td>Power Loss</td><td>{result["power_loss"] * 1e3:.3f}</td><td>mW</td></tr>
        """
    html += """
    </table>
    </body>
    </html>
    """
    with open(output_path, "w") as f:
        f.write(html)


__all__ = ["power_analysis"]
