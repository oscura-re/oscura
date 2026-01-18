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

    Analyzes power consumption from voltage and current measurements:
    - Instantaneous power calculation
    - Average, RMS, and peak power
    - Energy consumption
    - Efficiency (if input power provided)
    - Power profile generation

    Args:
        voltage: Output voltage trace.
        current: Output current trace.
        input_voltage: Optional input voltage for efficiency calculation.
        input_current: Optional input current for efficiency calculation.
        report: Optional path to save HTML report.

    Returns:
        Dictionary containing:
        - power_trace: WaveformTrace of instantaneous power P(t)
        - average_power: Mean power in watts
        - output_power_avg: Average output power (same as average_power)
        - output_power_rms: RMS output power in watts
        - peak_power: Maximum power in watts
        - min_power: Minimum power in watts
        - energy: Total energy in joules
        - duration: Measurement duration in seconds
        - efficiency: Efficiency percentage (if input provided)
        - power_loss: Power loss in watts (if input provided)
        - input_power_avg: Average input power (if input provided)

    Raises:
        AnalysisError: If traces have incompatible sample rates or lengths.

    Example:
        >>> voltage = osc.load('vout.wfm')
        >>> current = osc.load('iout.wfm')
        >>> result = osc.power_analysis(voltage, current)
        >>> print(f"Average: {result['average_power']*1e3:.2f} mW")
        >>> print(f"Peak: {result['peak_power']*1e3:.2f} mW")
        >>> print(f"Energy: {result['energy']*1e6:.2f} µJ")

    References:
        IEC 61000-4-7: Harmonics and interharmonics measurement
        IEEE 1459-2010: Definitions for measurement of electric power
    """
    # Import power analysis functions
    from oscura.analyzers.power.basic import (
        instantaneous_power,
        power_statistics,
    )

    # Validate traces
    if voltage.metadata.sample_rate != current.metadata.sample_rate:
        # Would need interpolation in real implementation
        raise AnalysisError(
            "Voltage and current traces must have same sample rate. "
            f"Got {voltage.metadata.sample_rate} and {current.metadata.sample_rate}"
        )

    # Calculate instantaneous power
    power_trace = instantaneous_power(voltage, current)

    # Calculate power statistics
    stats = power_statistics(power_trace)

    # Build result with output power
    result = {
        "power_trace": power_trace,
        "average_power": stats["average"],
        "output_power_avg": stats["average"],
        "output_power_rms": stats["rms"],
        "peak_power": stats["peak"],
        "min_power": stats.get("min", np.min(power_trace.data)),
        "energy": stats["energy"],
        "duration": stats["duration"],
    }

    # Calculate efficiency if input provided
    if input_voltage is not None and input_current is not None:
        input_power_trace = instantaneous_power(input_voltage, input_current)
        input_stats = power_statistics(input_power_trace)

        input_power_avg = input_stats["average"]
        output_power_avg = stats["average"]

        if input_power_avg > 0:
            efficiency = (output_power_avg / input_power_avg) * 100.0
            power_loss = input_power_avg - output_power_avg
        else:
            efficiency = 0.0
            power_loss = 0.0

        result["efficiency"] = efficiency
        result["power_loss"] = power_loss
        result["input_power_avg"] = input_power_avg

    # Generate report if requested
    if report is not None:
        _generate_power_report(result, report)

    return result


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
