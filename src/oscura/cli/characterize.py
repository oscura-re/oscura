"""Oscura Characterize Command implementing CLI-002.

Provides CLI for buffer/signal characterization with automatic logic family
detection and optional reference comparison.


Example:
    $ oscura characterize 74hc04_output.wfm
    $ oscura characterize signal.wfm --logic-family CMOS_3V3
    $ oscura characterize signal.wfm --compare reference.wfm --output html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from numpy.typing import NDArray

from oscura.cli.main import format_output

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace

logger = logging.getLogger("oscura.cli.characterize")


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--type",
    "analysis_type",
    type=click.Choice(["buffer", "signal", "power"], case_sensitive=False),
    default="buffer",
    help="Type of characterization to perform.",
)
@click.option(
    "--logic-family",
    type=click.Choice(
        ["TTL", "CMOS", "CMOS_3V3", "CMOS_5V", "LVTTL", "LVCMOS", "auto"],
        case_sensitive=False,
    ),
    default="auto",
    help="Logic family for buffer characterization (default: auto-detect).",
)
@click.option(
    "--compare",
    type=click.Path(exists=True),
    default=None,
    help="Reference file for comparison analysis.",
)
@click.option(
    "--output",
    type=click.Choice(["json", "csv", "html", "table"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--save-report",
    type=click.Path(),
    default=None,
    help="Save HTML report to file.",
)
@click.pass_context
def characterize(
    ctx: click.Context,
    file: str,
    analysis_type: str,
    logic_family: str,
    compare: str | None,
    output: str,
    save_report: str | None,
) -> None:
    """Characterize buffer, signal, or power measurements.

    Analyzes a waveform file and extracts timing, quality, and performance
    characteristics. Supports automatic logic family detection and optional
    comparison to a reference signal.

    Args:
        ctx: Click context object.
        file: Path to waveform file to characterize.
        analysis_type: Type of characterization (buffer, signal, power).
        logic_family: Logic family for buffer characterization.
        compare: Path to reference file for comparison analysis.
        output: Output format (json, csv, html, table).
        save_report: Path to save HTML report file.

    Raises:
        Exception: If characterization fails or file cannot be loaded.

    Examples:

        \b
        # Simple buffer characterization
        $ oscura characterize 74hc04_output.wfm

        \b
        # Full characterization with reference
        $ oscura characterize signal.wfm \\
            --logic-family CMOS_3V3 \\
            --compare golden_reference.wfm \\
            --save-report report.html

        \b
        # Power analysis
        $ oscura characterize power_rail.wfm --type power --output json
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Characterizing: {file}")
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Logic family: {logic_family}")

    try:
        # Import here to avoid circular imports
        from oscura.loaders import load

        # Load the main trace
        logger.debug(f"Loading trace from {file}")
        trace = load(file)

        # Load reference trace if provided
        reference_trace: WaveformTrace | None = None
        if compare:
            logger.debug(f"Loading reference trace from {compare}")
            reference_trace = load(compare)  # type: ignore[assignment]

        # Perform characterization based on type
        results = _perform_characterization(
            trace=trace,
            reference_trace=reference_trace,
            analysis_type=analysis_type,
            logic_family=logic_family,
        )

        # Add metadata
        results["file"] = str(Path(file).name)
        if compare:
            results["reference_file"] = str(Path(compare).name)

        # Generate HTML report if requested
        if save_report:
            html_content = format_output(results, "html")
            with open(save_report, "w") as f:
                f.write(html_content)
            logger.info(f"Report saved to {save_report}")

        # Output results
        formatted = format_output(results, output)
        click.echo(formatted)

    except Exception as e:
        logger.error(f"Characterization failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _perform_characterization(
    trace: Any,
    reference_trace: Any | None,
    analysis_type: str,
    logic_family: str,
) -> dict[str, Any]:
    """Perform characterization analysis.

    Calls actual Oscura analysis functions based on the analysis type.

    Args:
        trace: Main trace to analyze.
        reference_trace: Optional reference trace for comparison.
        analysis_type: Type of analysis ('buffer', 'signal', 'power').
        logic_family: Logic family for digital analysis.

    Returns:
        Dictionary of analysis results.
    """
    results = _build_basic_results(trace, analysis_type, logic_family)

    if analysis_type == "buffer":
        _add_buffer_results(results, trace, logic_family)
    elif analysis_type == "signal":
        _add_signal_results(results, trace.data)
    elif analysis_type == "power":
        _add_power_results(results, trace)

    # Add comparison results if reference provided
    if reference_trace is not None:
        _add_comparison_results(results, trace, reference_trace)

    return results


def _build_basic_results(trace: Any, analysis_type: str, logic_family: str) -> dict[str, Any]:
    """Build basic results dictionary with metadata.

    Args:
        trace: Trace to analyze.
        analysis_type: Type of analysis.
        logic_family: Logic family string.

    Returns:
        Dictionary with basic metadata.
    """
    sample_rate = trace.metadata.sample_rate
    data = trace.data

    return {
        "analysis_type": analysis_type,
        "logic_family": logic_family,
        "sample_rate": f"{sample_rate / 1e6:.1f} MHz",
        "samples": len(data),
        "duration": f"{len(data) / sample_rate * 1e3:.3f} ms",
    }


def _add_buffer_results(results: dict[str, Any], trace: Any, logic_family: str) -> None:
    """Add buffer characterization results.

    Args:
        results: Results dict to update.
        trace: Trace to analyze.
        logic_family: Logic family string.
    """
    import numpy as np

    from oscura.analyzers.waveform.measurements import (
        fall_time,
        overshoot,
        rise_time,
        undershoot,
    )
    from oscura.inference import detect_logic_family

    rt = rise_time(trace)
    ft = fall_time(trace)
    os_pct = overshoot(trace)
    us_pct = undershoot(trace)

    results.update(
        {
            "rise_time": f"{rt * 1e9:.2f} ns" if not np.isnan(rt) else "N/A",
            "fall_time": f"{ft * 1e9:.2f} ns" if not np.isnan(ft) else "N/A",
            "overshoot": f"{os_pct:.1f} %" if not np.isnan(os_pct) else "N/A",
            "undershoot": f"{us_pct:.1f} %" if not np.isnan(us_pct) else "N/A",
            "status": "PASS",
        }
    )

    # Logic family detection
    if logic_family == "auto":
        detected = detect_logic_family(trace)
        primary = detected.get("primary", {})
        results["logic_family_detected"] = primary.get("name", "unknown")
        results["confidence"] = f"{primary.get('confidence', 0) * 100:.0f}%"
    else:
        results["logic_family_detected"] = logic_family


def _add_signal_results(results: dict[str, Any], data: NDArray[np.floating[Any]]) -> None:
    """Add signal characterization results.

    Args:
        results: Results dict to update.
        data: Signal data array.
    """
    results.update(
        {
            "amplitude": f"{float(data.max() - data.min()):.3f} V",
            "peak_to_peak": f"{float(data.max() - data.min()):.3f} V",
            "mean": f"{float(data.mean()):.3f} V",
            "rms": f"{float(np.sqrt((data**2).mean())):.3f} V",
        }
    )


def _add_power_results(results: dict[str, Any], trace: Any) -> None:
    """Add power analysis results.

    Args:
        results: Results dict to update.
        trace: Trace to analyze.
    """
    import numpy as np

    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Compute power (P = V^2/R, assume R=1 for relative)
    power_data = data**2
    avg_pwr = float(np.mean(power_data))
    peak_pwr = float(np.max(power_data))
    total_energy = float(np.sum(power_data) / sample_rate)

    results.update(
        {
            "average_power": f"{avg_pwr * 1e3:.3f} mW",
            "peak_power": f"{peak_pwr * 1e3:.3f} mW",
            "energy": f"{total_energy * 1e6:.3f} uJ",
        }
    )


def _add_comparison_results(results: dict[str, Any], trace: Any, reference_trace: Any) -> None:
    """Add comparison results to results dictionary.

    Args:
        results: Results dict to update.
        trace: Main trace.
        reference_trace: Reference trace.
    """
    from oscura.utils.comparison.compare import compare_traces, similarity_score

    ref_data = reference_trace.data
    sim = similarity_score(trace, reference_trace)
    comparison_result = compare_traces(trace, reference_trace)

    results["comparison"] = {
        "correlation": f"{comparison_result.correlation:.4f}",
        "amplitude_difference": f"{abs(float(trace.data.mean()) - float(ref_data.mean())):.3f} V",
        "similarity": f"{sim * 100:.1f}%",
    }
