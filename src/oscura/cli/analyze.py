"""Oscura Analyze Command - Full Analysis Workflow.

Provides CLI for running complete analysis workflows on waveform files with
automatic protocol detection, signal characterization, and comprehensive reporting.


Example:
    $ oscura analyze signal.wfm
    $ oscura analyze capture.wfm --protocol uart --export-dir output/
    $ oscura analyze data.wfm --interactive
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

from oscura.cli.main import format_output
from oscura.cli.progress import ProgressReporter
from oscura.sessions.legacy import Session

logger = logging.getLogger("oscura.cli.analyze")


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--protocol",
    type=str,
    default="auto",
    help="Protocol hint (auto for auto-detection).",
)
@click.option(
    "--export-dir",
    type=click.Path(),
    default=None,
    help="Directory to export results (default: ./oscura_output).",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive mode with prompts.",
)
@click.option(
    "--output",
    type=click.Choice(["json", "csv", "html", "table"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--save-session",
    type=click.Path(),
    default=None,
    help="Save analysis session to file (.tks).",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    file: str,
    protocol: str,
    export_dir: str | None,
    interactive: bool,
    output: str,
    save_session: str | None,
) -> None:
    """Run full analysis workflow on waveform file.

    Performs comprehensive analysis including signal characterization,
    protocol detection and decoding, spectral analysis, and generates
    detailed reports.

    Args:
        ctx: Click context object.
        file: Path to waveform file to analyze.
        protocol: Protocol hint or 'auto' for detection.
        export_dir: Directory for exported results.
        interactive: Enable interactive prompts.
        output: Output format.
        save_session: Path to save session file.

    Examples:

        \b
        # Full auto analysis
        $ oscura analyze capture.wfm

        \b
        # With protocol hint and export
        $ oscura analyze signal.wfm \\
            --protocol uart \\
            --export-dir analysis_results/

        \b
        # Interactive mode
        $ oscura analyze data.wfm --interactive
    """
    verbose = ctx.obj.get("verbose", 0)
    quiet = ctx.obj.get("quiet", False)

    if verbose:
        logger.info(f"Analyzing: {file}")

    try:
        results = _perform_analysis_workflow(
            file, protocol, export_dir, interactive, quiet, save_session
        )
        formatted = format_output(results, output)
        click.echo(formatted)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _perform_analysis_workflow(
    file: str,
    protocol: str,
    export_dir: str | None,
    interactive: bool,
    quiet: bool,
    save_session: str | None,
) -> dict[str, Any]:
    """Perform complete analysis workflow."""
    from oscura.loaders import load

    progress = ProgressReporter(quiet=quiet, stages=5)

    # Stage 1: Load file
    progress.start_stage("Loading file")
    trace = load(file)
    progress.complete_stage()

    # Stage 2: Signal characterization
    progress.start_stage("Characterizing signal")
    signal_char = _characterize_signal(trace)
    progress.complete_stage()

    # Stage 3: Protocol detection
    progress.start_stage("Detecting protocol")
    protocol_info = _detect_and_prepare_protocol(trace, protocol, interactive)
    progress.complete_stage()

    # Stage 4: Protocol decoding
    progress.start_stage("Decoding protocol")
    decoded = _decode_protocol(trace, protocol_info["protocol"])
    progress.complete_stage()

    # Stage 5: Generate report
    progress.start_stage("Generating report")
    results = _build_analysis_results(
        file, signal_char, protocol_info, decoded, export_dir, save_session, trace
    )
    progress.complete_stage()
    progress.finish()

    return results


def _detect_and_prepare_protocol(trace: Any, protocol: str, interactive: bool) -> dict[str, Any]:
    """Detect and prepare protocol information."""
    if protocol == "auto":
        detected = _detect_protocol(trace, interactive=interactive)
        protocol = detected["protocol"]
    return {"protocol": protocol, "auto_detected": protocol == "auto"}


def _build_analysis_results(
    file: str,
    signal_char: dict[str, Any],
    protocol_info: dict[str, Any],
    decoded: dict[str, Any],
    export_dir: str | None,
    save_session: str | None,
    trace: Any,
) -> dict[str, Any]:
    """Build final analysis results with optional exports."""
    results = {
        "file": str(Path(file).name),
        **signal_char,
        **protocol_info,
        **decoded,
    }

    if export_dir:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        _export_results(results, export_path)
        results["export_dir"] = str(export_path)

    if save_session:
        session = Session(name=Path(file).stem)
        session.add_trace("main", trace)
        session.metadata["analysis_results"] = results
        session.save(save_session)
        results["session_file"] = save_session

    return results


def _characterize_signal(trace: Any) -> dict[str, Any]:
    """Characterize signal properties.

    Args:
        trace: Waveform trace to characterize.

    Returns:
        Dictionary of signal characteristics.
    """
    import numpy as np

    from oscura.analyzers.waveform.measurements import fall_time, rise_time

    data = trace.data
    sample_rate = trace.metadata.sample_rate

    rt = rise_time(trace)
    ft = fall_time(trace)

    return {
        "sample_rate": f"{sample_rate / 1e6:.1f} MHz",
        "samples": len(data),
        "duration": f"{len(data) / sample_rate * 1e3:.3f} ms",
        "amplitude": f"{float(data.max() - data.min()):.3f} V",
        "rise_time": f"{rt * 1e9:.2f} ns" if not np.isnan(rt) else "N/A",
        "fall_time": f"{ft * 1e9:.2f} ns" if not np.isnan(ft) else "N/A",
    }


def _detect_protocol(trace: Any, interactive: bool = False) -> dict[str, Any]:
    """Detect protocol from trace.

    Args:
        trace: Trace to analyze.
        interactive: If True, prompt for confirmation.

    Returns:
        Detection results.
    """
    from oscura.inference.protocol import detect_protocol

    detection = detect_protocol(trace, min_confidence=0.5, return_candidates=True)

    if interactive and detection.get("confidence", 0) < 0.9:
        click.echo(f"\nDetected protocol: {detection['protocol']}")
        click.echo(f"Confidence: {detection['confidence']:.1%}")

        if not click.confirm("Use this protocol?", default=True):
            # Show candidates
            candidates = detection.get("candidates", [])
            if candidates:
                click.echo("\nOther candidates:")
                for i, cand in enumerate(candidates[:5], 1):
                    click.echo(f"  {i}. {cand['protocol']} ({cand['confidence']:.1%})")

                choice = click.prompt("Select protocol (1-5, or 0 for manual)", type=int, default=1)
                if 1 <= choice <= len(candidates):
                    detection = candidates[choice - 1]
                elif choice == 0:
                    manual = click.prompt("Enter protocol name", type=str)
                    detection = {"protocol": manual, "confidence": 1.0}

    return detection


def _decode_protocol(trace: Any, protocol: str) -> dict[str, Any]:
    """Decode protocol from trace.

    Args:
        trace: Trace to decode.
        protocol: Protocol name.

    Returns:
        Decoding results.
    """
    from oscura.core.types import DigitalTrace

    # Convert to digital if needed
    if not isinstance(trace, DigitalTrace):
        import numpy as np

        threshold = (np.max(trace.data) + np.min(trace.data)) / 2
        digital_data = trace.data > threshold
        digital_trace = DigitalTrace(data=digital_data, metadata=trace.metadata)
    else:
        digital_trace = trace

    # Decode based on protocol
    packets: list[Any] = []
    if protocol.lower() == "uart":
        from oscura.analyzers.protocols.uart import UARTDecoder

        uart_decoder = UARTDecoder(baudrate=0)  # Auto-detect
        packets = list(uart_decoder.decode(digital_trace))
    elif protocol.lower() == "spi":
        from oscura.analyzers.protocols.spi import SPIDecoder

        spi_decoder = SPIDecoder()
        packets = list(
            spi_decoder.decode(
                clk=digital_trace.data,
                mosi=digital_trace.data,
                sample_rate=trace.metadata.sample_rate,
            )
        )
    elif protocol.lower() == "i2c":
        from oscura.analyzers.protocols.i2c import I2CDecoder

        i2c_decoder = I2CDecoder()
        packets = list(
            i2c_decoder.decode(
                scl=digital_trace.data,
                sda=digital_trace.data,
                sample_rate=trace.metadata.sample_rate,
            )
        )

    return {
        "packets_decoded": len(packets),
        "errors": sum(1 for p in packets if p.errors),
    }


def _export_results(results: dict[str, Any], export_dir: Path) -> None:
    """Export results to directory.

    Args:
        results: Results dictionary.
        export_dir: Export directory path.
    """
    import json

    # Export JSON
    json_path = export_dir / "analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Export HTML report
    html_path = export_dir / "analysis_report.html"
    html_content = format_output(results, "html")
    with open(html_path, "w") as f:
        f.write(html_content)
