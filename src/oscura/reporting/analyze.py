"""Comprehensive analysis report system main entry point.

This module provides the primary `analyze()` function for running
comprehensive analysis on any supported input data type.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oscura.reporting.config import (
    AnalysisConfig,
    AnalysisDomain,
    AnalysisError,
    AnalysisResult,
    InputType,
    ProgressInfo,
    get_available_analyses,
)
from oscura.reporting.output import OutputManager

if TYPE_CHECKING:
    from oscura.core.types import Trace

logger = logging.getLogger(__name__)


class UnsupportedFormatError(Exception):
    """Raised when input file format is not recognized."""


def analyze(
    input_path: str | Path | None = None,
    data: Trace | bytes | list[Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    config: AnalysisConfig | None = None,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
) -> AnalysisResult:
    """Run comprehensive analysis on data.

    Provide EITHER input_path (file) OR data (in-memory), not both.

    Args:
        input_path: Path to input data file (any supported format).
        data: In-memory data (Trace, bytes, list of packets).
        output_dir: Base directory for output. Default: input file's directory
                    or current directory for in-memory data.
        config: Analysis configuration. Default: analyze all applicable domains.
        progress_callback: Called with progress updates during analysis.

    Returns:
        AnalysisResult with paths to all outputs and summary statistics.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If neither or both input_path and data are provided.

    Examples:
        # From file
        result = analyze("capture.wfm")
        print(result.output_dir)  # 20260101_120000_capture_analysis/

        # From in-memory data
        result = analyze(data=my_waveform_trace, output_dir="/reports")

        # With configuration
        config = AnalysisConfig(domains=[AnalysisDomain.SPECTRAL])
        result = analyze("capture.wfm", config=config)

        # With progress callback
        def on_progress(info):
            print(f"{info.domain}: {info.percent}%")
        result = analyze("capture.wfm", progress_callback=on_progress)
    """
    # Validate inputs
    if input_path is None and data is None:
        raise ValueError("Either input_path or data must be provided")
    if input_path is not None and data is not None:
        raise ValueError("Provide input_path OR data, not both")

    # Use default config if not provided
    if config is None:
        config = AnalysisConfig()

    # Track timing
    start_time = time.time()

    # Determine input name and type
    if input_path is not None:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_name = input_path.stem
        input_type = _detect_input_type_from_file(input_path)
        loaded_data = _load_input_file(input_path, input_type)
    else:
        input_name = "memory_data"
        input_type = _detect_input_type_from_data(data)
        loaded_data = data

    # Determine output directory
    if output_dir is None:
        if input_path is not None:
            base_dir = input_path.parent
        else:
            base_dir = Path.cwd()
    else:
        base_dir = Path(output_dir)

    # Create output manager with timestamp
    timestamp = datetime.now()
    output_manager = OutputManager(base_dir, input_name, timestamp)
    output_manager.create()

    # Report progress: starting
    _report_progress(
        progress_callback,
        phase="initializing",
        domain=None,
        function=None,
        percent=0.0,
        message="Initializing analysis",
        elapsed=time.time() - start_time,
    )

    # Determine applicable domains
    applicable_domains = get_available_analyses(input_type)
    enabled_domains = [d for d in applicable_domains if config.is_domain_enabled(d)]

    logger.info(f"Running analysis on {input_name} ({input_type.value})")
    logger.info(f"Enabled domains: {[d.value for d in enabled_domains]}")

    # Execute analysis engine
    from oscura.reporting.engine import AnalysisEngine

    engine = AnalysisEngine(config)
    engine_result = engine.run(
        input_path=input_path,
        data=loaded_data,
        progress_callback=progress_callback,
    )

    # Generate plots
    plot_paths: list[Path] = []
    if config.generate_plots:
        _report_progress(
            progress_callback,
            phase="plotting",
            domain=None,
            function=None,
            percent=70.0,
            message="Generating visualizations",
            elapsed=time.time() - start_time,
        )

        from oscura.reporting.plots import PlotGenerator

        plot_gen = PlotGenerator(config)
        for domain, results in engine_result["results"].items():
            domain_plots = plot_gen.generate_plots(domain, results, output_manager)
            plot_paths.extend(domain_plots)

    # Save data outputs
    _report_progress(
        progress_callback,
        phase="saving",
        domain=None,
        function=None,
        percent=85.0,
        message="Saving analysis results",
        elapsed=time.time() - start_time,
    )

    # Save summary data
    summary_data = {
        "input": {
            "name": input_name,
            "type": input_type.value,
            "path": str(input_path) if input_path else None,
        },
        "timestamp": timestamp.isoformat(),
        "duration_seconds": time.time() - start_time,
        "stats": engine_result["stats"],
        "domains": {d.value: r for d, r in engine_result["results"].items()},
    }

    summary_json = output_manager.save_json("summary", summary_data)
    summary_yaml = None
    if "yaml" in config.output_formats:
        summary_yaml = output_manager.save_yaml("summary", summary_data)

    # Save metadata
    metadata = {
        "oscura_version": _get_version(),
        "analysis_version": "2.0",
        "timestamp": timestamp.isoformat(),
        "input_file": str(input_path) if input_path else None,
        "input_type": input_type.value,
        "duration_seconds": time.time() - start_time,
        "total_analyses": engine_result["stats"]["total_analyses"],
        "successful": engine_result["stats"]["successful_analyses"],
        "failed": engine_result["stats"]["failed_analyses"],
        "skipped": engine_result["stats"].get("skipped_analyses", 0),
    }
    metadata_json = output_manager.save_json("metadata", metadata)

    # Save configuration
    config_data = {
        "domains": [d.value for d in enabled_domains],
        "generate_plots": config.generate_plots,
        "plot_format": config.plot_format,
        "plot_dpi": config.plot_dpi,
        "output_formats": config.output_formats,
        "index_formats": config.index_formats,
    }
    config_yaml = output_manager.save_yaml("config", config_data)

    # Save domain results
    domain_dirs: dict[AnalysisDomain, Path] = {}
    for domain, results in engine_result["results"].items():
        domain_dir = output_manager.create_domain_dir(domain)
        domain_dirs[domain] = domain_dir
        output_manager.save_json("results", results, subdir=domain.value)

    # Save errors if any
    error_log: Path | None = None
    errors: list[AnalysisError] = engine_result["errors"]
    if errors:
        error_list = [
            {
                "domain": e.domain.value,
                "function": e.function,
                "error_type": e.error_type,
                "error_message": e.error_message,
                "duration_ms": e.duration_ms,
            }
            for e in errors
        ]
        error_data = {"errors": error_list, "count": len(error_list)}
        error_log = output_manager.save_json("failed_analyses", error_data, subdir="errors")

    # Build AnalysisResult for index generation
    partial_result = AnalysisResult(
        output_dir=output_manager.root,
        index_html=None,
        index_md=None,
        index_pdf=None,
        summary_json=summary_json,
        summary_yaml=summary_yaml,
        metadata_json=metadata_json,
        config_yaml=config_yaml,
        domain_dirs=domain_dirs,
        plot_paths=plot_paths,
        error_log=error_log,
        input_file=str(input_path) if input_path else None,
        input_type=input_type,
        total_analyses=engine_result["stats"]["total_analyses"],
        successful_analyses=engine_result["stats"]["successful_analyses"],
        failed_analyses=engine_result["stats"]["failed_analyses"],
        skipped_analyses=engine_result["stats"].get("skipped_analyses", 0),
        duration_seconds=time.time() - start_time,
        domain_summaries=engine_result["results"],
        errors=errors,
    )

    # Generate index files
    _report_progress(
        progress_callback,
        phase="indexing",
        domain=None,
        function=None,
        percent=95.0,
        message="Generating index files",
        elapsed=time.time() - start_time,
    )

    from oscura.reporting.index import IndexGenerator

    index_gen = IndexGenerator(output_manager)
    index_paths = index_gen.generate(partial_result, config.index_formats)

    # Complete result
    result = AnalysisResult(
        output_dir=output_manager.root,
        index_html=index_paths.get("html"),
        index_md=index_paths.get("md"),
        index_pdf=index_paths.get("pdf"),
        summary_json=summary_json,
        summary_yaml=summary_yaml,
        metadata_json=metadata_json,
        config_yaml=config_yaml,
        domain_dirs=domain_dirs,
        plot_paths=plot_paths,
        error_log=error_log,
        input_file=str(input_path) if input_path else None,
        input_type=input_type,
        total_analyses=engine_result["stats"]["total_analyses"],
        successful_analyses=engine_result["stats"]["successful_analyses"],
        failed_analyses=engine_result["stats"]["failed_analyses"],
        skipped_analyses=engine_result["stats"].get("skipped_analyses", 0),
        duration_seconds=time.time() - start_time,
        domain_summaries=engine_result["results"],
        errors=errors,
    )

    # Report completion
    _report_progress(
        progress_callback,
        phase="complete",
        domain=None,
        function=None,
        percent=100.0,
        message=f"Analysis complete: {result.successful_analyses}/{result.total_analyses} successful",
        elapsed=time.time() - start_time,
    )

    logger.info(f"Analysis complete. Output: {result.output_dir}")
    return result


def _detect_input_type_from_file(path: Path) -> InputType:
    """Detect input type from file extension."""
    suffix = path.suffix.lower()

    waveform_extensions = {".wfm", ".csv", ".npz", ".hdf5", ".h5", ".wav", ".tdms"}
    digital_extensions = {".vcd", ".sr"}
    binary_extensions = {".bin", ".raw"}
    pcap_extensions = {".pcap", ".pcapng"}
    sparams_extensions = {".s1p", ".s2p", ".s3p", ".s4p", ".s5p", ".s6p", ".s7p", ".s8p"}

    if suffix in waveform_extensions:
        return InputType.WAVEFORM
    elif suffix in digital_extensions:
        return InputType.DIGITAL
    elif suffix in binary_extensions:
        return InputType.BINARY
    elif suffix in pcap_extensions:
        return InputType.PCAP
    elif suffix in sparams_extensions:
        return InputType.SPARAMS
    else:
        raise UnsupportedFormatError(f"Unsupported file format: {suffix}")


def _detect_input_type_from_data(data: Any) -> InputType:
    """Detect input type from in-memory data."""
    # Check for Trace object (time + voltage = waveform)
    # Check this BEFORE SParameterData to avoid MagicMock false positives
    if hasattr(data, "time") and hasattr(data, "voltage"):
        # Verify these are not just mock/placeholder attributes
        try:
            _ = data.time
            _ = data.voltage
            return InputType.WAVEFORM
        except (AttributeError, TypeError):
            pass

    # Check for SParameterData
    if hasattr(data, "s_matrix") and hasattr(data, "frequencies"):
        return InputType.SPARAMS

    # Check for bytes
    if isinstance(data, bytes | bytearray):
        return InputType.BINARY

    # Check for list of packets
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if hasattr(first, "timestamp") or isinstance(first, dict):
            return InputType.PACKETS

    # Default to waveform
    return InputType.WAVEFORM


def _load_input_file(path: Path, input_type: InputType) -> Any:
    """Load input file based on type."""
    try:
        from oscura.loaders import load

        if input_type == InputType.WAVEFORM:
            return load(path)
        elif input_type == InputType.DIGITAL:
            # Use VCD/SR loader
            from oscura.loaders.vcd import load_vcd

            return load_vcd(path)
        elif input_type == InputType.BINARY:
            return path.read_bytes()
        elif input_type == InputType.PCAP:
            from oscura.loaders.pcap import load_pcap

            return load_pcap(path)
        elif input_type == InputType.SPARAMS:
            from oscura.analyzers.signal_integrity.sparams import load_touchstone

            return load_touchstone(path)
        else:
            return load(path)
    except ImportError as e:
        logger.warning(f"Loader not available: {e}")
        # Fall back to raw bytes
        return path.read_bytes()


def _report_progress(
    callback: Callable[[ProgressInfo], None] | None,
    phase: str,
    domain: AnalysisDomain | None,
    function: str | None,
    percent: float,
    message: str,
    elapsed: float,
) -> None:
    """Report progress to callback if provided."""
    if callback is not None:
        info = ProgressInfo(
            phase=phase,
            domain=domain,
            function=function,
            percent=percent,
            message=message,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=None,
        )
        callback(info)


def _get_version() -> str:
    """Get Oscura version."""
    try:
        from oscura import __version__

        return __version__
    except ImportError:
        return "unknown"


__all__ = [
    "UnsupportedFormatError",
    "analyze",
]
