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
        output_dir: Base directory for output.
        config: Analysis configuration. Default: analyze all applicable domains.
        progress_callback: Called with progress updates during analysis.

    Returns:
        AnalysisResult with paths to all outputs and summary statistics.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If neither or both input_path and data are provided.

    Examples:
        >>> result = analyze("capture.wfm")
        >>> result = analyze(data=my_trace, output_dir="/reports")
        >>> config = AnalysisConfig(domains=[AnalysisDomain.SPECTRAL])
        >>> result = analyze("capture.wfm", config=config)
    """
    config = config or AnalysisConfig()
    _validate_inputs(input_path, data)
    start_time = time.time()

    # Setup and load data
    output_manager, input_name, input_type, loaded_data, resolved_path = _setup_analysis(
        input_path, data, output_dir, progress_callback, start_time
    )

    # Run analysis pipeline
    engine_result, plot_paths, saved_paths = _run_analysis_pipeline(
        config,
        resolved_path,
        loaded_data,
        input_name,
        input_type,
        output_manager,
        progress_callback,
        start_time,
    )

    # Build and finalize result
    result = _finalize_result(
        output_manager,
        resolved_path,
        input_type,
        engine_result,
        saved_paths,
        plot_paths,
        config,
        progress_callback,
        start_time,
    )

    logger.info(f"Analysis complete. Output: {result.output_dir}")
    return result


def _run_analysis_pipeline(
    config: AnalysisConfig,
    resolved_path: Path | None,
    loaded_data: Any,
    input_name: str,
    input_type: InputType,
    output_manager: OutputManager,
    progress_callback: Callable[[ProgressInfo], None] | None,
    start_time: float,
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    """Run complete analysis pipeline."""
    engine_result = _run_analysis_engine(
        config, resolved_path, loaded_data, input_name, input_type, progress_callback
    )
    plot_paths = _generate_plots(
        config, engine_result, output_manager, progress_callback, start_time
    )
    saved_paths = _save_all_outputs(
        output_manager,
        input_name,
        input_type,
        resolved_path,
        datetime.now(),
        engine_result,
        config,
        start_time,
    )
    return engine_result, plot_paths, saved_paths


def _finalize_result(
    output_manager: OutputManager,
    resolved_path: Path | None,
    input_type: InputType,
    engine_result: dict[str, Any],
    saved_paths: dict[str, Any],
    plot_paths: list[Path],
    config: AnalysisConfig,
    progress_callback: Callable[[ProgressInfo], None] | None,
    start_time: float,
) -> AnalysisResult:
    """Build and finalize analysis result."""
    partial_result = _build_result(
        output_manager,
        resolved_path,
        input_type,
        engine_result,
        saved_paths["summary_json"],
        saved_paths["summary_yaml"],
        saved_paths["metadata_json"],
        saved_paths["config_yaml"],
        saved_paths["domain_dirs"],
        plot_paths,
        saved_paths["error_log"],
        start_time,
    )
    index_paths = _generate_index(
        output_manager, partial_result, config, progress_callback, start_time
    )
    result = _build_final_result(partial_result, index_paths, time.time() - start_time)

    _report_progress(
        progress_callback,
        "complete",
        None,
        None,
        100.0,
        f"Analysis complete: {result.successful_analyses}/{result.total_analyses} successful",
        time.time() - start_time,
    )
    return result


def _setup_analysis(
    input_path: str | Path | None,
    data: Trace | bytes | list[Any] | None,
    output_dir: str | Path | None,
    progress_callback: Callable[[ProgressInfo], None] | None,
    start_time: float,
) -> tuple[OutputManager, str, InputType, Any, Path | None]:
    """Setup analysis environment and load data."""
    input_name, input_type, loaded_data, resolved_path = _prepare_input(input_path, data)
    base_dir = _determine_output_dir(resolved_path, output_dir)
    output_manager = OutputManager(base_dir, input_name, datetime.now())
    output_manager.create()
    _report_progress(
        progress_callback,
        "initializing",
        None,
        None,
        0.0,
        "Initializing analysis",
        time.time() - start_time,
    )
    return output_manager, input_name, input_type, loaded_data, resolved_path


def _save_all_outputs(
    output_manager: OutputManager,
    input_name: str,
    input_type: InputType,
    resolved_path: Path | None,
    timestamp: datetime,
    engine_result: Any,
    config: AnalysisConfig,
    start_time: float,
) -> dict[str, Any]:
    """Save all analysis outputs and return paths."""
    summary_json, summary_yaml = _save_summary(
        output_manager,
        input_name,
        input_type,
        resolved_path,
        timestamp,
        engine_result,
        config,
        start_time,
    )
    return {
        "summary_json": summary_json,
        "summary_yaml": summary_yaml,
        "metadata_json": _save_metadata(
            output_manager, resolved_path, input_type, timestamp, engine_result, start_time
        ),
        "config_yaml": _save_config(output_manager, config, input_type),
        "domain_dirs": _save_domain_results(output_manager, engine_result),
        "error_log": _save_errors(output_manager, engine_result),
    }


def _validate_inputs(input_path: str | Path | None, data: Trace | bytes | list[Any] | None) -> None:
    """Validate that exactly one input source is provided."""
    if input_path is None and data is None:
        raise ValueError("Either input_path or data must be provided")
    if input_path is not None and data is not None:
        raise ValueError("Provide input_path OR data, not both")


def _prepare_input(
    input_path: str | Path | None, data: Trace | bytes | list[Any] | None
) -> tuple[str, InputType, Any, Path | None]:
    """Prepare input data and determine type.

    Returns:
        Tuple of (input_name, input_type, loaded_data, resolved_path)
    """
    if input_path is not None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        input_name = path.stem
        input_type = _detect_input_type_from_file(path)
        loaded_data = _load_input_file(path, input_type)
        return input_name, input_type, loaded_data, path
    else:
        input_name = "memory_data"
        input_type = _detect_input_type_from_data(data)
        loaded_data = data
        return input_name, input_type, loaded_data, None


def _determine_output_dir(input_path: Path | None, output_dir: str | Path | None) -> Path:
    """Determine output directory based on input path or override."""
    if output_dir is not None:
        return Path(output_dir)
    if input_path is not None:
        return input_path.parent
    return Path.cwd()


def _run_analysis_engine(
    config: AnalysisConfig,
    input_path: Path | None,
    loaded_data: Any,
    input_name: str,
    input_type: InputType,
    progress_callback: Callable[[ProgressInfo], None] | None,
) -> dict[str, Any]:
    """Run analysis engine on data."""
    applicable_domains = get_available_analyses(input_type)
    enabled_domains = [d for d in applicable_domains if config.is_domain_enabled(d)]

    logger.info(f"Running analysis on {input_name} ({input_type.value})")
    logger.info(f"Enabled domains: {[d.value for d in enabled_domains]}")

    from oscura.reporting.engine import AnalysisEngine

    engine = AnalysisEngine(config)
    return engine.run(
        input_path=input_path,
        data=loaded_data,
        progress_callback=progress_callback,
    )


def _generate_plots(
    config: AnalysisConfig,
    engine_result: dict[str, Any],
    output_manager: OutputManager,
    progress_callback: Callable[[ProgressInfo], None] | None,
    start_time: float,
) -> list[Path]:
    """Generate plots if configured."""
    plot_paths: list[Path] = []

    if not config.generate_plots:
        return plot_paths

    _report_progress(
        progress_callback,
        "plotting",
        None,
        None,
        70.0,
        "Generating visualizations",
        time.time() - start_time,
    )

    from oscura.reporting.plots import PlotGenerator

    plot_gen = PlotGenerator(config)
    for domain, results in engine_result["results"].items():
        domain_plots = plot_gen.generate_plots(domain, results, output_manager)
        plot_paths.extend(domain_plots)

    return plot_paths


def _save_summary(
    output_manager: OutputManager,
    input_name: str,
    input_type: InputType,
    input_path: Path | None,
    timestamp: datetime,
    engine_result: dict[str, Any],
    config: AnalysisConfig,
    start_time: float,
) -> tuple[Path, Path | None]:
    """Save summary data."""
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

    return summary_json, summary_yaml


def _save_metadata(
    output_manager: OutputManager,
    input_path: Path | None,
    input_type: InputType,
    timestamp: datetime,
    engine_result: dict[str, Any],
    start_time: float,
) -> Path:
    """Save metadata."""
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
    return output_manager.save_json("metadata", metadata)


def _save_config(
    output_manager: OutputManager, config: AnalysisConfig, input_type: InputType
) -> Path:
    """Save configuration."""
    applicable_domains = get_available_analyses(input_type)
    enabled_domains = [d for d in applicable_domains if config.is_domain_enabled(d)]

    config_data = {
        "domains": [d.value for d in enabled_domains],
        "generate_plots": config.generate_plots,
        "plot_format": config.plot_format,
        "plot_dpi": config.plot_dpi,
        "output_formats": config.output_formats,
        "index_formats": config.index_formats,
    }
    return output_manager.save_yaml("config", config_data)


def _save_domain_results(
    output_manager: OutputManager, engine_result: dict[str, Any]
) -> dict[AnalysisDomain, Path]:
    """Save domain-specific results."""
    domain_dirs: dict[AnalysisDomain, Path] = {}
    for domain, results in engine_result["results"].items():
        domain_dir = output_manager.create_domain_dir(domain)
        domain_dirs[domain] = domain_dir
        output_manager.save_json("results", results, subdir=domain.value)
    return domain_dirs


def _save_errors(output_manager: OutputManager, engine_result: dict[str, Any]) -> Path | None:
    """Save error log if errors occurred."""
    errors: list[AnalysisError] = engine_result["errors"]
    if not errors:
        return None

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
    return output_manager.save_json("failed_analyses", error_data, subdir="errors")


def _build_result(
    output_manager: OutputManager,
    input_path: Path | None,
    input_type: InputType,
    engine_result: dict[str, Any],
    summary_json: Path,
    summary_yaml: Path | None,
    metadata_json: Path,
    config_yaml: Path,
    domain_dirs: dict[AnalysisDomain, Path],
    plot_paths: list[Path],
    error_log: Path | None,
    start_time: float,
) -> AnalysisResult:
    """Build partial AnalysisResult for index generation."""
    return AnalysisResult(
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
        errors=engine_result["errors"],
    )


def _generate_index(
    output_manager: OutputManager,
    partial_result: AnalysisResult,
    config: AnalysisConfig,
    progress_callback: Callable[[ProgressInfo], None] | None,
    start_time: float,
) -> dict[str, Path]:
    """Generate index files."""
    _report_progress(
        progress_callback,
        "indexing",
        None,
        None,
        95.0,
        "Generating index files",
        time.time() - start_time,
    )

    from oscura.reporting.index import IndexGenerator

    index_gen = IndexGenerator(output_manager)
    return index_gen.generate(partial_result, config.index_formats)


def _build_final_result(
    partial_result: AnalysisResult,
    index_paths: dict[str, Path],
    duration: float,
) -> AnalysisResult:
    """Build final AnalysisResult with index paths."""
    return AnalysisResult(
        output_dir=partial_result.output_dir,
        index_html=index_paths.get("html"),
        index_md=index_paths.get("md"),
        index_pdf=index_paths.get("pdf"),
        summary_json=partial_result.summary_json,
        summary_yaml=partial_result.summary_yaml,
        metadata_json=partial_result.metadata_json,
        config_yaml=partial_result.config_yaml,
        domain_dirs=partial_result.domain_dirs,
        plot_paths=partial_result.plot_paths,
        error_log=partial_result.error_log,
        input_file=partial_result.input_file,
        input_type=partial_result.input_type,
        total_analyses=partial_result.total_analyses,
        successful_analyses=partial_result.successful_analyses,
        failed_analyses=partial_result.failed_analyses,
        skipped_analyses=partial_result.skipped_analyses,
        duration_seconds=duration,
        domain_summaries=partial_result.domain_summaries,
        errors=partial_result.errors,
    )


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
            from oscura.loaders.touchstone import load_touchstone

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
