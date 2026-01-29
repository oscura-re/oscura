"""Batch report generation for Oscura.

This module provides utilities for generating reports across multiple DUTs
or files with summary reports and yield analysis.


Example:
    >>> from oscura.reporting.batch import batch_report, generate_batch_report
    >>> # Process multiple files
    >>> results = batch_report(['dut1.wfm', 'dut2.wfm'], template='production', output_dir='reports/')
    >>> # Or generate from pre-computed results
    >>> report = generate_batch_report(batch_results, "batch_summary.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.reporting.core import Report, ReportConfig, Section
from oscura.reporting.tables import format_batch_summary_table

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "BatchReportResult",
    "aggregate_batch_measurements",
    "batch_report",
    "generate_batch_report",
]


class BatchReportResult:
    """Result of batch report generation.

    Attributes:
        summary_report_path: Path to summary report
        individual_report_paths: Paths to individual DUT reports
        total_duts: Total number of DUTs processed
        passed_duts: Number of passing DUTs
        failed_duts: Number of failing DUTs
        errors: List of processing errors

    References:
        RPT-003: Batch Report Generation
    """

    def __init__(self) -> None:
        """Initialize batch result."""
        self.summary_report_path: Path | None = None
        self.individual_report_paths: list[Path] = []
        self.total_duts: int = 0
        self.passed_duts: int = 0
        self.failed_duts: int = 0
        self.errors: list[tuple[str, str]] = []

    @property
    def dut_yield(self) -> float:
        """Calculate DUT yield percentage."""
        if self.total_duts == 0:
            return 0.0
        return (self.passed_duts / self.total_duts) * 100


def batch_report(
    files: list[str | Path],
    template: str = "production",
    output_dir: str | Path = "reports",
    *,
    analyzer: Callable[[Any], dict[str, Any]] | None = None,
    generate_individual: bool = True,
    generate_summary: bool = True,
    output_format: str = "pdf",
    file_pattern: str = "{dut_id}_report.{ext}",
    summary_filename: str = "batch_summary.{ext}",
    dut_id_extractor: Callable[[Path], str] | None = None,
) -> BatchReportResult:
    """Generate reports for multiple DUTs/files.

    This is the primary interface for batch report generation.

    Args:
        files: List of input file paths
        template: Report template name (default: 'production')
        output_dir: Output directory for reports
        analyzer: Optional analysis function (trace -> results dict)
        generate_individual: Generate individual DUT reports
        generate_summary: Generate summary report across all DUTs
        output_format: Output format ('pdf', 'html')
        file_pattern: Filename pattern for individual reports
        summary_filename: Filename for summary report
        dut_id_extractor: Function to extract DUT ID from file path

    Returns:
        BatchReportResult with paths and statistics

    Example:
        >>> result = batch_report(
        ...     files=['dut1.wfm', 'dut2.wfm', 'dut3.wfm'],
        ...     template='production',
        ...     output_dir='./reports'
        ... )
        >>> print(f"Yield: {result.dut_yield:.1f}%")
        >>> print(f"Summary: {result.summary_report_path}")

    References:
        RPT-003: Batch Report Generation
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = BatchReportResult()
    batch_results: list[dict[str, Any]] = []

    # Load template
    report_template = _load_report_template(template)

    # Get DUT ID extractor
    extractor = dut_id_extractor or _default_dut_id_extractor

    # Process each file
    batch_results = _process_files(
        files,
        extractor,
        analyzer,
        result,
        output_path,
        report_template,
        output_format,
        file_pattern,
        generate_individual,
    )

    # Generate summary report if requested
    if generate_summary and batch_results:
        _create_summary_report(batch_results, output_path, summary_filename, output_format, result)

    return result


def _load_report_template(template: str) -> Any:
    """Load report template with fallback to default."""
    from oscura.reporting.template_system import load_template

    try:
        return load_template(template)
    except ValueError:
        logger.warning(f"Template '{template}' not found, using 'default'")
        return load_template("default")


def _default_dut_id_extractor(path: Path) -> str:
    """Extract DUT ID from file path (default: use stem)."""
    return Path(path).stem


def _process_files(
    files: list[str | Path],
    dut_id_extractor: Callable[[Path], str],
    analyzer: Callable[[Any], dict[str, Any]] | None,
    result: BatchReportResult,
    output_path: Path,
    report_template: Any,
    output_format: str,
    file_pattern: str,
    generate_individual: bool,
) -> list[dict[str, Any]]:
    """Process all files and generate individual reports."""

    batch_results: list[dict[str, Any]] = []

    for file_path in files:
        file_path = Path(file_path)
        dut_id = dut_id_extractor(file_path)

        dut_result = _process_single_file(file_path, dut_id, analyzer, result)

        if dut_result is not None:
            batch_results.append(dut_result)

            if generate_individual:
                _save_individual_report(
                    dut_result,
                    dut_id,
                    output_path,
                    file_pattern,
                    output_format,
                    report_template,
                    result,
                )

    return batch_results


def _process_single_file(
    file_path: Path,
    dut_id: str,
    analyzer: Callable[[Any], dict[str, Any]] | None,
    result: BatchReportResult,
) -> dict[str, Any] | None:
    """Process a single DUT file and update result statistics."""
    import oscura as osc

    try:
        trace = osc.load(str(file_path))
        dut_result = analyzer(trace) if analyzer else _default_analysis(trace)

        dut_result["dut_id"] = dut_id
        dut_result["source_file"] = str(file_path)

        # Update pass/fail counts
        if dut_result.get("pass_count", 0) == dut_result.get("total_count", 0):
            result.passed_duts += 1
        else:
            result.failed_duts += 1

        result.total_duts += 1
        return dut_result

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        result.errors.append((str(file_path), str(e)))
        return None


def _save_individual_report(
    dut_result: dict[str, Any],
    dut_id: str,
    output_path: Path,
    file_pattern: str,
    output_format: str,
    report_template: Any,
    result: BatchReportResult,
) -> None:
    """Save individual DUT report."""
    ext = output_format.lower()
    individual_filename = file_pattern.format(dut_id=dut_id, ext=ext)
    individual_path = output_path / individual_filename

    try:
        _generate_individual_report(dut_result, individual_path, report_template, output_format)
        result.individual_report_paths.append(individual_path)
    except Exception as e:
        logger.error(f"Failed to generate report for {dut_id}: {e}")
        result.errors.append((dut_id, str(e)))


def _create_summary_report(
    batch_results: list[dict[str, Any]],
    output_path: Path,
    summary_filename: str,
    output_format: str,
    result: BatchReportResult,
) -> None:
    """Create and save summary report."""
    ext = output_format.lower()
    summary_path = output_path / summary_filename.format(ext=ext)

    try:
        summary_report = generate_batch_report(batch_results)
        _save_report(summary_report, summary_path, output_format)
        result.summary_report_path = summary_path
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
        result.errors.append(("summary", str(e)))


def _default_analysis(trace: Any) -> dict[str, Any]:
    """Default analysis for a trace."""
    import numpy as np

    from oscura.core.types import WaveformTrace

    if not isinstance(trace, WaveformTrace):
        return {
            "measurements": {},
            "pass_count": 0,
            "total_count": 0,
        }

    data = trace.data

    # Basic measurements
    measurements = {
        "peak_to_peak": {
            "value": float(np.ptp(data)),
            "unit": "V",
            "passed": True,
        },
        "rms": {
            "value": float(np.sqrt(np.mean(data**2))),
            "unit": "V",
            "passed": True,
        },
        "mean": {
            "value": float(np.mean(data)),
            "unit": "V",
            "passed": True,
        },
        "std_dev": {
            "value": float(np.std(data)),
            "unit": "V",
            "passed": True,
        },
    }

    return {
        "measurements": measurements,
        "pass_count": len(measurements),
        "total_count": len(measurements),
    }


def _generate_individual_report(
    dut_result: dict[str, Any],
    output_path: Path,
    template: Any,
    output_format: str,
) -> None:
    """Generate individual DUT report."""
    from oscura.reporting.core import Report, ReportConfig

    dut_id = dut_result.get("dut_id", "Unknown")
    config = ReportConfig(title=f"Test Report: {dut_id}")
    report = Report(config=config)

    # Add summary section
    pass_count = dut_result.get("pass_count", 0)
    total_count = dut_result.get("total_count", 0)
    pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0

    summary = f"DUT: {dut_id}\n"
    summary += f"Source: {dut_result.get('source_file', 'N/A')}\n"
    summary += f"Pass Rate: {pass_count}/{total_count} ({pass_rate:.1f}%)"

    report.add_section("Summary", summary, level=1)

    # Add measurements
    if "measurements" in dut_result:
        from oscura.reporting.tables import create_measurement_table

        table = create_measurement_table(dut_result["measurements"], format="dict")
        report.add_section("Measurements", [table], level=1)

    _save_report(report, output_path, output_format)


def _save_report(report: Report, output_path: Path, output_format: str) -> None:
    """Save report in specified format."""
    if output_format.lower() == "pdf":
        from oscura.reporting.pdf import save_pdf_report

        save_pdf_report(report, str(output_path))
    elif output_format.lower() == "html":
        from oscura.reporting.html import save_html_report

        save_html_report(report, str(output_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def generate_batch_report(
    batch_results: list[dict[str, Any]],
    *,
    title: str = "Batch Test Summary Report",
    include_individual: bool = True,
    include_yield_analysis: bool = True,
    include_outliers: bool = True,
    **kwargs: Any,
) -> Report:
    """Generate batch summary report for multiple DUTs.

    Args:
        batch_results: List of result dictionaries, one per DUT.
        title: Report title.
        include_individual: Include individual DUT sections.
        include_yield_analysis: Include yield analysis section.
        include_outliers: Include outlier detection section.
        **kwargs: Additional report configuration options.

    Returns:
        Batch Report object.

    References:
        REPORT-009, REPORT-018
    """
    config = ReportConfig(title=title, **kwargs)
    report = Report(config=config)

    # Add batch summary
    summary = _generate_batch_summary(batch_results)
    report.add_section("Batch Summary", summary, level=1)

    # Add batch summary table
    summary_table = format_batch_summary_table(batch_results, format="dict")
    report.add_section("DUT Summary Table", [summary_table], level=1)

    # Add yield analysis
    if include_yield_analysis:
        yield_section = _create_yield_analysis_section(batch_results)
        report.sections.append(yield_section)

    # Add statistical analysis
    stats_section = _create_batch_statistics_section(batch_results)
    report.sections.append(stats_section)

    # Add outlier detection
    if include_outliers:
        outlier_section = _create_outlier_detection_section(batch_results)
        report.sections.append(outlier_section)

    # Add individual DUT sections
    if include_individual:
        for i, dut_result in enumerate(batch_results):
            dut_section = _create_dut_section(dut_result, i)
            report.sections.append(dut_section)

    return report


def _generate_batch_summary(batch_results: list[dict[str, Any]]) -> str:
    """Generate batch summary text."""
    summary_parts = []

    total_duts = len(batch_results)
    summary_parts.append(f"Tested {total_duts} DUT(s).")

    # Aggregate statistics
    total_tests = sum(r.get("total_count", 0) for r in batch_results)
    total_passed = sum(r.get("pass_count", 0) for r in batch_results)

    if total_tests > 0:
        pass_rate = total_passed / total_tests * 100
        summary_parts.append(
            f"\nOverall: {total_passed}/{total_tests} tests passed ({pass_rate:.1f}% pass rate)."
        )

    # DUT-level yield
    passing_duts = sum(
        1 for r in batch_results if r.get("pass_count", 0) == r.get("total_count", 0)
    )

    if total_duts > 0:
        dut_yield = passing_duts / total_duts * 100
        summary_parts.append(
            f"\nDUT Yield: {passing_duts}/{total_duts} DUTs passed all tests "
            f"({dut_yield:.1f}% yield)."
        )

    # Failed DUTs
    if passing_duts < total_duts:
        failed_duts = []
        for i, r in enumerate(batch_results):
            dut_id = r.get("dut_id", f"DUT-{i + 1}")
            if r.get("pass_count", 0) < r.get("total_count", 0):
                failed_duts.append(dut_id)

        summary_parts.append(f"\nFailed DUTs: {', '.join(failed_duts)}")

    return "\n".join(summary_parts)


def _create_yield_analysis_section(batch_results: list[dict[str, Any]]) -> Section:
    """Create yield analysis section."""
    content_parts = []

    # Overall yield
    total_duts = len(batch_results)
    passing_duts = sum(
        1 for r in batch_results if r.get("pass_count", 0) == r.get("total_count", 0)
    )

    overall_yield = (passing_duts / total_duts * 100) if total_duts > 0 else 0

    content_parts.append(f"**Overall Yield:** {overall_yield:.2f}%")
    content_parts.append(f"**Passing DUTs:** {passing_duts}/{total_duts}")

    # Per-test yield
    content_parts.append("\n**Per-Test Yield:**")

    # Collect all test names
    all_tests: set[str] = set()
    for result in batch_results:
        if "measurements" in result:
            all_tests.update(result["measurements"].keys())

    test_yields: list[tuple[str, float, int, int]] = []
    for test_name in sorted(all_tests):
        total_with_test = 0
        passed_test = 0

        for result in batch_results:
            if "measurements" in result and test_name in result["measurements"]:
                total_with_test += 1
                if result["measurements"][test_name].get("passed", True):
                    passed_test += 1

        if total_with_test > 0:
            test_yield = passed_test / total_with_test * 100
            test_yields.append((test_name, test_yield, passed_test, total_with_test))

    # Sort by yield (worst first)
    test_yields.sort(key=lambda x: x[1])

    for test_name, yield_pct, passed, total in test_yields:
        content_parts.append(f"- {test_name}: {yield_pct:.1f}% ({passed}/{total})")

    content = "\n".join(content_parts)

    return Section(
        title="Yield Analysis",
        content=content,
        level=1,
        visible=True,
    )


def _create_batch_statistics_section(batch_results: list[dict[str, Any]]) -> Section:
    """Create batch statistical analysis section."""
    from oscura.reporting.formatting import NumberFormatter

    formatter = NumberFormatter()

    # Collect measurements across all DUTs
    param_values: dict[str, list[float]] = {}

    for result in batch_results:
        if "measurements" in result:
            for param, meas in result["measurements"].items():
                value = meas.get("value")
                if value is not None:
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)

    # Build statistics table
    headers = ["Parameter", "Mean", "Std Dev", "Min", "Max", "Range"]
    rows = []

    for param in sorted(param_values.keys()):
        values = np.array(param_values[param])
        unit = ""

        # Get unit from first measurement
        for result in batch_results:
            if "measurements" in result and param in result["measurements"]:
                unit = result["measurements"][param].get("unit", "")
                break

        rows.append(
            [
                param,
                formatter.format(float(np.mean(values)), unit),
                formatter.format(float(np.std(values)), unit),
                formatter.format(float(np.min(values)), unit),
                formatter.format(float(np.max(values)), unit),
                formatter.format(float(np.max(values) - np.min(values)), unit),
            ]
        )

    table = {"type": "table", "headers": headers, "data": rows}

    return Section(
        title="Batch Statistics",
        content=[table],
        level=1,
        visible=True,
    )


def _create_outlier_detection_section(batch_results: list[dict[str, Any]]) -> Section:
    """Create outlier detection section."""
    content_parts: list[str] = []

    # Collect measurements
    param_values: dict[str, list[tuple[int, float]]] = {}

    for i, result in enumerate(batch_results):
        if "measurements" in result:
            for param, meas in result["measurements"].items():
                value = meas.get("value")
                if value is not None:
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append((i, value))

    # Detect outliers using 3-sigma rule
    outliers_found = False

    for param in sorted(param_values.keys()):
        values_with_idx = param_values[param]
        values = np.array([v for _, v in values_with_idx])

        mean = float(np.mean(values))
        std = float(np.std(values))

        if std > 0:
            outlier_indices: list[tuple[int, float, float]] = []
            for idx, value in values_with_idx:
                z_score = abs(value - mean) / std
                if z_score > 3:  # 3-sigma rule
                    outlier_indices.append((idx, value, z_score))

            if outlier_indices:
                outliers_found = True
                content_parts.append(f"**{param}:**")
                for idx, value, z_score in outlier_indices:
                    dut_id = batch_results[idx].get("dut_id", f"DUT-{idx + 1}")
                    content_parts.append(f"- {dut_id}: {value:.3g} (z-score: {z_score:.2f})")

    if not outliers_found:
        content_parts.append("No statistical outliers detected (3-sigma threshold).")

    content = "\n".join(content_parts)

    return Section(
        title="Outlier Detection",
        content=content,
        level=1,
        visible=True,
    )


def _create_dut_section(result: dict[str, Any], index: int) -> Section:
    """Create individual DUT section."""
    dut_id = result.get("dut_id", f"DUT-{index + 1}")

    content_parts: list[Any] = []

    # DUT summary
    pass_count = result.get("pass_count", 0)
    total_count = result.get("total_count", 0)

    if total_count > 0:
        pass_rate = pass_count / total_count * 100
        content_parts.append(f"Pass rate: {pass_count}/{total_count} ({pass_rate:.1f}%)")

    # Measurements table
    if "measurements" in result:
        from oscura.reporting.tables import create_measurement_table

        table = create_measurement_table(result["measurements"], format="dict")
        content_parts.append(table)

    return Section(
        title=f"DUT: {dut_id}",
        content=content_parts,
        level=2,
        visible=True,
        collapsible=True,
    )


def aggregate_batch_measurements(
    batch_results: list[dict[str, Any]],
) -> dict[str, NDArray[np.float64]]:
    """Aggregate measurements across batch for statistical analysis.

    Args:
        batch_results: List of DUT results.

    Returns:
        Dictionary mapping parameter name to array of values.

    References:
        REPORT-009
    """
    param_values: dict[str, list[float]] = {}

    for result in batch_results:
        if "measurements" in result:
            for param, meas in result["measurements"].items():
                value = meas.get("value")
                if value is not None:
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)

    return {k: np.array(v) for k, v in param_values.items()}
