"""CSV export functions for vintage logic analysis results.

This module provides specialized CSV exporters for vintage logic analysis data,
including timing measurements, IC identification, and bill of materials.

Example:
    >>> from oscura.exporters.vintage_logic_csv import export_bom_csv
    >>> export_bom_csv(result, "bom.csv")
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.analyzers.digital.vintage_result import VintageLogicAnalysisResult


def export_timing_measurements_csv(
    result: VintageLogicAnalysisResult,
    path: str | Path,
) -> None:
    """Export timing measurements to CSV.

    Creates a CSV file with columns: parameter, measured_value_ns, measurement_type.

    Args:
        result: Vintage logic analysis result.
        path: Output CSV file path.

    Example:
        >>> export_timing_measurements_csv(result, "timing.csv")
    """
    path = Path(path)

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["parameter", "measured_value_ns", "measurement_type"])

        # Write timing measurements
        for param_name, value in result.timing_measurements.items():
            # Determine measurement type from parameter name
            if "_t_pd" in param_name:
                meas_type = "propagation_delay"
            elif "_t_su" in param_name:
                meas_type = "setup_time"
            elif "_t_h" in param_name:
                meas_type = "hold_time"
            elif "_t_w" in param_name:
                meas_type = "pulse_width"
            else:
                meas_type = "other"

            writer.writerow([param_name, f"{value * 1e9:.3f}", meas_type])


def export_ic_identification_csv(
    result: VintageLogicAnalysisResult,
    path: str | Path,
) -> None:
    """Export IC identification results to CSV.

    Creates a CSV file with columns: ic_name, confidence, family, timing_params,
    validation_status.

    Args:
        result: Vintage logic analysis result.
        path: Output CSV file path.

    Example:
        >>> export_ic_identification_csv(result, "ic_identification.csv")
    """
    path = Path(path)

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(
            [
                "ic_name",
                "confidence",
                "family",
                "t_pd_ns",
                "t_su_ns",
                "t_h_ns",
                "t_w_ns",
                "validation_status",
            ]
        )

        # Write IC identification results
        for ic_result in result.identified_ics:
            # Extract timing parameters
            t_pd = ic_result.timing_params.get("t_pd", 0) * 1e9
            t_su = ic_result.timing_params.get("t_su", 0) * 1e9
            t_h = ic_result.timing_params.get("t_h", 0) * 1e9
            t_w = ic_result.timing_params.get("t_w", 0) * 1e9

            # Determine validation status
            validation_failed = any(v.get("passes") is False for v in ic_result.validation.values())
            validation_status = "FAIL" if validation_failed else "PASS"

            writer.writerow(
                [
                    ic_result.ic_name,
                    f"{ic_result.confidence:.3f}",
                    ic_result.family,
                    f"{t_pd:.3f}" if t_pd > 0 else "",
                    f"{t_su:.3f}" if t_su > 0 else "",
                    f"{t_h:.3f}" if t_h > 0 else "",
                    f"{t_w:.3f}" if t_w > 0 else "",
                    validation_status,
                ]
            )


def export_bom_csv(
    result: VintageLogicAnalysisResult,
    path: str | Path,
) -> None:
    """Export bill of materials to CSV.

    Creates a CSV file compatible with spreadsheet programs and procurement systems.
    Columns: part_number, description, quantity, category, notes.

    Args:
        result: Vintage logic analysis result.
        path: Output CSV file path.

    Example:
        >>> export_bom_csv(result, "bom.csv")
    """
    path = Path(path)

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["part_number", "description", "quantity", "category", "notes"])

        # Write BOM entries
        for entry in result.bom:
            writer.writerow(
                [
                    entry.part_number,
                    entry.description,
                    entry.quantity,
                    entry.category,
                    entry.notes or "",
                ]
            )


def export_voltage_levels_csv(
    result: VintageLogicAnalysisResult,
    path: str | Path,
) -> None:
    """Export voltage levels to CSV.

    Creates a CSV file with measured voltage levels for the detected logic family.

    Args:
        result: Vintage logic analysis result.
        path: Output CSV file path.

    Example:
        >>> export_voltage_levels_csv(result, "voltage_levels.csv")
    """
    path = Path(path)

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["parameter", "voltage_v", "logic_family"])

        # Write voltage levels
        for param, value in result.voltage_levels.items():
            writer.writerow([param, f"{value:.3f}", result.detected_family])


def export_all_vintage_logic_csv(
    result: VintageLogicAnalysisResult,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Export all vintage logic analysis data to CSV files.

    Convenience function that exports all data types to separate CSV files.

    Args:
        result: Vintage logic analysis result.
        output_dir: Output directory for CSV files.
        prefix: Optional prefix for file names.

    Returns:
        Dictionary mapping data type to output file path.

    Example:
        >>> paths = export_all_vintage_logic_csv(result, "./output", prefix="analysis_")
        >>> print(paths["bom"])  # PosixPath('./output/analysis_bom.csv')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Export timing measurements
    if result.timing_measurements:
        timing_path = output_dir / f"{prefix}timing_measurements.csv"
        export_timing_measurements_csv(result, timing_path)
        paths["timing_measurements"] = timing_path

    # Export IC identification
    if result.identified_ics:
        ic_path = output_dir / f"{prefix}ic_identification.csv"
        export_ic_identification_csv(result, ic_path)
        paths["ic_identification"] = ic_path

    # Export BOM
    if result.bom:
        bom_path = output_dir / f"{prefix}bom.csv"
        export_bom_csv(result, bom_path)
        paths["bom"] = bom_path

    # Export voltage levels
    if result.voltage_levels:
        voltage_path = output_dir / f"{prefix}voltage_levels.csv"
        export_voltage_levels_csv(result, voltage_path)
        paths["voltage_levels"] = voltage_path

    return paths


__all__ = [
    "export_all_vintage_logic_csv",
    "export_bom_csv",
    "export_ic_identification_csv",
    "export_timing_measurements_csv",
    "export_voltage_levels_csv",
]
