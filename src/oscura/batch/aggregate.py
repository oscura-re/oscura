"""Result aggregation for batch analysis.


This module provides statistical aggregation and reporting for batch
analysis results, including outlier detection and export capabilities.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def aggregate_results(
    results: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    outlier_threshold: float = 3.0,
    include_plots: bool = False,
    output_format: str = "dict",
    output_file: str | Path | None = None,
) -> dict[str, Any] | pd.DataFrame:
    """Aggregate results from batch analysis into summary statistics.

    : Computes comprehensive statistics (mean, std, min, max,
    outliers) for each metric in the batch results. Supports export to various
    formats and optional visualization generation.

    Args:
        results: DataFrame from batch_analyze() containing analysis results
        metrics: List of column names to aggregate (default: all numeric columns)
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        include_plots: Generate comparison plots across files (default: False)
        output_format: Output format - 'dict', 'dataframe', 'csv', 'excel', 'html'
        output_file: Optional output file path for export formats

    Returns:
        Dictionary or DataFrame with summary statistics:
        - count: Number of valid values
        - mean: Mean value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - median: Median value
        - q25: 25th percentile
        - q75: 75th percentile
        - outliers: List of outlier values
        - outlier_files: List of files containing outliers

    Raises:
        ValueError: If no numeric metrics are found in results.

    Examples:
        >>> results = osc.batch_analyze(files, osc.characterize_buffer)
        >>> summary = osc.aggregate_results(
        ...     results,
        ...     metrics=['rise_time', 'fall_time'],
        ...     outlier_threshold=2.5
        ... )
        >>> print(summary['rise_time']['mean'])
        >>> print(summary['rise_time']['outlier_files'])

    Notes:
        - Outliers detected using IQR method: values outside [Q1 - k*IQR, Q3 + k*IQR]
          where k = (threshold / 3.0) * 1.5 (more robust than z-score for heavy-tailed data)
        - Non-numeric columns are automatically skipped
        - Missing values (NaN) are excluded from statistics
        - CSV/Excel/HTML export requires output_file parameter

    References:
        BATCH-002: Result Aggregation
    """
    if results.empty:
        return {} if output_format == "dict" else pd.DataFrame()

    # Determine metrics to analyze
    if metrics is None:
        # Auto-select all numeric columns except 'file' and 'error'
        metrics = results.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [m for m in metrics if m not in ["file", "error"]]

    if not metrics:
        raise ValueError("No numeric metrics found in results")

    # Compute aggregated statistics
    aggregated: dict[str, dict[str, Any]] = {}

    for metric in metrics:
        if metric not in results.columns:
            continue

        # Extract valid (non-null) values
        values = results[metric].dropna()

        if values.empty:
            aggregated[metric] = {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "outliers": [],
                "outlier_files": [],
            }
            continue

        # Basic statistics
        stats = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
        }

        # Outlier detection using IQR method (more robust than z-score)
        # IQR method: outliers are values outside [Q1 - k*IQR, Q3 + k*IQR]
        # where k = outlier_threshold * 1.5 (standard is k=1.5, we scale by threshold)
        if len(values) > 3:  # Need at least 4 values for meaningful IQR
            q1 = stats["q25"]
            q3 = stats["q75"]
            iqr = q3 - q1

            # Scale IQR multiplier by threshold (default 3.0 -> 2.0 * 1.5 = 3.0)
            k = (outlier_threshold / 3.0) * 1.5

            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr

            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_indices = values[outlier_mask].index.tolist()
            stats["outliers"] = values[outlier_mask].tolist()

            # Get corresponding filenames if available
            if "file" in results.columns:
                stats["outlier_files"] = results.loc[outlier_indices, "file"].tolist()
            else:
                stats["outlier_files"] = outlier_indices
        else:
            stats["outliers"] = []  # type: ignore[assignment]
            stats["outlier_files"] = []  # type: ignore[assignment]

        aggregated[metric] = stats

    # Generate plots if requested
    if include_plots:
        # Import here to avoid circular dependency
        try:
            import matplotlib.pyplot as plt

            for metric in metrics:
                if metric not in aggregated:
                    continue

                _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Histogram
                results[metric].dropna().hist(ax=ax1, bins=30)
                ax1.axvline(
                    aggregated[metric]["mean"],
                    color="r",
                    linestyle="--",
                    label="Mean",
                )
                ax1.axvline(
                    aggregated[metric]["median"],
                    color="g",
                    linestyle="--",
                    label="Median",
                )
                ax1.set_xlabel(metric)
                ax1.set_ylabel("Count")
                ax1.legend()
                ax1.set_title(f"{metric} Distribution")

                # Box plot
                ax2.boxplot(results[metric].dropna())
                ax2.set_ylabel(metric)
                ax2.set_title(f"{metric} Box Plot")

                plt.tight_layout()

                # Save or show based on output_file
                if output_file:
                    plot_file = Path(output_file).with_suffix("") / f"{metric}_plot.png"
                    plot_file.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(plot_file)
                else:
                    plt.show()

                plt.close()

        except ImportError:
            pass  # Silently skip plotting if matplotlib not available

    # Format output
    if output_format == "dict":
        return aggregated

    elif output_format == "dataframe":
        # Convert to DataFrame with metrics as rows
        df = pd.DataFrame(aggregated).T
        # Drop list columns for DataFrame format
        df = df.drop(columns=["outliers", "outlier_files"], errors="ignore")
        return df

    elif output_format in ["csv", "excel", "html"]:
        if not output_file:
            raise ValueError(f"{output_format} format requires output_file parameter")

        df = pd.DataFrame(aggregated).T
        df = df.drop(columns=["outliers", "outlier_files"], errors="ignore")

        if output_format == "csv":
            df.to_csv(output_file)
        elif output_format == "excel":
            df.to_excel(output_file)
        elif output_format == "html":
            # Generate HTML report
            html = _generate_html_report(results, aggregated, metrics)
            Path(output_file).write_text(html)

        return df

    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def _generate_html_report(
    results: pd.DataFrame,
    aggregated: dict[str, dict[str, Any]],
    metrics: list[str],
) -> str:
    """Generate HTML report for batch analysis results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .outlier { background-color: #ffcccc; }
        </style>
    </head>
    <body>
        <h1>Batch Analysis Report</h1>
    """
    # Summary statistics table
    html += "<h2>Summary Statistics</h2>\n<table>\n"
    html += "<tr><th>Metric</th><th>Count</th><th>Mean</th><th>Std</th>"
    html += "<th>Min</th><th>Median</th><th>Max</th><th>Outliers</th></tr>\n"

    for metric in metrics:
        if metric not in aggregated:
            continue
        stats = aggregated[metric]
        html += "<tr>"
        html += f"<td>{metric}</td>"
        html += f"<td>{stats['count']}</td>"
        html += f"<td>{stats['mean']:.4g}</td>"
        html += f"<td>{stats['std']:.4g}</td>"
        html += f"<td>{stats['min']:.4g}</td>"
        html += f"<td>{stats['median']:.4g}</td>"
        html += f"<td>{stats['max']:.4g}</td>"
        html += f"<td>{len(stats['outliers'])}</td>"
        html += "</tr>\n"

    html += "</table>\n"

    # Outlier details
    has_outliers = any(len(aggregated[m]["outliers"]) > 0 for m in metrics if m in aggregated)

    if has_outliers:
        html += "<h2>Outliers Detected</h2>\n"
        for metric in metrics:
            if metric not in aggregated:
                continue
            stats = aggregated[metric]
            if stats["outliers"]:
                html += f"<h3>{metric}</h3>\n<table>\n"
                html += "<tr><th>File</th><th>Value</th></tr>\n"
                for file, value in zip(stats["outlier_files"], stats["outliers"], strict=False):
                    html += f"<tr class='outlier'><td>{file}</td><td>{value:.4g}</td></tr>\n"
                html += "</table>\n"

    html += "</body>\n</html>"
    return html
