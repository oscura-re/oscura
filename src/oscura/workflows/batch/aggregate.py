"""Result aggregation for batch analysis.


This module provides statistical aggregation and reporting for batch
analysis results, including outlier detection and export capabilities.

**Requires pandas:**
This module requires pandas for DataFrame operations. Install with:
    pip install oscura[dataframes]    # Pandas + Excel export
    pip install oscura[standard]       # Recommended
"""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Batch aggregation requires pandas.\n\n"
        "Install with:\n"
        "  pip install oscura[dataframes]       # Pandas + Excel export\n"
        "  pip install oscura[standard]         # Recommended\n"
        "  pip install oscura[all]              # Everything\n"
    ) from e


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

    # Select and validate metrics
    metrics_to_use = _select_metrics(results, metrics)

    # Compute statistics for each metric
    aggregated = _compute_metric_statistics(results, metrics_to_use, outlier_threshold)

    # Generate plots if requested
    if include_plots:
        _generate_metric_plots(results, aggregated, metrics_to_use, output_file)

    # Format and return results
    return _format_output(aggregated, output_format, output_file, results, metrics_to_use)


def _select_metrics(results: pd.DataFrame, metrics: list[str] | None) -> list[str]:
    """Select and validate metrics to aggregate.

    Args:
        results: DataFrame containing results.
        metrics: User-specified metrics or None for auto-selection.

    Returns:
        List of metric column names.

    Raises:
        ValueError: If no numeric metrics found.
    """
    if metrics is None:
        metrics = results.select_dtypes(include=[np.number]).columns.tolist()
        metrics = [m for m in metrics if m not in ["file", "error"]]

    if not metrics:
        raise ValueError("No numeric metrics found in results")

    return metrics


def _compute_metric_statistics(
    results: pd.DataFrame,
    metrics: list[str],
    outlier_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Compute statistics for all metrics.

    Args:
        results: DataFrame containing results.
        metrics: List of metrics to compute statistics for.
        outlier_threshold: Threshold for outlier detection.

    Returns:
        Dictionary mapping metrics to statistics.
    """
    aggregated: dict[str, dict[str, Any]] = {}

    for metric in metrics:
        if metric not in results.columns:
            continue

        values = results[metric].dropna()

        if values.empty:
            aggregated[metric] = _create_empty_stats()
            continue

        aggregated[metric] = _compute_single_metric_stats(values, results, outlier_threshold)

    return aggregated


def _create_empty_stats() -> dict[str, Any]:
    """Create empty statistics dictionary.

    Returns:
        Dictionary with NaN values and empty lists.
    """
    return {
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


def _compute_single_metric_stats(
    values: pd.Series,
    results: pd.DataFrame,
    outlier_threshold: float,
) -> dict[str, Any]:
    """Compute statistics for a single metric.

    Args:
        values: Series of non-null metric values.
        results: Full results DataFrame (for file lookup).
        outlier_threshold: Threshold for outlier detection.

    Returns:
        Dictionary with computed statistics.
    """
    stats = _compute_basic_statistics(values)
    outliers_info = _detect_outliers(values, stats, outlier_threshold)
    stats.update(outliers_info)

    # Add outlier files if available
    if outliers_info["outliers"]:
        outlier_files: Any = _get_outlier_files(results, outliers_info["outlier_indices"])
        stats["outlier_files"] = outlier_files
    else:
        stats["outlier_files"] = []

    return stats


def _compute_basic_statistics(values: pd.Series) -> dict[str, Any]:
    """Compute basic statistics for metric values.

    Args:
        values: Series of metric values.

    Returns:
        Dictionary with basic statistics.
    """
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
    }


def _detect_outliers(
    values: pd.Series,
    stats: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Detect outliers using IQR method.

    Args:
        values: Series of metric values.
        stats: Basic statistics containing q25 and q75.
        threshold: IQR threshold multiplier.

    Returns:
        Dictionary with outlier information.
    """
    if len(values) <= 3:
        return {"outliers": [], "outlier_indices": []}

    q1, q3 = stats["q25"], stats["q75"]
    iqr = q3 - q1
    k = (threshold / 3.0) * 1.5

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outlier_indices = values[outlier_mask].index.tolist()

    return {
        "outliers": values[outlier_mask].tolist(),
        "outlier_indices": outlier_indices,
    }


def _get_outlier_files(
    results: pd.DataFrame,
    outlier_indices: list[Any],
) -> Any:
    """Get filenames for outlier indices.

    Args:
        results: Full results DataFrame.
        outlier_indices: Indices of outlier values.

    Returns:
        List of filenames or indices.
    """
    if "file" in results.columns:
        return results.loc[outlier_indices, "file"].tolist()
    return outlier_indices


def _generate_metric_plots(
    results: pd.DataFrame,
    aggregated: dict[str, dict[str, Any]],
    metrics: list[str],
    output_file: str | Path | None,
) -> None:
    """Generate comparison plots for metrics.

    Args:
        results: DataFrame with results.
        aggregated: Aggregated statistics.
        metrics: List of metrics to plot.
        output_file: Output file path for saving plots.
    """
    try:
        import matplotlib.pyplot as plt

        for metric in metrics:
            if metric not in aggregated:
                continue

            _create_metric_plot(results, aggregated, metric, output_file)
            plt.close()

    except ImportError:
        pass  # Skip if matplotlib unavailable


def _create_metric_plot(
    results: pd.DataFrame,
    aggregated: dict[str, dict[str, Any]],
    metric: str,
    output_file: str | Path | None,
) -> None:
    """Create histogram and box plot for a metric.

    Args:
        results: DataFrame with results.
        aggregated: Aggregated statistics.
        metric: Metric name.
        output_file: Output file path.
    """
    import matplotlib.pyplot as plt

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    _plot_histogram(results, aggregated, metric, ax1)

    # Box plot
    _plot_boxplot(results, metric, ax2)

    plt.tight_layout()

    # Save or show
    if output_file:
        plot_file = Path(output_file).with_suffix("") / f"{metric}_plot.png"
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_file)
    else:
        plt.show()


def _plot_histogram(
    results: pd.DataFrame,
    aggregated: dict[str, dict[str, Any]],
    metric: str,
    ax: Any,
) -> None:
    """Plot histogram with mean and median lines.

    Args:
        results: DataFrame with results.
        aggregated: Aggregated statistics.
        metric: Metric name.
        ax: Matplotlib axes.
    """
    results[metric].dropna().hist(ax=ax, bins=30)
    ax.axvline(aggregated[metric]["mean"], color="r", linestyle="--", label="Mean")
    ax.axvline(aggregated[metric]["median"], color="g", linestyle="--", label="Median")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title(f"{metric} Distribution")


def _plot_boxplot(results: pd.DataFrame, metric: str, ax: Any) -> None:
    """Plot box plot for metric.

    Args:
        results: DataFrame with results.
        metric: Metric name.
        ax: Matplotlib axes.
    """
    ax.boxplot(results[metric].dropna())
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Box Plot")


def _format_output(
    aggregated: dict[str, dict[str, Any]],
    output_format: str,
    output_file: str | Path | None,
    results: pd.DataFrame,
    metrics: list[str],
) -> dict[str, Any] | pd.DataFrame:
    """Format aggregated results based on output format.

    Args:
        aggregated: Aggregated statistics.
        output_format: Desired output format.
        output_file: Output file path.
        results: Original results DataFrame.
        metrics: List of metrics.

    Returns:
        Formatted results.

    Raises:
        ValueError: If unknown format or missing output_file.
    """
    if output_format == "dict":
        return aggregated

    if output_format == "dataframe":
        return _convert_to_dataframe(aggregated)

    if output_format in ["csv", "excel", "html"]:
        return _export_to_file(aggregated, output_format, output_file, results, metrics)

    raise ValueError(f"Unknown output_format: {output_format}")


def _convert_to_dataframe(aggregated: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert aggregated dict to DataFrame.

    Args:
        aggregated: Aggregated statistics dictionary.

    Returns:
        DataFrame with metrics as rows.
    """
    df = pd.DataFrame(aggregated).T
    return df.drop(columns=["outliers", "outlier_files"], errors="ignore")


def _export_to_file(
    aggregated: dict[str, dict[str, Any]],
    output_format: str,
    output_file: str | Path | None,
    results: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Export aggregated results to file.

    Args:
        aggregated: Aggregated statistics.
        output_format: Format (csv, excel, html).
        output_file: Output file path.
        results: Original results DataFrame.
        metrics: List of metrics.

    Returns:
        DataFrame with aggregated results.

    Raises:
        ValueError: If output_file not provided.
    """
    if not output_file:
        raise ValueError(f"{output_format} format requires output_file parameter")

    df = _convert_to_dataframe(aggregated)

    if output_format == "csv":
        df.to_csv(output_file)
    elif output_format == "excel":
        df.to_excel(output_file)
    elif output_format == "html":
        html = _generate_html_report(results, aggregated, metrics)
        Path(output_file).write_text(html)

    return df


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
