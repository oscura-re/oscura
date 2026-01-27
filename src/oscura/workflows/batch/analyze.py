"""Multi-file batch analysis with parallel execution support.


This module provides parallel batch processing of signal files using
concurrent.futures for efficient multi-core utilization.
"""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


def batch_analyze(
    files: list[str | Path],
    analysis_fn: Callable[[str | Path], dict[str, Any]],
    *,
    parallel: bool = False,
    workers: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    use_threads: bool = False,
    **config: Any,
) -> pd.DataFrame:
    """Analyze multiple files with the same analysis configuration.

    : Multi-file analysis with parallel execution support
    via concurrent.futures. Returns aggregated results as a DataFrame for
    easy statistical analysis and export.

    Args:
        files: List of file paths to analyze
        analysis_fn: Analysis function to apply to each file.
            Must accept a file path and return a dict of results.
        parallel: Enable parallel processing (default: False)
        workers: Number of parallel workers (default: CPU count)
        progress_callback: Optional callback for progress updates.
            Called with (current, total, filename) after each file.
        use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
            (useful for I/O-bound tasks, default: False)
        **config: Additional keyword arguments passed to analysis_fn

    Returns:
        DataFrame with one row per file, columns from analysis results.
        Always includes a 'file' column with the input filename.

    Examples:
        >>> import oscura as osc
        >>> import glob
        >>> files = glob.glob('captures/*.wfm')
        >>> results = osc.batch_analyze(
        ...     files,
        ...     analysis_fn=osc.characterize_buffer,
        ...     parallel=True,
        ...     workers=4
        ... )
        >>> print(results[['file', 'rise_time', 'fall_time', 'status']])
        >>> results.to_csv('batch_results.csv')

    Notes:
        - Use parallel=True for CPU-bound analysis functions
        - Use use_threads=True for I/O-bound operations (file loading)
        - Progress callback is called from worker threads/processes
        - All exceptions during analysis are caught and stored in 'error' column

    References:
        BATCH-001: Multi-File Analysis
    """
    if not files:
        return pd.DataFrame()

    wrapped_fn = _create_wrapped_analysis(analysis_fn, config)
    results = _execute_batch_analysis(
        files, wrapped_fn, parallel, workers, use_threads, progress_callback
    )
    return _build_result_dataframe(results)


def _create_wrapped_analysis(
    analysis_fn: Callable[[str | Path], dict[str, Any]], config: dict[str, Any]
) -> Callable[[str | Path], dict[str, Any]]:
    """Create wrapped analysis function with config injection and error handling.

    Args:
        analysis_fn: Original analysis function.
        config: Configuration parameters to pass to analysis_fn.

    Returns:
        Wrapped function that handles exceptions and ensures dict results.
    """

    def _wrapped_analysis(filepath: str | Path) -> dict[str, Any]:
        try:
            result = analysis_fn(filepath, **config)
            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"result": result}  # type: ignore[unreachable]
            result["file"] = str(filepath)
            result["error"] = None
            return result
        except Exception as e:
            # Return error info on failure
            return {
                "file": str(filepath),
                "error": str(e),
            }

    return _wrapped_analysis


def _execute_batch_analysis(
    files: list[str | Path],
    wrapped_fn: Callable[[str | Path], dict[str, Any]],
    parallel: bool,
    workers: int | None,
    use_threads: bool,
    progress_callback: Callable[[int, int, str], None] | None,
) -> list[dict[str, Any]]:
    """Execute batch analysis in parallel or sequential mode.

    Args:
        files: List of file paths to analyze.
        wrapped_fn: Wrapped analysis function.
        parallel: Enable parallel processing.
        workers: Number of parallel workers.
        use_threads: Use threads instead of processes.
        progress_callback: Optional progress callback.

    Returns:
        List of analysis result dictionaries.
    """
    total = len(files)
    results: list[dict[str, Any]] = []

    if parallel:
        results = _execute_parallel(
            files, wrapped_fn, workers, use_threads, progress_callback, total
        )
    else:
        results = _execute_sequential(files, wrapped_fn, progress_callback, total)

    return results


def _execute_parallel(
    files: list[str | Path],
    wrapped_fn: Callable[[str | Path], dict[str, Any]],
    workers: int | None,
    use_threads: bool,
    progress_callback: Callable[[int, int, str], None] | None,
    total: int,
) -> list[dict[str, Any]]:
    """Execute analysis in parallel using concurrent.futures.

    Args:
        files: List of file paths.
        wrapped_fn: Wrapped analysis function.
        workers: Number of workers.
        use_threads: Use ThreadPoolExecutor.
        progress_callback: Progress callback.
        total: Total file count.

    Returns:
        List of results.
    """
    results: list[dict[str, Any]] = []
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=workers) as executor:
        future_to_file = {executor.submit(wrapped_fn, f): f for f in files}

        for i, future in enumerate(as_completed(future_to_file), 1):
            filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback(i, total, str(filepath))
            except Exception as e:
                results.append({"file": str(filepath), "error": f"Execution error: {e}"})

    return results


def _execute_sequential(
    files: list[str | Path],
    wrapped_fn: Callable[[str | Path], dict[str, Any]],
    progress_callback: Callable[[int, int, str], None] | None,
    total: int,
) -> list[dict[str, Any]]:
    """Execute analysis sequentially.

    Args:
        files: List of file paths.
        wrapped_fn: Wrapped analysis function.
        progress_callback: Progress callback.
        total: Total file count.

    Returns:
        List of results.
    """
    results: list[dict[str, Any]] = []

    for i, filepath in enumerate(files, 1):
        result = wrapped_fn(filepath)
        results.append(result)

        if progress_callback:
            progress_callback(i, total, str(filepath))

    return results


def _build_result_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build DataFrame from results with column reordering.

    Args:
        results: List of result dictionaries.

    Returns:
        DataFrame with 'file' first, 'error' last.
    """
    df = pd.DataFrame(results)

    # Reorder columns: file first, error last
    cols = df.columns.tolist()
    if "file" in cols:
        cols.remove("file")
        cols = ["file", *cols]
    if "error" in cols:
        cols.remove("error")
        cols = [*cols, "error"]

    return df[cols]
