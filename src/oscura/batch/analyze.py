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

    # Wrapper to include config in analysis calls
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

    results: list[dict[str, Any]] = []
    total = len(files)

    if parallel:
        # Use concurrent.futures for parallel execution
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with executor_class(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(_wrapped_analysis, f): f for f in files}

            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress_callback(i, total, str(filepath))
                except Exception as e:
                    # Catch execution errors
                    results.append(
                        {
                            "file": str(filepath),
                            "error": f"Execution error: {e}",
                        }
                    )

    else:
        # Sequential processing
        for i, filepath in enumerate(files, 1):
            result = _wrapped_analysis(filepath)
            results.append(result)

            if progress_callback:
                progress_callback(i, total, str(filepath))

    # Convert to DataFrame
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
