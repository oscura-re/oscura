"""Oscura Batch Command implementing CLI-004.

Provides CLI for batch processing multiple files with parallel execution support.


Example:
    $ oscura batch '*.wfm' --analysis characterize
    $ oscura batch 'test_*.wfm' --analysis decode --parallel 4
    $ oscura batch 'captures/*.wfm' --analysis spectrum --save-summary results.csv
"""

from __future__ import annotations

import csv
import glob
import logging
from pathlib import Path
from typing import Any

import click

from oscura.cli.main import format_output

logger = logging.getLogger("oscura.cli.batch")


@click.command()
@click.argument("pattern")
@click.option(
    "--analysis",
    type=click.Choice(["characterize", "decode", "spectrum"], case_sensitive=False),
    required=True,
    help="Type of analysis to perform on each file.",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of files to process concurrently (default: 1).",
)
@click.option(
    "--output",
    type=click.Choice(["json", "csv", "html", "table"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--save-summary",
    type=click.Path(),
    default=None,
    help="Save aggregated results to file (CSV format).",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue processing even if individual files fail.",
)
@click.pass_context
def batch(
    ctx: click.Context,
    pattern: str,
    analysis: str,
    parallel: int,
    output: str,
    save_summary: str | None,
    continue_on_error: bool,
) -> None:
    """Batch process multiple files.

    Processes all files matching the given pattern with the specified analysis.
    Supports parallel processing for faster execution on multi-core systems.

    Args:
        ctx: Click context object.
        pattern: Glob pattern to match files.
        analysis: Type of analysis (characterize, decode, spectrum).
        parallel: Number of parallel workers.
        output: Output format (json, csv, html, table).
        save_summary: Path to save CSV summary file.
        continue_on_error: Continue processing if individual files fail.

    Raises:
        Exception: If batch processing fails or no files found.

    Examples:

        \b
        # Process all WFM files with characterization
        $ oscura batch '*.wfm' --analysis characterize

        \b
        # Parallel processing with 4 workers
        $ oscura batch 'test_run_*/*.wfm' \\
            --analysis characterize \\
            --parallel 4 \\
            --save-summary results.csv

        \b
        # Decode all captures, continue on errors
        $ oscura batch 'captures/*.wfm' \\
            --analysis decode \\
            --continue-on-error
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Batch processing pattern: {pattern}")
        logger.info(f"Analysis type: {analysis}")
        logger.info(f"Parallel workers: {parallel}")

    try:
        # Expand glob pattern
        files = glob.glob(pattern, recursive=True)

        if not files:
            click.echo(f"No files matched pattern: {pattern}", err=True)
            ctx.exit(1)

        logger.info(f"Found {len(files)} files to process")

        # Perform batch analysis
        results = _perform_batch_analysis(
            files=files,
            analysis_type=analysis,
            parallel=parallel,
            continue_on_error=continue_on_error,
            verbose=verbose,
        )

        # Save summary if requested
        if save_summary:
            _save_summary(results, save_summary)
            logger.info(f"Summary saved to {save_summary}")

        # Output aggregated results
        summary = _generate_summary(results)
        formatted = format_output(summary, output)
        click.echo(formatted)

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _analyze_single_file(file_path: str, analysis_type: str) -> dict[str, Any]:
    """Analyze a single file and return results.

    Args:
        file_path: Path to waveform file to analyze.
        analysis_type: Type of analysis to perform.

    Returns:
        Dictionary containing analysis results.
    """
    import numpy as np

    from oscura.analyzers.waveform.measurements import fall_time, rise_time
    from oscura.analyzers.waveform.spectral import fft, thd
    from oscura.inference import detect_protocol
    from oscura.loaders import load

    trace = load(file_path)
    sample_rate = trace.metadata.sample_rate

    result: dict[str, Any] = {
        "file": str(Path(file_path).name),
        "status": "success",
        "analysis_type": analysis_type,
        "samples": len(trace.data),  # type: ignore[union-attr]
        "sample_rate": f"{sample_rate / 1e6:.1f} MHz",
    }

    if analysis_type == "characterize":
        rt = rise_time(trace)  # type: ignore[arg-type]
        ft = fall_time(trace)  # type: ignore[arg-type]
        result.update(
            {
                "rise_time": f"{rt * 1e9:.2f} ns" if not np.isnan(rt) else "N/A",
                "fall_time": f"{ft * 1e9:.2f} ns" if not np.isnan(ft) else "N/A",
            }
        )
    elif analysis_type == "decode":
        detected = detect_protocol(trace)  # type: ignore[arg-type]
        result.update(
            {
                "protocol": detected.get("protocol", "unknown"),
                "confidence": f"{detected.get('confidence', 0) * 100:.0f}%",
            }
        )
    elif analysis_type == "spectrum":
        freqs, mags = fft(trace)  # type: ignore[misc, arg-type]
        if len(mags) > 0:
            peak_idx = int(np.argmax(mags))
            peak_freq = freqs[peak_idx]
        else:
            peak_freq = 0.0
        thd_val = thd(trace)  # type: ignore[arg-type]
        result.update(
            {
                "peak_frequency": f"{peak_freq / 1e6:.3f} MHz",
                "thd": f"{thd_val:.1f} dB" if not np.isnan(thd_val) else "N/A",
            }
        )

    return result


def _process_parallel(
    files: list[str],
    analysis_type: str,
    parallel: int,
    continue_on_error: bool,
    verbose: int,
) -> list[dict[str, Any]]:
    """Process files in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_file = {executor.submit(_analyze_single_file, f, analysis_type): f for f in files}

        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            if verbose:
                logger.info(f"[{i}/{len(files)}] Completed {Path(file_path).name}")

            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                if continue_on_error:
                    results.append(
                        {
                            "file": str(Path(file_path).name),
                            "status": "error",
                            "error": str(e),
                        }
                    )
                else:
                    raise

    return results


def _process_sequential(
    files: list[str],
    analysis_type: str,
    continue_on_error: bool,
    verbose: int,
) -> list[dict[str, Any]]:
    """Process files sequentially."""
    results: list[dict[str, Any]] = []

    for i, file_path in enumerate(files, 1):
        if verbose:
            logger.info(f"[{i}/{len(files)}] Processing {file_path}")

        try:
            result = _analyze_single_file(file_path, analysis_type)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            if continue_on_error:
                results.append(
                    {
                        "file": str(Path(file_path).name),
                        "status": "error",
                        "error": str(e),
                    }
                )
            else:
                raise

    return results


def _perform_batch_analysis(
    files: list[str],
    analysis_type: str,
    parallel: int,
    continue_on_error: bool,
    verbose: int,
) -> list[dict[str, Any]]:
    """Perform batch analysis on multiple files.

    Uses concurrent.futures for parallel processing when parallel > 1.

    Args:
        files: List of file paths to process.
        analysis_type: Type of analysis to perform.
        parallel: Number of parallel workers.
        continue_on_error: Whether to continue on errors.
        verbose: Verbosity level.

    Returns:
        List of result dictionaries, one per file.

    Raises:
        Exception: If analysis fails and continue_on_error is False.
    """
    if parallel > 1:
        return _process_parallel(files, analysis_type, parallel, continue_on_error, verbose)
    else:
        return _process_sequential(files, analysis_type, continue_on_error, verbose)


def _generate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate summary statistics from batch results.

    Args:
        results: List of individual file results.

    Returns:
        Summary dictionary with aggregated statistics.
    """
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = total - successful

    summary: dict[str, Any] = {
        "total_files": total,
        "successful": successful,
        "failed": failed,
        "success_rate": f"{successful / total * 100:.1f}%" if total > 0 else "N/A",
    }

    # Add analysis-specific aggregations
    if results and successful > 0:
        summary["note"] = "Detailed per-file results available in JSON/CSV output"

    return summary


def _save_summary(results: list[dict[str, Any]], output_path: str) -> None:
    """Save batch results to CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Path to save CSV file.
    """
    if not results:
        return

    # Get all unique keys across all results
    all_keys: set[str] = set()
    for result in results:
        all_keys.update(result.keys())

    fieldnames = sorted(all_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
