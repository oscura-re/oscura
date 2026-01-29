"""Oscura Compare Command implementing CLI-005.

Provides CLI for comparing two signal captures with timing, noise, and
spectral difference analysis.


Example:
    $ oscura compare before.wfm after.wfm
    $ oscura compare golden.wfm measured.wfm --threshold 5 --save-report diff.html
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from numpy.typing import NDArray
from scipy import fft, signal

from oscura.cli.main import format_output

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace

logger = logging.getLogger("oscura.cli.compare")


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--threshold",
    type=float,
    default=5.0,
    help="Report differences greater than this percentage (default: 5%).",
)
@click.option(
    "--output",
    type=click.Choice(["json", "csv", "html", "table"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.option(
    "--save-report",
    type=click.Path(),
    default=None,
    help="Save detailed HTML comparison report.",
)
@click.option(
    "--align",
    is_flag=True,
    help="Align signals using cross-correlation before comparison.",
)
@click.pass_context
def compare(
    ctx: click.Context,
    file1: str,
    file2: str,
    threshold: float,
    output: str,
    save_report: str | None,
    align: bool,
) -> None:
    """Compare two signal captures.

    Analyzes differences between two waveforms including timing drift,
    amplitude changes, noise variations, and spectral differences.

    Args:
        ctx: Click context object.
        file1: Path to first waveform file.
        file2: Path to second waveform file.
        threshold: Percentage threshold for reporting differences.
        output: Output format (json, csv, html, table).
        save_report: Path to save HTML comparison report.
        align: Align signals using cross-correlation before comparison.

    Raises:
        Exception: If comparison fails or files cannot be loaded.

    Examples:

        \b
        # Simple comparison
        $ oscura compare before.wfm after.wfm

        \b
        # Report only significant differences (>10%)
        $ oscura compare golden.wfm measured.wfm --threshold 10

        \b
        # Full comparison with alignment and HTML report
        $ oscura compare reference.wfm test.wfm \\
            --align \\
            --save-report comparison.html

        \b
        # JSON output for automation
        $ oscura compare before.wfm after.wfm --output json
    """
    verbose = ctx.obj.get("verbose", 0)

    if verbose:
        logger.info(f"Comparing: {file1} vs {file2}")
        logger.info(f"Threshold: {threshold}%")
        logger.info(f"Align signals: {align}")

    try:
        # Import here to avoid circular imports
        from oscura.loaders import load

        # Load both traces
        logger.debug(f"Loading first trace from {file1}")
        trace1 = load(file1)

        logger.debug(f"Loading second trace from {file2}")
        trace2 = load(file2)

        # Perform comparison
        results = _perform_comparison(
            trace1=trace1,  # type: ignore[arg-type]
            trace2=trace2,  # type: ignore[arg-type]
            threshold=threshold,
            align_signals=align,
        )

        # Add metadata
        results["file1"] = str(Path(file1).name)
        results["file2"] = str(Path(file2).name)

        # Generate HTML report if requested
        if save_report:
            html_content = _generate_html_report(results, file1, file2)
            with open(save_report, "w") as f:
                f.write(html_content)
            logger.info(f"Comparison report saved to {save_report}")
            results["report_saved"] = str(save_report)

        # Output results
        formatted = format_output(results, output)
        click.echo(formatted)

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if verbose > 1:
            raise
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _align_signals(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Align two signals using cross-correlation.

    Args:
        data1: Reference signal.
        data2: Signal to align.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (aligned_data1, aligned_data2, alignment_info).
    """
    # Use cross-correlation to find optimal alignment
    # For efficiency, use FFT-based correlation
    n = len(data1) + len(data2) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n)))  # Next power of 2

    # Compute cross-correlation using FFT
    fft1 = fft.fft(data1, n=n_fft)
    fft2 = fft.fft(data2, n=n_fft)
    cross_corr = fft.ifft(fft1 * np.conj(fft2)).real

    # Find peak
    peak_idx = np.argmax(np.abs(cross_corr))
    offset = peak_idx - n_fft if peak_idx > n_fft // 2 else peak_idx

    # Compute correlation coefficient at peak
    corr_peak = cross_corr[peak_idx] / np.sqrt(np.sum(data1**2) * np.sum(data2**2))

    # Apply offset
    if offset > 0:
        aligned1 = data1[offset:]
        aligned2 = data2[: len(aligned1)]
    elif offset < 0:
        aligned2 = data2[-offset:]
        aligned1 = data1[: len(aligned2)]
    else:
        min_len = min(len(data1), len(data2))
        aligned1 = data1[:min_len]
        aligned2 = data2[:min_len]

    # Ensure equal length
    min_len = min(len(aligned1), len(aligned2))
    aligned1 = aligned1[:min_len]
    aligned2 = aligned2[:min_len]

    # Calculate timing offset in ns
    offset_time_ns = offset / sample_rate * 1e9

    alignment_info = {
        "offset_samples": int(offset),
        "offset_time_ns": f"{offset_time_ns:.2f}",
        "correlation_peak": f"{corr_peak:.6f}",
        "quality": "excellent"
        if abs(corr_peak) > 0.95
        else "good"
        if abs(corr_peak) > 0.8
        else "poor",
    }

    return aligned1, aligned2, alignment_info


def _compute_timing_drift(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
) -> dict[str, Any]:
    """Compute timing drift between two signals using edge detection.

    Args:
        data1: Reference signal.
        data2: Comparison signal.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary with timing drift metrics.
    """
    # Find edges using threshold crossing
    threshold1 = (np.max(data1) + np.min(data1)) / 2
    threshold2 = (np.max(data2) + np.min(data2)) / 2

    # Rising edges
    edges1 = np.where(np.diff(data1 > threshold1).astype(int) > 0)[0]
    edges2 = np.where(np.diff(data2 > threshold2).astype(int) > 0)[0]

    if len(edges1) < 2 or len(edges2) < 2:
        return {
            "value_ns": "N/A",
            "percentage": "N/A",
            "significant": False,
            "note": "Insufficient edges for timing analysis",
        }

    # Match edges and compute timing differences
    # Use nearest-neighbor matching
    timing_diffs = []
    for e1 in edges1[: min(100, len(edges1))]:  # Limit to first 100 edges
        nearest_idx = np.argmin(np.abs(edges2 - e1))
        if abs(edges2[nearest_idx] - e1) < sample_rate * 0.1:  # Within 100ms
            timing_diffs.append((edges2[nearest_idx] - e1) / sample_rate)

    if len(timing_diffs) < 3:
        return {
            "value_ns": "N/A",
            "percentage": "N/A",
            "significant": False,
            "note": "Could not match sufficient edges",
        }

    timing_diffs_arr = np.array(timing_diffs)
    mean_drift_ns = float(np.mean(timing_diffs_arr)) * 1e9
    std_drift_ns = float(np.std(timing_diffs_arr)) * 1e9

    # Calculate period for percentage
    periods1 = np.diff(edges1) / sample_rate
    mean_period = float(np.mean(periods1)) if len(periods1) > 0 else 1.0
    mean_diff = float(np.mean(timing_diffs_arr))
    drift_percent = abs(mean_diff / mean_period * 100) if mean_period > 0 else 0.0

    return {
        "value_ns": f"{mean_drift_ns:.2f}",
        "std_ns": f"{std_drift_ns:.2f}",
        "percentage": f"{drift_percent:.4f}%",
        "edges_analyzed": len(timing_diffs_arr),
        "significant": bool(drift_percent > 0.1),  # >0.1% is significant
    }


def _compute_spectral_difference(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
    threshold: float,
) -> dict[str, Any]:
    """Compute spectral differences between two signals.

    Args:
        data1: Reference signal.
        data2: Comparison signal.
        sample_rate: Sample rate in Hz.
        threshold: Percentage threshold for significance.

    Returns:
        Dictionary with spectral comparison metrics.
    """
    # Compute FFT for both signals
    n = len(data1)
    # Use zero-padding for better frequency resolution
    # Pad to at least 10x the original length for good interpolation
    n_fft = 2 ** int(np.ceil(np.log2(n * 10)))

    # Apply window to reduce spectral leakage
    window = signal.windows.hann(n)
    windowed1 = data1 * window
    windowed2 = data2 * window

    # Compute magnitude spectra
    fft1 = np.abs(fft.rfft(windowed1, n=n_fft))
    fft2 = np.abs(fft.rfft(windowed2, n=n_fft))
    freqs = fft.rfftfreq(n_fft, d=1 / sample_rate)

    # Avoid division by zero
    fft1 = np.maximum(fft1, 1e-12)
    fft2 = np.maximum(fft2, 1e-12)

    # Find dominant frequencies
    peak1_idx = np.argmax(fft1[1:]) + 1  # Skip DC
    peak2_idx = np.argmax(fft2[1:]) + 1
    dominant_freq1 = freqs[peak1_idx]
    dominant_freq2 = freqs[peak2_idx]
    freq_diff = abs(dominant_freq2 - dominant_freq1)
    freq_diff_percent = freq_diff / dominant_freq1 * 100 if dominant_freq1 > 0 else 0

    # Compute magnitude differences in dB
    db_diff = 20 * np.log10(fft2 / fft1)
    max_db_diff = np.max(np.abs(db_diff))
    mean_db_diff = np.mean(np.abs(db_diff))

    # Check for harmonic changes
    # Find first 5 harmonics of dominant frequency
    harmonic_changes = []
    for h in range(1, 6):
        harm_freq = dominant_freq1 * h
        harm_idx = int(harm_freq / (sample_rate / n_fft))
        if harm_idx < len(fft1):
            harm_db_diff = 20 * np.log10(fft2[harm_idx] / fft1[harm_idx])
            harmonic_changes.append(
                {
                    "harmonic": h,
                    "frequency_hz": f"{harm_freq:.1f}",
                    "change_db": f"{harm_db_diff:.2f}",
                }
            )

    return {
        "dominant_freq1_hz": f"{dominant_freq1:.1f}",
        "dominant_freq2_hz": f"{dominant_freq2:.1f}",
        "freq_diff_hz": f"{freq_diff:.2f}",
        "freq_diff_percent": f"{freq_diff_percent:.4f}%",
        "max_magnitude_diff_db": f"{max_db_diff:.2f}",
        "mean_magnitude_diff_db": f"{mean_db_diff:.2f}",
        "harmonic_changes": harmonic_changes[:3],  # First 3 harmonics
        "significant": bool(freq_diff_percent > threshold or max_db_diff > 6.0),  # 6dB = 2x power
    }


def _perform_comparison(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    threshold: float,
    align_signals: bool,
) -> dict[str, Any]:
    """Perform comprehensive signal comparison analysis.

    Args:
        trace1: First trace (reference).
        trace2: Second trace (comparison).
        threshold: Percentage threshold for reporting differences.
        align_signals: Whether to align signals using cross-correlation.

    Returns:
        Dictionary of comparison results.
    """
    sample_rate1 = trace1.metadata.sample_rate
    sample_rate2 = trace2.metadata.sample_rate

    rate_mismatch = _check_sample_rate_mismatch(sample_rate1, sample_rate2)

    results: dict[str, Any] = {
        "threshold_percent": threshold,
        "aligned": align_signals,
        "sample_rate_mismatch": rate_mismatch,
    }

    # Add trace statistics
    results["trace1_stats"] = _compute_trace_stats(trace1, sample_rate1)
    results["trace2_stats"] = _compute_trace_stats(trace2, sample_rate2)

    # Prepare and align data
    data1, data2 = _prepare_comparison_data(trace1, trace2, sample_rate1, align_signals, results)

    # Perform analyses
    results["timing_drift"] = _compute_timing_drift(data1, data2, sample_rate1)
    results["amplitude_difference"] = _compute_amplitude_difference(data1, data2, threshold)
    results["noise_change"] = _compute_noise_change(data1, data2, sample_rate1, threshold)
    results["correlation"] = _compute_correlation(data1, data2)
    results["spectral_difference"] = _compute_spectral_difference(
        data1, data2, sample_rate1, threshold
    )

    # Overall assessment
    results["summary"] = _compute_summary(results)

    return results


def _check_sample_rate_mismatch(sample_rate1: float, sample_rate2: float) -> bool:
    """Check if sample rates differ between traces."""
    if sample_rate1 != sample_rate2:
        logger.warning(f"Sample rates differ: {sample_rate1:.2e} vs {sample_rate2:.2e} Hz")
        return True
    return False


def _compute_trace_stats(trace: WaveformTrace, sample_rate: float) -> dict[str, str | int]:
    """Compute statistics for a single trace."""
    return {
        "samples": len(trace.data),
        "sample_rate": f"{sample_rate / 1e6:.2f} MHz",
        "duration_ms": f"{len(trace.data) / sample_rate * 1e3:.3f} ms",
        "mean": f"{float(trace.data.mean()):.6f} V",
        "rms": f"{float(np.sqrt((trace.data**2).mean())):.6f} V",
        "peak_to_peak": f"{float(trace.data.max() - trace.data.min()):.6f} V",
        "min": f"{float(trace.data.min()):.6f} V",
        "max": f"{float(trace.data.max()):.6f} V",
    }


def _prepare_comparison_data(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    sample_rate: float,
    align_signals: bool,
    results: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare data arrays for comparison, with optional alignment."""
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    if align_signals:
        data1, data2, alignment_info = _align_signals(data1, data2, sample_rate)
        results["alignment"] = alignment_info
    else:
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

    return data1, data2


def _compute_amplitude_difference(
    data1: NDArray[np.float64], data2: NDArray[np.float64], threshold: float
) -> dict[str, Any]:
    """Compute amplitude differences between signals."""
    diff = data2 - data1
    abs_diff = np.abs(diff)

    mean1 = data1.mean()
    mean_diff = float(diff.mean())
    mean_diff_percent = abs(mean_diff / mean1 * 100) if mean1 != 0 else 0
    max_diff = float(abs_diff.max())
    rms_diff = float(np.sqrt((diff**2).mean()))

    rms1 = float(np.sqrt((data1**2).mean()))
    rms_diff_percent = rms_diff / rms1 * 100 if rms1 > 0 else 0

    return {
        "mean_diff_v": f"{mean_diff:.6f}",
        "mean_diff_percent": f"{mean_diff_percent:.2f}%",
        "max_diff_v": f"{max_diff:.6f}",
        "rms_diff_v": f"{rms_diff:.6f}",
        "rms_diff_percent": f"{rms_diff_percent:.2f}%",
        "significant": bool(mean_diff_percent > threshold),
    }


def _compute_noise_change(
    data1: NDArray[np.float64], data2: NDArray[np.float64], sample_rate: float, threshold: float
) -> dict[str, Any]:
    """Compute noise level changes between signals."""
    nyquist = sample_rate / 2
    cutoff = min(1000, nyquist * 0.9)
    b, a = signal.butter(4, cutoff / nyquist, btype="high")

    try:
        noise1 = signal.filtfilt(b, a, data1)
        noise2 = signal.filtfilt(b, a, data2)
        noise_std1 = float(np.std(noise1))
        noise_std2 = float(np.std(noise2))
    except Exception:
        noise_std1 = float(np.std(data1))
        noise_std2 = float(np.std(data2))

    noise_change = ((noise_std2 - noise_std1) / noise_std1 * 100) if noise_std1 != 0 else 0

    return {
        "noise1_v": f"{noise_std1:.6f}",
        "noise2_v": f"{noise_std2:.6f}",
        "change_percent": f"{noise_change:.2f}%",
        "significant": bool(abs(noise_change) > threshold),
    }


def _compute_correlation(data1: NDArray[np.float64], data2: NDArray[np.float64]) -> dict[str, str]:
    """Compute correlation coefficient between signals."""
    if len(data1) > 1 and len(data2) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = float(np.corrcoef(data1, data2)[0, 1])

        if correlation > 0.99:
            quality = "excellent"
        elif correlation > 0.95:
            quality = "good"
        elif correlation > 0.8:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "coefficient": f"{correlation:.6f}",
            "quality": quality,
        }

    return {"coefficient": "N/A", "quality": "unknown"}


def _compute_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Compute overall comparison summary."""
    significant_count = sum(
        [
            results.get("amplitude_difference", {}).get("significant", False),
            results.get("noise_change", {}).get("significant", False),
            results.get("timing_drift", {}).get("significant", False),
            results.get("spectral_difference", {}).get("significant", False),
        ]
    )

    if significant_count == 0:
        match_quality = "excellent"
    elif significant_count == 1:
        match_quality = "good"
    elif significant_count == 2:
        match_quality = "fair"
    else:
        match_quality = "poor"

    return {
        "significant_differences": significant_count,
        "overall_match": match_quality,
        "categories_with_differences": [
            cat
            for cat in [
                "amplitude_difference",
                "noise_change",
                "timing_drift",
                "spectral_difference",
            ]
            if results.get(cat, {}).get("significant", False)
        ],
    }


def _generate_html_report(
    results: dict[str, Any],
    file1: str,
    file2: str,
) -> str:
    """Generate HTML comparison report.

    Args:
        results: Comparison results dictionary.
        file1: First file path.
        file2: Second file path.

    Returns:
        HTML content as string.
    """
    summary = results.get("summary", {})
    match_quality = summary.get("overall_match", "unknown")
    significant_diffs = summary.get("significant_differences", 0)

    quality_color = _get_quality_color(match_quality)
    css = _generate_comparison_report_css(quality_color)
    header = _generate_report_header(file1, file2, match_quality, significant_diffs)
    sections = _generate_report_sections(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oscura Signal Comparison Report</title>
    {css}
</head>
<body>
    {header}
    <div class="content">
        {sections}
        <footer style="margin-top: 30px; text-align: center; color: #6c757d;">
            <p>Generated by Oscura - Signal Analysis Toolkit</p>
        </footer>
    </div>
</body>
</html>"""


def _get_quality_color(match_quality: str) -> str:
    """Get color code for match quality level."""
    quality_colors = {
        "excellent": "#28a745",
        "good": "#17a2b8",
        "fair": "#ffc107",
        "poor": "#dc3545",
    }
    return quality_colors.get(match_quality, "#6c757d")


def _generate_comparison_report_css(quality_color: str) -> str:
    """Generate CSS for comparison report."""
    return f"""<style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
        }}
        .content {{
            background: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary {{
            background: {quality_color};
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }}
        .section h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
        }}
        .significant {{
            color: #dc3545;
            font-weight: bold;
        }}
        .ok {{
            color: #28a745;
        }}
    </style>"""


def _generate_report_header(
    file1: str, file2: str, match_quality: str, significant_diffs: int
) -> str:
    """Generate report header with summary."""
    return f"""<div class="header">
        <h1>Oscura Signal Comparison Report</h1>
        <p>File 1: {Path(file1).name}</p>
        <p>File 2: {Path(file2).name}</p>
    </div>

    <div class="content">
        <div class="summary">
            <h2>Overall Match: {match_quality.upper()}</h2>
            <p>{significant_diffs} significant difference(s) detected</p>
        </div>"""


def _generate_report_sections(results: dict[str, Any]) -> str:
    """Generate all report sections."""
    stats = _generate_trace_stats_section(results)
    amplitude = _generate_amplitude_diff_section(results)
    timing = _generate_timing_drift_section(results)
    noise = _generate_noise_change_section(results)
    spectral = _generate_spectral_diff_section(results)
    correlation = _generate_correlation_section(results)

    return f"{stats}\n{amplitude}\n{timing}\n{noise}\n{spectral}\n{correlation}"


def _generate_trace_stats_section(results: dict[str, Any]) -> str:
    """Generate trace statistics comparison table."""
    t1 = results.get("trace1_stats", {})
    t2 = results.get("trace2_stats", {})

    return f"""<div class="section">
            <h3>Trace Statistics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Trace 1</th>
                    <th>Trace 2</th>
                </tr>
                <tr>
                    <td>Samples</td>
                    <td>{t1.get("samples", "N/A")}</td>
                    <td>{t2.get("samples", "N/A")}</td>
                </tr>
                <tr>
                    <td>Sample Rate</td>
                    <td>{t1.get("sample_rate", "N/A")}</td>
                    <td>{t2.get("sample_rate", "N/A")}</td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{t1.get("mean", "N/A")}</td>
                    <td>{t2.get("mean", "N/A")}</td>
                </tr>
                <tr>
                    <td>RMS</td>
                    <td>{t1.get("rms", "N/A")}</td>
                    <td>{t2.get("rms", "N/A")}</td>
                </tr>
                <tr>
                    <td>Peak-to-Peak</td>
                    <td>{t1.get("peak_to_peak", "N/A")}</td>
                    <td>{t2.get("peak_to_peak", "N/A")}</td>
                </tr>
            </table>
        </div>"""


def _generate_amplitude_diff_section(results: dict[str, Any]) -> str:
    """Generate amplitude difference section."""
    amp = results.get("amplitude_difference", {})
    sig_class = "significant" if amp.get("significant") else "ok"

    return f"""<div class="section">
            <h3>Amplitude Difference</h3>
            <table>
                <tr>
                    <td>Mean Difference</td>
                    <td>{amp.get("mean_diff_v", "N/A")}</td>
                    <td class="{sig_class}">
                        {amp.get("mean_diff_percent", "N/A")}
                    </td>
                </tr>
                <tr>
                    <td>RMS Difference</td>
                    <td>{amp.get("rms_diff_v", "N/A")}</td>
                    <td>{amp.get("rms_diff_percent", "N/A")}</td>
                </tr>
                <tr>
                    <td>Max Difference</td>
                    <td colspan="2">{amp.get("max_diff_v", "N/A")}</td>
                </tr>
            </table>
        </div>"""


def _generate_timing_drift_section(results: dict[str, Any]) -> str:
    """Generate timing drift section."""
    drift = results.get("timing_drift", {})
    sig_class = "significant" if drift.get("significant") else "ok"

    return f"""<div class="section">
            <h3>Timing Drift</h3>
            <table>
                <tr>
                    <td>Mean Drift</td>
                    <td>{drift.get("value_ns", "N/A")} ns</td>
                    <td class="{sig_class}">
                        {drift.get("percentage", "N/A")}
                    </td>
                </tr>
            </table>
        </div>"""


def _generate_noise_change_section(results: dict[str, Any]) -> str:
    """Generate noise change section."""
    noise = results.get("noise_change", {})
    sig_class = "significant" if noise.get("significant") else "ok"

    return f"""<div class="section">
            <h3>Noise Change</h3>
            <table>
                <tr>
                    <td>Trace 1 Noise</td>
                    <td>{noise.get("noise1_v", "N/A")}</td>
                </tr>
                <tr>
                    <td>Trace 2 Noise</td>
                    <td>{noise.get("noise2_v", "N/A")}</td>
                </tr>
                <tr>
                    <td>Change</td>
                    <td class="{sig_class}">
                        {noise.get("change_percent", "N/A")}
                    </td>
                </tr>
            </table>
        </div>"""


def _generate_spectral_diff_section(results: dict[str, Any]) -> str:
    """Generate spectral difference section."""
    spec = results.get("spectral_difference", {})
    sig_class = "significant" if spec.get("significant") else "ok"

    return f"""<div class="section">
            <h3>Spectral Difference</h3>
            <table>
                <tr>
                    <td>Dominant Frequency 1</td>
                    <td>{spec.get("dominant_freq1_hz", "N/A")} Hz</td>
                </tr>
                <tr>
                    <td>Dominant Frequency 2</td>
                    <td>{spec.get("dominant_freq2_hz", "N/A")} Hz</td>
                </tr>
                <tr>
                    <td>Frequency Difference</td>
                    <td class="{sig_class}">
                        {spec.get("freq_diff_percent", "N/A")}
                    </td>
                </tr>
                <tr>
                    <td>Max Magnitude Difference</td>
                    <td>{spec.get("max_magnitude_diff_db", "N/A")} dB</td>
                </tr>
            </table>
        </div>"""


def _generate_correlation_section(results: dict[str, Any]) -> str:
    """Generate correlation section."""
    corr = results.get("correlation", {})

    return f"""<div class="section">
            <h3>Correlation</h3>
            <p>Coefficient: {corr.get("coefficient", "N/A")}</p>
            <p>Quality: {corr.get("quality", "N/A")}</p>
        </div>"""
