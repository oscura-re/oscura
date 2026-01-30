"""Complete waveform analysis workflow orchestration.

This module provides high-level APIs for comprehensive waveform analysis,
automating the entire pipeline from loading to reporting.

Example:
    >>> from oscura.workflows import waveform
    >>> results = waveform.analyze_complete(
    ...     "signal.wfm",
    ...     output_dir="./analysis_output",
    ...     generate_plots=True,
    ...     generate_report=True
    ... )
    >>> print(f"Report saved: {results['report_path']}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

import oscura as osc
from oscura.core.types import DigitalTrace, WaveformTrace


def analyze_complete(
    filepath: str | Path,
    *,
    output_dir: str | Path | None = None,
    analyses: list[str] | Literal["all"] = "all",
    generate_plots: bool = True,
    generate_report: bool = True,
    embed_plots: bool = True,
    report_format: str = "html",
    verbose: bool = True,
) -> dict[str, Any]:
    """Perform complete waveform analysis workflow.

    This orchestrates the entire analysis pipeline:
    1. Load waveform file (auto-detects format)
    2. Detect signal type (analog/digital)
    3. Run requested analyses (time/frequency/digital/statistical domains)
    4. Generate plots (optional)
    5. Create professional report (optional)

    Args:
        filepath: Path to waveform file (.wfm, .tss, .csv, etc.).
        output_dir: Output directory for plots and reports.
                   Defaults to "./waveform_analysis_output".
        analyses: List of analysis types to run or "all".
                 Options: "time_domain", "frequency_domain", "digital", "statistics".
        generate_plots: Whether to generate visualization plots.
        generate_report: Whether to generate HTML/PDF report.
        embed_plots: Whether to embed plots in report (vs external files).
        report_format: Report format ("html" or "pdf").
        verbose: Print progress messages.

    Returns:
        Dictionary containing:
            - "filepath": Input file path
            - "trace": Loaded trace object
            - "is_digital": Boolean indicating digital signal
            - "results": Dict of analysis results by domain
            - "plots": Dict of plot data (if generate_plots=True)
            - "report_path": Path to generated report (if generate_report=True)
            - "output_dir": Output directory path

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If analyses contains invalid analysis type.

    Example:
        >>> # Minimal usage
        >>> results = analyze_complete("signal.wfm")

        >>> # Custom configuration
        >>> results = analyze_complete(
        ...     "complex_signal.tss",
        ...     output_dir="./my_analysis",
        ...     analyses=["time_domain", "frequency_domain"],
        ...     generate_plots=True,
        ...     generate_report=True
        ... )

        >>> # Access results
        >>> freq_results = results["results"]["frequency_domain"]
        >>> print(f"THD: {freq_results['thd']:.2f}%")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Set up output directory
    if output_dir is None:
        output_dir = Path("./waveform_analysis_output")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which analyses to run
    valid_analyses = {"time_domain", "frequency_domain", "digital", "statistics"}
    if analyses == "all":
        requested_analyses = list(valid_analyses)
    else:
        requested_analyses = analyses
        invalid = set(requested_analyses) - valid_analyses
        if invalid:
            raise ValueError(f"Invalid analysis types: {invalid}. Valid: {valid_analyses}")

    if verbose:
        print("=" * 80)
        print("OSCURA COMPLETE WAVEFORM ANALYSIS")
        print("=" * 80)
        print(f"\nLoading: {filepath.name}")

    # Step 1: Load waveform
    trace = osc.load(filepath)

    # Detect signal type using new properties
    is_digital = (
        trace.is_digital if hasattr(trace, "is_digital") else isinstance(trace, DigitalTrace)
    )

    if verbose:
        signal_type = "Digital" if is_digital else "Analog"
        print(f"✓ Loaded {signal_type} signal")
        print(f"  Samples: {len(trace)}")
        print(f"  Sample rate: {trace.metadata.sample_rate:.2e} Hz")
        print(f"  Duration: {trace.duration:.6f} s")

    # Step 2: Run analyses
    results: dict[str, dict[str, Any]] = {}

    if "time_domain" in requested_analyses:
        if verbose:
            print("\n" + "=" * 80)
            print("TIME-DOMAIN ANALYSIS")
            print("=" * 80)

        # Run time-domain measurements
        if isinstance(trace, WaveformTrace):
            from oscura.analyzers import waveform as waveform_analyzer

            time_results = waveform_analyzer.measure(
                trace,
                parameters=[
                    "amplitude",
                    "mean",
                    "rms",
                    "frequency",
                    "period",
                    "duty_cycle",
                    "rise_time",
                    "fall_time",
                    "overshoot",
                    "undershoot",
                ],
            )
            results["time_domain"] = time_results
            if verbose:
                print(f"✓ Completed {len(time_results)} measurements")

    if "frequency_domain" in requested_analyses and not is_digital:
        if verbose:
            print("\n" + "=" * 80)
            print("FREQUENCY-DOMAIN ANALYSIS")
            print("=" * 80)

        if isinstance(trace, WaveformTrace):
            # Run spectral analysis using top-level APIs
            fft_result = osc.fft(trace)
            # fft returns tuple: (freqs, magnitudes) or (freqs, magnitudes, phases)
            freqs_array = fft_result[0]
            mags_array = fft_result[1]

            # Find dominant frequency
            dominant_idx = int(np.argmax(np.abs(mags_array[1:]))) + 1  # Skip DC
            dominant_freq = float(freqs_array[dominant_idx])

            freq_results: dict[str, Any] = {
                "dominant_freq": dominant_freq,
                "fft_freqs": freqs_array,
                "fft_data": mags_array,
            }

            # Add quality metrics
            try:
                from oscura.analyzers.waveform.spectral import enob, sfdr, sinad, snr, thd

                freq_results["thd"] = thd(trace)
                freq_results["snr"] = snr(trace)
                freq_results["sinad"] = sinad(trace)
                freq_results["enob"] = enob(trace)
                freq_results["sfdr"] = sfdr(trace)
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Some spectral metrics unavailable: {e}")

            results["frequency_domain"] = freq_results
            if verbose:
                numeric_count = sum(1 for v in freq_results.values() if isinstance(v, (int, float)))
                print(f"✓ Completed {numeric_count} measurements")

    if "digital" in requested_analyses:
        if verbose:
            print("\n" + "=" * 80)
            print("DIGITAL SIGNAL ANALYSIS")
            print("=" * 80)

        # Run digital analysis (works for both analog and digital traces)
        try:
            from oscura.analyzers.digital import signal_quality_summary

            # Type narrow to WaveformTrace for signal_quality_summary
            if isinstance(trace, WaveformTrace):
                digital_results_obj = signal_quality_summary(trace)
                # Convert dataclass to dict if needed
                digital_results: dict[str, Any]
                if hasattr(digital_results_obj, "__dict__"):
                    digital_results = digital_results_obj.__dict__
                else:
                    # Assume it's already a dict-like object
                    digital_results = dict(digital_results_obj)
                results["digital"] = digital_results
                if verbose:
                    numeric_count = sum(
                        1 for v in digital_results.values() if isinstance(v, (int, float))
                    )
                    print(f"✓ Completed {numeric_count} measurements")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Digital analysis unavailable: {e}")

    if "statistics" in requested_analyses and not is_digital:
        if verbose:
            print("\n" + "=" * 80)
            print("STATISTICAL ANALYSIS")
            print("=" * 80)

        if isinstance(trace, WaveformTrace):
            # Run statistical analysis
            from oscura.analyzers.statistical import basic_stats, percentiles

            stats_results = basic_stats(trace.data)
            # Add percentiles
            try:
                p_dict = percentiles(trace.data, [1, 5, 25, 75, 95, 99])
                # percentiles returns a dict, not a list
                if isinstance(p_dict, dict):
                    stats_results.update(p_dict)
            except Exception:
                pass

            results["statistics"] = stats_results
            if verbose:
                numeric_count = sum(
                    1 for v in stats_results.values() if isinstance(v, (int, float))
                )
                print(f"✓ Completed {numeric_count} measurements")

    # Step 3: Generate plots
    plots: dict[str, str] = {}
    if generate_plots:
        if verbose:
            print("\n" + "=" * 80)
            print("GENERATING PLOTS")
            print("=" * 80)

        from oscura.visualization import batch

        # Type narrowing for batch.generate_all_plots
        if isinstance(trace, (WaveformTrace, DigitalTrace)):
            plots = batch.generate_all_plots(trace, verbose=verbose)
        else:
            # IQTrace not supported by batch plotting yet
            if verbose:
                print("  ⚠ Batch plotting not available for I/Q traces")
            plots = {}

        if verbose:
            print(f"✓ Generated {len(plots)} plots")

    # Step 4: Generate report
    report_path: Path | None = None
    if generate_report:
        if verbose:
            print("\n" + "=" * 80)
            print("GENERATING REPORT")
            print("=" * 80)

        from oscura.reporting import Report, ReportConfig, generate_html_report

        # Create report
        # Cast report_format to valid Literal type
        valid_format: Literal["html", "pdf", "markdown", "docx"] = (
            "html" if report_format == "html" else "pdf"
        )
        config = ReportConfig(
            title="Comprehensive Waveform Analysis",
            format=valid_format,
            verbosity="detailed",
        )

        report = Report(
            config=config,
            metadata={
                "file": str(filepath),
                "type": "Digital" if is_digital else "Analog",
            },
        )

        # Add measurement sections using add_measurements() API
        for analysis_name, analysis_results in results.items():
            # Filter to numeric measurements only
            measurements = {
                k: v
                for k, v in analysis_results.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }

            if measurements:
                title_map = {
                    "time_domain": "Time-Domain Analysis (IEEE 181-2011)",
                    "frequency_domain": "Frequency-Domain Analysis (IEEE 1241-2010)",
                    "digital": "Digital Signal Analysis",
                    "statistics": "Statistical Analysis",
                }
                title = title_map.get(analysis_name, analysis_name.replace("_", " ").title())
                report.add_measurements(title=title, measurements=measurements)

        # Generate HTML
        html_content = generate_html_report(report)

        # Embed plots if requested
        if embed_plots and plots:
            from oscura.reporting import embed_plots as embed_plots_func

            html_content = embed_plots_func(html_content, plots)
            if verbose:
                print(f"  ✓ Embedded {len(plots)} plots in report")

        # Save report
        report_path = output_dir / f"analysis_report.{report_format}"
        report_path.write_text(html_content, encoding="utf-8")

        if verbose:
            print(f"✓ Report saved: {report_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"✓ Output directory: {output_dir}")

    # Return comprehensive results
    return {
        "filepath": filepath,
        "trace": trace,
        "is_digital": is_digital,
        "results": results,
        "plots": plots if generate_plots else {},
        "report_path": report_path,
        "output_dir": output_dir,
    }


__all__ = [
    "analyze_complete",
]
