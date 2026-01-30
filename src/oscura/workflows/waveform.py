"""Complete waveform analysis workflow with reverse engineering capabilities.

This module provides high-level APIs for comprehensive waveform analysis,
automating the entire pipeline from loading to protocol reverse engineering.

Example:
    >>> from oscura.workflows import waveform
    >>> # Complete analysis including reverse engineering
    >>> results = waveform.analyze_complete(
    ...     "unknown_signal.wfm",
    ...     output_dir="./analysis_output",
    ...     enable_protocol_decode=True,
    ...     enable_reverse_engineering=True,
    ...     generate_plots=True,
    ...     generate_report=True
    ... )
    >>> print(f"Detected protocols: {results['protocols_detected']}")
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
    # Phase 3: Advanced capabilities
    enable_protocol_decode: bool = True,
    enable_reverse_engineering: bool = True,
    enable_pattern_recognition: bool = True,
    protocol_hints: list[str] | None = None,
    reverse_engineering_depth: Literal["quick", "standard", "deep"] = "standard",
    verbose: bool = True,
) -> dict[str, Any]:
    """Perform complete waveform analysis workflow with reverse engineering.

    This orchestrates the entire analysis pipeline:
    1. Load waveform file (auto-detects format)
    2. Detect signal type (analog/digital)
    3. Run basic analyses (time/frequency/digital/statistical domains)
    4. Protocol detection and decoding (for digital signals)
    5. Reverse engineering pipeline (clock recovery, framing, CRC analysis)
    6. Pattern recognition and state machine inference
    7. Generate comprehensive visualizations
    8. Create professional report with all findings

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
        enable_protocol_decode: Enable automatic protocol detection and decoding.
        enable_reverse_engineering: Enable reverse engineering pipeline.
        enable_pattern_recognition: Enable pattern mining and state machine inference.
        protocol_hints: Optional protocol hints for decoder (e.g., ["uart", "spi"]).
        reverse_engineering_depth: RE analysis depth ("quick", "standard", "deep").
        verbose: Print progress messages.

    Returns:
        Dictionary containing:
            - "filepath": Input file path
            - "trace": Loaded trace object
            - "is_digital": Boolean indicating digital signal
            - "results": Dict of analysis results by domain
            - "protocols_detected": List of detected protocols (if enabled)
            - "decoded_frames": List of decoded protocol frames (if enabled)
            - "reverse_engineering": RE analysis results (if enabled)
            - "patterns": Pattern recognition results (if enabled)
            - "plots": Dict of plot data (if generate_plots=True)
            - "report_path": Path to generated report (if generate_report=True)
            - "output_dir": Output directory path

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If analyses contains invalid analysis type.

    Example:
        >>> # Minimal usage - full analysis with defaults
        >>> results = analyze_complete("signal.wfm")

        >>> # Custom configuration
        >>> results = analyze_complete(
        ...     "complex_signal.tss",
        ...     output_dir="./my_analysis",
        ...     analyses=["time_domain", "frequency_domain"],
        ...     enable_protocol_decode=True,
        ...     protocol_hints=["uart", "spi"],
        ...     reverse_engineering_depth="deep",
        ...     generate_plots=True,
        ...     generate_report=True
        ... )

        >>> # Access results
        >>> if results["protocols_detected"]:
        ...     for proto in results["protocols_detected"]:
        ...         print(f"Found {proto['protocol']} at {proto['baud_rate']} baud")
        >>> if results["reverse_engineering"]:
        ...     print(f"CRC: {results['reverse_engineering']['crc_parameters']}")
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
        print("OSCURA COMPLETE WAVEFORM ANALYSIS WITH REVERSE ENGINEERING")
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

    # Step 2: Run basic analyses
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

            if isinstance(trace, WaveformTrace):
                digital_results_obj = signal_quality_summary(trace)
                digital_results: dict[str, Any]
                if hasattr(digital_results_obj, "__dict__"):
                    digital_results = digital_results_obj.__dict__
                else:
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

    # Step 3: Protocol Detection & Decoding (Phase 3A)
    protocols_detected: list[dict[str, Any]] = []
    decoded_frames: list[Any] = []

    if enable_protocol_decode and is_digital:
        if verbose:
            print("\n" + "=" * 80)
            print("PROTOCOL DETECTION & DECODING")
            print("=" * 80)

        try:
            from oscura.discovery import auto_decoder

            # Try each protocol hint or auto-detect
            protocols_to_try = protocol_hints if protocol_hints else ["UART", "SPI", "I2C"]

            for proto_name in protocols_to_try:
                try:
                    # Type narrow to WaveformTrace | DigitalTrace
                    if not isinstance(trace, (WaveformTrace, DigitalTrace)):
                        continue

                    result = auto_decoder.decode_protocol(
                        trace,
                        protocol_hint=proto_name.upper(),  # type: ignore[arg-type]
                        confidence_threshold=0.7,
                    )

                    if result.overall_confidence >= 0.7:
                        proto_info = {
                            "protocol": result.protocol,
                            "confidence": result.overall_confidence,
                            "params": result.detected_params,
                        }
                        protocols_detected.append(proto_info)
                        decoded_frames.extend(result.data)

                        if verbose:
                            print(
                                f"✓ Detected {result.protocol.upper()}: "
                                f"{result.overall_confidence:.1%} confidence"
                            )
                            print(f"  Decoded {len(result.data)} bytes")
                except Exception:
                    # Protocol didn't match, continue trying others
                    pass

            if not protocols_detected and verbose:
                print("  ⚠ No protocols detected (signal may be unknown or noisy)")

        except Exception as e:
            if verbose:
                print(f"  ⚠ Protocol detection unavailable: {e}")

    # Step 4: Reverse Engineering Pipeline (Phase 3B)
    reverse_engineering_results: dict[str, Any] | None = None

    if enable_reverse_engineering and is_digital:
        if verbose:
            print("\n" + "=" * 80)
            print("REVERSE ENGINEERING ANALYSIS")
            print("=" * 80)
            depth_str = {"quick": "Quick", "standard": "Standard", "deep": "Deep"}
            print(f"  Mode: {depth_str[reverse_engineering_depth]}")
            print("  (Note: RE pipeline integration in progress)")

        # RE pipeline integration - placeholder for now
        # Will integrate oscura.workflows.reverse_engineering when available
        reverse_engineering_results = {
            "status": "experimental",
            "note": "Full RE pipeline integration in progress",
        }

    # Step 5: Pattern Recognition & State Machine Inference (Phase 3C)
    pattern_results: dict[str, Any] | None = None

    if enable_pattern_recognition:
        if verbose:
            print("\n" + "=" * 80)
            print("PATTERN RECOGNITION & INFERENCE")
            print("=" * 80)
            print("  (Note: Pattern recognition integration in progress)")

        # Pattern recognition integration - placeholder for now
        pattern_results = {
            "status": "experimental",
            "note": "Pattern recognition APIs integration in progress",
        }

    # Step 6: Generate plots
    plots: dict[str, str] = {}
    if generate_plots:
        if verbose:
            print("\n" + "=" * 80)
            print("GENERATING VISUALIZATIONS")
            print("=" * 80)

        from oscura.visualization import batch

        # Basic plots
        if isinstance(trace, (WaveformTrace, DigitalTrace)):
            plots = batch.generate_all_plots(trace, verbose=verbose)

        if verbose:
            print(f"✓ Generated {len(plots)} total plots")

    # Step 7: Generate comprehensive report
    report_path: Path | None = None
    if generate_report:
        if verbose:
            print("\n" + "=" * 80)
            print("GENERATING COMPREHENSIVE REPORT")
            print("=" * 80)

        from oscura.reporting import Report, ReportConfig, generate_html_report

        # Create report
        valid_format: Literal["html", "pdf", "markdown", "docx"] = (
            "html" if report_format == "html" else "pdf"
        )
        config = ReportConfig(
            title="Complete Waveform Analysis with Reverse Engineering",
            format=valid_format,
            verbosity="detailed",
        )

        report = Report(
            config=config,
            metadata={
                "file": str(filepath),
                "type": "Digital" if is_digital else "Analog",
                "protocols_detected": len(protocols_detected),
                "frames_decoded": len(decoded_frames),
            },
        )

        # Add basic measurement sections
        for analysis_name, analysis_results in results.items():
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

        # Add protocol detection section
        if protocols_detected:
            report.add_section(
                title="Protocol Detection Results",
                content=_format_protocol_detection(protocols_detected, decoded_frames),
            )

        # Add reverse engineering section
        if reverse_engineering_results:
            report.add_section(
                title="Reverse Engineering Analysis",
                content=_format_reverse_engineering(reverse_engineering_results),
            )

        # Add pattern recognition section
        if pattern_results:
            report.add_section(
                title="Pattern Recognition & Inference",
                content=_format_pattern_recognition(pattern_results),
            )

        # Generate HTML
        html_content = generate_html_report(report)

        # Embed plots if requested
        if embed_plots and plots:
            from oscura.reporting import embed_plots as embed_plots_func

            html_content = embed_plots_func(html_content, plots)
            if verbose:
                print(f"  ✓ Embedded {len(plots)} plots in report")

        # Save report
        report_path = output_dir / f"complete_analysis.{report_format}"
        report_path.write_text(html_content, encoding="utf-8")

        if verbose:
            print(f"✓ Report saved: {report_path}")

    if verbose:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"✓ Output directory: {output_dir}")
        if protocols_detected:
            print(f"✓ Protocols detected: {len(protocols_detected)}")
        if decoded_frames:
            print(f"✓ Frames decoded: {len(decoded_frames)}")

    # Return comprehensive results
    return {
        "filepath": filepath,
        "trace": trace,
        "is_digital": is_digital,
        "results": results,
        "protocols_detected": protocols_detected,
        "decoded_frames": decoded_frames,
        "reverse_engineering": reverse_engineering_results,
        "patterns": pattern_results,
        "plots": plots if generate_plots else {},
        "report_path": report_path,
        "output_dir": output_dir,
    }


def _format_protocol_detection(protocols: list[dict[str, Any]], frames: list[Any]) -> str:
    """Format protocol detection results for report.

    Args:
        protocols: List of detected protocols.
        frames: List of decoded frames.

    Returns:
        HTML formatted string.
    """
    html = "<h3>Detected Protocols</h3>\n<ul>\n"
    for proto in protocols:
        conf = proto.get("confidence", 0.0)
        html += f"<li><strong>{proto['protocol'].upper()}</strong>: {conf:.1%} confidence"
        if "params" in proto and "baud_rate" in proto["params"]:
            html += f" at {proto['params']['baud_rate']:.0f} baud"
        html += "</li>\n"
    html += "</ul>\n"

    if frames:
        html += f"<p><strong>Total frames decoded:</strong> {len(frames)}</p>\n"

    return html


def _format_reverse_engineering(re_results: dict[str, Any]) -> str:
    """Format reverse engineering results for report.

    Args:
        re_results: RE analysis results dictionary.

    Returns:
        HTML formatted string.
    """
    html = "<h3>Reverse Engineering Status</h3>\n"

    if re_results.get("status") == "experimental":
        html += f"<p><em>{re_results.get('note', 'Integration in progress')}</em></p>\n"
    else:
        html += "<ul>\n"

        if re_results.get("baud_rate"):
            html += f"<li><strong>Baud Rate:</strong> {re_results['baud_rate']:.0f} Hz</li>\n"

        if re_results.get("frame_format"):
            html += f"<li><strong>Frame Format:</strong> {re_results['frame_format']}</li>\n"

        if re_results.get("sync_patterns"):
            html += f"<li><strong>Sync Pattern:</strong> {re_results['sync_patterns']}</li>\n"

        if re_results.get("crc_parameters"):
            html += "<li><strong>CRC:</strong> Detected</li>\n"

        if re_results.get("confidence"):
            conf = re_results["confidence"]
            html += f"<li><strong>Confidence:</strong> {conf:.1%}</li>\n"

        html += "</ul>\n"

    return html


def _format_pattern_recognition(pattern_results: dict[str, Any]) -> str:
    """Format pattern recognition results for report.

    Args:
        pattern_results: Pattern analysis results dictionary.

    Returns:
        HTML formatted string.
    """
    html = "<h3>Pattern Recognition Status</h3>\n"

    if pattern_results.get("status") == "experimental":
        html += f"<p><em>{pattern_results.get('note', 'Integration in progress')}</em></p>\n"
    else:
        html += "<ul>\n"

        if pattern_results.get("anomalies"):
            anomalies = pattern_results["anomalies"]
            html += f"<li><strong>Anomalies Detected:</strong> {len(anomalies)}</li>\n"

        if pattern_results.get("patterns"):
            patterns = pattern_results["patterns"]
            html += f"<li><strong>Patterns Discovered:</strong> {len(patterns)}</li>\n"

        if pattern_results.get("state_machine"):
            sm = pattern_results["state_machine"]
            state_count = len(sm.states) if hasattr(sm, "states") else 0
            html += f"<li><strong>State Machine:</strong> {state_count} states inferred</li>\n"

        html += "</ul>\n"

    return html


__all__ = [
    "analyze_complete",
]
