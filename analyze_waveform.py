#!/usr/bin/env python3
"""Complete Waveform Analysis with Reverse Engineering - Reference Implementation.

This script demonstrates optimal usage of the Oscura framework for comprehensive
waveform analysis, including protocol decoding and reverse engineering.

Features:
- Automatic file format detection (.wfm, .tss, .csv, etc.)
- Complete multi-domain analysis (time, frequency, digital, statistical)
- Protocol detection and decoding (UART, SPI, I2C, CAN, etc.)
- Reverse engineering pipeline (clock recovery, framing, CRC analysis)
- Pattern recognition and state machine inference
- Professional visualization with IEEE publication standards
- Comprehensive HTML reports with all findings

Example:
    # Complete analysis with all features
    $ python3 analyze_waveform.py signal.wfm

    # Digital signal with protocol decoding
    $ python3 analyze_waveform.py capture.tss --output ./analysis

    # Quick analysis without reverse engineering
    $ python3 analyze_waveform.py data.csv --no-reverse-engineering

    # Deep RE analysis with protocol hints
    $ python3 analyze_waveform.py unknown.wfm --re-depth deep --protocol-hints uart spi

References:
    IEEE 181-2011: Transitional Waveform Definitions
    IEEE 1241-2010: ADC Terminology and Test Methods
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal


def print_detailed_measurements(results: dict[str, Any]) -> None:
    """Print ALL measurement values with proper formatting.

    Args:
        results: Results dictionary from analyze_complete().
    """
    from oscura.analyzers.waveform import MEASUREMENT_METADATA

    print("\n" + "=" * 80)
    print("DETAILED MEASUREMENT RESULTS")
    print("=" * 80)

    # Signal information
    trace = results["trace"]
    print("\nSIGNAL INFORMATION:")
    print("-" * 80)
    print(f"  File                          : {results['filepath'].name}")
    print(f"  Signal Type                   : {'Digital' if results['is_digital'] else 'Analog'}")
    print(f"  Samples                       : {len(trace):,}")
    print(
        f"  Sample Rate                   : {_format_with_si_prefix(trace.metadata.sample_rate, 'Hz')}"
    )
    print(f"  Duration                      : {_format_with_si_prefix(trace.duration, 's')}")
    if hasattr(trace.metadata, "channel_name") and trace.metadata.channel_name:
        print(f"  Channel                       : {trace.metadata.channel_name}")
    if hasattr(trace.metadata, "vertical_scale") and trace.metadata.vertical_scale:
        print(
            f"  Vertical Scale                : {_format_with_si_prefix(trace.metadata.vertical_scale, 'V/div')}"
        )
    if hasattr(trace.metadata, "horizontal_scale") and trace.metadata.horizontal_scale:
        print(
            f"  Horizontal Scale              : {_format_with_si_prefix(trace.metadata.horizontal_scale, 's/div')}"
        )

    # Print each analysis domain
    for domain, measurements in results["results"].items():
        print(f"\n{domain.replace('_', ' ').upper()}:")
        print("-" * 80)

        for param, value in measurements.items():
            # Handle dict-style measurements (e.g., time_domain)
            if isinstance(value, dict) and "value" in value:
                numeric_value = value["value"]
                unit = value.get("unit", "")
            # Handle flat numeric measurements (e.g., frequency_domain)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = value
                # Get unit from metadata
                unit = ""
                if param in MEASUREMENT_METADATA:
                    unit = MEASUREMENT_METADATA[param].get("unit", "")
            else:
                # Skip non-numeric fields (arrays, bools, strings, etc.)
                continue

            # Format value with SI prefix
            formatted_value = _format_with_si_prefix(numeric_value, unit)

            # Print measurement
            param_display = param.replace("_", " ").title()
            print(f"  {param_display:30s}: {formatted_value}")


def _format_with_si_prefix(value: float, unit: str) -> str:
    """Format value with appropriate SI prefix.

    Args:
        value: Numeric value to format.
        unit: Unit string (e.g., "Hz", "V", "s").

    Returns:
        Formatted string with SI prefix.
    """
    if unit in ["ratio", "dimensionless", ""]:
        return f"{value:.6g}"

    if unit == "%":
        return f"{value:.2f}%"

    if unit == "dB":
        return f"{value:.2f} dB"

    # SI prefixes for standard units
    abs_value = abs(value)

    if abs_value == 0:
        return f"0 {unit}"

    if abs_value >= 1e9:
        return f"{value / 1e9:.3f} G{unit}"
    elif abs_value >= 1e6:
        return f"{value / 1e6:.3f} M{unit}"
    elif abs_value >= 1e3:
        return f"{value / 1e3:.3f} k{unit}"
    elif abs_value >= 1:
        return f"{value:.3f} {unit}"
    elif abs_value >= 1e-3:
        return f"{value * 1e3:.3f} m{unit}"
    elif abs_value >= 1e-6:
        return f"{value * 1e6:.3f} µ{unit}"
    elif abs_value >= 1e-9:
        return f"{value * 1e9:.3f} n{unit}"
    elif abs_value >= 1e-12:
        return f"{value * 1e12:.3f} p{unit}"
    else:
        return f"{value:.3e} {unit}"


def print_summary(results: dict[str, Any]) -> None:
    """Print comprehensive analysis summary.

    Args:
        results: Results dictionary from analyze_complete().
    """
    print("\n" + "=" * 80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 80)

    # Basic info
    print(f"\nFile: {results['filepath'].name}")
    print(f"Signal Type: {'Digital' if results['is_digital'] else 'Analog'}")
    print(f"Analyses Run: {len(results['results'])}")

    # Count total measurements (handle both dict and flat formats)
    total_measurements = 0
    for domain in results["results"].values():
        for v in domain.values():
            if (isinstance(v, dict) and "value" in v) or (isinstance(v, (int, float)) and not isinstance(v, bool)):
                total_measurements += 1
    print(f"Total Measurements: {total_measurements}")

    # Protocol detection
    if results.get("protocols_detected"):
        proto_count = len(results["protocols_detected"])
        print(f"\nProtocols Detected: {proto_count}")
        for proto in results["protocols_detected"]:
            conf = proto.get("confidence", 0.0)
            print(f"  - {proto['protocol'].upper()}: {conf:.1%} confidence")

    # Decoded frames
    if results.get("decoded_frames"):
        print(f"\nFrames Decoded: {len(results['decoded_frames'])}")

    # Reverse engineering
    if results.get("reverse_engineering"):
        re_res = results["reverse_engineering"]
        print("\nReverse Engineering:")
        if re_res.get("baud_rate"):
            print(f"  - Baud Rate: {re_res['baud_rate']:.0f} Hz")
        if re_res.get("frame_format"):
            print(f"  - Frame Format: {re_res['frame_format']}")
        if re_res.get("sync_patterns"):
            print(f"  - Sync Pattern: {re_res['sync_patterns']}")
        if re_res.get("crc_parameters"):
            print("  - CRC Detected: Yes")
        if re_res.get("confidence"):
            print(f"  - Confidence: {re_res['confidence']:.1%}")

    # Anomalies
    if results.get("anomalies"):
        print(f"\nAnomalies Detected: {len(results['anomalies'])}")
        # Group by severity
        severity_counts: dict[str, int] = {}
        for anomaly in results["anomalies"]:
            severity = anomaly.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        for severity, count in sorted(severity_counts.items()):
            print(f"  - {severity}: {count}")

    # Pattern recognition
    if results.get("patterns"):
        pattern_res = results["patterns"]
        print("\nPattern Recognition:")
        if pattern_res.get("signatures"):
            print(f"  - Signature Patterns: {len(pattern_res['signatures'])}")
            for sig in pattern_res["signatures"][:5]:  # Show first 5
                print(
                    f"    • {sig['pattern'][:16]}... ({sig['length']} bytes, {sig['count']} occurrences)"
                )
        if pattern_res.get("state_machine"):
            sm = pattern_res["state_machine"]
            state_count = len(sm.states) if hasattr(sm, "states") else 0
            print(f"  - State Machine States: {state_count}")

    # Visualizations
    print(f"\nPlots Generated: {len(results['plots'])}")
    if results["plots"]:
        for plot_name in sorted(results["plots"].keys()):
            print(f"  - {plot_name.replace('_', ' ').title()}")

    # Report
    if results["report_path"]:
        print(f"\n📄 Report: {results['report_path']}")

    print(f"\n📁 Output Directory: {results['output_dir']}")
    print("=" * 80 + "\n")


def main() -> int:
    """Main entry point for complete waveform analysis."""
    parser = argparse.ArgumentParser(
        description="Complete waveform analysis with reverse engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis (all features enabled)
  %(prog)s signal.wfm

  # Specify output directory
  %(prog)s capture.tss --output ./my_analysis

  # Quick analysis without reverse engineering
  %(prog)s data.csv --no-reverse-engineering

  # Deep reverse engineering with protocol hints
  %(prog)s unknown.wfm --re-depth deep --protocol-hints uart spi

  # Disable specific features
  %(prog)s signal.wfm --no-protocol-decode --no-patterns

Supported formats:
  .wfm   - Tektronix waveform files
  .tss   - Tektronix session files (multi-channel)
  .csv   - CSV waveform data
  .isf   - Tektronix internal save format
  .bin   - Binary waveform data

Capabilities:
  - Time/Frequency/Digital/Statistical Analysis
  - Protocol Detection & Decoding (UART, SPI, I2C, CAN, etc.)
  - Reverse Engineering (Clock recovery, CRC analysis, Framing)
  - Pattern Recognition & State Machine Inference
  - Comprehensive Reporting with Visualizations
        """,
    )

    # Required arguments
    parser.add_argument(
        "file",
        type=Path,
        help="Input waveform file path",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for analysis results (default: ./waveform_analysis_output)",
    )

    # Analysis selection
    parser.add_argument(
        "--analyses",
        "-a",
        nargs="+",
        choices=["time_domain", "frequency_domain", "digital", "statistics", "all"],
        default=["all"],
        help="Analysis domains to run (default: all)",
    )

    # Protocol decoding options
    parser.add_argument(
        "--no-protocol-decode",
        action="store_true",
        help="Disable automatic protocol detection and decoding",
    )

    parser.add_argument(
        "--protocol-hints",
        nargs="+",
        metavar="PROTOCOL",
        help="Protocol hints for decoder (e.g., uart spi i2c)",
    )

    # Reverse engineering options
    parser.add_argument(
        "--no-reverse-engineering",
        action="store_true",
        help="Disable reverse engineering pipeline",
    )

    parser.add_argument(
        "--re-depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Reverse engineering analysis depth (default: standard)",
    )

    # Pattern recognition options
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable pattern recognition and state machine inference",
    )

    # Output options
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable report generation",
    )

    # Verbosity
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    try:
        # Import workflow module
        from oscura.workflows import waveform

        # Convert analyses list format
        analyses: list[str] | Literal["all"]
        if "all" in args.analyses:
            analyses = "all"
        else:
            analyses = args.analyses

        # Single API call performs COMPLETE analysis including RE
        results = waveform.analyze_complete(
            args.file,
            output_dir=args.output,
            analyses=analyses,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            enable_protocol_decode=not args.no_protocol_decode,
            enable_reverse_engineering=not args.no_reverse_engineering,
            enable_pattern_recognition=not args.no_patterns,
            protocol_hints=args.protocol_hints,
            reverse_engineering_depth=args.re_depth,
            verbose=not args.quiet,
        )

        # Print detailed measurements and summary
        if not args.quiet:
            print_detailed_measurements(results)
            print_summary(results)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid configuration - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Analysis failed - {e}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
