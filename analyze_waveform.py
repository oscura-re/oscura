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

    # Pattern recognition
    if results.get("patterns"):
        pattern_res = results["patterns"]
        print("\nPattern Recognition:")
        if pattern_res.get("anomalies"):
            print(f"  - Anomalies: {len(pattern_res['anomalies'])}")
        if pattern_res.get("patterns"):
            print(f"  - Patterns Discovered: {len(pattern_res['patterns'])}")
        if pattern_res.get("state_machine"):
            sm = pattern_res["state_machine"]
            state_count = len(sm.states) if hasattr(sm, "states") else 0
            print(f"  - State Machine States: {state_count}")

    # Visualizations
    print(f"\nPlots Generated: {len(results['plots'])}")

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

        # Print comprehensive summary
        if not args.quiet:
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
