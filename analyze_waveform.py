#!/usr/bin/env python3
"""IDEAL Comprehensive Waveform Analysis - Reference Implementation for v0.9.0.

Demonstrates the COMPLETE oscura framework with ALL capabilities:
- Unified measurement APIs (28 measurements across 4 domains)
- Professional IEEE-compliant reporting with citations
- Protocol decoding and reverse engineering
- Pattern recognition and anomaly detection
- Batch visualization with publication-quality plots
- Complete workflow orchestration

This is the reference implementation showing optimal oscura API usage.

Usage:
    analyze_waveform.py <file>                    # Full analysis
    analyze_waveform.py <file> --no-protocol      # Skip protocol decode
    analyze_waveform.py <file> --no-re            # Skip reverse engineering
    analyze_waveform.py <file> --quick            # Quick analysis only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# COMPLETE oscura framework - all you need
from oscura.workflows.waveform import analyze_complete


def print_summary(results: dict[str, Any]) -> None:
    """Print human-readable analysis summary with defensive key access."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Signal information - extract from trace object
    filepath = results.get("filepath", "N/A")
    trace = results.get("trace")

    # Default values if trace is unavailable
    signal_type = "Unknown"
    sample_count = 0
    sample_rate = 0.0
    duration = 0.0

    # Extract info from trace if available
    if trace is not None:
        signal_type = getattr(trace, "signal_type", "Unknown")
        sample_count = len(trace)
        sample_rate = trace.metadata.sample_rate if hasattr(trace, "metadata") else 0.0
        duration = trace.duration if hasattr(trace, "duration") else 0.0

    print(f"\n📊 Signal: {filepath}")
    print(f"   Type: {signal_type.capitalize()}")
    print(f"   Samples: {sample_count:,}")
    print(f"   Sample Rate: {sample_rate:.2e} Hz")
    print(f"   Duration: {duration:.6f} s")

    # Measurements (28 total via unified APIs - corrected from 37)
    if "measurements" in results:
        meas = results["measurements"]

        # Time-domain
        if meas.get("time_domain"):
            print("\n⏱️  Time-Domain Analysis:")
            td = meas["time_domain"]
            print(f"   {len(td)} measurements computed")
            for name, value_dict in td.items():
                val = value_dict.get("value", 0)
                unit = value_dict.get("unit", "")
                print(f"   {name:20s}: {val:12.6g} {unit}")

        # Frequency-domain
        if meas.get("frequency_domain"):
            print("\n📡 Frequency-Domain Analysis:")
            fd = meas["frequency_domain"]
            numeric_count = sum(
                1
                for k, v in fd.items()
                if k not in ["fft_freqs", "fft_data"] and isinstance(v, (int, float, dict))
            )
            print(f"   {numeric_count} measurements computed")
            for name, value_dict in fd.items():
                if name not in ["fft_freqs", "fft_data"]:
                    val = value_dict.get("value", 0)
                    unit = value_dict.get("unit", "")
                    print(f"   {name:20s}: {val:12.6g} {unit}")

        # Statistical
        if meas.get("statistics"):
            print("\n📈 Statistical Analysis:")
            stats = meas["statistics"]
            print(f"   {len(stats)} measurements computed")
            key_stats = ["mean", "std", "min", "max", "p50"]  # Show subset
            for name in key_stats:
                if name in stats:
                    value_dict = stats[name]
                    val = value_dict.get("value", 0)
                    unit = value_dict.get("unit", "")
                    print(f"   {name:20s}: {val:12.6g} {unit}")
            if len(stats) > len(key_stats):
                print(f"   ... and {len(stats) - len(key_stats)} more")

        # Digital (if applicable)
        if meas.get("digital"):
            print("\n🔲 Digital Analysis:")
            dig = meas["digital"]
            numeric_count = sum(1 for v in dig.values() if isinstance(v, (int, float)))
            print(f"   {numeric_count} measurements computed")
            for name, value in dig.items():
                if isinstance(value, (int, float)):
                    print(f"   {name:20s}: {value}")

    # Protocol decoding
    if results.get("protocols_detected"):
        print("\n🔍 Protocol Decoding:")
        print(f"   Detected: {', '.join(results['protocols_detected'])}")
        print(f"   Frames decoded: {results.get('frame_count', 0)}")
        print(f"   Confidence: {results.get('protocol_confidence', 0):.1%}")

    # Reverse engineering
    if results.get("reverse_engineering"):
        re_data = results["reverse_engineering"]
        print("\n🔬 Reverse Engineering:")
        if "clock_recovery" in re_data:
            print(f"   Clock rate: {re_data['clock_recovery'].get('estimated_rate', 'N/A')} Hz")
        if "sync_patterns" in re_data:
            print(f"   Sync patterns: {len(re_data['sync_patterns'])}")
        if "crc_analysis" in re_data:
            print(f"   CRC type: {re_data['crc_analysis'].get('likely_polynomial', 'unknown')}")

    # Pattern recognition
    if results.get("pattern_recognition"):
        patterns = results["pattern_recognition"]
        print("\n🧩 Pattern Recognition:")
        if "anomalies" in patterns:
            anomaly_count = len(patterns["anomalies"])
            print(f"   Anomalies detected: {anomaly_count}")
        if "repeating_patterns" in patterns:
            pattern_count = len(patterns["repeating_patterns"])
            print(f"   Repeating patterns: {pattern_count}")

    # Visualizations
    if results.get("plots"):
        print(f"\n📊 Visualizations ({len(results['plots'])} plots):")
        for plot_name in sorted(results["plots"].keys()):
            print(f"   ✓ {plot_name.replace('_', ' ').title()}")

    # Report
    if results.get("report_path"):
        print("\n📄 Professional Report:")
        print(f"   {results['report_path']}")
        print("   Includes: IEEE citations, interpretations, executive summary")

    # Output directory
    print(f"\n📁 All outputs: {results.get('output_dir', 'N/A')}")
    print("=" * 80 + "\n")


def main() -> int:
    """Execute complete waveform analysis using oscura framework."""
    parser = argparse.ArgumentParser(
        description="IDEAL Waveform Analysis - Demonstrates complete oscura v0.9.0 framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s capture.wfm                    # Full analysis (ALL features)
  %(prog)s session.tss --quick            # Quick analysis only
  %(prog)s signal.wfm --no-protocol       # Skip protocol decoding
  %(prog)s data.wfm --no-re --no-patterns # Skip RE and pattern recognition

Features demonstrated:
  • 28 measurements via unified APIs (time/freq/digital/stats)
  • IEEE-compliant reporting with citations and interpretations
  • Protocol auto-decoding (UART/SPI/I2C/CAN/etc)
  • Reverse engineering (clock recovery, sync patterns, CRC)
  • Pattern recognition (anomalies, state machines)
  • Publication-quality visualizations
        """,
    )
    parser.add_argument("filepath", type=Path, help="Waveform file to analyze")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: ./waveform_analysis_output)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis only (measurements + basic plots)",
    )
    parser.add_argument(
        "--no-protocol",
        dest="enable_protocol_decode",
        action="store_false",
        help="Disable protocol auto-decoding",
    )
    parser.add_argument(
        "--protocol-hints",
        help="Protocol hints (comma-separated: UART,SPI,I2C)",
    )
    parser.add_argument(
        "--no-re",
        dest="enable_reverse_engineering",
        action="store_false",
        help="Disable reverse engineering",
    )
    parser.add_argument(
        "--re-depth",
        choices=["quick", "standard", "deep"],
        default="standard",
        help="Reverse engineering depth",
    )
    parser.add_argument(
        "--no-patterns",
        dest="enable_pattern_recognition",
        action="store_false",
        help="Disable pattern recognition",
    )

    args = parser.parse_args()

    # Validate input
    if not args.filepath.exists():
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        return 1

    # Setup output
    output_dir = args.output or Path("./waveform_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Parse protocol hints
    protocol_hints = None
    if args.protocol_hints:
        protocol_hints = [p.strip().upper() for p in args.protocol_hints.split(",")]

    # Quick mode disables advanced features
    if args.quick:
        args.enable_protocol_decode = False
        args.enable_reverse_engineering = False
        args.enable_pattern_recognition = False

    print(f"\n{'=' * 80}")
    print("OSCURA v0.9.0 - COMPLETE WAVEFORM ANALYSIS")
    print(f"{'=' * 80}")
    print("Configuration:")
    print("  • Measurements: 28 across 4 domains (time/freq/digital/stats)")
    print(f"  • Protocol decode: {'✓' if args.enable_protocol_decode else '✗'}")
    print(
        f"  • Reverse engineering: {'✓ (' + args.re_depth + ')' if args.enable_reverse_engineering else '✗'}"
    )
    print(f"  • Pattern recognition: {'✓' if args.enable_pattern_recognition else '✗'}")
    print("  • Professional reporting: ✓ (IEEE citations + interpretations)")
    print("  • Visualizations: ✓ (publication quality)")

    # Execute COMPLETE analysis via unified workflow
    # This single call handles ALL analysis, visualization, and reporting!
    try:
        results = analyze_complete(
            args.filepath,
            output_dir=output_dir,
            enable_protocol_decode=args.enable_protocol_decode,
            protocol_hints=protocol_hints,
            enable_reverse_engineering=args.enable_reverse_engineering,
            reverse_engineering_depth=args.re_depth,
            enable_pattern_recognition=args.enable_pattern_recognition,
        )
    except Exception as e:
        print(f"\nAnalysis failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # Display human-readable summary
    print_summary(results)

    print("✅ Analysis complete!")
    print(
        "This demonstrates oscura v0.9.0 COMPLETE framework:\n"
        "  • Unified measurement APIs (28 measurements)\n"
        "  • Professional IEEE-compliant reporting\n"
        "  • Protocol decoding + reverse engineering\n"
        "  • Pattern recognition + anomaly detection\n"
        "  • Publication-quality visualizations\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
