#!/usr/bin/env python3
"""Comprehensive Waveform Analysis - Reference Implementation.

This script demonstrates optimal usage of the Oscura framework for complete
waveform analysis, leveraging high-level workflow APIs for concise scripting.

Features:
- Automatic file format detection (.wfm, .tss, .csv, etc.)
- Complete multi-domain analysis (time, frequency, digital, statistical)
- Professional visualization with IEEE publication standards
- Comprehensive HTML reports with embedded plots
- Minimal code using framework APIs

Example:
    $ python3 analyze_waveform.py signal.wfm
    $ python3 analyze_waveform.py capture.tss --channel CH2 --output ./analysis
    $ python3 analyze_waveform.py data.csv

References:
    IEEE 181-2011: Transitional Waveform Definitions
    IEEE 1241-2010: ADC Terminology and Test Methods
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal


def main() -> int:
    """Main entry point for waveform analysis."""
    parser = argparse.ArgumentParser(
        description="Comprehensive waveform analysis using Oscura framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s signal.wfm
  %(prog)s capture.tss --channel CH2
  %(prog)s data.csv --output ./my_analysis

Supported formats:
  .wfm   - Tektronix waveform files
  .tss   - Tektronix session files (multi-channel)
  .csv   - CSV waveform data
  .isf   - Tektronix internal save format
  .bin   - Binary waveform data
        """,
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Input waveform file path",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for analysis results (default: ./waveform_analysis_output)",
    )

    parser.add_argument(
        "--channel",
        "-c",
        type=str,
        default=None,
        help="Channel name/index for multi-channel files (e.g., 'CH1', 'CH2', 0, 1)",
    )

    parser.add_argument(
        "--analyses",
        "-a",
        nargs="+",
        choices=["time_domain", "frequency_domain", "digital", "statistics", "all"],
        default=["all"],
        help="Analysis domains to run (default: all)",
    )

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
        # Run complete analysis using framework workflow API
        from oscura.workflows import waveform

        # Convert analyses list format
        analyses: list[str] | Literal["all"]
        if "all" in args.analyses:
            analyses = "all"
        else:
            analyses = args.analyses

        results = waveform.analyze_complete(
            args.file,
            output_dir=args.output,
            analyses=analyses,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            verbose=not args.quiet,
        )

        # Print summary
        if not args.quiet:
            print("\n" + "=" * 80)
            print("ANALYSIS SUMMARY")
            print("=" * 80)
            print(f"File: {results['filepath'].name}")
            print(f"Signal Type: {'Digital' if results['is_digital'] else 'Analog'}")
            print(f"Analyses Run: {len(results['results'])}")
            print(f"Plots Generated: {len(results['plots'])}")
            if results["report_path"]:
                print(f"Report: {results['report_path']}")
            print(f"Output Directory: {results['output_dir']}")
            print("=" * 80)

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
