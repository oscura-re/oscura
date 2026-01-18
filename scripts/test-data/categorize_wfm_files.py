#!/usr/bin/env python3
"""Categorize WFM files by format, size, and loading status.

This script analyzes all .wfm files in a directory and categorizes them
for test dataset organization.

Usage:
    uv run python scripts/categorize_wfm_files.py [--output report.json] [path]

Note: This script must be run via `uv run` to ensure oscura is in the path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class FileInfo:
    """Information about a WFM file."""

    path: str
    filename: str
    size_bytes: int
    size_category: str  # small, medium, large
    format_signature: str
    format_version: str
    can_load: bool
    load_error: str | None
    sample_count: int | None
    sample_rate: float | None
    channel_name: str | None


def get_size_category(size_bytes: int) -> str:
    """Categorize file by size."""
    if size_bytes < 100 * 1024:  # < 100KB
        return "small"
    elif size_bytes < 5 * 1024 * 1024:  # < 5MB
        return "medium"
    else:
        return "large"


def analyze_file_header(path: Path) -> tuple[str, str]:
    """Analyze WFM file header to determine format.

    Returns:
        Tuple of (format_signature, format_version)
    """
    try:
        with open(path, "rb") as f:
            header = f.read(64)

        if len(header) < 16:
            return ("too_small", "unknown")

        # Check for WFM#003 signature at offset 2
        if len(header) >= 10 and header[2:10] == b":WFM#003":
            return (":WFM#003", "3.0")

        # Check for other common signatures
        if header[:4] == b"\x00\x00:W":
            return ("legacy_wfm", "unknown")

        # Check if it's a binary file with no clear signature
        try:
            # Check first bytes for printable ASCII
            first_bytes = header[:16].decode("ascii", errors="strict")
            return ("text_or_unknown", first_bytes[:16])
        except UnicodeDecodeError:
            # Binary file with no known signature
            hex_sig = header[:8].hex()
            return ("binary_unknown", hex_sig)

    except Exception as e:
        return ("read_error", str(e)[:50])


def try_load_file(path: Path) -> tuple[bool, str | None, dict[str, Any]]:
    """Try to load file with oscura.

    Returns:
        Tuple of (success, error_message, metadata_dict)
    """
    try:
        # Import oscura - this will work when run via `uv run`
        import oscura as osc

        trace = osc.load(str(path))

        metadata: dict[str, Any] = {
            "sample_count": len(trace.data) if hasattr(trace, "data") else None,
            "sample_rate": trace.metadata.sample_rate if hasattr(trace, "metadata") else None,
            "channel_name": (
                trace.metadata.channel_name
                if hasattr(trace, "metadata") and hasattr(trace.metadata, "channel_name")
                else None
            ),
        }

        return (True, None, metadata)

    except ImportError as e:
        return (False, f"Import error: {e} (run via 'uv run')", {})
    except Exception as e:
        return (False, str(e)[:200], {})


def categorize_files(base_path: Path, *, verbose: bool = False) -> dict[str, list[FileInfo]]:
    """Categorize all WFM files under base_path.

    Returns:
        Dictionary with categories as keys and lists of FileInfo as values.
    """
    categories: dict[str, list[FileInfo]] = defaultdict(list)

    # Find all .wfm files
    wfm_files = list(base_path.rglob("*.wfm"))

    if verbose:
        print(f"Found {len(wfm_files)} WFM files")

    for i, wfm_path in enumerate(wfm_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(wfm_files)}...")

        try:
            stat = wfm_path.stat()
            size_bytes = stat.st_size
            size_category = get_size_category(size_bytes)

            format_sig, format_ver = analyze_file_header(wfm_path)

            can_load, load_error, metadata = try_load_file(wfm_path)

            info = FileInfo(
                path=str(wfm_path),
                filename=wfm_path.name,
                size_bytes=size_bytes,
                size_category=size_category,
                format_signature=format_sig,
                format_version=format_ver,
                can_load=can_load,
                load_error=load_error,
                sample_count=metadata.get("sample_count"),
                sample_rate=metadata.get("sample_rate"),
                channel_name=metadata.get("channel_name"),
            )

            # Categorize
            if can_load:
                categories[f"working_{size_category}"].append(info)
            else:
                if "No waveform data found" in (load_error or ""):
                    categories["no_waveform_data"].append(info)
                elif "corrupted" in (load_error or "").lower():
                    categories["corrupted"].append(info)
                else:
                    categories["other_errors"].append(info)

        except Exception as e:
            if verbose:
                print(f"  Error processing {wfm_path}: {e}")

    return categories


def print_summary(categories: dict[str, list[FileInfo]]) -> None:
    """Print a summary of categorized files."""
    print("\n" + "=" * 60)
    print("WFM FILE CATEGORIZATION SUMMARY")
    print("=" * 60)

    total = sum(len(files) for files in categories.values())
    print(f"\nTotal files analyzed: {total}\n")

    # Working files
    working_count = sum(
        len(files) for cat, files in categories.items() if cat.startswith("working_")
    )
    print(f"Working files: {working_count} ({100 * working_count / total:.1f}%)")
    for cat in ["working_small", "working_medium", "working_large"]:
        if cat in categories:
            print(f"  - {cat}: {len(categories[cat])}")

    # Problem files
    problem_count = total - working_count
    print(f"\nProblem files: {problem_count} ({100 * problem_count / total:.1f}%)")
    for cat in ["no_waveform_data", "corrupted", "other_errors"]:
        if cat in categories:
            print(f"  - {cat}: {len(categories[cat])}")

    # Format breakdown
    print("\n" + "-" * 40)
    print("FORMAT BREAKDOWN:")
    format_counts: dict[str, int] = defaultdict(int)
    for files in categories.values():
        for f in files:
            format_counts[f.format_signature] += 1

    for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        print(f"  {fmt}: {count}")


def save_report(categories: dict[str, list[FileInfo]], output_path: Path) -> None:
    """Save categorization report to JSON."""
    report = {
        "summary": {
            "total_files": sum(len(files) for files in categories.values()),
            "working_files": sum(
                len(files) for cat, files in categories.items() if cat.startswith("working_")
            ),
            "problem_files": sum(
                len(files) for cat, files in categories.items() if not cat.startswith("working_")
            ),
        },
        "categories": {cat: [asdict(f) for f in files] for cat, files in categories.items()},
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Categorize WFM files")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to search for WFM files (default: current directory)",
    )
    parser.add_argument("--output", "-o", type=str, help="Output JSON report path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return 1

    print(f"Analyzing WFM files in: {base_path.absolute()}")

    categories = categorize_files(base_path, verbose=args.verbose)

    print_summary(categories)

    if args.output:
        save_report(categories, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
