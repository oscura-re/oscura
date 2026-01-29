#!/usr/bin/env python3
"""Compare benchmark results across runs.

This script compares two sets of benchmark results to detect performance regressions
or improvements. It's designed to run in CI to catch performance issues early.

Usage:
    python scripts/compare_benchmarks.py baseline.json current.json
    python scripts/compare_benchmarks.py baseline.json current.json --threshold 20

Exit codes:
    0: No significant regressions
    1: Significant regressions detected (>threshold%)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Regression threshold (percentage)
DEFAULT_THRESHOLD = 20.0


def load_benchmark_results(path: Path) -> dict[str, Any]:
    """Load benchmark results from JSON file.

    Args:
        path: Path to JSON file containing pytest-benchmark results.

    Returns:
        Dictionary with benchmark data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    with open(path) as f:
        return json.load(f)


def extract_benchmark_stats(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract benchmark statistics from pytest-benchmark JSON.

    Args:
        data: pytest-benchmark JSON data.

    Returns:
        Dictionary mapping test name to stats (mean, min, max, stddev).
    """
    benchmarks = {}
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        stats = bench["stats"]
        benchmarks[name] = {
            "mean": stats["mean"],
            "min": stats["min"],
            "max": stats["max"],
            "stddev": stats["stddev"],
            "rounds": stats.get("rounds", 0),
        }
    return benchmarks


def compare_benchmarks(
    baseline: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Compare two sets of benchmarks.

    Args:
        baseline: Baseline benchmark stats.
        current: Current benchmark stats.
        threshold: Regression threshold (percentage).

    Returns:
        Tuple of (regressions, improvements) where each is a list of (name, change_pct).
    """
    regressions = []
    improvements = []

    for name in sorted(baseline.keys()):
        if name not in current:
            print(f"WARNING: {name} missing in current results")
            continue

        baseline_mean = baseline[name]["mean"]
        current_mean = current[name]["mean"]

        if baseline_mean == 0:
            continue

        change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

        # Classify based on threshold
        if change_pct > threshold:
            regressions.append((name, change_pct))
        elif change_pct < -5:  # Only count significant improvements
            improvements.append((name, abs(change_pct)))

    return regressions, improvements


def print_comparison_table(
    baseline: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
) -> None:
    """Print detailed comparison table.

    Args:
        baseline: Baseline benchmark stats.
        current: Current benchmark stats.
    """
    print("\nBenchmark Comparison")
    print("=" * 100)
    print(f"{'Test':<50} {'Baseline':<15} {'Current':<15} {'Change':<10} {'Status'}")
    print("-" * 100)

    for name in sorted(baseline.keys()):
        if name not in current:
            print(f"{name:<50} {'MISSING IN CURRENT':<40}")
            continue

        baseline_mean = baseline[name]["mean"]
        current_mean = current[name]["mean"]
        change_pct = (
            ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        )

        # Format times
        baseline_str = f"{baseline_mean * 1000:.4f}ms"
        current_str = f"{current_mean * 1000:.4f}ms"
        change_str = f"{change_pct:+.1f}%"

        # Determine status
        if abs(change_pct) < 5:
            status = "~"  # No significant change
        elif change_pct > 0:
            status = "↓"  # Regression (slower)
        else:
            status = "↑"  # Improvement (faster)

        print(f"{name:<50} {baseline_str:<15} {current_str:<15} {change_str:<10} {status}")


def print_summary(
    regressions: list[tuple[str, float]],
    improvements: list[tuple[str, float]],
    threshold: float,
) -> None:
    """Print summary of findings.

    Args:
        regressions: List of (name, change_pct) for regressions.
        improvements: List of (name, change_pct) for improvements.
        threshold: Regression threshold used.
    """
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    if regressions:
        print(f"\n⚠️  {len(regressions)} Performance Regressions (>{threshold}%):")
        for name, pct in sorted(regressions, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {name}: +{pct:.1f}%")

    if improvements:
        print(f"\n✓ {len(improvements)} Performance Improvements (>5%):")
        for name, pct in sorted(improvements, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {name}: -{pct:.1f}%")

    if not regressions and not improvements:
        print("\n✓ No significant performance changes")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare benchmark results and detect regressions")
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON file")
    parser.add_argument("current", type=Path, help="Current benchmark JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Regression threshold percentage (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)

    if not args.current.exists():
        print(f"Error: Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    # Load results
    try:
        baseline_data = load_benchmark_results(args.baseline)
        current_data = load_benchmark_results(args.current)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract stats
    baseline_stats = extract_benchmark_stats(baseline_data)
    current_stats = extract_benchmark_stats(current_data)

    if not baseline_stats:
        print("Warning: No benchmarks found in baseline file", file=sys.stderr)

    if not current_stats:
        print("Warning: No benchmarks found in current file", file=sys.stderr)

    # Compare
    regressions, improvements = compare_benchmarks(baseline_stats, current_stats, args.threshold)

    # Output
    if args.json:
        # JSON output for programmatic consumption
        output = {
            "threshold": args.threshold,
            "regressions": [{"name": name, "change_pct": pct} for name, pct in regressions],
            "improvements": [{"name": name, "change_pct": pct} for name, pct in improvements],
            "has_regressions": len(regressions) > 0,
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print_comparison_table(baseline_stats, current_stats)
        print_summary(regressions, improvements, args.threshold)

    # Exit code
    if regressions:
        print(f"\n❌ {len(regressions)} significant regressions detected", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✓ No significant regressions")
        sys.exit(0)


if __name__ == "__main__":
    main()
