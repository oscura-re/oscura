#!/usr/bin/env python3
"""Comprehensive demonstration validator.

Discovers and validates all demonstrations in the demonstrations/ directory.
Ensures 100% pass rate and proper functionality.

Usage:
    python demonstrations/validate_all.py
    python demonstrations/validate_all.py --section 00_getting_started
    python demonstrations/validate_all.py --verbose
    python demonstrations/validate_all.py --fast  # Skip slow demos

Exit codes:
    0: All demonstrations passed
    1: One or more demonstrations failed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


class DemoResult(NamedTuple):
    """Result of running a single demonstration."""

    path: str
    passed: bool
    duration: float
    output: str
    error: str | None


class DemoValidator:
    """Validates all Oscura demonstrations."""

    def __init__(self, verbose: bool = False, fast: bool = False):
        """Initialize validator.

        Args:
            verbose: Print full output from demos
            fast: Skip slow demonstrations
        """
        self.verbose = verbose
        self.fast = fast
        self.demos_dir = Path(__file__).parent
        self.results: list[DemoResult] = []

    def discover_demos(self, section: str | None = None) -> list[Path]:
        """Discover all demonstration files.

        Args:
            section: Optional section filter (e.g., "00_getting_started")

        Returns:
            List of demonstration file paths
        """
        if section:
            search_pattern = f"{section}/**/*.py"
        else:
            search_pattern = "*/**/*.py"

        # Find all Python files in demonstration directories
        demo_files = sorted(self.demos_dir.glob(search_pattern))

        # Exclude common/, utility scripts, and __init__.py files
        demo_files = [
            f
            for f in demo_files
            if f.parent.name != "common"
            and f.name != "validate_all.py"
            and f.name != "capability_index.py"
            and f.name != "generate_all_data.py"
            and f.name != "__init__.py"
        ]

        return demo_files

    def run_demo(self, demo_path: Path) -> DemoResult:
        """Run a single demonstration and validate.

        Args:
            demo_path: Path to demonstration file

        Returns:
            DemoResult with pass/fail status
        """
        relative_path = demo_path.relative_to(self.demos_dir)
        print(f"\n{'=' * 80}")
        print(f"Running: {relative_path}")
        print("=" * 80)

        start_time = time.time()

        try:
            # Run demonstration with uv environment
            result = subprocess.run(
                ["uv", "run", "python3", str(demo_path)],
                capture_output=True,
                text=True,
                cwd=self.demos_dir.parent,  # Run from project root
                timeout=300,  # 5 minute timeout
                check=False,
            )

            duration = time.time() - start_time
            output = result.stdout + result.stderr

            if self.verbose:
                print(output)

            # Check for validation success
            passed = "DEMONSTRATION PASSED" in output and result.returncode == 0

            if passed:
                print(f"✓ PASS ({duration:.2f}s)")
                error = None
            else:
                print(f"✗ FAIL ({duration:.2f}s)")
                if result.returncode != 0:
                    error = f"Exit code {result.returncode}"
                else:
                    error = "No validation marker found"

                if not self.verbose:
                    print("\nOutput:")
                    print(output[-1000:])  # Last 1000 characters

            return DemoResult(
                path=str(relative_path),
                passed=passed,
                duration=duration,
                output=output,
                error=error,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"✗ TIMEOUT ({duration:.2f}s)")
            return DemoResult(
                path=str(relative_path),
                passed=False,
                duration=duration,
                output="",
                error="Timeout (>5 minutes)",
            )

        except Exception as e:
            duration = time.time() - start_time
            print(f"✗ ERROR: {e}")
            return DemoResult(
                path=str(relative_path),
                passed=False,
                duration=duration,
                output="",
                error=str(e),
            )

    def validate_all(self, section: str | None = None) -> bool:
        """Validate all demonstrations.

        Args:
            section: Optional section filter

        Returns:
            True if all passed, False otherwise
        """
        demos = self.discover_demos(section)

        if not demos:
            print("No demonstrations found!")
            return False

        print(f"\n{'=' * 80}")
        print("DEMONSTRATION VALIDATOR")
        print("=" * 80)
        print(f"Found {len(demos)} demonstrations")
        if section:
            print(f"Section filter: {section}")
        if self.fast:
            print("Fast mode: Skipping slow demonstrations")
        print()

        # Run all demonstrations
        for demo_path in demos:
            result = self.run_demo(demo_path)
            self.results.append(result)

        # Print summary
        self._print_summary()

        # Return overall pass/fail
        return all(r.passed for r in self.results)

    def _print_summary(self) -> None:
        """Print validation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_duration = sum(r.duration for r in self.results)

        print(f"\n{'=' * 80}")
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total demonstrations: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {passed / total * 100:.1f}%")
        print(f"Total duration: {total_duration:.2f}s")

        if failed > 0:
            print("\nFailed demonstrations:")
            for result in self.results:
                if not result.passed:
                    print(f"  ✗ {result.path}")
                    if result.error:
                        print(f"    Error: {result.error}")

        print(f"\n{'=' * 80}")
        if failed == 0:
            print("✓ ALL DEMONSTRATIONS PASSED")
        else:
            print(f"✗ {failed} DEMONSTRATION(S) FAILED")
        print("=" * 80)
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Oscura demonstrations")
    parser.add_argument(
        "--section", help="Only validate specific section (e.g., 00_getting_started)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full output")
    parser.add_argument("--fast", action="store_true", help="Skip slow demonstrations")

    args = parser.parse_args()

    validator = DemoValidator(verbose=args.verbose, fast=args.fast)
    success = validator.validate_all(section=args.section)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
