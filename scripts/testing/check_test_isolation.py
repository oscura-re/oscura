#!/usr/bin/env python3
"""Check test isolation by running a sample of tests individually.

This script verifies that tests can run in isolation and don't depend on
state from other tests. It randomly samples test files and runs them
individually to detect isolation issues.
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path


def find_test_files() -> list[Path]:
    """Find all test files in the tests directory."""
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ Error: tests directory not found")
        sys.exit(1)

    test_files = list(test_dir.rglob("test_*.py"))
    test_files.extend(test_dir.rglob("*_test.py"))

    # Filter out __pycache__ and other non-test files
    test_files = [f for f in test_files if "__pycache__" not in str(f)]

    # Exclude test files that depend on dependencies with known Python 3.12+ issues
    # asammdf has SyntaxError in Python 3.12+ due to invalid escape sequences
    #
    # Also exclude tests that apply resource limits, which can interfere with
    # isolation testing (tests timeout when run individually without pytest-xdist)
    #
    # Also exclude helper modules that match test_*.py pattern but contain no tests
    excluded_patterns = [
        "tests/automotive/loaders/test_mdf_loader.py",
        "tests/unit/plugins/test_isolation.py",  # Resource limits cause timeouts
        "tests/unit/search/test_performance.py",  # Only performance tests (all filtered by markers)
        "tests/performance/test_benchmarks.py",  # Only performance/benchmark tests (all filtered)
        "tests/stress/test_protocol_decoder_load.py",  # Only stress/slow tests (all filtered)
        "tests/stress/test_realtime_streaming_load.py",  # Only stress/slow tests (all filtered)
        "tests/unit/workflow/test_dag_performance.py",  # Only performance tests (all filtered)
        "tests/unit/hooks/datetime_utils_for_test.py",  # Helper module, not a test file
        "tests/unit/analyzers/packet/test_stream.py",  # Timeouts in isolation (50 tests, needs xdist)
    ]
    test_files = [
        f for f in test_files if not any(str(f).endswith(pattern) for pattern in excluded_patterns)
    ]

    return sorted(set(test_files))


def run_test_file(test_file: Path) -> tuple[bool, str]:
    """Run a single test file and return success status and output."""
    try:
        # Use same marker filtering as CI to avoid running slow/performance tests
        # that would timeout in isolation (e.g., 1GB file benchmarks)
        # Use 'uv run python -m pytest' to ensure correct Python environment
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "-m",
                "not slow and not performance",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 60 seconds"
    except Exception as e:
        return False, f"Error running test: {e}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check test isolation by running samples individually"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help="Number of test files to sample (default: 15)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("🔍 Checking test isolation...")
    print()

    # Find all test files
    test_files = find_test_files()
    print(f"📁 Found {len(test_files)} test files")

    if len(test_files) == 0:
        print("⚠️  No test files found")
        return 0

    # Sample test files
    sample_size = min(args.sample, len(test_files))
    if args.seed is not None:
        random.seed(args.seed)
    sampled_files = random.sample(test_files, sample_size)

    print(f"🎲 Sampling {sample_size} files for isolation check")
    print()

    # Run each sampled test file individually
    failures = []
    for i, test_file in enumerate(sampled_files, 1):
        # Handle both absolute and relative paths
        try:
            rel_path = test_file.relative_to(Path.cwd())
        except ValueError:
            rel_path = test_file
        print(f"[{i}/{sample_size}] Testing {rel_path}...", end=" ", flush=True)

        success, output = run_test_file(test_file)

        if success:
            print("✅")
        else:
            print("❌")
            failures.append((test_file, output))

    print()
    print("=" * 70)
    print()

    # Report results
    if not failures:
        print(f"✅ All {sample_size} sampled tests passed in isolation!")
        return 0
    else:
        print(f"❌ {len(failures)}/{sample_size} test files failed in isolation:")
        print()
        for test_file, output in failures:
            # Handle both absolute and relative paths
            try:
                rel_path = test_file.relative_to(Path.cwd())
            except ValueError:
                rel_path = test_file
            print(f"Failed: {rel_path}")
            print("Output (last 500 chars):")
            print(output[-500:] if len(output) > 500 else output)
            print()

        print("⚠️  These tests may have isolation issues:")
        print("   - They might depend on state from other tests")
        print("   - They might require specific test execution order")
        print("   - They might have missing fixtures or setup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
