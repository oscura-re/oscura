#!/usr/bin/env python3
"""Measure test coverage for Oscura modules.

This script runs pytest with coverage for each module separately,
collects results, and calculates overall coverage.

Usage:
    # Full analysis (all modules, 90s timeout)
    uv run python scripts/measure_coverage.py

    # Quick analysis (key modules only, 60s timeout)
    uv run python scripts/measure_coverage.py --quick

    # Custom module list
    uv run python scripts/measure_coverage.py --modules core,api,loaders

    # Robust mode with retries
    uv run python scripts/measure_coverage.py --robust
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "oscura"
TESTS_DIR = PROJECT_ROOT / "tests" / "unit"

# Key modules for quick analysis (representative sample)
KEY_MODULES = [
    "core",
    "api",
    "analyzers",
    "inference",
    "loaders",
    "reporting",
    "visualization",
    "filtering",
    "exporters",
    "batch",
    "cli",
    "dsl",
    "plugins",
    "utils",
    "config",
    "pipeline",
    "compliance",
    "streaming",
]


@dataclass
class ModuleCoverage:
    """Coverage results for a single module."""

    name: str
    statements: int
    covered: int
    missing: int
    coverage_pct: float
    has_tests: bool
    error: str | None = None


def get_source_modules() -> list[str]:
    """Get all source module directories."""
    modules = []
    for path in SRC_DIR.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            modules.append(path.name)
    return sorted(modules)


def get_test_directories() -> set[str]:
    """Get all test module directories."""
    tests = set()
    for path in TESTS_DIR.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            tests.add(path.name)
    return tests


def measure_module_coverage(
    module: str, test_dir_exists: bool, timeout: int, max_retries: int = 1
) -> ModuleCoverage:
    """Measure coverage for a single module."""
    src_path = f"src/oscura/{module}"
    test_path = f"tests/unit/{module}"

    if not test_dir_exists:
        # Count statements in source without running tests
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    f"""
import ast
import sys
from pathlib import Path

total = 0
src = Path('{src_path}')
for py_file in src.rglob('*.py'):
    try:
        with open(py_file) as f:
            tree = ast.parse(f.read())
        # Count executable statements (simplified)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                                ast.Expr, ast.Return, ast.Raise, ast.Assert,
                                ast.Pass, ast.Break, ast.Continue,
                                ast.If, ast.For, ast.While, ast.Try,
                                ast.With, ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                total += 1
    except (SyntaxError, OSError, UnicodeDecodeError):
        pass
print(total)
""",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=PROJECT_ROOT,
            )
            statements = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            statements = 100  # Default estimate

        return ModuleCoverage(
            name=module,
            statements=statements,
            covered=0,
            missing=statements,
            coverage_pct=0.0,
            has_tests=False,
            error="No test directory",
        )

    # Run pytest with coverage (with retries if robust mode)
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "pytest",
                    test_path,
                    f"--cov={src_path}",
                    "--cov-report=term",
                    f"--timeout={timeout}",
                    "-q",
                    "--tb=no",
                    "-x",  # Stop on first failure for speed
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout + 30,
                cwd=PROJECT_ROOT,
            )

            output = result.stdout + result.stderr

            # Parse TOTAL line: "TOTAL    1234    123    90%"
            total_match = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", output)
            if total_match:
                statements = int(total_match.group(1))
                missing = int(total_match.group(2))
                coverage_pct = int(total_match.group(3))
                covered = statements - missing

                return ModuleCoverage(
                    name=module,
                    statements=statements,
                    covered=covered,
                    missing=missing,
                    coverage_pct=float(coverage_pct),
                    has_tests=True,
                )

            # Try alternate format: "TOTAL    1234    123    456    90%"
            total_match2 = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+\d+\s+(\d+)%", output)
            if total_match2:
                statements = int(total_match2.group(1))
                missing = int(total_match2.group(2))
                coverage_pct = int(total_match2.group(3))
                covered = statements - missing

                return ModuleCoverage(
                    name=module,
                    statements=statements,
                    covered=covered,
                    missing=missing,
                    coverage_pct=float(coverage_pct),
                    has_tests=True,
                )

            # Check if tests failed
            if "FAILED" in output or "ERROR" in output:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Brief pause before retry
                    continue
                return ModuleCoverage(
                    name=module,
                    statements=0,
                    covered=0,
                    missing=0,
                    coverage_pct=0.0,
                    has_tests=True,
                    error="Tests failed",
                )

            # No coverage data found
            if attempt < max_retries - 1:
                continue
            return ModuleCoverage(
                name=module,
                statements=0,
                covered=0,
                missing=0,
                coverage_pct=0.0,
                has_tests=True,
                error="Could not parse coverage from output",
            )

        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                continue
            return ModuleCoverage(
                name=module,
                statements=0,
                covered=0,
                missing=0,
                coverage_pct=0.0,
                has_tests=True,
                error="Timeout",
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            if attempt < max_retries - 1:
                continue
            return ModuleCoverage(
                name=module,
                statements=0,
                covered=0,
                missing=0,
                coverage_pct=0.0,
                has_tests=True,
                error=str(e),
            )

    # Should never reach here
    return ModuleCoverage(
        name=module,
        statements=0,
        covered=0,
        missing=0,
        coverage_pct=0.0,
        has_tests=True,
        error="Max retries exceeded",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Measure test coverage for Oscura modules")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: measure key modules only with 60s timeout",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Robust mode: retry failed measurements",
    )
    parser.add_argument(
        "--modules",
        type=str,
        help="Comma-separated list of modules to measure (overrides --quick)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout per module in seconds (default: 90 for full, 60 for quick)",
    )
    args = parser.parse_args()

    # Determine mode and settings
    if args.modules:
        modules = [m.strip() for m in args.modules.split(",")]
        mode_name = "Custom"
    elif args.quick:
        modules = KEY_MODULES
        mode_name = "Quick"
    else:
        modules = get_source_modules()
        mode_name = "Full"

    # Determine timeout
    if args.timeout:
        timeout = args.timeout
    elif args.quick:
        timeout = 60
    else:
        timeout = 90

    # Determine max retries
    max_retries = 3 if args.robust else 1

    # Print header
    print("=" * 70)
    print(f"Oscura Module Coverage Analysis ({mode_name} Mode)")
    if args.robust:
        print("Robust mode: enabled (3 retries)")
    print("=" * 70)
    print()

    # Filter modules that exist
    all_source_modules = get_source_modules()
    modules = [m for m in modules if m in all_source_modules]
    test_dirs = get_test_directories()

    print(f"Measuring {len(modules)} modules")
    print(f"Timeout: {timeout}s per module")
    print()

    results: list[ModuleCoverage] = []

    for i, module in enumerate(modules, 1):
        has_tests = module in test_dirs
        print(f"[{i}/{len(modules)}] Measuring {module}... ", end="", flush=True)

        result = measure_module_coverage(module, has_tests, timeout, max_retries)
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error}")
        elif result.has_tests:
            print(f"{result.coverage_pct:.0f}% ({result.covered}/{result.statements})")
        else:
            print(f"NO TESTS ({result.statements} stmts)")

    print()
    print("=" * 70)
    print("RESULTS BY MODULE")
    print("=" * 70)
    print()
    print(
        f"{'Module':<20} {'Stmts':>8} {'Covered':>8} {'Missing':>8} {'Coverage':>10} {'Status':<15}"
    )
    print("-" * 70)

    total_statements = 0
    total_covered = 0

    # Sort by coverage % ascending (lowest first)
    for r in sorted(results, key=lambda x: x.coverage_pct):
        status = "OK" if not r.error else r.error[:12]
        print(
            f"{r.name:<20} {r.statements:>8} {r.covered:>8} {r.missing:>8} {r.coverage_pct:>9.1f}% {status:<15}"
        )
        total_statements += r.statements
        total_covered += r.covered

    print("-" * 70)

    overall_pct = (total_covered / total_statements * 100) if total_statements > 0 else 0
    print(
        f"{'TOTAL':<20} {total_statements:>8} {total_covered:>8} {total_statements - total_covered:>8} {overall_pct:>9.1f}%"
    )
    print()

    print("=" * 70)
    print("MODULES BELOW 80% (Priority for improvement)")
    print("=" * 70)

    below_80 = [r for r in results if r.coverage_pct < 80 and r.statements > 0]
    below_80.sort(key=lambda x: x.statements * (80 - x.coverage_pct), reverse=True)

    if below_80:
        print(f"{'Module':<20} {'Coverage':>10} {'Gap':>10} {'Impact':>10}")
        print("-" * 50)

        for r in below_80:
            gap = 80 - r.coverage_pct
            impact = r.statements * gap / 100
            print(f"{r.name:<20} {r.coverage_pct:>9.1f}% {gap:>9.1f}% {impact:>9.0f}")
    else:
        print("All modules meet 80% target!")

    print()
    if mode_name == "Full":
        print(f"TRUE OVERALL COVERAGE: {overall_pct:.1f}%")
    else:
        print(f"MEASURED COVERAGE ({mode_name}): {overall_pct:.1f}%")
        if mode_name == "Quick":
            print("(Note: Quick mode measures key modules only, not entire codebase)")
    print("Target: 80%")
    print(f"Gap: {80 - overall_pct:.1f}%" if overall_pct < 80 else "TARGET MET!")

    # Save results to JSON
    output_file = (
        PROJECT_ROOT / ".claude" / "quick-coverage-results.json"
        if args.quick
        else PROJECT_ROOT / ".claude" / "coverage-results.json"
    )
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "mode": mode_name.lower(),
                "total_statements": total_statements,
                "total_covered": total_covered,
                "overall_coverage_pct": round(overall_pct, 1),
                "modules": [
                    {
                        "name": r.name,
                        "statements": r.statements,
                        "covered": r.covered,
                        "missing": r.missing,
                        "coverage_pct": r.coverage_pct,
                        "has_tests": r.has_tests,
                        "error": r.error,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    return 0 if overall_pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
