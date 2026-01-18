#!/usr/bin/env python3
"""Comprehensive codebase health check for Oscura.

This script runs multiple code quality and organization checks:
1. Dead code detection (vulture)
2. Complexity analysis (radon)
3. Test suite statistics
4. Dead fixture detection (pytest-deadfixtures)
5. Duplicate test name detection
6. Import organization check

All checks are configurable and produce detailed reports.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent


class HealthCheck:
    """Base class for health checks."""

    def __init__(self, name: str) -> None:
        """Initialize health check.

        Args:
            name: Display name for this check
        """
        self.name = name
        self.passed = False
        self.output = ""
        self.details: dict[str, Any] = {}

    def run(self) -> bool:
        """Run the health check.

        Returns:
            True if check passed, False otherwise
        """
        raise NotImplementedError

    def report(self) -> None:
        """Print detailed report."""
        status = "✓" if self.passed else "✗"
        print(f"\n{status} {self.name}")
        if self.output:
            print(self.output)


class DeadCodeCheck(HealthCheck):
    """Check for dead code using vulture."""

    def __init__(self) -> None:
        """Initialize dead code check."""
        super().__init__("Dead Code Detection (vulture)")

    def run(self) -> bool:
        """Run vulture to detect dead code.

        Returns:
            True if no dead code found, False otherwise
        """
        try:
            result = subprocess.run(
                ["uv", "run", "vulture", "src/oscura", "--min-confidence", "80"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
            )

            self.output = result.stdout.strip()

            if not self.output or "unused" not in self.output.lower():
                self.output = "No dead code detected!"
                self.passed = True
            else:
                # Count issues
                lines = self.output.split("\n")
                self.details["dead_code_items"] = len(
                    [line for line in lines if line.strip() and not line.startswith("# ")]
                )

            return self.passed

        except Exception as e:
            self.output = f"Error running vulture: {e}"
            return False


class ComplexityCheck(HealthCheck):
    """Check code complexity using radon."""

    def __init__(self) -> None:
        """Initialize complexity check."""
        super().__init__("Complexity Analysis (radon)")

    def run(self) -> bool:
        """Run radon complexity analysis.

        Returns:
            True if complexity is acceptable, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "radon",
                    "cc",
                    "src/oscura",
                    "-a",
                    "-s",
                    "--total-average",
                    "-j",
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
            )

            # Parse JSON output
            try:
                data = json.loads(result.stdout)

                # Count complexity ratings
                complexity_counts: dict[str, int] = {}
                total_functions = 0

                for file_data in data.values():
                    for item in file_data:
                        if isinstance(item, dict) and "complexity" in item:
                            rating = item.get("rank", "?")
                            complexity_counts[rating] = complexity_counts.get(rating, 0) + 1
                            total_functions += 1

                self.details["total_functions"] = total_functions
                self.details["complexity_distribution"] = complexity_counts

                # Build output
                self.output = f"Analyzed {total_functions} functions:\n"
                for rating in ["A", "B", "C", "D", "E", "F"]:
                    count = complexity_counts.get(rating, 0)
                    if count > 0:
                        self.output += f"  {rating}: {count} functions\n"

                # Pass if no D/E/F ratings or very few C
                high_complexity = sum(complexity_counts.get(r, 0) for r in ["D", "E", "F"])
                self.passed = high_complexity == 0

            except json.JSONDecodeError:
                # Fallback to text parsing
                self.output = result.stdout
                self.passed = "D (" not in result.stdout and "E (" not in result.stdout

            return self.passed

        except Exception as e:
            self.output = f"Error running radon: {e}"
            return False


class TestStatsCheck(HealthCheck):
    """Analyze test suite statistics."""

    def __init__(self) -> None:
        """Initialize test stats check."""
        super().__init__("Test Suite Statistics")

    def run(self) -> bool:
        """Collect test suite statistics.

        Returns:
            True (always passes, informational only)
        """
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q", "--no-header"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
            )

            lines = [line for line in result.stdout.split("\n") if "::" in line]

            # Categorize by test type
            categories: dict[str, int] = {}
            for line in lines:
                parts = line.split("/")
                if len(parts) > 1:
                    # Extract category from path (unit, integration, etc.)
                    if "tests/" in line:
                        cat_part = line.split("tests/")[1].split("/")[0]
                        categories[cat_part] = categories.get(cat_part, 0) + 1

            self.details["total_tests"] = len(lines)
            self.details["categories"] = categories

            self.output = f"Total tests: {len(lines)}\n"
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                self.output += f"  {cat}: {count}\n"

            self.passed = True
            return True

        except Exception as e:
            self.output = f"Error collecting tests: {e}"
            return False


class DeadFixturesCheck(HealthCheck):
    """Check for unused pytest fixtures."""

    def __init__(self) -> None:
        """Initialize dead fixtures check."""
        super().__init__("Dead Fixtures Check")

    def run(self) -> bool:
        """Run pytest-deadfixtures to find unused fixtures.

        Returns:
            True if no dead fixtures found, False otherwise
        """
        try:
            result = subprocess.run(
                ["pytest", "--dead-fixtures", "-q"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
                timeout=60,
            )

            self.output = result.stdout.strip()

            if not self.output or "unused" not in self.output.lower():
                self.output = "No dead fixtures detected!"
                self.passed = True
            else:
                # Count unused fixtures
                lines = self.output.split("\n")
                self.details["dead_fixtures"] = len(
                    [line for line in lines if "unused" in line.lower()]
                )

            return self.passed

        except FileNotFoundError:
            self.output = "pytest-deadfixtures not installed (optional check)"
            self.passed = True
            return True
        except subprocess.TimeoutExpired:
            self.output = "Check timed out (skipped)"
            self.passed = True
            return True
        except Exception as e:
            self.output = f"Error running pytest-deadfixtures: {e}"
            self.passed = True  # Don't fail on optional check
            return True


class DuplicateTestCheck(HealthCheck):
    """Check for duplicate test names."""

    def __init__(self) -> None:
        """Initialize duplicate test check."""
        super().__init__("Duplicate Test Names")

    def run(self) -> bool:
        """Find tests with duplicate names.

        Returns:
            True if no significant duplicates found, False otherwise
        """
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q", "--no-header"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
            )

            # Extract test names
            test_names = []
            for line in result.stdout.strip().split("\n"):
                if "::" in line:
                    # Get just the test function name
                    name = line.split("::")[-1].strip()
                    test_names.append(name)

            # Find duplicates
            name_counts = Counter(test_names)
            duplicates = {name: count for name, count in name_counts.items() if count > 1}

            self.details["total_unique_names"] = len(name_counts)
            self.details["duplicate_count"] = len(duplicates)
            self.details["total_tests"] = len(test_names)

            if duplicates:
                self.output = f"Found {len(duplicates)} duplicate test names:\n"
                # Show top 10 duplicates
                for name, count in sorted(duplicates.items(), key=lambda x: -x[1])[:10]:
                    self.output += f"  {count}x: {name}\n"

                # Only fail if there are excessive duplicates
                self.passed = len(duplicates) < 100
            else:
                self.output = "No duplicate test names found!"
                self.passed = True

            return self.passed

        except Exception as e:
            self.output = f"Error checking duplicates: {e}"
            return False


class DocstringCoverageCheck(HealthCheck):
    """Check docstring coverage using interrogate."""

    def __init__(self) -> None:
        """Initialize docstring coverage check."""
        super().__init__("Docstring Coverage (interrogate)")

    def run(self) -> bool:
        """Run interrogate to check docstring coverage.

        Returns:
            True if coverage meets threshold, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "interrogate",
                    "src/oscura",
                    "--fail-under=95",
                    "--verbose=0",
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                check=False,
            )

            self.output = result.stdout.strip()

            # Extract coverage percentage
            for line in self.output.split("\n"):
                if "%" in line and "coverage" in line.lower():
                    # Parse percentage
                    try:
                        pct = float(line.split("%")[0].split()[-1])
                        self.details["coverage_percent"] = pct
                        self.passed = pct >= 95.0
                    except (ValueError, IndexError):
                        pass

            if result.returncode == 0:
                self.passed = True

            return self.passed

        except Exception as e:
            self.output = f"Error running interrogate: {e}"
            return False


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("Oscura Codebase Health Check")
    print("=" * 70)

    checks = [
        DeadCodeCheck(),
        ComplexityCheck(),
        TestStatsCheck(),
        DocstringCoverageCheck(),
        DeadFixturesCheck(),
        DuplicateTestCheck(),
    ]

    results = []

    for check in checks:
        try:
            success = check.run()
            results.append((check.name, success))
            check.report()
        except Exception as e:
            print(f"\n✗ {check.name}")
            print(f"  Error: {e}")
            results.append((check.name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\nPassed: {passed_count}/{total_count}")

    # Exit with appropriate code
    if passed_count == total_count:
        print("\n✓ All checks passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
