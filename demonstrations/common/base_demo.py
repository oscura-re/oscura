"""Base demonstration class providing standard template and validation."""

from __future__ import annotations

import sys
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDemo(ABC):
    """Base class for all Oscura demonstrations.

    Provides:
    - Standard execution framework
    - Timing measurements
    - Validation infrastructure
    - Formatted output
    - Error handling

    Example:
        class MyDemo(BaseDemo):
            def __init__(self):
                super().__init__(
                    name="my_demo",
                    description="Demonstrates feature X",
                    capabilities=["feature.x", "feature.y"],
                )

            def generate_test_data(self) -> dict:
                return {"signal": ...}

            def run_demonstration(self, data: dict) -> dict:
                results = {}
                self.section("Part 1")
                # ... demonstration code ...
                return results

            def validate(self, results: dict) -> bool:
                return True

        if __name__ == "__main__":
            demo = MyDemo()
            success = demo.execute()
            exit(0 if success else 1)
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        ieee_standards: list[str] | None = None,
        related_demos: list[str] | None = None,
    ):
        """Initialize demonstration.

        Args:
            name: Demonstration name (snake_case)
            description: One-line description
            capabilities: List of capabilities demonstrated (e.g., ["oscura.fft", "oscura.thd"])
            ieee_standards: Applicable IEEE standards (e.g., ["IEEE 181-2011"])
            related_demos: Related demonstration paths
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.ieee_standards = ieee_standards or []
        self.related_demos = related_demos or []

        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.results: dict[str, Any] = {}
        self.errors: list[str] = []

    def execute(self) -> bool:
        """Execute the demonstration with full framework.

        Returns:
            True if demonstration passed, False otherwise
        """
        self.start_time = time.time()

        try:
            # Print header
            self._print_header()

            # Generate test data
            self.section("Generating Test Data")
            data = self.generate_test_data()
            self.info(f"Test data generated: {len(data)} datasets")

            # Run demonstration
            self.section("Running Demonstration")
            self.results = self.run_demonstration(data)

            # Validate results
            self.section("Validation")
            validation_passed = self.validate(self.results)

            # Print summary
            self.end_time = time.time()
            self._print_summary(validation_passed)

            return validation_passed

        except KeyboardInterrupt:
            self.error("Demonstration interrupted by user")
            return False

        except Exception as e:
            self.error(f"Demonstration failed with exception: {e}")
            self.error(traceback.format_exc())
            return False

    @abstractmethod
    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic test data for the demonstration.

        Must be self-contained (no external files required).

        Returns:
            Dictionary of test data (e.g., {"signal": trace, "metadata": ...})
        """

    @abstractmethod
    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute the demonstration logic.

        Args:
            data: Test data from generate_test_data()

        Returns:
            Dictionary of results for validation
        """

    @abstractmethod
    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results.

        Args:
            results: Results from run_demonstration()

        Returns:
            True if validation passed, False otherwise
        """

    def section(self, title: str) -> None:
        """Print a section header.

        Args:
            title: Section title
        """
        print(f"\n{'=' * 80}")
        print(f"{title}")
        print("=" * 80)

    def subsection(self, title: str) -> None:
        """Print a subsection header.

        Args:
            title: Subsection title
        """
        print(f"\n{'-' * 80}")
        print(f"{title}")
        print("-" * 80)

    def info(self, message: str) -> None:
        """Print an info message.

        Args:
            message: Message to print
        """
        print(f"  {message}")

    def success(self, message: str) -> None:
        """Print a success message.

        Args:
            message: Message to print
        """
        print(f"  ✓ {message}")

    def warning(self, message: str) -> None:
        """Print a warning message.

        Args:
            message: Message to print
        """
        print(f"  ⚠ WARNING: {message}")
        self.errors.append(f"WARNING: {message}")

    def error(self, message: str) -> None:
        """Print an error message.

        Args:
            message: Message to print
        """
        print(f"  ✗ ERROR: {message}", file=sys.stderr)
        self.errors.append(f"ERROR: {message}")

    def result(self, key: str, value: Any, unit: str = "") -> None:
        """Print a result with optional unit.

        Args:
            key: Result name
            value: Result value
            unit: Optional unit string
        """
        if unit:
            print(f"  {key}: {value} {unit}")
        else:
            print(f"  {key}: {value}")

    def _print_header(self) -> None:
        """Print demonstration header."""
        print("\n" + "=" * 80)
        print(f"DEMONSTRATION: {self.name}")
        print("=" * 80)
        print(f"\nDescription: {self.description}\n")

        if self.capabilities:
            print("Capabilities Demonstrated:")
            for cap in self.capabilities:
                print(f"  - {cap}")
            print()

        if self.ieee_standards:
            print("IEEE Standards:")
            for std in self.ieee_standards:
                print(f"  - {std}")
            print()

        if self.related_demos:
            print("Related Demonstrations:")
            for demo in self.related_demos:
                print(f"  - {demo}")
            print()

    def _print_summary(self, passed: bool) -> None:
        """Print demonstration summary.

        Args:
            passed: Whether validation passed
        """
        duration = self.end_time - self.start_time

        print(f"\n{'=' * 80}")
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print(f"Name: {self.name}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Errors: {len(self.errors)}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error}")

        print(f"\n{'=' * 80}")
        if passed:
            print("✓ DEMONSTRATION PASSED")
        else:
            print("✗ DEMONSTRATION FAILED")
        print("=" * 80)
        print()

    def get_data_dir(self) -> Path:
        """Get path to demonstration data directory.

        Returns:
            Path to demonstrations/data/
        """
        return Path(__file__).parent.parent / "data"

    def get_output_dir(self) -> Path:
        """Get path to demonstration output directory.

        Creates if doesn't exist.

        Returns:
            Path to demonstrations/data/outputs/<demo_name>/
        """
        output_dir = self.get_data_dir() / "outputs" / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
