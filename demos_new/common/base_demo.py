"""Base class for Oscura demos.

This module provides the BaseDemo abstract class that all demos
should inherit from for consistent structure, validation, and reporting.

Combines the best features from both demos/ and demonstrations/ directories:
- demos/ clean implementation with ValidationSuite
- demonstrations/ rich metadata (capabilities, IEEE standards, related demos)

Usage:
    from demos.common import BaseDemo, ValidationSuite

    class MyDemo(BaseDemo):
        name = "My Feature Demo"
        description = "Demonstrates feature X"
        category = "protocols"

        # Metadata from demonstrations/
        capabilities = ["oscura.decode_uart", "oscura.protocols.uart.UARTDecoder"]
        ieee_standards = ["IEEE 181-2011"]
        related_demos = ["02_spi_basic.py", "03_i2c_basic.py"]

        def generate_data(self):
            # Generate or load test data
            pass

        def run_analysis(self):
            # Perform analysis
            pass

        def validate_results(self, suite: ValidationSuite):
            suite.check_equal("Result", self.result, expected_value)
"""

from __future__ import annotations

import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from demos.common.formatting import (
    BOLD,
    GREEN,
    RED,
    RESET,
    format_duration,
    print_error,
    print_header,
    print_info,
    print_subheader,
    print_success,
)
from demos.common.validation import ValidationSuite


@dataclass
class DemoResult:
    """Result of running a demo.

    Attributes:
        name: Demo name
        success: Whether demo succeeded
        duration: Execution time in seconds
        validation_passed: Whether all validations passed
        validation_count: Number of validations performed
        error: Error message if failed
        outputs: Dictionary of output artifacts
    """

    name: str
    success: bool
    duration: float
    validation_passed: bool
    validation_count: int
    error: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)


class BaseDemo(ABC):
    """Abstract base class for Oscura demos.

    Provides consistent structure for demo implementation:
    1. Data generation/loading
    2. Analysis execution
    3. Result validation
    4. Report generation

    Subclasses must implement:
    - generate_data(): Create or load test data
    - run_analysis(): Perform the demo analysis
    - validate_results(): Add validation checks

    Attributes:
        name: Demo name (shown in headers)
        description: Brief description
        category: Category for organization
        capabilities: List of Oscura capabilities demonstrated
        ieee_standards: Applicable IEEE standards
        related_demos: Related demonstration files
        demo_dir: Directory containing demo script
        data_dir: Directory for demo data files
    """

    name: str = "Unnamed Demo"
    description: str = ""
    category: str = "general"

    # Enhanced metadata from demonstrations/
    capabilities: list[str] = []
    ieee_standards: list[str] = []
    related_demos: list[str] = []

    def __init__(
        self,
        verbose: bool = False,
        data_dir: Path | None = None,
        data_file: Path | str | None = None,
    ):
        """Initialize demo.

        Args:
            verbose: Enable verbose output
            data_dir: Override data directory
            data_file: Override data file path (load from this file instead of generating)
        """
        self.verbose = verbose

        # Set up directories
        # Get the directory of the subclass, not this file
        import inspect

        subclass_file = inspect.getfile(self.__class__)
        self.demo_dir = Path(subclass_file).parent

        self.data_dir = data_dir or (self.demo_dir / "demo_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Store data file override
        self.data_file = Path(data_file) if data_file else None

        # Results storage
        self.results: dict[str, Any] = {}
        self._validation_suite = ValidationSuite(self.name)

    @abstractmethod
    def generate_data(self) -> None:
        """Generate or load test data for demo.

        This method should create any necessary test signals or
        load existing data files. Store data as instance attributes.
        """

    @abstractmethod
    def run_analysis(self) -> None:
        """Execute main demo analysis.

        This method performs the actual analysis being demonstrated.
        Store results in self.results dictionary.
        """

    @abstractmethod
    def validate_results(self, suite: ValidationSuite) -> None:
        """Add validation checks for demo results.

        Args:
            suite: ValidationSuite to add checks to

        Example:
            suite.check_equal("Frame count", len(self.frames), 5)
            suite.check_range("THD", self.thd, 0.0, 10.0)
        """

    def setup(self) -> None:  # noqa: B027
        """Optional setup before demo runs.

        Override to perform any setup not related to data generation.
        """

    def cleanup(self) -> None:  # noqa: B027
        """Optional cleanup after demo runs.

        Override to clean up temporary resources.
        """

    def find_default_data_file(self, filename: str) -> Path | None:
        """Find a default data file in the demo_data directory.

        Args:
            filename: Name of file to look for (e.g., "signal.wfm")

        Returns:
            Path to file if it exists, None otherwise
        """
        data_file_path = self.data_dir / filename
        return data_file_path if data_file_path.exists() else None

    def run(self, validate: bool = True) -> DemoResult:
        """Execute complete demo workflow.

        Args:
            validate: Whether to run validation checks

        Returns:
            DemoResult with execution status and outputs
        """
        start_time = time.time()
        success = False
        error = ""

        try:
            # Print header with metadata
            self._print_header()

            # Setup
            self._run_phase("Setup", self.setup)

            # Generate data
            self._run_phase("Data Generation", self.generate_data)

            # Run analysis
            self._run_phase("Analysis", self.run_analysis)

            # Validation
            if validate:
                self._run_phase("Validation", lambda: self.validate_results(self._validation_suite))
                self._validation_suite.print_summary()

            success = True

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            print_error(error)
            if self.verbose:
                traceback.print_exc()

        finally:
            # Cleanup
            try:
                self.cleanup()
            except Exception as cleanup_error:
                print_error(f"Cleanup failed: {cleanup_error}")

        duration = time.time() - start_time

        # Print summary
        print()
        print_header("DEMO COMPLETE")
        if success:
            print_success(f"Demo completed in {format_duration(duration)}")
        else:
            print_error(f"Demo failed: {error}")

        validation_passed = self._validation_suite.all_passed() if validate else True
        validation_count = len(self._validation_suite.checks) if validate else 0

        return DemoResult(
            name=self.name,
            success=success,
            duration=duration,
            validation_passed=validation_passed,
            validation_count=validation_count,
            error=error,
            outputs=self.results,
        )

    def _print_header(self) -> None:
        """Print demonstration header with metadata."""
        print_header(f"OSCURA DEMO: {self.name}")
        if self.description:
            print_info(self.description)

        # Print capabilities if provided
        if self.capabilities:
            print()
            print_info("Capabilities Demonstrated:")
            for cap in self.capabilities:
                print(f"    - {cap}")

        # Print IEEE standards if provided
        if self.ieee_standards:
            print()
            print_info("IEEE Standards:")
            for std in self.ieee_standards:
                print(f"    - {std}")

        # Print related demos if provided
        if self.related_demos:
            print()
            print_info("Related Demonstrations:")
            for demo in self.related_demos:
                print(f"    - {demo}")

        print()

    def _run_phase(self, name: str, func: callable) -> None:
        """Run a demo phase with timing."""
        print_subheader(name)
        start = time.time()
        func()
        elapsed = time.time() - start
        if self.verbose:
            print_info(f"{name} completed in {format_duration(elapsed)}")


def run_demo_main(demo_class: type[BaseDemo], **kwargs) -> int:
    """Standard main() function for demos.

    Usage in demo script:
        if __name__ == "__main__":
            sys.exit(run_demo_main(MyDemo))

    Args:
        demo_class: BaseDemo subclass
        **kwargs: Additional arguments for demo constructor

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=demo_class.description or f"Run {demo_class.name}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Override data file (load from this file instead of generating)",
    )

    args = parser.parse_args()

    try:
        demo = demo_class(
            verbose=args.verbose, data_dir=args.data_dir, data_file=args.data_file, **kwargs
        )
        result = demo.run(validate=not args.no_validate)

        if result.success and result.validation_passed:
            print(f"\n{GREEN}{BOLD}Demo validation passed!{RESET}\n")
            return 0
        else:
            if not result.success:
                print(f"\n{RED}{BOLD}Demo execution failed!{RESET}\n")
            else:
                print(f"\n{RED}{BOLD}Demo validation failed!{RESET}\n")
            return 1

    except Exception as e:
        print(f"\n{RED}{BOLD}Unexpected error: {e}{RESET}\n", file=sys.stderr)
        traceback.print_exc()
        return 1
