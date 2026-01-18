"""Validation framework for demo self-verification.

This module provides a structured approach to validating demo outputs,
ensuring demos can verify their own correctness.

Usage:
    from demos.common.validation import ValidationSuite

    suite = ValidationSuite("UART Decoding")
    suite.check_equal("Frame count", len(frames), 5)
    suite.check_range("THD", thd_value, 0.0, 10.0)
    suite.check_true("CRC valid", frame.crc_ok)

    suite.print_summary()
    suite.assert_all_passed()  # Raises if any failed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Import formatting utilities
from demos.common.formatting import (
    GREEN,
    RED,
    RESET,
    print_table,
)


@dataclass
class ValidationResult:
    """Result of a single validation check.

    Attributes:
        name: Check name/description
        passed: Whether check passed
        expected: Expected value/condition
        actual: Actual value
        message: Optional additional message
        category: Optional category for grouping
    """

    name: str
    passed: bool
    expected: Any
    actual: Any
    message: str = ""
    category: str = "general"


@dataclass
class ValidationSuite:
    """Collection of validation checks for a demo.

    Provides methods for common validation patterns and
    a summary report of all checks.
    """

    name: str
    checks: list[ValidationResult] = field(default_factory=list)
    _abort_on_fail: bool = False

    def check_equal(
        self,
        name: str,
        actual: Any,
        expected: Any,
        category: str = "general",
    ) -> bool:
        """Validate equality.

        Args:
            name: Check name
            actual: Actual value
            expected: Expected value
            category: Check category

        Returns:
            True if check passed
        """
        passed = actual == expected
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=expected,
                actual=actual,
                category=category,
            )
        )
        self._handle_result(name, passed, expected, actual)
        return passed

    def check_not_equal(
        self,
        name: str,
        actual: Any,
        not_expected: Any,
        category: str = "general",
    ) -> bool:
        """Validate inequality.

        Args:
            name: Check name
            actual: Actual value
            not_expected: Value that should NOT match
            category: Check category

        Returns:
            True if check passed
        """
        passed = actual != not_expected
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"!= {not_expected}",
                actual=actual,
                category=category,
            )
        )
        self._handle_result(name, passed, f"!= {not_expected}", actual)
        return passed

    def check_range(
        self,
        name: str,
        value: float,
        min_val: float,
        max_val: float,
        category: str = "general",
    ) -> bool:
        """Validate value within range.

        Args:
            name: Check name
            value: Value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            category: Check category

        Returns:
            True if check passed
        """
        passed = min_val <= value <= max_val
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"[{min_val}, {max_val}]",
                actual=value,
                category=category,
            )
        )
        self._handle_result(name, passed, f"[{min_val}, {max_val}]", value)
        return passed

    def check_true(
        self,
        name: str,
        condition: bool,
        message: str = "",
        category: str = "general",
    ) -> bool:
        """Validate boolean condition is true.

        Args:
            name: Check name
            condition: Boolean condition
            message: Optional message
            category: Check category

        Returns:
            True if check passed
        """
        self.checks.append(
            ValidationResult(
                name=name,
                passed=condition,
                expected=True,
                actual=condition,
                message=message,
                category=category,
            )
        )
        self._handle_result(name, condition, True, condition)
        return condition

    def check_false(
        self,
        name: str,
        condition: bool,
        message: str = "",
        category: str = "general",
    ) -> bool:
        """Validate boolean condition is false.

        Args:
            name: Check name
            condition: Boolean condition
            message: Optional message
            category: Check category

        Returns:
            True if check passed
        """
        passed = not condition
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=False,
                actual=condition,
                message=message,
                category=category,
            )
        )
        self._handle_result(name, passed, False, condition)
        return passed

    def check_greater(
        self,
        name: str,
        value: float,
        threshold: float,
        category: str = "general",
    ) -> bool:
        """Validate value greater than threshold.

        Args:
            name: Check name
            value: Value to check
            threshold: Minimum threshold (exclusive)
            category: Check category

        Returns:
            True if check passed
        """
        passed = value > threshold
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"> {threshold}",
                actual=value,
                category=category,
            )
        )
        self._handle_result(name, passed, f"> {threshold}", value)
        return passed

    def check_less(
        self,
        name: str,
        value: float,
        threshold: float,
        category: str = "general",
    ) -> bool:
        """Validate value less than threshold.

        Args:
            name: Check name
            value: Value to check
            threshold: Maximum threshold (exclusive)
            category: Check category

        Returns:
            True if check passed
        """
        passed = value < threshold
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"< {threshold}",
                actual=value,
                category=category,
            )
        )
        self._handle_result(name, passed, f"< {threshold}", value)
        return passed

    def check_greater_equal(
        self,
        name: str,
        value: float,
        threshold: float,
        category: str = "general",
    ) -> bool:
        """Validate value greater than or equal to threshold.

        Args:
            name: Check name
            value: Value to check
            threshold: Minimum threshold (inclusive)
            category: Check category

        Returns:
            True if check passed
        """
        passed = value >= threshold
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f">= {threshold}",
                actual=value,
                category=category,
            )
        )
        self._handle_result(name, passed, f">= {threshold}", value)
        return passed

    def check_less_equal(
        self,
        name: str,
        value: float,
        threshold: float,
        category: str = "general",
    ) -> bool:
        """Validate value less than or equal to threshold.

        Args:
            name: Check name
            value: Value to check
            threshold: Maximum threshold (inclusive)
            category: Check category

        Returns:
            True if check passed
        """
        passed = value <= threshold
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"<= {threshold}",
                actual=value,
                category=category,
            )
        )
        self._handle_result(name, passed, f"<= {threshold}", value)
        return passed

    def check_close(
        self,
        name: str,
        actual: float,
        expected: float,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        category: str = "general",
    ) -> bool:
        """Validate values are close (numpy.isclose).

        Args:
            name: Check name
            actual: Actual value
            expected: Expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
            category: Check category

        Returns:
            True if check passed
        """
        passed = np.isclose(actual, expected, rtol=rtol, atol=atol)
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"{expected} (rtol={rtol})",
                actual=actual,
                category=category,
            )
        )
        self._handle_result(name, passed, f"~{expected}", actual)
        return passed

    def check_contains(
        self,
        name: str,
        container: Any,
        item: Any,
        category: str = "general",
    ) -> bool:
        """Validate container contains item.

        Args:
            name: Check name
            container: Container to search
            item: Item to find
            category: Check category

        Returns:
            True if check passed
        """
        passed = item in container
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=f"contains {item}",
                actual=container,
                category=category,
            )
        )
        self._handle_result(name, passed, f"contains {item}", type(container).__name__)
        return passed

    def check_file_exists(
        self,
        name: str,
        path: Path | str,
        category: str = "files",
    ) -> bool:
        """Validate file was created.

        Args:
            name: Check name
            path: File path to check
            category: Check category

        Returns:
            True if check passed
        """
        path = Path(path)
        exists = path.exists()
        self.checks.append(
            ValidationResult(
                name=name,
                passed=exists,
                expected="file exists",
                actual="exists" if exists else "missing",
                category=category,
            )
        )
        self._handle_result(name, exists, "exists", "missing" if not exists else "exists")
        return exists

    def check_not_empty(
        self,
        name: str,
        value: Any,
        category: str = "general",
    ) -> bool:
        """Validate value is not empty.

        Args:
            name: Check name
            value: Value to check
            category: Check category

        Returns:
            True if check passed
        """
        passed = bool(value) if not isinstance(value, np.ndarray) else len(value) > 0
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected="not empty",
                actual=f"length {len(value) if hasattr(value, '__len__') else 'N/A'}",
                category=category,
            )
        )
        self._handle_result(name, passed, "not empty", "empty" if not passed else "has content")
        return passed

    def check_array_shape(
        self,
        name: str,
        array: np.ndarray,
        expected_shape: tuple,
        category: str = "arrays",
    ) -> bool:
        """Validate array shape.

        Args:
            name: Check name
            array: NumPy array
            expected_shape: Expected shape tuple
            category: Check category

        Returns:
            True if check passed
        """
        passed = array.shape == expected_shape
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected=expected_shape,
                actual=array.shape,
                category=category,
            )
        )
        self._handle_result(name, passed, expected_shape, array.shape)
        return passed

    def check_all_finite(
        self,
        name: str,
        array: np.ndarray,
        category: str = "arrays",
    ) -> bool:
        """Validate array contains no NaN or Inf.

        Args:
            name: Check name
            array: NumPy array
            category: Check category

        Returns:
            True if check passed
        """
        passed = np.all(np.isfinite(array))
        nan_count = np.sum(np.isnan(array))
        inf_count = np.sum(np.isinf(array))
        self.checks.append(
            ValidationResult(
                name=name,
                passed=passed,
                expected="all finite",
                actual=f"NaN: {nan_count}, Inf: {inf_count}",
                category=category,
            )
        )
        self._handle_result(name, passed, "all finite", f"NaN:{nan_count},Inf:{inf_count}")
        return passed

    def summarize(self) -> tuple[int, int, list[ValidationResult]]:
        """Get summary of all checks.

        Returns:
            Tuple of (passed_count, total_count, failed_checks)
        """
        passed = sum(1 for c in self.checks if c.passed)
        failed = [c for c in self.checks if not c.passed]
        return passed, len(self.checks), failed

    def print_summary(self, show_all: bool = False) -> None:
        """Print validation summary.

        Args:
            show_all: If True, show all checks; otherwise only failed
        """
        passed, total, failed = self.summarize()

        print(f"\n{'=' * 60}")
        print(f"VALIDATION SUMMARY: {self.name}")
        print(f"{'=' * 60}")

        if show_all:
            rows = []
            for check in self.checks:
                status = "PASS" if check.passed else "FAIL"
                rows.append([check.name, str(check.expected), str(check.actual), status])
            print_table(["Check", "Expected", "Actual", "Status"], rows)

        print(f"\nResults: {passed}/{total} checks passed")

        if failed:
            print(f"\n{RED}Failed checks:{RESET}")
            for check in failed:
                print(f"  - {check.name}: expected {check.expected}, got {check.actual}")
        else:
            print(f"\n{GREEN}All checks passed!{RESET}")

    def assert_all_passed(self) -> None:
        """Assert all checks passed, raising on failure.

        Raises:
            AssertionError: If any check failed
        """
        passed, total, failed = self.summarize()
        if failed:
            messages = [f"{c.name}: expected {c.expected}, got {c.actual}" for c in failed]
            raise AssertionError(
                f"{len(failed)} validation(s) failed:\n" + "\n".join(f"  - {m}" for m in messages)
            )

    def all_passed(self) -> bool:
        """Check if all validations passed.

        Returns:
            True if all checks passed
        """
        return all(c.passed for c in self.checks)

    def _handle_result(self, name: str, passed: bool, expected: Any, actual: Any) -> None:
        """Handle individual result (print status, abort if configured)."""
        if passed:
            status = f"{GREEN}PASS{RESET}"
        else:
            status = f"{RED}FAIL{RESET}"

        # Print inline result
        actual_str = str(actual)
        if len(actual_str) > 30:
            actual_str = actual_str[:27] + "..."

        print(f"  [{status}] {name}: {actual_str}")

        if self._abort_on_fail and not passed:
            raise AssertionError(f"Validation failed: {name} (expected {expected}, got {actual})")


def validate_demo_output(
    results: dict[str, Any],
    expected: dict[str, Any],
    suite_name: str = "Demo Output",
) -> ValidationSuite:
    """Convenience function to validate demo results dict.

    Args:
        results: Dictionary of actual results
        expected: Dictionary of expected values/ranges
        suite_name: Name for validation suite

    Returns:
        ValidationSuite with all checks performed
    """
    suite = ValidationSuite(suite_name)

    for key, spec in expected.items():
        if key not in results:
            suite.check_true(f"{key} exists", False, "Key missing from results")
            continue

        actual = results[key]

        if isinstance(spec, tuple) and len(spec) == 2:
            # Range check
            suite.check_range(key, actual, spec[0], spec[1])
        elif isinstance(spec, type):
            # Type check
            suite.check_true(f"{key} type", isinstance(actual, spec))
        else:
            # Equality check
            suite.check_equal(key, actual, spec)

    return suite
