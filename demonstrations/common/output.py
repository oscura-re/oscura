"""Output formatting utilities for demonstrations."""

from __future__ import annotations


def print_info(message: str) -> None:
    """Print informational message.

    Args:
        message: Message to print

    Example:
        >>> print_info("Loading data...")
        INFO: Loading data...
    """
    print(f"ℹ️  {message}")  # noqa: RUF001


def print_result(key: str, value: str | int | float, unit: str = "") -> None:
    """Print key-value result.

    Args:
        key: Result key/label
        value: Result value
        unit: Optional unit suffix

    Example:
        >>> print_result("Sample rate", 1.0, "MHz")
        ✓ Sample rate: 1.0 MHz
    """
    if unit:
        print(f"✓ {key}: {value} {unit}")
    else:
        print(f"✓ {key}: {value}")


class ValidationSuite:
    """Collects and reports validation results.

    Example:
        >>> suite = ValidationSuite()
        >>> suite.add_check("Data loaded", True)
        >>> suite.add_check("Valid range", False, "Out of bounds")
        >>> suite.report()
        Validation Results:
        ✓ Data loaded: PASS
        ✗ Valid range: FAIL (Out of bounds)
    """

    def __init__(self) -> None:
        """Initialize empty validation suite."""
        self.checks: list[tuple[str, bool, str]] = []

    def add_check(self, name: str, passed: bool, message: str = "") -> None:
        """Add validation check result.

        Args:
            name: Check name
            passed: Whether check passed
            message: Optional failure message
        """
        self.checks.append((name, passed, message))

    def report(self) -> None:
        """Print validation results."""
        print("\nValidation Results:")
        for name, passed, message in self.checks:
            if passed:
                print(f"✓ {name}: PASS")
            else:
                msg = f" ({message})" if message else ""
                print(f"✗ {name}: FAIL{msg}")

    def all_passed(self) -> bool:
        """Check if all validations passed.

        Returns:
            True if all checks passed
        """
        return all(passed for _, passed, _ in self.checks)


__all__ = ["ValidationSuite", "print_info", "print_result"]
