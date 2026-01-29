"""Simple validation helper functions for demonstrations.

These helpers provide convenience functions for common validation patterns,
wrapping the ValidationSuite class with simpler standalone functions.
"""

from __future__ import annotations


def validate_approximately(
    actual: float,
    expected: float,
    tolerance: float = 0.01,
    name: str = "value",
) -> bool:
    """Validate that value is approximately equal to expected.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Relative tolerance (default 1%)
        name: Name for error messages

    Returns:
        True if within tolerance
    """
    diff = abs(actual - expected)
    max_diff = abs(expected * tolerance)

    if diff > max_diff:
        print(f"  ✗ {name}: {actual} != {expected} (diff {diff:.4f} > {max_diff:.4f})")
        return False

    print(f"  ✓ {name}: {actual:.4f} ≈ {expected:.4f} (within {tolerance * 100}%)")
    return True


def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> bool:
    """Validate that a value is within a range.

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for error messages

    Returns:
        True if within range
    """
    if value < min_val or value > max_val:
        print(f"  ✗ {name}: {value} not in range [{min_val}, {max_val}]")
        return False
    print(f"  ✓ {name}: {value} in range [{min_val}, {max_val}]")
    return True
