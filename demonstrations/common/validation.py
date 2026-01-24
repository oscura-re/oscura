"""Validation utilities for demonstrations."""

from __future__ import annotations

from typing import Any


def validate_results(results: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Validate demonstration results against expected values.

    Args:
        results: Actual results from demonstration
        expected: Expected values/constraints

    Returns:
        True if all validations pass

    Example:
        results = {"amplitude": 5.0, "frequency": 1000}
        expected = {
            "amplitude": {"min": 4.5, "max": 5.5},
            "frequency": 1000,
        }
        assert validate_results(results, expected)
    """
    for key, expected_value in expected.items():
        if key not in results:
            print(f"  ✗ Missing expected key: {key}")
            return False

        actual = results[key]

        # Handle dict constraints (min/max)
        if isinstance(expected_value, dict):
            if "min" in expected_value and actual < expected_value["min"]:
                print(f"  ✗ {key}: {actual} < minimum {expected_value['min']}")
                return False
            if "max" in expected_value and actual > expected_value["max"]:
                print(f"  ✗ {key}: {actual} > maximum {expected_value['max']}")
                return False
        # Handle exact value
        elif actual != expected_value:
            print(f"  ✗ {key}: expected {expected_value}, got {actual}")
            return False

        print(f"  ✓ {key}: {actual} (valid)")

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


def validate_exists(obj: Any, name: str) -> bool:
    """Validate that an object exists (not None).

    Args:
        obj: Object to check
        name: Name for error messages

    Returns:
        True if exists
    """
    if obj is None:
        print(f"  ✗ {name}: does not exist")
        return False
    print(f"  ✓ {name}: exists")
    return True


def validate_length(seq: Any, expected_length: int, name: str = "sequence") -> bool:
    """Validate sequence length.

    Args:
        seq: Sequence to check
        expected_length: Expected length
        name: Name for error messages

    Returns:
        True if length matches
    """
    actual_length = len(seq)
    if actual_length != expected_length:
        print(f"  ✗ {name}: length {actual_length} != expected {expected_length}")
        return False
    print(f"  ✓ {name}: length {actual_length}")
    return True


def validate_type(obj: Any, expected_type: type, name: str = "object") -> bool:
    """Validate object type.

    Args:
        obj: Object to check
        expected_type: Expected type
        name: Name for error messages

    Returns:
        True if type matches
    """
    if not isinstance(obj, expected_type):
        print(f"  ✗ {name}: type {type(obj).__name__} != expected {expected_type.__name__}")
        return False
    print(f"  ✓ {name}: type {expected_type.__name__}")
    return True


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
        print(f"  ✗ {name}: {actual} != {expected} (diff {diff} > {max_diff})")
        return False

    print(f"  ✓ {name}: {actual} ≈ {expected} (within {tolerance * 100}%)")
    return True
