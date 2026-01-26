"""Comprehensive tests for validation utility module.

Tests validation helper functions.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from oscura.utils.validation import validate_protocol_spec

# =============================================================================
# Mock Protocol Spec
# =============================================================================


@dataclass
class MockProtocolSpec:
    """Mock protocol specification for testing."""

    name: str = ""
    fields: list[Any] | None = None


# =============================================================================
# Valid Specification Tests
# =============================================================================


def test_validate_protocol_spec_valid() -> None:
    """Test validation of valid protocol specification."""
    spec = MockProtocolSpec(name="UART", fields=["data", "parity", "stop"])

    # Should not raise
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_minimal_valid() -> None:
    """Test validation of minimal valid specification."""
    spec = MockProtocolSpec(name="Test", fields=["field1"])

    # Should not raise
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_many_fields() -> None:
    """Test validation with many fields."""
    spec = MockProtocolSpec(name="Complex", fields=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])

    # Should not raise
    validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Invalid Name Tests
# =============================================================================


def test_validate_protocol_spec_empty_name() -> None:
    """Test validation fails with empty protocol name."""
    spec = MockProtocolSpec(name="", fields=["field1"])

    with pytest.raises(ValueError, match="Protocol name is required"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_none_name() -> None:
    """Test validation with None name (if name attribute is missing)."""
    spec = MockProtocolSpec(name="", fields=["field1"])
    spec.name = ""  # Empty string is falsy

    with pytest.raises(ValueError, match="Protocol name is required"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Invalid Fields Tests
# =============================================================================


def test_validate_protocol_spec_empty_fields() -> None:
    """Test validation fails with empty fields list."""
    spec = MockProtocolSpec(name="Test", fields=[])

    with pytest.raises(ValueError, match="Protocol must have at least one field"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_none_fields() -> None:
    """Test validation fails with None fields."""
    spec = MockProtocolSpec(name="Test", fields=None)

    with pytest.raises(ValueError, match="Protocol must have at least one field"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Combined Invalid Tests
# =============================================================================


def test_validate_protocol_spec_both_empty() -> None:
    """Test validation with both name and fields empty."""
    spec = MockProtocolSpec(name="", fields=[])

    # Should raise error for name first (order of checks)
    with pytest.raises(ValueError, match="Protocol name is required"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_both_none() -> None:
    """Test validation with both name and fields missing."""
    spec = MockProtocolSpec(name="", fields=None)

    with pytest.raises(ValueError, match="Protocol name is required"):
        validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Name Validation Edge Cases
# =============================================================================


def test_validate_protocol_spec_whitespace_name() -> None:
    """Test validation with whitespace-only name."""
    spec = MockProtocolSpec(name="   ", fields=["field1"])

    # Whitespace string is truthy, so validation should pass
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_special_chars_name() -> None:
    """Test validation with special characters in name."""
    spec = MockProtocolSpec(name="UART-9600-8N1", fields=["field1"])

    # Should pass - name just needs to be non-empty
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_unicode_name() -> None:
    """Test validation with Unicode name."""
    spec = MockProtocolSpec(name="协议", fields=["field1"])

    # Should pass - any non-empty string is valid
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_very_long_name() -> None:
    """Test validation with very long protocol name."""
    spec = MockProtocolSpec(name="X" * 1000, fields=["field1"])

    # Should pass - no length restriction
    validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Fields Validation Edge Cases
# =============================================================================


def test_validate_protocol_spec_single_field() -> None:
    """Test validation with single field."""
    spec = MockProtocolSpec(name="Test", fields=["data"])

    # Should pass - at least one field required
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_various_field_types() -> None:
    """Test validation with various field types in list."""
    spec = MockProtocolSpec(name="Test", fields=["string", 123, {"dict": "field"}, None])

    # Should pass - validation only checks if list is non-empty
    validate_protocol_spec(spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_nested_structure() -> None:
    """Test validation with nested field structures."""
    spec = MockProtocolSpec(
        name="Complex",
        fields=[
            {"name": "field1", "type": "uint8"},
            {"name": "field2", "type": "string"},
        ],
    )

    # Should pass
    validate_protocol_spec(spec)  # type: ignore[arg-type]


# =============================================================================
# Validation Behavior Tests
# =============================================================================


def test_validate_protocol_spec_returns_none() -> None:
    """Test that validation returns None on success."""
    spec = MockProtocolSpec(name="Test", fields=["field1"])

    result = validate_protocol_spec(spec)  # type: ignore[arg-type,func-returns-value]

    assert result is None


def test_validate_protocol_spec_multiple_calls() -> None:
    """Test multiple validation calls."""
    spec1 = MockProtocolSpec(name="Test1", fields=["field1"])
    spec2 = MockProtocolSpec(name="Test2", fields=["field2", "field3"])
    spec3 = MockProtocolSpec(name="Test3", fields=["field4"])

    # All should pass without raising
    validate_protocol_spec(spec1)  # type: ignore[arg-type]
    validate_protocol_spec(spec2)  # type: ignore[arg-type]
    validate_protocol_spec(spec3)  # type: ignore[arg-type]


def test_validate_protocol_spec_does_not_modify() -> None:
    """Test that validation does not modify the spec."""
    original_name = "Test"
    original_fields = ["field1", "field2"]
    spec = MockProtocolSpec(name=original_name, fields=original_fields.copy())

    validate_protocol_spec(spec)  # type: ignore[arg-type]

    assert spec.name == original_name
    assert spec.fields == original_fields


# =============================================================================
# Integration Tests
# =============================================================================


def test_validate_protocol_spec_common_protocols() -> None:
    """Test validation of common protocol specifications."""
    uart_spec = MockProtocolSpec(name="UART", fields=["start", "data", "parity", "stop"])

    spi_spec = MockProtocolSpec(name="SPI", fields=["clock", "mosi", "miso", "cs"])

    i2c_spec = MockProtocolSpec(name="I2C", fields=["address", "data", "ack"])

    # All should pass
    validate_protocol_spec(uart_spec)  # type: ignore[arg-type]
    validate_protocol_spec(spi_spec)  # type: ignore[arg-type]
    validate_protocol_spec(i2c_spec)  # type: ignore[arg-type]


def test_validate_protocol_spec_error_messages_clear() -> None:
    """Test that error messages are clear and helpful."""
    # Missing name
    spec1 = MockProtocolSpec(name="", fields=["field1"])
    try:
        validate_protocol_spec(spec1)  # type: ignore[arg-type]
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "name" in str(e).lower()
        assert "required" in str(e).lower()

    # Missing fields
    spec2 = MockProtocolSpec(name="Test", fields=[])
    try:
        validate_protocol_spec(spec2)  # type: ignore[arg-type]
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "field" in str(e).lower()
        assert "at least one" in str(e).lower()
