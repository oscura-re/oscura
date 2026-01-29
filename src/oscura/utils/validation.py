"""Validation utility functions.

This module provides validation helpers used across export and other modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.workflows.reverse_engineering import ProtocolSpec


def validate_protocol_spec(spec: "ProtocolSpec") -> None:
    """Validate protocol specification.

    Ensures protocol specification has required fields and is well-formed.

    Args:
        spec: Protocol specification to validate

    Raises:
        ValueError: If spec is invalid (missing name or fields)

    Example:
        >>> from oscura.analyzers.protocols.base import ProtocolSpec
        >>> spec = ProtocolSpec(name="MyProtocol", fields=[...])
        >>> validate_protocol_spec(spec)  # No error if valid
    """
    if not spec.name:
        raise ValueError("Protocol name is required")

    if not spec.fields:
        raise ValueError("Protocol must have at least one field")
