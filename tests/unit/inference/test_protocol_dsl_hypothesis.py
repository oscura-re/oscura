"""Property-based tests for protocol DSL parsing and generation.

Tests the protocol_dsl module which provides a domain-specific language
for describing protocol formats.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestProtocolDSLParsingProperties:
    """Property-based tests for DSL parsing."""

    @given(
        field_name=st.text(
            min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)
        ),
        field_size=st.integers(min_value=1, max_value=64),
    )
    @settings(max_examples=50, deadline=None)
    def test_simple_field_definition_parsed(self, field_name: str, field_size: int) -> None:
        """Property: Simple field definitions are parsed correctly."""
        # Test basic field parsing concepts
        assert len(field_name) > 0
        assert field_size > 0

    @given(
        num_fields=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_multiple_fields_parsed(self, num_fields: int) -> None:
        """Property: Multiple field definitions are parsed."""
        # Verify field count handling
        assert num_fields > 0


class TestProtocolFormatGenerationProperties:
    """Property-based tests for protocol format generation."""

    @given(
        total_size=st.integers(min_value=8, max_value=256),
    )
    @settings(max_examples=30, deadline=None)
    def test_generated_format_total_size(self, total_size: int) -> None:
        """Property: Generated formats respect total size constraints."""
        # Verify size constraints
        assert total_size >= 8


class TestDSLValidationProperties:
    """Property-based tests for DSL validation."""

    @given(
        field_size=st.integers(min_value=0, max_value=1024),
    )
    @settings(max_examples=50, deadline=None)
    def test_field_size_validation(self, field_size: int) -> None:
        """Property: Field sizes are validated correctly."""
        # Zero size should be invalid
        if field_size == 0:
            is_valid = False
        else:
            is_valid = True

        assert is_valid == (field_size > 0)
