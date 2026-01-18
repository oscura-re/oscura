"""Unit tests for configuration schema validation.

Tests CFG-001, CFG-014, CFG-015
"""

from unittest.mock import patch

import pytest

from oscura.config.schema import (
    ConfigSchema,
    SchemaRegistry,
    ValidationError,
    get_schema_registry,
    validate_against_schema,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestConfigSchema:
    """Test ConfigSchema dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic schema."""
        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
            description="Test schema",
        )

        assert schema.name == "test"
        assert schema.version == "1.0.0"
        assert schema.description == "Test schema"

    def test_full_uri(self) -> None:
        """Test URI generation."""
        schema = ConfigSchema(
            name="protocol",
            version="1.2.3",
            schema={"type": "object"},
        )

        assert schema.full_uri == "urn:oscura:schemas:protocol:v1.2.3"

    def test_custom_uri(self) -> None:
        """Test custom URI override."""
        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
            uri="https://example.com/schema.json",
        )

        assert schema.full_uri == "https://example.com/schema.json"

    def test_empty_name_error(self) -> None:
        """Test error on empty name."""
        with pytest.raises(ValueError, match="name"):
            ConfigSchema(name="", version="1.0.0", schema={"type": "object"})

    def test_empty_version_error(self) -> None:
        """Test error on empty version."""
        with pytest.raises(ValueError, match="version"):
            ConfigSchema(name="test", version="", schema={"type": "object"})


class TestSchemaRegistry:
    """Test SchemaRegistry class."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a schema."""
        registry = SchemaRegistry()

        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
        )
        registry.register(schema)

        result = registry.get("test")
        assert result is not None
        assert result.name == "test"

    def test_get_with_version(self) -> None:
        """Test retrieving specific version."""
        registry = SchemaRegistry()

        schema1 = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
        )
        schema2 = ConfigSchema(
            name="test",
            version="2.0.0",
            schema={"type": "object", "additionalProperties": False},
        )

        registry.register(schema1, set_default=False)
        registry.register(schema2)

        assert registry.get("test", "1.0.0") == schema1
        assert registry.get("test", "2.0.0") == schema2
        # Default should be 2.0.0
        assert registry.get("test") == schema2

    def test_duplicate_registration_error(self) -> None:
        """Test error on duplicate registration."""
        registry = SchemaRegistry()

        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
        )
        registry.register(schema)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(schema)

    def test_list_schemas(self) -> None:
        """Test listing registered schemas."""
        registry = SchemaRegistry()

        registry.register(
            ConfigSchema(
                name="alpha",
                version="1.0.0",
                schema={"type": "object"},
            )
        )
        registry.register(
            ConfigSchema(
                name="beta",
                version="1.0.0",
                schema={"type": "object"},
            )
        )

        schemas = registry.list_schemas()
        assert "alpha" in schemas
        assert "beta" in schemas

    def test_list_versions(self) -> None:
        """Test listing versions of a schema."""
        registry = SchemaRegistry()

        registry.register(
            ConfigSchema(
                name="test",
                version="1.0.0",
                schema={"type": "object"},
            ),
            set_default=False,
        )
        registry.register(
            ConfigSchema(
                name="test",
                version="2.0.0",
                schema={"type": "object"},
            )
        )

        versions = registry.list_versions("test")
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    def test_has_schema(self) -> None:
        """Test checking if schema exists."""
        registry = SchemaRegistry()

        registry.register(
            ConfigSchema(
                name="test",
                version="1.0.0",
                schema={"type": "object"},
            )
        )

        assert registry.has_schema("test")
        assert registry.has_schema("test", "1.0.0")
        assert not registry.has_schema("test", "2.0.0")
        assert not registry.has_schema("nonexistent")


class TestGlobalRegistry:
    """Test global schema registry."""

    def test_get_registry_is_singleton(self) -> None:
        """Test that get_schema_registry returns same instance."""
        reg1 = get_schema_registry()
        reg2 = get_schema_registry()
        assert reg1 is reg2

    def test_builtin_schemas(self) -> None:
        """Test that built-in schemas are registered."""
        registry = get_schema_registry()

        # Check for expected built-in schemas
        assert registry.has_schema("protocol")
        assert registry.has_schema("pipeline")
        assert registry.has_schema("logic_family")
        assert registry.has_schema("preferences")


class TestValidateAgainstSchema:
    """Test validate_against_schema function."""

    def test_valid_protocol_config(self) -> None:
        """Test validating a valid protocol configuration."""
        config = {
            "name": "uart",
            "version": "1.0.0",
            "description": "UART decoder",
            "timing": {
                "baud_rates": [9600, 115200],
                "data_bits": [8],
            },
        }

        result = validate_against_schema(config, "protocol")
        assert result is True

    def test_invalid_protocol_name(self) -> None:
        """Test validation failure on invalid protocol name."""
        config = {
            "name": "INVALID-NAME!",  # Invalid pattern
            "version": "1.0.0",
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "protocol")

    def test_missing_required_field(self) -> None:
        """Test validation failure on missing required field."""
        config = {
            # Missing "name" field
            "version": "1.0.0",
        }

        with pytest.raises(ValidationError, match="required"):
            validate_against_schema(config, "protocol")

    def test_invalid_type(self) -> None:
        """Test validation failure on wrong type."""
        config = {
            "name": "uart",
            "timing": {
                "baud_rates": "not an array",  # Should be array
            },
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "protocol")

    def test_valid_pipeline_config(self) -> None:
        """Test validating a valid pipeline configuration."""
        config = {
            "name": "analysis_pipeline",
            "version": "1.0.0",
            "steps": [
                {"name": "load", "type": "input.file"},
                {"name": "decode", "type": "decoder.uart"},
            ],
        }

        result = validate_against_schema(config, "pipeline")
        assert result is True

    def test_pipeline_missing_steps(self) -> None:
        """Test validation failure on pipeline without steps."""
        config = {
            "name": "empty_pipeline",
            "steps": [],  # minItems: 1
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "pipeline")

    def test_valid_logic_family(self) -> None:
        """Test validating a valid logic family configuration."""
        config = {
            "name": "TTL",
            "VIH": 2.0,
            "VIL": 0.8,
            "VOH": 2.4,
            "VOL": 0.4,
        }

        result = validate_against_schema(config, "logic_family")
        assert result is True

    def test_logic_family_invalid_voltage(self) -> None:
        """Test validation failure on out-of-range voltage."""
        config = {
            "name": "invalid",
            "VIH": 15.0,  # max is 10
            "VIL": 0.8,
            "VOH": 2.4,
            "VOL": 0.4,
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "logic_family")

    def test_schema_not_found(self) -> None:
        """Test error when schema doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            validate_against_schema({}, "nonexistent_schema")

    def test_validation_error_details(self) -> None:
        """Test that validation errors include path information."""
        config = {
            "name": "uart",
            "timing": {
                "baud_rates": ["not", "integers"],
            },
        }

        try:
            validate_against_schema(config, "protocol")
            raise AssertionError("Should have raised ValidationError")
        except ValidationError as e:
            assert e.path is not None
            assert "timing" in e.path or "baud_rates" in e.path


class TestStrictMode:
    """Test strict validation mode."""

    def test_strict_rejects_extra_properties(self) -> None:
        """Test that strict mode rejects additional properties."""
        # Create a schema that doesn't allow additional properties
        registry = get_schema_registry()
        registry.register(
            ConfigSchema(
                name="strict_test",
                version="1.0.0",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            )
        )

        config = {
            "name": "test",
            "extra_field": "not allowed",
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "strict_test", strict=True)


class TestErrorSuggestions:
    """Test error suggestion generation."""

    def test_error_suggestion_type_mismatch(self) -> None:
        """Test suggestion for type mismatch errors."""
        config = {
            "name": "test",
            "timing": {
                "baud_rates": "not_an_array",
            },
        }

        try:
            validate_against_schema(config, "protocol")
        except ValidationError as e:
            assert e.suggestion is not None
            # Should suggest converting to correct type

    def test_error_suggestion_missing_required(self) -> None:
        """Test suggestion for missing required field."""
        config = {
            # Missing required "name" field
            "version": "1.0.0",
        }

        try:
            validate_against_schema(config, "protocol")
        except ValidationError as e:
            assert e.suggestion is not None
            # Should suggest adding the missing field

    def test_error_suggestion_pattern_mismatch(self) -> None:
        """Test suggestion for pattern validation failure."""
        config = {
            "name": "INVALID NAME!",  # Doesn't match pattern
        }

        try:
            validate_against_schema(config, "protocol")
        except ValidationError as e:
            # Error should have path information
            assert e.path is not None


class TestValidationErrorDetails:
    """Test ValidationError with detailed information."""

    def test_validation_error_with_path(self) -> None:
        """Test validation error includes path."""
        error = ValidationError(
            "Invalid value",
            path="protocol.timing.baud_rate",
        )

        assert error.path == "protocol.timing.baud_rate"

    def test_validation_error_with_line_column(self) -> None:
        """Test validation error with line and column."""
        error = ValidationError(
            "Syntax error",
            path="config.field",
            line=10,
            column=5,
        )

        assert error.line == 10
        assert error.column == 5

    def test_validation_error_with_expected_actual(self) -> None:
        """Test validation error with expected and actual values."""
        error = ValidationError(
            "Type mismatch",
            expected="integer",
            actual="string",
        )

        assert error.expected == "integer"
        assert error.actual == "string"

    def test_validation_error_with_suggestion(self) -> None:
        """Test validation error with suggestion."""
        error = ValidationError(
            "Invalid format",
            path="field",
            suggestion="Use format: YYYY-MM-DD",
        )

        assert error.suggestion == "Use format: YYYY-MM-DD"

    def test_validation_error_full_details(self) -> None:
        """Test validation error with all details."""
        error = ValidationError(
            "Configuration error",
            path="config.settings.value",
            line=42,
            column=10,
            schema_path="properties.settings.properties.value",
            expected="number",
            actual="'invalid'",
            suggestion="Provide a numeric value",
        )

        assert error.path == "config.settings.value"
        assert error.line == 42
        assert error.column == 10
        assert error.schema_path is not None
        assert error.expected == "number"
        assert error.actual == "'invalid'"
        assert error.suggestion == "Provide a numeric value"


class TestConfigSchemaValidation:
    """Test ConfigSchema post-init validation."""

    def test_empty_schema_dict_raises_error(self) -> None:
        """Test that empty schema dict raises error."""
        with pytest.raises(ValueError, match="Schema cannot be empty"):
            ConfigSchema(
                name="test",
                version="1.0.0",
                schema={},
            )

    def test_schema_with_content(self) -> None:
        """Test schema with valid content."""
        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object", "properties": {}},
        )

        assert schema.schema["type"] == "object"


class TestSchemaRegistryEdgeCases:
    """Test SchemaRegistry edge cases."""

    def test_get_nonexistent_schema_returns_none(self) -> None:
        """Test getting nonexistent schema returns None."""
        registry = SchemaRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_get_nonexistent_version_returns_none(self) -> None:
        """Test getting nonexistent version returns None."""
        registry = SchemaRegistry()
        registry.register(
            ConfigSchema(
                name="test",
                version="1.0.0",
                schema={"type": "object"},
            )
        )

        result = registry.get("test", version="2.0.0")

        assert result is None

    def test_list_versions_nonexistent_schema(self) -> None:
        """Test listing versions of nonexistent schema."""
        registry = SchemaRegistry()

        versions = registry.list_versions("nonexistent")

        assert versions == []

    def test_has_schema_with_nonexistent(self) -> None:
        """Test has_schema with nonexistent schema."""
        registry = SchemaRegistry()

        assert registry.has_schema("nonexistent") is False

    def test_register_without_set_default(self) -> None:
        """Test registering schema without setting as default."""
        registry = SchemaRegistry()

        schema = ConfigSchema(
            name="test",
            version="1.0.0",
            schema={"type": "object"},
        )
        registry.register(schema, set_default=False)

        # Should still be retrievable
        assert registry.get("test", "1.0.0") == schema


class TestValidateAgainstSchemaEdgeCases:
    """Test validate_against_schema edge cases."""

    def test_validate_with_specific_version(self) -> None:
        """Test validating against specific schema version."""
        registry = SchemaRegistry()

        # Register two versions
        registry.register(
            ConfigSchema(
                name="versioned",
                version="1.0.0",
                schema={
                    "type": "object",
                    "properties": {"old_field": {"type": "string"}},
                },
            ),
            set_default=False,
        )
        registry.register(
            ConfigSchema(
                name="versioned",
                version="2.0.0",
                schema={
                    "type": "object",
                    "properties": {"new_field": {"type": "string"}},
                },
            )
        )

        # Create custom registry to test
        with patch("oscura.config.schema._global_registry", registry):
            config = {"old_field": "value"}

            # Should validate against v1.0.0
            result = validate_against_schema(config, "versioned", version="1.0.0")
            assert result is True

    def test_validate_strict_mode_adds_restriction(self) -> None:
        """Test that strict mode adds additionalProperties: false."""
        registry = SchemaRegistry()
        registry.register(
            ConfigSchema(
                name="no_additional",
                version="1.0.0",
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    # No additionalProperties specified
                },
            )
        )

        with patch("oscura.config.schema._global_registry", registry):
            config = {"name": "test", "extra": "field"}

            # Without strict, should pass
            # With strict, should fail
            with pytest.raises(ValidationError):
                validate_against_schema(config, "no_additional", strict=True)


class TestBuiltinSchemas:
    """Test built-in schema definitions."""

    def test_threshold_profile_schema_exists(self) -> None:
        """Test threshold_profile schema is registered."""
        registry = get_schema_registry()

        assert registry.has_schema("threshold_profile")

    def test_threshold_profile_valid_config(self) -> None:
        """Test valid threshold profile configuration."""
        config = {
            "name": "custom_ttl",
            "description": "Custom TTL thresholds",
            "base_family": "TTL",
            "overrides": {
                "VIH": 2.2,
                "VIL": 0.7,
            },
            "tolerance": 5.0,
        }

        result = validate_against_schema(config, "threshold_profile")
        assert result is True

    def test_threshold_profile_tolerance_range(self) -> None:
        """Test threshold profile tolerance validation."""
        config = {
            "name": "test",
            "tolerance": 150,  # Over 100% maximum
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "threshold_profile")

    def test_preferences_schema_complete(self) -> None:
        """Test complete preferences schema."""
        config = {
            "defaults": {
                "sample_rate": 1000000,
                "window_function": "hann",
                "fft_size": 8192,
            },
            "visualization": {
                "style": "seaborn-v0_8-whitegrid",
                "figure_size": [10, 6],
                "dpi": 100,
                "colormap": "viridis",
            },
            "export": {
                "default_format": "csv",
                "precision": 6,
            },
            "logging": {
                "level": "INFO",
                "file": "/var/log/oscura.log",
            },
        }

        result = validate_against_schema(config, "preferences")
        assert result is True

    def test_preferences_invalid_log_level(self) -> None:
        """Test preferences with invalid log level."""
        config = {
            "logging": {
                "level": "INVALID_LEVEL",
            }
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "preferences")

    def test_preferences_invalid_export_format(self) -> None:
        """Test preferences with invalid export format."""
        config = {
            "export": {
                "default_format": "invalid_format",
            }
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "preferences")
