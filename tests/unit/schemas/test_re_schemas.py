"""Unit tests for configuration schema validation.

Tests JSON Schema validation for all configuration types including
packet formats, device mappings, bus configurations, and protocol
definitions.


Requirements tested:
"""

from __future__ import annotations

from typing import Any

import pytest

from oscura.config.schema import ValidationError
from oscura.schemas import (
    SCHEMA_NAMES,
    get_schema_path,
    load_schema,
    validate_config,
)

pytestmark = pytest.mark.unit


class TestSchemaLoading:
    """Test schema file loading and registration."""

    def test_schema_names_defined(self) -> None:
        """Test that all expected schema names are defined."""
        expected = [
            "packet_format",
            "device_mapping",
            "bus_configuration",
            "protocol_definition",
        ]
        assert expected == SCHEMA_NAMES

    def test_get_schema_path_valid(self) -> None:
        """Test getting schema path for valid schema name."""
        for schema_name in SCHEMA_NAMES:
            path = get_schema_path(schema_name)
            assert path.exists()
            assert path.suffix == ".json"
            assert schema_name in path.name

    def test_get_schema_path_invalid(self) -> None:
        """Test getting schema path for invalid schema name."""
        with pytest.raises(ValueError, match="Unknown schema name"):
            get_schema_path("nonexistent_schema")

    def test_load_schema_valid(self) -> None:
        """Test loading valid schema files."""
        for schema_name in SCHEMA_NAMES:
            schema = load_schema(schema_name)
            assert isinstance(schema, dict)
            assert "$schema" in schema
            assert "type" in schema
            assert schema["type"] == "object"

    def test_load_schema_has_metadata(self) -> None:
        """Test that schemas have proper metadata."""
        for schema_name in SCHEMA_NAMES:
            schema = load_schema(schema_name)
            assert "title" in schema
            assert "description" in schema
            assert "$id" in schema
            # Check that description mentions CFG-001
            assert "CFG-001" in schema["description"]


class TestPacketFormatSchema:
    """Test packet format schema validation."""

    def test_missing_required_name(self) -> None:
        """Test error on missing required 'name' field."""
        config = {
            "packet": {"byte_order": "big"},
            "header": {"size": 16, "fields": []},
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError, match=r"name.*required"):
            validate_config(config, "packet_format")

    def test_missing_required_packet(self) -> None:
        """Test error on missing required 'packet' field."""
        config = {
            "name": "test",
            "header": {"size": 16, "fields": []},
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "packet_format")

    def test_invalid_name_pattern(self) -> None:
        """Test error on invalid name pattern."""
        config = {
            "name": "123-invalid",  # Must start with letter
            "packet": {"byte_order": "big"},
            "header": {"size": 16, "fields": []},
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "packet_format")

    def test_invalid_byte_order(self) -> None:
        """Test error on invalid byte order."""
        config = {
            "name": "test",
            "packet": {"byte_order": "invalid"},
            "header": {"size": 16, "fields": []},
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "packet_format")

    def test_invalid_field_type(self) -> None:
        """Test error on invalid field type."""
        config = {
            "name": "test",
            "packet": {"byte_order": "big"},
            "header": {
                "size": 16,
                "fields": [
                    {
                        "name": "field1",
                        "offset": 0,
                        "size": 4,
                        "type": "invalid_type",
                    }
                ],
            },
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "packet_format")

    def test_negative_offset(self) -> None:
        """Test error on negative offset."""
        config = {
            "name": "test",
            "packet": {"byte_order": "big"},
            "header": {
                "size": 16,
                "fields": [{"name": "field1", "offset": -1, "size": 4, "type": "uint32"}],
            },
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "packet_format")

    def test_variable_packet_size(self) -> None:
        """Test variable packet size is accepted."""
        config = {
            "name": "test",
            "packet": {"size": "variable", "byte_order": "big"},
            "header": {"size": 16, "fields": []},
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        result = validate_config(config, "packet_format")
        assert result is True

    def test_bitfield_with_single_bit(self) -> None:
        """Test bitfield field with single bit extraction."""
        config = {
            "name": "test",
            "packet": {"byte_order": "big"},
            "header": {
                "size": 16,
                "fields": [
                    {
                        "name": "flags",
                        "offset": 0,
                        "size": 1,
                        "type": "bitfield",
                        "fields": {"overflow": {"bit": 7, "description": "Overflow"}},
                    }
                ],
            },
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        result = validate_config(config, "packet_format")
        assert result is True

    def test_bitfield_with_bit_range(self) -> None:
        """Test bitfield field with bit range extraction."""
        config = {
            "name": "test",
            "packet": {"byte_order": "big"},
            "header": {
                "size": 16,
                "fields": [
                    {
                        "name": "flags",
                        "offset": 0,
                        "size": 1,
                        "type": "bitfield",
                        "fields": {"channel": {"bits": [0, 3], "description": "Channel ID"}},
                    }
                ],
            },
            "samples": {"offset": 16, "format": {"size": 8, "type": "uint64"}},
        }
        result = validate_config(config, "packet_format")
        assert result is True


class TestDeviceMappingSchema:
    """Test device mapping schema validation."""

    def test_missing_required_name(self) -> None:
        """Test error on missing required 'name' field."""
        config = {"devices": {"0x01": {"name": "Device1"}}}
        with pytest.raises(ValidationError, match=r"name.*required"):
            validate_config(config, "device_mapping")

    def test_missing_required_devices(self) -> None:
        """Test error on missing required 'devices' field."""
        config = {"name": "test"}
        with pytest.raises(ValidationError):
            validate_config(config, "device_mapping")

    def test_empty_devices(self) -> None:
        """Test error on empty devices object."""
        config = {"name": "test", "devices": {}}
        with pytest.raises(ValidationError):
            validate_config(config, "device_mapping")

    def test_hex_device_id(self) -> None:
        """Test device with hex ID."""
        config = {
            "name": "test",
            "devices": {"0xFF": {"name": "Test Device"}},
        }
        result = validate_config(config, "device_mapping")
        assert result is True

    def test_decimal_device_id(self) -> None:
        """Test device with decimal ID."""
        config = {
            "name": "test",
            "devices": {"255": {"name": "Test Device"}},
        }
        result = validate_config(config, "device_mapping")
        assert result is True

    def test_string_device_id(self) -> None:
        """Test device with string ID (allowed for YAML compatibility)."""
        config = {
            "name": "test",
            "devices": {"device_a": {"name": "Test Device"}},
        }
        # String IDs are allowed due to YAML parsing of numeric keys as integers
        result = validate_config(config, "device_mapping")
        assert result is True

    def test_device_missing_name(self) -> None:
        """Test error on device missing required name."""
        config = {
            "name": "test",
            "devices": {"0x01": {"description": "No name"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "device_mapping")

    def test_invalid_sample_rate(self) -> None:
        """Test error on invalid sample rate (negative)."""
        config = {
            "name": "test",
            "devices": {"0x01": {"name": "Device", "sample_rate": -1000}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "device_mapping")

    def test_category_with_color(self) -> None:
        """Test category definition with color."""
        config = {
            "name": "test",
            "devices": {"0x01": {"name": "Device"}},
            "categories": {"control": {"description": "Control devices", "color": "#00FF00"}},
        }
        result = validate_config(config, "device_mapping")
        assert result is True

    def test_invalid_color_format(self) -> None:
        """Test error on invalid color format."""
        config = {
            "name": "test",
            "devices": {"0x01": {"name": "Device"}},
            "categories": {"control": {"color": "invalid"}},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "device_mapping")

    def test_unknown_device_policy(self) -> None:
        """Test unknown device policy options."""
        for policy in ["error", "warn", "ignore"]:
            config = {
                "name": "test",
                "devices": {"0x01": {"name": "Device"}},
                "unknown_device": {"policy": policy},
            }
            result = validate_config(config, "device_mapping")
            assert result is True

    def test_filters_with_hex_ids(self) -> None:
        """Test filters with hex device IDs."""
        config = {
            "name": "test",
            "devices": {"0x01": {"name": "Device"}},
            "filters": {"enabled": True, "include_devices": ["0x01", "0xFF"]},
        }
        result = validate_config(config, "device_mapping")
        assert result is True


class TestBusConfigurationSchema:
    """Test bus configuration schema validation."""

    def test_missing_required_name(self) -> None:
        """Test error on missing required 'name' field."""
        config: dict[str, Any] = {"settings": {}}
        with pytest.raises(ValidationError):
            validate_config(config, "bus_configuration")

    def test_missing_required_settings(self) -> None:
        """Test error on missing required 'settings' field."""
        config = {"name": "test"}
        with pytest.raises(ValidationError):
            validate_config(config, "bus_configuration")

    def test_invalid_sample_on(self) -> None:
        """Test error on invalid sample_on value."""
        config = {
            "name": "test",
            "settings": {"sample_on": "invalid"},
        }
        with pytest.raises(ValidationError):
            validate_config(config, "bus_configuration")

    def test_valid_sample_on_values(self) -> None:
        """Test valid sample_on values."""
        for value in ["rising", "falling", "both"]:
            config = {
                "name": "test",
                "settings": {"sample_on": value},
            }
            result = validate_config(config, "bus_configuration")
            assert result is True

    def test_data_bus_definition(self) -> None:
        """Test data bus definition."""
        config = {
            "name": "test",
            "settings": {},
            "data_bus": {
                "name": "data",
                "width": 8,
                "bits": [{"channel": 0, "bit": 0, "name": "D0"}],
            },
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_bus_width_limits(self) -> None:
        """Test bus width validation."""
        # Valid widths
        for width in [1, 8, 16, 32, 64, 128]:
            config = {
                "name": "test",
                "settings": {},
                "data_bus": {
                    "name": "data",
                    "width": width,
                    "bits": [{"channel": 0, "bit": 0}],
                },
            }
            result = validate_config(config, "bus_configuration")
            assert result is True

        # Invalid width (too large)
        config = {
            "name": "test",
            "settings": {},
            "data_bus": {
                "name": "data",
                "width": 256,
                "bits": [{"channel": 0, "bit": 0}],
            },
        }
        with pytest.raises(ValidationError):
            validate_config(config, "bus_configuration")

    def test_control_signals(self) -> None:
        """Test control signal definitions."""
        config = {
            "name": "test",
            "settings": {},
            "control_signals": [
                {
                    "name": "read",
                    "channel": 24,
                    "active_low": True,
                    "description": "Read strobe",
                    "short_name": "RD",
                }
            ],
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_timing_constraints(self) -> None:
        """Test timing constraint definitions."""
        config = {
            "name": "test",
            "settings": {},
            "timing": {
                "clock_period_ns": 1000,
                "setup_time_ns": 50,
                "hold_time_ns": 30,
                "pulse_width_ns": {"read_strobe": 200},
            },
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_transaction_types(self) -> None:
        """Test transaction type definitions."""
        config = {
            "name": "test",
            "settings": {},
            "transactions": {
                "types": [
                    {
                        "name": "memory_read",
                        "conditions": {"read": True, "write": False},
                        "sample_on": "timing_pulse_3",
                        "capture": {"address": "address_bus", "data": "data_bus"},
                    }
                ]
            },
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_instruction_decode_binary(self) -> None:
        """Test instruction decode with binary opcodes."""
        config = {
            "name": "test",
            "settings": {},
            "instruction_decode": {
                "enabled": True,
                "opcode_bits": [9, 11],
                "opcodes": {"0b000": {"name": "AND", "description": "Logical AND"}},
            },
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_instruction_decode_hex(self) -> None:
        """Test instruction decode with hex opcodes."""
        config = {
            "name": "test",
            "settings": {},
            "instruction_decode": {
                "enabled": True,
                "opcode_bits": [0, 7],
                "opcodes": {"0xFF": {"name": "NOP", "description": "No operation"}},
            },
        }
        result = validate_config(config, "bus_configuration")
        assert result is True


class TestProtocolDefinitionSchema:
    """Test protocol definition schema validation."""

    def test_missing_required_fields(self) -> None:
        """Test error on missing required fields."""
        # Missing name
        with pytest.raises(ValidationError):
            validate_config(
                {
                    "settings": {},
                    "framing": {"type": "fixed"},
                    "fields": [{"name": "f1", "type": "uint8"}],
                },
                "protocol_definition",
            )

        # Missing settings
        with pytest.raises(ValidationError):
            validate_config(
                {
                    "name": "test",
                    "framing": {"type": "fixed"},
                    "fields": [{"name": "f1", "type": "uint8"}],
                },
                "protocol_definition",
            )

        # Missing framing
        with pytest.raises(ValidationError):
            validate_config(
                {
                    "name": "test",
                    "settings": {},
                    "fields": [{"name": "f1", "type": "uint8"}],
                },
                "protocol_definition",
            )

        # Missing fields
        with pytest.raises(ValidationError):
            validate_config(
                {"name": "test", "settings": {}, "framing": {"type": "fixed"}},
                "protocol_definition",
            )

    def test_framing_types(self) -> None:
        """Test all framing types."""
        for framing_type in ["delimiter", "length_prefix", "fixed"]:
            config = {
                "name": "test",
                "settings": {},
                "framing": {"type": framing_type},
                "fields": [{"name": "f1", "type": "uint8"}],
            }
            result = validate_config(config, "protocol_definition")
            assert result is True

    def test_sync_pattern(self) -> None:
        """Test sync pattern definition."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {
                "type": "length_prefix",
                "sync": {"pattern": [0xAA, 0x55], "required": True},
            },
            "fields": [{"name": "f1", "type": "uint8"}],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_length_field(self) -> None:
        """Test length field specification."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {
                "type": "length_prefix",
                "length_field": {
                    "offset": 2,
                    "size": 2,
                    "endian": "big",
                    "includes_header": False,
                },
            },
            "fields": [{"name": "f1", "type": "uint8"}],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_field_types(self) -> None:
        """Test all supported field types."""
        types = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
            "bytes",
            "string",
            "array",
            "struct",
            "bitfield",
        ]
        for field_type in types:
            config = {
                "name": "test",
                "settings": {},
                "framing": {"type": "fixed"},
                "fields": [{"name": "f1", "type": field_type}],
            }
            result = validate_config(config, "protocol_definition")
            assert result is True

    def test_field_with_condition(self) -> None:
        """Test field with conditional expression."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {"name": "msg_type", "type": "uint8"},
                {
                    "name": "data",
                    "type": "uint16",
                    "condition": "msg_type == 0x01",
                },
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_field_with_enum(self) -> None:
        """Test field with enumeration values."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "msg_type",
                    "type": "uint8",
                    "enum": {
                        "0x01": {"name": "data", "description": "Data message"},
                        "0x02": {"name": "status", "description": "Status message"},
                    },
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_bitfield_field(self) -> None:
        """Test bitfield field with extraction."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "flags",
                    "type": "bitfield",
                    "size": 1,
                    "fields": {
                        "priority": {"bits": [6, 7], "description": "Priority"},
                        "encrypted": {"bit": 5, "description": "Encrypted"},
                    },
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_array_field(self) -> None:
        """Test array field with element definition."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "readings",
                    "type": "array",
                    "element": {"type": "float32"},
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_struct_field(self) -> None:
        """Test struct field with nested fields."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "header",
                    "type": "struct",
                    "fields": [
                        {"name": "version", "type": "uint8"},
                        {"name": "length", "type": "uint16"},
                    ],
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_string_field_encoding(self) -> None:
        """Test string field with encoding."""
        for encoding in ["utf-8", "ascii", "latin-1", "utf-16", "utf-32"]:
            config = {
                "name": "test",
                "settings": {},
                "framing": {"type": "fixed"},
                "fields": [{"name": "message", "type": "string", "encoding": encoding}],
            }
            result = validate_config(config, "protocol_definition")
            assert result is True

    def test_field_validation_rules(self) -> None:
        """Test field validation specifications."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "value",
                    "type": "uint16",
                    "validation": {"min": 0, "max": 100, "on_mismatch": "warn"},
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_checksum_field(self) -> None:
        """Test checksum field with algorithm."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {
                    "name": "checksum",
                    "type": "uint16",
                    "validation": {
                        "algorithm": "crc16_ccitt",
                        "scope": "all_prior",
                        "on_mismatch": "error",
                    },
                }
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_computed_fields(self) -> None:
        """Test computed/virtual fields."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [{"name": "flags", "type": "uint8"}],
            "computed_fields": [
                {"name": "priority", "expression": "flags >> 6", "description": "Priority level"}
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_decoding_hints(self) -> None:
        """Test decoding configuration."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [{"name": "f1", "type": "uint8"}],
            "decoding": {
                "min_header_size": 10,
                "max_message_size": 65535,
                "resync_on_error": True,
                "max_resync_distance": 1024,
            },
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_encoding_rules(self) -> None:
        """Test encoding configuration."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [{"name": "f1", "type": "uint8"}],
            "encoding": {
                "auto_fill": {"sync": True, "length": True},
                "validate_required_fields": True,
                "validate_field_ranges": True,
            },
        }
        result = validate_config(config, "protocol_definition")
        assert result is True


class TestSchemasReSchemasEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_packet_format(self) -> None:
        """Test minimal valid packet format config."""
        config = {
            "name": "minimal",
            "packet": {"byte_order": "big"},
            "header": {"size": 0, "fields": []},
            "samples": {
                "offset": 0,
                "format": {"size": 1, "type": "uint8"},
            },
        }
        result = validate_config(config, "packet_format")
        assert result is True

    def test_minimal_device_mapping(self) -> None:
        """Test minimal valid device mapping config."""
        config = {
            "name": "minimal",
            "devices": {"0": {"name": "Device"}},
        }
        result = validate_config(config, "device_mapping")
        assert result is True

    def test_minimal_bus_configuration(self) -> None:
        """Test minimal valid bus config."""
        config = {
            "name": "minimal",
            "settings": {},
        }
        result = validate_config(config, "bus_configuration")
        assert result is True

    def test_minimal_protocol_definition(self) -> None:
        """Test minimal valid protocol definition."""
        config = {
            "name": "minimal",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [{"name": "f1", "type": "uint8"}],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True

    def test_empty_fields_array_invalid(self) -> None:
        """Test that empty fields array is invalid."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [],  # Empty array
        }
        with pytest.raises(ValidationError):
            validate_config(config, "protocol_definition")

    def test_size_expression_string(self) -> None:
        """Test size field accepts string expressions."""
        config = {
            "name": "test",
            "settings": {},
            "framing": {"type": "fixed"},
            "fields": [
                {"name": "length", "type": "uint16"},
                {"name": "data", "type": "bytes", "size": "length - 10"},
            ],
        }
        result = validate_config(config, "protocol_definition")
        assert result is True
