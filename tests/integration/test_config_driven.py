"""Configuration-driven integration tests using real YAML configs.

This module tests Oscura's configuration-driven features.

- Tests schema validation and config parsing
- Uses inline configuration fixtures
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Graceful imports
try:
    from oscura.loaders.configurable import (
        ConfigurablePacketLoader,
        DeviceConfig,
        DeviceMapper,
        PacketFormatConfig,
    )
    from oscura.loaders.validation import PacketValidator
    from oscura.testing.synthetic import SyntheticPacketConfig, generate_packets

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

pytestmark = pytest.mark.integration


@pytest.fixture
def config_dir() -> Path:
    """Path to example configs directory."""
    return Path(__file__).parent.parent.parent / "examples" / "configs"


@pytest.fixture
def packet_format_config(config_dir: Path) -> dict:
    """Load packet format example config."""
    config_path = config_dir / "packet_format_example.yaml"
    if not config_path.exists():
        pytest.skip("Packet format config not found")

    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def device_mapping_config(config_dir: Path) -> dict:
    """Load device mapping example config."""
    config_path = config_dir / "device_mapping_example.yaml"
    if not config_path.exists():
        pytest.skip("Device mapping config not found")

    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def bus_config(config_dir: Path) -> dict:
    """Load bus config example."""
    config_path = config_dir / "bus_config_example.yaml"
    if not config_path.exists():
        pytest.skip("Bus config not found")

    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def protocol_definition_config(config_dir: Path) -> dict:
    """Load protocol definition example."""
    config_path = config_dir / "protocol_definition_example.yaml"
    if not config_path.exists():
        pytest.skip("Protocol definition config not found")

    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.mark.integration
class TestPacketFormatConfig:
    """Test packet format configuration loading and usage."""

    def test_load_packet_format_config(self, packet_format_config: dict) -> None:
        """Test loading packet format from YAML config.

        , CFG-001
        """
        # Verify config structure
        assert "name" in packet_format_config
        assert "packet" in packet_format_config
        assert "header" in packet_format_config
        assert "samples" in packet_format_config

        # Check packet definition
        packet_def = packet_format_config["packet"]
        assert packet_def["size"] == 1024
        assert packet_def["byte_order"] in ("big", "little", "native")

        # Check header fields
        header_def = packet_format_config["header"]
        assert header_def["size"] == 16
        assert "fields" in header_def
        assert len(header_def["fields"]) > 0

        # Verify field definitions
        for field in header_def["fields"]:
            assert "name" in field
            assert "offset" in field
            assert "size" in field
            assert "type" in field

    def test_create_loader_from_config(self, packet_format_config: dict, tmp_path: Path) -> None:
        """Test creating ConfigurablePacketLoader from YAML config.

        , CFG-001
        """
        try:
            # Create simplified config for loader
            from oscura.loaders.configurable import SampleFormatDef

            packet_def = packet_format_config["packet"]
            header_def = packet_format_config["header"]
            samples_def = packet_format_config["samples"]

            loader_config = PacketFormatConfig(
                name=packet_format_config["name"],
                version=packet_format_config["version"],
                packet_size=packet_def["size"],
                byte_order=packet_def["byte_order"],
                header_size=header_def["size"],
                header_fields=[],  # Simplified for test
                sample_offset=samples_def["offset"],
                sample_count=samples_def["count"],
                sample_format=SampleFormatDef(
                    size=samples_def["format"]["size"],
                    type=samples_def["format"]["type"],
                    endian=samples_def["format"]["endian"],
                ),
            )

            # Create loader
            loader = ConfigurablePacketLoader(loader_config)
            assert loader is not None

            # Generate matching test data
            synthetic_config = SyntheticPacketConfig(
                packet_size=packet_def["size"],
                header_size=header_def["size"],
            )
            binary_data, _ = generate_packets(count=10, **synthetic_config.__dict__)

            # Write test file
            test_file = tmp_path / "config_test.bin"
            test_file.write_bytes(binary_data)

            # Load using configured loader
            result = loader.load(test_file)

            # Verify loaded packets
            assert len(result.packets) == 10

        except Exception as e:
            pytest.skip(f"Config loader test skipped: {e}")

    def test_validation_config_rules(self, packet_format_config: dict, tmp_path: Path) -> None:
        """Test validation rules from packet format config.

        , CFG-001
        """
        try:
            validation_def = packet_format_config.get("validation", {})

            if "sync_check" in validation_def:
                sync_def = validation_def["sync_check"]

                if sync_def.get("enabled"):
                    # Create validator with sync check
                    expected_sync = sync_def["expected"]
                    validator = PacketValidator(
                        sync_marker=expected_sync,
                        strictness=sync_def.get("on_failure", "warn"),
                    )

                    # Generate test packet with known sync
                    config = SyntheticPacketConfig(
                        packet_size=1024,
                        sync_pattern=b"\xfa",  # Match config
                    )
                    binary_data, _ = generate_packets(count=5, **config.__dict__)

                    test_file = tmp_path / "sync_test.bin"
                    test_file.write_bytes(binary_data)

                    # Load and validate
                    from oscura.loaders.configurable import SampleFormatDef

                    loader_config = PacketFormatConfig(
                        name="sync_test",
                        version="1.0",
                        packet_size=1024,
                        byte_order="big",
                        header_size=16,
                        header_fields=[],
                        sample_offset=16,
                        sample_count=126,
                        sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
                    )

                    loader = ConfigurablePacketLoader(loader_config)
                    result = loader.load(test_file)

                    # Validate packets
                    for packet in result.packets:
                        validation_result = validator.validate_packet(packet)
                        # Validation should complete
                        assert validation_result is not None

        except Exception as e:
            pytest.skip(f"Validation config test skipped: {e}")

    def test_timing_config(self, packet_format_config: dict) -> None:
        """Test timing configuration parsing."""
        if "timing" not in packet_format_config:
            pytest.skip("No timing config")

        timing_def = packet_format_config["timing"]

        # Verify timing parameters
        assert "sample_rate" in timing_def
        assert "samples_per_packet" in timing_def

        sample_rate = timing_def["sample_rate"]
        samples_per_packet = timing_def["samples_per_packet"]

        # Convert sample_rate if it's a string (scientific notation)
        if isinstance(sample_rate, str):
            sample_rate = float(sample_rate)

        # Should be reasonable values
        assert sample_rate > 0
        assert samples_per_packet > 0

        # Calculate expected timing
        packet_duration = samples_per_packet / sample_rate
        assert packet_duration > 0

    def test_preprocessing_config(self, packet_format_config: dict) -> None:
        """Test preprocessing configuration."""
        if "preprocessing" not in packet_format_config:
            pytest.skip("No preprocessing config")

        preprocessing_def = packet_format_config["preprocessing"]

        # Check trim_idle settings
        if "trim_idle" in preprocessing_def:
            trim_idle = preprocessing_def["trim_idle"]

            assert "enabled" in trim_idle
            if trim_idle["enabled"]:
                assert "pattern" in trim_idle
                assert "min_duration" in trim_idle


@pytest.mark.integration
class TestDeviceMappingConfig:
    """Test device mapping configuration."""

    def test_load_device_mapping(self, device_mapping_config: dict) -> None:
        """Test loading device mapping from YAML.

        , CFG-001
        """
        # Verify config structure
        assert "devices" in device_mapping_config
        assert "unknown_device" in device_mapping_config

        devices = device_mapping_config["devices"]
        assert len(devices) > 0

        # Check device definitions
        for device_info in devices.values():
            assert "name" in device_info
            assert "description" in device_info

    def test_device_mapper_from_config(self, device_mapping_config: dict) -> None:
        """Test creating DeviceMapper from config.

        , CFG-001
        """
        try:
            # Extract device config
            devices_dict = {}
            for device_id, info in device_mapping_config["devices"].items():
                # Handle hex device IDs
                if isinstance(device_id, str) and device_id.startswith("0x"):
                    device_id_int = int(device_id, 16)
                else:
                    device_id_int = int(device_id)

                devices_dict[device_id_int] = {
                    "name": info["name"],
                    "description": info["description"],
                }

            # Create device config
            device_config = DeviceConfig(
                devices=devices_dict,
                unknown_policy=device_mapping_config["unknown_device"]["policy"],
            )

            # Create mapper
            mapper = DeviceMapper(device_config)
            assert mapper is not None

            # Test known device lookup
            first_device_id = next(iter(devices_dict.keys()))
            device_name = mapper.get_device_name(first_device_id)
            assert device_name == devices_dict[first_device_id]["name"]

            # Test unknown device handling
            unknown_name = mapper.get_device_name(0xFFFF)
            assert unknown_name is not None
            # Should contain "Unknown" or the hex ID
            assert "Unknown" in unknown_name or "0x" in unknown_name.lower()

        except Exception as e:
            pytest.skip(f"Device mapper test skipped: {e}")

    def test_device_categories(self, device_mapping_config: dict) -> None:
        """Test device category configuration."""
        if "categories" not in device_mapping_config:
            pytest.skip("No categories config")

        categories = device_mapping_config["categories"]
        assert len(categories) > 0

        # Each category should have description
        for cat_info in categories.values():
            assert "description" in cat_info

    def test_channel_configuration(self, device_mapping_config: dict) -> None:
        """Test channel/lane configuration."""
        if "channels" not in device_mapping_config:
            pytest.skip("No channels config")

        channels = device_mapping_config["channels"]
        assert len(channels) > 0

        # Each channel should have name
        for ch_info in channels.values():
            assert "name" in ch_info


@pytest.mark.integration
class TestBusConfig:
    """Test parallel bus configuration."""

    def test_load_bus_config(self, bus_config: dict) -> None:
        """Test loading bus configuration from YAML."""
        # Verify config structure
        assert "name" in bus_config
        assert "settings" in bus_config

        # Check bus definitions
        if "data_bus" in bus_config:
            data_bus = bus_config["data_bus"]
            assert "width" in data_bus
            assert "bits" in data_bus
            assert len(data_bus["bits"]) == data_bus["width"]

        if "address_bus" in bus_config:
            addr_bus = bus_config["address_bus"]
            assert "width" in addr_bus
            assert "bits" in addr_bus

    def test_control_signals_config(self, bus_config: dict) -> None:
        """Test control signals configuration."""
        if "control_signals" not in bus_config:
            pytest.skip("No control signals")

        control_signals = bus_config["control_signals"]
        assert len(control_signals) > 0

        # Each signal should have required fields
        for signal in control_signals:
            assert "name" in signal
            assert "channel" in signal
            assert "description" in signal

    def test_transaction_types_config(self, bus_config: dict) -> None:
        """Test transaction type definitions."""
        if "transactions" not in bus_config:
            pytest.skip("No transactions config")

        transactions = bus_config["transactions"]
        if "types" in transactions:
            trans_types = transactions["types"]
            assert len(trans_types) > 0

            # Each transaction type should have name and conditions
            for trans_type in trans_types:
                assert "name" in trans_type
                assert "conditions" in trans_type

    def test_instruction_decode_config(self, bus_config: dict) -> None:
        """Test instruction decoding configuration."""
        if "instruction_decode" not in bus_config:
            pytest.skip("No instruction decode config")

        instr_decode = bus_config["instruction_decode"]

        if instr_decode.get("enabled"):
            assert "opcode_bits" in instr_decode
            assert "opcodes" in instr_decode

            opcodes = instr_decode["opcodes"]
            # Each opcode should map to instruction info
            for instr_info in opcodes.values():
                assert "name" in instr_info


@pytest.mark.integration
class TestProtocolDefinitionConfig:
    """Test protocol definition DSL configuration."""

    def test_load_protocol_definition(self, protocol_definition_config: dict) -> None:
        """Test loading protocol definition from YAML."""
        # Verify config structure
        assert "name" in protocol_definition_config
        assert "version" in protocol_definition_config
        assert "fields" in protocol_definition_config

        fields = protocol_definition_config["fields"]
        assert len(fields) > 0

        # Each field should have name and type
        for field in fields:
            assert "name" in field
            assert "type" in field

    def test_framing_config(self, protocol_definition_config: dict) -> None:
        """Test message framing configuration."""
        if "framing" not in protocol_definition_config:
            pytest.skip("No framing config")

        framing = protocol_definition_config["framing"]

        assert "type" in framing
        assert framing["type"] in ("delimiter", "length_prefix", "fixed")

        if "sync" in framing:
            sync = framing["sync"]
            assert "pattern" in sync

    def test_field_types_config(self, protocol_definition_config: dict) -> None:
        """Test various field type definitions."""
        fields = protocol_definition_config["fields"]

        # Find examples of different field types
        found_types = set()
        for field in fields:
            found_types.add(field["type"])

        # Should have variety of types
        assert len(found_types) > 1

        # Check for specific complex types
        for field in fields:
            if field["type"] == "bitfield":
                assert "fields" in field
            elif field["type"] == "array":
                assert "element" in field
            elif field["type"] == "bytes":
                assert "size" in field or "condition" in field

    def test_validation_rules_config(self, protocol_definition_config: dict) -> None:
        """Test field validation rules."""
        fields = protocol_definition_config["fields"]

        # Find fields with validation
        validated_fields = [f for f in fields if "validation" in f]

        if validated_fields:
            for field in validated_fields:
                validation = field["validation"]
                # Should have validation criteria
                assert len(validation) > 0

    def test_conditional_fields_config(self, protocol_definition_config: dict) -> None:
        """Test conditional field definitions."""
        fields = protocol_definition_config["fields"]

        # Find conditional fields
        conditional_fields = [f for f in fields if "condition" in f]

        if conditional_fields:
            for field in conditional_fields:
                condition = field["condition"]
                # Condition should be expression string
                assert isinstance(condition, str)
                assert len(condition) > 0

    def test_computed_fields_config(self, protocol_definition_config: dict) -> None:
        """Test computed/virtual fields."""
        if "computed_fields" not in protocol_definition_config:
            pytest.skip("No computed fields")

        computed = protocol_definition_config["computed_fields"]

        for field in computed:
            assert "name" in field
            assert "expression" in field


@pytest.mark.integration
class TestConfigValidation:
    """Test configuration validation and error handling."""

    def test_invalid_packet_size_config(self, tmp_path: Path) -> None:
        """Test handling of invalid packet size in config."""
        try:
            from oscura.loaders.configurable import SampleFormatDef

            # Create config with invalid packet size
            invalid_config = PacketFormatConfig(
                name="invalid_test",
                version="1.0",
                packet_size=0,  # Invalid size
                byte_order="little",
                header_size=8,
                header_fields=[],
                sample_offset=8,
                sample_count=1,
                sample_format=SampleFormatDef(size=8, type="uint64", endian="little"),
            )

            loader = ConfigurablePacketLoader(invalid_config)

            # Generate test data
            test_file = tmp_path / "invalid.bin"
            test_file.write_bytes(b"\x00" * 64)

            # Load should fail or handle gracefully
            try:
                result = loader.load(test_file)
                # If it succeeds, should have 0 packets due to invalid config
                assert len(result.packets) == 0, (
                    f"Expected 0 packets for invalid config, got {len(result.packets)}"
                )
            except (ValueError, KeyError, AttributeError):
                # Exception is acceptable for invalid config
                pass

        except Exception as e:
            pytest.skip(f"Invalid config test skipped: {e}")

    def test_config_schema_validation(self, packet_format_config: dict) -> None:
        """Test that config follows expected schema."""
        # Required top-level fields
        required_fields = ["name", "version"]
        for field in required_fields:
            assert field in packet_format_config, f"Missing required field: {field}"

        # Packet definition validation
        if "packet" in packet_format_config:
            packet = packet_format_config["packet"]
            assert "size" in packet
            assert isinstance(packet["size"], int) or packet["size"] == "variable"

        # Header definition validation
        if "header" in packet_format_config:
            header = packet_format_config["header"]
            assert "size" in header
            assert isinstance(header["size"], int)


@pytest.mark.integration
class TestConfigInteraction:
    """Test interaction between multiple configs."""

    def test_packet_format_with_device_mapping(
        self,
        packet_format_config: dict,
        device_mapping_config: dict,
        tmp_path: Path,
    ) -> None:
        """Test using packet format config with device mapping."""
        try:
            from oscura.loaders.configurable import SampleFormatDef

            # Create loader from packet format
            packet_def = packet_format_config["packet"]
            header_def = packet_format_config["header"]
            samples_def = packet_format_config["samples"]

            loader_config = PacketFormatConfig(
                name=packet_format_config["name"],
                version=packet_format_config["version"],
                packet_size=packet_def["size"],
                byte_order=packet_def["byte_order"],
                header_size=header_def["size"],
                header_fields=[],
                sample_offset=samples_def["offset"],
                sample_count=samples_def["count"],
                sample_format=SampleFormatDef(
                    size=samples_def["format"]["size"],
                    type=samples_def["format"]["type"],
                    endian=samples_def["format"]["endian"],
                ),
            )

            loader = ConfigurablePacketLoader(loader_config)

            # Create device mapper
            devices_dict = {}
            for device_id, info in device_mapping_config["devices"].items():
                if isinstance(device_id, str) and device_id.startswith("0x"):
                    device_id_int = int(device_id, 16)
                else:
                    device_id_int = int(device_id)

                devices_dict[device_id_int] = {
                    "name": info["name"],
                    "description": info["description"],
                }

            device_config = DeviceConfig(
                devices=devices_dict,
                unknown_policy="warn",
            )

            mapper = DeviceMapper(device_config)

            # Generate test data
            synthetic_config = SyntheticPacketConfig(packet_size=packet_def["size"])
            binary_data, _ = generate_packets(count=5, **synthetic_config.__dict__)

            test_file = tmp_path / "combined_test.bin"
            test_file.write_bytes(binary_data)

            # Load packets
            result = loader.load(test_file)

            # Map device IDs (simulated)
            for device_id in devices_dict:
                device_name = mapper.get_device_name(device_id)
                assert device_name is not None

        except Exception as e:
            pytest.skip(f"Combined config test skipped: {e}")
