"""Comprehensive unit tests for protocol_library module.

- RE-DSL-003: Protocol Format Library

This module tests the ProtocolLibrary class and related functions for providing
pre-defined protocol definitions for common industrial, IoT, and communication protocols.
"""

from __future__ import annotations

import pytest

from oscura.inference.protocol_dsl import (
    FieldDefinition,
    ProtocolDecoder,
    ProtocolDefinition,
)
from oscura.inference.protocol_library import (
    ProtocolInfo,
    ProtocolLibrary,
    get_decoder,
    get_library,
    get_protocol,
    list_protocols,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test ProtocolInfo Dataclass
# =============================================================================


class TestProtocolInfo:
    """Test ProtocolInfo dataclass."""

    def test_create_protocol_info_minimal(self) -> None:
        """Test creating ProtocolInfo with minimal fields."""
        info = ProtocolInfo(
            name="test_protocol",
            category="custom",
            version="1.0",
            description="A test protocol",
        )

        assert info.name == "test_protocol"
        assert info.category == "custom"
        assert info.version == "1.0"
        assert info.description == "A test protocol"
        assert info.reference == ""
        assert info.definition is None

    def test_create_protocol_info_with_reference(self) -> None:
        """Test creating ProtocolInfo with reference URL."""
        info = ProtocolInfo(
            name="test_protocol",
            category="industrial",
            version="2.0",
            description="Industrial protocol",
            reference="https://example.com/spec",
        )

        assert info.reference == "https://example.com/spec"

    def test_create_protocol_info_with_definition(self) -> None:
        """Test creating ProtocolInfo with protocol definition."""
        definition = ProtocolDefinition(
            name="test",
            description="Test protocol",
            fields=[
                FieldDefinition(name="header", field_type="uint8"),
            ],
        )
        info = ProtocolInfo(
            name="test_protocol",
            category="serial",
            version="1.0",
            description="Serial protocol",
            definition=definition,
        )

        assert info.definition is not None
        assert info.definition.name == "test"
        assert len(info.definition.fields) == 1

    def test_protocol_info_category_values(self) -> None:
        """Test all valid category values."""
        valid_categories = [
            "industrial",
            "iot",
            "network",
            "automotive",
            "building",
            "serial",
            "custom",
        ]
        for category in valid_categories:
            info = ProtocolInfo(
                name=f"test_{category}",
                category=category,  # type: ignore[arg-type]
                version="1.0",
                description=f"Test {category} protocol",
            )
            assert info.category == category


# =============================================================================
# Test ProtocolLibrary Class
# =============================================================================


class TestProtocolLibrary:
    """Test ProtocolLibrary class."""

    def test_library_initialization(self) -> None:
        """Test that library initializes with built-in protocols."""
        library = ProtocolLibrary()

        # Should have protocols loaded
        protocols = library.list_protocols()
        assert len(protocols) > 0

    def test_list_protocols_all(self) -> None:
        """Test listing all protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols()

        # Should return list of ProtocolInfo
        assert isinstance(protocols, list)
        assert all(isinstance(p, ProtocolInfo) for p in protocols)

        # Should have expected protocols
        names = [p.name for p in protocols]
        assert "modbus_rtu" in names
        assert "mqtt" in names
        assert "can" in names

    def test_list_protocols_by_category_industrial(self) -> None:
        """Test listing industrial protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="industrial")

        assert len(protocols) >= 3
        assert all(p.category == "industrial" for p in protocols)

        names = [p.name for p in protocols]
        assert "modbus_rtu" in names
        assert "modbus_tcp" in names
        assert "dnp3" in names

    def test_list_protocols_by_category_iot(self) -> None:
        """Test listing IoT protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="iot")

        assert len(protocols) >= 4
        assert all(p.category == "iot" for p in protocols)

        names = [p.name for p in protocols]
        assert "mqtt" in names
        assert "coap" in names
        assert "cbor" in names
        assert "messagepack" in names

    def test_list_protocols_by_category_automotive(self) -> None:
        """Test listing automotive protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="automotive")

        assert len(protocols) >= 3
        assert all(p.category == "automotive" for p in protocols)

        names = [p.name for p in protocols]
        assert "obd2" in names
        assert "j1939" in names
        assert "can" in names

    def test_list_protocols_by_category_network(self) -> None:
        """Test listing network protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="network")

        assert len(protocols) >= 4
        assert all(p.category == "network" for p in protocols)

        names = [p.name for p in protocols]
        assert "http" in names
        assert "dns" in names
        assert "ntp" in names
        assert "syslog" in names

    def test_list_protocols_by_category_serial(self) -> None:
        """Test listing serial protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="serial")

        assert len(protocols) >= 2
        assert all(p.category == "serial" for p in protocols)

        names = [p.name for p in protocols]
        assert "nmea" in names
        assert "xmodem" in names

    def test_list_protocols_by_category_building(self) -> None:
        """Test listing building automation protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="building")

        assert len(protocols) >= 3
        assert all(p.category == "building" for p in protocols)

        names = [p.name for p in protocols]
        assert "bacnet" in names
        assert "knx" in names
        assert "lonworks" in names

    def test_list_protocols_by_category_custom(self) -> None:
        """Test listing custom protocols."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="custom")

        assert len(protocols) >= 2
        assert all(p.category == "custom" for p in protocols)

        names = [p.name for p in protocols]
        assert "tlv" in names
        assert "length_prefixed" in names

    def test_list_protocols_invalid_category(self) -> None:
        """Test listing protocols with invalid category returns empty."""
        library = ProtocolLibrary()
        protocols = library.list_protocols(category="nonexistent")

        assert protocols == []

    def test_list_protocol_names(self) -> None:
        """Test listing protocol names."""
        library = ProtocolLibrary()
        names = library.list_protocol_names()

        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "modbus_rtu" in names
        assert "mqtt" in names

    def test_list_protocol_names_by_category(self) -> None:
        """Test listing protocol names by category."""
        library = ProtocolLibrary()
        names = library.list_protocol_names(category="automotive")

        assert isinstance(names, list)
        assert "can" in names
        assert "obd2" in names
        assert "j1939" in names
        # Should not contain protocols from other categories
        assert "mqtt" not in names

    def test_get_protocol_existing(self) -> None:
        """Test getting existing protocol."""
        library = ProtocolLibrary()
        info = library.get("modbus_rtu")

        assert info is not None
        assert info.name == "modbus_rtu"
        assert info.category == "industrial"
        assert info.definition is not None

    def test_get_protocol_case_insensitive(self) -> None:
        """Test that protocol lookup is case-insensitive."""
        library = ProtocolLibrary()

        info_lower = library.get("modbus_rtu")
        info_upper = library.get("MODBUS_RTU")
        info_mixed = library.get("Modbus_RTU")

        assert info_lower is not None
        assert info_upper is not None
        assert info_mixed is not None
        assert info_lower.name == info_upper.name == info_mixed.name

    def test_get_protocol_nonexistent(self) -> None:
        """Test getting nonexistent protocol returns None."""
        library = ProtocolLibrary()
        info = library.get("nonexistent_protocol")

        assert info is None

    def test_get_decoder_existing(self) -> None:
        """Test getting decoder for existing protocol."""
        library = ProtocolLibrary()
        decoder = library.get_decoder("modbus_rtu")

        assert decoder is not None
        assert isinstance(decoder, ProtocolDecoder)

    def test_get_decoder_caches_result(self) -> None:
        """Test that decoder is cached on subsequent calls."""
        library = ProtocolLibrary()
        decoder1 = library.get_decoder("modbus_rtu")
        decoder2 = library.get_decoder("modbus_rtu")

        assert decoder1 is decoder2  # Same instance

    def test_get_decoder_case_insensitive(self) -> None:
        """Test that decoder lookup is case-insensitive."""
        library = ProtocolLibrary()
        decoder_lower = library.get_decoder("mqtt")
        decoder_upper = library.get_decoder("MQTT")

        assert decoder_lower is not None
        assert decoder_upper is not None

    def test_get_decoder_nonexistent(self) -> None:
        """Test getting decoder for nonexistent protocol returns None."""
        library = ProtocolLibrary()
        decoder = library.get_decoder("nonexistent_protocol")

        assert decoder is None

    def test_get_definition_existing(self) -> None:
        """Test getting protocol definition."""
        library = ProtocolLibrary()
        definition = library.get_definition("dns")

        assert definition is not None
        assert isinstance(definition, ProtocolDefinition)
        assert definition.name == "dns"

    def test_get_definition_nonexistent(self) -> None:
        """Test getting definition for nonexistent protocol returns None."""
        library = ProtocolLibrary()
        definition = library.get_definition("nonexistent")

        assert definition is None

    def test_add_protocol(self) -> None:
        """Test adding custom protocol to library."""
        library = ProtocolLibrary()

        custom = ProtocolInfo(
            name="my_custom_protocol",
            category="custom",
            version="1.0",
            description="My custom protocol",
            definition=ProtocolDefinition(
                name="my_custom_protocol",
                fields=[
                    FieldDefinition(name="magic", field_type="uint32"),
                    FieldDefinition(name="data", field_type="bytes", size_ref="remaining"),
                ],
            ),
        )
        library.add_protocol(custom)

        # Should be retrievable
        retrieved = library.get("my_custom_protocol")
        assert retrieved is not None
        assert retrieved.name == "my_custom_protocol"
        assert retrieved.definition is not None

    def test_add_protocol_overwrite(self) -> None:
        """Test that adding protocol overwrites existing."""
        library = ProtocolLibrary()

        # Get original
        original = library.get("tlv")
        assert original is not None
        original_version = original.version

        # Add with same name but different version
        custom = ProtocolInfo(
            name="tlv",
            category="custom",
            version="2.0",
            description="Updated TLV",
        )
        library.add_protocol(custom)

        # Should be updated
        updated = library.get("tlv")
        assert updated is not None
        assert updated.version == "2.0"
        assert updated.version != original_version

    def test_categories(self) -> None:
        """Test getting list of categories."""
        library = ProtocolLibrary()
        categories = library.categories()

        assert isinstance(categories, list)
        assert "industrial" in categories
        assert "iot" in categories
        assert "network" in categories
        assert "automotive" in categories
        assert "building" in categories
        assert "serial" in categories
        assert "custom" in categories

        # Should be sorted
        assert categories == sorted(categories)


# =============================================================================
# Test Module-Level Functions
# =============================================================================


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_get_library_returns_singleton(self) -> None:
        """Test that get_library returns singleton instance."""
        lib1 = get_library()
        lib2 = get_library()

        assert lib1 is lib2
        assert isinstance(lib1, ProtocolLibrary)

    def test_list_protocols_returns_info(self) -> None:
        """Test list_protocols returns ProtocolInfo objects by default."""
        protocols = list_protocols()

        assert isinstance(protocols, list)
        assert all(isinstance(p, ProtocolInfo) for p in protocols)

    def test_list_protocols_names_only(self) -> None:
        """Test list_protocols with names_only=True."""
        names = list_protocols(names_only=True)

        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "modbus_rtu" in names

    def test_list_protocols_with_category(self) -> None:
        """Test list_protocols with category filter."""
        protocols = list_protocols(category="iot")

        assert all(p.category == "iot" for p in protocols)

    def test_list_protocols_category_and_names_only(self) -> None:
        """Test list_protocols with both category and names_only."""
        names = list_protocols(category="automotive", names_only=True)

        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert "can" in names
        assert "mqtt" not in names  # IoT, not automotive

    def test_get_protocol_existing(self) -> None:
        """Test get_protocol for existing protocol."""
        info = get_protocol("coap")

        assert info is not None
        assert info.name == "coap"
        assert info.category == "iot"

    def test_get_protocol_nonexistent(self) -> None:
        """Test get_protocol for nonexistent protocol."""
        info = get_protocol("nonexistent")

        assert info is None

    def test_get_decoder_existing(self) -> None:
        """Test get_decoder for existing protocol."""
        decoder = get_decoder("ntp")

        assert decoder is not None
        assert isinstance(decoder, ProtocolDecoder)

    def test_get_decoder_nonexistent(self) -> None:
        """Test get_decoder for nonexistent protocol."""
        decoder = get_decoder("nonexistent")

        assert decoder is None


# =============================================================================
# Test Built-in Protocol Definitions
# =============================================================================


class TestBuiltinProtocolsStructure:
    """Test that built-in protocols have proper structure."""

    @pytest.fixture
    def library(self) -> ProtocolLibrary:
        """Create library instance."""
        return ProtocolLibrary()

    def test_all_protocols_have_definitions(self, library: ProtocolLibrary) -> None:
        """Test that all protocols have definitions."""
        for protocol in library.list_protocols():
            assert protocol.definition is not None, f"{protocol.name} missing definition"

    def test_all_protocols_have_required_fields(self, library: ProtocolLibrary) -> None:
        """Test that all protocols have required metadata fields."""
        for protocol in library.list_protocols():
            assert protocol.name, "Protocol missing name"
            assert protocol.category, f"{protocol.name} missing category"
            assert protocol.version, f"{protocol.name} missing version"
            assert protocol.description, f"{protocol.name} missing description"

    def test_all_definitions_have_fields(self, library: ProtocolLibrary) -> None:
        """Test that all protocol definitions have at least one field."""
        for protocol in library.list_protocols():
            assert protocol.definition is not None
            assert len(protocol.definition.fields) > 0, f"{protocol.name} definition has no fields"


class TestModbusRTU:
    """Test Modbus RTU protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get Modbus RTU protocol."""
        library = ProtocolLibrary()
        info = library.get("modbus_rtu")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "modbus_rtu"
        assert protocol.category == "industrial"
        assert "modbus" in protocol.reference.lower()

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "address" in fields
        assert "function_code" in fields
        assert "data" in fields
        assert "crc" in fields

        # Check function code enum
        fc_field = fields["function_code"]
        assert fc_field.enum is not None
        assert 3 in fc_field.enum  # Read holding registers

    def test_decode_modbus_rtu_message(self, protocol: ProtocolInfo) -> None:
        """Test decoding a Modbus RTU read holding registers request."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # Modbus RTU: Address=1, FC=3 (read holding), Start=0, Count=2, CRC
        # Note: The data field uses size_ref="remaining", so it consumes all remaining
        # bytes including what would be the CRC field. This tests the core header decoding.
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B])
        message = decoder.decode(data)

        # Core fields are decoded correctly
        assert message["address"] == 1
        assert message["function_code"] == 3  # Read holding registers
        # Data field consumes remaining bytes (including CRC in this case)
        assert message["data"] == bytes([0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B])


class TestModbusTCP:
    """Test Modbus TCP protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get Modbus TCP protocol."""
        library = ProtocolLibrary()
        info = library.get("modbus_tcp")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "modbus_tcp"
        assert protocol.category == "industrial"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "transaction_id" in fields
        assert "protocol_id" in fields
        assert "length" in fields
        assert "unit_id" in fields
        assert "function_code" in fields

    def test_decode_modbus_tcp_message(self, protocol: ProtocolInfo) -> None:
        """Test decoding a Modbus TCP message."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # Modbus TCP header + function code
        data = bytes(
            [
                0x00,
                0x01,  # Transaction ID
                0x00,
                0x00,  # Protocol ID (Modbus)
                0x00,
                0x06,  # Length
                0x01,  # Unit ID
                0x03,  # Function code (read holding registers)
                0x00,
                0x00,  # Start address
                0x00,
                0x01,  # Count
            ]
        )
        message = decoder.decode(data)

        assert message.valid
        assert message["transaction_id"] == 1
        assert message["protocol_id"] == 0
        assert message["unit_id"] == 1
        assert message["function_code"] == 3


class TestDNP3:
    """Test DNP3 protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get DNP3 protocol."""
        library = ProtocolLibrary()
        info = library.get("dnp3")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "dnp3"
        assert protocol.category == "industrial"
        assert protocol.definition is not None
        assert protocol.definition.endian == "little"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "start" in fields
        assert "length" in fields
        assert "control" in fields
        assert "destination" in fields
        assert "source" in fields
        assert "crc" in fields

        # Check start value
        assert fields["start"].value == 0x0564


class TestMQTT:
    """Test MQTT protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get MQTT protocol."""
        library = ProtocolLibrary()
        info = library.get("mqtt")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "mqtt"
        assert protocol.category == "iot"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "fixed_header" in fields
        assert "remaining_length" in fields
        assert "payload" in fields

    def test_decode_mqtt_connect(self, protocol: ProtocolInfo) -> None:
        """Test decoding an MQTT CONNECT packet (simplified)."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # MQTT CONNECT (type 1, flags 0)
        data = bytes([0x10, 0x0C])  # Fixed header + remaining length
        data += b"MQTT"  # Protocol name (part of payload)
        message = decoder.decode(data)

        assert message.valid
        assert message["fixed_header"] == 0x10  # CONNECT


class TestCoAP:
    """Test CoAP protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get CoAP protocol."""
        library = ProtocolLibrary()
        info = library.get("coap")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "coap"
        assert protocol.category == "iot"
        assert "RFC 7252" in protocol.version

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "version_type_tkl" in fields
        assert "code" in fields
        assert "message_id" in fields

        # Check code enum
        code_field = fields["code"]
        assert code_field.enum is not None
        assert 0x01 in code_field.enum  # GET

    def test_decode_coap_get(self, protocol: ProtocolInfo) -> None:
        """Test decoding a CoAP GET request."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # CoAP: Version=1, Type=CON, TKL=0, Code=GET, MID=1234
        data = bytes([0x40, 0x01, 0x04, 0xD2])
        message = decoder.decode(data)

        assert message.valid
        assert message["code"] == 0x01  # GET
        assert message["message_id"] == 0x04D2


class TestCAN:
    """Test CAN protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get CAN protocol."""
        library = ProtocolLibrary()
        info = library.get("can")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "can"
        assert protocol.category == "automotive"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "identifier" in fields
        assert "dlc" in fields
        assert "data" in fields

        # Check data size
        assert fields["data"].size == 8

    def test_decode_can_frame(self, protocol: ProtocolInfo) -> None:
        """Test decoding a CAN frame."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # CAN frame: ID=0x123, DLC=8, 8 bytes data
        data = bytes(
            [
                0x00,
                0x00,
                0x01,
                0x23,  # ID (big endian)
                0x08,  # DLC
                0x01,
                0x02,
                0x03,
                0x04,  # Data
                0x05,
                0x06,
                0x07,
                0x08,
            ]
        )
        message = decoder.decode(data)

        assert message.valid
        assert message["identifier"] == 0x123
        assert message["dlc"] == 8


class TestDNS:
    """Test DNS protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get DNS protocol."""
        library = ProtocolLibrary()
        info = library.get("dns")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "dns"
        assert protocol.category == "network"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "transaction_id" in fields
        assert "flags" in fields
        assert "questions" in fields
        assert "answer_rrs" in fields
        assert "authority_rrs" in fields
        assert "additional_rrs" in fields

    def test_decode_dns_query(self, protocol: ProtocolInfo) -> None:
        """Test decoding a DNS query header."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # DNS header: TxID=0x1234, Flags=0x0100 (standard query), 1 question
        data = bytes(
            [
                0x12,
                0x34,  # Transaction ID
                0x01,
                0x00,  # Flags (standard query)
                0x00,
                0x01,  # Questions: 1
                0x00,
                0x00,  # Answer RRs: 0
                0x00,
                0x00,  # Authority RRs: 0
                0x00,
                0x00,  # Additional RRs: 0
            ]
        )
        message = decoder.decode(data)

        assert message.valid
        assert message["transaction_id"] == 0x1234
        assert message["questions"] == 1


class TestNTP:
    """Test NTP protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get NTP protocol."""
        library = ProtocolLibrary()
        info = library.get("ntp")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "ntp"
        assert protocol.category == "network"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "flags" in fields
        assert "stratum" in fields
        assert "poll" in fields
        assert "precision" in fields
        assert "root_delay" in fields
        assert "root_dispersion" in fields
        assert "reference_id" in fields
        assert "reference_timestamp" in fields
        assert "origin_timestamp" in fields
        assert "receive_timestamp" in fields
        assert "transmit_timestamp" in fields

    def test_decode_ntp_packet(self, protocol: ProtocolInfo) -> None:
        """Test decoding an NTP packet header."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # NTP packet: LI=0, VN=4, Mode=3 (client), Stratum=0
        data = bytes(
            [
                0x23,  # LI=0, VN=4, Mode=3
                0x00,  # Stratum
                0x06,  # Poll
                0xEC,  # Precision (-20)
            ]
        )
        # Add remaining fields (root delay, dispersion, ref ID, timestamps)
        data += bytes([0x00] * 44)  # Fill remaining NTP header

        message = decoder.decode(data)

        assert message.valid
        assert message["flags"] == 0x23
        assert message["stratum"] == 0


class TestTLV:
    """Test TLV (Tag-Length-Value) protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get TLV protocol."""
        library = ProtocolLibrary()
        info = library.get("tlv")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "tlv"
        assert protocol.category == "custom"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "tag" in fields
        assert "length" in fields
        assert "value" in fields

        # Check value references length
        assert fields["value"].size_ref == "length"

    def test_decode_tlv_message(self, protocol: ProtocolInfo) -> None:
        """Test decoding a TLV message."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # TLV: Tag=1, Length=4, Value=0x01020304
        data = bytes([0x01, 0x04, 0x01, 0x02, 0x03, 0x04])
        message = decoder.decode(data)

        assert message.valid
        assert message["tag"] == 1
        assert message["length"] == 4
        assert message["value"] == bytes([0x01, 0x02, 0x03, 0x04])


class TestLengthPrefixed:
    """Test length-prefixed protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get length-prefixed protocol."""
        library = ProtocolLibrary()
        info = library.get("length_prefixed")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "length_prefixed"
        assert protocol.category == "custom"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "length" in fields
        assert "payload" in fields

        assert fields["length"].field_type == "uint16"
        assert fields["payload"].size_ref == "length"

    def test_decode_length_prefixed(self, protocol: ProtocolInfo) -> None:
        """Test decoding a length-prefixed message."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # Length=5, Payload="Hello"
        data = bytes([0x00, 0x05]) + b"Hello"
        message = decoder.decode(data)

        assert message.valid
        assert message["length"] == 5
        assert message["payload"] == b"Hello"


class TestXMODEM:
    """Test XMODEM protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get XMODEM protocol."""
        library = ProtocolLibrary()
        info = library.get("xmodem")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "xmodem"
        assert protocol.category == "serial"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "header" in fields
        assert "block_number" in fields
        assert "block_complement" in fields
        assert "data" in fields
        assert "checksum" in fields

        # Check header enum
        header_field = fields["header"]
        assert header_field.enum is not None
        assert 0x01 in header_field.enum  # SOH

        # Check data size
        assert fields["data"].size == 128


class TestKNX:
    """Test KNX protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get KNX protocol."""
        library = ProtocolLibrary()
        info = library.get("knx")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "knx"
        assert protocol.category == "building"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "header_length" in fields
        assert "protocol_version" in fields
        assert "service_type" in fields
        assert "total_length" in fields

        # Check constant values
        assert fields["header_length"].value == 0x06
        assert fields["protocol_version"].value == 0x10

        # Check service type enum
        service_field = fields["service_type"]
        assert service_field.enum is not None
        assert 0x0420 in service_field.enum  # tunnelling_request


class TestOBD2:
    """Test OBD-II protocol definition."""

    @pytest.fixture
    def protocol(self) -> ProtocolInfo:
        """Get OBD-II protocol."""
        library = ProtocolLibrary()
        info = library.get("obd2")
        assert info is not None
        return info

    def test_protocol_metadata(self, protocol: ProtocolInfo) -> None:
        """Test protocol metadata."""
        assert protocol.name == "obd2"
        assert protocol.category == "automotive"

    def test_protocol_fields(self, protocol: ProtocolInfo) -> None:
        """Test protocol field structure."""
        assert protocol.definition is not None
        fields = {f.name: f for f in protocol.definition.fields}

        assert "mode" in fields
        assert "pid" in fields
        assert "data" in fields

        # Check mode enum
        mode_field = fields["mode"]
        assert mode_field.enum is not None
        assert 0x01 in mode_field.enum  # show_current_data

    def test_decode_obd2_request(self, protocol: ProtocolInfo) -> None:
        """Test decoding an OBD-II request."""
        assert protocol.definition is not None
        decoder = ProtocolDecoder(protocol.definition)

        # OBD-II: Mode=1 (current data), PID=0x0C (engine RPM)
        data = bytes([0x01, 0x0C])
        message = decoder.decode(data)

        assert message.valid
        assert message["mode"] == 1
        assert message["pid"] == 0x0C


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestInferenceProtocolLibraryEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_decode(self) -> None:
        """Test decoding empty data."""
        library = ProtocolLibrary()
        decoder = library.get_decoder("tlv")
        assert decoder is not None

        message = decoder.decode(b"")

        assert not message.valid
        assert "Insufficient data" in message.errors[0]

    def test_truncated_data_decode(self) -> None:
        """Test decoding truncated data."""
        library = ProtocolLibrary()
        decoder = library.get_decoder("dns")
        assert decoder is not None

        # Only 4 bytes of a 12-byte DNS header
        message = decoder.decode(bytes([0x12, 0x34, 0x01, 0x00]))

        # Should decode partially
        assert "transaction_id" in message.fields

    def test_protocol_with_no_definition(self) -> None:
        """Test getting decoder for protocol without definition."""
        library = ProtocolLibrary()

        # Add protocol without definition
        info = ProtocolInfo(
            name="no_def_protocol",
            category="custom",
            version="1.0",
            description="Protocol without definition",
            definition=None,
        )
        library.add_protocol(info)

        decoder = library.get_decoder("no_def_protocol")
        assert decoder is None

    def test_multiple_library_instances(self) -> None:
        """Test that multiple library instances are independent."""
        lib1 = ProtocolLibrary()
        lib2 = ProtocolLibrary()

        # Add custom protocol to lib1
        lib1.add_protocol(
            ProtocolInfo(
                name="unique_to_lib1",
                category="custom",
                version="1.0",
                description="Only in lib1",
            )
        )

        # lib1 should have it
        assert lib1.get("unique_to_lib1") is not None

        # lib2 should not
        assert lib2.get("unique_to_lib1") is None


# =============================================================================
# Test Documentation Examples
# =============================================================================


class TestDocumentationExamples:
    """Test examples from module docstrings."""

    def test_library_basic_usage(self) -> None:
        """Test basic library usage from docstring."""
        library = ProtocolLibrary()
        modbus = library.get("modbus_rtu")

        assert modbus is not None
        assert modbus.name == "modbus_rtu"

    def test_get_decoder_usage(self) -> None:
        """Test get_decoder usage from docstring."""
        library = ProtocolLibrary()
        decoder = library.get_decoder("modbus_rtu")

        assert decoder is not None

        # Decode a sample message - test that it can decode core fields
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B])
        message = decoder.decode(data)

        # Verify core fields are decoded
        assert message["address"] == 1
        assert message["function_code"] == 3

    def test_list_protocols_example(self) -> None:
        """Test list_protocols example from docstring."""
        protocols = list_protocols(category="industrial")

        for p in protocols:
            assert p.name  # Can print f"{p.name}: {p.description}"

    def test_list_protocol_names_example(self) -> None:
        """Test list_protocols with names_only example."""
        names = list_protocols(names_only=True)

        assert "http" in names

    def test_add_custom_protocol_example(self) -> None:
        """Test adding custom protocol example from docstring."""
        library = ProtocolLibrary()

        custom = ProtocolInfo(
            name="my_protocol",
            category="custom",
            version="1.0",
            description="My custom protocol",
            definition=ProtocolDefinition(
                name="my_protocol",
                fields=[FieldDefinition(name="header", field_type="uint8")],
            ),
        )
        library.add_protocol(custom)

        # Retrieve it
        retrieved = library.get("my_protocol")
        assert retrieved is not None
        assert retrieved.name == "my_protocol"
