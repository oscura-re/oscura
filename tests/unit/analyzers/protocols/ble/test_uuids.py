"""Unit tests for BLE UUID utilities.

Tests for src/oscura/analyzers/protocols/ble/uuids.py

This test suite covers:
- UUID conversion between bytes and strings
- Standard service/characteristic/descriptor name lookup
- AD type mappings
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from oscura.analyzers.protocols.ble.uuids import (
    AD_TYPES,
    STANDARD_CHARACTERISTICS,
    STANDARD_DESCRIPTORS,
    STANDARD_SERVICES,
    get_characteristic_name,
    get_descriptor_name,
    get_service_name,
    string_to_uuid_bytes,
    uuid_to_string,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# UUID Conversion Tests
# =============================================================================


class TestUUIDConversion:
    """Test UUID conversion between bytes and strings."""

    def test_uuid_to_string_16bit_short(self):
        """Test converting 16-bit UUID to short string format."""
        uuid_bytes = (0x180D).to_bytes(2, "little")
        result = uuid_to_string(uuid_bytes, short=True)

        assert result == "0x180D"

    def test_uuid_to_string_16bit_full(self):
        """Test converting 16-bit UUID to full string format."""
        uuid_bytes = (0x180D).to_bytes(2, "little")
        result = uuid_to_string(uuid_bytes, short=False)

        assert result == "0000180d-0000-1000-8000-00805f9b34fb"

    def test_uuid_to_string_128bit(self):
        """Test converting 128-bit UUID to string."""
        # Full Bluetooth Base UUID
        uuid_bytes = bytes.fromhex("fb349b5f800000801000000000000000")[::-1]
        result = uuid_to_string(uuid_bytes)

        assert "-" in result
        assert len(result) == 36  # Standard UUID format

    def test_string_to_uuid_bytes_16bit(self):
        """Test converting short UUID string to bytes."""
        uuid_str = "0x180D"
        result = string_to_uuid_bytes(uuid_str)

        assert result == (0x180D).to_bytes(2, "little")

    def test_string_to_uuid_bytes_full_format(self):
        """Test converting full UUID string to bytes."""
        uuid_str = "0000180d-0000-1000-8000-00805f9b34fb"
        result = string_to_uuid_bytes(uuid_str)

        assert len(result) == 16

    def test_string_to_uuid_bytes_invalid_format(self):
        """Test converting invalid UUID string."""
        with pytest.raises(ValueError, match="Invalid UUID"):
            string_to_uuid_bytes("invalid-uuid")

    def test_uuid_roundtrip_16bit(self):
        """Test roundtrip conversion for 16-bit UUID."""
        original = (0x180D).to_bytes(2, "little")
        uuid_str = uuid_to_string(original, short=True)
        converted = string_to_uuid_bytes(uuid_str)

        assert original == converted


# =============================================================================
# Service Name Lookup Tests
# =============================================================================


class TestServiceNameLookup:
    """Test standard service name lookup."""

    def test_get_service_name_heart_rate(self):
        """Test getting Heart Rate service name."""
        name = get_service_name(0x180D)
        assert name == "Heart Rate"

    def test_get_service_name_battery(self):
        """Test getting Battery service name."""
        name = get_service_name(0x180F)
        assert name == "Battery Service"

    def test_get_service_name_device_info(self):
        """Test getting Device Information service name."""
        name = get_service_name(0x180A)
        assert name == "Device Information"

    def test_get_service_name_string_uuid(self):
        """Test getting service name from string UUID."""
        name = get_service_name("0x180D")
        assert name == "Heart Rate"

    def test_get_service_name_unknown(self):
        """Test getting unknown service name."""
        name = get_service_name(0x9999)
        assert name == "Unknown Service"

    def test_get_service_name_custom_string(self):
        """Test getting service name from custom string UUID."""
        name = get_service_name("custom-uuid-string")
        assert name == "Custom Service"


# =============================================================================
# Characteristic Name Lookup Tests
# =============================================================================


class TestCharacteristicNameLookup:
    """Test standard characteristic name lookup."""

    def test_get_characteristic_name_device_name(self):
        """Test getting Device Name characteristic name."""
        name = get_characteristic_name(0x2A00)
        assert name == "Device Name"

    def test_get_characteristic_name_battery_level(self):
        """Test getting Battery Level characteristic name."""
        name = get_characteristic_name(0x2A19)
        assert name == "Battery Level"

    def test_get_characteristic_name_heart_rate(self):
        """Test getting Heart Rate Measurement characteristic name."""
        name = get_characteristic_name(0x2A37)
        assert name == "Heart Rate Measurement"

    def test_get_characteristic_name_string_uuid(self):
        """Test getting characteristic name from string UUID."""
        name = get_characteristic_name("0x2A37")
        assert name == "Heart Rate Measurement"

    def test_get_characteristic_name_unknown(self):
        """Test getting unknown characteristic name."""
        name = get_characteristic_name(0x8888)
        assert name == "Unknown Characteristic"


# =============================================================================
# Descriptor Name Lookup Tests
# =============================================================================


class TestDescriptorNameLookup:
    """Test standard descriptor name lookup."""

    def test_get_descriptor_name_cccd(self):
        """Test getting Client Characteristic Configuration descriptor name."""
        name = get_descriptor_name(0x2902)
        assert name == "Client Characteristic Configuration"

    def test_get_descriptor_name_user_description(self):
        """Test getting Characteristic User Description descriptor name."""
        name = get_descriptor_name(0x2901)
        assert name == "Characteristic User Description"

    def test_get_descriptor_name_string_uuid(self):
        """Test getting descriptor name from string UUID."""
        name = get_descriptor_name("0x2902")
        assert name == "Client Characteristic Configuration"

    def test_get_descriptor_name_unknown(self):
        """Test getting unknown descriptor name."""
        name = get_descriptor_name(0x7777)
        assert name == "Unknown Descriptor"


# =============================================================================
# Standard Mappings Tests
# =============================================================================


class TestStandardMappings:
    """Test standard UUID mapping dictionaries."""

    def test_standard_services_not_empty(self):
        """Test that standard services dict is populated."""
        assert len(STANDARD_SERVICES) > 0
        assert 0x180D in STANDARD_SERVICES  # Heart Rate

    def test_standard_characteristics_not_empty(self):
        """Test that standard characteristics dict is populated."""
        assert len(STANDARD_CHARACTERISTICS) > 0
        assert 0x2A37 in STANDARD_CHARACTERISTICS  # HR Measurement

    def test_standard_descriptors_not_empty(self):
        """Test that standard descriptors dict is populated."""
        assert len(STANDARD_DESCRIPTORS) > 0
        assert 0x2902 in STANDARD_DESCRIPTORS  # CCCD

    def test_ad_types_not_empty(self):
        """Test that AD types dict is populated."""
        assert len(AD_TYPES) > 0
        assert 0x01 in AD_TYPES  # Flags
        assert 0x09 in AD_TYPES  # Complete Local Name
        assert 0xFF in AD_TYPES  # Manufacturer Data

    def test_common_services_present(self):
        """Test that common BLE services are present."""
        common_services = [
            0x1800,  # Generic Access
            0x1801,  # Generic Attribute
            0x180D,  # Heart Rate
            0x180F,  # Battery Service
            0x180A,  # Device Information
        ]

        for service_uuid in common_services:
            assert service_uuid in STANDARD_SERVICES

    def test_common_characteristics_present(self):
        """Test that common BLE characteristics are present."""
        common_chars = [
            0x2A00,  # Device Name
            0x2A19,  # Battery Level
            0x2A37,  # Heart Rate Measurement
            0x2A29,  # Manufacturer Name String
        ]

        for char_uuid in common_chars:
            assert char_uuid in STANDARD_CHARACTERISTICS


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_uuid_to_string_empty_bytes(self):
        """Test converting empty bytes."""
        result = uuid_to_string(b"")
        assert result == ""

    def test_uuid_to_string_invalid_length(self):
        """Test converting bytes with invalid length."""
        # 3 bytes is not a valid UUID length
        result = uuid_to_string(b"\x01\x02\x03")
        # Should return hex representation
        assert isinstance(result, str)

    def test_string_to_uuid_bytes_uppercase(self):
        """Test converting uppercase UUID string."""
        uuid_str = "0X180D"
        result = string_to_uuid_bytes(uuid_str)

        assert result == (0x180D).to_bytes(2, "little")

    def test_get_service_name_zero(self):
        """Test getting service name for UUID 0."""
        name = get_service_name(0x0000)
        assert "Unknown" in name

    def test_uuid_to_string_32bit(self):
        """Test converting 32-bit UUID (rare)."""
        uuid_bytes = (0x12345678).to_bytes(4, "little")
        result = uuid_to_string(uuid_bytes, short=True)

        assert result == "0x12345678"
