"""Tests for Zigbee Cluster Library (ZCL) parsing.

Tests ZCL frame parsing, cluster-specific commands, and global commands.
"""

from __future__ import annotations

from oscura.iot.zigbee.zcl import (
    ZCL_CLUSTERS,
    parse_zcl_frame,
)


class TestZCLClusters:
    """Test ZCL cluster definitions."""

    def test_standard_clusters_defined(self) -> None:
        """Test standard clusters are defined."""
        assert 0x0000 in ZCL_CLUSTERS
        assert ZCL_CLUSTERS[0x0000] == "Basic"
        assert ZCL_CLUSTERS[0x0006] == "On/Off"
        assert ZCL_CLUSTERS[0x0008] == "Level Control"
        assert ZCL_CLUSTERS[0x0402] == "Temperature Measurement"

    def test_cluster_count(self) -> None:
        """Test reasonable number of clusters defined."""
        assert len(ZCL_CLUSTERS) >= 15


class TestZCLFrameParsing:
    """Test ZCL frame parsing."""

    def test_parse_onoff_on(self) -> None:
        """Test parsing On/Off On command."""
        data = bytes([0x01, 0x00, 0x01])  # Frame control, seq, cmd

        result = parse_zcl_frame(0x0006, data)

        assert result["cluster_id"] == 0x0006
        assert result["cluster_name"] == "On/Off"
        assert result["command_name"] == "On"
        assert result["transaction_sequence"] == 0x00

    def test_parse_onoff_off(self) -> None:
        """Test parsing On/Off Off command."""
        data = bytes([0x01, 0x05, 0x00])

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Off"
        assert result["transaction_sequence"] == 0x05

    def test_parse_onoff_toggle(self) -> None:
        """Test parsing On/Off Toggle command."""
        data = bytes([0x01, 0x10, 0x02])

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Toggle"

    def test_parse_level_control_move_to_level(self) -> None:
        """Test parsing Level Control Move to Level command."""
        data = bytes([0x01, 0x00, 0x00, 0x80, 0x0A, 0x00])

        result = parse_zcl_frame(0x0008, data)

        assert result["cluster_name"] == "Level Control"
        assert result["command_name"] == "Move to Level"
        assert result["details"]["level"] == 0x80
        assert result["details"]["transition_time"] == 10

    def test_parse_unknown_cluster(self) -> None:
        """Test parsing unknown cluster."""
        data = bytes([0x01, 0x00, 0x00])

        result = parse_zcl_frame(0xFFFF, data)

        assert "Unknown" in result["cluster_name"]
        assert result["cluster_id"] == 0xFFFF

    def test_parse_insufficient_data(self) -> None:
        """Test parsing with insufficient data."""
        data = bytes([0x01])  # Only frame control

        result = parse_zcl_frame(0x0006, data)

        assert "error" in result


class TestZCLGlobalCommands:
    """Test ZCL global commands."""

    def test_parse_read_attributes(self) -> None:
        """Test parsing Read Attributes command."""
        data = bytes(
            [
                0x00,  # Frame control (global)
                0x01,  # Transaction sequence
                0x00,  # Command ID (Read Attributes)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x05,
                0x00,  # Attribute ID 0x0005
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["frame_type"] == "global"
        assert result["command_name"] == "Read Attributes"
        assert "details" in result
        assert result["details"]["attribute_ids"] == [0x0000, 0x0005]

    def test_parse_read_attributes_response(self) -> None:
        """Test parsing Read Attributes Response."""
        data = bytes(
            [
                0x08,  # Frame control (global, server to client)
                0x01,  # Transaction sequence
                0x01,  # Command ID (Read Attributes Response)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x00,  # Status (success)
                0x20,  # Data type (Uint8)
                0x01,  # Value
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Read Attributes Response"
        assert result["direction"] == "server_to_client"
        assert "details" in result
        assert result["details"]["attributes"][0]["attribute_id"] == 0x0000
        assert result["details"]["attributes"][0]["value"] == 0x01

    def test_parse_read_attributes_response_uint16(self) -> None:
        """Test parsing Read Attributes Response with Uint16."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID
                0x05,
                0x00,  # Attribute ID 0x0005
                0x00,  # Status (success)
                0x21,  # Data type (Uint16)
                0x34,
                0x12,  # Value (0x1234)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] == 0x1234

    def test_parse_read_attributes_response_failed(self) -> None:
        """Test parsing Read Attributes Response with failure status."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID
                0x00,
                0x00,  # Attribute ID
                0x86,  # Status (unsupported attribute)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["status"] == 0x86

    def test_parse_report_attributes(self) -> None:
        """Test parsing Report Attributes command."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0x20,  # Data type (Uint8)
                0x01,  # Value
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Report Attributes"
        assert result["details"]["attributes"][0]["attribute_id"] == 0x0000
        assert result["details"]["attributes"][0]["value"] == 0x01


class TestZCLManufacturerSpecific:
    """Test manufacturer-specific ZCL frames."""

    def test_parse_manufacturer_specific_frame(self) -> None:
        """Test parsing manufacturer-specific frame."""
        data = bytes(
            [
                0x05,  # Frame control (cluster-specific + manufacturer)
                0x01,  # Transaction sequence
                0x01,  # Command ID (On command)
                0x34,
                0x12,  # Manufacturer code (little-endian)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["manufacturer_specific"] is True
        assert result["manufacturer_code"] == 0x1234
        assert result["command_id"] == 0x01

    def test_parse_manufacturer_specific_insufficient_data(self) -> None:
        """Test manufacturer-specific with insufficient data."""
        data = bytes(
            [
                0x05,  # Frame control (manufacturer bit set)
                0x01,  # Transaction sequence
                # Missing manufacturer code
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "error" in result


class TestZCLFrameControl:
    """Test ZCL frame control field parsing."""

    def test_parse_disable_default_response(self) -> None:
        """Test disable default response flag."""
        data = bytes(
            [
                0x11,  # Frame control (disable default response)
                0x01,  # Transaction sequence
                0x01,  # Command ID
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["disable_default_response"] is True

    def test_parse_direction_client_to_server(self) -> None:
        """Test client to server direction."""
        data = bytes(
            [
                0x01,  # Frame control (client to server)
                0x01,  # Transaction sequence
                0x01,  # Command ID
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["direction"] == "client_to_server"

    def test_parse_direction_server_to_client(self) -> None:
        """Test server to client direction."""
        data = bytes(
            [
                0x09,  # Frame control (server to client)
                0x01,  # Transaction sequence
                0x01,  # Command ID
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["direction"] == "server_to_client"


class TestZCLColorControl:
    """Test Color Control cluster commands."""

    def test_parse_move_to_hue(self) -> None:
        """Test parsing Move to Hue command."""
        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x00,  # Command ID (Move to Hue)
            ]
        )

        result = parse_zcl_frame(0x0300, data)

        assert result["cluster_name"] == "Color Control"
        assert result["command_name"] == "Move to Hue"

    def test_parse_move_to_color_temperature(self) -> None:
        """Test parsing Move to Color Temperature command."""
        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x0A,  # Command ID (Move to Color Temperature)
            ]
        )

        result = parse_zcl_frame(0x0300, data)

        assert result["command_name"] == "Move to Color Temperature"


class TestZCLEdgeCases:
    """Test ZCL edge cases and error handling."""

    def test_parse_empty_data(self) -> None:
        """Test parsing empty data."""
        data = bytes([])

        result = parse_zcl_frame(0x0006, data)

        assert "error" in result

    def test_parse_single_byte(self) -> None:
        """Test parsing single byte."""
        data = bytes([0x01])

        result = parse_zcl_frame(0x0006, data)

        assert "error" in result

    def test_parse_two_bytes(self) -> None:
        """Test parsing two bytes."""
        data = bytes([0x01, 0x00])

        result = parse_zcl_frame(0x0006, data)

        assert "error" in result

    def test_parse_read_attributes_empty_list(self) -> None:
        """Test parsing Read Attributes with empty attribute list."""
        data = bytes(
            [
                0x00,  # Frame control (global)
                0x01,  # Transaction sequence
                0x00,  # Command ID (Read Attributes)
                # No attribute IDs
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Read Attributes"
        assert "details" in result

    def test_parse_unknown_global_command(self) -> None:
        """Test parsing unknown global command."""
        data = bytes(
            [
                0x00,  # Frame control (global)
                0x01,  # Transaction sequence
                0xFF,  # Unknown command ID
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "Unknown Global" in result["command_name"]

    def test_parse_unknown_cluster_command(self) -> None:
        """Test parsing unknown cluster-specific command."""
        data = bytes(
            [
                0x01,  # Frame control (cluster-specific)
                0x01,  # Transaction sequence
                0xFF,  # Unknown command ID
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "Unknown" in result["command_name"]

    def test_parse_read_attributes_insufficient_data(self) -> None:
        """Test parsing Read Attributes with insufficient data."""
        data = bytes(
            [
                0x00,  # Frame control (global)
                0x01,  # Transaction sequence
                0x00,  # Command ID (Read Attributes)
                0x00,  # Only one byte of attribute ID (insufficient)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Read Attributes"
        assert "details" in result

    def test_parse_read_attributes_response_boolean(self) -> None:
        """Test parsing Read Attributes Response with Boolean data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID (Read Attributes Response)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x00,  # Status (success)
                0x10,  # Data type (Boolean)
                0x01,  # Value (True)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] is True

    def test_parse_read_attributes_response_int16(self) -> None:
        """Test parsing Read Attributes Response with Int16 (signed)."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID (Read Attributes Response)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x00,  # Status (success)
                0x29,  # Data type (Int16)
                0xFF,
                0xFF,  # Value (-1 in signed int16)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] == -1

    def test_parse_read_attributes_response_unknown_data_type(self) -> None:
        """Test parsing Read Attributes Response with unknown data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID (Read Attributes Response)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x00,  # Status (success)
                0xFF,  # Unknown data type
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "attributes" in result["details"]
        # When unknown data type encountered, parsing stops
        if result["details"]["attributes"]:
            assert result["details"]["attributes"][0]["data_type"] == 0xFF

    def test_parse_read_attributes_response_insufficient_data_for_value(self) -> None:
        """Test parsing Read Attributes Response with insufficient data for value."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x01,  # Command ID (Read Attributes Response)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x00,  # Status (success)
                0x21,  # Data type (Uint16)
                0x01,  # Only 1 byte (need 2)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "attributes" in result["details"]

    def test_parse_report_attributes_boolean(self) -> None:
        """Test parsing Report Attributes with Boolean data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0x10,  # Data type (Boolean)
                0x00,  # Value (False)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] is False

    def test_parse_report_attributes_uint16(self) -> None:
        """Test parsing Report Attributes with Uint16 data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0x21,  # Data type (Uint16)
                0xAB,
                0xCD,  # Value (0xCDAB in little-endian)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] == 0xCDAB

    def test_parse_report_attributes_int16(self) -> None:
        """Test parsing Report Attributes with Int16 (signed) data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0x29,  # Data type (Int16)
                0x00,
                0x80,  # Value (-32768 in signed int16)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert result["details"]["attributes"][0]["value"] == -32768

    def test_parse_report_attributes_unknown_data_type(self) -> None:
        """Test parsing Report Attributes with unknown data type."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0xFF,  # Unknown data type
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "attributes" in result["details"]
        # When unknown data type encountered, parsing stops
        if result["details"]["attributes"]:
            assert result["details"]["attributes"][0]["data_type"] == 0xFF

    def test_parse_report_attributes_insufficient_data(self) -> None:
        """Test parsing Report Attributes with insufficient data."""
        data = bytes(
            [
                0x08,  # Frame control
                0x01,  # Transaction sequence
                0x0A,  # Command ID (Report Attributes)
                0x00,
                0x00,  # Attribute ID
                0x21,  # Data type (Uint16)
                0x01,  # Only 1 byte (need 2)
            ]
        )

        result = parse_zcl_frame(0x0006, data)

        assert "attributes" in result["details"]

    def test_parse_level_control_move_to_level_minimal_payload(self) -> None:
        """Test parsing Level Control Move to Level with minimal payload."""
        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x00,  # Command ID (Move to Level)
                0x80,  # Level only, no transition time
            ]
        )

        result = parse_zcl_frame(0x0008, data)

        assert result["command_name"] == "Move to Level"
        # Should handle missing transition time gracefully

    def test_parse_generic_cluster_command(self) -> None:
        """Test parsing cluster-specific command for generic cluster."""
        data = bytes(
            [
                0x01,  # Frame control (cluster-specific)
                0x01,  # Transaction sequence
                0x10,  # Command ID
            ]
        )

        result = parse_zcl_frame(0x0402, data)  # Temperature Measurement cluster

        assert "Cluster Command" in result["command_name"]
