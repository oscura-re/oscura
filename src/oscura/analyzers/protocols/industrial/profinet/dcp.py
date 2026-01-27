"""DCP (Discovery and Configuration Protocol) implementation for PROFINET.

DCP is used for device identification, configuration, and network assignment
in PROFINET networks. It runs directly on Ethernet Layer 2.

References:
    PROFINET Specification V2.4 - Section 4.3 DCP
    IEC 61158-6-10:2014
"""

from __future__ import annotations

from typing import Any, ClassVar


class DCPParser:
    """DCP (Discovery and Configuration Protocol) frame parser.

    DCP is used for device discovery, identification, and configuration
    in PROFINET networks. It operates at Layer 2 (Ethernet).

    Attributes:
        SERVICE_IDS: Mapping of DCP service IDs to names.
        SERVICE_TYPES: Mapping of service types to names.
        OPTIONS: Mapping of DCP option codes to names.
    """

    # DCP Service IDs (Request/Response)
    SERVICE_IDS: ClassVar[dict[int, str]] = {
        0x00: "Reserved",
        0x01: "Get",
        0x02: "Set",
        0x03: "Identify",
        0x04: "Hello",
    }

    # DCP Service Types
    SERVICE_TYPES: ClassVar[dict[int, str]] = {
        0x00: "Request",
        0x01: "Response Success",
        0x05: "Response Not Supported",
    }

    # DCP Options
    OPTIONS: ClassVar[dict[int, str]] = {
        0x01: "IP",
        0x02: "Device Properties",
        0x03: "DHCP",
        0x04: "Reserved",
        0x05: "Control",
        0x06: "Device Initiative",
        0xFF: "All Selector",
    }

    # DCP Suboptions for Device Properties (Option 0x02)
    DEVICE_PROPS_SUBOPTIONS: ClassVar[dict[int, str]] = {
        0x01: "Manufacturer specific",
        0x02: "Name of Station",
        0x03: "Device ID",
        0x04: "Device Role",
        0x05: "Device Options",
        0x06: "Alias Name",
        0x07: "Device Instance",
        0x08: "OEM Device ID",
    }

    # DCP Suboptions for IP (Option 0x01)
    IP_SUBOPTIONS: ClassVar[dict[int, str]] = {
        0x01: "MAC address",
        0x02: "IP parameter",
        0x03: "Full IP Suite",
    }

    @staticmethod
    def parse_frame(data: bytes) -> dict[str, Any]:
        """Parse complete DCP frame.

        DCP Frame Format:
        - Service ID (1 byte)
        - Service Type (1 byte)
        - XID (4 bytes) - Transaction ID
        - Response Delay (2 bytes)
        - DCPDataLength (2 bytes)
        - Blocks (variable)

        Args:
            data: Raw DCP frame data (without Ethernet header).

        Returns:
            Parsed DCP frame data.

        Raises:
            ValueError: If frame is too short or invalid.

        Example:
            >>> parser = DCPParser()
            >>> result = parser.parse_frame(dcp_data)
            >>> print(f"Service: {result['service']}, Blocks: {len(result['blocks'])}")
        """
        if len(data) < 10:
            raise ValueError(f"DCP frame too short: {len(data)} bytes (minimum 10)")

        service_id = data[0]
        service_type = data[1]
        xid = int.from_bytes(data[2:6], "big")
        response_delay = int.from_bytes(data[6:8], "big")
        dcp_data_length = int.from_bytes(data[8:10], "big")

        blocks = []
        offset = 10

        # Parse DCP blocks
        while offset < len(data) and offset < 10 + dcp_data_length:
            if offset + 4 > len(data):
                break

            option = data[offset]
            suboption = data[offset + 1]
            block_length = int.from_bytes(data[offset + 2 : offset + 4], "big")

            if offset + 4 + block_length > len(data):
                break

            block_data = data[offset + 4 : offset + 4 + block_length]

            # Parse block data based on option/suboption
            parsed_block = DCPParser._parse_block(option, suboption, block_data)
            blocks.append(parsed_block)

            offset += 4 + block_length
            # Align to 2-byte boundary
            if block_length % 2:
                offset += 1

        return {
            "service": DCPParser.SERVICE_IDS.get(service_id, f"Unknown (0x{service_id:02X})"),
            "service_id": service_id,
            "service_type": DCPParser.SERVICE_TYPES.get(
                service_type, f"Unknown (0x{service_type:02X})"
            ),
            "service_type_raw": service_type,
            "transaction_id": xid,
            "response_delay": response_delay,
            "data_length": dcp_data_length,
            "blocks": blocks,
        }

    @staticmethod
    def _parse_block(option: int, suboption: int, data: bytes) -> dict[str, Any]:
        """Parse individual DCP block based on option/suboption.

        Args:
            option: DCP option code.
            suboption: DCP suboption code.
            data: Block data bytes.

        Returns:
            Parsed block data.
        """
        block: dict[str, Any] = {
            "option": DCPParser.OPTIONS.get(option, f"Unknown (0x{option:02X})"),
            "option_raw": option,
            "suboption": suboption,
            "length": len(data),
        }

        # Parse based on option type
        if option == 0x01:  # IP
            DCPParser._parse_ip_option(block, suboption, data)
        elif option == 0x02:  # Device Properties
            DCPParser._parse_device_properties(block, suboption, data)
        else:
            block["data_hex"] = data.hex()

        return block

    @staticmethod
    def _parse_ip_option(block: dict[str, Any], suboption: int, data: bytes) -> None:
        """Parse IP option suboptions.

        Args:
            block: Block dictionary to update.
            suboption: IP suboption code.
            data: Block data bytes.
        """
        block["suboption_name"] = DCPParser.IP_SUBOPTIONS.get(
            suboption, f"Unknown (0x{suboption:02X})"
        )

        if suboption == 0x01 and len(data) >= 6:  # MAC address
            block["mac_address"] = ":".join(f"{b:02x}" for b in data[:6])
        elif suboption == 0x02 and len(data) >= 12:  # IP parameter
            DCPParser._parse_ip_parameter(block, data)
        else:
            block["data_hex"] = data.hex()

    @staticmethod
    def _parse_ip_parameter(block: dict[str, Any], data: bytes) -> None:
        """Parse IP parameter suboption (IP, subnet, gateway).

        Args:
            block: Block dictionary to update.
            data: Block data bytes (must be >= 12 bytes).
        """
        block["ip_address"] = ".".join(str(b) for b in data[0:4])
        block["subnet_mask"] = ".".join(str(b) for b in data[4:8])
        block["gateway"] = ".".join(str(b) for b in data[8:12])

    @staticmethod
    def _parse_device_properties(block: dict[str, Any], suboption: int, data: bytes) -> None:
        """Parse Device Properties option.

        Args:
            block: Block dictionary to update.
            suboption: Device property suboption code.
            data: Block data bytes.
        """
        block["suboption_name"] = DCPParser.DEVICE_PROPS_SUBOPTIONS.get(
            suboption, f"Unknown (0x{suboption:02X})"
        )

        if suboption == 0x02:  # Name of Station
            DCPParser._parse_device_name(block, data)
        elif suboption == 0x03 and len(data) >= 4:  # Device ID
            DCPParser._parse_device_id(block, data)
        elif suboption == 0x04 and len(data) >= 2:  # Device Role
            DCPParser._parse_device_role(block, data)
        else:
            block["data_hex"] = data.hex()

    @staticmethod
    def _parse_device_name(block: dict[str, Any], data: bytes) -> None:
        """Parse device name."""
        try:
            block["device_name"] = data.decode("ascii").rstrip("\x00")
        except UnicodeDecodeError:
            block["device_name_hex"] = data.hex()

    @staticmethod
    def _parse_device_id(block: dict[str, Any], data: bytes) -> None:
        """Parse device ID (vendor + device)."""
        block["vendor_id"] = int.from_bytes(data[0:2], "big")
        block["device_id"] = int.from_bytes(data[2:4], "big")

    @staticmethod
    def _parse_device_role(block: dict[str, Any], data: bytes) -> None:
        """Parse device role bitmask."""
        device_role = int.from_bytes(data[0:2], "big")
        block["device_role"] = device_role

        roles = []
        if device_role & 0x01:
            roles.append("IO-Device")
        if device_role & 0x02:
            roles.append("IO-Controller")
        if device_role & 0x04:
            roles.append("IO-Multidevice")
        if device_role & 0x08:
            roles.append("IO-Supervisor")
        block["role_names"] = roles


__all__ = ["DCPParser"]
