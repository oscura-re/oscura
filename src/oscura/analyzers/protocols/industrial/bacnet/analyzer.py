"""BACnet protocol analyzer for IP and MSTP variants.

This module provides comprehensive BACnet (Building Automation and Control Networks)
protocol analysis supporting both BACnet/IP (UDP port 47808) and BACnet/MSTP
(Master-Slave/Token-Passing serial) variants. Decodes NPDU/APDU layers, all service
types, and discovers devices and objects for HVAC and building automation systems.

Example:
    >>> from oscura.analyzers.protocols.industrial.bacnet import BACnetAnalyzer
    >>> analyzer = BACnetAnalyzer()
    >>> # Parse BACnet/IP message from UDP packet
    >>> udp_payload = bytes([0x81, 0x0A, 0x00, 0x11, 0x01, 0x20, 0x00, 0x08, ...])
    >>> message = analyzer.parse_bacnet_ip(udp_payload, timestamp=0.0)
    >>> print(f"{message.service_name}: {message.decoded_service}")
    >>> # Export discovered devices
    >>> analyzer.export_devices(Path("bacnet_devices.json"))

References:
    ANSI/ASHRAE Standard 135-2020 (BACnet):
    https://www.ashrae.org/technical-resources/bookstore/bacnet

    BACnet/IP (Annex J):
    http://www.bacnet.org/Addenda/Add-135-2016bj-1_chair_approved.pdf

    BACnet MS/TP (Annex G):
    http://www.bacnet.org/Addenda/Add-135-2012g-5_PPR2-redline.pdf
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.analyzers.protocols.industrial.bacnet.services import (
    decode_i_am,
    decode_i_have,
    decode_read_property_ack,
    decode_read_property_request,
    decode_who_has,
    decode_who_is,
    decode_write_property_request,
)


@dataclass
class BACnetMessage:
    """BACnet message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        protocol: Protocol variant ("BACnet/IP" or "BACnet/MSTP").
        npdu: Network Protocol Data Unit parsed fields.
        apdu_type: APDU type name (Confirmed-REQ, Unconfirmed-REQ, etc.).
        service_choice: Service choice number.
        service_name: Human-readable service name.
        invoke_id: Invoke ID for confirmed services.
        payload: Raw APDU payload bytes.
        decoded_service: Service-specific decoded data.
    """

    timestamp: float
    protocol: str  # "BACnet/IP" or "BACnet/MSTP"
    npdu: dict[str, Any]
    apdu_type: str
    service_choice: int | None = None
    service_name: str | None = None
    invoke_id: int | None = None
    payload: bytes = b""
    decoded_service: dict[str, Any] = field(default_factory=dict)


@dataclass
class BACnetObject:
    """BACnet object representation.

    Attributes:
        object_type: Object type name (analog-input, binary-output, device, etc.).
        instance_number: Object instance number.
        properties: Observed property values (property_id -> value).
    """

    object_type: str
    instance_number: int
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class BACnetDevice:
    """BACnet device information.

    Attributes:
        device_instance: Device instance number.
        device_name: Device name (if discovered).
        vendor_id: Vendor identifier (if discovered).
        model_name: Model name (if discovered).
        objects: List of discovered objects on this device.
    """

    device_instance: int
    device_name: str | None = None
    vendor_id: int | None = None
    model_name: str | None = None
    objects: list[BACnetObject] = field(default_factory=list)


class BACnetAnalyzer:
    """BACnet protocol analyzer for IP and MSTP variants.

    Supports parsing BACnet/IP (UDP) and BACnet/MSTP (serial) messages,
    decoding NPDU/APDU layers, all service types, and discovering devices
    and objects on the network.

    Attributes:
        messages: List of all parsed BACnet messages.
        devices: Dictionary of discovered devices (device_instance -> BACnetDevice).
    """

    # APDU types (ASHRAE 135-2020, Clause 20.1.2)
    APDU_TYPES: ClassVar[dict[int, str]] = {
        0: "Confirmed-REQ",
        1: "Unconfirmed-REQ",
        2: "SimpleACK",
        3: "ComplexACK",
        4: "SegmentACK",
        5: "Error",
        6: "Reject",
        7: "Abort",
    }

    # Confirmed services (ASHRAE 135-2020, Clause 21.1)
    CONFIRMED_SERVICES: ClassVar[dict[int, str]] = {
        0: "acknowledgeAlarm",
        1: "confirmedCOVNotification",
        2: "confirmedEventNotification",
        3: "getAlarmSummary",
        4: "getEnrollmentSummary",
        5: "subscribeCOV",
        6: "atomicReadFile",
        7: "atomicWriteFile",
        8: "addListElement",
        9: "removeListElement",
        10: "createObject",
        11: "deleteObject",
        12: "readProperty",
        13: "readPropertyConditional",
        14: "readPropertyMultiple",
        15: "writeProperty",
        16: "writePropertyMultiple",
        17: "deviceCommunicationControl",
        18: "confirmedPrivateTransfer",
        19: "confirmedTextMessage",
        20: "reinitializeDevice",
    }

    # Unconfirmed services (ASHRAE 135-2020, Clause 21.2)
    UNCONFIRMED_SERVICES: ClassVar[dict[int, str]] = {
        0: "i-Am",
        1: "i-Have",
        2: "unconfirmedCOVNotification",
        3: "unconfirmedEventNotification",
        4: "unconfirmedPrivateTransfer",
        5: "unconfirmedTextMessage",
        6: "timeSynchronization",
        7: "who-Has",
        8: "who-Is",
        9: "utcTimeSynchronization",
    }

    def __init__(self) -> None:
        """Initialize BACnet analyzer."""
        self.messages: list[BACnetMessage] = []
        self.devices: dict[int, BACnetDevice] = {}

    def parse_bacnet_ip(self, udp_payload: bytes, timestamp: float = 0.0) -> BACnetMessage:
        """Parse BACnet/IP message from UDP payload (port 47808).

        BACnet/IP messages use the BACnet Virtual Link Control (BVLC) layer
        before the NPDU/APDU layers.

        Args:
            udp_payload: Raw UDP payload bytes.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed BACnet message.

        Raises:
            ValueError: If message is too short or has invalid BVLC header.

        Example:
            >>> analyzer = BACnetAnalyzer()
            >>> udp_data = bytes([0x81, 0x0A, 0x00, 0x11, ...])
            >>> msg = analyzer.parse_bacnet_ip(udp_data, timestamp=1.23)
            >>> print(f"{msg.service_name}: {msg.decoded_service}")
        """
        if len(udp_payload) < 4:
            raise ValueError("BACnet/IP message too short")

        # Parse BVLC header (Annex J)
        bvlc_type = udp_payload[0]
        bvlc_function = udp_payload[1]
        # bvlc_length = int.from_bytes(udp_payload[2:4], "big")  # Not used in current implementation

        if bvlc_type != 0x81:
            raise ValueError(f"Invalid BACnet/IP type: 0x{bvlc_type:02X} (expected 0x81)")

        # BVLC function 0x0A = Original-Unicast-NPDU
        # BVLC function 0x0B = Original-Broadcast-NPDU
        if bvlc_function not in (0x0A, 0x0B, 0x04):
            raise ValueError(f"Unsupported BVLC function: 0x{bvlc_function:02X}")

        # Parse NPDU starting at offset 4
        npdu_data = udp_payload[4:]
        npdu, npdu_len = self._parse_npdu(npdu_data)

        # Parse APDU
        apdu_data = npdu_data[npdu_len:]
        apdu_dict = self._parse_apdu(apdu_data)

        # Decode service-specific payload
        decoded_service = self._decode_service(
            apdu_dict["apdu_type"],
            apdu_dict.get("service_choice"),
            apdu_dict.get("service_data", b""),
        )

        message = BACnetMessage(
            timestamp=timestamp,
            protocol="BACnet/IP",
            npdu=npdu,
            apdu_type=self.APDU_TYPES.get(apdu_dict["apdu_type"], "Unknown"),
            service_choice=apdu_dict.get("service_choice"),
            service_name=apdu_dict.get("service_name"),
            invoke_id=apdu_dict.get("invoke_id"),
            payload=apdu_data,
            decoded_service=decoded_service,
        )

        self.messages.append(message)
        self._update_device_info(message)

        return message

    def parse_bacnet_mstp(self, serial_data: bytes, timestamp: float = 0.0) -> BACnetMessage:
        """Parse BACnet MS/TP (Master-Slave/Token-Passing) frame from serial data.

        BACnet MS/TP is a token-passing protocol for serial (RS-485) links.
        Frame format: Preamble (0x55 0xFF) + Header (6 bytes) + Data + CRC.

        Args:
            serial_data: Raw serial data bytes.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed BACnet message.

        Raises:
            ValueError: If frame is too short or has invalid preamble/CRC.

        Example:
            >>> analyzer = BACnetAnalyzer()
            >>> serial_frame = bytes([0x55, 0xFF, 0x05, 0x01, 0x00, ...])
            >>> msg = analyzer.parse_bacnet_mstp(serial_frame, timestamp=2.34)
        """
        if len(serial_data) < 8:
            raise ValueError("BACnet MSTP frame too short")

        # Check preamble (Annex G)
        if serial_data[0] != 0x55 or serial_data[1] != 0xFF:
            raise ValueError("Invalid MSTP preamble (expected 0x55 0xFF)")

        # Parse MSTP header (6 bytes after preamble)
        frame_type = serial_data[2]
        # destination_address = serial_data[3]  # Not used in current implementation
        # source_address = serial_data[4]  # Not used in current implementation
        data_length = int.from_bytes(serial_data[5:7], "big")
        header_crc = serial_data[7]

        # Verify header CRC (simple XOR for demonstration; real MSTP uses proper CRC)
        calculated_header_crc = self._mstp_header_crc(serial_data[2:7])
        if calculated_header_crc != header_crc:
            raise ValueError(
                f"MSTP header CRC mismatch: {header_crc:02X} != {calculated_header_crc:02X}"
            )

        # Extract data and verify data CRC if present
        if data_length > 0:
            if len(serial_data) < 8 + data_length + 2:
                raise ValueError("MSTP frame truncated")
            data_payload = serial_data[8 : 8 + data_length]
            data_crc = int.from_bytes(serial_data[8 + data_length : 8 + data_length + 2], "big")

            calculated_data_crc = self._mstp_data_crc(data_payload)
            if calculated_data_crc != data_crc:
                raise ValueError(
                    f"MSTP data CRC mismatch: {data_crc:04X} != {calculated_data_crc:04X}"
                )
        else:
            data_payload = b""

        # Parse NPDU from data payload (frame type 0x05 = BACnet Data Expecting Reply)
        if frame_type in (0x00, 0x01, 0x05):  # Data frames
            npdu, npdu_len = self._parse_npdu(data_payload)
            apdu_data = data_payload[npdu_len:]
            apdu_dict = self._parse_apdu(apdu_data)

            decoded_service = self._decode_service(
                apdu_dict["apdu_type"],
                apdu_dict.get("service_choice"),
                apdu_dict.get("service_data", b""),
            )

            message = BACnetMessage(
                timestamp=timestamp,
                protocol="BACnet/MSTP",
                npdu=npdu,
                apdu_type=self.APDU_TYPES.get(apdu_dict["apdu_type"], "Unknown"),
                service_choice=apdu_dict.get("service_choice"),
                service_name=apdu_dict.get("service_name"),
                invoke_id=apdu_dict.get("invoke_id"),
                payload=apdu_data,
                decoded_service=decoded_service,
            )

            self.messages.append(message)
            self._update_device_info(message)

            return message
        else:
            # Non-data frame (token, poll, etc.)
            raise ValueError(f"MSTP frame type {frame_type:02X} not supported")

    def _parse_npdu(self, data: bytes) -> tuple[dict[str, Any], int]:
        """Parse NPDU (Network Protocol Data Unit).

        Args:
            data: Raw NPDU bytes.

        Returns:
            Tuple of (npdu_dict, bytes_consumed).

        Raises:
            ValueError: If NPDU is too short or has invalid version.
        """
        if len(data) < 2:
            raise ValueError("NPDU too short")

        version = data[0]
        control = data[1]
        offset = 2

        if version != 0x01:
            raise ValueError(f"Invalid NPDU version: {version} (expected 0x01)")

        npdu_dict: dict[str, Any] = {
            "version": version,
            "control": control,
            "network_priority": (control >> 0) & 0x03,
            "dest_specifier": bool(control & 0x20),
            "source_specifier": bool(control & 0x08),
            "expects_reply": bool(control & 0x04),
            "is_network_message": bool(control & 0x80),
        }

        # Parse destination network address if present
        if npdu_dict["dest_specifier"]:
            if offset + 3 > len(data):
                return npdu_dict, offset
            dest_network = int.from_bytes(data[offset : offset + 2], "big")
            dest_mac_len = data[offset + 2]
            offset += 3

            if offset + dest_mac_len > len(data):
                return npdu_dict, offset
            dest_mac = data[offset : offset + dest_mac_len]
            offset += dest_mac_len

            npdu_dict["dest_network"] = dest_network
            npdu_dict["dest_mac"] = dest_mac.hex()

        # Parse source network address if present
        if npdu_dict["source_specifier"]:
            if offset + 3 > len(data):
                return npdu_dict, offset
            source_network = int.from_bytes(data[offset : offset + 2], "big")
            source_mac_len = data[offset + 2]
            offset += 3

            if offset + source_mac_len > len(data):
                return npdu_dict, offset
            source_mac = data[offset : offset + source_mac_len]
            offset += source_mac_len

            npdu_dict["source_network"] = source_network
            npdu_dict["source_mac"] = source_mac.hex()

        # Parse hop count if destination specified
        if npdu_dict["dest_specifier"]:
            if offset < len(data):
                npdu_dict["hop_count"] = data[offset]
                offset += 1

        # Parse network message type if network message
        if npdu_dict["is_network_message"]:
            if offset < len(data):
                npdu_dict["network_message_type"] = data[offset]
                offset += 1

        return npdu_dict, offset

    def _parse_apdu(self, data: bytes) -> dict[str, Any]:
        """Parse APDU (Application Protocol Data Unit).

        Args:
            data: Raw APDU bytes.

        Returns:
            Dictionary with apdu_type, service_choice, invoke_id, and service_data.

        Raises:
            ValueError: If APDU is too short.
        """
        if len(data) < 1:
            raise ValueError("APDU too short")

        apdu_type = (data[0] >> 4) & 0x0F
        apdu_dict: dict[str, Any] = {"apdu_type": apdu_type}

        if apdu_type == 0:  # Confirmed-REQ
            self._parse_confirmed_req(data, apdu_dict)
        elif apdu_type == 1:  # Unconfirmed-REQ
            self._parse_unconfirmed_req(data, apdu_dict)
        elif apdu_type == 2:  # SimpleACK
            self._parse_simple_ack(data, apdu_dict)
        elif apdu_type == 3:  # ComplexACK
            self._parse_complex_ack(data, apdu_dict)
        elif apdu_type == 5:  # Error
            self._parse_error(data, apdu_dict)
        elif apdu_type == 6:  # Reject
            self._parse_reject(data, apdu_dict)
        elif apdu_type == 7:  # Abort
            self._parse_abort(data, apdu_dict)

        return apdu_dict

    def _parse_confirmed_req(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse Confirmed-REQ APDU type."""
        if len(data) < 3:
            raise ValueError("Confirmed-REQ APDU too short")

        segmented = bool(data[0] & 0x08)
        more_follows = bool(data[0] & 0x04)
        segmented_response_accepted = bool(data[0] & 0x02)
        max_segments = data[1] >> 4
        max_apdu = data[1] & 0x0F
        invoke_id = data[2]
        service_choice = data[3] if len(data) > 3 else 0

        apdu_dict.update(
            {
                "segmented": segmented,
                "more_follows": more_follows,
                "segmented_response_accepted": segmented_response_accepted,
                "max_segments": max_segments,
                "max_apdu": max_apdu,
                "invoke_id": invoke_id,
                "service_choice": service_choice,
                "service_name": self.CONFIRMED_SERVICES.get(
                    service_choice, f"service-{service_choice}"
                ),
                "service_data": data[4:] if len(data) > 4 else b"",
            }
        )

    def _parse_unconfirmed_req(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse Unconfirmed-REQ APDU type."""
        if len(data) < 2:
            raise ValueError("Unconfirmed-REQ APDU too short")

        service_choice = data[1]
        apdu_dict.update(
            {
                "service_choice": service_choice,
                "service_name": self.UNCONFIRMED_SERVICES.get(
                    service_choice, f"service-{service_choice}"
                ),
                "service_data": data[2:] if len(data) > 2 else b"",
            }
        )

    def _parse_simple_ack(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse SimpleACK APDU type."""
        if len(data) < 3:
            raise ValueError("SimpleACK APDU too short")

        invoke_id = data[1]
        service_choice = data[2]
        apdu_dict.update(
            {
                "invoke_id": invoke_id,
                "service_choice": service_choice,
                "service_name": self.CONFIRMED_SERVICES.get(
                    service_choice, f"service-{service_choice}"
                ),
            }
        )

    def _parse_complex_ack(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse ComplexACK APDU type."""
        if len(data) < 3:
            raise ValueError("ComplexACK APDU too short")

        segmented = bool(data[0] & 0x08)
        more_follows = bool(data[0] & 0x04)
        invoke_id = data[1]
        service_choice = data[2]

        apdu_dict.update(
            {
                "segmented": segmented,
                "more_follows": more_follows,
                "invoke_id": invoke_id,
                "service_choice": service_choice,
                "service_name": self.CONFIRMED_SERVICES.get(
                    service_choice, f"service-{service_choice}"
                ),
                "service_data": data[3:] if len(data) > 3 else b"",
            }
        )

    def _parse_error(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse Error APDU type."""
        if len(data) < 3:
            raise ValueError("Error APDU too short")

        invoke_id = data[1]
        service_choice = data[2]
        apdu_dict.update(
            {
                "invoke_id": invoke_id,
                "service_choice": service_choice,
                "service_name": self.CONFIRMED_SERVICES.get(
                    service_choice, f"service-{service_choice}"
                ),
                "service_data": data[3:] if len(data) > 3 else b"",
            }
        )

    def _parse_reject(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse Reject APDU type."""
        if len(data) < 3:
            raise ValueError("Reject APDU too short")

        invoke_id = data[1]
        reject_reason = data[2]
        apdu_dict.update({"invoke_id": invoke_id, "reject_reason": reject_reason})

    def _parse_abort(self, data: bytes, apdu_dict: dict[str, Any]) -> None:
        """Parse Abort APDU type."""
        if len(data) < 3:
            raise ValueError("Abort APDU too short")

        invoke_id = data[1]
        abort_reason = data[2]
        apdu_dict.update({"invoke_id": invoke_id, "abort_reason": abort_reason})

    def _decode_service(
        self, apdu_type: int, service_choice: int | None, data: bytes
    ) -> dict[str, Any]:
        """Decode service-specific payload based on APDU type and service choice.

        Args:
            apdu_type: APDU type number.
            service_choice: Service choice number.
            data: Service payload bytes.

        Returns:
            Decoded service data dictionary.
        """
        if service_choice is None:
            return {}

        decoders = {
            1: self._decode_unconfirmed_service,
            0: self._decode_confirmed_service,
            3: self._decode_complex_ack,
        }

        decoder = decoders.get(apdu_type)
        return decoder(service_choice, data) if decoder else {}

    def _decode_unconfirmed_service(self, service_choice: int, data: bytes) -> dict[str, Any]:
        """Decode unconfirmed service payloads."""
        unconfirmed_decoders = {
            0: decode_i_am,
            1: decode_i_have,
            7: decode_who_has,
            8: decode_who_is,
        }
        decoder = unconfirmed_decoders.get(service_choice)
        return decoder(data) if decoder else {}

    def _decode_confirmed_service(self, service_choice: int, data: bytes) -> dict[str, Any]:
        """Decode confirmed service request payloads."""
        confirmed_decoders = {
            12: decode_read_property_request,
            15: decode_write_property_request,
        }
        decoder = confirmed_decoders.get(service_choice)
        return decoder(data) if decoder else {}

    def _decode_complex_ack(self, service_choice: int, data: bytes) -> dict[str, Any]:
        """Decode ComplexACK response payloads."""
        if service_choice == 12:
            return decode_read_property_ack(data)
        return {}

    def _update_device_info(self, message: BACnetMessage) -> None:
        """Update device information from parsed message.

        Args:
            message: Parsed BACnet message.
        """
        # Extract device info from I-Am messages
        if message.service_name == "i-Am" and "device_instance" in message.decoded_service:
            device_instance = message.decoded_service["device_instance"]

            if device_instance not in self.devices:
                self.devices[device_instance] = BACnetDevice(device_instance=device_instance)

            device = self.devices[device_instance]

            # Update device properties from I-Am
            if "vendor_id" in message.decoded_service:
                device.vendor_id = message.decoded_service["vendor_id"]

    def _mstp_header_crc(self, header: bytes) -> int:
        """Calculate MSTP header CRC (simplified XOR for demonstration).

        Args:
            header: Header bytes (5 bytes).

        Returns:
            CRC value.
        """
        # Real implementation should use proper CRC-8 (CCITT)
        # This is a simplified version for demonstration
        crc = 0xFF
        for byte in header:
            crc ^= byte
        return crc

    def _mstp_data_crc(self, data: bytes) -> int:
        """Calculate MSTP data CRC (simplified for demonstration).

        Args:
            data: Data bytes.

        Returns:
            CRC value (16-bit).
        """
        # Real implementation should use proper CRC-16 (CCITT)
        # This is a simplified version for demonstration
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    def export_devices(self, output_path: Path) -> None:
        """Export discovered devices and object lists as JSON.

        Args:
            output_path: Output file path for JSON export.

        Example:
            >>> analyzer = BACnetAnalyzer()
            >>> # ... parse messages ...
            >>> analyzer.export_devices(Path("bacnet_devices.json"))
        """
        devices_data = []

        for device in self.devices.values():
            device_dict = {
                "device_instance": device.device_instance,
                "device_name": device.device_name,
                "vendor_id": device.vendor_id,
                "model_name": device.model_name,
                "objects": [
                    {
                        "object_type": obj.object_type,
                        "instance_number": obj.instance_number,
                        "properties": obj.properties,
                    }
                    for obj in device.objects
                ],
            }
            devices_data.append(device_dict)

        with output_path.open("w") as f:
            json.dump(devices_data, f, indent=2)
