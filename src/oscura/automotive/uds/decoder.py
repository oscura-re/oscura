"""UDS (Unified Diagnostic Services) protocol decoder per ISO 14229.

This module implements decoding for UDS diagnostic messages used in automotive ECUs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage

from oscura.automotive.uds.models import UDSNegativeResponse, UDSService

__all__ = ["UDSDecoder"]


# Service ID mappings per ISO 14229-1
_SERVICE_NAMES = {
    0x10: "DiagnosticSessionControl",
    0x11: "ECUReset",
    0x14: "ClearDiagnosticInformation",
    0x19: "ReadDTCInformation",
    0x22: "ReadDataByIdentifier",
    0x23: "ReadMemoryByAddress",
    0x27: "SecurityAccess",
    0x28: "CommunicationControl",
    0x2E: "WriteDataByIdentifier",
    0x2F: "InputOutputControlByIdentifier",
    0x31: "RoutineControl",
    0x34: "RequestDownload",
    0x35: "RequestUpload",
    0x36: "TransferData",
    0x37: "RequestTransferExit",
    0x3E: "TesterPresent",
    0x85: "ControlDTCSetting",
}

# Negative Response Codes per ISO 14229-1
_NRC_NAMES = {
    0x10: "generalReject",
    0x11: "serviceNotSupported",
    0x12: "subFunctionNotSupported",
    0x13: "incorrectMessageLengthOrInvalidFormat",
    0x14: "responseTooLong",
    0x21: "busyRepeatRequest",
    0x22: "conditionsNotCorrect",
    0x24: "requestSequenceError",
    0x25: "noResponseFromSubnetComponent",
    0x26: "failurePreventsExecutionOfRequestedAction",
    0x31: "requestOutOfRange",
    0x33: "securityAccessDenied",
    0x35: "invalidKey",
    0x36: "exceedNumberOfAttempts",
    0x37: "requiredTimeDelayNotExpired",
    0x70: "uploadDownloadNotAccepted",
    0x71: "transferDataSuspended",
    0x72: "generalProgrammingFailure",
    0x73: "wrongBlockSequenceCounter",
    0x78: "requestCorrectlyReceivedResponsePending",
    0x7E: "subFunctionNotSupportedInActiveSession",
    0x7F: "serviceNotSupportedInActiveSession",
}

# Services that typically have sub-functions
_SERVICES_WITH_SUBFUNCTIONS = {
    0x10,  # DiagnosticSessionControl
    0x11,  # ECUReset
    0x19,  # ReadDTCInformation
    0x27,  # SecurityAccess
    0x28,  # CommunicationControl
    0x2F,  # InputOutputControlByIdentifier
    0x31,  # RoutineControl
    0x3E,  # TesterPresent
    0x85,  # ControlDTCSetting
}


class UDSDecoder:
    """ISO 14229 UDS protocol decoder.

    This decoder analyzes CAN messages to identify and decode UDS diagnostic services.

    UDS message formats:
    - Request: [length/PCI, SID, sub-function?, data...]
    - Positive Response: [length/PCI, SID+0x40, data...]
    - Negative Response: [length/PCI, 0x7F, requested_SID, NRC]

    Note: The first byte in CAN data may be ISO-TP PCI (Protocol Control Information)
    or the message may be single-frame. This decoder handles both cases.
    """

    @staticmethod
    def is_uds_request(message: CANMessage) -> bool:
        """Check if CAN message is a UDS request.

        Args:
            message: CAN message to check.

        Returns:
            True if message appears to be a UDS request.
        """
        if len(message.data) < 2:
            return False

        # Handle ISO-TP single frame (0x0X where X is length)
        if message.data[0] <= 0x07:
            sid = message.data[1]
        # Direct UDS (no ISO-TP header)
        else:
            sid = message.data[0]

        return sid in _SERVICE_NAMES

    @staticmethod
    def is_uds_response(message: CANMessage) -> bool:
        """Check if CAN message is a UDS response (positive or negative).

        Args:
            message: CAN message to check.

        Returns:
            True if message appears to be a UDS response.
        """
        if len(message.data) < 2:
            return False

        # Handle ISO-TP single frame
        if message.data[0] <= 0x07:
            first_byte = message.data[1]
            # Negative response needs: PCI + 0x7F + Requested SID + NRC = 4 bytes minimum
            if first_byte == 0x7F:
                return len(message.data) >= 4
        else:
            first_byte = message.data[0]

        # Negative response (non-ISO-TP format)
        if first_byte == 0x7F:
            return len(message.data) >= 3

        # Positive response (SID + 0x40)
        # Response SIDs are in range 0x40-0x7F or 0xC0-0xFF
        if 0x40 <= first_byte < 0x80 or first_byte >= 0xC0:
            response_sid = first_byte - 0x40
            return response_sid in _SERVICE_NAMES

        return False

    @staticmethod
    def decode_service(message: CANMessage) -> UDSService | UDSNegativeResponse | None:
        """Decode UDS service from CAN message.

        Args:
            message: CAN message to decode.

        Returns:
            UDSService, UDSNegativeResponse, or None if not a valid UDS message.

        Example:
            >>> msg = CANMessage(id=0x7E0, data=bytes([0x02, 0x10, 0x01]))
            >>> service = UDSDecoder.decode_service(msg)
            >>> print(service.name if service else "Invalid")
        """
        if len(message.data) < 2:
            return None

        # Extract UDS payload from ISO-TP frame if needed
        data = UDSDecoder._extract_uds_payload(message.data)
        if not data:
            return None

        # Check for negative response
        if data[0] == 0x7F:
            return UDSDecoder._decode_negative_response(data)

        # Determine SID and request/response type
        sid_info = UDSDecoder._parse_sid_byte(data[0])
        if sid_info is None:
            return None

        sid, canonical_sid, is_request = sid_info
        service_name = _SERVICE_NAMES[canonical_sid]

        # Extract sub-function and payload
        sub_function, payload = UDSDecoder._extract_subfunction_and_payload(
            data, canonical_sid, is_request
        )

        return UDSService(
            sid=sid,
            name=service_name,
            request=is_request,
            sub_function=sub_function,
            data=payload,
        )

    @staticmethod
    def _extract_uds_payload(message_data: bytes) -> bytes:
        """Extract UDS payload from CAN message data.

        Handles ISO-TP single frame format (first byte ≤0x07 indicates length).

        Args:
            message_data: Raw CAN message data.

        Returns:
            UDS payload bytes (empty if invalid).
        """
        if message_data[0] <= 0x07:
            # ISO-TP single frame: [length, ...UDS data...]
            uds_length = message_data[0]
            if len(message_data) < 1 + uds_length:
                return b""
            return message_data[1 : 1 + uds_length]
        else:
            # Direct UDS: all bytes are UDS data
            return message_data

    @staticmethod
    def _decode_negative_response(data: bytes) -> UDSNegativeResponse | None:
        """Decode UDS negative response.

        Format: [0x7F, requested_SID, NRC]

        Args:
            data: UDS payload starting with 0x7F.

        Returns:
            UDSNegativeResponse or None if invalid format.
        """
        if len(data) < 3:
            return None

        requested_sid = data[1]
        nrc = data[2]
        nrc_name = _NRC_NAMES.get(nrc, f"unknownNRC_0x{nrc:02X}")

        return UDSNegativeResponse(
            requested_sid=requested_sid,
            nrc=nrc,
            nrc_name=nrc_name,
        )

    @staticmethod
    def _parse_sid_byte(first_byte: int) -> tuple[int, int, bool] | None:
        """Parse SID and request/response type from first UDS byte.

        Args:
            first_byte: First byte of UDS payload.

        Returns:
            Tuple of (actual_sid, canonical_sid, is_request) or None if unknown service.
            - actual_sid: The SID byte from message (0x50 for responses, 0x10 for requests)
            - canonical_sid: The canonical service ID for name lookup (always 0x10)
            - is_request: True if request, False if response

        Notes:
            - Response SIDs: 0x40-0x7F (request 0x00-0x3F + 0x40)
            - Response SIDs: 0xC0-0xFF (request 0x80-0xBF + 0x40)
            - Request SIDs: 0x00-0x3F, 0x80-0xBF
        """
        if 0x40 <= first_byte < 0x80:
            # Positive response to service 0x00-0x3F
            sid = first_byte  # Keep actual response SID (e.g., 0x50)
            canonical_sid = first_byte - 0x40  # Request SID for validation (e.g., 0x10)
            is_request = False
        elif first_byte >= 0xC0:
            # Positive response to service 0x80-0xBF
            sid = first_byte  # Keep actual response SID
            canonical_sid = first_byte - 0x40  # Request SID for validation
            is_request = False
        else:
            # Request (0x00-0x3F or 0x80-0xBF)
            sid = first_byte
            canonical_sid = first_byte
            is_request = True

        # Validate service is known (use canonical request SID)
        if canonical_sid not in _SERVICE_NAMES:
            return None

        return (sid, canonical_sid, is_request)

    @staticmethod
    def _extract_subfunction_and_payload(
        data: bytes, sid: int, is_request: bool
    ) -> tuple[int | None, bytes]:
        """Extract sub-function and payload from UDS service data.

        Args:
            data: UDS payload bytes.
            sid: Service ID.
            is_request: True if request, False if response.

        Returns:
            Tuple of (sub_function, payload_bytes).

        Notes:
            Some services echo sub-function in responses: 0x10, 0x11, 0x27, 0x28, 0x31, 0x3E, 0x85.
        """
        sub_function = None
        payload_offset = 1

        if sid not in _SERVICES_WITH_SUBFUNCTIONS:
            # No sub-function for this service
            payload = data[payload_offset:] if len(data) > payload_offset else b""
            return (sub_function, payload)

        # Extract sub-function if data contains it
        if is_request:
            if len(data) >= 2:
                # Mask off suppress positive response bit (0x80)
                sub_function = data[1] & 0x7F
                payload_offset = 2
        else:
            # Response may echo sub-function for certain services
            _SERVICES_WITH_ECHO = {0x10, 0x11, 0x27, 0x28, 0x31, 0x3E, 0x85}
            if sid in _SERVICES_WITH_ECHO and len(data) >= 2:
                sub_function = data[1] & 0x7F
                payload_offset = 2

        payload = data[payload_offset:] if len(data) > payload_offset else b""
        return (sub_function, payload)

    @staticmethod
    def get_service_name(sid: int) -> str:
        """Get service name from Service ID.

        Args:
            sid: Service ID (0x10-0xFF).

        Returns:
            Service name or "Unknown" if not recognized.
        """
        return _SERVICE_NAMES.get(sid, f"Unknown_0x{sid:02X}")

    @staticmethod
    def get_nrc_name(nrc: int) -> str:
        """Get negative response code name.

        Args:
            nrc: Negative Response Code (0x10-0xFF).

        Returns:
            NRC name or "unknown" if not recognized.
        """
        return _NRC_NAMES.get(nrc, f"unknownNRC_0x{nrc:02X}")
