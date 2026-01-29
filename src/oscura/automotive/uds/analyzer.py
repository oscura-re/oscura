"""UDS (Unified Diagnostic Services) protocol analyzer per ISO 14229.

This module provides comprehensive UDS protocol analysis for automotive diagnostics,
supporting all standard UDS services, diagnostic sessions, security access, DTC parsing,
and ECU capability discovery.

Example:
    >>> from oscura.automotive.uds.analyzer import UDSAnalyzer
    >>> analyzer = UDSAnalyzer()
    >>> msg = analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0)
    >>> print(msg.service_name)
    DiagnosticSessionControl
    >>> analyzer.export_session_flows(Path("session_flows.json"))

References:
    ISO 14229-1:2020 - UDS specification
    ISO 14229-2:2013 - Session layer services
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "UDSECU",
    "UDSAnalyzer",
    "UDSMessage",
]


@dataclass
class UDSMessage:
    """UDS message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        service_id: Service ID (0x10-0xFF).
        service_name: Human-readable service name.
        is_response: True if response, False if request.
        sub_function: Sub-function byte (if applicable).
        data: Service data payload.
        negative_response_code: NRC for negative responses.
        decoded: Service-specific decoded fields.
    """

    timestamp: float
    service_id: int
    service_name: str
    is_response: bool
    sub_function: int | None = None
    data: bytes = b""
    negative_response_code: int | None = None
    decoded: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Human-readable representation."""
        msg_type = "Response" if self.is_response else "Request"
        nrc = f" NRC=0x{self.negative_response_code:02X}" if self.negative_response_code else ""
        subfunc = f" sub=0x{self.sub_function:02X}" if self.sub_function is not None else ""
        return f"UDSMessage(0x{self.service_id:02X} {self.service_name} [{msg_type}]{subfunc}{nrc})"


@dataclass
class UDSECU:
    """UDS ECU information.

    Attributes:
        ecu_id: ECU identifier.
        supported_services: Set of supported service IDs.
        current_session: Current diagnostic session type.
        security_level: Current security access level (0 = locked).
        dtcs: List of Diagnostic Trouble Codes.
        data_identifiers: Data identifier values.
    """

    ecu_id: str
    supported_services: set[int] = field(default_factory=set)
    current_session: int = 0x01  # Default session
    security_level: int = 0  # Locked
    dtcs: list[dict[str, Any]] = field(default_factory=list)
    data_identifiers: dict[int, bytes] = field(default_factory=dict)


class UDSAnalyzer:
    """UDS (Unified Diagnostic Services) protocol analyzer.

    Supports comprehensive UDS protocol analysis including:
    - All standard UDS services (0x10-0x3E, 0x83-0x87)
    - Positive and negative responses
    - Diagnostic session management
    - Security access (seed/key exchange)
    - DTC parsing and analysis
    - Data identifier read/write
    - Routine control
    - Session flow export

    Example:
        >>> analyzer = UDSAnalyzer()
        >>> # Parse a diagnostic session control request
        >>> msg = analyzer.parse_message(
        ...     bytes([0x10, 0x03]),
        ...     timestamp=1.0,
        ...     ecu_id="ECU1"
        ... )
        >>> print(msg.service_name)
        DiagnosticSessionControl
        >>> print(msg.decoded["session_type"])
        ExtendedDiagnosticSession
    """

    # Service IDs per ISO 14229-1
    SERVICES: ClassVar[dict[int, str]] = {
        0x10: "DiagnosticSessionControl",
        0x11: "ECUReset",
        0x14: "ClearDiagnosticInformation",
        0x19: "ReadDTCInformation",
        0x22: "ReadDataByIdentifier",
        0x23: "ReadMemoryByAddress",
        0x24: "ReadScalingDataByIdentifier",
        0x27: "SecurityAccess",
        0x28: "CommunicationControl",
        0x2A: "ReadDataByPeriodicIdentifier",
        0x2C: "DynamicallyDefineDataIdentifier",
        0x2E: "WriteDataByIdentifier",
        0x2F: "InputOutputControlByIdentifier",
        0x31: "RoutineControl",
        0x34: "RequestDownload",
        0x35: "RequestUpload",
        0x36: "TransferData",
        0x37: "RequestTransferExit",
        0x38: "RequestFileTransfer",
        0x3D: "WriteMemoryByAddress",
        0x3E: "TesterPresent",
        0x83: "AccessTimingParameter",
        0x84: "SecuredDataTransmission",
        0x85: "ControlDTCSetting",
        0x86: "ResponseOnEvent",
        0x87: "LinkControl",
    }

    # Diagnostic sessions
    DIAGNOSTIC_SESSIONS: ClassVar[dict[int, str]] = {
        0x01: "DefaultSession",
        0x02: "ProgrammingSession",
        0x03: "ExtendedDiagnosticSession",
        0x04: "SafetySystemDiagnosticSession",
    }

    # Negative response codes per ISO 14229-1
    NEGATIVE_RESPONSE_CODES: ClassVar[dict[int, str]] = {
        0x10: "GeneralReject",
        0x11: "ServiceNotSupported",
        0x12: "SubFunctionNotSupported",
        0x13: "IncorrectMessageLengthOrInvalidFormat",
        0x14: "ResponseTooLong",
        0x21: "BusyRepeatRequest",
        0x22: "ConditionsNotCorrect",
        0x24: "RequestSequenceError",
        0x25: "NoResponseFromSubnetComponent",
        0x26: "FailurePreventsExecutionOfRequestedAction",
        0x31: "RequestOutOfRange",
        0x33: "SecurityAccessDenied",
        0x35: "InvalidKey",
        0x36: "ExceedNumberOfAttempts",
        0x37: "RequiredTimeDelayNotExpired",
        0x70: "UploadDownloadNotAccepted",
        0x71: "TransferDataSuspended",
        0x72: "GeneralProgrammingFailure",
        0x73: "WrongBlockSequenceCounter",
        0x78: "RequestCorrectlyReceived-ResponsePending",
        0x7E: "SubFunctionNotSupportedInActiveSession",
        0x7F: "ServiceNotSupportedInActiveSession",
    }

    def __init__(self) -> None:
        """Initialize UDS analyzer."""
        self.messages: list[UDSMessage] = []
        self.ecus: dict[str, UDSECU] = {}

    def parse_message(
        self, data: bytes, timestamp: float = 0.0, ecu_id: str = "ECU1"
    ) -> UDSMessage:
        """Parse UDS message (CAN/DoIP payload).

        UDS Message Format:
        - Service ID (1 byte) - or 0x7F for negative response
        - For negative response:
          - Failed Service ID (1 byte)
          - Negative Response Code (1 byte)
        - For positive response:
          - Service ID + 0x40
          - Service-specific data
        - For request:
          - Service ID
          - Sub-function (optional, 1 byte)
          - Service-specific data

        Args:
            data: Raw UDS message data.
            timestamp: Message timestamp in seconds.
            ecu_id: ECU identifier.

        Returns:
            Parsed UDSMessage object.

        Raises:
            ValueError: If message is invalid or empty.

        Example:
            >>> analyzer = UDSAnalyzer()
            >>> msg = analyzer.parse_message(bytes([0x10, 0x03]), timestamp=1.0)
            >>> print(msg.service_name)
            DiagnosticSessionControl
        """
        if len(data) == 0:
            raise ValueError("UDS message is empty")

        # Ensure ECU exists
        if ecu_id not in self.ecus:
            self.ecus[ecu_id] = UDSECU(ecu_id=ecu_id)

        sid = data[0]

        # Check for negative response
        if sid == 0x7F:
            if len(data) < 3:
                raise ValueError("Negative response too short")
            failed_sid = data[1]
            nrc = data[2]

            decoded = {"nrc_name": self.NEGATIVE_RESPONSE_CODES.get(nrc, "Unknown")}

            msg = UDSMessage(
                timestamp=timestamp,
                service_id=failed_sid,
                service_name=self.SERVICES.get(failed_sid, f"Unknown (0x{failed_sid:02X})"),
                is_response=True,
                negative_response_code=nrc,
                data=data[3:],
                decoded=decoded,
            )
        else:
            # Check for positive response (SID + 0x40)
            is_response = bool(sid & 0x40)
            actual_sid = sid & 0xBF if is_response else sid

            # Parse sub-function and data
            service_data = data[1:]

            decoded = self._decode_service(actual_sid, service_data, is_response)
            sub_function_val = decoded.get("sub_function")
            # Ensure sub_function is int or None (mypy strict)
            sub_function: int | None = (
                int(sub_function_val) if isinstance(sub_function_val, int) else None
            )

            msg = UDSMessage(
                timestamp=timestamp,
                service_id=actual_sid,
                service_name=self.SERVICES.get(actual_sid, f"Unknown (0x{actual_sid:02X})"),
                is_response=is_response,
                sub_function=sub_function,
                data=service_data,
                decoded=decoded,
            )

            # Update ECU state
            self._update_ecu_state(ecu_id, msg)

        self.messages.append(msg)
        return msg

    def _decode_service(self, service_id: int, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode service-specific data.

        Args:
            service_id: Service ID.
            data: Service payload data.
            is_response: True if response message.

        Returns:
            Dictionary of decoded fields.
        """
        decoders = {
            0x10: self._decode_diagnostic_session_control,
            0x11: self._decode_ecu_reset,
            0x19: self._decode_read_dtc,
            0x22: self._decode_read_data_by_id,
            0x27: self._decode_security_access,
            0x2E: self._decode_write_data_by_id,
            0x31: self._decode_routine_control,
            0x3E: self._decode_tester_present,
        }

        decoder = decoders.get(service_id)
        if decoder:
            return decoder(data, is_response)

        return {}

    def _decode_diagnostic_session_control(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode DiagnosticSessionControl (0x10).

        Request: [sub-function]
        Response: [sub-function, P2_server_max (2 bytes), P2*_server_max (2 bytes)]

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) == 0:
            return {}

        sub_function = data[0] & 0x7F
        suppress_response = bool(data[0] & 0x80)

        result = {
            "sub_function": sub_function,
            "suppress_positive_response": suppress_response,
            "session_type": self.DIAGNOSTIC_SESSIONS.get(sub_function, f"0x{sub_function:02X}"),
        }

        if is_response and len(data) >= 5:
            # P2_server_max and P2*_server_max in milliseconds
            p2_server_max = int.from_bytes(data[1:3], "big")
            p2_star_server_max = int.from_bytes(data[3:5], "big")
            result["p2_server_max_ms"] = p2_server_max
            result["p2_star_server_max_ms"] = p2_star_server_max

        return result

    def _decode_ecu_reset(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode ECUReset (0x11).

        Request: [sub-function]
        Response: [sub-function, power_down_time? (1 byte)]

        Sub-functions:
        - 0x01: hardReset
        - 0x02: keyOffOnReset
        - 0x03: softReset
        - 0x04: enableRapidPowerShutDown
        - 0x05: disableRapidPowerShutDown

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) == 0:
            return {}

        sub_function = data[0] & 0x7F
        suppress_response = bool(data[0] & 0x80)

        reset_types = {
            0x01: "hardReset",
            0x02: "keyOffOnReset",
            0x03: "softReset",
            0x04: "enableRapidPowerShutDown",
            0x05: "disableRapidPowerShutDown",
        }

        result = {
            "sub_function": sub_function,
            "suppress_positive_response": suppress_response,
            "reset_type": reset_types.get(sub_function, f"0x{sub_function:02X}"),
        }

        if is_response and len(data) >= 2:
            power_down_time = data[1]
            result["power_down_time_s"] = power_down_time

        return result

    def _decode_read_dtc(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode ReadDTCInformation (0x19).

        Sub-functions:
        - 0x01: reportNumberOfDTCByStatusMask
        - 0x02: reportDTCByStatusMask
        - 0x04: reportDTCSnapshotIdentification
        - 0x06: reportDTCExtDataRecordByDTCNumber

        Response format for 0x02:
        - [sub-function, availability_mask, dtc1_high, dtc1_mid, dtc1_low, status1, ...]

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) == 0:
            return {}

        sub_function = data[0]
        result = {"sub_function": sub_function}

        if sub_function == 0x02 and is_response and len(data) >= 2:
            # Parse DTC list
            dtcs = []
            offset = 2  # Skip sub-function echo and availability mask

            while offset + 4 <= len(data):
                dtc_bytes = data[offset : offset + 3]
                status = data[offset + 3]

                # DTC format: 3 bytes (6 hex digits)
                dtc_value = int.from_bytes(dtc_bytes, "big")
                dtc_string = f"{dtc_value:06X}"

                dtcs.append(
                    {
                        "dtc": dtc_string,
                        "status": status,
                        "test_failed": bool(status & 0x01),
                        "test_failed_this_operation_cycle": bool(status & 0x02),
                        "pending": bool(status & 0x04),
                        "confirmed": bool(status & 0x08),
                        "test_not_completed_since_last_clear": bool(status & 0x10),
                        "test_failed_since_last_clear": bool(status & 0x20),
                        "test_not_completed_this_operation_cycle": bool(status & 0x40),
                        "warning_indicator_requested": bool(status & 0x80),
                    }
                )

                offset += 4

            result["dtcs"] = dtcs  # type: ignore[assignment]
            result["dtc_count"] = len(dtcs)
            if len(data) >= 2:
                result["availability_mask"] = data[1]

        return result

    def _decode_read_data_by_id(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode ReadDataByIdentifier (0x22).

        Request: [did1_high, did1_low, did2_high, did2_low, ...]
        Response: [did1_high, did1_low, data1..., did2_high, did2_low, data2..., ...]

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) < 2:
            return {}

        result: dict[str, Any] = {}

        if not is_response:
            # Parse requested DIDs
            dids = []
            offset = 0
            while offset + 2 <= len(data):
                did = int.from_bytes(data[offset : offset + 2], "big")
                dids.append(did)
                offset += 2
            result["requested_dids"] = dids
        else:
            # Parse response DID and data
            # Note: Without knowing DID data lengths, we can only parse first DID
            did = int.from_bytes(data[0:2], "big")
            did_data = data[2:]
            result["did"] = did
            result["did_data"] = did_data.hex()

        return result

    def _decode_security_access(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode SecurityAccess (0x27) - seed/key exchange.

        Request:
        - Sub-function (1 byte) - 0x01 requestSeed, 0x02 sendKey, etc.
        - Data (variable) - empty for seed request, key for sendKey

        Response:
        - Sub-function (1 byte)
        - Seed/Key data (variable)

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.

        Example:
            >>> analyzer = UDSAnalyzer()
            >>> # Request seed for level 1
            >>> msg = analyzer.parse_message(bytes([0x27, 0x01]), timestamp=1.0)
            >>> print(msg.decoded["access_type"])
            requestSeed
            >>> print(msg.decoded["security_level"])
            1
        """
        if len(data) == 0:
            return {}

        sub_function = data[0] & 0x7F  # Mask suppress positive response bit
        suppress_response = bool(data[0] & 0x80)
        payload = data[1:]

        result = {
            "sub_function": sub_function,
            "suppress_positive_response": suppress_response,
        }

        if sub_function % 2 == 1:  # Odd = requestSeed
            result["access_type"] = "requestSeed"  # type: ignore[assignment]
            result["security_level"] = (sub_function + 1) // 2
            if is_response and len(payload) > 0:
                result["seed"] = payload.hex()  # type: ignore[assignment]
        else:  # Even = sendKey
            result["access_type"] = "sendKey"  # type: ignore[assignment]
            result["security_level"] = sub_function // 2
            if not is_response and len(payload) > 0:
                result["key"] = payload.hex()  # type: ignore[assignment]

        return result

    def _decode_write_data_by_id(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode WriteDataByIdentifier (0x2E).

        Request: [did_high, did_low, data...]
        Response: [did_high, did_low]

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) < 2:
            return {}

        did = int.from_bytes(data[0:2], "big")
        result = {"did": did}

        if not is_response and len(data) > 2:
            result["did_data"] = data[2:].hex()  # type: ignore[assignment]

        return result

    def _decode_routine_control(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode RoutineControl (0x31).

        Request: [sub-function, routine_id_high, routine_id_low, routine_option...]
        Response: [sub-function, routine_id_high, routine_id_low, status_record...]

        Sub-functions:
        - 0x01: startRoutine
        - 0x02: stopRoutine
        - 0x03: requestRoutineResults

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) < 3:
            return {}

        sub_function = data[0] & 0x7F
        suppress_response = bool(data[0] & 0x80)
        routine_id = int.from_bytes(data[1:3], "big")

        routine_types = {
            0x01: "startRoutine",
            0x02: "stopRoutine",
            0x03: "requestRoutineResults",
        }

        result = {
            "sub_function": sub_function,
            "suppress_positive_response": suppress_response,
            "routine_type": routine_types.get(sub_function, f"0x{sub_function:02X}"),
            "routine_id": routine_id,
        }

        if len(data) > 3:
            if is_response:
                result["status_record"] = data[3:].hex()
            else:
                result["routine_option"] = data[3:].hex()

        return result

    def _decode_tester_present(self, data: bytes, is_response: bool) -> dict[str, Any]:
        """Decode TesterPresent (0x3E).

        Request: [sub-function] (typically 0x00 or 0x80)
        Response: [sub-function]

        Args:
            data: Service payload.
            is_response: True if response.

        Returns:
            Decoded fields dictionary.
        """
        if len(data) == 0:
            return {}

        sub_function = data[0] & 0x7F
        suppress_response = bool(data[0] & 0x80)

        return {
            "sub_function": sub_function,
            "suppress_positive_response": suppress_response,
        }

    def _update_ecu_state(self, ecu_id: str, msg: UDSMessage) -> None:
        """Update ECU state based on message.

        Args:
            ecu_id: ECU identifier.
            msg: Parsed UDS message.
        """
        ecu = self.ecus[ecu_id]

        # Track supported services from requests
        if not msg.is_response and msg.negative_response_code is None:
            ecu.supported_services.add(msg.service_id)

        # Only process successful responses
        if not msg.is_response or msg.negative_response_code is not None:
            return

        # Update session state (0x10 DiagnosticSessionControl)
        if msg.service_id == 0x10:
            self._update_session_state(ecu, msg)

        # Update security level (0x27 SecurityAccess)
        if msg.service_id == 0x27:
            self._update_security_level(ecu, msg)

        # Store DTCs (0x19 ReadDTCInformation)
        if msg.service_id == 0x19:
            self._store_dtcs(ecu, msg)

        # Store data identifiers (0x22 ReadDataByIdentifier)
        if msg.service_id == 0x22:
            self._store_data_identifier(ecu, msg)

    def _update_session_state(self, ecu: UDSECU, msg: UDSMessage) -> None:
        """Update ECU diagnostic session state."""
        session_type = msg.decoded.get("sub_function")
        if session_type is not None:
            ecu.current_session = session_type

    def _update_security_level(self, ecu: UDSECU, msg: UDSMessage) -> None:
        """Update ECU security access level."""
        if msg.decoded.get("access_type") == "sendKey":
            level = msg.decoded.get("security_level", 0)
            ecu.security_level = level

    def _store_dtcs(self, ecu: UDSECU, msg: UDSMessage) -> None:
        """Store diagnostic trouble codes."""
        dtcs = msg.decoded.get("dtcs")
        if dtcs:
            ecu.dtcs = dtcs

    def _store_data_identifier(self, ecu: UDSECU, msg: UDSMessage) -> None:
        """Store data identifier value."""
        did = msg.decoded.get("did")
        did_data_hex = msg.decoded.get("did_data")
        if did is not None and did_data_hex is not None:
            ecu.data_identifiers[did] = bytes.fromhex(did_data_hex)

    def export_session_flows(self, output_path: Path) -> None:
        """Export diagnostic session flows as JSON.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = UDSAnalyzer()
            >>> # ... parse messages ...
            >>> analyzer.export_session_flows(Path("flows.json"))
        """
        flows = {
            "messages": [
                {
                    "timestamp": msg.timestamp,
                    "service_id": msg.service_id,
                    "service_name": msg.service_name,
                    "is_response": msg.is_response,
                    "sub_function": msg.sub_function,
                    "negative_response_code": msg.negative_response_code,
                    "decoded": msg.decoded,
                }
                for msg in self.messages
            ],
            "ecus": {
                ecu_id: {
                    "supported_services": sorted(ecu.supported_services),
                    "current_session": ecu.current_session,
                    "security_level": ecu.security_level,
                    "dtc_count": len(ecu.dtcs),
                    "dtcs": ecu.dtcs,
                    "data_identifier_count": len(ecu.data_identifiers),
                }
                for ecu_id, ecu in self.ecus.items()
            },
        }

        with output_path.open("w") as f:
            json.dump(flows, f, indent=2)
