"""Tests for UDS (Unified Diagnostic Services) protocol decoder.

This module tests UDS (ISO 14229) message identification and decoding
for diagnostic services, sub-functions, and negative responses.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage
from oscura.automotive.uds.decoder import UDSDecoder
from oscura.automotive.uds.models import UDSNegativeResponse, UDSService


class TestIsUDSRequest:
    """Tests for UDS request message identification."""

    def test_diagnostic_session_control_request(self):
        """Test identification of DiagnosticSessionControl request."""
        # ISO-TP single frame: [length=2, SID=0x10, sub=0x01]
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=bytes([0x02, 0x10, 0x01, 0x00]))
        assert UDSDecoder.is_uds_request(msg) is True

    def test_security_access_request(self):
        """Test identification of SecurityAccess request."""
        # ISO-TP single frame: [length=2, SID=0x27, sub=0x01]
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([0x02, 0x27, 0x01, 0x00]))
        assert UDSDecoder.is_uds_request(msg) is True

    def test_read_data_by_identifier_request(self):
        """Test identification of ReadDataByIdentifier request."""
        # ISO-TP single frame: [length=3, SID=0x22, DID=0x1234]
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([0x03, 0x22, 0x12, 0x34]))
        assert UDSDecoder.is_uds_request(msg) is True

    def test_tester_present_request(self):
        """Test identification of TesterPresent request."""
        # ISO-TP single frame: [length=2, SID=0x3E, sub=0x00]
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=bytes([0x02, 0x3E, 0x00, 0x00]))
        assert UDSDecoder.is_uds_request(msg) is True

    def test_direct_uds_without_isotp(self):
        """Test identification of UDS request without ISO-TP header."""
        # Direct UDS: [SID=0x10, sub=0x03]
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([0x10, 0x03, 0x00, 0x00]))
        assert UDSDecoder.is_uds_request(msg) is True

    def test_non_uds_message(self):
        """Test that non-UDS messages are not identified as requests."""
        # Random CAN message
        msg = CANMessage(arbitration_id=0x123, timestamp=1.0, data=bytes([0x01, 0x02, 0x03, 0x04]))
        assert UDSDecoder.is_uds_request(msg) is False

    def test_too_short_message(self):
        """Test that messages too short cannot be UDS requests."""
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([0x02]))
        assert UDSDecoder.is_uds_request(msg) is False


class TestIsUDSResponse:
    """Tests for UDS response message identification."""

    def test_positive_response_diagnostic_session(self):
        """Test identification of positive DiagnosticSessionControl response."""
        # ISO-TP single frame: [length=2, response_SID=0x50, sub=0x01]
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=bytes([0x02, 0x50, 0x01, 0x00]))
        assert UDSDecoder.is_uds_response(msg) is True

    def test_positive_response_security_access(self):
        """Test identification of positive SecurityAccess response."""
        # ISO-TP single frame: [length=6, response_SID=0x67, sub=0x02, key_bytes...]
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x06, 0x67, 0x02, 0xAA, 0xBB, 0xCC, 0xDD]),
        )
        assert UDSDecoder.is_uds_response(msg) is True

    def test_negative_response(self):
        """Test identification of negative response."""
        # ISO-TP single frame: [length=3, NR_SID=0x7F, requested_SID=0x10, NRC=0x22]
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=bytes([0x03, 0x7F, 0x10, 0x22]))
        assert UDSDecoder.is_uds_response(msg) is True

    def test_positive_response_without_isotp(self):
        """Test identification of positive response without ISO-TP header."""
        # Direct UDS: [response_SID=0x62, data...]
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=bytes([0x62, 0x12, 0x34, 0x56]))
        assert UDSDecoder.is_uds_response(msg) is True

    def test_non_uds_response(self):
        """Test that non-UDS messages are not identified as responses."""
        # Random CAN message
        msg = CANMessage(arbitration_id=0x456, timestamp=1.0, data=bytes([0x12, 0x34, 0x56, 0x78]))
        assert UDSDecoder.is_uds_response(msg) is False


class TestDecodeDiagnosticSessionControl:
    """Tests for DiagnosticSessionControl (0x10) service."""

    def test_decode_request_default_session(self):
        """Test decoding request for default diagnostic session."""
        # [length=2, SID=0x10, sub=0x01]
        data = bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSService)
        assert result.sid == 0x10
        assert result.name == "DiagnosticSessionControl"
        assert result.request is True
        assert result.sub_function == 0x01
        assert result.data == b""

    def test_decode_request_programming_session(self):
        """Test decoding request for programming session."""
        # [length=2, SID=0x10, sub=0x02]
        data = bytes([0x02, 0x10, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x10
        assert result.request is True
        assert result.sub_function == 0x02

    def test_decode_positive_response(self):
        """Test decoding positive response for DiagnosticSessionControl."""
        # [length=6, response_SID=0x50, sub=0x03, session_params...]
        data = bytes([0x06, 0x50, 0x03, 0x00, 0x32, 0x01, 0xF4, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x50  # Response SID (0x10 + 0x40)
        assert result.name == "DiagnosticSessionControl"
        assert result.request is False
        assert result.sub_function == 0x03
        # UDS length=6 means: [0x50, 0x03, 0x00, 0x32, 0x01, 0xF4]
        # After removing SID and sub-function: [0x00, 0x32, 0x01, 0xF4] = 4 bytes
        assert len(result.data) == 4  # Session parameters


class TestDecodeSecurityAccess:
    """Tests for SecurityAccess (0x27) service."""

    def test_decode_request_seed(self):
        """Test decoding SecurityAccess seed request."""
        # [length=2, SID=0x27, sub=0x01 (request seed)]
        data = bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x27
        assert result.name == "SecurityAccess"
        assert result.request is True
        assert result.sub_function == 0x01

    def test_decode_request_send_key(self):
        """Test decoding SecurityAccess send key request."""
        # [length=6, SID=0x27, sub=0x02 (send key), key_bytes...]
        data = bytes([0x06, 0x27, 0x02, 0xAA, 0xBB, 0xCC, 0xDD, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x27
        assert result.request is True
        assert result.sub_function == 0x02
        assert result.data == bytes([0xAA, 0xBB, 0xCC, 0xDD])

    def test_decode_response_seed(self):
        """Test decoding SecurityAccess seed response."""
        # [length=6, response_SID=0x67, sub=0x01, seed_bytes...]
        data = bytes([0x06, 0x67, 0x01, 0x12, 0x34, 0x56, 0x78, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x67  # Response SID (0x27 + 0x40)
        assert result.request is False
        assert result.sub_function == 0x01
        assert result.data == bytes([0x12, 0x34, 0x56, 0x78])


class TestDecodeReadDataByIdentifier:
    """Tests for ReadDataByIdentifier (0x22) service."""

    def test_decode_request_single_did(self):
        """Test decoding ReadDataByIdentifier request with single DID."""
        # [length=3, SID=0x22, DID=0xF190 (VIN)]
        data = bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x22
        assert result.name == "ReadDataByIdentifier"
        assert result.request is True
        assert result.sub_function is None
        assert result.data == bytes([0xF1, 0x90])

    def test_decode_response_with_data(self):
        """Test decoding ReadDataByIdentifier response with data."""
        # [length=7, response_SID=0x62, DID=0x1234, data_bytes...]
        data = bytes([0x07, 0x62, 0x12, 0x34, 0xAA, 0xBB, 0xCC, 0xDD])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x62  # Response SID (0x22 + 0x40)
        assert result.request is False
        # UDS length=7 means: [0x62, 0x12, 0x34, 0xAA, 0xBB, 0xCC, 0xDD]
        # After removing SID: [0x12, 0x34, 0xAA, 0xBB, 0xCC, 0xDD] = 6 bytes
        assert len(result.data) == 6  # DID + data


class TestDecodeTesterPresent:
    """Tests for TesterPresent (0x3E) service."""

    def test_decode_request(self):
        """Test decoding TesterPresent request."""
        # [length=2, SID=0x3E, sub=0x00]
        data = bytes([0x02, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x3E
        assert result.name == "TesterPresent"
        assert result.request is True
        assert result.sub_function == 0x00

    def test_decode_request_suppress_response(self):
        """Test decoding TesterPresent with suppress positive response bit."""
        # [length=2, SID=0x3E, sub=0x80 (0x00 | 0x80)]
        data = bytes([0x02, 0x3E, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x3E
        assert result.request is True
        # Sub-function should be extracted without suppress bit
        assert result.sub_function == 0x00

    def test_decode_positive_response(self):
        """Test decoding TesterPresent positive response."""
        # [length=2, response_SID=0x7E, sub=0x00]
        data = bytes([0x02, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x7E  # Response SID (0x3E + 0x40)
        assert result.request is False
        assert result.sub_function == 0x00


class TestDecodeECUReset:
    """Tests for ECUReset (0x11) service."""

    def test_decode_hard_reset_request(self):
        """Test decoding ECUReset hard reset request."""
        # [length=2, SID=0x11, sub=0x01 (hard reset)]
        data = bytes([0x02, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x11
        assert result.name == "ECUReset"
        assert result.request is True
        assert result.sub_function == 0x01

    def test_decode_positive_response(self):
        """Test decoding ECUReset positive response."""
        # [length=2, response_SID=0x51, sub=0x01]
        data = bytes([0x02, 0x51, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x51  # Response SID (0x11 + 0x40)
        assert result.request is False
        assert result.sub_function == 0x01


class TestDecodeWriteDataByIdentifier:
    """Tests for WriteDataByIdentifier (0x2E) service."""

    def test_decode_request(self):
        """Test decoding WriteDataByIdentifier request."""
        # [length=7, SID=0x2E, DID=0x1234, data...]
        data = bytes([0x07, 0x2E, 0x12, 0x34, 0xAA, 0xBB, 0xCC, 0xDD])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x2E
        assert result.name == "WriteDataByIdentifier"
        assert result.request is True
        assert result.data == bytes([0x12, 0x34, 0xAA, 0xBB, 0xCC, 0xDD])


class TestDecodeRoutineControl:
    """Tests for RoutineControl (0x31) service."""

    def test_decode_start_routine_request(self):
        """Test decoding RoutineControl start routine request."""
        # [length=4, SID=0x31, sub=0x01 (start), routine_id=0x1234]
        data = bytes([0x04, 0x31, 0x01, 0x12, 0x34, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x31
        assert result.name == "RoutineControl"
        assert result.request is True
        assert result.sub_function == 0x01
        assert result.data == bytes([0x12, 0x34])


class TestDecodeNegativeResponse:
    """Tests for negative response decoding."""

    def test_decode_service_not_supported(self):
        """Test decoding serviceNotSupported negative response."""
        # [length=3, NR_SID=0x7F, requested_SID=0x10, NRC=0x11]
        data = bytes([0x03, 0x7F, 0x10, 0x11, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.requested_sid == 0x10
        assert result.nrc == 0x11
        assert result.nrc_name == "serviceNotSupported"

    def test_decode_security_access_denied(self):
        """Test decoding securityAccessDenied negative response."""
        # [length=3, NR_SID=0x7F, requested_SID=0x22, NRC=0x33]
        data = bytes([0x03, 0x7F, 0x22, 0x33, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.requested_sid == 0x22
        assert result.nrc == 0x33
        assert result.nrc_name == "securityAccessDenied"

    def test_decode_conditions_not_correct(self):
        """Test decoding conditionsNotCorrect negative response."""
        # [length=3, NR_SID=0x7F, requested_SID=0x27, NRC=0x22]
        data = bytes([0x03, 0x7F, 0x27, 0x22, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.requested_sid == 0x27
        assert result.nrc == 0x22
        assert result.nrc_name == "conditionsNotCorrect"

    def test_decode_invalid_key(self):
        """Test decoding invalidKey negative response."""
        # [length=3, NR_SID=0x7F, requested_SID=0x27, NRC=0x35]
        data = bytes([0x03, 0x7F, 0x27, 0x35, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.nrc == 0x35
        assert result.nrc_name == "invalidKey"

    def test_decode_request_out_of_range(self):
        """Test decoding requestOutOfRange negative response."""
        # [length=3, NR_SID=0x7F, requested_SID=0x2E, NRC=0x31]
        data = bytes([0x03, 0x7F, 0x2E, 0x31, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.nrc == 0x31
        assert result.nrc_name == "requestOutOfRange"

    def test_decode_unknown_nrc(self):
        """Test decoding unknown negative response code."""
        # [length=3, NR_SID=0x7F, requested_SID=0x10, NRC=0xFF (unknown)]
        data = bytes([0x03, 0x7F, 0x10, 0xFF, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.nrc == 0xFF
        assert "unknownNRC" in result.nrc_name

    def test_decode_negative_response_without_isotp(self):
        """Test decoding negative response without ISO-TP header."""
        # Direct UDS: [0x7F, requested_SID=0x22, NRC=0x13]
        data = bytes([0x7F, 0x22, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.requested_sid == 0x22
        assert result.nrc == 0x13


class TestDecodeAdditionalServices:
    """Tests for additional UDS services."""

    def test_decode_clear_diagnostic_information(self):
        """Test decoding ClearDiagnosticInformation (0x14)."""
        # [length=4, SID=0x14, group_of_dtc...]
        data = bytes([0x04, 0x14, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x14
        assert result.name == "ClearDiagnosticInformation"

    def test_decode_read_dtc_information(self):
        """Test decoding ReadDTCInformation (0x19)."""
        # [length=3, SID=0x19, sub=0x02 (reportDTCByStatusMask), mask=0xFF]
        data = bytes([0x03, 0x19, 0x02, 0xFF, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x19
        assert result.name == "ReadDTCInformation"
        assert result.sub_function == 0x02

    def test_decode_read_memory_by_address(self):
        """Test decoding ReadMemoryByAddress (0x23)."""
        # [length=7, SID=0x23, address_and_length_format, address, size]
        data = bytes([0x07, 0x23, 0x24, 0x12, 0x34, 0x56, 0x78, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x23
        assert result.name == "ReadMemoryByAddress"

    def test_decode_request_download(self):
        """Test decoding RequestDownload (0x34)."""
        # [length=7, SID=0x34, data_format, address_length_format, address, size]
        data = bytes([0x07, 0x34, 0x00, 0x44, 0x12, 0x34, 0x00, 0x10])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x34
        assert result.name == "RequestDownload"

    def test_decode_transfer_data(self):
        """Test decoding TransferData (0x36)."""
        # [length=7, SID=0x36, block_sequence, data...]
        data = bytes([0x07, 0x36, 0x01, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x36
        assert result.name == "TransferData"

    def test_decode_request_transfer_exit(self):
        """Test decoding RequestTransferExit (0x37)."""
        # [length=1, SID=0x37]
        data = bytes([0x01, 0x37, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x37
        assert result.name == "RequestTransferExit"

    def test_decode_control_dtc_setting(self):
        """Test decoding ControlDTCSetting (0x85)."""
        # [length=2, SID=0x85, sub=0x01 (on)]
        data = bytes([0x02, 0x85, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sid == 0x85
        assert result.name == "ControlDTCSetting"
        assert result.sub_function == 0x01


class TestGetServiceName:
    """Tests for service name lookup."""

    def test_get_known_service_names(self):
        """Test getting names for known services."""
        assert UDSDecoder.get_service_name(0x10) == "DiagnosticSessionControl"
        assert UDSDecoder.get_service_name(0x27) == "SecurityAccess"
        assert UDSDecoder.get_service_name(0x22) == "ReadDataByIdentifier"
        assert UDSDecoder.get_service_name(0x3E) == "TesterPresent"

    def test_get_unknown_service_name(self):
        """Test getting name for unknown service."""
        name = UDSDecoder.get_service_name(0xFF)
        assert "Unknown" in name
        assert "0xFF" in name


class TestGetNRCName:
    """Tests for NRC name lookup."""

    def test_get_known_nrc_names(self):
        """Test getting names for known NRCs."""
        assert UDSDecoder.get_nrc_name(0x11) == "serviceNotSupported"
        assert UDSDecoder.get_nrc_name(0x22) == "conditionsNotCorrect"
        assert UDSDecoder.get_nrc_name(0x33) == "securityAccessDenied"
        assert UDSDecoder.get_nrc_name(0x35) == "invalidKey"

    def test_get_unknown_nrc_name(self):
        """Test getting name for unknown NRC."""
        name = UDSDecoder.get_nrc_name(0xAB)
        assert "unknownNRC" in name
        assert "0xAB" in name


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_message(self):
        """Test that empty messages return None."""
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([]))
        result = UDSDecoder.decode_service(msg)
        assert result is None

    def test_single_byte_message(self):
        """Test that single-byte messages return None."""
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=bytes([0x10]))
        result = UDSDecoder.decode_service(msg)
        assert result is None

    def test_malformed_negative_response(self):
        """Test that malformed negative response returns None."""
        # Negative response too short
        data = bytes([0x02, 0x7F, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)
        result = UDSDecoder.decode_service(msg)
        assert result is None

    def test_unknown_service_id(self):
        """Test that unknown service IDs return None."""
        # Service 0xAA doesn't exist
        data = bytes([0x02, 0xAA, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)
        result = UDSDecoder.decode_service(msg)
        assert result is None

    def test_service_without_subfunction_data(self):
        """Test service without sub-function that has data."""
        # ReadDataByIdentifier has no sub-function, only data
        data = bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E0, timestamp=1.0, data=data)

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sub_function is None
        assert len(result.data) == 2  # DID bytes
