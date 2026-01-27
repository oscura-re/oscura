"""Comprehensive tests for UDS decoder.

This module tests ISO 14229 UDS protocol decoding.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage
from oscura.automotive.uds.decoder import UDSDecoder
from oscura.automotive.uds.models import UDSNegativeResponse, UDSService


class TestUDSDecoderRequestDetection:
    """Tests for UDS request detection."""

    def test_is_uds_request_valid_service(self):
        """Test detecting valid UDS request."""
        # DiagnosticSessionControl (0x10) with ISO-TP single frame
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is True

    def test_is_uds_request_direct_sid(self):
        """Test detecting UDS request without ISO-TP header."""
        # ReadDataByIdentifier (0x22) directly
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is True

    def test_is_uds_request_invalid_service(self):
        """Test rejecting invalid service ID."""
        # Invalid service ID 0xFF
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is False

    def test_is_uds_request_too_short(self):
        """Test rejecting too-short messages."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x01]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is False

    def test_is_uds_request_all_services(self):
        """Test detection for all defined UDS services."""
        service_ids = [
            0x10,
            0x11,
            0x14,
            0x19,
            0x22,
            0x23,
            0x27,
            0x28,
            0x2E,
            0x2F,
            0x31,
            0x34,
            0x35,
            0x36,
            0x37,
            0x3E,
            0x85,
        ]

        for sid in service_ids:
            msg = CANMessage(
                arbitration_id=0x7E0,
                timestamp=1.0,
                data=bytes([0x02, sid, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                is_extended=False,
            )
            assert UDSDecoder.is_uds_request(msg) is True, f"Service 0x{sid:02X} not detected"


class TestUDSDecoderResponseDetection:
    """Tests for UDS response detection."""

    def test_is_uds_response_positive(self):
        """Test detecting positive UDS response."""
        # Positive response to DiagnosticSessionControl (0x10 + 0x40 = 0x50)
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x02, 0x50, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_response(msg) is True

    def test_is_uds_response_negative(self):
        """Test detecting negative UDS response."""
        # Negative response: 0x7F, requested SID, NRC
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x03, 0x7F, 0x10, 0x12, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_response(msg) is True

    def test_is_uds_response_direct_positive(self):
        """Test detecting positive response without ISO-TP."""
        # Direct positive response
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x62, 0xF1, 0x90, 0x01, 0x02, 0x03, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_response(msg) is True

    def test_is_uds_response_invalid(self):
        """Test rejecting invalid response."""
        # Not a response
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_response(msg) is False

    def test_is_uds_response_too_short_negative(self):
        """Test rejecting too-short negative response."""
        # Negative response needs at least 3 bytes after PCI
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x02, 0x7F, 0x10]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_response(msg) is False


class TestUDSDecoderServiceDecoding:
    """Tests for UDS service decoding."""

    def test_decode_service_diagnostic_session_control(self):
        """Test decoding DiagnosticSessionControl request."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSService)
        assert result.service_id == 0x10
        assert result.service_name == "DiagnosticSessionControl"
        assert result.sub_function == 0x01

    def test_decode_service_read_data_by_identifier(self):
        """Test decoding ReadDataByIdentifier request."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSService)
        assert result.service_id == 0x22
        assert result.service_name == "ReadDataByIdentifier"
        # Data should contain DID (0xF190)
        assert len(result.data) >= 2

    def test_decode_service_tester_present(self):
        """Test decoding TesterPresent request."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.service_id == 0x3E
        assert result.service_name == "TesterPresent"
        assert result.sub_function == 0x00

    def test_decode_service_negative_response(self):
        """Test decoding negative response."""
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x03, 0x7F, 0x10, 0x12, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSNegativeResponse)
        assert result.requested_sid == 0x10
        assert result.nrc == 0x12
        assert result.nrc_name == "subFunctionNotSupported"

    def test_decode_service_positive_response(self):
        """Test decoding positive response."""
        # Positive response to DiagnosticSessionControl
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x06, 0x50, 0x01, 0x00, 0x32, 0x01, 0xF4, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert isinstance(result, UDSService)
        assert result.service_id == 0x50
        assert result.is_response is True

    def test_decode_service_too_short(self):
        """Test decoding too-short message."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x01]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)
        assert result is None

    def test_decode_service_invalid(self):
        """Test decoding invalid message."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)
        assert result is None


class TestUDSDecoderNegativeResponses:
    """Tests for negative response decoding."""

    def test_decode_nrc_all_codes(self):
        """Test decoding all defined NRC codes."""
        nrc_codes = {
            0x10: "generalReject",
            0x11: "serviceNotSupported",
            0x12: "subFunctionNotSupported",
            0x13: "incorrectMessageLengthOrInvalidFormat",
            0x21: "busyRepeatRequest",
            0x22: "conditionsNotCorrect",
            0x31: "requestOutOfRange",
            0x33: "securityAccessDenied",
            0x35: "invalidKey",
            0x36: "exceedNumberOfAttempts",
            0x37: "requiredTimeDelayNotExpired",
            0x78: "requestCorrectlyReceivedResponsePending",
            0x7F: "serviceNotSupportedInActiveSession",
        }

        for nrc, expected_name in nrc_codes.items():
            msg = CANMessage(
                arbitration_id=0x7E8,
                timestamp=1.0,
                data=bytes([0x03, 0x7F, 0x10, nrc, 0x00, 0x00, 0x00, 0x00]),
                is_extended=False,
            )

            result = UDSDecoder.decode_service(msg)

            assert isinstance(result, UDSNegativeResponse)
            assert result.nrc == nrc
            assert result.nrc_name == expected_name

    def test_decode_nrc_unknown_code(self):
        """Test decoding unknown NRC code."""
        msg = CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x03, 0x7F, 0x10, 0xFF, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert isinstance(result, UDSNegativeResponse)
        assert result.nrc == 0xFF
        assert result.nrc_name == "unknownNRC_0xFF"


class TestUDSDecoderSubFunctions:
    """Tests for sub-function handling."""

    def test_decode_service_with_subfunction(self):
        """Test services that have sub-functions."""
        services_with_subfunc = [0x10, 0x11, 0x19, 0x27, 0x28, 0x2F, 0x31, 0x3E, 0x85]

        for sid in services_with_subfunc:
            msg = CANMessage(
                arbitration_id=0x7E0,
                timestamp=1.0,
                data=bytes([0x02, sid, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
                is_extended=False,
            )

            result = UDSDecoder.decode_service(msg)

            assert result is not None
            assert result.service_id == sid
            assert result.sub_function == 0x01

    def test_decode_service_without_subfunction(self):
        """Test services without sub-functions."""
        # ReadDataByIdentifier doesn't have sub-function
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.sub_function is None


class TestUDSDecoderEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_decode_empty_data(self):
        """Test decoding message with empty data."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is False
        assert UDSDecoder.is_uds_response(msg) is False
        assert UDSDecoder.decode_service(msg) is None

    def test_decode_single_byte(self):
        """Test decoding single-byte message."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x10]),
            is_extended=False,
        )

        assert UDSDecoder.is_uds_request(msg) is False

    def test_decode_max_length_isotp(self):
        """Test decoding with maximum ISO-TP length indicator."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x07, 0x10, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.service_id == 0x10

    def test_decode_security_access_request(self):
        """Test decoding SecurityAccess service."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.service_id == 0x27
        assert result.service_name == "SecurityAccess"
        assert result.sub_function == 0x01

    def test_decode_routine_control(self):
        """Test decoding RoutineControl service."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x04, 0x31, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.service_id == 0x31
        assert result.service_name == "RoutineControl"
        assert result.sub_function == 0x01

    def test_decode_transfer_data(self):
        """Test decoding TransferData service."""
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x05, 0x36, 0x01, 0xAA, 0xBB, 0xCC, 0x00, 0x00]),
            is_extended=False,
        )

        result = UDSDecoder.decode_service(msg)

        assert result is not None
        assert result.service_id == 0x36
        assert result.service_name == "TransferData"
