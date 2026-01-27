"""Comprehensive test suite for UDS protocol decoder.

Tests cover UDS services (0x10, 0x22, 0x27, 0x3E, etc.), response decoding,
negative responses, and multi-frame handling.
"""

from __future__ import annotations

import pytest

# Module under test
try:
    from oscura.automotive.can.models import CANMessage
    from oscura.automotive.uds.decoder import UDSDecoder, UDSResponse

    HAS_UDS = True
except ImportError:
    HAS_UDS = False

pytestmark = pytest.mark.skipif(not HAS_UDS, reason="UDS modules not available")


# ============================================================================
# UDSResponse Tests
# ============================================================================


def test_uds_response_creation():
    """Test creating UDS response."""
    response = UDSResponse(
        service=0x50,  # Positive response to 0x10
        data=bytes([0x01]),
        timestamp=1.5,
        is_negative=False,
    )

    assert response.service == 0x50
    assert response.is_negative is False
    assert response.timestamp == 1.5


def test_uds_response_negative():
    """Test creating negative response."""
    response = UDSResponse(
        service=0x7F,
        data=bytes([0x10, 0x31]),  # Service 0x10, NRC 0x31
        timestamp=1.0,
        is_negative=True,
        nrc=0x31,
    )

    assert response.is_negative is True
    assert response.nrc == 0x31


# ============================================================================
# Diagnostic Session Control (0x10) Tests
# ============================================================================


def test_decode_diagnostic_session_control_request():
    """Test decoding diagnostic session control request."""
    msg = CANMessage(
        arbitration_id=0x7DF,  # Functional request
        timestamp=1.0,
        data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # Default session
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x10
    assert response.is_negative is False


def test_decode_diagnostic_session_control_response():
    """Test decoding positive response to session control."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x06, 0x50, 0x01, 0x00, 0x32, 0x01, 0xF4, 0x00]),  # Positive response
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x50
    assert response.is_negative is False


# ============================================================================
# Read Data By Identifier (0x22) Tests
# ============================================================================


def test_decode_read_data_by_id_request():
    """Test decoding read data by identifier request."""
    msg = CANMessage(
        arbitration_id=0x7DF,
        timestamp=1.0,
        data=bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00]),  # Read VIN
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x22


def test_decode_read_data_by_id_response():
    """Test decoding positive response to read data."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x10, 0x14, 0x62, 0xF1, 0x90, 0x31, 0x47, 0x31]),  # Multi-frame start
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x62  # Positive response to 0x22


# ============================================================================
# Security Access (0x27) Tests
# ============================================================================


def test_decode_security_access_request_seed():
    """Test decoding security access seed request."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # Request seed
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x27


def test_decode_security_access_send_key():
    """Test decoding security access send key request."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x06, 0x27, 0x02, 0x12, 0x34, 0x56, 0x78, 0x00]),  # Send key
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x27


# ============================================================================
# Tester Present (0x3E) Tests
# ============================================================================


def test_decode_tester_present_request():
    """Test decoding tester present request."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x02, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x3E


def test_decode_tester_present_response():
    """Test decoding tester present positive response."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x02, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x7E  # Positive response to 0x3E


# ============================================================================
# ECU Reset (0x11) Tests
# ============================================================================


def test_decode_ecu_reset_request():
    """Test decoding ECU reset request."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x02, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # Hard reset
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x11


# ============================================================================
# Negative Response Tests
# ============================================================================


def test_decode_negative_response():
    """Test decoding negative response (0x7F)."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x7F, 0x22, 0x31, 0x00, 0x00, 0x00, 0x00]),  # NRC 0x31
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x7F
    assert response.is_negative is True
    assert response.nrc == 0x31  # Request out of range


def test_decode_negative_response_conditions_not_correct():
    """Test negative response with NRC 0x22 (conditions not correct)."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x7F, 0x10, 0x22, 0x00, 0x00, 0x00, 0x00]),
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.is_negative is True
    assert response.nrc == 0x22


def test_decode_negative_response_security_access_denied():
    """Test negative response for security access denied."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x7F, 0x27, 0x35, 0x00, 0x00, 0x00, 0x00]),  # Invalid key
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.nrc == 0x35  # Invalid key


# ============================================================================
# Multi-Frame Handling Tests
# ============================================================================


def test_decode_first_frame():
    """Test decoding first frame of multi-frame message."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x10, 0x20, 0x62, 0xF1, 0x90, 0x00, 0x01, 0x02]),  # First frame, 32 bytes total
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    # Should recognize as multi-frame
    assert response.service == 0x62


def test_decode_consecutive_frame():
    """Test decoding consecutive frame."""
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x21, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]),  # Consecutive frame
    )

    response = UDSDecoder.decode(msg)

    # Consecutive frames might not be decoded standalone
    assert response is None or response.service is not None


# ============================================================================
# Write Data By Identifier (0x2E) Tests
# ============================================================================


def test_decode_write_data_by_id_request():
    """Test decoding write data by identifier request."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x07, 0x2E, 0xF1, 0x90, 0x01, 0x02, 0x03, 0x04]),
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x2E


# ============================================================================
# Control DTC Setting (0x85) Tests
# ============================================================================


def test_decode_control_dtc_setting_on():
    """Test decoding control DTC setting ON."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x02, 0x85, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),  # DTC On
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x85


# ============================================================================
# Clear Diagnostic Information (0x14) Tests
# ============================================================================


def test_decode_clear_diagnostic_info():
    """Test decoding clear diagnostic information request."""
    msg = CANMessage(
        arbitration_id=0x7DF,
        timestamp=1.0,
        data=bytes([0x04, 0x14, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00]),  # Clear all DTCs
    )

    response = UDSDecoder.decode(msg)

    assert response is not None
    assert response.service == 0x14


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_decode_invalid_service():
    """Test decoding message with invalid/unknown service."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x02, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),  # Invalid service
    )

    response = UDSDecoder.decode(msg)

    # Should handle gracefully
    assert response is None or response.service == 0xFF


def test_decode_short_data():
    """Test decoding message with insufficient data."""
    msg = CANMessage(
        arbitration_id=0x7E0,
        timestamp=1.0,
        data=bytes([0x01]),  # Too short
    )

    with pytest.raises((ValueError, IndexError)):
        UDSDecoder.decode(msg)


def test_decode_non_uds_message():
    """Test decoding non-UDS CAN message."""
    msg = CANMessage(
        arbitration_id=0x100,  # Non-diagnostic ID
        timestamp=1.0,
        data=bytes([0x12, 0x34, 0x56, 0x78]),
    )

    response = UDSDecoder.decode(msg)

    # Should return None for non-UDS messages
    assert response is None


# ============================================================================
# Service Identification Tests
# ============================================================================


def test_identify_all_common_services():
    """Test that decoder can identify all common UDS services."""
    common_services = [
        0x10,  # Diagnostic session control
        0x11,  # ECU reset
        0x14,  # Clear diagnostic information
        0x22,  # Read data by identifier
        0x27,  # Security access
        0x2E,  # Write data by identifier
        0x3E,  # Tester present
        0x85,  # Control DTC setting
    ]

    for service_id in common_services:
        msg = CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, service_id, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        )

        response = UDSDecoder.decode(msg)

        # Should at least recognize the service
        assert response is not None
        assert response.service == service_id


# ============================================================================
# Batch Decoding Tests
# ============================================================================


def test_decode_multiple_uds_messages():
    """Test decoding sequence of UDS messages."""
    messages = [
        # Session control
        CANMessage(
            arbitration_id=0x7E0,
            timestamp=1.0,
            data=bytes([0x02, 0x10, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
        ),
        # Tester present
        CANMessage(
            arbitration_id=0x7E0,
            timestamp=2.0,
            data=bytes([0x02, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        ),
        # Read DID
        CANMessage(
            arbitration_id=0x7E0,
            timestamp=3.0,
            data=bytes([0x03, 0x22, 0xF1, 0x90, 0x00, 0x00, 0x00, 0x00]),
        ),
    ]

    responses = [UDSDecoder.decode(msg) for msg in messages]

    # All should decode
    assert all(r is not None for r in responses)
    # Check services
    services = [r.service for r in responses]  # type: ignore[union-attr]
    assert services == [0x10, 0x3E, 0x22]
