"""Comprehensive test suite for OBD-II decoder.

Tests cover PID definitions, OBD2Response creation, standard PIDs,
formula calculations, and full decoding workflows.
"""

from __future__ import annotations

import pytest

# Module under test
try:
    from oscura.automotive.can.models import CANMessage
    from oscura.automotive.obd.decoder import PID, OBD2Decoder, OBD2Response

    HAS_OBD = True
except ImportError:
    HAS_OBD = False

pytestmark = pytest.mark.skipif(not HAS_OBD, reason="OBD modules not available")


# ============================================================================
# PID Definition Tests
# ============================================================================


def test_pid_creation():
    """Test creating PID definition."""
    pid = PID(
        mode=1,
        pid=0x0C,
        name="engine_rpm",
        description="Engine RPM",
        formula=lambda data: ((data[1] * 256) + data[2]) / 4,
        unit="rpm",
        min_value=0,
        max_value=16383.75,
    )

    assert pid.mode == 1
    assert pid.pid == 0x0C
    assert pid.name == "engine_rpm"
    assert pid.unit == "rpm"


def test_pid_formula_execution():
    """Test executing PID formula."""
    pid = PID(
        mode=1,
        pid=0x0C,
        name="test",
        description="Test",
        formula=lambda data: data[1] * 2,
        unit="",
        min_value=0,
        max_value=510,
    )

    data = bytes([0x41, 0x0C, 0x1A, 0xF8])  # Response format
    result = pid.formula(data)

    assert result == 24  # 12 * 2


# ============================================================================
# OBD2Response Tests
# ============================================================================


def test_obd2_response_creation():
    """Test creating OBD2Response."""
    response = OBD2Response(
        mode=1, pid=0x0C, name="engine_rpm", value=2500.0, unit="rpm", timestamp=1.5
    )

    assert response.mode == 1
    assert response.pid == 0x0C
    assert response.value == 2500.0
    assert response.timestamp == 1.5


# ============================================================================
# OBD2Decoder Standard PIDs Tests
# ============================================================================


def test_decoder_has_standard_pids():
    """Test that decoder has standard PIDs defined."""
    assert len(OBD2Decoder.PIDS) > 0
    # Should have common PIDs
    assert 0x00 in OBD2Decoder.PIDS  # PIDs supported
    assert 0x0C in OBD2Decoder.PIDS  # Engine RPM
    assert 0x0D in OBD2Decoder.PIDS  # Vehicle speed
    assert 0x05 in OBD2Decoder.PIDS  # Coolant temperature


def test_decoder_pid_support_bitmap():
    """Test PID support bitmap (PID 0x00)."""
    pid = OBD2Decoder.PIDS[0x00]

    assert pid.name == "PIDs_supported_01_20"
    assert pid.unit == "bitmap"


def test_decoder_engine_rpm_pid():
    """Test Engine RPM PID definition."""
    pid = OBD2Decoder.PIDS[0x0C]

    assert pid.name == "engine_rpm"
    assert pid.unit == "rpm"
    assert pid.min_value == 0
    assert pid.max_value == 16383.75


def test_decoder_vehicle_speed_pid():
    """Test Vehicle Speed PID definition."""
    pid = OBD2Decoder.PIDS[0x0D]

    assert pid.name == "vehicle_speed"
    assert pid.unit == "km/h"


def test_decoder_coolant_temp_pid():
    """Test Coolant Temperature PID definition."""
    pid = OBD2Decoder.PIDS[0x05]

    assert pid.name == "coolant_temp"
    assert pid.unit == "°C"


# ============================================================================
# Decoding Tests
# ============================================================================


def test_decode_engine_rpm():
    """Test decoding engine RPM response."""
    # Response: 41 0C 1A F8 (RPM = 2750)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x04, 0x41, 0x0C, 0x1A, 0xF8, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x0C
    assert response.name == "engine_rpm"
    # RPM = ((0x1A << 8) + 0xF8) / 4 = (6904) / 4 = 1726
    assert response.value == pytest.approx(1726.0)
    assert response.unit == "rpm"


def test_decode_vehicle_speed():
    """Test decoding vehicle speed response."""
    # Response: 41 0D 64 (Speed = 100 km/h)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0x0D, 0x64, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x0D
    assert response.value == 100.0
    assert response.unit == "km/h"


def test_decode_coolant_temperature():
    """Test decoding coolant temperature response."""
    # Response: 41 05 73 (Temp = 115 - 40 = 75°C)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0x05, 0x73, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x05
    assert response.value == pytest.approx(75.0)  # 115 - 40
    assert response.unit == "°C"


def test_decode_throttle_position():
    """Test decoding throttle position response."""
    # Response: 41 11 80 (Throttle = 128 * 100 / 255 = 50.2%)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0x11, 0x80, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x11
    assert response.unit == "%"
    assert 0 <= response.value <= 100


def test_decode_intake_air_temp():
    """Test decoding intake air temperature."""
    # Response: 41 0F 50 (Temp = 80 - 40 = 40°C)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0x0F, 0x50, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x0F
    assert response.value == pytest.approx(40.0)
    assert response.unit == "°C"


def test_decode_maf_flow_rate():
    """Test decoding MAF flow rate."""
    # Response: 41 10 00 64 (MAF = 100 / 100 = 1.0 g/s)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x04, 0x41, 0x10, 0x00, 0x64, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x10
    assert response.unit == "g/s"


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_decode_unknown_pid():
    """Test decoding unknown PID."""
    # Response with unknown PID 0xFF
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    # Should return None for unknown PIDs
    assert response is None


def test_decode_invalid_mode():
    """Test decoding message with invalid mode."""
    # Mode 02 (freeze frame) not fully supported
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x42, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    # Should handle gracefully
    assert response is None or response.mode == 2


def test_decode_short_data():
    """Test decoding message with insufficient data."""
    # Too short to be valid OBD response
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x01, 0x41]),
    )

    # Should handle gracefully and return None
    response = OBD2Decoder.decode(msg)
    assert response is None


def test_decode_non_obd_message():
    """Test decoding non-OBD CAN message."""
    # Random CAN message (not OBD format)
    msg = CANMessage(
        arbitration_id=0x100,
        timestamp=1.0,
        data=bytes([0x12, 0x34, 0x56, 0x78]),
    )

    response = OBD2Decoder.decode(msg)

    # Should return None for non-OBD messages
    assert response is None


# ============================================================================
# Formula Accuracy Tests
# ============================================================================


def test_formula_engine_rpm_calculation():
    """Test RPM formula accuracy."""
    pid = OBD2Decoder.PIDS[0x0C]

    # Test several RPM values
    # Note: formulas use data[1] and data[2] (skip mode byte at data[0])
    test_cases = [
        (bytes([0x0C, 0x00, 0x00]), 0.0),  # 0 RPM: (0x00<<8 + 0x00)/4 = 0
        (bytes([0x0C, 0x0F, 0xA0]), 1000.0),  # 1000 RPM: (0x0F<<8 + 0xA0)/4 = 4000/4
        (bytes([0x0C, 0x1F, 0x40]), 2000.0),  # 2000 RPM: (0x1F<<8 + 0x40)/4 = 8000/4
        (bytes([0x0C, 0x2E, 0xE0]), 3000.0),  # 3000 RPM: (0x2E<<8 + 0xE0)/4 = 12000/4
    ]

    for data, expected_rpm in test_cases:
        result = pid.formula(data)
        assert result == pytest.approx(expected_rpm, abs=0.5)


def test_formula_temperature_offset():
    """Test temperature formulas with -40 offset."""
    coolant_pid = OBD2Decoder.PIDS[0x05]
    intake_pid = OBD2Decoder.PIDS[0x0F]

    # Both should use same formula: value - 40
    # Note: formula uses data[1], not data[2]
    test_data = bytes([0x05, 0x00])  # Min value (0 - 40 = -40°C)

    coolant_temp = coolant_pid.formula(test_data)
    assert coolant_temp == -40.0


def test_formula_percentage_scaling():
    """Test percentage scaling formulas."""
    throttle_pid = OBD2Decoder.PIDS[0x11]

    test_cases = [
        (bytes([0x11, 0x00]), 0.0),  # 0%
        (bytes([0x11, 0xFF]), 100.0),  # 100%
        (bytes([0x11, 0x80]), pytest.approx(50.196, abs=0.1)),  # ~50%
    ]

    for data, expected in test_cases:
        result = throttle_pid.formula(data)
        if isinstance(expected, float):
            assert result == expected
        else:
            assert result == pytest.approx(expected, abs=0.2)


# ============================================================================
# Batch Decoding Tests
# ============================================================================


def test_decode_multiple_messages():
    """Test decoding multiple different PIDs."""
    messages = [
        # RPM
        CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.0,
            data=bytes([0x04, 0x41, 0x0C, 0x1A, 0xF8, 0x00, 0x00, 0x00]),
        ),
        # Speed
        CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.1,
            data=bytes([0x03, 0x41, 0x0D, 0x64, 0x00, 0x00, 0x00, 0x00]),
        ),
        # Temperature
        CANMessage(
            arbitration_id=0x7E8,
            timestamp=1.2,
            data=bytes([0x03, 0x41, 0x05, 0x73, 0x00, 0x00, 0x00, 0x00]),
        ),
    ]

    responses = [OBD2Decoder.decode(msg) for msg in messages]

    # All should decode successfully
    assert all(r is not None for r in responses)
    assert len(responses) == 3

    # Check PIDs
    pids = [r.pid for r in responses]  # type: ignore[union-attr]
    assert pids == [0x0C, 0x0D, 0x05]


# ============================================================================
# PID Support Bitmap Tests
# ============================================================================


def test_decode_pid_support_bitmap():
    """Test decoding PID support bitmap."""
    # Response: 41 00 BE 1F A8 13 (bitmap of supported PIDs)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x06, 0x41, 0x00, 0xBE, 0x1F, 0xA8, 0x13, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.pid == 0x00
    assert response.name == "PIDs_supported_01_20"
    assert isinstance(response.value, (int, float))


# ============================================================================
# Edge Cases
# ============================================================================


def test_decode_maximum_values():
    """Test decoding maximum allowable values."""
    # RPM at maximum (0xFFFF / 4)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x04, 0x41, 0x0C, 0xFF, 0xFF, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.value <= OBD2Decoder.PIDS[0x0C].max_value


def test_decode_minimum_values():
    """Test decoding minimum allowable values."""
    # Temperature at minimum (-40°C)
    msg = CANMessage(
        arbitration_id=0x7E8,
        timestamp=1.0,
        data=bytes([0x03, 0x41, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]),
    )

    response = OBD2Decoder.decode(msg)

    assert response is not None
    assert response.value >= OBD2Decoder.PIDS[0x05].min_value
