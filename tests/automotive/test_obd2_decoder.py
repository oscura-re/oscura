"""Tests for OBD-II protocol decoder.

This module tests OBD-II (SAE J1979) message identification and decoding
for common diagnostic PIDs.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage
from oscura.automotive.obd.decoder import OBD2Decoder, OBD2Response


class TestIsOBD2Request:
    """Tests for OBD-II request message identification."""

    def test_broadcast_request(self):
        """Test identification of broadcast request (0x7DF)."""
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_request(msg) is True

    def test_specific_ecu_requests(self):
        """Test identification of ECU-specific requests (0x7E0-0x7E7)."""
        for ecu_id in range(0x7E0, 0x7E8):
            msg = CANMessage(arbitration_id=ecu_id, timestamp=1.0, data=bytes(8))
            assert OBD2Decoder.is_obd2_request(msg) is True

    def test_non_obd2_request(self):
        """Test that non-OBD-II messages are not identified as requests."""
        # Standard CAN message
        msg = CANMessage(arbitration_id=0x123, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_request(msg) is False

        # Just outside OBD-II range
        msg = CANMessage(arbitration_id=0x7DE, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_request(msg) is False

        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_request(msg) is False


class TestIsOBD2Response:
    """Tests for OBD-II response message identification."""

    def test_valid_responses(self):
        """Test identification of valid response messages (0x7E8-0x7EF)."""
        for response_id in range(0x7E8, 0x7F0):
            msg = CANMessage(arbitration_id=response_id, timestamp=1.0, data=bytes(8))
            assert OBD2Decoder.is_obd2_response(msg) is True

    def test_non_obd2_response(self):
        """Test that non-response messages are not identified."""
        # Standard message
        msg = CANMessage(arbitration_id=0x100, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_response(msg) is False

        # Request ID
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_response(msg) is False

        # Just outside range
        msg = CANMessage(arbitration_id=0x7E7, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_response(msg) is False

        msg = CANMessage(arbitration_id=0x7F0, timestamp=1.0, data=bytes(8))
        assert OBD2Decoder.is_obd2_response(msg) is False


class TestDecodeEngineRPM:
    """Tests for decoding engine RPM (PID 0x0C)."""

    def test_decode_rpm_basic(self):
        """Test decoding basic RPM value."""
        # Mode 01 response, PID 0x0C, RPM = 2000
        # Formula: (256*A + B) / 4 = 2000 -> 256*A + B = 8000
        # A = 31 (0x1F), B = 64 (0x40)
        data = bytes([0x03, 0x41, 0x0C, 0x1F, 0x40, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert isinstance(result, OBD2Response)
        assert result.mode == 1
        assert result.pid == 0x0C
        assert result.name == "engine_rpm"
        assert abs(result.value - 2000.0) < 0.1
        assert result.unit == "rpm"
        assert result.timestamp == 1.0

    def test_decode_rpm_idle(self):
        """Test decoding idle RPM (~800 RPM)."""
        # RPM = 800 -> raw = 3200 = 0x0C80
        # A = 12 (0x0C), B = 128 (0x80)
        data = bytes([0x03, 0x41, 0x0C, 0x0C, 0x80, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert abs(result.value - 800.0) < 0.1

    def test_decode_rpm_high(self):
        """Test decoding high RPM (~6000 RPM)."""
        # RPM = 6000 -> raw = 24000 = 0x5DC0
        # A = 93 (0x5D), B = 192 (0xC0)
        data = bytes([0x03, 0x41, 0x0C, 0x5D, 0xC0, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert abs(result.value - 6000.0) < 0.1


class TestDecodeVehicleSpeed:
    """Tests for decoding vehicle speed (PID 0x0D)."""

    def test_decode_speed_basic(self):
        """Test decoding basic speed value."""
        # Mode 01 response, PID 0x0D, Speed = 100 km/h
        # Formula: A = 100
        data = bytes([0x03, 0x41, 0x0D, 100, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=2.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "vehicle_speed"
        assert result.value == 100.0
        assert result.unit == "km/h"
        assert result.timestamp == 2.0

    def test_decode_speed_zero(self):
        """Test decoding zero speed (vehicle stopped)."""
        data = bytes([0x03, 0x41, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 0.0

    def test_decode_speed_max(self):
        """Test decoding maximum speed value (255 km/h)."""
        data = bytes([0x03, 0x41, 0x0D, 255, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 255.0


class TestDecodeCoolantTemp:
    """Tests for decoding engine coolant temperature (PID 0x05)."""

    def test_decode_temp_normal(self):
        """Test decoding normal operating temperature (90°C)."""
        # Formula: A - 40 = 90 -> A = 130
        data = bytes([0x03, 0x41, 0x05, 130, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "coolant_temp"
        assert result.value == 90.0
        assert result.unit == "°C"

    def test_decode_temp_cold(self):
        """Test decoding cold temperature (-40°C, minimum)."""
        # Formula: A - 40 = -40 -> A = 0
        data = bytes([0x03, 0x41, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == -40.0

    def test_decode_temp_hot(self):
        """Test decoding hot temperature (215°C, maximum)."""
        # Formula: A - 40 = 215 -> A = 255
        data = bytes([0x03, 0x41, 0x05, 255, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 215.0


class TestDecodeThrottlePosition:
    """Tests for decoding throttle position (PID 0x11)."""

    def test_decode_throttle_half(self):
        """Test decoding 50% throttle position."""
        # Formula: A * 100 / 255 ≈ 50% -> A ≈ 128
        data = bytes([0x03, 0x41, 0x11, 128, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "throttle_position"
        assert abs(result.value - 50.0) < 1.0
        assert result.unit == "%"

    def test_decode_throttle_closed(self):
        """Test decoding closed throttle (0%)."""
        data = bytes([0x03, 0x41, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 0.0

    def test_decode_throttle_wide_open(self):
        """Test decoding wide-open throttle (100%)."""
        data = bytes([0x03, 0x41, 0x11, 255, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 100.0


class TestDecodeIntakeAirTemp:
    """Tests for decoding intake air temperature (PID 0x0F)."""

    def test_decode_intake_temp_normal(self):
        """Test decoding normal intake air temperature."""
        # Formula: A - 40 = 25°C -> A = 65
        data = bytes([0x03, 0x41, 0x0F, 65, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "intake_air_temp"
        assert result.value == 25.0
        assert result.unit == "°C"


class TestDecodeFuelLevel:
    """Tests for decoding fuel tank level (PID 0x2F)."""

    def test_decode_fuel_half(self):
        """Test decoding half-full tank."""
        # Formula: A * 100 / 255 ≈ 50% -> A ≈ 128
        data = bytes([0x03, 0x41, 0x2F, 128, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "fuel_level"
        assert abs(result.value - 50.0) < 1.0
        assert result.unit == "%"

    def test_decode_fuel_empty(self):
        """Test decoding empty tank."""
        data = bytes([0x03, 0x41, 0x2F, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.value == 0.0


class TestDecodePIDsSupported:
    """Tests for decoding PIDs supported bitmap (PID 0x00)."""

    def test_decode_pids_supported(self):
        """Test decoding PIDs supported bitmap."""
        # PIDs 0x01-0x20 bitmap
        data = bytes([0x06, 0x41, 0x00, 0xBE, 0x1F, 0xA8, 0x13, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "PIDs_supported_01_20"
        assert result.unit == "bitmap"
        # Value should be the 4-byte bitmap as integer
        expected = 0xBE1FA813
        assert result.value == expected


class TestInvalidMessages:
    """Tests for handling invalid OBD-II messages."""

    def test_decode_non_response_returns_none(self):
        """Test that non-response message returns None."""
        msg = CANMessage(arbitration_id=0x123, timestamp=1.0, data=bytes(8))
        result = OBD2Decoder.decode(msg)
        assert result is None

    def test_decode_request_returns_none(self):
        """Test that request message returns None."""
        msg = CANMessage(arbitration_id=0x7DF, timestamp=1.0, data=bytes(8))
        result = OBD2Decoder.decode(msg)
        assert result is None

    def test_decode_too_short_message(self):
        """Test that message with insufficient data returns None."""
        # Only 3 bytes (need at least 4)
        data = bytes([0x02, 0x41, 0x0C])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)
        assert result is None

    def test_decode_invalid_mode_response(self):
        """Test that message with invalid mode byte returns None."""
        # Mode byte should be >= 0x40, but this is 0x01
        data = bytes([0x03, 0x01, 0x0C, 0x1F, 0x40, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)
        assert result is None

    def test_decode_unsupported_mode(self):
        """Test that non-Mode-01 response returns None."""
        # Mode 02 response (0x42)
        data = bytes([0x03, 0x42, 0x0C, 0x1F, 0x40, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)
        assert result is None

    def test_decode_unknown_pid(self):
        """Test that unknown PID returns None."""
        # Valid Mode 01 response but unknown PID 0xFF
        data = bytes([0x03, 0x41, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)
        assert result is None


class TestMessageFormat:
    """Tests for OBD-II message format validation."""

    def test_valid_format_with_different_lengths(self):
        """Test that different byte count values are handled correctly."""
        # Byte 0 indicates number of additional bytes
        # For PID 0x0D (speed), only 1 data byte needed
        data = bytes([0x03, 0x41, 0x0D, 100, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)
        assert result is not None
        assert result.value == 100.0

    def test_response_from_different_ecus(self):
        """Test decoding responses from different ECU IDs."""
        data = bytes([0x03, 0x41, 0x0C, 0x1F, 0x40, 0x00, 0x00, 0x00])

        # Test all valid response IDs
        for ecu_id in range(0x7E8, 0x7F0):
            msg = CANMessage(arbitration_id=ecu_id, timestamp=1.0, data=data)
            result = OBD2Decoder.decode(msg)
            assert result is not None
            assert abs(result.value - 2000.0) < 0.1


class TestDecodeCalculatedEngineLoad:
    """Tests for decoding calculated engine load (PID 0x04)."""

    def test_decode_engine_load_half(self):
        """Test decoding 50% engine load."""
        # Formula: A * 100 / 255 ≈ 50% -> A ≈ 128
        data = bytes([0x03, 0x41, 0x04, 128, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "calculated_engine_load"
        assert abs(result.value - 50.0) < 1.0
        assert result.unit == "%"


class TestDecodeFuelTrim:
    """Tests for decoding fuel trim PIDs (0x06-0x09)."""

    def test_decode_short_term_fuel_trim_bank1_positive(self):
        """Test decoding positive short term fuel trim (rich)."""
        # Formula: (A - 128) * 100 / 128 = +10% -> A = 140.8 ≈ 141
        data = bytes([0x03, 0x41, 0x06, 141, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "short_term_fuel_trim_bank1"
        assert abs(result.value - 10.0) < 1.0
        assert result.unit == "%"

    def test_decode_short_term_fuel_trim_bank1_negative(self):
        """Test decoding negative short term fuel trim (lean)."""
        # Formula: (A - 128) * 100 / 128 = -10% -> A = 115.2 ≈ 115
        data = bytes([0x03, 0x41, 0x06, 115, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert abs(result.value - (-10.0)) < 1.0

    def test_decode_long_term_fuel_trim_bank2(self):
        """Test decoding long term fuel trim bank 2."""
        # Formula: (A - 128) * 100 / 128 = 0% -> A = 128
        data = bytes([0x03, 0x41, 0x09, 128, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "long_term_fuel_trim_bank2"
        assert abs(result.value) < 0.1


class TestDecodeFuelPressure:
    """Tests for decoding fuel pressure (PID 0x0A)."""

    def test_decode_fuel_pressure_normal(self):
        """Test decoding normal fuel pressure."""
        # Formula: A * 3 = 300 kPa -> A = 100
        data = bytes([0x03, 0x41, 0x0A, 100, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "fuel_pressure"
        assert result.value == 300.0
        assert result.unit == "kPa"


class TestDecodeIntakeManifoldPressure:
    """Tests for decoding intake manifold pressure (PID 0x0B)."""

    def test_decode_manifold_pressure_normal(self):
        """Test decoding normal manifold pressure."""
        # Formula: A = 100 kPa
        data = bytes([0x03, 0x41, 0x0B, 100, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "intake_manifold_pressure"
        assert result.value == 100.0
        assert result.unit == "kPa"


class TestDecodeTimingAdvance:
    """Tests for decoding timing advance (PID 0x0E)."""

    def test_decode_timing_advance_positive(self):
        """Test decoding positive timing advance."""
        # Formula: (A - 128) / 2 = 10° -> A = 148
        data = bytes([0x03, 0x41, 0x0E, 148, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "timing_advance"
        assert abs(result.value - 10.0) < 0.1
        assert result.unit == "° before TDC"

    def test_decode_timing_advance_zero(self):
        """Test decoding zero timing advance."""
        # Formula: (A - 128) / 2 = 0° -> A = 128
        data = bytes([0x03, 0x41, 0x0E, 128, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert abs(result.value) < 0.1


class TestDecodeMAFAirFlowRate:
    """Tests for decoding MAF air flow rate (PID 0x10)."""

    def test_decode_maf_normal(self):
        """Test decoding normal MAF air flow rate."""
        # Formula: (256*A + B) / 100 = 50 g/s -> raw = 5000 = 0x1388
        # A = 19 (0x13), B = 136 (0x88)
        data = bytes([0x04, 0x41, 0x10, 0x13, 0x88, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "maf_air_flow_rate"
        assert abs(result.value - 50.0) < 0.1
        assert result.unit == "g/s"


class TestDecodeOxygenSensors:
    """Tests for decoding oxygen sensor voltages (PIDs 0x14-0x1B)."""

    def test_decode_o2_sensor1_voltage(self):
        """Test decoding O2 sensor 1 voltage."""
        # Formula: A / 200 = 0.45V -> A = 90
        data = bytes([0x04, 0x41, 0x14, 90, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "o2_sensor1_voltage"
        assert abs(result.value - 0.45) < 0.01
        assert result.unit == "V"

    def test_decode_o2_sensor5_voltage(self):
        """Test decoding O2 sensor 5 voltage (Bank 2)."""
        # Formula: A / 200 = 0.70V -> A = 140
        data = bytes([0x04, 0x41, 0x18, 140, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "o2_sensor5_voltage"
        assert abs(result.value - 0.70) < 0.01
        assert result.unit == "V"


class TestDecodeRuntime:
    """Tests for decoding runtime since engine start (PID 0x1F)."""

    def test_decode_runtime_minutes(self):
        """Test decoding runtime (600 seconds = 10 minutes)."""
        # Formula: 256*A + B = 600 -> 0x0258
        # A = 2 (0x02), B = 88 (0x58)
        data = bytes([0x04, 0x41, 0x1F, 0x02, 0x58, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "run_time_since_engine_start"
        assert result.value == 600.0
        assert result.unit == "s"


class TestDecodeDistanceMIL:
    """Tests for decoding distance traveled with MIL on (PID 0x21)."""

    def test_decode_distance_mil(self):
        """Test decoding distance traveled with MIL on."""
        # Formula: 256*A + B = 1500 km -> 0x05DC
        # A = 5 (0x05), B = 220 (0xDC)
        data = bytes([0x04, 0x41, 0x21, 0x05, 0xDC, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "distance_traveled_with_mil_on"
        assert result.value == 1500.0
        assert result.unit == "km"


class TestDecodeEGR:
    """Tests for decoding EGR parameters (PIDs 0x2C, 0x2D)."""

    def test_decode_commanded_egr(self):
        """Test decoding commanded EGR."""
        # Formula: A * 100 / 255 ≈ 30% -> A ≈ 77
        data = bytes([0x03, 0x41, 0x2C, 77, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "commanded_egr"
        assert abs(result.value - 30.0) < 2.0
        assert result.unit == "%"

    def test_decode_egr_error(self):
        """Test decoding EGR error."""
        # Formula: (A - 128) * 100 / 128 = 5% -> A = 134.4 ≈ 134
        data = bytes([0x03, 0x41, 0x2D, 134, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "egr_error"
        assert abs(result.value - 5.0) < 1.0
        assert result.unit == "%"


class TestDecodeBarometricPressure:
    """Tests for decoding absolute barometric pressure (PID 0x33)."""

    def test_decode_barometric_pressure(self):
        """Test decoding barometric pressure (sea level ~101 kPa)."""
        # Formula: A = 101 kPa
        data = bytes([0x03, 0x41, 0x33, 101, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "absolute_barometric_pressure"
        assert result.value == 101.0
        assert result.unit == "kPa"


class TestDecodeCatalystTemperature:
    """Tests for decoding catalyst temperatures (PIDs 0x3C-0x3F)."""

    def test_decode_catalyst_temp_b1s1_normal(self):
        """Test decoding catalyst temperature Bank 1 Sensor 1."""
        # Formula: (256*A + B) / 10 - 40 = 400°C
        # -> (256*A + B) / 10 = 440 -> raw = 4400 = 0x1130
        # A = 17 (0x11), B = 48 (0x30)
        data = bytes([0x04, 0x41, 0x3C, 0x11, 0x30, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "catalyst_temp_b1s1"
        assert abs(result.value - 400.0) < 0.1
        assert result.unit == "°C"

    def test_decode_catalyst_temp_b2s2_high(self):
        """Test decoding catalyst temperature Bank 2 Sensor 2."""
        # Formula: (256*A + B) / 10 - 40 = 600°C
        # -> raw = 6400 = 0x1900
        # A = 25 (0x19), B = 0 (0x00)
        data = bytes([0x04, 0x41, 0x3F, 0x19, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "catalyst_temp_b2s2"
        assert abs(result.value - 600.0) < 0.1


class TestDecodeControlModuleVoltage:
    """Tests for decoding control module voltage (PID 0x42)."""

    def test_decode_module_voltage(self):
        """Test decoding control module voltage (14.2V)."""
        # Formula: (256*A + B) / 1000 = 14.2V -> raw = 14200 = 0x3778
        # A = 55 (0x37), B = 120 (0x78)
        data = bytes([0x04, 0x41, 0x42, 0x37, 0x78, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "control_module_voltage"
        assert abs(result.value - 14.2) < 0.01
        assert result.unit == "V"


class TestDecodeAmbientAirTemp:
    """Tests for decoding ambient air temperature (PID 0x46)."""

    def test_decode_ambient_temp(self):
        """Test decoding ambient air temperature."""
        # Formula: A - 40 = 25°C -> A = 65
        data = bytes([0x03, 0x41, 0x46, 65, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "ambient_air_temp"
        assert result.value == 25.0
        assert result.unit == "°C"


class TestDecodeThrottlePositions:
    """Tests for decoding additional throttle position PIDs."""

    def test_decode_relative_throttle_position(self):
        """Test decoding relative throttle position (PID 0x45)."""
        # Formula: A * 100 / 255 ≈ 75% -> A ≈ 191
        data = bytes([0x03, 0x41, 0x45, 191, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "relative_throttle_position"
        assert abs(result.value - 75.0) < 1.0
        assert result.unit == "%"


class TestDecodeAcceleratorPedal:
    """Tests for decoding accelerator pedal positions."""

    def test_decode_accelerator_pedal_position_d(self):
        """Test decoding accelerator pedal position D (PID 0x49)."""
        # Formula: A * 100 / 255 ≈ 60% -> A ≈ 153
        data = bytes([0x03, 0x41, 0x49, 153, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "accelerator_pedal_position_d"
        assert abs(result.value - 60.0) < 1.0
        assert result.unit == "%"


class TestDecodeFuelType:
    """Tests for decoding fuel type (PID 0x51)."""

    def test_decode_fuel_type_gasoline(self):
        """Test decoding fuel type (1 = Gasoline)."""
        data = bytes([0x03, 0x41, 0x51, 0x01, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "fuel_type"
        assert result.value == 1.0
        assert result.unit == "type"


class TestDecodeEthanolPercentage:
    """Tests for decoding ethanol fuel percentage (PID 0x52)."""

    def test_decode_ethanol_percentage(self):
        """Test decoding ethanol fuel percentage (E85 = 85%)."""
        # Formula: A * 100 / 255 ≈ 85% -> A ≈ 217
        data = bytes([0x03, 0x41, 0x52, 217, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "ethanol_fuel_percentage"
        assert abs(result.value - 85.0) < 1.0
        assert result.unit == "%"


class TestDecodeEngineOilTemp:
    """Tests for decoding engine oil temperature (PID 0x5C)."""

    def test_decode_engine_oil_temp(self):
        """Test decoding engine oil temperature (100°C)."""
        # Formula: A - 40 = 100°C -> A = 140
        data = bytes([0x03, 0x41, 0x5C, 140, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "engine_oil_temp"
        assert result.value == 100.0
        assert result.unit == "°C"


class TestDecodeExtendedPIDSupport:
    """Tests for decoding extended PID support bitmaps."""

    def test_decode_pids_supported_21_40(self):
        """Test decoding PIDs supported 21-40 (PID 0x20)."""
        data = bytes([0x06, 0x41, 0x20, 0x80, 0x00, 0x40, 0x01, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "PIDs_supported_21_40"
        assert result.unit == "bitmap"
        expected = 0x80004001
        assert result.value == expected

    def test_decode_pids_supported_41_60(self):
        """Test decoding PIDs supported 41-60 (PID 0x40)."""
        data = bytes([0x06, 0x41, 0x40, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])
        msg = CANMessage(arbitration_id=0x7E8, timestamp=1.0, data=data)

        result = OBD2Decoder.decode(msg)

        assert result is not None
        assert result.name == "PIDs_supported_41_60"
        assert result.value == 0xFFFFFFFF


class TestPIDDefinitions:
    """Tests for PID definition structure."""

    def test_pid_definitions_exist(self):
        """Test that expected PIDs are defined."""
        assert 0x00 in OBD2Decoder.PIDS  # PIDs supported
        assert 0x0C in OBD2Decoder.PIDS  # Engine RPM
        assert 0x0D in OBD2Decoder.PIDS  # Vehicle speed
        assert 0x05 in OBD2Decoder.PIDS  # Coolant temp
        assert 0x0F in OBD2Decoder.PIDS  # Intake air temp
        assert 0x11 in OBD2Decoder.PIDS  # Throttle position
        assert 0x2F in OBD2Decoder.PIDS  # Fuel level

    def test_new_pid_definitions_exist(self):
        """Test that newly added PIDs are defined."""
        # Extended PID support bitmaps
        assert 0x20 in OBD2Decoder.PIDS
        assert 0x40 in OBD2Decoder.PIDS
        assert 0x60 in OBD2Decoder.PIDS
        # Engine parameters
        assert 0x04 in OBD2Decoder.PIDS  # Calculated engine load
        assert 0x06 in OBD2Decoder.PIDS  # Short term fuel trim bank 1
        assert 0x0A in OBD2Decoder.PIDS  # Fuel pressure
        assert 0x0B in OBD2Decoder.PIDS  # Intake manifold pressure
        assert 0x0E in OBD2Decoder.PIDS  # Timing advance
        assert 0x10 in OBD2Decoder.PIDS  # MAF air flow rate
        # Oxygen sensors
        assert 0x14 in OBD2Decoder.PIDS  # O2 sensor 1
        assert 0x18 in OBD2Decoder.PIDS  # O2 sensor 5
        # Temperatures
        assert 0x3C in OBD2Decoder.PIDS  # Catalyst temp B1S1
        assert 0x46 in OBD2Decoder.PIDS  # Ambient air temp
        assert 0x5C in OBD2Decoder.PIDS  # Engine oil temp
        # Fuel system
        assert 0x51 in OBD2Decoder.PIDS  # Fuel type
        assert 0x52 in OBD2Decoder.PIDS  # Ethanol percentage

    def test_pid_count(self):
        """Test that we have 50+ PIDs defined."""
        assert len(OBD2Decoder.PIDS) >= 50

    def test_pid_has_required_fields(self):
        """Test that PID definitions have required fields."""
        pid = OBD2Decoder.PIDS[0x0C]

        assert pid.mode == 1
        assert pid.pid == 0x0C
        assert pid.name == "engine_rpm"
        assert pid.unit == "rpm"
        assert callable(pid.formula)
        assert pid.min_value is not None
        assert pid.max_value is not None
