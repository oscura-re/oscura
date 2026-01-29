"""OBD-II protocol decoder.

This module implements OBD-II (SAE J1979) diagnostic protocol decoding
for standard vehicle diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage

__all__ = ["PID", "OBD2Decoder", "OBD2Response"]


@dataclass
class PID:
    """OBD-II Parameter ID definition.

    Attributes:
        mode: OBD mode (1-10).
        pid: PID number (0x00-0xFF).
        name: Parameter name.
        description: Description.
        formula: Decoding formula function.
        unit: Physical unit.
        min_value: Minimum value.
        max_value: Maximum value.
    """

    mode: int
    pid: int
    name: str
    description: str
    formula: Callable[[bytes], float]
    unit: str
    min_value: float
    max_value: float


@dataclass
class OBD2Response:
    """Decoded OBD-II response.

    Attributes:
        mode: OBD mode.
        pid: PID number.
        name: Parameter name.
        value: Decoded value.
        unit: Physical unit.
        timestamp: Message timestamp.
    """

    mode: int
    pid: int
    name: str
    value: float
    unit: str
    timestamp: float


class OBD2Decoder:
    """OBD-II protocol decoder.

    Supports Mode 01 (current data) PIDs.
    """

    # Standard Mode 01 PIDs (most common)
    PIDS: ClassVar[dict[int, PID]] = {
        # PID support bitmaps
        0x00: PID(
            mode=1,
            pid=0x00,
            name="PIDs_supported_01_20",
            description="PIDs supported [01-20]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0x20: PID(
            mode=1,
            pid=0x20,
            name="PIDs_supported_21_40",
            description="PIDs supported [21-40]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0x40: PID(
            mode=1,
            pid=0x40,
            name="PIDs_supported_41_60",
            description="PIDs supported [41-60]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0x60: PID(
            mode=1,
            pid=0x60,
            name="PIDs_supported_61_80",
            description="PIDs supported [61-80]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0x80: PID(
            mode=1,
            pid=0x80,
            name="PIDs_supported_81_A0",
            description="PIDs supported [81-A0]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0xA0: PID(
            mode=1,
            pid=0xA0,
            name="PIDs_supported_A1_C0",
            description="PIDs supported [A1-C0]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        0xC0: PID(
            mode=1,
            pid=0xC0,
            name="PIDs_supported_C1_E0",
            description="PIDs supported [C1-E0]",
            formula=lambda data: int.from_bytes(data[1:5], "big"),
            unit="bitmap",
            min_value=0,
            max_value=0xFFFFFFFF,
        ),
        # Fuel system
        0x03: PID(
            mode=1,
            pid=0x03,
            name="fuel_system_status",
            description="Fuel System Status",
            formula=lambda data: data[1],
            unit="bitmap",
            min_value=0,
            max_value=255,
        ),
        0x04: PID(
            mode=1,
            pid=0x04,
            name="calculated_engine_load",
            description="Calculated Engine Load",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x05: PID(
            mode=1,
            pid=0x05,
            name="coolant_temp",
            description="Engine Coolant Temperature",
            formula=lambda data: data[1] - 40,
            unit="°C",
            min_value=-40,
            max_value=215,
        ),
        0x06: PID(
            mode=1,
            pid=0x06,
            name="short_term_fuel_trim_bank1",
            description="Short Term Fuel Trim - Bank 1",
            formula=lambda data: (data[1] - 128) * 100.0 / 128,
            unit="%",
            min_value=-100,
            max_value=99.22,
        ),
        0x07: PID(
            mode=1,
            pid=0x07,
            name="long_term_fuel_trim_bank1",
            description="Long Term Fuel Trim - Bank 1",
            formula=lambda data: (data[1] - 128) * 100.0 / 128,
            unit="%",
            min_value=-100,
            max_value=99.22,
        ),
        0x08: PID(
            mode=1,
            pid=0x08,
            name="short_term_fuel_trim_bank2",
            description="Short Term Fuel Trim - Bank 2",
            formula=lambda data: (data[1] - 128) * 100.0 / 128,
            unit="%",
            min_value=-100,
            max_value=99.22,
        ),
        0x09: PID(
            mode=1,
            pid=0x09,
            name="long_term_fuel_trim_bank2",
            description="Long Term Fuel Trim - Bank 2",
            formula=lambda data: (data[1] - 128) * 100.0 / 128,
            unit="%",
            min_value=-100,
            max_value=99.22,
        ),
        0x0A: PID(
            mode=1,
            pid=0x0A,
            name="fuel_pressure",
            description="Fuel Pressure (gauge)",
            formula=lambda data: data[1] * 3,
            unit="kPa",
            min_value=0,
            max_value=765,
        ),
        0x0B: PID(
            mode=1,
            pid=0x0B,
            name="intake_manifold_pressure",
            description="Intake Manifold Absolute Pressure",
            formula=lambda data: data[1],
            unit="kPa",
            min_value=0,
            max_value=255,
        ),
        0x0C: PID(
            mode=1,
            pid=0x0C,
            name="engine_rpm",
            description="Engine RPM",
            formula=lambda data: (256 * data[1] + data[2]) / 4,
            unit="rpm",
            min_value=0,
            max_value=16383.75,
        ),
        0x0D: PID(
            mode=1,
            pid=0x0D,
            name="vehicle_speed",
            description="Vehicle Speed",
            formula=lambda data: data[1],
            unit="km/h",
            min_value=0,
            max_value=255,
        ),
        0x0E: PID(
            mode=1,
            pid=0x0E,
            name="timing_advance",
            description="Timing Advance",
            formula=lambda data: (data[1] - 128) / 2,
            unit="° before TDC",
            min_value=-64,
            max_value=63.5,
        ),
        0x0F: PID(
            mode=1,
            pid=0x0F,
            name="intake_air_temp",
            description="Intake Air Temperature",
            formula=lambda data: data[1] - 40,
            unit="°C",
            min_value=-40,
            max_value=215,
        ),
        0x10: PID(
            mode=1,
            pid=0x10,
            name="maf_air_flow_rate",
            description="MAF Air Flow Rate",
            formula=lambda data: (256 * data[1] + data[2]) / 100,
            unit="g/s",
            min_value=0,
            max_value=655.35,
        ),
        0x11: PID(
            mode=1,
            pid=0x11,
            name="throttle_position",
            description="Throttle Position",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        # Oxygen sensors (Bank 1)
        0x14: PID(
            mode=1,
            pid=0x14,
            name="o2_sensor1_voltage",
            description="O2 Sensor 1 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x15: PID(
            mode=1,
            pid=0x15,
            name="o2_sensor2_voltage",
            description="O2 Sensor 2 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x16: PID(
            mode=1,
            pid=0x16,
            name="o2_sensor3_voltage",
            description="O2 Sensor 3 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x17: PID(
            mode=1,
            pid=0x17,
            name="o2_sensor4_voltage",
            description="O2 Sensor 4 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        # Oxygen sensors (Bank 2)
        0x18: PID(
            mode=1,
            pid=0x18,
            name="o2_sensor5_voltage",
            description="O2 Sensor 5 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x19: PID(
            mode=1,
            pid=0x19,
            name="o2_sensor6_voltage",
            description="O2 Sensor 6 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x1A: PID(
            mode=1,
            pid=0x1A,
            name="o2_sensor7_voltage",
            description="O2 Sensor 7 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x1B: PID(
            mode=1,
            pid=0x1B,
            name="o2_sensor8_voltage",
            description="O2 Sensor 8 Voltage",
            formula=lambda data: data[1] / 200,
            unit="V",
            min_value=0,
            max_value=1.275,
        ),
        0x1F: PID(
            mode=1,
            pid=0x1F,
            name="run_time_since_engine_start",
            description="Run Time Since Engine Start",
            formula=lambda data: 256 * data[1] + data[2],
            unit="s",
            min_value=0,
            max_value=65535,
        ),
        0x21: PID(
            mode=1,
            pid=0x21,
            name="distance_traveled_with_mil_on",
            description="Distance Traveled with MIL On",
            formula=lambda data: 256 * data[1] + data[2],
            unit="km",
            min_value=0,
            max_value=65535,
        ),
        0x2C: PID(
            mode=1,
            pid=0x2C,
            name="commanded_egr",
            description="Commanded EGR",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x2D: PID(
            mode=1,
            pid=0x2D,
            name="egr_error",
            description="EGR Error",
            formula=lambda data: (data[1] - 128) * 100.0 / 128,
            unit="%",
            min_value=-100,
            max_value=99.22,
        ),
        0x2F: PID(
            mode=1,
            pid=0x2F,
            name="fuel_level",
            description="Fuel Tank Level Input",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x33: PID(
            mode=1,
            pid=0x33,
            name="absolute_barometric_pressure",
            description="Absolute Barometric Pressure",
            formula=lambda data: data[1],
            unit="kPa",
            min_value=0,
            max_value=255,
        ),
        # Catalyst temperatures
        0x3C: PID(
            mode=1,
            pid=0x3C,
            name="catalyst_temp_b1s1",
            description="Catalyst Temperature - Bank 1, Sensor 1",
            formula=lambda data: (256 * data[1] + data[2]) / 10 - 40,
            unit="°C",
            min_value=-40,
            max_value=6513.5,
        ),
        0x3D: PID(
            mode=1,
            pid=0x3D,
            name="catalyst_temp_b2s1",
            description="Catalyst Temperature - Bank 2, Sensor 1",
            formula=lambda data: (256 * data[1] + data[2]) / 10 - 40,
            unit="°C",
            min_value=-40,
            max_value=6513.5,
        ),
        0x3E: PID(
            mode=1,
            pid=0x3E,
            name="catalyst_temp_b1s2",
            description="Catalyst Temperature - Bank 1, Sensor 2",
            formula=lambda data: (256 * data[1] + data[2]) / 10 - 40,
            unit="°C",
            min_value=-40,
            max_value=6513.5,
        ),
        0x3F: PID(
            mode=1,
            pid=0x3F,
            name="catalyst_temp_b2s2",
            description="Catalyst Temperature - Bank 2, Sensor 2",
            formula=lambda data: (256 * data[1] + data[2]) / 10 - 40,
            unit="°C",
            min_value=-40,
            max_value=6513.5,
        ),
        0x42: PID(
            mode=1,
            pid=0x42,
            name="control_module_voltage",
            description="Control Module Voltage",
            formula=lambda data: (256 * data[1] + data[2]) / 1000,
            unit="V",
            min_value=0,
            max_value=65.535,
        ),
        0x43: PID(
            mode=1,
            pid=0x43,
            name="absolute_load_value",
            description="Absolute Load Value",
            formula=lambda data: (256 * data[1] + data[2]) * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=25700,
        ),
        0x44: PID(
            mode=1,
            pid=0x44,
            name="commanded_equivalence_ratio",
            description="Commanded Equivalence Ratio",
            formula=lambda data: (256 * data[1] + data[2]) / 32768,
            unit="ratio",
            min_value=0,
            max_value=2,
        ),
        0x45: PID(
            mode=1,
            pid=0x45,
            name="relative_throttle_position",
            description="Relative Throttle Position",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x46: PID(
            mode=1,
            pid=0x46,
            name="ambient_air_temp",
            description="Ambient Air Temperature",
            formula=lambda data: data[1] - 40,
            unit="°C",
            min_value=-40,
            max_value=215,
        ),
        0x47: PID(
            mode=1,
            pid=0x47,
            name="absolute_throttle_position_b",
            description="Absolute Throttle Position B",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x48: PID(
            mode=1,
            pid=0x48,
            name="absolute_throttle_position_c",
            description="Absolute Throttle Position C",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x49: PID(
            mode=1,
            pid=0x49,
            name="accelerator_pedal_position_d",
            description="Accelerator Pedal Position D",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x4A: PID(
            mode=1,
            pid=0x4A,
            name="accelerator_pedal_position_e",
            description="Accelerator Pedal Position E",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x4C: PID(
            mode=1,
            pid=0x4C,
            name="commanded_throttle_actuator",
            description="Commanded Throttle Actuator",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x51: PID(
            mode=1,
            pid=0x51,
            name="fuel_type",
            description="Fuel Type",
            formula=lambda data: data[1],
            unit="type",
            min_value=0,
            max_value=23,
        ),
        0x52: PID(
            mode=1,
            pid=0x52,
            name="ethanol_fuel_percentage",
            description="Ethanol Fuel Percentage",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x5A: PID(
            mode=1,
            pid=0x5A,
            name="relative_accelerator_pedal_position",
            description="Relative Accelerator Pedal Position",
            formula=lambda data: data[1] * 100.0 / 255,
            unit="%",
            min_value=0,
            max_value=100,
        ),
        0x5C: PID(
            mode=1,
            pid=0x5C,
            name="engine_oil_temp",
            description="Engine Oil Temperature",
            formula=lambda data: data[1] - 40,
            unit="°C",
            min_value=-40,
            max_value=215,
        ),
    }

    @staticmethod
    def is_obd2_request(message: CANMessage) -> bool:
        """Check if message is an OBD-II request.

        OBD-II requests are typically sent to ID 0x7DF (broadcast)
        or 0x7E0-0x7E7 (specific ECU).

        Args:
            message: CAN message to check.

        Returns:
            True if message appears to be OBD-II request.
        """
        if message.arbitration_id == 0x7DF:
            return True
        return 0x7E0 <= message.arbitration_id <= 0x7E7

    @staticmethod
    def is_obd2_response(message: CANMessage) -> bool:
        """Check if message is an OBD-II response.

        OBD-II responses are typically sent from ID 0x7E8-0x7EF.

        Args:
            message: CAN message to check.

        Returns:
            True if message appears to be OBD-II response.
        """
        return 0x7E8 <= message.arbitration_id <= 0x7EF

    @staticmethod
    def decode(message: CANMessage) -> OBD2Response | None:
        """Decode an OBD-II response message.

        Args:
            message: CAN message (must be OBD-II response).

        Returns:
            OBD2Response if successfully decoded, None otherwise.
        """
        if not OBD2Decoder.is_obd2_response(message):
            return None

        if len(message.data) < 4:
            return None

        # OBD-II response format:
        # Byte 0: Number of additional bytes
        # Byte 1: Mode + 0x40 (e.g., 0x41 for Mode 01 response)
        # Byte 2: PID
        # Byte 3+: Data

        mode_response = message.data[1]
        pid = message.data[2]

        # Verify mode response (should be mode + 0x40)
        if mode_response < 0x40:
            return None

        mode = mode_response - 0x40

        # Only support Mode 01 for now
        if mode != 1:
            return None

        # Look up PID
        if pid not in OBD2Decoder.PIDS:
            return None

        pid_def = OBD2Decoder.PIDS[pid]

        # Decode value
        # Pass data starting from PID byte so formulas can use correct indices
        # Formula expects: [PID, data_byte_0, data_byte_1, ...]
        try:
            value = pid_def.formula(message.data[2:])
        except Exception:
            return None

        return OBD2Response(
            mode=mode,
            pid=pid,
            name=pid_def.name,
            value=value,
            unit=pid_def.unit,
            timestamp=message.timestamp,
        )
