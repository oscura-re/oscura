"""J1939 protocol decoder.

This module implements J1939 (SAE J1939) protocol decoding for heavy-duty
vehicles including PGN extraction and common parameter decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage

__all__ = ["J1939Decoder", "J1939Message", "extract_pgn"]


@dataclass
class J1939Message:
    """Decoded J1939 message.

    Attributes:
        pgn: Parameter Group Number.
        priority: Message priority (0-7).
        source_address: Source address.
        destination_address: Destination address (0xFF for broadcast).
        data: Message data bytes.
        timestamp: Message timestamp.
    """

    pgn: int
    priority: int
    source_address: int
    destination_address: int
    data: bytes
    timestamp: float


def extract_pgn(can_id: int) -> tuple[int, int, int, int]:
    """Extract J1939 components from 29-bit CAN ID.

    J1939 uses extended 29-bit CAN IDs with this structure:
    - Priority (bits 26-28): 3 bits
    - Reserved (bit 25): 1 bit
    - Data Page (bit 24): 1 bit
    - PDU Format (bits 16-23): 8 bits
    - PDU Specific (bits 8-15): 8 bits
    - Source Address (bits 0-7): 8 bits

    Args:
        can_id: 29-bit extended CAN ID.

    Returns:
        Tuple of (pgn, priority, destination_address, source_address).
    """
    # Extract components
    priority = (can_id >> 26) & 0x7
    # reserved = (can_id >> 25) & 0x1  # Not used in PGN calculation
    data_page = (can_id >> 24) & 0x1
    pdu_format = (can_id >> 16) & 0xFF
    pdu_specific = (can_id >> 8) & 0xFF
    source_address = can_id & 0xFF

    # Calculate PGN
    # If PDU Format < 240, PDU Specific is Destination Address
    # If PDU Format >= 240, PDU Specific is Group Extension
    if pdu_format < 240:
        # PDU1 format - destination-specific
        pgn = (data_page << 16) | (pdu_format << 8)
        destination_address = pdu_specific
    else:
        # PDU2 format - broadcast
        pgn = (data_page << 16) | (pdu_format << 8) | pdu_specific
        destination_address = 0xFF  # Broadcast

    return pgn, priority, destination_address, source_address


class J1939Decoder:
    """J1939 protocol decoder.

    Decodes J1939 messages from extended CAN frames with support for
    100+ Parameter Group Numbers (PGNs) and signal extraction.
    """

    # Comprehensive PGN names (100+ common PGNs)
    PGN_NAMES: ClassVar[dict[int, str]] = {
        # Engine Parameters - Core (61440-61695, 0xF000-0xF0FF)
        0xF000: "Electronic Retarder Controller 1 (ERC1)",
        0xF001: "Electronic Brake Controller 1 (EBC1)",
        0xF002: "Electronic Transmission Controller 1 (ETC1)",
        0xF003: "Electronic Engine Controller 2 (EEC2)",
        0xF004: "Electronic Engine Controller 1 (EEC1)",
        0xF005: "Electronic Transmission Controller 2 (ETC2)",
        0xF009: "Vehicle Dynamic Stability Control 2 (VDC2)",
        0xF010: "Aftertreatment 1 Intake Gas 1 (AT1IG1)",
        0xF011: "Aftertreatment 1 Outlet Gas 1 (AT1OG1)",
        # Engine Fluid Levels & Temperatures (65248-65279, 0xFEE0-0xFEFF)
        0xFEE0: "Electronic Engine Controller 3 (EEC3)",
        0xFEE5: "Engine Hours, Revolutions",
        0xFEE6: "Time/Date",
        0xFEE7: "Vehicle Hours",
        0xFEE8: "Vehicle Direction/Speed",
        0xFEE9: "Vehicle Weight",
        0xFEEA: "Vehicle Identification",
        0xFEEB: "Component Identification",
        0xFEEC: "Vehicle Distance",
        0xFEED: "Shutdown",
        0xFEEE: "Engine Temperature 1",
        0xFEEF: "Engine Fluid Level/Pressure 1",
        0xFEF0: "Power Takeoff Information",
        0xFEF1: "Cruise Control/Vehicle Speed 1 (CCVS1)",
        0xFEF2: "Fuel Economy (Liquid)",
        0xFEF3: "Engine Configuration 1",
        0xFEF4: "Fuel Consumption (Gaseous)",
        0xFEF5: "Ambient Conditions",
        0xFEF6: "Inlet/Exhaust Conditions 1",
        0xFEF7: "Vehicle Electrical Power 1",
        0xFEF8: "Transmission Fluids 1",
        0xFEF9: "Air Supply Pressure",
        0xFEFA: "Vehicle Weight",
        0xFEFB: "Engine Speed/Load Factor",
        0xFEFC: "Fan Drive",
        0xFEFD: "Vehicle Position",
        0xFEFE: "Engine Temperature 2",
        0xFEFF: "Electronic Engine Controller 4 (EEC4)",
        # Transmission & Drivetrain (65184-65247, 0xFEC0-0xFEDF)
        0xFEC0: "Transmission Configuration",
        0xFEC1: "High Resolution Vehicle Distance",
        0xFEC2: "High Resolution Fuel Consumption (Liquid)",
        0xFEC3: "High Resolution Fuel Economy (Liquid)",
        0xFEC5: "Aftertreatment 1 Diesel Exhaust Fluid Tank 1 Info",
        0xFEC6: "Aftertreatment 1 Diesel Oxidation Catalyst 1",
        0xFEC7: "Aftertreatment 1 Diesel Particulate Filter 1",
        0xFEC8: "Aftertreatment 1 Intake Gas",
        0xFEC9: "Aftertreatment 1 Outlet Gas",
        0xFECA: "DM1 - Active Diagnostic Trouble Codes",
        0xFECB: "DM2 - Previously Active Diagnostic Trouble Codes",
        0xFECC: "DM3 - Diagnostic Data Clear/Reset Previously Active DTCs",
        0xFECD: "DM4 - Freeze Frame Parameters",
        0xFECE: "DM5 - Diagnostic Readiness 1",
        0xFECF: "DM6 - Emission-Related Pending DTCs",
        0xFED0: "DM7 - Command Non-Continuously Monitored Test",
        0xFED1: "DM8 - Test Results",
        0xFED2: "DM9 - Request Test Results",
        0xFED3: "DM11 - Diagnostic Data Clear/Reset Active DTCs",
        0xFED4: "DM12 - Emissions-Related Active DTCs",
        0xFED5: "DM13 - Stop Start Broadcast",
        0xFED6: "DM14 - Memory Access Response",
        0xFED7: "DM15 - Memory Access",
        0xFED8: "DM16 - Binary Data Transfer",
        0xFED9: "DM17 - Binary Data Transfer Response",
        0xFEDA: "DM18 - Binary Data Transfer",
        0xFEDB: "DM19 - Calibration Information",
        0xFEDC: "DM20 - Monitor Performance Ratio",
        0xFEDD: "DM21 - Diagnostic Readiness 2",
        0xFEDE: "DM22 - Individual Clear/Reset Active & Previously Active DTC",
        0xFEDF: "DM23 - Emission-Related Previously Active DTC",
        # Brake & Wheels (65120-65183, 0xFE80-0xFEBF)
        0xFE80: "Tire Condition",
        0xFE81: "Tire Pressure",
        0xFE82: "Tire Temperature",
        0xFE83: "Tire Pressure Control Unit",
        0xFE90: "Hydraulic Pressure 1",
        0xFE91: "Hydraulic Pressure 2",
        0xFE92: "Fuel Consumption (Liquid) 1",
        0xFE93: "Fuel Economy (Liquid) 1",
        0xFEA0: "Axle Information",
        0xFEA1: "Engine Torque/Speed",
        0xFEA2: "Electronic Transmission Controller 3 (ETC3)",
        0xFEA3: "Electronic Transmission Controller 4 (ETC4)",
        0xFEA4: "Electronic Transmission Controller 5 (ETC5)",
        0xFEA5: "Electronic Engine Controller 5 (EEC5)",
        0xFEA6: "Electronic Engine Controller 6 (EEC6)",
        0xFEA7: "Electronic Engine Controller 7 (EEC7)",
        0xFEB0: "Axle Weight",
        0xFEB1: "Trailer Weight",
        0xFEB2: "Cargo Weight",
        0xFEB3: "Trip Fuel Economy (Liquid)",
        0xFEB4: "Trip Fuel (Liquid)",
        0xFEB5: "Trip Time",
        0xFEB6: "Trip Shutdown Information",
        0xFEB7: "Fuel Level 1",
        0xFEB8: "Fuel Level 2",
        0xFEB9: "Auxiliary Water Pump Pressure",
        0xFEBA: "Coolant Filter Differential Pressure",
        0xFEBB: "Engine Exhaust Gas Recirculation 1 (EGR1)",
        0xFEBC: "Engine Exhaust Gas Recirculation 2 (EGR2)",
        0xFEBD: "Engine Exhaust Gas Recirculation 3 (EGR3)",
        0xFEBE: "Aftertreatment 1 Diesel Particulate Filter 2",
        0xFEBF: "Wheel Speed Information",
        # Aftertreatment & Emissions (65024-65119, 0xFE40-0xFE7F)
        0xFE40: "Aftertreatment 1 SCR Exhaust Gas Temperature 1",
        0xFE41: "Aftertreatment 1 SCR Dosing System Information 1",
        0xFE42: "Aftertreatment 1 Intake NOx",
        0xFE43: "Aftertreatment 1 Outlet NOx",
        0xFE44: "Aftertreatment 2 Intake Gas 1",
        0xFE45: "Aftertreatment 2 Outlet Gas 1",
        0xFE46: "Aftertreatment 2 SCR Exhaust Gas Temperature 1",
        0xFE47: "Aftertreatment 2 SCR Dosing System Information 1",
        0xFE48: "Aftertreatment 2 Intake NOx",
        0xFE49: "Aftertreatment 2 Outlet NOx",
        0xFE4A: "Fuel Information 1 (Liquid)",
        0xFE4B: "Fuel Information 2 (Liquid)",
        0xFE4C: "Fuel Information 3 (Liquid)",
        0xFE4D: "Engine Gas Flow Rate",
        0xFE4E: "Engine Throttle Valve 1",
        0xFE4F: "Engine Throttle Valve 2",
        0xFE50: "Aftertreatment 1 Diesel Particulate Filter 3",
        0xFE51: "Aftertreatment 1 Diesel Particulate Filter 4",
        0xFE52: "Aftertreatment 1 SCR Dosing System Requests",
        0xFE53: "Aftertreatment 1 Fuel Control 1",
        0xFE54: "Aftertreatment 1 Fuel Control 2",
        0xFE55: "Aftertreatment 2 Diesel Particulate Filter 1",
        0xFE56: "Aftertreatment 1 Diesel Exhaust Fluid Tank 1 Information",
        0xFE57: "Aftertreatment 2 Diesel Exhaust Fluid Tank 1 Information",
        0xFE58: "Fuel Information (Gaseous)",
        0xFE59: "Aftertreatment 1 Air Control 1",
        0xFE5A: "Aftertreatment 2 Air Control 1",
        0xFE5B: "Aftertreatment 1 Diesel Particulate Filter Control 1",
        0xFE5C: "Aftertreatment 2 Diesel Particulate Filter Control 1",
        # Cab Climate & Lighting (64256-64511, 0xFB00-0xFBFF)
        0xFB00: "Cab Climate Control Status 1",
        0xFB01: "Cab Climate Control Status 2",
        0xFB02: "Cab Climate Control Command 1",
        0xFB03: "Cab Climate Control Command 2",
        # Additional Common PGNs
        0xC100: "DM1 - Active Diagnostic Trouble Codes (Request)",
        0xC200: "DM13 - Stop Start Broadcast (Request)",
        0xC300: "DM2 - Previously Active DTCs (Request)",
        0xFF00: "Aftertreatment 1 Diesel Particulate Filter Control",
        # Proprietary PGNs (common ranges)
        61184: "Electronic Retarder Controller 1 (ERC1)",
        61185: "Electronic Brake Controller 2 (EBC2)",
        61186: "Electronic Transmission Controller 1 (ETC1)",
        61187: "Electronic Engine Controller 2 (EEC2)",
        61188: "Electronic Engine Controller 1 (EEC1)",
        61189: "Electronic Transmission Controller 2 (ETC2)",
        61190: "Turbocharger 1",
        61191: "Turbocharger 2",
        61192: "Air Intake Conditions",
        61193: "Exhaust Gas Recirculation",
        61194: "Fuel System",
        61195: "Alternator Information",
        61196: "Intake Manifold Temperature 1",
        61197: "Exhaust Port Temperature",
        61198: "Engine Oil Information",
        61199: "Engine Coolant Information",
        61200: "Fuel Delivery Information",
    }

    # Signal definitions for common PGNs
    # Format: {pgn: {signal_name: {start_byte, bit_offset, length_bits, scale, offset, unit}}}
    PGN_SIGNALS: ClassVar[dict[int, dict[str, dict[str, int | float | str]]]] = {
        0xF004: {  # Electronic Engine Controller 1 (EEC1)
            "engine_torque_mode": {
                "byte": 0,
                "bit": 0,
                "length": 4,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "driver_demand_torque": {
                "byte": 1,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": -125,
                "unit": "%",
            },
            "actual_engine_torque": {
                "byte": 2,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": -125,
                "unit": "%",
            },
            "engine_speed": {
                "byte": 3,
                "bit": 0,
                "length": 16,
                "scale": 0.125,
                "offset": 0,
                "unit": "rpm",
            },
            "source_address": {
                "byte": 5,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
        },
        0xF003: {  # Electronic Engine Controller 2 (EEC2)
            "accelerator_pedal_position": {
                "byte": 1,
                "bit": 0,
                "length": 8,
                "scale": 0.4,
                "offset": 0,
                "unit": "%",
            },
            "engine_percent_load": {
                "byte": 2,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": 0,
                "unit": "%",
            },
        },
        0xFEF1: {  # Cruise Control/Vehicle Speed 1
            "wheel_based_speed": {
                "byte": 1,
                "bit": 0,
                "length": 16,
                "scale": 1 / 256,
                "offset": 0,
                "unit": "km/h",
            },
            "cruise_control_active": {
                "byte": 3,
                "bit": 0,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "brake_switch": {
                "byte": 3,
                "bit": 2,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "clutch_switch": {
                "byte": 3,
                "bit": 4,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
        },
        0xFEEE: {  # Engine Temperature 1
            "coolant_temperature": {
                "byte": 0,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": -40,
                "unit": "°C",
            },
            "fuel_temperature": {
                "byte": 1,
                "bit": 0,
                "length": 8,
                "scale": 1,
                "offset": -40,
                "unit": "°C",
            },
            "oil_temperature": {
                "byte": 2,
                "bit": 0,
                "length": 16,
                "scale": 0.03125,
                "offset": -273,
                "unit": "°C",
            },
            "turbo_oil_temperature": {
                "byte": 4,
                "bit": 0,
                "length": 16,
                "scale": 0.03125,
                "offset": -273,
                "unit": "°C",
            },
        },
        0xFEEF: {  # Engine Fluid Level/Pressure 1
            "fuel_delivery_pressure": {
                "byte": 0,
                "bit": 0,
                "length": 8,
                "scale": 4,
                "offset": 0,
                "unit": "kPa",
            },
            "oil_pressure": {
                "byte": 3,
                "bit": 0,
                "length": 8,
                "scale": 4,
                "offset": 0,
                "unit": "kPa",
            },
            "crankcase_pressure": {
                "byte": 5,
                "bit": 0,
                "length": 16,
                "scale": 0.125,
                "offset": -250,
                "unit": "kPa",
            },
            "coolant_pressure": {
                "byte": 7,
                "bit": 0,
                "length": 8,
                "scale": 2,
                "offset": 0,
                "unit": "kPa",
            },
        },
        0xFEF2: {  # Fuel Economy (Liquid)
            "fuel_rate": {
                "byte": 0,
                "bit": 0,
                "length": 16,
                "scale": 0.05,
                "offset": 0,
                "unit": "L/h",
            },
            "instantaneous_fuel_economy": {
                "byte": 2,
                "bit": 0,
                "length": 16,
                "scale": 1 / 512,
                "offset": 0,
                "unit": "km/L",
            },
            "average_fuel_economy": {
                "byte": 4,
                "bit": 0,
                "length": 16,
                "scale": 1 / 512,
                "offset": 0,
                "unit": "km/L",
            },
        },
        0xFEF5: {  # Ambient Conditions
            "barometric_pressure": {
                "byte": 0,
                "bit": 0,
                "length": 8,
                "scale": 0.5,
                "offset": 0,
                "unit": "kPa",
            },
            "ambient_air_temperature": {
                "byte": 3,
                "bit": 0,
                "length": 16,
                "scale": 0.03125,
                "offset": -273,
                "unit": "°C",
            },
            "ambient_air_humidity": {
                "byte": 6,
                "bit": 0,
                "length": 8,
                "scale": 0.4,
                "offset": 0,
                "unit": "%",
            },
        },
        0xFEF7: {  # Vehicle Electrical Power 1
            "battery_potential": {
                "byte": 4,
                "bit": 0,
                "length": 16,
                "scale": 0.05,
                "offset": 0,
                "unit": "V",
            },
            "alternator_current": {
                "byte": 6,
                "bit": 0,
                "length": 16,
                "scale": 1,
                "offset": -125,
                "unit": "A",
            },
        },
        0xFEE5: {  # Engine Hours, Revolutions
            "total_engine_hours": {
                "byte": 0,
                "bit": 0,
                "length": 32,
                "scale": 0.05,
                "offset": 0,
                "unit": "hours",
            },
            "total_engine_revolutions": {
                "byte": 4,
                "bit": 0,
                "length": 32,
                "scale": 1000,
                "offset": 0,
                "unit": "revolutions",
            },
        },
        0xFECA: {  # DM1 - Active Diagnostic Trouble Codes
            "lamp_status_malfunction_indicator": {
                "byte": 0,
                "bit": 0,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "lamp_status_red_stop": {
                "byte": 0,
                "bit": 2,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "lamp_status_amber_warning": {
                "byte": 0,
                "bit": 4,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
            "lamp_status_protect": {
                "byte": 0,
                "bit": 6,
                "length": 2,
                "scale": 1,
                "offset": 0,
                "unit": "",
            },
        },
        0xFEC1: {  # High Resolution Vehicle Distance
            "high_resolution_total_distance": {
                "byte": 0,
                "bit": 0,
                "length": 32,
                "scale": 5,
                "offset": 0,
                "unit": "m",
            },
            "high_resolution_trip_distance": {
                "byte": 4,
                "bit": 0,
                "length": 32,
                "scale": 5,
                "offset": 0,
                "unit": "m",
            },
        },
        0xFEBF: {  # Wheel Speed Information
            "front_axle_speed": {
                "byte": 0,
                "bit": 0,
                "length": 16,
                "scale": 1 / 256,
                "offset": 0,
                "unit": "km/h",
            },
            "rear_axle_1_speed": {
                "byte": 2,
                "bit": 0,
                "length": 16,
                "scale": 1 / 256,
                "offset": 0,
                "unit": "km/h",
            },
        },
    }

    @staticmethod
    def is_j1939(message: CANMessage) -> bool:
        """Check if message uses J1939 protocol.

        Args:
            message: CAN message to check.

        Returns:
            True if message appears to be J1939 (extended ID).
        """
        return message.is_extended

    @staticmethod
    def decode(message: CANMessage) -> J1939Message:
        """Decode J1939 message from CAN frame.

        Args:
            message: CAN message with extended ID.

        Returns:
            J1939Message with decoded components.

        Raises:
            ValueError: If message is not extended frame.
        """
        if not message.is_extended:
            raise ValueError("J1939 requires extended (29-bit) CAN ID")

        pgn, priority, dest_addr, src_addr = extract_pgn(message.arbitration_id)

        return J1939Message(
            pgn=pgn,
            priority=priority,
            source_address=src_addr,
            destination_address=dest_addr,
            data=message.data,
            timestamp=message.timestamp,
        )

    @staticmethod
    def get_pgn_name(pgn: int) -> str:
        """Get name for PGN if known.

        Args:
            pgn: Parameter Group Number.

        Returns:
            PGN name or hex string if unknown.
        """
        return J1939Decoder.PGN_NAMES.get(pgn, f"PGN_0x{pgn:05X}")

    @staticmethod
    def extract_signal(data: bytes, byte_pos: int, bit_pos: int, length_bits: int) -> int:
        """Extract signal value from message data.

        Args:
            data: Message data bytes.
            byte_pos: Starting byte position (0-indexed).
            bit_pos: Starting bit position within byte (0-7).
            length_bits: Number of bits to extract.

        Returns:
            Raw signal value as integer.
        """
        if byte_pos >= len(data):
            return 0

        value = 0
        bits_read = 0

        # Read bits across multiple bytes if needed
        current_byte = byte_pos
        current_bit = bit_pos

        while bits_read < length_bits and current_byte < len(data):
            # How many bits to read from current byte
            bits_available = 8 - current_bit
            bits_to_read = min(length_bits - bits_read, bits_available)

            # Extract bits from current byte
            mask = ((1 << bits_to_read) - 1) << current_bit
            byte_value = (data[current_byte] & mask) >> current_bit

            # Add to result
            value |= byte_value << bits_read

            # Move to next byte
            bits_read += bits_to_read
            current_byte += 1
            current_bit = 0

        return value

    @staticmethod
    def decode_signal(message: J1939Message, signal_name: str) -> dict[str, float | str | None]:
        """Decode a specific signal from a J1939 message.

        Args:
            message: Decoded J1939 message.
            signal_name: Name of signal to decode.

        Returns:
            Dictionary with 'value', 'unit', and 'raw' keys.
            Returns None values if signal or PGN not found.
        """
        # Check if PGN has signal definitions
        if message.pgn not in J1939Decoder.PGN_SIGNALS:
            return {"value": None, "unit": None, "raw": None}

        signals = J1939Decoder.PGN_SIGNALS[message.pgn]

        if signal_name not in signals:
            return {"value": None, "unit": None, "raw": None}

        sig_def = signals[signal_name]

        # Extract raw value
        byte_val = sig_def["byte"]
        bit_val = sig_def["bit"]
        length_val = sig_def["length"]
        scale_val = sig_def["scale"]
        offset_val = sig_def["offset"]

        if (
            not isinstance(byte_val, int)
            or not isinstance(bit_val, int)
            or not isinstance(length_val, int)
        ):
            return {"value": None, "unit": None, "raw": None}
        if not isinstance(scale_val, (int, float)) or not isinstance(offset_val, (int, float)):
            return {"value": None, "unit": None, "raw": None}

        raw_value = J1939Decoder.extract_signal(
            message.data,
            byte_val,
            bit_val,
            length_val,
        )

        # Apply scaling and offset
        scaled_value = raw_value * scale_val + offset_val

        return {
            "value": scaled_value,
            "unit": sig_def["unit"],
            "raw": raw_value,
        }

    @staticmethod
    def decode_all_signals(message: J1939Message) -> dict[str, dict[str, float | str | None]]:
        """Decode all signals from a J1939 message.

        Args:
            message: Decoded J1939 message.

        Returns:
            Dictionary mapping signal names to their decoded values.
            Empty dict if PGN has no signal definitions.
        """
        if message.pgn not in J1939Decoder.PGN_SIGNALS:
            return {}

        signals = J1939Decoder.PGN_SIGNALS[message.pgn]
        result = {}

        for signal_name in signals:
            result[signal_name] = J1939Decoder.decode_signal(message, signal_name)

        return result
