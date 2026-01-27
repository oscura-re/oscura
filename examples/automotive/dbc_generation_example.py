#!/usr/bin/env python3
"""Example: Generate DBC file from reverse-engineered CAN protocol.

This example demonstrates how to create a complete DBC file from
CAN message and signal specifications discovered through reverse engineering.
"""

from pathlib import Path

from oscura.automotive.can.dbc_generator import (
    DBCGenerator,
    DBCMessage,
    DBCNode,
    DBCSignal,
)


def main() -> None:
    """Generate example DBC file for automotive CAN network."""
    # Create DBC generator
    gen = DBCGenerator()

    # Add network nodes (ECUs)
    gen.add_node(DBCNode("ECU_Engine", "Engine Control Unit"))
    gen.add_node(DBCNode("ECU_Transmission", "Transmission Control Unit"))
    gen.add_node(DBCNode("Gateway", "Central Gateway"))
    gen.add_node(DBCNode("Dashboard", "Instrument Cluster"))

    # Add value tables for enumerated signals
    gen.add_value_table(
        "GearPosition",
        {
            0: "Park",
            1: "Reverse",
            2: "Neutral",
            3: "Drive",
            4: "Sport",
            5: "Manual",
        },
    )

    gen.add_value_table(
        "EngineState",
        {
            0: "Off",
            1: "Cranking",
            2: "Running",
            3: "Error",
        },
    )

    # Define engine status message (0x200)
    engine_speed_signal = DBCSignal(
        name="EngineSpeed",
        start_bit=0,
        bit_length=16,
        byte_order="little_endian",
        value_type="unsigned",
        factor=0.25,
        offset=0.0,
        min_value=0.0,
        max_value=16383.75,
        unit="rpm",
        receivers=["Gateway", "Dashboard"],
        comment="Engine rotational speed",
    )

    engine_temp_signal = DBCSignal(
        name="CoolantTemp",
        start_bit=16,
        bit_length=8,
        byte_order="little_endian",
        value_type="unsigned",
        factor=1.0,
        offset=-40.0,
        min_value=-40.0,
        max_value=215.0,
        unit="°C",
        receivers=["Gateway", "Dashboard"],
        comment="Engine coolant temperature",
    )

    engine_state_signal = DBCSignal(
        name="EngineState",
        start_bit=24,
        bit_length=8,
        byte_order="little_endian",
        value_type="unsigned",
        factor=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=3.0,
        unit="",
        receivers=["Gateway"],
        value_table={0: "Off", 1: "Cranking", 2: "Running", 3: "Error"},
        comment="Current engine state",
    )

    engine_msg = DBCMessage(
        message_id=0x200,
        name="EngineStatus",
        dlc=8,
        sender="ECU_Engine",
        signals=[engine_speed_signal, engine_temp_signal, engine_state_signal],
        comment="Engine status and temperature",
        cycle_time=10,
        send_type="Cyclic",
    )

    gen.add_message(engine_msg)

    # Define vehicle speed message (0x210)
    vehicle_speed_signal = DBCSignal(
        name="VehicleSpeed",
        start_bit=0,
        bit_length=16,
        byte_order="little_endian",
        value_type="unsigned",
        factor=0.01,
        offset=0.0,
        min_value=0.0,
        max_value=655.35,
        unit="km/h",
        receivers=["Dashboard", "Gateway"],
        comment="Vehicle speed from wheel sensors",
    )

    speed_msg = DBCMessage(
        message_id=0x210,
        name="VehicleSpeed",
        dlc=8,
        sender="ECU_Engine",
        signals=[vehicle_speed_signal],
        comment="Vehicle speed information",
        cycle_time=20,
        send_type="Cyclic",
    )

    gen.add_message(speed_msg)

    # Define transmission status message (0x220)
    gear_position_signal = DBCSignal(
        name="CurrentGear",
        start_bit=0,
        bit_length=8,
        byte_order="little_endian",
        value_type="unsigned",
        factor=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=5.0,
        unit="",
        receivers=["Dashboard", "Gateway"],
        value_table={0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive", 4: "Sport", 5: "Manual"},
        comment="Current gear position",
    )

    trans_temp_signal = DBCSignal(
        name="TransmissionTemp",
        start_bit=8,
        bit_length=8,
        byte_order="little_endian",
        value_type="unsigned",
        factor=1.0,
        offset=-40.0,
        min_value=-40.0,
        max_value=215.0,
        unit="°C",
        receivers=["Gateway"],
        comment="Transmission fluid temperature",
    )

    trans_msg = DBCMessage(
        message_id=0x220,
        name="TransmissionStatus",
        dlc=8,
        sender="ECU_Transmission",
        signals=[gear_position_signal, trans_temp_signal],
        comment="Transmission status and gear position",
        cycle_time=50,
        send_type="Cyclic",
    )

    gen.add_message(trans_msg)

    # Generate DBC file
    output_path = Path("vehicle_network.dbc")
    gen.generate(output_path)

    print(f"Generated DBC file: {output_path}")
    print(f"  Nodes: {len(gen.nodes)}")
    print(f"  Messages: {len(gen.messages)}")
    print(f"  Total signals: {sum(len(msg.signals) for msg in gen.messages)}")

    # Validate the generated file
    content = output_path.read_text()
    if gen.validate_dbc(content):
        print("✓ DBC validation passed")
    else:
        print("✗ DBC validation failed")

    # Display file contents
    print("\nGenerated DBC content (first 1000 chars):")
    print("-" * 80)
    print(content[:1000])
    print("..." if len(content) > 1000 else "")


if __name__ == "__main__":
    main()
