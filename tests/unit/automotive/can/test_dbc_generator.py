"""Tests for DBC file generator.

This module tests the DBC (CAN Database) file generation functionality,
ensuring valid DBC files are created from message and signal specifications.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from oscura.automotive.can.dbc_generator import (
    DBCGenerator,
    DBCMessage,
    DBCNode,
    DBCSignal,
)


class TestDBCSignal:
    """Test DBCSignal dataclass."""

    def test_signal_creation(self) -> None:
        """Test creating a signal with default values."""
        signal = DBCSignal(
            name="TestSignal",
            start_bit=0,
            bit_length=16,
        )

        assert signal.name == "TestSignal"
        assert signal.start_bit == 0
        assert signal.bit_length == 16
        assert signal.byte_order == "little_endian"
        assert signal.value_type == "unsigned"
        assert signal.factor == 1.0
        assert signal.offset == 0.0
        assert signal.min_value == 0.0
        assert signal.max_value == 0.0
        assert signal.unit == ""
        assert signal.receivers == ["Vector__XXX"]
        assert signal.value_table is None
        assert signal.comment == ""
        assert signal.multiplexer_indicator is None

    def test_signal_with_all_fields(self) -> None:
        """Test creating a signal with all fields specified."""
        value_table = {0: "Off", 1: "On"}
        signal = DBCSignal(
            name="EngineSpeed",
            start_bit=8,
            bit_length=16,
            byte_order="big_endian",
            value_type="signed",
            factor=0.25,
            offset=0.0,
            min_value=0.0,
            max_value=16383.75,
            unit="rpm",
            receivers=["Gateway", "Dashboard"],
            value_table=value_table,
            comment="Engine rotational speed",
            multiplexer_indicator="M",
        )

        assert signal.name == "EngineSpeed"
        assert signal.byte_order == "big_endian"
        assert signal.value_type == "signed"
        assert signal.factor == 0.25
        assert signal.unit == "rpm"
        assert signal.receivers == ["Gateway", "Dashboard"]
        assert signal.value_table == value_table
        assert signal.comment == "Engine rotational speed"
        assert signal.multiplexer_indicator == "M"

    def test_signal_validation_bit_length_too_small(self) -> None:
        """Test signal validation rejects bit_length < 1."""
        with pytest.raises(ValueError, match="bit_length must be 1-64"):
            DBCSignal(name="Test", start_bit=0, bit_length=0)

    def test_signal_validation_bit_length_too_large(self) -> None:
        """Test signal validation rejects bit_length > 64."""
        with pytest.raises(ValueError, match="bit_length must be 1-64"):
            DBCSignal(name="Test", start_bit=0, bit_length=65)

    def test_signal_validation_negative_start_bit(self) -> None:
        """Test signal validation rejects negative start_bit."""
        with pytest.raises(ValueError, match="start_bit must be >= 0"):
            DBCSignal(name="Test", start_bit=-1, bit_length=8)


class TestDBCMessage:
    """Test DBCMessage dataclass."""

    def test_message_creation(self) -> None:
        """Test creating a message with default values."""
        msg = DBCMessage(
            message_id=0x200,
            name="TestMessage",
            dlc=8,
        )

        assert msg.message_id == 0x200
        assert msg.name == "TestMessage"
        assert msg.dlc == 8
        assert msg.sender == "Vector__XXX"
        assert msg.signals == []
        assert msg.comment == ""
        assert msg.cycle_time is None
        assert msg.send_type == "Cyclic"

    def test_message_with_signals(self) -> None:
        """Test creating a message with signals."""
        signal1 = DBCSignal(name="Signal1", start_bit=0, bit_length=8)
        signal2 = DBCSignal(name="Signal2", start_bit=8, bit_length=16)

        msg = DBCMessage(
            message_id=0x100,
            name="TestMsg",
            dlc=8,
            sender="ECU_Test",
            signals=[signal1, signal2],
            comment="Test message",
            cycle_time=10,
            send_type="Event",
        )

        assert len(msg.signals) == 2
        assert msg.comment == "Test message"
        assert msg.cycle_time == 10
        assert msg.send_type == "Event"

    def test_message_validation_negative_id(self) -> None:
        """Test message validation rejects negative message_id."""
        with pytest.raises(ValueError, match="message_id must be >= 0"):
            DBCMessage(message_id=-1, name="Test", dlc=8)

    def test_message_validation_dlc_too_large(self) -> None:
        """Test message validation rejects dlc > 64."""
        with pytest.raises(ValueError, match="dlc must be 0-64"):
            DBCMessage(message_id=0x100, name="Test", dlc=65)

    def test_message_validation_negative_dlc(self) -> None:
        """Test message validation rejects negative dlc."""
        with pytest.raises(ValueError, match="dlc must be 0-64"):
            DBCMessage(message_id=0x100, name="Test", dlc=-1)


class TestDBCNode:
    """Test DBCNode dataclass."""

    def test_node_creation(self) -> None:
        """Test creating a node."""
        node = DBCNode(name="ECU_Test")
        assert node.name == "ECU_Test"
        assert node.comment == ""

    def test_node_with_comment(self) -> None:
        """Test creating a node with comment."""
        node = DBCNode(name="ECU_Engine", comment="Engine Control Unit")
        assert node.comment == "Engine Control Unit"


class TestDBCGenerator:
    """Test DBCGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator initializes with empty collections."""
        gen = DBCGenerator()
        assert gen.nodes == []
        assert gen.messages == []
        assert gen.value_tables == {}
        assert gen.environment_variables == {}

    def test_add_node(self) -> None:
        """Test adding nodes to generator."""
        gen = DBCGenerator()
        node1 = DBCNode("ECU_1")
        node2 = DBCNode("ECU_2")

        gen.add_node(node1)
        gen.add_node(node2)

        assert len(gen.nodes) == 2
        assert gen.nodes[0] == node1
        assert gen.nodes[1] == node2

    def test_add_message(self) -> None:
        """Test adding messages to generator."""
        gen = DBCGenerator()
        msg1 = DBCMessage(0x100, "Msg1", 8)
        msg2 = DBCMessage(0x200, "Msg2", 8)

        gen.add_message(msg1)
        gen.add_message(msg2)

        assert len(gen.messages) == 2
        assert gen.messages[0] == msg1
        assert gen.messages[1] == msg2

    def test_add_value_table(self) -> None:
        """Test adding value tables to generator."""
        gen = DBCGenerator()
        values = {0: "Off", 1: "On", 2: "Error"}

        gen.add_value_table("PowerState", values)

        assert "PowerState" in gen.value_tables
        assert gen.value_tables["PowerState"] == values

    def test_generate_header(self) -> None:
        """Test DBC header generation."""
        gen = DBCGenerator()
        header = gen._generate_header()

        assert 'VERSION "1.0"' in header
        assert "NS_ :" in header
        assert "BS_:" in header
        assert "\tCM_" in header
        assert "\tBA_DEF_" in header

    def test_generate_nodes_empty(self) -> None:
        """Test node generation with no nodes."""
        gen = DBCGenerator()
        nodes = gen._generate_nodes()
        assert nodes == "BU_:"

    def test_generate_nodes_multiple(self) -> None:
        """Test node generation with multiple nodes."""
        gen = DBCGenerator()
        gen.add_node(DBCNode("ECU_Engine"))
        gen.add_node(DBCNode("Gateway"))
        gen.add_node(DBCNode("Dashboard"))

        nodes = gen._generate_nodes()
        assert nodes == "BU_: ECU_Engine Gateway Dashboard"

    def test_generate_value_tables_empty(self) -> None:
        """Test value table generation with no tables."""
        gen = DBCGenerator()
        tables = gen._generate_value_tables()
        assert tables == ""

    def test_generate_value_tables(self) -> None:
        """Test value table generation."""
        gen = DBCGenerator()
        gen.add_value_table("GearPosition", {0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive"})
        gen.add_value_table("PowerState", {0: "Off", 1: "On"})

        tables = gen._generate_value_tables()
        assert "VAL_TABLE_ GearPosition" in tables
        assert '0 "Park"' in tables
        assert '3 "Drive"' in tables
        assert "VAL_TABLE_ PowerState" in tables
        assert '0 "Off"' in tables

    def test_generate_messages_empty(self) -> None:
        """Test message generation with no messages."""
        gen = DBCGenerator()
        messages = gen._generate_messages()
        assert messages == ""

    def test_generate_messages_without_signals(self) -> None:
        """Test message generation without signals."""
        gen = DBCGenerator()
        gen.add_message(DBCMessage(0x200, "TestMsg", 8, "ECU_Test"))

        messages = gen._generate_messages()
        assert "BO_ 512 TestMsg: 8 ECU_Test" in messages

    def test_generate_messages_with_signals(self) -> None:
        """Test message generation with signals."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="Speed",
            start_bit=0,
            bit_length=16,
            factor=0.1,
            unit="km/h",
            receivers=["Dashboard"],
        )
        msg = DBCMessage(0x100, "SpeedData", 8, "ECU_Engine", signals=[signal])
        gen.add_message(msg)

        messages = gen._generate_messages()
        assert "BO_ 256 SpeedData: 8 ECU_Engine" in messages
        assert 'SG_ Speed : 0|16@1+ (0.1,0.0) [0.0|0.0] "km/h" Dashboard' in messages

    def test_generate_signal_little_endian_unsigned(self) -> None:
        """Test signal generation for little-endian unsigned signal."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="TestSignal",
            start_bit=8,
            bit_length=16,
            byte_order="little_endian",
            value_type="unsigned",
            factor=0.5,
            offset=10.0,
            min_value=0.0,
            max_value=1000.0,
            unit="rpm",
            receivers=["ECU_1", "ECU_2"],
        )

        sig_line = gen._generate_signal(signal, 0x200)
        assert sig_line == ' SG_ TestSignal : 8|16@1+ (0.5,10.0) [0.0|1000.0] "rpm" ECU_1,ECU_2'

    def test_generate_signal_big_endian_signed(self) -> None:
        """Test signal generation for big-endian signed signal."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="Temperature",
            start_bit=0,
            bit_length=16,
            byte_order="big_endian",
            value_type="signed",
            factor=0.1,
            offset=-40.0,
            min_value=-40.0,
            max_value=215.0,
            unit="°C",
            receivers=["Dashboard"],
        )

        sig_line = gen._generate_signal(signal, 0x300)
        # For big-endian 16-bit signal starting at bit 0, Motorola start bit is 15
        assert sig_line == ' SG_ Temperature : 15|16@0- (0.1,-40.0) [-40.0|215.0] "°C" Dashboard'

    def test_generate_signal_with_multiplexer(self) -> None:
        """Test signal generation with multiplexer indicator."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="MuxSignal",
            start_bit=0,
            bit_length=8,
            multiplexer_indicator="M",
        )

        sig_line = gen._generate_signal(signal, 0x100)
        assert sig_line == ' SG_ MuxSignal M : 0|8@1+ (1.0,0.0) [0.0|0.0] "" Vector__XXX'

    def test_generate_signal_multiplexed(self) -> None:
        """Test signal generation for multiplexed signal."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="MultiplexedSignal",
            start_bit=8,
            bit_length=16,
            multiplexer_indicator="m0",
        )

        sig_line = gen._generate_signal(signal, 0x100)
        assert sig_line == ' SG_ MultiplexedSignal m0 : 8|16@1+ (1.0,0.0) [0.0|0.0] "" Vector__XXX'

    def test_generate_comments_empty(self) -> None:
        """Test comment generation with no comments."""
        gen = DBCGenerator()
        gen.add_node(DBCNode("ECU_Test"))
        gen.add_message(DBCMessage(0x100, "Msg", 8))

        comments = gen._generate_comments()
        assert comments == ""

    def test_generate_comments_node(self) -> None:
        """Test node comment generation."""
        gen = DBCGenerator()
        gen.add_node(DBCNode("ECU_Engine", "Engine Control Unit"))

        comments = gen._generate_comments()
        assert 'CM_ BU_ ECU_Engine "Engine Control Unit";' in comments

    def test_generate_comments_message(self) -> None:
        """Test message comment generation."""
        gen = DBCGenerator()
        gen.add_message(DBCMessage(0x200, "EngineData", 8, comment="Engine status"))

        comments = gen._generate_comments()
        assert 'CM_ BO_ 512 "Engine status";' in comments

    def test_generate_comments_signal(self) -> None:
        """Test signal comment generation."""
        gen = DBCGenerator()
        signal = DBCSignal(name="Speed", start_bit=0, bit_length=16, comment="Vehicle speed")
        msg = DBCMessage(0x100, "SpeedData", 8, signals=[signal])
        gen.add_message(msg)

        comments = gen._generate_comments()
        assert 'CM_ SG_ 256 Speed "Vehicle speed";' in comments

    def test_generate_attributes(self) -> None:
        """Test attribute generation."""
        gen = DBCGenerator()
        msg = DBCMessage(
            0x100,
            "TestMsg",
            8,
            cycle_time=10,
            send_type="Event",
        )
        gen.add_message(msg)

        attributes = gen._generate_attributes()
        assert 'BA_DEF_ "BusType" STRING ;' in attributes
        assert 'BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;' in attributes
        assert 'BA_DEF_ BO_ "GenMsgSendType" STRING ;' in attributes
        assert 'BA_DEF_DEF_ "BusType" "CAN";' in attributes
        assert 'BA_ "GenMsgCycleTime" BO_ 256 10;' in attributes
        assert 'BA_ "GenMsgSendType" BO_ 256 "Event";' in attributes

    def test_generate_value_descriptions_empty(self) -> None:
        """Test value description generation with no value tables."""
        gen = DBCGenerator()
        signal = DBCSignal(name="Speed", start_bit=0, bit_length=16)
        msg = DBCMessage(0x100, "SpeedData", 8, signals=[signal])
        gen.add_message(msg)

        value_desc = gen._generate_value_descriptions()
        assert value_desc == ""

    def test_generate_value_descriptions(self) -> None:
        """Test value description generation for signals."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="GearPosition",
            start_bit=0,
            bit_length=8,
            value_table={0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive"},
        )
        msg = DBCMessage(0x200, "TransmissionData", 8, signals=[signal])
        gen.add_message(msg)

        value_desc = gen._generate_value_descriptions()
        assert "VAL_ 512 GearPosition" in value_desc
        assert '0 "Park"' in value_desc
        assert '1 "Reverse"' in value_desc
        assert '3 "Drive"' in value_desc

    def test_calculate_motorola_start_bit_8bit(self) -> None:
        """Test Motorola start bit calculation for 8-bit signal."""
        gen = DBCGenerator()
        # 8-bit signal at byte 0 (bits 0-7)
        # Intel start bit: 0, MSB at bit 7
        motorola_bit = gen._calculate_motorola_start_bit(0, 8)
        assert motorola_bit == 7

    def test_calculate_motorola_start_bit_16bit(self) -> None:
        """Test Motorola start bit calculation for 16-bit signal."""
        gen = DBCGenerator()
        # 16-bit signal at byte 0 (bits 0-15)
        # Intel start bit: 0, MSB at bit 15
        motorola_bit = gen._calculate_motorola_start_bit(0, 16)
        assert motorola_bit == 15

    def test_calculate_motorola_start_bit_offset(self) -> None:
        """Test Motorola start bit calculation for offset signal."""
        gen = DBCGenerator()
        # 8-bit signal at byte 1 (bits 8-15)
        # Intel start bit: 8, MSB at bit 15
        motorola_bit = gen._calculate_motorola_start_bit(8, 8)
        assert motorola_bit == 15

    def test_calculate_motorola_start_bit_32bit(self) -> None:
        """Test Motorola start bit calculation for 32-bit signal."""
        gen = DBCGenerator()
        # 32-bit signal at byte 0 (bits 0-31)
        # Intel start bit: 0, MSB at bit 31
        motorola_bit = gen._calculate_motorola_start_bit(0, 32)
        assert motorola_bit == 31

    def test_validate_dbc_valid(self) -> None:
        """Test DBC validation with valid content."""
        gen = DBCGenerator()
        gen.add_node(DBCNode("ECU_Test"))
        msg = DBCMessage(0x100, "TestMsg", 8, "ECU_Test")
        gen.add_message(msg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gen.generate(temp_path)
            content = temp_path.read_text()
            assert gen.validate_dbc(content)
        finally:
            temp_path.unlink()

    def test_validate_dbc_missing_version(self) -> None:
        """Test DBC validation rejects content without VERSION."""
        gen = DBCGenerator()
        content = "NS_ :\nBS_:\nBU_:"
        assert not gen.validate_dbc(content)

    def test_validate_dbc_missing_ns(self) -> None:
        """Test DBC validation rejects content without NS_."""
        gen = DBCGenerator()
        content = 'VERSION "1.0"\nBS_:\nBU_:'
        assert not gen.validate_dbc(content)

    def test_validate_dbc_invalid_message(self) -> None:
        """Test DBC validation rejects malformed message definition."""
        gen = DBCGenerator()
        content = 'VERSION "1.0"\nNS_ :\nBS_:\nBU_:\nBO_ INVALID'
        assert not gen.validate_dbc(content)

    def test_validate_dbc_invalid_signal(self) -> None:
        """Test DBC validation rejects malformed signal definition."""
        gen = DBCGenerator()
        content = 'VERSION "1.0"\nNS_ :\nBS_:\nBU_:\nBO_ 100 Msg: 8 ECU\n SG_ BadSignal'
        assert not gen.validate_dbc(content)

    def test_generate_complete_dbc_file(self) -> None:
        """Test generating a complete DBC file with all features."""
        gen = DBCGenerator()

        # Add nodes
        gen.add_node(DBCNode("ECU_Engine", "Engine Control Unit"))
        gen.add_node(DBCNode("Gateway", "CAN Gateway"))
        gen.add_node(DBCNode("Dashboard", "Dashboard Display"))

        # Add value table
        gen.add_value_table("GearPosition", {0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive"})

        # Create message with signals
        signal1 = DBCSignal(
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

        signal2 = DBCSignal(
            name="Gear",
            start_bit=16,
            bit_length=8,
            value_table={0: "Park", 1: "Reverse", 2: "Neutral", 3: "Drive"},
            receivers=["Dashboard"],
            comment="Current gear position",
        )

        msg = DBCMessage(
            message_id=0x200,
            name="EngineData",
            dlc=8,
            sender="ECU_Engine",
            signals=[signal1, signal2],
            comment="Engine status and transmission",
            cycle_time=10,
            send_type="Cyclic",
        )

        gen.add_message(msg)

        # Generate file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gen.generate(temp_path)
            content = temp_path.read_text()

            # Verify all sections present
            assert 'VERSION "1.0"' in content
            assert "BU_: ECU_Engine Gateway Dashboard" in content
            assert "VAL_TABLE_ GearPosition" in content
            assert "BO_ 512 EngineData: 8 ECU_Engine" in content
            assert "SG_ EngineSpeed" in content
            assert "SG_ Gear" in content
            assert 'CM_ BU_ ECU_Engine "Engine Control Unit"' in content
            assert 'CM_ BO_ 512 "Engine status and transmission"' in content
            assert 'CM_ SG_ 512 EngineSpeed "Engine rotational speed"' in content
            assert 'BA_ "GenMsgCycleTime" BO_ 512 10;' in content
            assert "VAL_ 512 Gear" in content

            # Validate
            assert gen.validate_dbc(content)

        finally:
            temp_path.unlink()

    def test_generate_multiple_messages_sorted(self) -> None:
        """Test that messages are generated in sorted order by ID."""
        gen = DBCGenerator()
        gen.add_message(DBCMessage(0x300, "Msg3", 8))
        gen.add_message(DBCMessage(0x100, "Msg1", 8))
        gen.add_message(DBCMessage(0x200, "Msg2", 8))

        messages = gen._generate_messages()
        lines = [line for line in messages.split("\n") if line.startswith("BO_")]

        assert "BO_ 256 Msg1" in lines[0]
        assert "BO_ 512 Msg2" in lines[1]
        assert "BO_ 768 Msg3" in lines[2]

    def test_generate_extended_id_message(self) -> None:
        """Test generating message with 29-bit extended ID."""
        gen = DBCGenerator()
        # Extended ID: 0x18FF1234 (29-bit) = 419369524 decimal
        msg = DBCMessage(0x18FF1234, "ExtendedMsg", 8, "ECU_Test")
        gen.add_message(msg)

        messages = gen._generate_messages()
        assert "BO_ 419369524 ExtendedMsg: 8 ECU_Test" in messages

    def test_generate_canfd_message(self) -> None:
        """Test generating CAN-FD message with DLC > 8."""
        gen = DBCGenerator()
        msg = DBCMessage(0x100, "CANFDMsg", 64, "ECU_Test")
        gen.add_message(msg)

        messages = gen._generate_messages()
        assert "BO_ 256 CANFDMsg: 64 ECU_Test" in messages

    def test_signal_with_no_unit(self) -> None:
        """Test signal generation with empty unit string."""
        gen = DBCGenerator()
        signal = DBCSignal(name="Counter", start_bit=0, bit_length=8, unit="")
        sig_line = gen._generate_signal(signal, 0x100)
        assert '""' in sig_line  # Empty unit string

    def test_signal_with_special_unit(self) -> None:
        """Test signal generation with special characters in unit."""
        gen = DBCGenerator()
        signal = DBCSignal(name="Temp", start_bit=0, bit_length=16, unit="°C")
        sig_line = gen._generate_signal(signal, 0x100)
        assert '"°C"' in sig_line

    def test_multiplexed_signals_complete_example(self) -> None:
        """Test complete example with multiplexer and multiplexed signals."""
        gen = DBCGenerator()

        # Multiplexer signal
        mux_signal = DBCSignal(
            name="MultiplexerID",
            start_bit=0,
            bit_length=8,
            multiplexer_indicator="M",
            comment="Multiplexer identifier",
        )

        # Multiplexed signals
        signal1 = DBCSignal(
            name="Mode0_Data",
            start_bit=8,
            bit_length=16,
            multiplexer_indicator="m0",
            comment="Data for mode 0",
        )

        signal2 = DBCSignal(
            name="Mode1_Data",
            start_bit=8,
            bit_length=32,
            multiplexer_indicator="m1",
            comment="Data for mode 1",
        )

        msg = DBCMessage(
            message_id=0x400,
            name="MultiplexedMsg",
            dlc=8,
            sender="ECU_Test",
            signals=[mux_signal, signal1, signal2],
            comment="Multiplexed message example",
        )

        gen.add_message(msg)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gen.generate(temp_path)
            content = temp_path.read_text()

            assert "SG_ MultiplexerID M :" in content
            assert "SG_ Mode0_Data m0 :" in content
            assert "SG_ Mode1_Data m1 :" in content
            assert gen.validate_dbc(content)

        finally:
            temp_path.unlink()

    def test_empty_dbc_generation(self) -> None:
        """Test generating DBC with no messages (header only)."""
        gen = DBCGenerator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False) as f:
            temp_path = Path(f.name)

        try:
            gen.generate(temp_path)
            content = temp_path.read_text()

            assert 'VERSION "1.0"' in content
            assert "NS_ :" in content
            assert "BS_:" in content
            assert "BU_:" in content
            assert gen.validate_dbc(content)

        finally:
            temp_path.unlink()

    def test_message_without_sender(self) -> None:
        """Test message with default sender."""
        gen = DBCGenerator()
        msg = DBCMessage(0x100, "NoSender", 8)

        assert msg.sender == "Vector__XXX"
        gen.add_message(msg)

        messages = gen._generate_messages()
        assert "Vector__XXX" in messages

    def test_signal_with_float_scaling(self) -> None:
        """Test signal with precise floating-point scaling."""
        gen = DBCGenerator()
        signal = DBCSignal(
            name="Voltage",
            start_bit=0,
            bit_length=16,
            factor=0.001,
            offset=-32.768,
            min_value=-32.768,
            max_value=32.767,
            unit="V",
        )

        sig_line = gen._generate_signal(signal, 0x100)
        assert "(0.001,-32.768)" in sig_line
        assert "[-32.768|32.767]" in sig_line
