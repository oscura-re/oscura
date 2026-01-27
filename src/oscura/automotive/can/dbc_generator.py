"""DBC (CAN Database) file generator from reverse-engineered CAN protocols.

This module provides functionality to generate DBC files from CAN message
specifications, supporting all DBC elements including messages, signals,
enums, comments, and attributes.

DBC Format Reference:
    Vector DBC File Format Specification
    https://www.csselectronics.com/pages/can-dbc-file-format-explained
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DBCGenerator",
    "DBCMessage",
    "DBCNode",
    "DBCSignal",
]


@dataclass
class DBCSignal:
    """DBC signal definition.

    Attributes:
        name: Signal name (must be valid C identifier).
        start_bit: Starting bit position (0-63).
        bit_length: Signal length in bits (1-64).
        byte_order: Byte order - "little_endian" (Intel) or "big_endian" (Motorola).
        value_type: Value type - "unsigned" or "signed".
        factor: Scaling factor (physical = raw * factor + offset).
        offset: Value offset.
        min_value: Minimum physical value.
        max_value: Maximum physical value.
        unit: Physical unit (e.g., "rpm", "km/h", "°C").
        receivers: List of receiving nodes (ECUs).
        value_table: Optional mapping of raw values to descriptions.
        comment: Signal description/documentation.
        multiplexer_indicator: Multiplexer indicator ("M" for multiplexer, "mX" for
            multiplexed where X is the multiplexer value).

    Example:
        >>> signal = DBCSignal(
        ...     name="EngineSpeed",
        ...     start_bit=0,
        ...     bit_length=16,
        ...     byte_order="little_endian",
        ...     value_type="unsigned",
        ...     factor=0.25,
        ...     offset=0.0,
        ...     min_value=0.0,
        ...     max_value=16383.75,
        ...     unit="rpm",
        ...     receivers=["Gateway", "Dashboard"],
        ... )
    """

    name: str
    start_bit: int
    bit_length: int
    byte_order: Literal["little_endian", "big_endian"] = "little_endian"
    value_type: Literal["unsigned", "signed"] = "unsigned"
    factor: float = 1.0
    offset: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    unit: str = ""
    receivers: list[str] = field(default_factory=lambda: ["Vector__XXX"])
    value_table: dict[int, str] | None = None
    comment: str = ""
    multiplexer_indicator: str | None = None

    def __post_init__(self) -> None:
        """Validate signal definition."""
        if self.bit_length < 1 or self.bit_length > 64:
            raise ValueError(f"bit_length must be 1-64, got {self.bit_length}")
        if self.start_bit < 0:
            raise ValueError(f"start_bit must be >= 0, got {self.start_bit}")


@dataclass
class DBCMessage:
    """DBC message definition.

    Attributes:
        message_id: CAN message ID (11-bit: 0-0x7FF, 29-bit: 0-0x1FFFFFFF).
        name: Message name (must be valid C identifier).
        dlc: Data Length Code (0-8 for CAN 2.0, up to 64 for CAN-FD).
        sender: Transmitting node (ECU) name.
        signals: List of signals in this message.
        comment: Message description/documentation.
        cycle_time: Message transmission cycle time in milliseconds.
        send_type: Transmission type ("Cyclic", "Event", "IfActive", etc.).

    Example:
        >>> msg = DBCMessage(
        ...     message_id=0x200,
        ...     name="EngineData",
        ...     dlc=8,
        ...     sender="ECU_Engine",
        ...     signals=[...],
        ...     comment="Engine status and RPM",
        ...     cycle_time=10,
        ...     send_type="Cyclic",
        ... )
    """

    message_id: int
    name: str
    dlc: int
    sender: str = "Vector__XXX"
    signals: list[DBCSignal] = field(default_factory=list)
    comment: str = ""
    cycle_time: int | None = None
    send_type: str = "Cyclic"

    def __post_init__(self) -> None:
        """Validate message definition."""
        if self.message_id < 0:
            raise ValueError(f"message_id must be >= 0, got {self.message_id}")
        if self.dlc < 0 or self.dlc > 64:
            raise ValueError(f"dlc must be 0-64, got {self.dlc}")


@dataclass
class DBCNode:
    """DBC network node (ECU) definition.

    Attributes:
        name: Node name (must be valid C identifier).
        comment: Node description/documentation.

    Example:
        >>> node = DBCNode(name="ECU_Engine", comment="Engine Control Unit")
    """

    name: str
    comment: str = ""


class DBCGenerator:
    """DBC file generator from CAN specifications.

    This class generates complete DBC files from message and signal definitions,
    including all standard DBC sections (nodes, messages, signals, value tables,
    comments, attributes).

    Example:
        >>> gen = DBCGenerator()
        >>> gen.add_node(DBCNode("ECU_Engine", "Engine Control Unit"))
        >>> gen.add_node(DBCNode("Gateway"))
        >>> signal = DBCSignal(
        ...     name="EngineSpeed",
        ...     start_bit=0,
        ...     bit_length=16,
        ...     factor=0.25,
        ...     unit="rpm",
        ...     receivers=["Gateway"],
        ... )
        >>> msg = DBCMessage(
        ...     message_id=0x200,
        ...     name="EngineData",
        ...     dlc=8,
        ...     sender="ECU_Engine",
        ...     signals=[signal],
        ... )
        >>> gen.add_message(msg)
        >>> gen.generate(Path("output.dbc"))
    """

    VERSION = "1.0"

    def __init__(self) -> None:
        """Initialize DBC generator."""
        self.nodes: list[DBCNode] = []
        self.messages: list[DBCMessage] = []
        self.value_tables: dict[str, dict[int, str]] = {}
        self.environment_variables: dict[str, Any] = {}

    def add_node(self, node: DBCNode) -> None:
        """Add network node (ECU).

        Args:
            node: Node definition to add.

        Example:
            >>> gen = DBCGenerator()
            >>> gen.add_node(DBCNode("ECU_Engine", "Engine Control Unit"))
        """
        self.nodes.append(node)

    def add_message(self, message: DBCMessage) -> None:
        """Add CAN message definition.

        Args:
            message: Message definition to add.

        Example:
            >>> gen = DBCGenerator()
            >>> msg = DBCMessage(0x200, "EngineData", 8, "ECU_Engine")
            >>> gen.add_message(msg)
        """
        self.messages.append(message)

    def add_value_table(self, name: str, values: dict[int, str]) -> None:
        """Add value table (enum) for signals.

        Args:
            name: Value table name.
            values: Mapping of raw values to descriptions.

        Example:
            >>> gen = DBCGenerator()
            >>> gen.add_value_table("GearPosition", {
            ...     0: "Park",
            ...     1: "Reverse",
            ...     2: "Neutral",
            ...     3: "Drive",
            ... })
        """
        self.value_tables[name] = values

    def generate(self, output_path: Path) -> None:
        """Generate complete DBC file.

        DBC file structure (in order):
        1. VERSION
        2. NS_ (New symbols)
        3. BS_ (Bit timing)
        4. BU_ (Network nodes)
        5. VAL_TABLE_ (Value tables)
        6. BO_ (Messages with signals)
        7. CM_ (Comments)
        8. BA_DEF_ (Attribute definitions)
        9. BA_ (Attribute values)
        10. VAL_ (Signal value descriptions)

        Args:
            output_path: Path to output DBC file.

        Example:
            >>> gen = DBCGenerator()
            >>> # ... add nodes, messages, signals ...
            >>> gen.generate(Path("network.dbc"))
        """
        dbc_lines = []

        # Header
        dbc_lines.append(self._generate_header())

        # Nodes
        dbc_lines.append(self._generate_nodes())

        # Value tables
        value_tables = self._generate_value_tables()
        if value_tables:
            dbc_lines.append(value_tables)

        # Messages and signals
        dbc_lines.append(self._generate_messages())

        # Comments
        comments = self._generate_comments()
        if comments:
            dbc_lines.append(comments)

        # Attributes
        attributes = self._generate_attributes()
        if attributes:
            dbc_lines.append(attributes)

        # Value descriptions
        value_desc = self._generate_value_descriptions()
        if value_desc:
            dbc_lines.append(value_desc)

        dbc_content = "\n".join(dbc_lines)

        # Write to file
        output_path.write_text(dbc_content, encoding="utf-8")

    def _generate_header(self) -> str:
        """Generate DBC file header.

        Returns:
            Header section with VERSION, NS_, and BS_.
        """
        return f"""VERSION "{self.VERSION}"


NS_ :
\tNS_DESC_
\tCM_
\tBA_DEF_
\tBA_
\tVAL_
\tCAT_DEF_
\tCAT_
\tFILTER
\tBA_DEF_DEF_
\tEV_DATA_
\tENVVAR_DATA_
\tSGTYPE_
\tSGTYPE_VAL_
\tBA_DEF_SGTYPE_
\tBA_SGTYPE_
\tSIG_TYPE_REF_
\tVAL_TABLE_
\tSIG_GROUP_
\tSIG_VALTYPE_
\tSIGTYPE_VALTYPE_
\tBO_TX_BU_
\tBA_DEF_REL_
\tBA_REL_
\tBA_SGTYPE_REL_
\tSG_MUL_VAL_

BS_:"""

    def _generate_nodes(self) -> str:
        """Generate BU_ (nodes/ECUs) section.

        Returns:
            BU_ section with all network nodes.
        """
        if not self.nodes:
            return "BU_:"

        node_names = " ".join(node.name for node in self.nodes)
        return f"BU_: {node_names}"

    def _generate_value_tables(self) -> str:
        """Generate VAL_TABLE_ section.

        Returns:
            VAL_TABLE_ section with all value tables, or empty string if none.
        """
        if not self.value_tables:
            return ""

        lines = []
        for name, values in sorted(self.value_tables.items()):
            value_pairs = " ".join(f'{val} "{desc}"' for val, desc in sorted(values.items()))
            lines.append(f"VAL_TABLE_ {name} {value_pairs} ;")

        return "\n".join(lines)

    def _generate_messages(self) -> str:
        """Generate BO_ (messages) section.

        Returns:
            BO_ section with all messages and their signals.
        """
        if not self.messages:
            return ""

        lines = []

        for message in sorted(self.messages, key=lambda m: m.message_id):
            # Message line: BO_ MessageID MessageName: DLC Sender
            lines.append(f"BO_ {message.message_id} {message.name}: {message.dlc} {message.sender}")

            # Signal lines
            for signal in message.signals:
                lines.append(self._generate_signal(signal, message.message_id))

            # Blank line between messages
            lines.append("")

        return "\n".join(lines)

    def _generate_signal(self, signal: DBCSignal, message_id: int) -> str:
        """Generate SG_ (signal) line.

        Format:
            SG_ SignalName [M|mX] : StartBit|BitLength@ByteOrder ValueType
                (Factor,Offset) [Min|Max] "Unit" Receivers

        Where:
            ByteOrder: 1 = little endian (Intel), 0 = big endian (Motorola)
            ValueType: + = unsigned, - = signed

        Args:
            signal: Signal definition.
            message_id: Parent message ID (used for Motorola bit calculation).

        Returns:
            Formatted signal line.
        """
        # Determine byte order indicator (1 = Intel/little, 0 = Motorola/big)
        byte_order_indicator = "1" if signal.byte_order == "little_endian" else "0"

        # Convert start_bit to DBC format
        # For Intel (little-endian): start_bit is LSB position (use as-is)
        # For Motorola (big-endian): start_bit must be MSB position
        if signal.byte_order == "big_endian":
            # Convert from Intel LSB position to Motorola MSB position
            # Motorola start_bit = Intel start_bit + bit_length - 1
            start_bit = signal.start_bit + signal.bit_length - 1
        else:
            start_bit = signal.start_bit

        # Determine value type (+ = unsigned, - = signed)
        value_type_indicator = "+" if signal.value_type == "unsigned" else "-"

        # Multiplexer indicator
        multiplex = ""
        if signal.multiplexer_indicator:
            multiplex = f" {signal.multiplexer_indicator}"

        # Receivers
        receivers = ",".join(signal.receivers)

        return (
            f" SG_ {signal.name}{multiplex} : {start_bit}|{signal.bit_length}@"
            f"{byte_order_indicator}{value_type_indicator} "
            f"({signal.factor},{signal.offset}) "
            f"[{signal.min_value}|{signal.max_value}] "
            f'"{signal.unit}" {receivers}'
        )

    def _generate_comments(self) -> str:
        """Generate CM_ (comments) section.

        Returns:
            CM_ section with all comments, or empty string if none.
        """
        lines = []

        # Node comments
        for node in self.nodes:
            if node.comment:
                lines.append(f'CM_ BU_ {node.name} "{node.comment}";')

        # Message comments
        for message in self.messages:
            if message.comment:
                lines.append(f'CM_ BO_ {message.message_id} "{message.comment}";')

        # Signal comments
        for message in self.messages:
            for signal in message.signals:
                if signal.comment:
                    lines.append(f'CM_ SG_ {message.message_id} {signal.name} "{signal.comment}";')

        return "\n".join(lines) if lines else ""

    def _generate_attributes(self) -> str:
        """Generate BA_ (attributes) section.

        Returns:
            BA_ section with attribute definitions and values.
        """
        lines = []

        # Define standard attributes
        lines.append('BA_DEF_ "BusType" STRING ;')
        lines.append('BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;')
        lines.append('BA_DEF_ BO_ "GenMsgSendType" STRING ;')

        # Default attribute values
        lines.append('BA_DEF_DEF_ "BusType" "CAN";')
        lines.append('BA_DEF_DEF_ "GenMsgCycleTime" 0;')
        lines.append('BA_DEF_DEF_ "GenMsgSendType" "Cyclic";')

        # Message-specific attribute values
        for message in self.messages:
            if message.cycle_time is not None:
                lines.append(
                    f'BA_ "GenMsgCycleTime" BO_ {message.message_id} {message.cycle_time};'
                )
            if message.send_type:
                lines.append(
                    f'BA_ "GenMsgSendType" BO_ {message.message_id} "{message.send_type}";'
                )

        return "\n".join(lines)

    def _generate_value_descriptions(self) -> str:
        """Generate VAL_ (signal value descriptions) section.

        Returns:
            VAL_ section with signal value descriptions, or empty string if none.
        """
        lines = []

        for message in self.messages:
            for signal in message.signals:
                if signal.value_table:
                    value_pairs = " ".join(
                        f'{val} "{desc}"' for val, desc in sorted(signal.value_table.items())
                    )
                    lines.append(f"VAL_ {message.message_id} {signal.name} {value_pairs} ;")

        return "\n".join(lines) if lines else ""

    def _calculate_motorola_start_bit(self, start_bit: int, bit_length: int) -> int:
        """Calculate start bit for Motorola (big-endian) byte order.

        In DBC files, Motorola (big-endian) signals use the MSB position as the start bit.
        Intel (little-endian) signals use the LSB position as the start bit.

        For Motorola byte order, the start bit is simply the MSB position,
        which is start_bit + bit_length - 1.

        Args:
            start_bit: Intel (little-endian) start bit position (LSB).
            bit_length: Signal length in bits.

        Returns:
            Motorola (big-endian) start bit position (MSB).

        Example:
            >>> gen = DBCGenerator()
            >>> # 8-bit signal starting at bit 0 (Intel LSB)
            >>> # Bits 0-7 -> MSB is at bit 7
            >>> gen._calculate_motorola_start_bit(0, 8)
            7
            >>> # 16-bit signal starting at bit 0 (Intel LSB)
            >>> # Bits 0-15 -> MSB is at bit 15
            >>> gen._calculate_motorola_start_bit(0, 16)
            15
        """
        # Motorola start bit is simply the MSB position
        return start_bit + bit_length - 1

    def validate_dbc(self, dbc_content: str) -> bool:
        """Validate generated DBC syntax.

        Basic syntax validation checking for:
        - Required sections (VERSION, NS_, BS_, BU_)
        - Well-formed message definitions
        - Well-formed signal definitions

        Args:
            dbc_content: DBC file content to validate.

        Returns:
            True if basic syntax is valid, False otherwise.

        Example:
            >>> gen = DBCGenerator()
            >>> gen.add_node(DBCNode("Test"))
            >>> msg = DBCMessage(0x100, "TestMsg", 8, "Test")
            >>> gen.add_message(msg)
            >>> from pathlib import Path
            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.dbc', delete=False) as f:
            ...     gen.generate(Path(f.name))
            ...     content = Path(f.name).read_text()
            ...     gen.validate_dbc(content)
            True
        """
        lines = dbc_content.strip().split("\n")

        # Check for required sections
        has_version = any(line.startswith("VERSION") for line in lines)
        has_ns = any(line.startswith("NS_") for line in lines)
        has_bs = any(line.startswith("BS_") for line in lines)
        has_bu = any(line.startswith("BU_") for line in lines)

        if not (has_version and has_ns and has_bs and has_bu):
            return False

        # Check message definitions (BO_)
        for line in lines:
            if line.startswith("BO_ "):
                # Format: BO_ <ID> <Name>: <DLC> <Sender>
                parts = line.split()
                if len(parts) < 5:
                    return False
                try:
                    int(parts[1])  # Message ID
                    int(parts[3])  # DLC (after colon)
                except (ValueError, IndexError):
                    return False

        # Check signal definitions (SG_)
        for line in lines:
            if line.strip().startswith("SG_ "):
                # Format: SG_ <Name> : <StartBit>|<Length>@<ByteOrder><ValueType> ...
                if "|" not in line or "@" not in line:
                    return False

        return True
