"""DBC file parser using cantools.

This module provides parsing of standard DBC files for use with Oscura.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from oscura.automotive.can.models import DecodedSignal, SignalDefinition

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage

__all__ = ["DBCParser", "load_dbc"]


class DBCParser:
    """Parse DBC files and decode CAN messages."""

    def __init__(self, dbc_path: Path | str):
        """Initialize DBC parser.

        Args:
            dbc_path: Path to DBC file.

        Raises:
            ImportError: If cantools is not installed.
            FileNotFoundError: If DBC file doesn't exist.
        """
        try:
            import cantools
        except ImportError as e:
            raise ImportError(
                "cantools is required for DBC file support. "
                "Install with: pip install 'oscura[automotive]'"
            ) from e

        path = Path(dbc_path)
        if not path.exists():
            raise FileNotFoundError(f"DBC file not found: {path}")

        self.db = cantools.database.load_file(str(path))

    def decode_message(self, message: CANMessage) -> dict[str, DecodedSignal]:
        """Decode a CAN message using DBC definitions.

        Args:
            message: CAN message to decode.

        Returns:
            Dictionary mapping signal names to DecodedSignal objects.

        Raises:
            KeyError: If message ID not found in DBC.
            ValueError: If message data cannot be decoded.
        """
        # Get message definition from DBC
        try:
            msg_def = self.db.get_message_by_frame_id(message.arbitration_id)
        except KeyError as e:
            raise KeyError(f"Message ID 0x{message.arbitration_id:03X} not found in DBC") from e

        # Decode message
        try:
            decoded_data = msg_def.decode(message.data)
        except Exception as e:
            raise ValueError(f"Failed to decode message: {e}") from e

        # Convert to DecodedSignal objects
        signals = {}
        for signal_name, value in decoded_data.items():
            # Get signal definition from DBC
            signal_def = msg_def.get_signal_by_name(signal_name)

            # Create SignalDefinition (for reference)
            sig_def = SignalDefinition(
                name=signal_name,
                start_bit=signal_def.start,
                length=signal_def.length,
                byte_order="big_endian"
                if signal_def.byte_order == "big_endian"
                else "little_endian",
                value_type="signed" if signal_def.is_signed else "unsigned",
                scale=signal_def.scale,
                offset=signal_def.offset,
                unit=signal_def.unit if signal_def.unit else "",
                min_value=signal_def.minimum,
                max_value=signal_def.maximum,
                comment=signal_def.comment if signal_def.comment else "",
            )

            # Create DecodedSignal
            decoded_sig = DecodedSignal(
                name=signal_name,
                value=float(value),
                unit=sig_def.unit,
                timestamp=message.timestamp,
                definition=sig_def,
            )

            signals[signal_name] = decoded_sig

        return signals

    def get_message_ids(self) -> set[int]:
        """Get all message IDs defined in DBC.

        Returns:
            Set of CAN arbitration IDs.
        """
        return {msg.frame_id for msg in self.db.messages}

    def get_message_name(self, arbitration_id: int) -> str | None:
        """Get message name from DBC.

        Args:
            arbitration_id: CAN ID.

        Returns:
            Message name or None if not found.
        """
        try:
            msg = self.db.get_message_by_frame_id(arbitration_id)
            return str(msg.name)  # Explicit cast to str
        except KeyError:
            return None


def load_dbc(dbc_path: Path | str) -> DBCParser:
    """Load a DBC file.

    Args:
        dbc_path: Path to DBC file.

    Returns:
        DBCParser instance.

    Example:
        >>> dbc = load_dbc("vehicle.dbc")
        >>> # Decode message
        >>> signals = dbc.decode_message(can_message)
        >>> print(signals["Engine_RPM"].value)
    """
    return DBCParser(dbc_path)
