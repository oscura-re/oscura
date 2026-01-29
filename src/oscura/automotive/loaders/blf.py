"""Vector BLF (Binary Logging Format) file loader.

This module provides loading of Vector BLF files using the python-can library.
BLF is a proprietary binary format used by Vector tools (CANoe, CANalyzer, etc.)
for logging CAN bus data.
"""

from pathlib import Path

from oscura.automotive.can.models import CANMessage, CANMessageList

__all__ = ["load_blf"]


def load_blf(file_path: Path | str) -> CANMessageList:
    """Load CAN messages from a Vector BLF file.

    Args:
        file_path: Path to the BLF file.

    Returns:
        CANMessageList containing all parsed CAN messages.

    Raises:
        ImportError: If python-can is not installed.
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be parsed.

    Example:
        >>> messages = load_blf("capture.blf")
        >>> print(f"Loaded {len(messages)} messages")
        >>> print(f"Unique IDs: {len(messages.unique_ids())}")
    """
    try:
        import can
    except ImportError as e:
        raise ImportError(
            "python-can is required for BLF file support. "
            "Install with: pip install 'oscura[automotive]'"
        ) from e

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"BLF file not found: {path}")

    messages = CANMessageList()

    try:
        # Open BLF file using python-can
        with can.BLFReader(str(path)) as reader:
            for msg in reader:
                # Convert python-can Message to our CANMessage
                # Extract channel number (python-can may return various types)
                channel = 0
                if hasattr(msg, "channel"):
                    ch = msg.channel
                    if isinstance(ch, int):
                        channel = ch
                    elif isinstance(ch, str):
                        channel = int(ch)
                    elif ch is not None:
                        # Handle other types (e.g., sequences), default to 0 if conversion fails
                        try:
                            # Convert sequences to single value
                            from collections.abc import Sequence as ABCSequence

                            if isinstance(ch, ABCSequence):
                                channel = int(ch[0]) if ch else 0
                            # Note: other cases caught by exception handler
                        except (TypeError, ValueError, IndexError):
                            channel = 0

                can_msg = CANMessage(
                    arbitration_id=msg.arbitration_id,
                    timestamp=msg.timestamp,
                    data=bytes(msg.data),
                    is_extended=msg.is_extended_id,
                    is_fd=msg.is_fd,
                    channel=channel,
                )
                messages.append(can_msg)

    except Exception as e:
        raise ValueError(f"Failed to parse BLF file {path}: {e}") from e

    return messages
