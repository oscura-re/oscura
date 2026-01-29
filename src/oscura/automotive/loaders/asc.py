"""Vector ASC (ASCII Format) file loader.

This module provides loading of Vector ASC files. ASC is a human-readable
text format used by Vector tools for logging CAN bus data.

ASC format example:
    date Mon Jul 15 10:30:45.123 2024
    0.000000 1 123 Rx d 8 01 02 03 04 05 06 07 08
    0.010000 1 280 Rx d 8 0A 0B 0C 0D 0E 0F 10 11
"""

import re
from pathlib import Path

from oscura.automotive.can.models import CANMessage, CANMessageList

__all__ = ["load_asc"]


def load_asc(file_path: Path | str) -> CANMessageList:
    """Load CAN messages from a Vector ASC file.

    Args:
        file_path: Path to the ASC file.

    Returns:
        CANMessageList containing all parsed CAN messages.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be parsed.

    Example:
        >>> messages = load_asc("capture.asc")
        >>> print(f"Loaded {len(messages)} messages")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"ASC file not found: {path}")

    messages = CANMessageList()

    # Regex pattern for CAN message lines
    # Format: <timestamp> <channel> <ID> <direction> <type> <DLC> <data bytes>
    # Example: 0.123456 1 123 Rx d 8 01 02 03 04 05 06 07 08
    pattern = re.compile(
        r"^\s*(\d+\.\d+)\s+"  # timestamp
        r"(\d+)\s+"  # channel
        r"([0-9A-Fa-f]+)\s+"  # CAN ID (hex)
        r"(Rx|Tx)\s+"  # direction
        r"d\s+"  # type (d = data frame)
        r"(\d+)\s+"  # DLC
        r"((?:[0-9A-Fa-f]{2}\s*)*)"  # data bytes
    )

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Skip comments and header lines
                if line.startswith("//") or line.startswith("date"):
                    continue

                # Try to parse as CAN message
                match = pattern.match(line)
                if match:
                    timestamp_str, channel_str, id_str, direction, dlc_str, data_str = (
                        match.groups()
                    )

                    # Parse components
                    timestamp = float(timestamp_str)
                    channel = int(channel_str)
                    arb_id = int(id_str, 16)
                    int(dlc_str)

                    # Parse data bytes
                    data_bytes = bytes.fromhex(data_str.replace(" ", ""))

                    # Determine if extended ID (typically > 0x7FF)
                    is_extended = arb_id > 0x7FF

                    # Create message
                    can_msg = CANMessage(
                        arbitration_id=arb_id,
                        timestamp=timestamp,
                        data=data_bytes,
                        is_extended=is_extended,
                        is_fd=False,  # ASC typically doesn't indicate FD
                        channel=channel,
                    )
                    messages.append(can_msg)

    except Exception as e:
        raise ValueError(f"Failed to parse ASC file {path}: {e}") from e

    return messages
