"""CSV CAN log file loader.

This module provides loading of CAN data from CSV files.
Supports various CSV formats commonly used for CAN logging.

Common CSV format:
    timestamp,id,data
    0.000000,0x123,0102030405060708
    0.010000,0x280,0A0B0C0D0E0F1011
"""

import csv
from pathlib import Path

from oscura.automotive.can.models import CANMessage, CANMessageList

__all__ = ["load_csv_can"]


def load_csv_can(file_path: Path | str, delimiter: str = ",") -> CANMessageList:
    """Load CAN messages from a CSV file.

    This function attempts to automatically detect the CSV column layout
    and parse CAN messages accordingly.

    Expected columns (case-insensitive, order-independent):
    - timestamp or time: Message timestamp
    - id or can_id or arbitration_id: CAN ID (hex or decimal)
    - data or payload: Data bytes (hex string)
    - Optional: channel, dlc, extended, etc.

    Args:
        file_path: Path to the CSV file.
        delimiter: CSV delimiter character.

    Returns:
        CANMessageList containing all parsed CAN messages.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file cannot be parsed or has unexpected format.

    Example:
        >>> messages = load_csv_can("capture.csv")
        >>> print(f"Loaded {len(messages)} messages")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    messages = CANMessageList()

    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            # Try to detect column names
            if not reader.fieldnames:
                raise ValueError("CSV file has no header row")

            # Normalize column names to lowercase for easier matching
            fieldnames = [name.lower().strip() for name in reader.fieldnames]

            # Find required columns
            timestamp_col = None
            id_col = None
            data_col = None

            for col in fieldnames:
                if "timestamp" in col or col == "time" or col == "t":
                    timestamp_col = col
                elif "id" in col or col == "can_id" or col == "arbitration_id":
                    id_col = col
                elif "data" in col or col == "payload" or col == "bytes":
                    data_col = col

            if not all([timestamp_col, id_col, data_col]):
                raise ValueError(
                    f"CSV file missing required columns. "
                    f"Found: {fieldnames}. "
                    f"Need: timestamp, id, data"
                )

            # Parse messages
            for row_dict in reader:
                # Create lowercase dict for case-insensitive access
                row = {k.lower().strip(): v for k, v in row_dict.items()}

                try:
                    # Parse timestamp
                    timestamp = float(row[timestamp_col])

                    # Parse ID (handle hex or decimal)
                    id_str = row[id_col].strip()
                    if id_str.startswith("0x") or id_str.startswith("0X"):
                        arb_id = int(id_str, 16)
                    else:
                        # Try as int first, then hex
                        try:
                            arb_id = int(id_str)
                        except ValueError:
                            arb_id = int(id_str, 16)

                    # Parse data bytes
                    data_str = row[data_col].strip()
                    # Remove common separators and spaces
                    data_str = data_str.replace(" ", "").replace(":", "").replace("-", "")
                    # Remove 0x prefix if present
                    if data_str.startswith("0x") or data_str.startswith("0X"):
                        data_str = data_str[2:]
                    data_bytes = bytes.fromhex(data_str)

                    # Determine if extended (>11 bits = 0x7FF)
                    is_extended = arb_id > 0x7FF

                    # Create message
                    can_msg = CANMessage(
                        arbitration_id=arb_id,
                        timestamp=timestamp,
                        data=data_bytes,
                        is_extended=is_extended,
                        is_fd=False,
                        channel=0,
                    )
                    messages.append(can_msg)

                except (ValueError, KeyError):
                    # Skip malformed rows
                    continue

    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {path}: {e}") from e

    return messages
