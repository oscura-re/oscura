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


def _detect_csv_columns(fieldnames: list[str] | None) -> dict[str, str]:
    """Detect required column names from CSV header.

    Args:
        fieldnames: List of column names from CSV header.

    Returns:
        Dictionary mapping logical names to actual column names.

    Raises:
        ValueError: If CSV has no header or missing required columns.
    """
    if not fieldnames:
        raise ValueError("CSV file has no header row")

    # Normalize column names to lowercase
    fieldnames_lower = [name.lower().strip() for name in fieldnames]

    # Find required columns
    timestamp_col = None
    id_col = None
    data_col = None

    for col in fieldnames_lower:
        if "timestamp" in col or col == "time" or col == "t":
            timestamp_col = col
        elif "id" in col or col == "can_id" or col == "arbitration_id":
            id_col = col
        elif "data" in col or col == "payload" or col == "bytes":
            data_col = col

    if not all([timestamp_col is not None, id_col is not None, data_col is not None]):
        raise ValueError(
            f"CSV file missing required columns. "
            f"Found: {fieldnames_lower}. "
            f"Need: timestamp, id, data"
        )

    # All columns are guaranteed non-None here due to validation above
    return {
        "timestamp": str(timestamp_col),
        "id": str(id_col),
        "data": str(data_col),
    }


def _parse_csv_row(
    row_dict: dict[str, str],
    column_mapping: dict[str, str],
    messages: CANMessageList,
) -> None:
    """Parse a single CSV row into a CAN message.

    Args:
        row_dict: Dictionary of row data from CSV.
        column_mapping: Mapping of logical names to column names.
        messages: Message list to append to.
    """
    # Create lowercase dict for case-insensitive access
    row = {k.lower().strip(): v for k, v in row_dict.items()}

    try:
        # Parse timestamp
        timestamp = float(row[column_mapping["timestamp"]])

        # Parse ID (handle hex or decimal)
        arb_id = _parse_can_id(row[column_mapping["id"]])

        # Parse data bytes
        data_bytes = _parse_data_bytes(row[column_mapping["data"]])

        # Determine if extended (>11 bits = 0x7FF)
        is_extended = arb_id > 0x7FF

        # Create and append message
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
        # Skip malformed rows silently
        pass


def _parse_can_id(id_str: str) -> int:
    """Parse CAN ID from string (hex or decimal).

    Args:
        id_str: CAN ID string.

    Returns:
        CAN ID as integer.

    Raises:
        ValueError: If ID cannot be parsed.
    """
    id_str = id_str.strip()

    if id_str.startswith("0x") or id_str.startswith("0X"):
        return int(id_str, 16)

    # Try as int first, then hex
    try:
        return int(id_str)
    except ValueError:
        return int(id_str, 16)


def _parse_data_bytes(data_str: str) -> bytes:
    """Parse data bytes from hex string.

    Args:
        data_str: Data bytes as hex string.

    Returns:
        Parsed bytes.

    Raises:
        ValueError: If data cannot be parsed.
    """
    # Remove common separators and spaces
    data_str = data_str.strip().replace(" ", "").replace(":", "").replace("-", "")

    # Remove 0x prefix if present
    if data_str.startswith("0x") or data_str.startswith("0X"):
        data_str = data_str[2:]

    return bytes.fromhex(data_str)


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

            # Detect and validate column layout
            column_mapping = _detect_csv_columns(
                list(reader.fieldnames) if reader.fieldnames else None
            )

            # Parse all message rows
            for row_dict in reader:
                _parse_csv_row(row_dict, column_mapping, messages)

    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {path}: {e}") from e

    return messages
