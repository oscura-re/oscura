"""Automatic file format detection and loading dispatcher.

This module provides automatic detection of automotive log file formats
and dispatching to the appropriate loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessageList

__all__ = ["detect_format", "load_automotive_log"]


def detect_format(file_path: Path | str) -> str:
    """Detect automotive log file format.

    Args:
        file_path: Path to the file.

    Returns:
        str: Format name: 'blf', 'asc', 'mdf', 'csv', 'pcap', or 'unknown'.
    """
    path = Path(file_path)

    # Check extension first
    ext = path.suffix.lower()

    if ext == ".blf":
        return "blf"
    elif ext == ".asc":
        return "asc"
    elif ext in [".mdf", ".mf4", ".dat"]:
        return "mdf"
    elif ext == ".csv":
        return "csv"
    elif ext in [".pcap", ".pcapng"]:
        return "pcap"

    # If extension is ambiguous, check file contents
    try:
        with open(path, "rb") as f:
            header = f.read(16)

            # BLF magic: "LOGG"
            if header[:4] == b"LOGG":
                return "blf"

            # MDF magic: "MDF" or "HDBlock"
            if b"MDF" in header or b"HD" in header[:8]:
                return "mdf"

            # PCAP magic
            if header[:4] in [b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"]:
                return "pcap"

    except Exception:
        pass

    # Try as text file
    try:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline().strip()

            # ASC files typically start with "date" or timestamp
            if first_line.startswith("date") or "CAN" in first_line:
                return "asc"

            # CSV with CAN data
            if "," in first_line and ("id" in first_line.lower() or "can" in first_line.lower()):
                return "csv"

    except Exception:
        pass

    return "unknown"


def load_automotive_log(file_path: Path | str) -> CANMessageList:
    """Load automotive log file, automatically detecting format.

    This function automatically detects the file format and uses the
    appropriate loader.

    Args:
        file_path: Path to the automotive log file.

    Returns:
        CANMessageList containing all parsed messages.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format cannot be determined or is unsupported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Detect format
    fmt = detect_format(path)

    if fmt == "unknown":
        raise ValueError(f"Unknown or unsupported file format: {path}")

    # Import and dispatch
    if fmt == "blf":
        from oscura.automotive.loaders.blf import load_blf

        return load_blf(path)

    elif fmt == "asc":
        from oscura.automotive.loaders.asc import load_asc

        return load_asc(path)

    elif fmt == "mdf":
        from oscura.automotive.loaders.mdf import load_mdf

        return load_mdf(path)

    elif fmt == "csv":
        from oscura.automotive.loaders.csv_can import load_csv_can

        return load_csv_can(path)

    elif fmt == "pcap":
        from oscura.automotive.loaders.pcap import load_pcap

        return load_pcap(path)

    else:
        raise ValueError(f"Unsupported format: {fmt}")
