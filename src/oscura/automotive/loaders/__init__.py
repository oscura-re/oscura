"""Automotive file format loaders.

This module provides loaders for common automotive logging file formats:
- BLF (Vector Binary Logging Format)
- ASC (Vector ASCII Format)
- MDF/MF4 (ASAM Measurement Data Format)
- CSV (Comma-Separated Values)
- PCAP (Packet Capture - SocketCAN)
"""

__all__ = [
    "detect_format",
    "load_asc",
    "load_automotive_log",
    "load_blf",
    "load_csv_can",
    "load_mdf",
    "load_pcap",
]

try:
    from oscura.automotive.loaders.asc import load_asc
    from oscura.automotive.loaders.blf import load_blf
    from oscura.automotive.loaders.csv_can import load_csv_can
    from oscura.automotive.loaders.dispatcher import (
        detect_format,
        load_automotive_log,
    )
    from oscura.automotive.loaders.mdf import load_mdf
    from oscura.automotive.loaders.pcap import load_pcap
except ImportError:
    # Optional dependencies not installed
    pass
