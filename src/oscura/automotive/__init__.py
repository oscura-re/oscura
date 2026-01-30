"""Automotive signal analysis and reverse engineering.

This module provides comprehensive automotive protocol analysis capabilities
for CAN, CAN-FD, OBD-II, J1939, UDS, DBC-based signal decoding, and diagnostic
trouble code (DTC) database.

Key features:
- CAN message analysis and discovery (message inventory, byte entropy, pattern detection)
- Signal boundary detection and hypothesis testing
- DBC database parsing and generation
- OBD-II diagnostic protocol support
- J1939 heavy-duty vehicle protocol support
- UDS (ISO 14229) diagnostic services decoding
- DTC database (200+ codes for Powertrain, Chassis, Body, Network)
- Discovery documentation with evidence tracking (.tkcan format)
- Integration with Oscura's CRC reverse engineering and state machine learning

Example:
    >>> from oscura.automotive.can import CANSession
    >>> from oscura.automotive.sources import FileSource
    >>> # Load automotive log file
    >>> session = CANSession(name="Analysis")
    >>> session.add_recording("main", FileSource("capture.blf"))
    >>> # View message inventory
    >>> inventory = session.inventory()
    >>> # Analyze specific message
    >>> msg = session.message(0x280)
    >>> analysis = msg.analyze()
    >>> # Test hypothesis about signal
    >>> hypothesis = msg.test_hypothesis(
    ...     signal_name="rpm",
    ...     start_byte=2,
    ...     bit_length=16,
    ...     byte_order="big_endian",
    ...     scale=0.25
    ... )
    >>>
    >>> # Look up diagnostic trouble codes
    >>> from oscura.automotive.dtc import DTCDatabase
    >>> info = DTCDatabase.lookup("P0420")
    >>> print(f"{info.code}: {info.description}")
    P0420: Catalyst System Efficiency Below Threshold (Bank 1)
"""

# Version dynamically imported from package metadata (SSOT: pyproject.toml)
try:
    from importlib.metadata import version

    __version__ = version("oscura")
except Exception:
    # Fallback for development/testing when package not installed
    __version__ = "0.7.0"

__all__ = [
    "CANMessage",
    "CANSession",
    "DecodedSignal",
    "DiscoveryDocument",
    "plot_bus_timeline",
    "plot_bus_utilization",
    "plot_message_distribution",
    "plot_message_frequency",
    "plot_signal_timeline",
]

# Import main classes when module is loaded
try:
    from oscura.automotive.can.discovery import DiscoveryDocument
    from oscura.automotive.can.models import CANMessage, DecodedSignal
    from oscura.automotive.can.session import CANSession
    from oscura.automotive.visualization import (
        plot_bus_timeline,
        plot_bus_utilization,
        plot_message_distribution,
        plot_message_frequency,
        plot_signal_timeline,
    )
except ImportError:
    # Optional automotive dependencies not installed
    pass
