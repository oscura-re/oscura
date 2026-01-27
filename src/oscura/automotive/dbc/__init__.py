"""DBC database support for CAN signal definitions.

This module provides DBC file parsing and generation capabilities.
"""

__all__ = ["DBCGenerator", "DBCMessage", "DBCNode", "DBCParser", "DBCSignal", "load_dbc"]

try:
    # Import comprehensive DBC generator from new location
    from oscura.automotive.can.dbc_generator import (
        DBCGenerator,
        DBCMessage,
        DBCNode,
        DBCSignal,
    )

    # Parser stays in dbc module
    from oscura.automotive.dbc.parser import DBCParser, load_dbc
except ImportError:
    # Optional dependencies not installed
    pass
