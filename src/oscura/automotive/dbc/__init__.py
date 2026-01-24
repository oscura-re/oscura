"""DBC database support for CAN signal definitions.

This module provides DBC file parsing and generation capabilities.
"""

__all__ = ["DBCGenerator", "DBCParser", "load_dbc"]

try:
    from oscura.automotive.dbc.generator import DBCGenerator
    from oscura.automotive.dbc.parser import DBCParser, load_dbc
except ImportError:
    # Optional dependencies not installed
    pass
