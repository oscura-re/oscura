"""Diagnostic Trouble Code (DTC) database and utilities.

This module provides a comprehensive database of automotive diagnostic trouble
codes (DTCs) following SAE J2012 and ISO 14229 standards.

Example:
    >>> from oscura.automotive.dtc import DTCDatabase
    >>> # Look up a specific code
    >>> info = DTCDatabase.lookup("P0420")
    >>> print(f"{info.code}: {info.description}")
    P0420: Catalyst System Efficiency Below Threshold (Bank 1)
    >>>
    >>> # Search for codes
    >>> results = DTCDatabase.search("oxygen sensor")
    >>> print(f"Found {len(results)} oxygen sensor codes")
    >>>
    >>> # Get all codes in a category
    >>> powertrain = DTCDatabase.get_by_category("Powertrain")
    >>> print(f"{len(powertrain)} powertrain codes")
"""

from oscura.automotive.dtc.database import DTCS, DTCDatabase, DTCInfo

__all__ = [
    "DTCS",
    "DTCDatabase",
    "DTCInfo",
]
