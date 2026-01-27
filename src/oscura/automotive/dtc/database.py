"""Diagnostic Trouble Code (DTC) database for automotive diagnostics.

This module provides a comprehensive database of standardized DTCs following
SAE J2012 and ISO 14229 standards. DTCs are used to identify vehicle system
faults and malfunctions.

DTC Format:
    - First character: Category (P/C/B/U)
        - P: Powertrain (engine, transmission, emissions)
        - C: Chassis (brakes, steering, suspension)
        - B: Body (lighting, HVAC, security)
        - U: Network/Communication (CAN, LIN, FlexRay)
    - Second character: Code type
        - 0: Generic (SAE defined)
        - 1-3: Manufacturer specific
    - Remaining 3 digits: Specific fault code

Example:
    >>> from oscura.automotive.dtc import DTCDatabase
    >>> # Look up a specific code
    >>> info = DTCDatabase.lookup("P0420")
    >>> print(f"{info.code}: {info.description}")
    P0420: Catalyst System Efficiency Below Threshold (Bank 1)
    >>> # Search for related codes
    >>> results = DTCDatabase.search("oxygen sensor")
    >>> print(f"Found {len(results)} oxygen sensor codes")
    >>> # Get all powertrain codes
    >>> powertrain = DTCDatabase.get_by_category("Powertrain")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DTCInfo:
    """Information about a Diagnostic Trouble Code.

    Attributes:
        code: DTC code (e.g., "P0420", "C0035", "B1234", "U0100")
        description: Human-readable description of the fault
        category: DTC category ("Powertrain", "Chassis", "Body", "Network")
        severity: Fault severity ("Critical", "High", "Medium", "Low")
        system: Specific system affected (e.g., "Emissions Control", "ABS")
        possible_causes: List of common causes for this DTC
    """

    code: str
    description: str
    category: str
    severity: str
    system: str
    possible_causes: list[str]


def _load_dtc_database() -> dict[str, DTCInfo]:
    """Load DTC database from packaged JSON file.

    Returns:
        Dictionary mapping DTC codes to DTCInfo objects.

    Note:
        This function is called once at module import time to populate
        the global DTCS dictionary. The JSON file is located in the
        same directory as this module (data.json).
    """
    # Locate data.json in the same directory as this module
    data_file = Path(__file__).parent / "data.json"

    # Load JSON data
    with open(data_file) as f:
        db = json.load(f)

    # Convert JSON objects to DTCInfo instances
    dtcs: dict[str, DTCInfo] = {}
    for code, info in db["dtcs"].items():
        dtcs[code] = DTCInfo(
            code=info["code"],
            description=info["description"],
            category=info["category"],
            severity=info["severity"],
            system=info["system"],
            possible_causes=info["possible_causes"],
        )

    return dtcs


# Load DTC database from JSON (executed once at module import)
DTCS: dict[str, DTCInfo] = _load_dtc_database()


class DTCDatabase:
    """Database for looking up Diagnostic Trouble Codes (DTCs).

    This class provides methods to search and retrieve DTC information from
    a comprehensive database of standardized automotive fault codes.
    """

    @staticmethod
    def lookup(code: str) -> DTCInfo | None:
        """Look up a DTC by its code.

        Args:
            code: DTC code to look up (e.g., "P0420", "p0420")
                  Case-insensitive, whitespace is stripped

        Returns:
            DTCInfo object if found, None if code not in database

        Example:
            >>> info = DTCDatabase.lookup("P0420")
            >>> if info:
            ...     print(f"{info.code}: {info.description}")
            P0420: Catalyst System Efficiency Below Threshold (Bank 1)
        """
        return DTCS.get(code.strip().upper())

    @staticmethod
    def search(keyword: str) -> list[DTCInfo]:
        """Search DTCs by keyword in description or possible causes.

        Searches are case-insensitive and match partial words.

        Args:
            keyword: Search term (e.g., "oxygen sensor", "misfire", "ABS")

        Returns:
            List of matching DTCInfo objects, sorted by code

        Example:
            >>> results = DTCDatabase.search("oxygen sensor")
            >>> for dtc in results[:3]:
            ...     print(f"{dtc.code}: {dtc.description}")
            P0130: O2 Sensor Circuit Malfunction (Bank 1 Sensor 1)
            P0131: O2 Sensor Circuit Low Voltage (Bank 1 Sensor 1)
            P0132: O2 Sensor Circuit High Voltage (Bank 1 Sensor 1)
        """
        keyword_lower = keyword.lower()
        results = []

        for dtc in DTCS.values():
            # Search in description
            if keyword_lower in dtc.description.lower():
                results.append(dtc)
                continue

            # Search in system
            if keyword_lower in dtc.system.lower():
                results.append(dtc)
                continue

            # Search in possible causes
            for cause in dtc.possible_causes:
                if keyword_lower in cause.lower():
                    results.append(dtc)
                    break

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def get_by_category(category: str) -> list[DTCInfo]:
        """Get all DTCs in a specific category.

        Args:
            category: Category name ("Powertrain", "Chassis", "Body", "Network")
                     Case-insensitive

        Returns:
            List of DTCInfo objects in the category, sorted by code

        Example:
            >>> chassis_codes = DTCDatabase.get_by_category("Chassis")
            >>> print(f"Found {len(chassis_codes)} chassis codes")
            Found 42 chassis codes
        """
        category_lower = category.lower()
        results = [dtc for dtc in DTCS.values() if dtc.category.lower() == category_lower]

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def get_by_system(system: str) -> list[DTCInfo]:
        """Get all DTCs for a specific system.

        Args:
            system: System name (e.g., "ABS", "Oxygen Sensors", "Fuel System")
                   Case-insensitive

        Returns:
            List of DTCInfo objects for the system, sorted by code

        Example:
            >>> abs_codes = DTCDatabase.get_by_system("ABS")
            >>> for dtc in abs_codes[:3]:
            ...     print(f"{dtc.code}: {dtc.description}")
        """
        system_lower = system.lower()
        results = [dtc for dtc in DTCS.values() if dtc.system.lower() == system_lower]

        # Sort by code
        results.sort(key=lambda x: x.code)
        return results

    @staticmethod
    def parse_dtc(code: str) -> tuple[str, str, str] | None:
        """Parse a DTC code into its components.

        Args:
            code: DTC code to parse (e.g., "P0420")

        Returns:
            Tuple of (category, code_type, fault_code) or None if invalid
            - category: "Powertrain", "Chassis", "Body", or "Network"
            - code_type: "Generic" (0) or "Manufacturer" (1-3)
            - fault_code: Remaining 3 digits

        Example:
            >>> result = DTCDatabase.parse_dtc("P0420")
            >>> if result:
            ...     category, code_type, fault_code = result
            ...     print(f"Category: {category}, Type: {code_type}, Code: {fault_code}")
            Category: Powertrain, Type: Generic, Code: 420
        """
        code = code.upper().strip()

        # Validate format
        if len(code) != 5:
            return None

        # Parse category
        category_map = {
            "P": "Powertrain",
            "C": "Chassis",
            "B": "Body",
            "U": "Network",
        }
        category = category_map.get(code[0])
        if not category:
            return None

        # Parse code type
        try:
            type_digit = int(code[1])
        except ValueError:
            return None

        code_type = "Generic" if type_digit == 0 else "Manufacturer"

        # Parse fault code
        try:
            fault_code = code[2:5]
            int(fault_code)  # Validate it's numeric
        except ValueError:
            return None

        return (category, code_type, fault_code)

    @staticmethod
    def get_all_codes() -> list[str]:
        """Get a list of all DTC codes in the database.

        Returns:
            Sorted list of all DTC codes

        Example:
            >>> all_codes = DTCDatabase.get_all_codes()
            >>> print(f"Database contains {len(all_codes)} codes")
            >>> print(f"First code: {all_codes[0]}")
            >>> print(f"Last code: {all_codes[-1]}")
        """
        return sorted(DTCS.keys())

    @staticmethod
    def get_stats() -> dict[str, int]:
        """Get statistics about the DTC database.

        Returns:
            Dictionary with counts by category and total

        Example:
            >>> stats = DTCDatabase.get_stats()
            >>> for category, count in stats.items():
            ...     print(f"{category}: {count}")
            Powertrain: 100
            Chassis: 42
            Body: 45
            Network: 23
            Total: 210
        """
        stats = {"Powertrain": 0, "Chassis": 0, "Body": 0, "Network": 0}

        for dtc in DTCS.values():
            stats[dtc.category] += 1

        stats["Total"] = len(DTCS)
        return stats
