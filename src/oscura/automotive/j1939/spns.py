"""J1939 Suspect Parameter Number (SPN) definitions.

This module provides standard SPN definitions for common J1939 parameters
according to SAE J1939/71 (Vehicle Application Layer).

Example:
    >>> from oscura.automotive.j1939.spns import get_standard_spns
    >>> spns = get_standard_spns()
    >>> engine_speed = next(s for s in spns[61444] if s.spn == 190)
    >>> engine_speed.name
    'Engine Speed'
    >>> engine_speed.unit
    'rpm'
"""

from __future__ import annotations

from oscura.automotive.j1939.analyzer import J1939SPN

__all__ = [
    "STANDARD_SPNS",
    "get_standard_spns",
]


def get_standard_spns() -> dict[int, list[J1939SPN]]:
    """Get standard SPN definitions for common PGNs.

    Returns:
        Dictionary mapping PGN to list of SPNs.

    Example:
        >>> spns = get_standard_spns()
        >>> 61444 in spns  # EEC1
        True
        >>> len(spns[61444])
        8
    """
    return STANDARD_SPNS.copy()


# Standard SPN definitions per SAE J1939/71
STANDARD_SPNS: dict[int, list[J1939SPN]] = {
    # PGN 61444 - Electronic Engine Controller 1 (EEC1)
    61444: [
        J1939SPN(
            spn=190,
            name="Engine Speed",
            start_bit=24,
            bit_length=16,
            resolution=0.125,
            offset=0.0,
            unit="rpm",
            data_range=(0.0, 8031.875),
        ),
        J1939SPN(
            spn=899,
            name="Engine Torque Mode",
            start_bit=0,
            bit_length=4,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 15.0),
        ),
        J1939SPN(
            spn=512,
            name="Driver's Demand Engine Percent Torque",
            start_bit=8,
            bit_length=8,
            resolution=1.0,
            offset=-125.0,
            unit="%",
            data_range=(-125.0, 125.0),
        ),
        J1939SPN(
            spn=513,
            name="Actual Engine Percent Torque",
            start_bit=16,
            bit_length=8,
            resolution=1.0,
            offset=-125.0,
            unit="%",
            data_range=(-125.0, 125.0),
        ),
        J1939SPN(
            spn=1483,
            name="Source Address of Controlling Device",
            start_bit=40,
            bit_length=8,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 255.0),
        ),
        J1939SPN(
            spn=1675,
            name="Engine Starter Mode",
            start_bit=4,
            bit_length=4,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 15.0),
        ),
        J1939SPN(
            spn=2432,
            name="Engine Demand Percent Torque",
            start_bit=48,
            bit_length=8,
            resolution=1.0,
            offset=-125.0,
            unit="%",
            data_range=(-125.0, 125.0),
        ),
    ],
    # PGN 61443 - Electronic Engine Controller 2 (EEC2)
    61443: [
        J1939SPN(
            spn=91,
            name="Accelerator Pedal Position 1",
            start_bit=0,
            bit_length=8,
            resolution=0.4,
            offset=0.0,
            unit="%",
            data_range=(0.0, 100.0),
        ),
        J1939SPN(
            spn=92,
            name="Engine Percent Load At Current Speed",
            start_bit=8,
            bit_length=8,
            resolution=1.0,
            offset=0.0,
            unit="%",
            data_range=(0.0, 250.0),
        ),
        J1939SPN(
            spn=974,
            name="Remote Accelerator Pedal Position",
            start_bit=16,
            bit_length=8,
            resolution=0.4,
            offset=0.0,
            unit="%",
            data_range=(0.0, 100.0),
        ),
        J1939SPN(
            spn=29,
            name="Accelerator Pedal Position 2",
            start_bit=24,
            bit_length=8,
            resolution=0.4,
            offset=0.0,
            unit="%",
            data_range=(0.0, 100.0),
        ),
        J1939SPN(
            spn=3357,
            name="Vehicle Acceleration Rate Limit Status",
            start_bit=32,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
    ],
    # PGN 65265 - Cruise Control/Vehicle Speed (CCVS1)
    65265: [
        J1939SPN(
            spn=84,
            name="Wheel-Based Vehicle Speed",
            start_bit=8,
            bit_length=16,
            resolution=0.00390625,
            offset=0.0,
            unit="km/h",
            data_range=(0.0, 250.996),
        ),
        J1939SPN(
            spn=595,
            name="Cruise Control Active",
            start_bit=0,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=596,
            name="Cruise Control Enable Switch",
            start_bit=2,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=597,
            name="Brake Switch",
            start_bit=4,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=598,
            name="Clutch Switch",
            start_bit=6,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=599,
            name="Cruise Control Set Switch",
            start_bit=24,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=600,
            name="Cruise Control Coast Switch",
            start_bit=26,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=601,
            name="Cruise Control Resume Switch",
            start_bit=28,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=602,
            name="Cruise Control Accelerate Switch",
            start_bit=30,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=1633,
            name="Cruise Control Set Speed",
            start_bit=32,
            bit_length=8,
            resolution=1.0,
            offset=0.0,
            unit="km/h",
            data_range=(0.0, 250.0),
        ),
        J1939SPN(
            spn=976,
            name="PTO State",
            start_bit=40,
            bit_length=5,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 31.0),
        ),
    ],
    # PGN 65226 - Active Diagnostic Trouble Codes (DM1)
    65226: [
        J1939SPN(
            spn=1213,
            name="Malfunction Indicator Lamp Status",
            start_bit=0,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=623,
            name="Red Stop Lamp Status",
            start_bit=2,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=624,
            name="Amber Warning Lamp Status",
            start_bit=4,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
        J1939SPN(
            spn=987,
            name="Protect Lamp Status",
            start_bit=6,
            bit_length=2,
            resolution=1.0,
            offset=0.0,
            unit="",
            data_range=(0.0, 3.0),
        ),
    ],
}
