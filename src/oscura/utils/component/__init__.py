"""Component analysis module for Oscura.

This module provides TDR-based impedance extraction, capacitance/inductance
measurement, parasitic extraction, and transmission line analysis.
"""

from oscura.utils.component.impedance import (
    discontinuity_analysis,
    extract_impedance,
    impedance_profile,
)
from oscura.utils.component.reactive import (
    extract_parasitics,
    measure_capacitance,
    measure_inductance,
)
from oscura.utils.component.transmission_line import (
    characteristic_impedance,
    propagation_delay,
    transmission_line_analysis,
    velocity_factor,
)

__all__ = [
    "characteristic_impedance",
    "discontinuity_analysis",
    # Impedance
    "extract_impedance",
    "extract_parasitics",
    "impedance_profile",
    # Reactive
    "measure_capacitance",
    "measure_inductance",
    "propagation_delay",
    # Transmission line
    "transmission_line_analysis",
    "velocity_factor",
]
