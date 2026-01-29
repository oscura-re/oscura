"""Power analysis module for Oscura.

Provides comprehensive power analysis capabilities including:
- Basic power measurements (instantaneous, average, RMS, peak)
- AC power analysis (reactive, apparent, power factor)
- Switching loss analysis for power electronics
- Safe Operating Area (SOA) analysis
- Ripple measurement
- Efficiency calculations


Example:
    >>> from oscura.analyzers.power import instantaneous_power, power_statistics
    >>> power_trace = instantaneous_power(voltage_trace, current_trace)
    >>> stats = power_statistics(power_trace)
    >>> print(f"Average power: {stats['average']:.2f} W")
"""

from oscura.analyzers.power.ac_power import (
    apparent_power,
    displacement_power_factor,
    distortion_power_factor,
    phase_angle,
    power_factor,
    reactive_power,
    total_harmonic_distortion_power,
)
from oscura.analyzers.power.basic import (
    average_power,
    energy,
    instantaneous_power,
    peak_power,
    power_statistics,
    rms_power,
)
from oscura.analyzers.power.conduction import (
    conduction_loss,
    duty_cycle_weighted_loss,
    forward_voltage,
    mosfet_conduction_loss,
    on_resistance,
)
from oscura.analyzers.power.efficiency import (
    efficiency,
    multi_output_efficiency,
    power_conversion_efficiency,
)
from oscura.analyzers.power.ripple import (
    extract_ripple,
    ripple,
    ripple_frequency,
    ripple_percentage,
    ripple_statistics,
)
from oscura.analyzers.power.soa import (
    SOALimit,
    check_soa_violations,
    create_mosfet_soa,
    plot_soa,
    soa_analysis,
)
from oscura.analyzers.power.switching import (
    switching_energy,
    switching_frequency,
    switching_loss,
    total_switching_loss,
    turn_off_loss,
    turn_on_loss,
)

__all__ = [
    "SOALimit",
    "apparent_power",
    "average_power",
    "check_soa_violations",
    # Conduction
    "conduction_loss",
    "create_mosfet_soa",
    "displacement_power_factor",
    "distortion_power_factor",
    "duty_cycle_weighted_loss",
    # Efficiency
    "efficiency",
    "energy",
    "extract_ripple",
    "forward_voltage",
    # Basic power
    "instantaneous_power",
    "mosfet_conduction_loss",
    "multi_output_efficiency",
    "on_resistance",
    "peak_power",
    "phase_angle",
    "plot_soa",
    "power_conversion_efficiency",
    "power_factor",
    "power_statistics",
    # AC power
    "reactive_power",
    # Ripple
    "ripple",
    "ripple_frequency",
    "ripple_percentage",
    "ripple_statistics",
    "rms_power",
    # SOA
    "soa_analysis",
    "switching_energy",
    "switching_frequency",
    # Switching
    "switching_loss",
    "total_harmonic_distortion_power",
    "total_switching_loss",
    "turn_off_loss",
    "turn_on_loss",
]
