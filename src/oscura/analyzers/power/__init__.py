"""Power analysis module for Oscura - IEEE 1459-2010 compliant.

Provides comprehensive power analysis capabilities including:
- Basic power measurements (instantaneous, average, RMS, peak) per IEEE 1459
- AC power analysis (reactive, apparent, power factor, harmonics) per IEEE 1459
- Switching loss analysis for power electronics
- Conduction loss analysis (MOSFET, IGBT, diode)
- Safe Operating Area (SOA) analysis
- Ripple measurement and analysis
- Efficiency calculations (single and multi-output)


Example:
    >>> from oscura.analyzers.power import instantaneous_power, power_statistics
    >>> power_trace = instantaneous_power(voltage_trace, current_trace)
    >>> stats = power_statistics(power_trace)
    >>> print(f"Average power: {stats['average']:.2f} W")

References:
    IEEE 1459-2010: Standard for Power Quality Definitions
"""

from oscura.analyzers.power.ac_power import (
    apparent_power,
    displacement_power_factor,
    distortion_power_factor,
    phase_angle,
    power_factor,
    reactive_power,
    three_phase_power,
    total_harmonic_distortion_power,
)
from oscura.analyzers.power.basic import (
    average_power,
    energy,
    instantaneous_power,
    peak_power,
    power_profile,
    power_statistics,
    rms_power,
)
from oscura.analyzers.power.conduction import (
    conduction_loss,
    diode_conduction_loss,
    duty_cycle_weighted_loss,
    forward_voltage,
    igbt_conduction_loss,
    mosfet_conduction_loss,
    on_resistance,
    temperature_derating,
)
from oscura.analyzers.power.efficiency import (
    efficiency,
    efficiency_vs_load,
    loss_breakdown,
    multi_output_efficiency,
    power_conversion_efficiency,
    thermal_efficiency,
)
from oscura.analyzers.power.ripple import (
    extract_ripple,
    ripple,
    ripple_envelope,
    ripple_frequency,
    ripple_harmonics,
    ripple_percentage,
    ripple_statistics,
)
from oscura.analyzers.power.soa import (
    SOALimit,
    SOAViolation,
    check_soa_violations,
    create_mosfet_soa,
    plot_soa,
    soa_analysis,
)
from oscura.analyzers.power.switching import (
    SwitchingEvent,
    switching_energy,
    switching_frequency,
    switching_loss,
    switching_times,
    total_switching_loss,
    turn_off_loss,
    turn_on_loss,
)

__all__ = [
    "SOALimit",
    "SOAViolation",
    "SwitchingEvent",
    "apparent_power",
    "average_power",
    "check_soa_violations",
    "conduction_loss",
    "create_mosfet_soa",
    "diode_conduction_loss",
    "displacement_power_factor",
    "distortion_power_factor",
    "duty_cycle_weighted_loss",
    "efficiency",
    "efficiency_vs_load",
    "energy",
    "extract_ripple",
    "forward_voltage",
    "igbt_conduction_loss",
    "instantaneous_power",
    "loss_breakdown",
    "mosfet_conduction_loss",
    "multi_output_efficiency",
    "on_resistance",
    "peak_power",
    "phase_angle",
    "plot_soa",
    "power_conversion_efficiency",
    "power_factor",
    "power_profile",
    "power_statistics",
    "reactive_power",
    "ripple",
    "ripple_envelope",
    "ripple_frequency",
    "ripple_harmonics",
    "ripple_percentage",
    "ripple_statistics",
    "rms_power",
    "soa_analysis",
    "switching_energy",
    "switching_frequency",
    "switching_loss",
    "switching_times",
    "temperature_derating",
    "thermal_efficiency",
    "three_phase_power",
    "total_harmonic_distortion_power",
    "total_switching_loss",
    "turn_off_loss",
    "turn_on_loss",
]
