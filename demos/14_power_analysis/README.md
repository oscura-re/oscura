# Power Analysis

> IEEE 1459-2010 compliant power measurements, DC-DC efficiency, ripple analysis

**Oscura Version**: 0.1.2 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers power electronics analysis per IEEE 1459-2010. Oscura provides AC/DC power measurements, efficiency calculations, ripple analysis, and switching loss estimation.

### Key Capabilities

- Real power (P) measurement
- Reactive power (Q)
- Apparent power (S)
- Power factor (PF) - displacement and distortion
- Efficiency calculation (multi-output support)
- Output ripple measurement (Vpp and %)
- Switching loss analysis
- Conduction loss estimation

### Standards Compliance

|Standard|Coverage|Notes|
|---|---|---|
|IEEE 1459-2010|Full|Power quality definitions|
|IEC 61000-3-2|Partial|Harmonic limits|

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 09`)

### 30-Second Example

```python
from oscura.analyzers.power.ac_power import reactive_power, apparent_power, power_factor
from oscura.analyzers.power.efficiency import efficiency

# AC power analysis
q = reactive_power(voltage_trace, current_trace)
s = apparent_power(voltage_trace, current_trace)
pf = power_factor(voltage_trace, current_trace)
print(f"Q: {q:.2f} VAR, S: {s:.2f} VA, PF: {pf:.3f}")

# Efficiency
eff = efficiency(input_voltage, input_current, output_voltage, output_current)
print(f"Efficiency: {eff * 100:.1f}%")
```

---

## Demo Scripts

|Script|Purpose|Complexity|
|---|---|---|
|`ac_power_basics.py`|P, Q, S, power factor|Basic|
|`dcdc_efficiency.py`|Converter efficiency curves|Intermediate|
|`ripple_measurement.py`|Output voltage ripple|Basic|
|`switching_loss.py`|MOSFET loss estimation|Advanced|
|`ieee1459_compliance.py`|Full IEEE 1459 validation|Advanced|

## Related Demos

- [06_spectral_analysis](../06_spectral_analysis/) - Harmonic analysis
- [13_emc_compliance](../13_emc_compliance/) - EMC power quality
- [07_timing_measurements](../07_timing_measurements/) - Switching timing

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
