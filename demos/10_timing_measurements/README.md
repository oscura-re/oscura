# Timing Measurements

> IEEE 181-2011 compliant rise/fall time, pulse width, duty cycle measurements

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers time-domain measurements per IEEE 181-2011. Oscura provides configurable reference levels, statistical analysis, and JEDEC timing parameter extraction.

### Key Capabilities

- Rise time (10%-90% or configurable)
- Fall time (90%-10% or configurable)
- Pulse width
- Duty cycle
- Slew rate (V/ns)
- Propagation delay
- Setup and hold time (JEDEC)

### Standards Compliance

| Standard      | Coverage | Notes                              |
| ------------- | -------- | ---------------------------------- |
| IEEE 181-2011 | Full     | Rise/fall time, pulse measurements |
| JEDEC JESD8C  | Partial  | Setup/hold timing                  |

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 07`)

### 30-Second Example

```python
from oscura.analyzers.waveform.measurements import rise_time, fall_time, slew_rate

t_rise = rise_time(trace, ref_levels=(0.1, 0.9))
t_fall = fall_time(trace, ref_levels=(0.9, 0.1))
sr = slew_rate(trace)

print(f"Rise time: {t_rise * 1e9:.2f} ns")
print(f"Fall time: {t_fall * 1e9:.2f} ns")
print(f"Slew rate: {sr:.3f} V/ns")
```

---

## Demo Scripts

| Script                     | Purpose                     | Complexity   |
| -------------------------- | --------------------------- | ------------ |
| `rise_fall_time.py`        | Basic rise/fall measurement | Basic        |
| `pulse_characteristics.py` | Width, duty cycle, period   | Basic        |
| `slew_rate_analysis.py`    | Edge rate characterization  | Intermediate |
| `setup_hold_timing.py`     | Digital timing margins      | Intermediate |
| `ieee181_compliance.py`    | Full IEEE 181 validation    | Advanced     |

## Related Demos

- [06_spectral_analysis](../06_spectral_analysis/) - Frequency domain
- [08_jitter_analysis](../08_jitter_analysis/) - Jitter decomposition
- [10_signal_integrity](../10_signal_integrity/) - Setup/hold timing

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
