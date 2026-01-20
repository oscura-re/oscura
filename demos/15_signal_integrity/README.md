# Signal Integrity

> TDR impedance profiling, S-parameters, crosstalk, eye metrics

**Oscura Version**: 0.1.2 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers signal integrity analysis for high-speed digital design. Oscura provides TDR-based impedance measurement, S-parameter analysis, eye diagram metrics, and crosstalk characterization.

### Key Capabilities

- TDR impedance extraction (Z0)
- Impedance profile along transmission line
- Discontinuity detection
- S-parameter analysis (S11, S21, S12, S22)
- Return loss / Insertion loss
- Eye height, width, opening
- Q-factor estimation
- Crosstalk (NEXT, FEXT)

### Standards Compliance

|Standard|Coverage|Notes|
|---|---|---|
|IEEE 370-2020|Partial|Interconnect fixture|
|IPC-TM-650|Partial|TDR testing|

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 10`)

### 30-Second Example

```python
from oscura.analyzers.eye.diagram import generate_eye
from oscura.analyzers.eye.metrics import eye_height, eye_width, measure_eye

# Generate eye diagram
eye = generate_eye(trace, unit_interval=1e-9, n_ui=2, max_traces=1000)

# Measure
height = eye_height(eye, position=0.5)
width = eye_width(eye, level=0.5)
metrics = measure_eye(eye, ber=1e-12)

print(f"Eye height: {height * 1e3:.2f} mV, Width: {width:.3f} UI")
print(f"Q-factor: {metrics.q_factor:.2f}")
```

---

## Demo Scripts

|Script|Purpose|Complexity|
|---|---|---|
|`tdr_impedance.py`|TDR-based impedance extraction|Intermediate|
|`impedance_profile.py`|Z0 along transmission line|Intermediate|
|`sparameters.py`|Touchstone S-parameter analysis|Advanced|
|`eye_metrics.py`|Eye height/width/opening|Basic|
|`crosstalk.py`|NEXT/FEXT characterization|Advanced|

## Related Demos

- [08_jitter_analysis](../08_jitter_analysis/) - Jitter impact on eyes
- [07_timing_measurements](../07_timing_measurements/) - Setup/hold
- [06_spectral_analysis](../06_spectral_analysis/) - Frequency response

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
