# Jitter Analysis

> IEEE 2414-2020 compliant TIE, RJ/DJ decomposition, eye diagrams, bathtub curves

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers jitter measurement and analysis per IEEE 2414-2020. Oscura provides jitter decomposition (RJ/DJ), eye diagram generation, and BER estimation.

### Key Capabilities

- Time Interval Error (TIE) measurement
- Random Jitter (RJ) extraction
- Deterministic Jitter (DJ) decomposition
- Duty Cycle Distortion (DCD)
- Data-Dependent Jitter (DDJ)
- Periodic Jitter (PJ)
- Total Jitter (TJ) at BER
- Eye diagram generation
- Bathtub curve plotting

### Standards Compliance

|Standard|Coverage|Notes|
|---|---|---|
|IEEE 2414-2020|Full|Jitter measurement|
|OIF-CEI-28G|Partial|28G jitter requirements|

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 08`)

### 30-Second Example

```python
from oscura.analyzers.jitter.measurements import tie_from_edges
from oscura.analyzers.jitter.decomposition import extract_rj, extract_dj
from oscura.analyzers.jitter.ber import tj_at_ber

# TIE from edge timestamps
tie = tie_from_edges(edge_timestamps, expected_period=1e-9)
print(f"TIE RMS: {np.std(tie) * 1e12:.2f} ps")

# Jitter decomposition
rj = extract_rj(tie)
dj = extract_dj(tie)
print(f"RJ (1-sigma): {rj * 1e12:.2f} ps, DJ (pk-pk): {dj * 1e12:.2f} ps")

# Total jitter at BER
tj = tj_at_ber(rj, dj, ber=1e-12)
print(f"TJ at BER 1e-12: {tj * 1e12:.2f} ps")
```

---

## Demo Scripts

|Script|Purpose|Complexity|
|---|---|---|
|`tie_measurement.py`|Time Interval Error extraction|Basic|
|`rj_dj_decomposition.py`|Jitter separation|Intermediate|
|`eye_diagram.py`|Eye diagram generation|Intermediate|
|`bathtub_curve.py`|BER estimation and bathtub|Advanced|
|`ieee2414_compliance.py`|Full IEEE 2414 validation|Advanced|

## Related Demos

- [07_timing_measurements](../07_timing_measurements/) - Time-domain measurements
- [10_signal_integrity](../10_signal_integrity/) - Eye metrics, SI analysis
- [06_spectral_analysis](../06_spectral_analysis/) - Spectral jitter methods

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
