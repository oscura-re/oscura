# Complete Workflows

> End-to-end reverse engineering workflows combining multiple Oscura capabilities

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category showcases complete end-to-end workflows combining multiple Oscura capabilities. Each workflow demonstrates a real-world analysis scenario from data acquisition to reporting.

### Key Capabilities

- Complete reverse engineering pipelines
- Multi-domain analysis (time + frequency + protocol)
- Automated report generation
- Multi-format data export
- Correlation analysis across signals
- Professional deliverables

---

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .[all]`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 15`)

### 30-Second Example

```python
import oscura as osc
from oscura.reporting import generate_report, save_pdf_report

# Complete analysis workflow
trace = osc.load("capture.wfm")

# Multi-domain analysis
freq = osc.frequency(trace)
rise = osc.rise_time(trace)
thd = osc.thd(trace)
snr = osc.snr(trace)

# Generate professional report
report = generate_report(trace, title="Signal Analysis")
save_pdf_report(report, "analysis_report.pdf")
```

---

## Demo Scripts

| Script                             | Purpose                         | Complexity   |
| ---------------------------------- | ------------------------------- | ------------ |
| `unknown_protocol_re.py`           | Full unknown protocol workflow  | Advanced     |
| `automotive_bus_analysis.py`       | Complete vehicle bus RE         | Advanced     |
| `embedded_debug_workflow.py`       | Firmware extraction via debug   | Advanced     |
| `power_supply_characterization.py` | DC-DC converter full analysis   | Intermediate |
| `adc_full_characterization.py`     | IEEE 1241 complete ADC test     | Advanced     |
| `emc_pre_compliance.py`            | Pre-compliance testing workflow | Intermediate |
| `automated_report.py`              | Multi-format report generation  | Basic        |

## Workflow Examples

### Unknown Protocol RE Workflow

```
1. Load capture
2. Auto-detect signal characteristics (level, baud)
3. Detect protocol type
4. Decode transport layer
5. Infer message format
6. Reverse CRC
7. Generate Wireshark dissector
8. Export results
```

### Automotive Bus Analysis Workflow

```
1. Load BLF/MF4 capture
2. Decode CAN frames
3. Discover message patterns
4. Generate DBC file
5. Map to OBD-II/J1939
6. Identify unknown messages
7. Generate report
```

---

## Related Demos

All other demo categories feed into complete workflows:

- [01-02](../01_waveform_loading/) - Data loading
- [03-05](../03_serial_protocols/) - Protocol decoding
- [06-10](../06_spectral_analysis/) - Signal analysis
- [11-14](../11_protocol_inference/) - Inference techniques

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
