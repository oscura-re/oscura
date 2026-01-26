# File Format I/O

> Load and export data in CSV, HDF5, NumPy, MATLAB, and custom binary formats

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers all file format input/output operations. Oscura supports 18+ formats for loading and exporting waveform data, with configurable binary loaders for custom DAQ systems.

### Key Capabilities

- CSV loading/export
- NumPy NPZ/NPY format
- HDF5 format for large datasets
- MATLAB .mat format
- JSON export
- Custom binary formats (YAML-driven)
- Streaming for large files
- Format conversion workflows

### Supported Formats

| Category       | Formats                              |
| -------------- | ------------------------------------ |
| Oscilloscope   | Tektronix WFM, Rigol WFM, LeCroy TRC |
| Logic Analyzer | Sigrok, VCD                          |
| Generic        | CSV, NumPy, HDF5, MATLAB, WAV, JSON  |
| Automotive     | BLF, MF4 (ASAM)                      |
| Network        | PCAP, PCAPNG                         |

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 02`)

### 30-Second Example

```python
import oscura as osc

# Load any format
trace = osc.load("capture.wfm")

# Export to different formats
osc.export_csv(trace, "output.csv")
osc.export_hdf5(trace, "output.h5", compression="gzip")
```

---

## Demo Scripts

| Script                    | Purpose                    | Complexity   |
| ------------------------- | -------------------------- | ------------ |
| `csv_workflows.py`        | CSV import/export          | Basic        |
| `hdf5_large_files.py`     | HDF5 for large datasets    | Intermediate |
| `custom_binary_loader.py` | YAML-driven custom formats | Intermediate |
| `streaming_loader.py`     | Memory-efficient streaming | Advanced     |
| `format_converter.py`     | Format conversion utility  | Basic        |

## Related Demos

- [01_waveform_loading](../01_waveform_loading/) - Oscilloscope-specific loading
- [15_complete_workflows](../15_complete_workflows/) - End-to-end workflows

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
