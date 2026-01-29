# Data Loading Demos

This directory contains 10 comprehensive demonstrations of file format loading capabilities in Oscura.

## Demos Overview

| Demo | File | Description | Key Concepts |
|------|------|-------------|--------------|
| 01 | `01_oscilloscopes.py` | Oscilloscope file formats (Tektronix, Rigol, MSO) | Vendor formats, metadata extraction |
| 02 | `02_logic_analyzers.py` | Logic analyzer captures (Sigrok, VCD, Saleae) | Digital waveforms, timing analysis |
| 03 | `03_automotive_formats.py` | Automotive bus captures (CAN, LIN) | Frame extraction, protocol parsing |
| 04 | `04_scientific_formats.py` | Scientific instruments (TDMS, HDF5, WAV) | Hierarchical data, multi-format |
| 05 | `05_custom_binary.py` | Custom binary loaders with headers | Endianness, struct parsing |
| 06 | `06_streaming_large_files.py` | Memory-efficient large file loading | Chunking, memory-mapped files |
| 07 | `07_multi_channel.py` | Multi-channel data handling | Synchronization, cross-channel |
| 08 | `08_network_formats.py` | Network captures (PCAP, Touchstone) | Packet analysis, S-parameters |
| 09 | `09_lazy_loading.py` | Lazy loading and deferred evaluation | On-demand loading, memory optimization |
| 10 | `10_format_conversion.py` | Format conversion workflows | Binary ↔ CSV, metadata preservation |

## Running Demos

Each demo is self-contained and generates its own synthetic data:

```bash
# Run a single demo
python3 demos/01_data_loading/01_oscilloscopes.py

# Run all data loading demos
for demo in demos/01_data_loading/[0-9]*.py; do
    python3 "$demo"
done
```

## File Format Support

### Oscilloscope Formats
- **Tektronix**: `.wfm` files from DPO/MSO series
- **Rigol**: `.wfm` files from DS1000/DS2000 series
- **Generic**: `.csv`, `.dat` text-based formats

### Logic Analyzer Formats
- **Sigrok**: `.sr` archive files (PulseView)
- **VCD**: `.vcd` Value Change Dump (HDL simulators)
- **Saleae**: Binary captures from Logic analyzers

### Automotive Formats
- **CAN**: Controller Area Network frame captures
- **LIN**: Local Interconnect Network captures

### Scientific Formats
- **TDMS**: National Instruments LabVIEW data
- **HDF5**: Hierarchical Data Format 5
- **WAV**: Audio waveforms (acoustics, vibration)

### Network Formats
- **PCAP**: Network packet captures (Wireshark)
- **Touchstone**: S-parameter measurements (VNA)

### Custom Formats
- **Binary**: Raw binary with custom headers
- **Configurable**: User-defined binary structures

## Key Features Demonstrated

### 1. Metadata Extraction
All demos show how to extract and validate:
- Sample rates and timing information
- Vertical scales and offsets
- Channel configurations
- Instrument identification

### 2. Memory Management
Efficient handling of large datasets:
- Chunk-based processing (06)
- Memory-mapped file access (06)
- Lazy loading patterns (09)
- Streaming algorithms (06)

### 3. Multi-Channel Handling
Managing multiple simultaneous channels:
- Channel synchronization (07)
- Cross-channel analysis (07)
- Mixed analog/digital data (07)

### 4. Format Interoperability
Converting between formats:
- Binary ↔ CSV conversion (10)
- Metadata preservation (10)
- Roundtrip validation (10)

## IEEE Standards Referenced

- **IEEE 181-2011**: Waveform and Vector Measurements (oscilloscopes)
- **IEEE 1364-2005**: Verilog VCD format (logic analyzers)
- **IEEE 1057-2017**: Digitizing Waveform Recorders

## Related Categories

- **02_basic_analysis**: Use loaded data for measurements
- **03_protocol_decoding**: Decode protocols from loaded captures
- **04_advanced_analysis**: Advanced signal processing on loaded data

## Best Practices

### File Format Selection
- **Binary formats**: Fast, efficient, exact precision
- **Text formats (CSV)**: Human-readable, portable, larger size
- **Compressed formats**: Balance size vs. speed
- **Vendor formats**: Maximum fidelity, metadata-rich

### Performance Considerations
- Files < 100 MB: Load entirely into memory
- Files 100 MB - 1 GB: Use chunk-based processing
- Files 1-10 GB: Memory-mapped file access
- Files > 10 GB: Streaming algorithms + databases

### Metadata Management
Always extract and validate:
- Sample rate (critical for analysis)
- Vertical scale/offset (voltage scaling)
- Coupling mode (AC/DC)
- Timestamp information (synchronization)

## Common Workflows

### 1. Load and Analyze Oscilloscope Capture
```python
from oscura.loaders import load

# Load Tektronix capture
trace = load("TEK00001.wfm")
print(f"Sample rate: {trace.metadata.sample_rate}")
print(f"Samples: {len(trace.data)}")

# Basic analysis
import numpy as np
rms = np.sqrt(np.mean(trace.data**2))
print(f"RMS voltage: {rms:.4f} V")
```

### 2. Process Large File in Chunks
```python
import numpy as np

chunk_size = 1000000  # 1M samples per chunk
with open("large_capture.bin", "rb") as f:
    while True:
        chunk_bytes = f.read(chunk_size * 8)  # 8 bytes per float64
        if not chunk_bytes:
            break
        chunk = np.frombuffer(chunk_bytes, dtype=np.float64)
        # Process chunk...
```

### 3. Convert Format with Metadata
```python
from oscura.loaders import load
import numpy as np

# Load proprietary format
trace = load("capture.wfm")

# Save as CSV with metadata
with open("capture.csv", "w") as f:
    f.write(f"# Sample Rate: {trace.metadata.sample_rate} Hz\n")
    f.write(f"# Vertical Scale: {trace.metadata.vertical_scale} V/div\n")
    f.write("Time (s),Amplitude (V)\n")
    t = np.arange(len(trace.data)) / trace.metadata.sample_rate
    for time, value in zip(t, trace.data):
        f.write(f"{time:.9e},{value:.6e}\n")
```

## Implementation Notes

All demos:
- Use `demonstrations.common.BaseDemo` pattern
- Generate synthetic data (no external files required)
- Include comprehensive validation with `ValidationSuite`
- Follow project coding standards (type hints, docstrings)
- Self-contained and executable independently

## Troubleshooting

### Import Errors
Demos use `sys.path.insert(0, ...)` to find `demonstrations.common`. Run from repo root:
```bash
cd /path/to/oscura
python3 demos/01_data_loading/01_oscilloscopes.py
```

### Memory Issues
If demos fail with memory errors, reduce synthetic data sizes in `generate_test_data()` methods.

### Missing Dependencies
Some loaders require optional dependencies:
```bash
uv sync --all-extras  # Install all optional dependencies
```

## Contributing

When adding new data loading demos:
1. Follow the `BaseDemo` pattern
2. Generate synthetic test data (no external files)
3. Include comprehensive validation
4. Add to this README with summary
5. Update related demos links
6. Test with `python3 -m py_compile <demo>.py`
