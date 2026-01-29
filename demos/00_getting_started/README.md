# Getting Started with Oscura

This category contains the essential first steps for learning Oscura's hardware reverse engineering framework.

## Demonstrations

### 00_hello_world.py
**Simplest possible Oscura workflow**

- Generates a 1kHz sine wave
- Performs basic measurements (amplitude, frequency, RMS)
- Validates results automatically
- Perfect for testing your installation

**Run**: `python 00_hello_world.py`

**Capabilities Demonstrated**:
- `oscura.amplitude()` - Peak-to-peak voltage measurement
- `oscura.frequency()` - Frequency extraction
- `oscura.rms()` - RMS voltage calculation

---

### 01_core_types.py
**Understanding Oscura's data structures**

- `TraceMetadata` - Timing and calibration information
- `WaveformTrace` - Analog waveform signals
- `DigitalTrace` - Digital/logic signals
- `ProtocolPacket` - Decoded protocol data
- `CalibrationInfo` - Instrument configuration

**Run**: `python 01_core_types.py`

**IEEE Standards**: IEEE 1241-2010 (ADC Terminology)

---

### 02_supported_formats.py
**Overview of file format support**

- Lists all 21+ supported file formats
- Organized by category (oscilloscopes, logic analyzers, automotive, scientific)
- Shows auto-detection capabilities
- Provides usage examples for each category

**Run**: `python 02_supported_formats.py`

**Format Categories**:
- Oscilloscopes: .wfm, .isf, .bin
- Logic Analyzers: .sr, .vcd
- Automotive: .blf, .asc, .mf4
- Scientific: .tdms, .h5, .hdf5, .npz, .wav
- Network: .pcap, .pcapng
- RF: .s2p, .snp

---

## Next Steps

After completing these demos, explore:
- **01_data_loading/** - Format-specific loading examples
- **02_basic_analysis/** - Signal processing and measurements
- **03_protocol_decoding/** - Decode UART, SPI, I2C, CAN protocols
- **04_advanced_analysis/** - FFT, filtering, spectral analysis

## Requirements

All demos in this category:
- Generate their own test data (no external files needed)
- Include built-in validation
- Run standalone with `python <demo_name>.py`
- Support `--verbose` and `--no-validate` flags

## Common Issues

**Import Error**: Ensure Oscura is installed: `uv sync --all-extras`

**Validation Failed**: This indicates a calculation discrepancy. Check your Python version (3.12+ required) and numpy installation.
