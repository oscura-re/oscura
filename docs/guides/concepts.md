# Core Concepts

Understanding Oscura's design philosophy and key concepts.

## Signal Analysis Philosophy

Oscura treats **everything as a signal**:

- Oscilloscope waveforms (analog)
- Logic analyzer captures (digital)
- Network packets (protocol data)
- Sensor readings (time-series)

This unified view means the same analysis tools work across domains.

## Core Abstractions

### Traces

A **trace** is captured signal data with metadata:

- Time-series samples (1D arrays)
- Sample rate (timing information)
- Channel names and units
- Source information

Traces are the fundamental data structure in Oscura.

### Loaders

**Loaders** convert file formats into traces:

- Auto-detect format from file extension
- Parse vendor-specific formats (Tektronix, Rigol, Siglent)
- Handle industry standards (VCD, PCAP, HDF5)

See [Loader API](../api/loader.md) for supported formats.

### Analyzers

**Analyzers** extract measurements from traces:

- Waveform measurements (amplitude, frequency, rise time)
- Spectral analysis (FFT, THD, SNR, ENOB)
- Power measurements (AC/DC, ripple, efficiency)
- Signal integrity (jitter, eye diagrams, S-parameters)

Each analyzer returns a dictionary of named measurements.

### Protocol Decoders

**Decoders** extract structured data from digital signals:

- Transport protocols (UART, SPI, I2C, CAN)
- Network protocols (UDP, TCP via PCAP)
- Automotive protocols (OBD-II, J1939, UDS)

Decoders return frame objects with parsed fields.

## Design Principles

### 1. **Single Source of Truth**

Configuration lives in one place:

- Version in `pyproject.toml`
- Documentation in demos (not duplicated guides)
- Standards compliance in code (not external docs)

### 2. **IEEE Standards Compliance**

Where standards exist, Oscura implements them:

- **IEEE 181-2011**: Pulse measurements
- **IEEE 1241-2010**: ADC testing (SNR, SINAD, ENOB)
- **IEEE 1459-2010**: Power quality
- **IEEE 2414-2020**: Jitter measurements

See [Architecture](../images/architecture/index.md) for design details.

### 3. **Composable Analysis**

Build complex workflows from simple parts:

```python
# Each step is independent
trace = load(file)           # Load
filtered = filter(trace)     # Process
measurements = analyze(filtered)  # Measure
export(measurements, "report.json")  # Export
```

See [Workflows Guide](workflows.md) for patterns.

### 4. **Fail Gracefully**

Invalid results return `NaN`, not exceptions:

```python
result = frequency_detector(dc_signal)
# result = NaN (DC has no frequency)
# No exception raised
```

Check results with `np.isfinite()` before using.

## Data Flow

Typical Oscura workflow:

```
File → Loader → Trace → Analyzer → Measurements → Exporter → Output
         ↓
    Metadata (sample rate, channels, units)
```

**Working examples**: See [Complete Workflows Demo](https://github.com/oscura-re/oscura/tree/main/demos/19_complete_workflows)

## Use Case Categories

Oscura serves four primary domains:

### 1. **Analog Circuit Analysis**

Audio, power supplies, RF baseband, sensors
→ [Power Demo](https://github.com/oscura-re/oscura/tree/main/demos/14_power_analysis)

### 2. **Digital Protocol Analysis**

UART, SPI, I2C, CAN decoding and validation
→ [Protocol Demo](https://github.com/oscura-re/oscura/tree/main/demos/04_serial_protocols)

### 3. **Protocol Reverse Engineering**

Unknown signal analysis, CRC recovery, state machine learning
→ [RE Demo](https://github.com/oscura-re/oscura/tree/main/demos/17_signal_reverse_engineering)

### 4. **Compliance Validation**

IEEE standards, EMC testing, automotive diagnostics
→ [Spectral Demo](https://github.com/oscura-re/oscura/tree/main/demos/12_spectral_compliance)

## Performance Considerations

**Small files (<100MB)**: Load entirely into memory

**Large files (>100MB)**: Use streaming loaders
→ [Custom DAQ Demo](https://github.com/oscura-re/oscura/tree/main/demos/03_custom_daq)

**Very large datasets**: Consider chunking and parallel processing
→ [Pipelines API](../api/pipelines.md)

## Next Steps

- **Apply concepts**: Try the [Quick Start](quick-start.md)
- **See patterns**: Check [Common Workflows](workflows.md)
- **Choose APIs**: Read [Choosing Features](choosing-features.md)
- **Deep dive**: Explore [API Reference](../api/index.md)
