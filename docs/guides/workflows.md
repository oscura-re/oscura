# Common Workflows

Typical patterns for different signal analysis tasks.

## Oscilloscope Waveform Analysis

**Use case**: Load and analyze captured waveforms

**Pattern**:

1. Load oscilloscope file
2. Extract measurements
3. Perform spectral analysis
4. Export results

**Working example**: [Waveform Analysis Demo](https://github.com/oscura-re/oscura/tree/main/demos/01_waveform_analysis)

**When to use**:

- Analyzing Tektronix/Rigol/Siglent captures
- Validating signal integrity
- Measuring waveform parameters

---

## Protocol Decoding

**Use case**: Decode digital communication protocols

**Pattern**:

1. Load logic analyzer capture
2. Auto-detect protocol (or specify)
3. Extract decoded frames
4. Validate/analyze frames

**Working example**: [Protocol Decoding Demo](https://github.com/oscura-re/oscura/tree/main/demos/05_protocol_decoding)

**When to use**:

- Debugging UART/SPI/I2C communication
- Validating protocol timing
- Extracting message data

---

## Large File Processing

**Use case**: Memory-efficient analysis of multi-GB files

**Pattern**:

1. Use streaming loader
2. Process in chunks
3. Accumulate statistics
4. Generate summary

**Working example**: [Custom DAQ Demo](https://github.com/oscura-re/oscura/tree/main/demos/03_custom_daq)

**When to use**:

- Files >100MB
- Limited memory available
- Real-time streaming data

---

## Signal Reverse Engineering

**Use case**: Understand unknown digital signals

**Pattern**:

1. Characterize signal (analog vs digital, periodic vs aperiodic)
2. Detect clock and extract bits
3. Find frame boundaries
4. Analyze field structure
5. Detect checksums/CRCs
6. Generate protocol specification

**Working example**: [Signal RE Demo](https://github.com/oscura-re/oscura/tree/main/demos/17_signal_reverse_engineering)

**When to use**:

- Unknown protocol analysis
- Legacy system documentation
- Security research

---

## Automotive Diagnostics

**Use case**: Analyze CAN bus and OBD-II data

**Pattern**:

1. Load CAN capture (BLF/ASC/PCAP)
2. Decode OBD-II PIDs or J1939 PGNs
3. Extract diagnostic data
4. Generate DBC file for future analysis

**Working example**: [Automotive Demo](https://github.com/oscura-re/oscura/tree/main/demos/09_automotive)

**When to use**:

- Vehicle diagnostics
- CAN bus reverse engineering
- Fleet data analysis

---

## Compliance Testing

**Use case**: Validate against IEEE/EMC standards

**Pattern**:

1. Load test signal
2. Apply IEEE-compliant measurements
3. Compare against limits
4. Generate compliance report

**Working examples**:

- [Spectral Compliance (IEEE 1241)](https://github.com/oscura-re/oscura/tree/main/demos/12_spectral_compliance)
- [Jitter Analysis (IEEE 2414)](https://github.com/oscura-re/oscura/tree/main/demos/13_jitter_analysis)
- [EMC Testing (CISPR 16)](https://github.com/oscura-re/oscura/tree/main/demos/16_emc_compliance)

**When to use**:

- Product certification
- Quality control
- Standards validation

---

## Multi-Format Export

**Use case**: Export analysis results to various formats

**Pattern**:

1. Perform analysis
2. Generate report
3. Export to target format(s)

**Working example**: [All Output Formats](https://github.com/oscura-re/oscura/tree/main/demos/01_waveform_analysis)

**Supported formats**:

- CSV, JSON (data interchange)
- HDF5, MATLAB (scientific)
- Excel, HTML, Markdown (reports)

**When to use**:

- Sharing results with team
- Archiving analysis data
- Integration with other tools

---

## Batch Processing

**Use case**: Analyze multiple files with same workflow

**Pattern**:

1. Define analysis pipeline
2. Iterate over files
3. Apply consistent analysis
4. Aggregate results

**Working example**: [Pipelines API](../api/pipelines.md)

**When to use**:

- Regression testing
- Batch validation
- Dataset analysis

---

## Custom Analysis Pipeline

**Use case**: Build domain-specific analysis workflows

**Pattern**:

1. Compose existing analyzers
2. Add custom processing steps
3. Validate results
4. Package as reusable workflow

**Working example**: [Complete Workflows Demo](https://github.com/oscura-re/oscura/tree/main/demos/19_complete_workflows)

**When to use**:

- Repeated analysis tasks
- Domain-specific requirements
- Production testing

---

## Quick Decision Tree

**Start here** → What do you have?

- **Oscilloscope file (.wfm)** → [Waveform Analysis](#oscilloscope-waveform-analysis)
- **Logic analyzer file (.sr, .vcd)** → [Protocol Decoding](#protocol-decoding)
- **Network capture (.pcap)** → [Protocol Decoding](#protocol-decoding)
- **CAN/automotive data** → [Automotive Diagnostics](#automotive-diagnostics)
- **Large file (>100MB)** → [Large File Processing](#large-file-processing)
- **Unknown signal** → [Signal RE](#signal-reverse-engineering)
- **Need compliance** → [Compliance Testing](#compliance-testing)

## Next Steps

- **Learn fundamentals**: Read [Core Concepts](concepts.md)
- **Choose the right API**: See [Choosing Features](choosing-features.md)
- **Browse all demos**: [GitHub demos directory](https://github.com/oscura-re/oscura/tree/main/demos)
- **Dive into API**: [API Reference](../api/index.md)
