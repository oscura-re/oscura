# Changelog

All notable changes to Oscura will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

## [0.1.2] - 2026-01-18

### Project Renamed: TraceKit → Oscura

**Oscura** is the new name for this project. The rename reflects our identity as a unified hardware reverse engineering framework.

- **New package name**: `oscura` (formerly `tracekit`)
- **New tagline**: "Revealing what's hidden in every signal"
- **New organization**: github.com/oscura-re
- **New repository**: github.com/oscura-re/oscura

**Migration**: No backward compatibility needed - this is the first public release under the new name.

### Initial Public Release

Oscura 0.1.0 is the first public release of the comprehensive hardware reverse engineering framework for security researchers, right-to-repair advocates, defense analysts, and commercial intelligence teams.

### Core Features

#### Signal Analysis & Measurement

- **Waveform Analysis** - Rise/fall time, frequency, duty cycle, overshoot (IEEE 181-2011 compliant)
- **Spectral Analysis** - FFT, PSD, spectrograms, wavelets, THD, SNR, SINAD, ENOB (IEEE 1241-2010)
- **Audio Analysis** - THD, SNR, SINAD, ENOB, harmonic distortion
- **Power Analysis** - AC/DC power, efficiency, ripple, power factor (IEEE 1459-2010)
- **Jitter Analysis** - TIE, period jitter, RJ/DJ decomposition (IEEE 2414-2020)
- **Signal Integrity** - Eye diagrams, S-parameter analysis, TDR impedance profiling

#### Protocol Support (16+ Decoders)

**Serial Protocols:**

- UART, SPI, I2C, 1-Wire, I2S, Manchester encoding

**Automotive:**

- CAN, CAN-FD, LIN, FlexRay, OBD-II (54 PIDs), J1939 (154 PGNs), UDS (ISO 14229)

**Debug Interfaces:**

- JTAG, SWD

**Network:**

- USB, HDLC

**Features:**

- Auto-detection and baud rate recovery
- Checksum validation (XOR, SUM, CRC-8/16/32)
- DTC database (210 codes, SAE J2012)

#### Reverse Engineering

- **SignalBuilder API** - Fluent API for composable signal generation (analog waveforms, protocol signals, noise/impairments)
- **Protocol Inference** - CRC polynomial recovery, state machine learning (L\* algorithm), field boundary detection
- **Pattern Recognition** - Counter patterns, toggle patterns, sequence detection
- **CAN Bus RE** - Message discovery, signal extraction, DBC file generation
- **Complete RE Workflow** - 8-step automated reverse engineering pipeline for unknown digital signals

#### Convenience APIs

- **quick_spectral()** - One-call spectral analysis returning all IEEE 1241 metrics
- **auto_decode()** - Unified protocol detection and decoding (UART/SPI/I2C/CAN)
- **smart_filter()** - Intelligent filtering with automatic noise source detection
- **reverse_engineer_signal()** - Complete reverse engineering workflow for unknown signals

#### Discovery & Analysis

- **Signal Characterization** - Automatic signal type detection (analog/digital/mixed)
- **Anomaly Detection** - Statistical anomaly identification
- **Quality Assessment** - Data quality validation and metrics

#### File Format Support

**Oscilloscopes:**

- Tektronix WFM, Rigol WFM, Siglent, generic binary

**Logic Analyzers:**

- Sigrok (.sr), VCD (Value Change Dump)

**Network Captures:**

- PCAP, PCAPNG with full protocol parsing (dpkt integration)

**Automotive:**

- Vector BLF/ASC, ASAM MDF/MF4, DBC, CSV

**Scientific Data:**

- TDMS (LabVIEW), HDF5, NumPy, WAV, CSV

**RF/Network:**

- Touchstone S-parameters (.s1p, .s2p, etc.)

#### Signal Processing

- **Filtering** - IIR, FIR, Butterworth, Chebyshev, Bessel, Elliptic filters
- **Triggering** - Edge, pattern, pulse width, glitch, runt, window triggers
- **Arithmetic** - Add, subtract, differentiate, integrate, FFT operations
- **Math Operations** - RMS, mean, peak detection, envelope, correlation

#### EMC & Compliance Testing

- **Standards Support** - CISPR 32, IEC 61000-3-2, IEC 61000-4-2/4-4, MIL-STD-461G
- **EMI Analysis** - Conducted/radiated emissions, immunity testing
- **EMI Fingerprinting** - Automatic emission source identification
- **Limit Testing** - Automated compliance checking with limit masks

#### Professional Features

- **Report Generation** - PDF, HTML, Markdown, CSV exports
- **Session Management** - Workspace persistence and replay
- **Batch Processing** - Multi-file analysis with progress tracking
- **Visualization** - Waveform plotting, eye diagrams, spectrograms, constellation diagrams
- **Memory Management** - Large file handling with streaming support
- **Comparison Tools** - Golden waveform comparison, mask testing

### Demonstrations

31 comprehensive demos covering all major features:

**Comprehensive Demos (8):**

1. Waveform Analysis - 7 analysis sections (measurements, spectral, power, statistics, filtering, protocols, math)
2. Protocol Decoding - UART, SPI, I2C multi-protocol with auto-detection
3. UDP Packet Analysis - Traffic metrics, payload analysis, pattern detection, field inference
4. Automotive - CAN, OBD-II, UDS, J1939, LIN, FlexRay protocols with DBC generation
5. Mixed-Signal - Clock recovery, jitter analysis, IEEE 2414-2020 compliance
6. Spectral Compliance - IEEE 1241-2010 validation (THD, SNR, SINAD, ENOB, SFDR)
7. Signal Reverse Engineering - 5-phase RE workflow
8. EMC Compliance - CISPR 32, IEC 61000 compliance testing

**Serial Protocols (6):**

- JTAG, SWD, USB, 1-Wire, Manchester, I2S

**Automotive Protocols (2):**

- LIN, FlexRay

**Timing & Jitter (3):**

- IEEE 181 pulse measurements, bathtub curves, DDJ/DCD analysis

**Power Analysis (2):**

- DC-DC efficiency, ripple analysis

**Signal Integrity (3):**

- Setup/hold timing, TDR impedance, S-parameters

**Protocol Inference (3):**

- CRC reverse engineering, Wireshark dissector generation, state machine learning

**Advanced Inference (3):**

- Bayesian inference, protocol DSL, active learning

**Complete Workflows (3):**

- Network analysis, unknown signal RE, automotive full workflow

**File Format I/O (1):**

- VCD loader

**Custom DAQ (3):**

- Simple, chunked, optimal streaming loaders

**Demo Features:**

- All demos support `--data-file` CLI argument for loading pre-captured data
- Auto-detection of default data from `demo_data/` directories
- Synthetic generation fallback when no files available
- 678.67 MB of generated demo data across 25 files
- Validation suite: 30/31 demos passing (96.8% success rate)

### Data Loading Feature

- **BaseDemo Enhancement** - Added `data_file` parameter and `find_default_data_file()` helper
- **Three-tier Loading** - CLI override → default file → synthetic fallback
- **Consistent Pattern** - All 31 demos follow standardized data loading approach
- **Multiple Formats** - NPZ, VCD, BIN, MF4, PCAP support across different demo types

### Infrastructure

- **Python 3.12+ Support** - Full type hints and modern Python features
- **Dependencies** - Optimized core dependencies (removed unused plotly/bokeh, moved reportlab to extras)
- **Testing** - 18,083 tests passing, 255 skipped, 10 xfailed (99.6% pass rate)
- **Code Quality** - 100% pass rate on all quality checks (ruff, mypy, prettier, markdownlint)
- **Pre-commit Hooks** - 21 hooks covering format, lint, security, documentation
- **Pre-push Verification** - Full CI simulation with 3-stage verification (95% CI coverage)
- **CI/CD** - GitHub Actions with parallel test matrix (Python 3.12/3.13, 8 test groups)
- **Documentation** - MkDocs with strict link validation, comprehensive API docs

### Standards Compliance

- **IEEE 181-2011** - Pulse measurements (rise/fall time, overshoot, slew rate)
- **IEEE 1057-2017** - Digitizer characterization and timing analysis
- **IEEE 1241-2010** - ADC testing (SNR, SINAD, ENOB, THD, SFDR)
- **IEEE 2414-2020** - Jitter measurements (TIE, period jitter, RJ/DJ decomposition)
- **IEEE 1459-2010** - Power quality measurements
- **CISPR 16** - EMC compliance testing with limit masks
- **IEC 61000** - Electromagnetic compatibility standards
- **MIL-STD-461G** - Military EMI/EMC requirements
- **SAE J1939** - Heavy-duty vehicle CAN diagnostics
- **ISO 14229** - Unified Diagnostic Services (UDS)

### Installation

```bash
pip install oscura
```

**Optional Dependencies:**

```bash
pip install oscura[all]           # All features
pip install oscura[automotive]    # Automotive protocols
pip install oscura[visualization] # Plotting support
pip install oscura[reporting]     # PDF report generation
```

### Quick Start

```python
import oscura as osc

# Load and analyze a waveform
trace = osc.load("capture.wfm")
print(f"Rise time: {osc.rise_time(trace):.2e} s")

# Quick spectral analysis
metrics = osc.quick_spectral(trace, fundamental=1000)
print(f"THD: {metrics.thd_db:.1f} dB, SNR: {metrics.snr_db:.1f} dB")

# Auto-decode protocol
result = osc.auto_decode(trace)
print(f"Protocol: {result.protocol}, Frames: {len(result.frames)}")

# Generate test signals
signal = (osc.SignalBuilder(sample_rate=1e6, duration=0.01)
    .add_sine(frequency=1000)
    .add_noise(snr_db=40)
    .build())

# Reverse engineer unknown signal
result = osc.workflows.reverse_engineer_signal(trace)
print(result.protocol_spec)
```

### Known Issues

- USB demo has pre-existing PID validation bug (not related to data loading feature)
- GitHub Actions CI requires billing resolution (code is fully verified locally)

### Contributors

- lair-click-bats (primary author)
- Claude Code (AI development assistance)

### License

MIT License - See LICENSE file for details

---

[Unreleased]: https://github.com/oscura-re/oscura/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/oscura-re/oscura/releases/tag/v0.1.2
