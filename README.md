# Oscura

**Hardware Reverse Engineering Framework**

Extract complete system understanding from any signal source—oscilloscopes, logic analyzers, network captures, side-channel traces—through:

- Unknown protocol reverse engineering (physical layer detection, message format inference, field boundaries, CRC recovery)
- State machine extraction and protocol behavior analysis
- Side-channel cryptanalysis (power analysis, timing attacks, EM leakage)
- IEEE-compliant measurements and signal characterization

[![CI](https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/ci.yml)
[![Code Quality](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/oscura-re/oscura/graph/badge.svg)](https://codecov.io/gh/oscura-re/oscura)
[![PyPI version](https://img.shields.io/pypi/v/oscura)](https://pypi.org/project/oscura/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Vision

Hardware systems are obscured through proprietary protocols, undocumented interfaces, cryptographic obfuscation, and black-box designs. Whether imposed by vendors, governments, or time itself—**Oscura illuminates what others obscure.**

**Built For:**

- **Security researchers** analyzing embedded systems, IoT vulnerabilities, side-channel leakage
- **Hardware engineers** validating signal integrity, power supply design, EMC compliance
- **Protocol developers** testing implementations against IEEE standards
- **Automotive engineers** analyzing CAN/LIN/FlexRay for diagnostics and security research
- **Right-to-repair advocates** documenting and replicating undocumented interfaces
- **Academic researchers** studying signal processing, cryptanalysis, and hardware security
- **Vintage computing enthusiasts** preserving and analyzing retro hardware (1960s–present)
- **Defense and intelligence analysts** for forensics and threat assessment

**What We Enable:**

- **Protocol Reverse Engineering**: CRC recovery, message format inference, state machine extraction for security research, protocol development, right-to-repair documentation
- **Signal Intelligence**: Auto-classify unknown signals (digital/analog, periodicity, SNR) to guide reverse engineering strategy
- **Cryptographic Analysis**: Side-channel attacks (DPA/CPA, timing, mutual information) for vulnerability assessment and implementation validation
- **Automotive Reverse Engineering**: CAN/LIN/FlexRay analysis (differential, stimulus-response, hypothesis-driven discovery) for aftermarket, diagnostics, security research (no vendor lock-in)
- **Debug Interface Analysis**: JTAG/SWD/UART for firmware extraction, bootloader analysis, and development/recovery workflows
- **Binary Format Recovery**: 100+ magic bytes, structure alignment, auto-parser generation for right-to-repair and vulnerability research
- **Obsolete System Replication**: Logic family auto-detection (ECL/RTL/DTL/TTL, 1960s-present), IC timing validation, modern replacements for vintage hardware preservation
- **Evidence-Based Discovery**: Hypothesis tracking, statistical validation, confidence scoring, full audit trails for reproducible reverse engineering
- **Attack Surface Mapping**: State machine extraction, stimulus-response correlation, differential analysis for vulnerability discovery and exploitation
- **Intelligence Sharing**: Wireshark dissectors (validated Lua), DBC files, auto-generated parsers, multi-format reports (PDF/HTML/PPTX) for collaboration

---

## Installation

```bash
# Production use
pip install oscura

# Development (recommended - includes quality tools)
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh
```

**Requirements:** Python 3.12+ | [See pyproject.toml for dependencies](pyproject.toml)

---

## Quick Start

### Decode Unknown Protocol in 5 Lines

```python
import oscura as osc

# Load oscilloscope capture
trace = osc.load("mystery_device.wfm")

# Auto-detect and decode
decoder = osc.auto_detect_protocol(trace)
messages = decoder.decode(trace)
print(f"Decoded {len(messages)} {decoder.name} messages")
```

### Side-Channel Attack on AES

```python
from oscura.loaders import load_chipwhisperer
from oscura.analyzers.side_channel import CPAAnalyzer

# Load power traces from ChipWhisperer capture
traces = load_chipwhisperer("aes_traces.npy")

# Correlation Power Analysis
cpa = CPAAnalyzer(leakage_model="hamming_weight", target_byte=0)
result = cpa.analyze(traces.traces, traces.plaintexts)

print(f"Key byte: 0x{result.key_guess:02X}")
print(f"Correlation: {result.max_correlation:.4f}")
print(f"Confidence: {'HIGH' if result.max_correlation > 0.8 else 'LOW'}")
```

### Black-Box Protocol Reverse Engineering

```python
from oscura.sessions import BlackBoxSession

# Create analysis session
session = BlackBoxSession(name="IoT Device RE")

# Differential analysis: compare device states
session.add_recording("idle", "idle.bin")
session.add_recording("button_press", "button.bin")
diff = session.compare("idle", "button_press")

# Automatic field detection
spec = session.generate_protocol_spec()
print(f"Found {len(spec['fields'])} protocol fields")

# Export Wireshark dissector
session.export_results("dissector", "protocol.lua")
```

### Automotive CAN Bus Analysis

```python
from oscura.automotive.can import CANSession

session = CANSession(name="Vehicle RE")
session.add_recording("idle", "idle.blf")
session.add_recording("accelerate", "accel.blf")

# Identify changed CAN IDs
diff = session.compare("idle", "accelerate")
print(f"Changed IDs: {diff.details['changed_ids']}")

# Export DBC file for further analysis
session.export_dbc("vehicle.dbc")
```

**More examples:** [20 demo categories](demonstrations/) with 112 comprehensive demonstrations covering every capability

---

## Core Capabilities

### Protocol Reverse Engineering & Development

| Capability                    | Use Cases (Development / Reverse Engineering)                                                                                                                                                                                                 | Location                                                |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **CRC Recovery & Validation** | Checksum development validation / XOR differential technique recovers polynomial, init, xor_out, reflection flags from 4+ message-CRC pairs. Identifies 12+ standard algorithms or brute-forces custom CRCs for proprietary checksum breaking | [`inference/`](src/oscura/inference/)                   |
| **Message Format Inference**  | Protocol documentation / IPART-style ensemble analysis (entropy, alignment, variance, distribution, n-grams) with confidence-scored field boundaries and automatic checksum/counter/timestamp detection                                       | [`inference/`](src/oscura/inference/)                   |
| **State Machine Extraction**  | Protocol testing with L\* active learning (systematic querying) and RPNI passive learning / Map authentication flows, command sequences, and hidden states for attack surface analysis with evidence-based hypothesis testing                 | [`inference/`](src/oscura/inference/)                   |
| **Binary Format Recovery**    | Firmware documentation / Magic byte detection (100+ formats including executables, images, archives), structure alignment inference (1/2/4/8/16-byte), auto-generate Python parsers and YAML specs for proprietary file format analysis       | [`inference/`](src/oscura/inference/)                   |
| **Signal Intelligence**       | Automatic analysis guidance / Auto-classify digital vs analog signals, detect periodicity (autocorrelation + FFT), estimate SNR, recommend suitable measurement techniques for unknown signal sources                                         | [`inference/`](src/oscura/inference/)                   |
| **Pattern Discovery**         | Sync marker identification / Find repeating signatures, frame boundaries, magic bytes in raw captures (malware C2, IoT traffic, proprietary protocols)                                                                                        | [`analyzers/patterns/`](src/oscura/analyzers/patterns/) |
| **Physical Layer Detection**  | PCB debugging / Auto-detect baud rates, clock frequencies, logic levels for unknown test points and debug interfaces on embedded devices                                                                                                      | [`inference/`](src/oscura/inference/)                   |
| **Stream Reassembly**         | Protocol validation / TCP/UDP stream reconstruction, framing detection, and packet boundary recovery from fragmented captures                                                                                                                 | [`inference/`](src/oscura/inference/)                   |
| **BlackBox Workflow**         | End-to-end RE pipeline / Differential analysis → field hypothesis generation → state machine inference → multi-format export (Wireshark dissectors, DBC, parsers) with full evidence tracking and confidence scoring                          | [`sessions/`](src/oscura/sessions/)                     |

### Cryptographic & Security Analysis

| Capability                      | Use Cases (Validation / Reverse Engineering)                                                                                                                                                                   | Location                                                        |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Power Analysis (DPA/CPA)**    | Crypto implementation validation / Differential and Correlation Power Analysis for key extraction from smart cards, TPMs, secure enclaves with configurable leakage models (Hamming weight/distance)           | [`analyzers/side_channel/`](src/oscura/analyzers/side_channel/) |
| **Timing Attack Analysis**      | Constant-time validation / Statistical timing analysis with mutual information calculation (quantify leakage in bits), effect size measurement (Cohen's d), and outlier detection for vulnerability assessment | [`analyzers/side_channel/`](src/oscura/analyzers/side_channel/) |
| **ChipWhisperer Integration**   | Educational side-channel labs / Direct integration for power/EM trace collection and analysis with standardized formats (.npy, .trs)                                                                           | [`loaders/`](src/oscura/loaders/)                               |
| **Jitter Analysis (IEEE 2414)** | Clock quality validation / TIE, period jitter, RJ/DJ decomposition for detecting clock glitching, fault injection attacks, and timing manipulation                                                             | [`analyzers/jitter/`](src/oscura/analyzers/jitter/)             |
| **Power Supply Analysis**       | DC-DC converter design / Voltage rail characterization for identifying fault injection targets (brownout thresholds, transient response, ripple analysis)                                                      | [`analyzers/power/`](src/oscura/analyzers/power/)               |

### Protocol Decoding

| Category             | Use Cases (Development / Reverse Engineering)                                 | Protocols                                     |
| -------------------- | ----------------------------------------------------------------------------- | --------------------------------------------- |
| **Serial**           | Sensor development, device debugging / Debug console access, flash extraction | UART, SPI, I2C, 1-Wire, I2S, Manchester, HDLC |
| **Automotive**       | Diagnostics, aftermarket dev / ECU security analysis, CAN injection testing   | CAN, CAN-FD, LIN, FlexRay, J1939              |
| **Debug Interfaces** | Firmware development, recovery / Firmware extraction, bootloader analysis     | JTAG (TAP state), SWD, USB                    |
| **Parallel Bus**     | Vintage hardware restoration, instrument control / Industrial system security | IEEE-488 (GPIB), Centronics, ISA              |

**Location:** [`analyzers/protocols/`](src/oscura/analyzers/protocols/) | [`automotive/`](src/oscura/automotive/)

**Auto-detection:** Baud rate recovery, clock extraction, logic level adaptation, confidence scoring

### File Format Support

| Category            | Formats                                       | Location                                                |
| ------------------- | --------------------------------------------- | ------------------------------------------------------- |
| **Oscilloscopes**   | Tektronix (.wfm), Rigol (.wfm), LeCroy (.trc) | [`loaders/`](src/oscura/loaders/)                       |
| **Logic Analyzers** | Sigrok (.sr), VCD (Value Change Dump)         | [`loaders/`](src/oscura/loaders/)                       |
| **Network**         | PCAP, PCAPNG (Wireshark-compatible)           | [`loaders/`](src/oscura/loaders/)                       |
| **Automotive**      | Vector BLF/ASC, ASAM MDF/MF4, DBC, CSV        | [`automotive/loaders/`](src/oscura/automotive/loaders/) |
| **Scientific**      | TDMS (LabVIEW), HDF5, NumPy (.npz), WAV, CSV  | [`loaders/`](src/oscura/loaders/)                       |
| **RF/Network**      | Touchstone S-parameters (.s1p-.s8p)           | [`loaders/`](src/oscura/loaders/)                       |
| **Side-Channel**    | ChipWhisperer (.npy, .trs)                    | [`loaders/`](src/oscura/loaders/)                       |

**Memory-efficient:** Lazy loading, memory-mapped files, chunked processing for TB-scale datasets

### Automotive Engineering & Security

| Capability                       | Use Cases (Development / Reverse Engineering)                                                                                                                                           | Location                                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Evidence-Based Discovery**     | Protocol documentation / Hypothesis-driven CAN message field detection with statistical validation, confidence scoring, and evidence tracking for reproducible reverse engineering      | [`automotive/can/`](src/oscura/automotive/can/)     |
| **Stimulus-Response Mapping**    | Input correlation analysis / Correlate user actions (button press, sensor input, vehicle events) to CAN message changes for attack surface identification and control message discovery | [`automotive/can/`](src/oscura/automotive/can/)     |
| **CAN State Machine Extraction** | ECU behavior modeling / Extract protocol state machines from CAN traffic for security analysis and vulnerability discovery                                                              | [`automotive/can/`](src/oscura/automotive/can/)     |
| **Pattern Recognition**          | Signal validation / Auto-detect counters (rolling, wraparound), checksums (XOR, SUM, CRC), sensor encoding patterns in CAN messages                                                     | [`automotive/can/`](src/oscura/automotive/can/)     |
| **Differential Analysis**        | Signal validation / Compare CAN captures (idle vs accelerate, locked vs unlocked) to identify control messages and security-critical signals                                            | [`automotive/can/`](src/oscura/automotive/can/)     |
| **Discovery Persistence**        | Collaboration / Save/load RE progress in .tkcan format with hypothesis tracking for team collaboration and long-term projects                                                           | [`automotive/can/`](src/oscura/automotive/can/)     |
| **DBC Export**                   | Protocol documentation / Generate Vector DBC files from discoveries for integration with industry tools (CANalyzer, Wireshark, Vehicle Spy)                                             | [`automotive/dbc/`](src/oscura/automotive/dbc/)     |
| **J1939 Protocol**               | Heavy vehicle diagnostics / Complete J1939 decoder for truck, construction, agricultural equipment security research (PGN, SPN, suspect/FMI decoding)                                   | [`automotive/j1939/`](src/oscura/automotive/j1939/) |
| **OBD-II Diagnostics**           | Diagnostics development / Complete OBD-II Mode 01-09 support for PIDs, freeze frames, DTCs, VIN extraction                                                                              | [`automotive/obd/`](src/oscura/automotive/obd/)     |
| **UDS Protocol**                 | Security testing / Complete Unified Diagnostic Services decoder for ECU programming, security access, routine control, and diagnostic command fuzzing                                   | [`automotive/uds/`](src/oscura/automotive/uds/)     |

### IEEE-Compliant Measurements

| Standard               | Measurements                                       | Hardware Hacking Relevance                                 |
| ---------------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| **IEEE 181** (Pulse)   | Rise/fall time, pulse width, overshoot, duty cycle | Signal integrity validation for protocol analysis          |
| **IEEE 1241** (ADC)    | SNR, SINAD, THD, SFDR, ENOB                        | ADC characterization for side-channel analysis             |
| **IEEE 1459** (Power)  | Active/reactive power, harmonics, power factor     | Power supply analysis for fault injection targeting        |
| **IEEE 2414** (Jitter) | TIE, period jitter, RJ/DJ decomposition, BER       | Clock glitching attack detection, fault injection analysis |

**Location:** [`analyzers/waveform/`](src/oscura/analyzers/waveform/) | [`analyzers/spectral/`](src/oscura/analyzers/spectral/) | [`analyzers/power/`](src/oscura/analyzers/power/) | [`analyzers/jitter/`](src/oscura/analyzers/jitter/)

### Hardware Acquisition

| Source           | Description                                     | Status      | Location                                  |
| ---------------- | ----------------------------------------------- | ----------- | ----------------------------------------- |
| **File-Based**   | All supported file formats                      | ✅ Complete | [`loaders/`](src/oscura/loaders/)         |
| **SocketCAN**    | Linux CAN interfaces (vcan, physical CAN)       | ✅ Complete | [`acquisition/`](src/oscura/acquisition/) |
| **Saleae Logic** | Direct Logic analyzer streaming                 | ✅ Complete | [`acquisition/`](src/oscura/acquisition/) |
| **PyVISA**       | Oscilloscopes (Tektronix, Keysight, Rigol, R&S) | ✅ Complete | [`acquisition/`](src/oscura/acquisition/) |
| **Synthetic**    | Programmable signal generation for testing      | ✅ Complete | [`acquisition/`](src/oscura/acquisition/) |

### Export & Intelligence Sharing

| Format                    | Hardware Hacking Use Case                                                                                                                                                                                   | Location                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Wireshark Dissectors**  | Auto-generate validated Lua dissectors with ProtoField type mapping, variable-length field support, TCP/UDP port registration, and checksum validation for live traffic analysis and protocol documentation | [`export/wireshark/`](src/oscura/export/wireshark/) |
| **Binary Parsers**        | Auto-generate Python parsers and YAML specifications from inferred binary formats for firmware analysis, file format documentation, and automated data extraction                                           | [`inference/`](src/oscura/inference/)               |
| **DBC Files**             | Export CAN discoveries to Vector DBC format for integration with industry tools (CANalyzer, Vehicle Spy, Wireshark CAN plugins)                                                                             | [`automotive/dbc/`](src/oscura/automotive/dbc/)     |
| **Multi-Format Reports**  | Generate comprehensive evidence reports in PDF, HTML, PPTX, and Markdown with charts, tables, hypothesis tracking, confidence scoring, and full audit trails for reproducible research                      | [`reporting/`](src/oscura/reporting/)               |
| **Batch Processing**      | Generate comparison reports for entire capture directories with statistical analysis, correlation matrices, and trend visualization                                                                         | [`reporting/`](src/oscura/reporting/)               |
| **Discovery Archives**    | Save/load reverse engineering progress in structured .tkcan format (YAML-based) with hypothesis tracking, field candidates, and validation status for team collaboration                                    | [`automotive/can/`](src/oscura/automotive/can/)     |
| **WaveDrom Diagrams**     | Generate publication-quality timing diagrams from protocol specifications for documentation, academic papers, and technical communication                                                                   | [`export/`](src/oscura/export/)                     |
| **Vintage Logic Reports** | Specialized reports for retro computing with IC identification, timing validation, modern replacement recommendations, and restoration guidance                                                             | [`reporting/`](src/oscura/reporting/)               |

### Obsolete System Characterization & Replication

**For vintage computing restoration, industrial equipment repair, and hardware preservation (1960s-present)**

| Feature                         | Replication Use Case                                                                                                                                                                         | Location                                              |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Logic Family Auto-Detection** | Identify unknown vintage chips by voltage levels: ECL (-1.75V), RTL (3.6V), DTL (5V), PMOS (-12V), NMOS (+12V), TTL variants, HC-CMOS, 4000-series for functional cloning without datasheets | [`analyzers/digital/`](src/oscura/analyzers/digital/) |
| **IC Timing Database**          | Validate against 74xx series, 4000 series specifications (propagation delay, setup/hold times) for functional replication and modern FPGA/CPLD implementations                               | [`analyzers/digital/`](src/oscura/analyzers/digital/) |
| **IC Identification**           | Match measured timing characteristics to specific ICs (7400, 74LS138, 74HC595, etc.) for part identification when markings are illegible or missing                                          | [`analyzers/digital/`](src/oscura/analyzers/digital/) |
| **Modern Replacement Mapping**  | Recommend current-production substitutes (74HC → 74LS, CD4000 → 74HC) for restoration projects when original parts are unavailable                                                           | [`analyzers/digital/`](src/oscura/analyzers/digital/) |
| **Open-Collector Detection**    | Identify open-collector/open-drain outputs requiring pull-up resistors for accurate replication                                                                                              | [`analyzers/digital/`](src/oscura/analyzers/digital/) |
| **Protocol Preservation**       | Extract and document undocumented legacy protocols (parallel buses, proprietary serial) for data recovery and system emulation                                                               | [`inference/`](src/oscura/inference/)                 |

---

## Learn By Example

**Demos are the documentation.** Each category includes working code with validation and comprehensive explanations.

### Getting Started (Beginner)

| Demo                                                | What You'll Learn                                 |
| --------------------------------------------------- | ------------------------------------------------- |
| [01_waveform_analysis](demos/01_waveform_analysis/) | Load Tektronix/Rigol captures, basic measurements |
| [02_file_format_io](demos/02_file_format_io/)       | CSV, HDF5, NumPy, custom binary formats           |
| [04_serial_protocols](demos/04_serial_protocols/)   | UART, SPI, I2C decoding with auto-detection       |
| [05_protocol_decoding](demos/05_protocol_decoding/) | Protocol auto-detection pipeline                  |

### Reverse Engineering (Intermediate)

| Demo                                                                  | What You'll Learn                                     |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| [07_protocol_inference](demos/07_protocol_inference/)                 | State machine learning, CRC recovery, field detection |
| [08_automotive_protocols](demos/08_automotive_protocols/)             | CAN, CAN-FD, LIN, FlexRay analysis                    |
| [17_signal_reverse_engineering](demos/17_signal_reverse_engineering/) | Complete unknown signal analysis workflow             |
| [18_advanced_inference](demos/18_advanced_inference/)                 | Bayesian inference, binary format DSL                 |

### Security & Compliance (Advanced)

| Demo                                                    | What You'll Learn                                  |
| ------------------------------------------------------- | -------------------------------------------------- |
| [12_spectral_compliance](demos/12_spectral_compliance/) | FFT, THD, SNR, SINAD (IEEE 1241)                   |
| [13_jitter_analysis](demos/13_jitter_analysis/)         | TIE, RJ/DJ decomposition, eye diagrams (IEEE 2414) |
| [14_power_analysis](demos/14_power_analysis/)           | DC-DC converter analysis, efficiency (IEEE 1459)   |
| [15_signal_integrity](demos/15_signal_integrity/)       | TDR, S-parameters, setup/hold timing               |
| [16_emc_compliance](demos/16_emc_compliance/)           | CISPR, FCC, MIL-STD electromagnetic compatibility  |

### Complete Workflows (Expert)

| Demo                                                  | What You'll Learn                        |
| ----------------------------------------------------- | ---------------------------------------- |
| [19_complete_workflows](demos/19_complete_workflows/) | End-to-end reverse engineering pipelines |

### Run Your First Demo

```bash
# Generate test data
python demos/generate_all_demo_data.py

# Run waveform analysis demo
python demos/01_waveform_analysis/comprehensive_wfm_analysis.py

# Run protocol inference demo
python demos/07_protocol_inference/01_state_machine_inference.py
```

**Browse all demos:** [demos/](demos/) | [Demo index with descriptions](demos/README.md)

---

## Command Line Interface

```bash
# Signal characterization
oscura characterize capture.wfm

# Protocol decoding
oscura decode uart_capture.wfm --protocol uart --baud 115200

# Batch processing
oscura batch '*.wfm' --analysis characterize

# Differential analysis
oscura compare baseline.wfm modified.wfm

# Interactive REPL
oscura shell

# Generate synthetic test signals
oscura generate --protocol spi --frequency 1MHz --output test.bin
```

**Full reference:** [docs/cli.md](docs/cli.md)

---

## Quality & Testing

Comprehensive test suite with property-based testing, stress tests, and CI enforcement. Current metrics visible in [CI dashboard](https://github.com/oscura-re/oscura/actions) and [code coverage reports](https://codecov.io/gh/oscura-re/oscura).

```bash
# Run test suite (uses pytest in parallel with coverage)
./scripts/test.sh

# Quality checks (linting, type checking, formatting)
./scripts/check.sh

# Auto-fix issues
./scripts/fix.sh

# Full CI validation (pre-push)
./scripts/pre-push.sh
```

**Quality infrastructure:**

- Pre-commit hooks (format, lint, type check)
- Merge queue CI (prevents untested commits)
- Property-based testing (Hypothesis)
- Nightly stress tests
- Security scanning (Bandit, Safety)

**See:** [Testing architecture](docs/testing/) | [Quality gates](.github/workflows/)

---

## Contributing

**We need your expertise.** Whether you're adding a new protocol decoder, improving inference algorithms, or documenting use cases - every contribution illuminates more hardware.

### Quick Start

```bash
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh                    # Complete setup with hooks
./scripts/test.sh                     # Verify environment
```

### What We Need

- **Protocol Decoders:** Proprietary protocols you've reverse engineered
- **File Format Loaders:** Oscilloscope/LA formats we don't support yet
- **Inference Algorithms:** Better state machine learning, field detection
- **Hardware Sources:** Integration with more DAQ systems
- **Documentation:** Real-world case studies and tutorials
- **Validation:** Test our tools on your captures

**Full guide:** [CONTRIBUTING.md](CONTRIBUTING.md) | [Architecture docs](docs/architecture/)

---

## Documentation

### User Guides

- [Quick Start Guide](docs/guides/quick-start.md) - Installation and first steps
- [Black-Box Protocol Analysis](docs/guides/blackbox-analysis.md) - Unknown protocol RE
- [Side-Channel Analysis](docs/guides/side-channel-analysis.md) - DPA/CPA/timing attacks
- [Hardware Acquisition](docs/guides/hardware-acquisition.md) - Direct instrument control
- [Complete Workflows](docs/guides/workflows.md) - End-to-end analysis pipelines

### API Reference

- [API Documentation](docs/api/) - Complete API reference
- [Session Management](docs/api/session-management.md) - Interactive analysis sessions
- [CLI Reference](docs/cli.md) - Command-line interface

### Development

- [Architecture](docs/architecture/) - Design principles and patterns
- [Testing Guide](docs/testing/) - Test suite architecture
- [CHANGELOG](CHANGELOG.md) - Version history and migration guides

---

## Project Status

**Latest Release:** [See current version and release notes](https://github.com/oscura-re/oscura/releases/latest)

**Active Development Areas:**

- Side-channel analysis and cryptographic attack frameworks
- Vintage computing support (retro logic families, IC identification)
- Industrial and automotive protocol decoders
- Hardware acquisition from diverse sources
- Inference and machine learning for unknown protocols

**See:** [Full changelog](CHANGELOG.md) | [Roadmap discussions](https://github.com/oscura-re/oscura/discussions)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/oscura-re/oscura/issues) - Bug reports and feature requests
- **Discussions:** [GitHub Discussions](https://github.com/oscura-re/oscura/discussions) - Questions and community
- **Security:** [SECURITY.md](SECURITY.md) - Responsible disclosure

---

## Citation

If Oscura helps your research, please cite:

```bibtex
@software{oscura2026,
  title = {Oscura: Hardware Reverse Engineering Framework},
  author = {Oscura Contributors},
  year = {2026},
  url = {https://github.com/oscura-re/oscura},
  version = {0.5.0}
}
```

**Machine-readable:** [CITATION.cff](CITATION.cff)

---

## License

MIT License - [LICENSE](LICENSE)

**Built with:** Python, NumPy, SciPy, Hypothesis

**Standards:** IEEE 181/1241/1459/2414, ISO/IEC, CISPR

**Supported by:** Security researchers, right-to-repair advocates, open source community

---

**Oscura** - _Illuminate what others obscure_
