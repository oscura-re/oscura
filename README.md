# Oscura

**The missing link in hardware reverse engineering.** Binary analysis has Ghidra, radare2, and IDA—unified frameworks for dissecting compiled code. But what about the critical steps before you get a binary off a chip? Oscura provides the comprehensive toolkit for analyzing signals, decoding protocols, and extracting intelligence from hardware systems.

[![CI](https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/ci.yml)
[![Code Quality](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/oscura-re/oscura/graph/badge.svg)](https://codecov.io/gh/oscura-re/oscura)
[![PyPI version](https://img.shields.io/pypi/v/oscura)](https://pypi.org/project/oscura/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Hardware reverse engineering is fragmented. You capture signals with an oscilloscope, decode protocols with custom scripts, analyze waveforms in MATLAB, infer message formats manually, and stitch everything together with duct tape and determination. Each tool solves one piece of the puzzle, but none connect them into a unified workflow.

Meanwhile, binary reverse engineering matured decades ago with integrated platforms that handle disassembly, decompilation, debugging, and analysis in one place. Hardware deserves the same.

## The Solution

Oscura unifies the hardware reverse engineering pipeline from raw signal capture to protocol documentation:

- **Load** from any source: oscilloscopes (Tektronix, Rigol, LeCroy), logic analyzers (Sigrok, Saleae), network captures (PCAP), automotive logs (BLF, MDF4), side-channel traces (ChipWhisperer)
- **Analyze** with IEEE-compliant measurements, spectral analysis, jitter characterization, power profiling, signal integrity validation
- **Decode** 16+ protocols automatically or infer unknown protocols through statistical analysis, state machine learning, and CRC recovery
- **Extract** intelligence through differential analysis, hypothesis testing, pattern recognition, and side-channel cryptanalysis
- **Export** to industry formats: Wireshark dissectors, DBC files, binary parsers, comprehensive reports

All in Python. All tested. All ready for serious work.

---

## Quick Start

### Installation

```bash
# Production use
pip install oscura

# Development (recommended - includes all features)
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh
```

**Requirements:** Python 3.12+ | [Dependencies](pyproject.toml)

### Five-Minute Examples

**Decode an unknown protocol from oscilloscope capture:**

```python
import oscura as osc

# Load Tektronix/Rigol/LeCroy waveform
trace = osc.load("mystery_device.wfm")

# Auto-detect protocol (UART, SPI, I2C, CAN, etc.)
decoder = osc.auto_detect_protocol(trace)
messages = decoder.decode(trace)
print(f"Decoded {len(messages)} {decoder.name} messages")
```

**Extract AES key via power analysis:**

```python
from oscura.loaders import load_chipwhisperer
from oscura.analyzers.side_channel import CPAAnalyzer

# Load power traces
traces = load_chipwhisperer("aes_traces.npy")

# Correlation Power Analysis
cpa = CPAAnalyzer(leakage_model="hamming_weight", target_byte=0)
result = cpa.analyze(traces.traces, traces.plaintexts)

print(f"Key byte: 0x{result.key_guess:02X} (correlation: {result.max_correlation:.4f})")
```

**Reverse engineer a proprietary protocol:**

```python
from oscura.sessions import BlackBoxSession

# Create analysis session with hypothesis tracking
session = BlackBoxSession(name="IoT Device RE")

# Differential analysis: idle vs active states
session.add_recording("idle", "idle.bin")
session.add_recording("button_press", "button.bin")
diff = session.compare("idle", "button_press")

# Automatic field detection with confidence scoring
spec = session.generate_protocol_spec()
print(f"Identified {len(spec['fields'])} protocol fields")

# Export validated Wireshark dissector
session.export_results("dissector", "protocol.lua")
```

**Analyze automotive CAN traffic:**

```python
from oscura.automotive.can import CANSession

session = CANSession(name="Vehicle RE")
session.add_recording("idle", "idle.blf")
session.add_recording("accelerate", "accel.blf")

# Identify control messages
diff = session.compare("idle", "accelerate")
print(f"Changed CAN IDs: {diff.details['changed_ids']}")

# Export to industry tools (CANalyzer, Vehicle Spy)
session.export_dbc("vehicle.dbc")
```

**Recover CRC specification from unknown protocol:**

```python
from oscura.inference.crc_reverse import CRCReverser

# Just 4 message-checksum pairs needed
messages = [b"\x01\x02\x03", b"\x04\x05\x06", b"\x07\x08\x09", b"\x0a\x0b\x0c"]
checksums = [0x12, 0x34, 0x56, 0x78]

# Recover complete CRC specification
reverser = CRCReverser(message_bits=8)
crc = reverser.find_crc(list(zip(messages, checksums)))

print(f"Polynomial: 0x{crc.polynomial:02X}")
print(f"Init: 0x{crc.init_value:02X}, XOR out: 0x{crc.xor_out:02X}")
print(f"Standard: {crc.standard_name or 'Custom'}")  # Matches CRC-8, CRC-16, etc.
```

[**112 working demonstrations**](demonstrations/) across 19 categories show every capability in depth.

---

## Core Capabilities

### What This Framework Does

| Capability | What You Get | Use Cases |
|------------|--------------|-----------|
| **Protocol Decoding** | 16 decoders (UART, SPI, I2C, CAN, LIN, FlexRay, JTAG, SWD, USB, I2S, 1-Wire, Manchester, HDLC, CAN-FD, J1939, IEEE-488) with auto-detection | Debug console access, firmware extraction, bus monitoring |
| **Unknown Protocol RE** | CRC recovery, message format inference, state machine extraction, field boundary detection | Proprietary protocols, vendor lock-in bypass, legacy systems |
| **Side-Channel Analysis** | DPA/CPA power analysis, timing attacks, mutual information leakage quantification | Cryptographic key extraction, vulnerability assessment, secure implementation validation |
| **IEEE Measurements** | Standards-compliant (181/1241/1459/2414): pulse timing, ADC characterization, power quality, jitter decomposition | Signal integrity validation, compliance testing, component characterization |
| **File Format Support** | 13+ formats: oscilloscopes (Tektronix WFM, Rigol, LeCroy TRC), logic analyzers (Sigrok, VCD), automotive (BLF, MDF4, DBC), network (PCAP), scientific (HDF5, TDMS, WAV) | Universal signal import, no vendor lock-in |
| **Automotive Security** | CAN/LIN/FlexRay analysis, OBD-II/UDS decoding, hypothesis-driven field discovery, stimulus-response mapping | ECU security research, aftermarket development, diagnostics reverse engineering |
| **Intelligence Sharing** | Auto-generate Wireshark dissectors (validated Lua), DBC files, binary parsers, PDF/HTML/PPTX reports | Collaboration, documentation, tool integration |
| **Hardware Acquisition** | Direct control: oscilloscopes (PyVISA), logic analyzers (Saleae), CAN interfaces (SocketCAN), synthetic signal generation | Live capture, automated testing, comprehensive validation |

### Where It Shines

**Security Research:**

- Extract cryptographic keys through power/EM side channels
- Identify authentication bypass vulnerabilities via state machine analysis
- Map attack surfaces through differential signal analysis
- Validate constant-time implementations with timing attack detection

**Right-to-Repair & Modernization:**

- Document undocumented protocols for replacement parts
- Replicate vintage hardware (1960s-present logic families: ECL, RTL, DTL, TTL, CMOS)
- Overcome vendor lock-in through protocol reverse engineering
- Generate interoperable interfaces without vendor cooperation

**Academic Research:**

- Property-based testing with Hypothesis for algorithm validation
- Full reproducibility through evidence tracking and audit trails
- IEEE/ISO standards compliance for publishable results
- 302 unit tests and 80%+ code coverage ensure reliability

**Industrial & Automotive:**

- CAN bus security research and aftermarket development
- Signal integrity validation for high-speed designs
- EMC compliance testing (CISPR, FCC, MIL-STD)
- Component characterization without datasheets

---

## Technical Foundation

### Standards Compliance

We implement specifications correctly, not approximately:

| Standard | Coverage | Hardware RE Relevance |
|----------|----------|-----------------------|
| **IEEE 181** | Pulse timing, rise/fall, overshoot, duty cycle | Protocol physical layer validation, signal integrity |
| **IEEE 1241** | SNR, SINAD, THD, SFDR, ENOB | ADC characterization for side-channel analysis |
| **IEEE 1459** | Active/reactive power, harmonics, power factor | Power supply profiling, fault injection targeting |
| **IEEE 2414** | TIE, period jitter, RJ/DJ decomposition, BER | Clock glitch detection, timing attack analysis |

### Quality Metrics

Production-ready means tested rigorously:

- **302 unit tests** with property-based validation (Hypothesis)
- **80%+ code coverage** with branch coverage enabled
- **Pre-commit hooks** (format, lint, type check) enforce consistency
- **Merge queue CI** prevents untested code from landing
- **Nightly stress tests** validate edge cases and memory usage
- **Security scanning** (Bandit, Safety) on every commit

View current metrics: [CI Dashboard](https://github.com/oscura-re/oscura/actions) | [Coverage Reports](https://codecov.io/gh/oscura-re/oscura)

### Architecture Principles

Built for extensibility and maintainability:

- **Type-safe**: MyPy strict mode, comprehensive type hints
- **Modular**: Protocol decoders, loaders, and analyzers are plug-and-play
- **Memory-efficient**: Lazy loading, memory-mapped files, chunked processing (TB-scale datasets)
- **Documented**: Google-style docstrings, 95% documentation coverage
- **Reproducible**: Hypothesis tracking, confidence scoring, full audit trails

---

## Learn By Doing

### 112 Demonstrations Across 19 Categories

Every demo includes working code, validation, and comprehensive explanations. Demos are the documentation.

**Getting Started (Beginner):**

- [Waveform Analysis](demos/01_waveform_analysis/) - Load oscilloscope captures, basic measurements
- [File Format I/O](demos/02_file_format_io/) - CSV, HDF5, NumPy, VCD, custom formats
- [Serial Protocols](demos/04_serial_protocols/) - UART, SPI, I2C with auto-detection
- [Protocol Decoding](demos/05_protocol_decoding/) - Auto-detect unknown protocols

**Reverse Engineering (Intermediate):**

- [Protocol Inference](demos/07_protocol_inference/) - State machines, CRC recovery, field detection
- [Automotive Protocols](demos/08_automotive_protocols/) - CAN, LIN, FlexRay analysis
- [Signal RE](demos/17_signal_reverse_engineering/) - Complete unknown signal workflow
- [Advanced Inference](demos/18_advanced_inference/) - Bayesian methods, protocol DSL

**Security & Compliance (Advanced):**

- [Spectral Compliance](demos/12_spectral_compliance/) - FFT, THD, SNR, SINAD (IEEE 1241)
- [Jitter Analysis](demos/13_jitter_analysis/) - TIE, RJ/DJ decomposition (IEEE 2414)
- [Power Analysis](demos/14_power_analysis/) - DC-DC converters, efficiency (IEEE 1459)
- [Signal Integrity](demos/15_signal_integrity/) - TDR, S-parameters, setup/hold timing
- [EMC Compliance](demos/16_emc_compliance/) - CISPR, FCC, MIL-STD testing

**Complete Workflows (Expert):**

- [End-to-End Pipelines](demos/19_complete_workflows/) - Unknown signal → documented protocol

### Run Your First Demo

```bash
# Generate synthetic test data
python demos/generate_all_demo_data.py

# Analyze oscilloscope waveforms
python demos/01_waveform_analysis/comprehensive_wfm_analysis.py

# Reverse engineer unknown protocol
python demos/07_protocol_inference/state_machine_learning.py

# Side-channel power analysis
python demos/14_power_analysis/dcdc_efficiency_demo.py
```

[**Browse all demos**](demos/) | [Demo index with descriptions](demos/README.md)

---

## Command-Line Interface

```bash
# Signal characterization
oscura characterize capture.wfm

# Protocol decoding with auto-detection
oscura decode uart_capture.wfm --protocol uart --baud 115200

# Batch processing entire directories
oscura batch '*.wfm' --analysis characterize

# Differential analysis (compare baseline to modified)
oscura compare baseline.wfm modified.wfm

# Interactive REPL for exploration
oscura shell

# Generate synthetic test signals
oscura generate --protocol spi --frequency 1MHz --output test.bin
```

[**Full CLI reference**](docs/cli.md)

---

## Why This Exists

### Legitimate Use Cases

Hardware reverse engineering serves critical needs across security, repair, modernization, and defense:

**Security Research:** Vulnerability discovery requires understanding how hardware actually works, not how vendors claim it works. Side-channel analysis exposes cryptographic weaknesses. Protocol reverse engineering reveals authentication bypasses.

**Right-to-Repair:** Proprietary protocols and vendor lock-in prevent owners from fixing their own equipment. Reverse engineering restores agency. Open documentation enables interoperable replacements.

**Modernization:** Legacy systems run critical infrastructure but use obsolete components. Replication requires extracting specifications from working hardware when documentation is lost or was never public.

**National Defense:** Intelligence and threat assessment depend on understanding adversary capabilities. Forensic analysis of captured equipment requires comprehensive signal analysis and protocol decoding.

**Academic Research:** Understanding existing systems informs better designs. Teaching security requires demonstrating real vulnerabilities. Open tools advance the field collectively.

### The Open Source Philosophy

We believe security through obscurity is a temporary business model at best and a vulnerability at worst. Real security comes from open scrutiny, not information hiding. Real value comes from services and expertise, not gatekeeping knowledge.

Vendors who hide protocol specifications aren't protecting trade secrets—they're preventing interoperability and limiting repair. We're building tools to level that playing field.

### Join the Effort

Hardware reverse engineering requires diverse expertise: signal processing, cryptanalysis, protocol design, automotive systems, vintage computing, embedded security. No single person knows it all. **We need your knowledge.**

- Reverse engineered a proprietary protocol? Contribute the decoder.
- Built side-channel analysis techniques? Add them to the framework.
- Work with file formats we don't support? Write a loader.
- Found vulnerabilities using these tools? Share sanitized case studies.
- Teaching hardware security? Use Oscura and improve the documentation.

Every contribution pools our collective expertise and makes the next reverse engineering project easier for everyone.

---

## Getting Involved

### Contributing

```bash
# Clone and setup development environment
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh                    # Complete setup with hooks

# Run quality checks (required before commit)
./scripts/check.sh                    # Linting, type checking, tests
./scripts/test.sh                     # Full test suite with coverage

# Validate everything passes
python3 .claude/hooks/validate_all.py # Must show 5/5 passing
```

**What We Need:**

| Contribution Type | Examples | Impact |
|-------------------|----------|--------|
| **Protocol Decoders** | Proprietary protocols you've reversed | Enable others to analyze same systems |
| **File Format Loaders** | Oscilloscope/LA formats not yet supported | Eliminate conversion steps |
| **Inference Algorithms** | Better state machine learning, CRC detection | Improve automatic analysis quality |
| **Hardware Integration** | DAQ systems, instrument drivers | Enable live capture workflows |
| **Real-World Validation** | Test on your captures, report issues | Ensure reliability across use cases |
| **Documentation** | Tutorials, case studies, guides | Lower entry barrier for newcomers |

[**Contributing Guide**](CONTRIBUTING.md) | [Architecture Documentation](docs/architecture/)

### Community

- **Issues:** [GitHub Issues](https://github.com/oscura-re/oscura/issues) - Bug reports, feature requests
- **Discussions:** [GitHub Discussions](https://github.com/oscura-re/oscura/discussions) - Questions, ideas, collaboration
- **Security:** [SECURITY.md](SECURITY.md) - Responsible disclosure process

---

## Documentation

### User Guides

- [Quick Start Guide](docs/guides/quick-start.md) - Installation and first steps
- [Black-Box Protocol Analysis](docs/guides/blackbox-analysis.md) - Unknown protocol RE workflow
- [Side-Channel Analysis](docs/guides/side-channel-analysis.md) - DPA/CPA/timing attacks
- [Hardware Acquisition](docs/guides/hardware-acquisition.md) - Direct instrument control
- [Complete Workflows](docs/guides/workflows.md) - End-to-end pipelines

### API Reference

- [API Documentation](docs/api/) - Complete function reference
- [Session Management](docs/api/session-management.md) - Interactive analysis sessions
- [CLI Reference](docs/cli.md) - Command-line interface

### Development

- [Architecture](docs/architecture/) - Design principles and patterns
- [Testing Guide](docs/testing/) - Test suite architecture
- [CHANGELOG](CHANGELOG.md) - Version history and migration guides

---

## Project Status

**Current Version:** [0.5.1](https://github.com/oscura-re/oscura/releases/latest) (2026-01-24)

**Active Development Areas:**

- Side-channel cryptanalysis frameworks (DPA, CPA, timing, EM)
- Vintage computing support (retro logic families, IC identification, 1960s-present)
- Industrial and automotive protocol decoders (CAN-FD, J1939, OBD-II, UDS)
- Unknown protocol inference (state machines, field detection, CRC recovery)
- Hardware acquisition from diverse instruments and interfaces

**Stability:** Production-ready for security research, right-to-repair, academic use. APIs may evolve as we add capabilities—breaking changes documented in [CHANGELOG](CHANGELOG.md).

[**Release History**](https://github.com/oscura-re/oscura/releases) | [**Roadmap Discussions**](https://github.com/oscura-re/oscura/discussions)

---

## Citation

If Oscura contributes to your research, please cite:

```bibtex
@software{oscura2026,
  title = {Oscura: Hardware Reverse Engineering Framework},
  author = {Oscura Contributors},
  year = {2026},
  url = {https://github.com/oscura-re/oscura},
  version = {0.5.1}
}
```

**Machine-readable:** [CITATION.cff](CITATION.cff)

---

## Legal

**License:** [MIT License](LICENSE) - Permissive use, modification, distribution

**Disclaimer:** This framework is intended for legitimate security research, right-to-repair, academic study, and authorized testing. Users are responsible for compliance with applicable laws and regulations. Unauthorized access to systems or networks is illegal and unethical.

**Dependencies:** Built with Python, NumPy, SciPy, Matplotlib, Hypothesis. See [pyproject.toml](pyproject.toml) for complete dependency list.

**Supported by:** Security researchers, right-to-repair advocates, academic institutions, and the open source community.

---

**Oscura** - _Illuminate what others obscure._

Hardware systems are black boxes by design, obscured through proprietary protocols, cryptographic obfuscation, and undocumented interfaces. Whether imposed by vendors, governments, or the passage of time—**we bring light to the darkness.** Join us in building the comprehensive hardware reverse engineering framework the field deserves.
