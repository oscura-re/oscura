# Oscura

**Workflow automation for hardware reverse engineering.** Stop juggling seven different tools to analyze one capture. Oscura chains specialized tools (sigrok, ChipWhisperer, scipy) into unified Python workflows—from oscilloscope files to Wireshark dissectors without manual conversions or context switching.

[![CI](https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/ci.yml)
[![Code Quality](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/oscura-re/oscura/graph/badge.svg)](https://codecov.io/gh/oscura-re/oscura)
[![PyPI version](https://img.shields.io/pypi/v/oscura)](https://pypi.org/project/oscura/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Hardware reverse engineering means juggling specialized tools:

1. Export oscilloscope waveforms (vendor-specific formats)
2. Convert formats for analysis (sigrok, custom scripts)
3. Decode protocols (PulseView, separate decoders)
4. Infer unknown protocols (Netzob, manual analysis)
5. Reverse checksums (CRC RevEng, separate tool)
6. Generate documentation (manual Wireshark dissectors, DBC files)
7. Repeat for each new capture

**Each step requires different tools, manual file conversions, and context switching.** Binary reverse engineering solved this decades ago with integrated platforms (Ghidra, radare2, IDA). Hardware RE remains fragmented.

## The Solution

Oscura automates complete workflows in Python:

**What We Integrate:**

- Protocol decoding via [sigrok](https://sigrok.org/) (UART, SPI, I2C, CAN, etc.)
- Signal processing with scipy/numpy
- Side-channel trace formats (ChipWhisperer)
- Automotive protocols (cantools integration)

**What We Add:**

- **Hypothesis-driven RE workflows** with differential analysis and confidence scoring
- **Automatic Wireshark dissector generation** from inferred protocols
- **DBC file generation** from raw CAN captures (no manual signal definition)
- **Multi-format file loading** (Tektronix, Rigol, Sigrok, BLF, PCAP, ChipWhisperer)
- **CRC/checksum recovery** from message-checksum pairs
- **Unified Python API** eliminating tool-hopping and format conversions

**Value proposition:** Write one Python script instead of:

1. Exporting from oscilloscope software (vendor GUI)
2. Converting formats (sigrok-cli, custom scripts)
3. Decoding protocols (PulseView manual selection)
4. Inferring message formats (Netzob or manual)
5. Recovering checksums (CRC RevEng separate invocation)
6. Writing dissectors (manual Lua coding)
7. Documenting findings (manual reports)

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

### Workflow Examples

**Reverse engineer unknown protocol (differential analysis):**

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

# Export validated Wireshark dissector (Lua)
session.export_results("dissector", "protocol.lua")
```

**Generate automotive DBC from raw CAN captures:**

```python
from oscura.automotive.can import CANSession

session = CANSession(name="Vehicle RE")
session.add_recording("idle", "idle.blf")
session.add_recording("accelerate", "accel.blf")

# Statistical stimulus-response analysis
diff = session.compare("idle", "accelerate")
print(f"Changed CAN IDs: {diff.details['changed_ids']}")

# Generate DBC file (signal definitions inferred automatically)
session.export_dbc("vehicle.dbc")  # Import into CANalyzer, Vehicle Spy, Wireshark
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

**Auto-detect protocol from oscilloscope capture:**

```python
import oscura as osc

# Load Tektronix/Rigol waveform
trace = osc.load("mystery_device.wfm")

# Statistical protocol detection (timing, voltage levels, bit patterns)
result = osc.auto_decode(trace)
print(f"Detected {result.protocol}: {len(result.frames)} frames decoded")
```

**[See demos/README.md](demos/README.md)** for 105+ comprehensive demonstrations organized by skill level.

---

## Core Capabilities

### Unique Contributions

| Capability                            | What We Provide                                                                                   | Why It Matters                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Hypothesis-Driven RE**              | BlackBoxSession with differential analysis, field detection, confidence scoring, audit trails     | Systematic unknown protocol analysis vs manual guesswork                                 |
| **DBC Auto-Generation**               | Statistical CAN signal inference from captures → DBC export                                       | Open-source alternative to Vector CANalyzer ($$$)                                        |
| **Wireshark Dissector Generation**    | Infer protocol → generate validated Lua dissector                                                 | End-to-end automation (others require manual YAML specs)                                 |
| **Multi-Format File Loading**         | Oscilloscopes (Tektronix WFM, Rigol), logic analyzers (Sigrok, VCD), automotive BLF  | Eliminate format conversion steps                                                        |
| **Statistical Protocol Auto-Detect**  | Waveform analysis (timing, voltage, patterns) → protocol identification                           | Goes beyond sigrok's signal name matching                                                |
| **Unified Workflow API**              | Single Python script: oscilloscope file → decode → infer → export dissector                       | Replace 7-tool chains with one script                                                    |
| **CRC Recovery**                      | Message-checksum pairs → polynomial, init, XOR out, reflection                                    | Practical automation (CRC RevEng is more robust for edge cases)                          |
| **Automotive Security Analysis**      | Stimulus-response correlation, hypothesis testing, UDS/OBD-II decoding                            | Research-focused (CANToolz covers security, python-can covers low-level)                 |
| **State Machine Extraction (Passive)** | RPNI algorithm for passive observation (vs Netzob's active L\* requiring oracle)                  | Different use case from existing tools                                                   |
| **Evidence-Based Discovery**          | Confidence scoring, hypothesis tracking, statistical validation, reproducible audit trails        | Scientific rigor for research publication                                                |

### Integration Capabilities

| Category                   | Implementation                                                                          | Best Alternative                                                    |
| -------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Protocol Decoding**      | Integrated sigrok decoders (UART, SPI, I2C, CAN, LIN, JTAG, etc.) via Python API       | [sigrok](https://sigrok.org/) directly (100+ decoders)              |
| **Side-Channel Analysis**  | Load ChipWhisperer traces, basic DPA/CPA implementations                               | [ChipWhisperer](https://chipwhisperer.com/) (superior capabilities) |
| **Signal Processing**      | IEEE-based measurements using scipy/numpy                                               | [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) directly or MATLAB                        |
| **CAN Parsing**            | cantools integration for DBC parsing and message encoding                               | [cantools](https://github.com/cantools/cantools) + [python-can](https://python-can.readthedocs.io/)     |
| **File Format Conversion** | Loaders for 13+ formats with unified API                                                | Vendor software + manual export                                     |

**Our philosophy:** Integrate best-in-class tools rather than reimplementing them. Add value through workflow automation and novel analysis methods.

---

## When to Use Oscura

**Choose Oscura when:**

- You need end-to-end workflows (capture → analysis → documentation) in Python
- You're reverse engineering unknown protocols with differential analysis
- You want DBC files generated from CAN captures without CANalyzer ($$$)
- You need Wireshark dissectors generated automatically from inferred protocols
- You're working with multiple oscilloscope/LA formats and want unified API
- You value reproducible research with hypothesis tracking and confidence scoring

**Use specialized tools directly when:**

- You only need protocol decoding → [sigrok](https://sigrok.org/) has 100+ decoders
- You're doing side-channel attacks → [ChipWhisperer](https://chipwhisperer.com/) is superior
- You only need signal processing → [scipy](https://scipy.org/)/MATLAB are more optimized
- You need the most robust CRC recovery → [CRC RevEng](https://reveng.sourceforge.io/) handles edge cases better
- You have vendor-specific needs → vendor tools have more format support

**Oscura's sweet spot:** Chaining multiple RE steps in scripted workflows with novel hypothesis-driven analysis.

---

## Where This Excels

### Security Research

- **Protocol reverse engineering** with hypothesis tracking and validation
- **Automotive ECU security** via CAN stimulus-response analysis
- **Attack surface mapping** through state machine extraction
- **Cryptographic implementation validation** (use ChipWhisperer for attacks, Oscura for trace analysis workflows)

### Right-to-Repair & Modernization

- **Document undocumented protocols** with generated Wireshark dissectors
- **Replicate vintage hardware** (1960s-present logic family auto-detection)
- **Overcome vendor lock-in** through protocol reverse engineering
- **Generate interoperable interfaces** without vendor cooperation

### Academic Research

- **Reproducible workflows** with evidence tracking and audit trails
- **Statistical validation** with confidence scoring
- **IEEE-based measurements** for publishable results (181/1241/1459/2414)
- **22,000+ comprehensive tests, 80%+ coverage** ensure reliability

### Industrial & Automotive

- **CAN bus security research** with open-source DBC generation
- **Signal integrity validation** for high-speed designs
- **Component characterization** without datasheets
- **Compliance testing** (EMC, automotive standards)

---

## Built On

Oscura integrates proven open-source tools:

| Component           | What We Use                                                             | Why                                                     |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| **Protocol Engine** | [sigrok](https://sigrok.org/) libsigrokdecode                           | 100+ mature, community-supported protocol decoders      |
| **Signal Processing** | [scipy](https://scipy.org/)/[numpy](https://numpy.org/)                 | Industry-standard numerical computing                   |
| **Side-Channel Traces** | [ChipWhisperer](https://chipwhisperer.com/) formats                     | De facto standard for side-channel research             |
| **CAN Protocols**   | [cantools](https://github.com/cantools/cantools), [python-can](https://python-can.readthedocs.io/) | Robust CAN message parsing and encoding                |
| **Testing**         | [pytest](https://pytest.org/), [Hypothesis](https://hypothesis.readthedocs.io/)         | Property-based testing for algorithm validation         |
| **Type Safety**     | [mypy](https://mypy-lang.org/)                                          | Static type checking (strict mode)                      |

**Our contribution:** Unified API + novel hypothesis-driven RE workflows + format handling + export automation.

---

## Technical Foundation

### Quality Metrics

Production-ready validation:

- **22,000+ comprehensive tests** with property-based validation (Hypothesis)
- **80%+ code coverage** with branch coverage enabled
- **Pre-commit hooks** (format, lint, type check) enforce consistency
- **Merge queue CI** prevents untested code from landing
- **Nightly stress tests** validate edge cases and memory usage
- **Security scanning** (Bandit, Safety) on every commit

View current metrics: [CI Dashboard](https://github.com/oscura-re/oscura/actions) | [Coverage Reports](https://codecov.io/gh/oscura-re/oscura)

### Standards Implementation

We implement measurements based on IEEE specifications:

| Standard        | Coverage                                       | Hardware RE Relevance                           |
| --------------- | ---------------------------------------------- | ----------------------------------------------- |
| **IEEE 181**    | Pulse timing, rise/fall, overshoot, duty cycle | Protocol physical layer validation              |
| **IEEE 1241**   | SNR, SINAD, THD, SFDR, ENOB                    | ADC characterization for side-channel analysis  |
| **IEEE 1459**   | Active/reactive power, harmonics, power factor | Power supply profiling, fault injection targets |
| **IEEE 2414**   | TIE, period jitter, RJ/DJ decomposition, BER   | Clock glitch detection, timing attack analysis  |

### Architecture Principles

Built for extensibility:

- **Type-safe**: MyPy strict mode, comprehensive type hints
- **Modular**: Protocol decoders, loaders, and analyzers are plug-and-play
- **Memory-efficient**: Lazy loading, memory-mapped files, chunked processing (TB-scale datasets)
- **Documented**: Google-style docstrings, 95% documentation coverage
- **Reproducible**: Hypothesis tracking, confidence scoring, full audit trails

---

## Learn By Doing

### Working Demonstrations

**105+ comprehensive demos** organized into 12 categories covering:

- **Data Loading** - All file format loaders (oscilloscopes, logic analyzers, automotive, scientific)
- **Basic Analysis** - Waveform measurements, digital analysis, spectral analysis, filtering
- **Protocol Decoding** - UART, SPI, I2C, CAN, LIN, FlexRay, JTAG, SWD, I2S, USB
- **Advanced Analysis** - Jitter, eye diagrams, power analysis, signal integrity, TDR
- **Domain Specific** - Automotive diagnostics, EMC compliance, side-channel analysis, IEEE 181 timing
- **Reverse Engineering** - CRC recovery, state machines, Wireshark dissectors, ML classification
- **Advanced Features** - Lazy loading, memory management, performance optimization, batch processing
- **Extensibility** - Custom analyzers, plugins, templates
- **Integration** - CI/CD, hardware, external tools, web dashboards
- **Export & Visualization** - All export formats, plotting, reporting
- **Complete Workflows** - End-to-end production pipelines
- **Standards Compliance** - IEEE 181/1241/1459/2414, automotive standards

### Comprehensive Demonstrations

**33+ in-depth demos** organized by skill level and domain:

- **[Getting Started](demos/README.md#beginner-path-2-4-hours)** - File loading, basic measurements, format export (Beginner, 2-4 hours)
- **[Protocol Decoding](demos/README.md#intermediate-path-6-10-hours)** - UART, SPI, I2C, Manchester, JTAG, USB, PCAP (Intermediate, 6-10 hours)
- **[Reverse Engineering](demos/README.md#advanced-path-12-20-hours)** - CRC recovery, state machines, Wireshark dissectors, automotive protocols (Advanced, 12-20 hours)
- **[Standards Compliance](demos/README.md#advanced-path-12-20-hours)** - IEEE 181/1241/1459/2414, CISPR 32, IEC 61000 (Advanced/Expert)
- **[Complete Workflows](demos/README.md#expert-path-20-40-hours)** - End-to-end production pipelines with ML inference (Expert, 20-40 hours)

**Categories**: Waveform Analysis | File I/O | Custom DAQ | Serial Protocols | Protocol Decoding | UDP Analysis | Protocol Inference | Automotive | Timing | Mixed Signal | Spectral | Jitter | Power | Signal Integrity | EMC | Signal RE | Advanced Inference | Complete Workflows

[**See full demo catalog with learning paths**](demos/README.md)

### Run Your First Demo

```bash
# Install development dependencies
./scripts/setup.sh

# Run your first demo
python demos/00_getting_started/00_hello_world.py

# Or try a specific topic
python demos/05_domain_specific/05_side_channel_basics.py
```

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
```

[**Full CLI reference**](docs/cli.md)

---

## Why This Exists

### Legitimate Use Cases

Hardware reverse engineering serves critical needs across security, repair, modernization, and defense:

**Security Research:** Vulnerability discovery requires understanding how hardware actually works, not how vendors claim it works. Protocol reverse engineering reveals authentication bypasses. State machine analysis maps attack surfaces.

**Right-to-Repair:** Proprietary protocols and vendor lock-in prevent owners from fixing their own equipment. Reverse engineering restores agency. Open documentation enables interoperable replacements.

**Modernization:** Legacy systems run critical infrastructure but use obsolete components. Replication requires extracting specifications from working hardware when documentation is lost or was never public.

**National Defense:** Intelligence and threat assessment depend on understanding adversary capabilities. Forensic analysis of captured equipment requires comprehensive signal analysis and protocol decoding.

**Academic Research:** Understanding existing systems informs better designs. Teaching security requires demonstrating real vulnerabilities. Open tools advance the field collectively.

### The Open Source Philosophy

We believe security through obscurity is a temporary business model at best and a vulnerability at worst. Real security comes from open scrutiny, not information hiding. Real value comes from services and expertise, not gatekeeping knowledge.

Vendors who hide protocol specifications aren't protecting trade secrets—they're preventing interoperability and limiting repair. We're building tools to level that playing field.

### Join the Effort

Hardware reverse engineering requires diverse expertise: signal processing, protocol design, automotive systems, vintage computing, embedded security. No single person knows it all. **We need your knowledge.**

- Reverse engineered a proprietary protocol? Contribute the decoder.
- Built workflow automation techniques? Add them to the framework.
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

| Contribution Type              | Examples                                                           | Impact                                          |
| ------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------- |
| **Workflow Automation**        | New analysis pipelines, export formats, integration scripts        | Core value proposition                          |
| **File Format Loaders**        | Oscilloscope/LA formats not yet supported                          | Eliminate conversion steps                      |
| **Inference Algorithms**       | Better state machine learning, field detection, pattern discovery  | Improve automatic analysis quality              |
| **Protocol Decoders**          | Proprietary protocols you've reversed                              | Enable others to analyze same systems           |
| **Hardware Integration**       | DAQ systems, instrument drivers, live capture workflows            | Enable real-time analysis                       |
| **Real-World Validation**      | Test on your captures, report issues                               | Ensure reliability across use cases             |
| **Documentation & Case Studies** | Tutorials, sanitized RE workflows, academic papers using Oscura    | Lower entry barrier, demonstrate capabilities   |

[**Contributing Guide**](CONTRIBUTING.md) | [Architecture Documentation](docs/developer-guide/architecture.md)

### Community

- **Issues:** [GitHub Issues](https://github.com/oscura-re/oscura/issues) - Bug reports, feature requests
- **Discussions:** [GitHub Discussions](https://github.com/oscura-re/oscura/discussions) - Questions, ideas, collaboration
- **Security:** [SECURITY.md](SECURITY.md) - Responsible disclosure process

---

## Documentation

### User Guides

- [Quick Start Guide](docs/guides/quick-start.md) - Installation and first steps
- [Black-Box Protocol Analysis](docs/guides/blackbox-analysis.md) - Unknown protocol RE workflow
- [Side-Channel Analysis](docs/guides/side-channel-analysis.md) - Using ChipWhisperer traces with Oscura
- [Hardware Acquisition](docs/guides/hardware-acquisition.md) - Direct instrument control
- [Complete Workflows](docs/guides/workflows.md) - End-to-end pipelines

### API Reference

- [API Documentation](docs/api/) - Complete function reference
- [Session Management](docs/api/session-management.md) - Interactive analysis sessions
- [CLI Reference](docs/cli.md) - Command-line interface

### Development

- [Architecture](docs/developer-guide/architecture.md) - Design principles and patterns
- [Testing Guide](docs/testing/) - Test suite architecture
- [CHANGELOG](CHANGELOG.md) - Version history and migration guides

---

## Project Status

**Current Version:** [0.6.0](https://github.com/oscura-re/oscura/releases/latest) (2026-01-25)

**Active Development Areas:**

- Hypothesis-driven RE workflows and confidence scoring
- Automotive protocol analysis (CAN-FD, J1939, OBD-II, UDS)
- Unknown protocol inference (state machines, field detection, CRC recovery)
- Multi-format file loading and export automation
- Vintage computing support (retro logic families, IC identification, 1960s-present)

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
  version = {0.6.0}
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

Hardware systems are black boxes by design, obscured through proprietary protocols, cryptographic obfuscation, and undocumented interfaces. Whether imposed by vendors, governments, or the passage of time—**we bring light to the darkness.** Join us in building the workflow automation framework that hardware reverse engineering deserves.
