# Oscura

**Unified hardware reverse engineering framework. Extract all information from any system through signals and data.**

**Build Status:**
[![CI](https://github.com/oscura-re/oscura/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/ci.yml)
[![Code Quality](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/code-quality.yml)
[![Documentation](https://github.com/oscura-re/oscura/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/docs.yml)
[![Test Quality](https://github.com/oscura-re/oscura/actions/workflows/test-quality.yml/badge.svg?branch=main)](https://github.com/oscura-re/oscura/actions/workflows/test-quality.yml)

**Package:**
[![PyPI version](https://img.shields.io/pypi/v/oscura)](https://pypi.org/project/oscura/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/oscura)](https://pypi.org/project/oscura/)

**Code Quality:**
[![codecov](https://codecov.io/gh/oscura-re/oscura/graph/badge.svg)](https://codecov.io/gh/oscura-re/oscura)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docstring Coverage](https://raw.githubusercontent.com/oscura-re/oscura/main/docs/badges/interrogate_badge.svg)](https://github.com/oscura-re/oscura/tree/main/docs)

**Project Status:**
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/oscura-re/oscura/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/oscura-re/oscura)](https://github.com/oscura-re/oscura/commits/main)

---

## What is Oscura?

Oscura is a hardware reverse engineering framework for security researchers, right-to-repair advocates, defense analysts, and commercial intelligence teams. From oscilloscope captures to complete system understanding.

**Reverse Engineering**: Unknown protocol discovery • State machine extraction • CRC/checksum recovery • Proprietary device replication • Security vulnerability analysis • Black-box protocol analysis

**Signal Analysis**: IEEE-compliant measurements (181/1241/1459/2414) • Comprehensive protocol decoding (16+ protocols) • Spectral analysis • Timing characterization • Side-channel analysis (DPA/CPA/timing attacks)

**Unified Acquisition**: File-based • Hardware sources (SocketCAN, Saleae, PyVISA - Phase 2) • Synthetic generation • Polymorphic Source protocol

**Interactive Sessions**: Domain-specific analysis sessions • BlackBoxSession for protocol RE • Differential analysis • Field hypothesis generation • Protocol specification export

**Built For**: Exploitation • Replication • Defense analysis • Commercial intelligence • Right-to-repair • Cryptographic research

---

## Installation

```bash
# Using uv (recommended)
uv pip install oscura

# Or with pip
pip install oscura

# Development install (RECOMMENDED)
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh            # Complete setup (dependencies + hooks)
./scripts/verify-setup.sh     # Verify environment is ready
```python

---

## Quick Start

### Signal Analysis

```python
import oscura as osc

# Load oscilloscope capture
trace = osc.load("capture.wfm")

# Basic measurements
print(f"Frequency: {osc.frequency(trace) / 1e6:.3f} MHz")
print(f"Rise time: {osc.rise_time(trace) * 1e9:.1f} ns")

# Decode protocol
from oscura.protocols import UARTDecoder
decoder = UARTDecoder(baud_rate=115200)
messages = decoder.decode(trace)
```python

### Black-Box Protocol Reverse Engineering

```python
from oscura.sessions import BlackBoxSession
from oscura.acquisition import FileSource

# Create analysis session
session = BlackBoxSession(name="IoT Device RE")

# Add recordings from different stimuli
session.add_recording("idle", FileSource("idle.bin"))
session.add_recording("button_press", FileSource("button.bin"))

# Differential analysis
diff = session.compare("idle", "button_press")
print(f"Changed bytes: {diff.changed_bytes}")

# Generate protocol specification
spec = session.generate_protocol_spec()
print(f"Detected {len(spec['fields'])} protocol fields")

# Export Wireshark dissector
session.export_results("dissector", "protocol.lua")
```python

### CAN Protocol Analysis

```python
from oscura.automotive.can import CANSession
from oscura.acquisition import FileSource

# Create session
session = CANSession(name="Vehicle Analysis")

# Add recordings from CAN bus captures
session.add_recording("idle", FileSource("idle.blf"))
session.add_recording("accelerate", FileSource("accelerate.blf"))

# Analyze traffic
analysis = session.analyze()
print(f"Messages: {analysis['inventory']['total_messages']}")
print(f"Unique IDs: {len(analysis['inventory']['message_ids'])}")

# Compare recordings
diff = session.compare("idle", "accelerate")
print(f"Changed IDs: {len(diff.details['changed_ids'])}")

# Export DBC file
session.export_dbc("vehicle.dbc")
```markdown

---

## Learn by Example

**Demos are the documentation.** Each category includes working code with comprehensive explanations.

### Core Capabilities

| Demo | Description |
|------|-------------|
| [01_waveform_analysis](demos/01_waveform_analysis/) | Load and analyze Tektronix, Rigol, LeCroy captures |
| [02_file_format_io](demos/02_file_format_io/) | CSV, HDF5, NumPy, custom binary formats |
| [03_custom_daq](demos/03_custom_daq/) | Streaming loaders for custom DAQ systems |
| [04_serial_protocols](demos/04_serial_protocols/) | UART, SPI, I2C, 1-Wire decoding |
| [05_protocol_decoding](demos/05_protocol_decoding/) | Protocol auto-detection and decoding |
| [06_udp_packet_analysis](demos/06_udp_packet_analysis/) | Network packet capture and analysis |

### Advanced Analysis

| Demo | Description |
|------|-------------|
| [07_protocol_inference](demos/07_protocol_inference/) | State machine learning, CRC reverse engineering |
| [08_automotive_protocols](demos/08_automotive_protocols/) | CAN, CAN-FD, LIN, FlexRay analysis |
| [09_automotive](demos/09_automotive/) | OBD-II, UDS, J1939 diagnostics |
| [10_timing_measurements](demos/10_timing_measurements/) | Rise/fall time, duty cycle (IEEE 181) |
| [11_mixed_signal](demos/11_mixed_signal/) | Analog + digital combined analysis |
| [12_spectral_compliance](demos/12_spectral_compliance/) | FFT, THD, SNR, SINAD (IEEE 1241) |
| [13_jitter_analysis](demos/13_jitter_analysis/) | TIE, RJ/DJ, eye diagrams (IEEE 2414) |
| [14_power_analysis](demos/14_power_analysis/) | DC-DC, ripple, efficiency (IEEE 1459) |
| [15_signal_integrity](demos/15_signal_integrity/) | TDR, S-parameters, setup/hold timing |
| [16_emc_compliance](demos/16_emc_compliance/) | CISPR, FCC, MIL-STD testing |
| [17_signal_reverse_engineering](demos/17_signal_reverse_engineering/) | Complete unknown signal analysis |
| [18_advanced_inference](demos/18_advanced_inference/) | Bayesian inference, Protocol DSL |
| [19_complete_workflows](demos/19_complete_workflows/) | End-to-end reverse engineering |

### Run Your First Demo

```bash
# Generate demo data
python demos/generate_all_demo_data.py

# Run a demo
uv run python demos/01_waveform_analysis/comprehensive_wfm_analysis.py
```markdown

---

## Key Features

### Protocols (16+)

UART • SPI • I2C • 1-Wire • CAN • CAN-FD • LIN • FlexRay • JTAG • SWD • Manchester • Miller • USB • HDLC • I2S • MDIO • DMX512

### File Formats (18+)

Tektronix WFM • Rigol WFM • LeCroy TRC • Sigrok • VCD • CSV • NumPy • HDF5 • MATLAB • WAV • JSON • BLF • MF4 • PCAP • PCAPNG

### Standards Compliance

**IEEE 181-2011**: Rise/fall time, pulse width, overshoot

**IEEE 1241-2010**: SNR, SINAD, THD, SFDR, ENOB

**IEEE 1459-2010**: Power factor, harmonics, efficiency

**IEEE 2414-2020**: TIE, period jitter, RJ/DJ

---

## Command Line Interface

```bash
# Characterize signal measurements
oscura characterize capture.wfm

# Decode protocol
oscura decode uart.wfm --protocol uart

# Batch process multiple files
oscura batch '*.wfm' --analysis characterize

# Compare two signals
oscura compare before.wfm after.wfm

# Interactive shell
oscura shell
```bash

See [CLI Reference](docs/cli.md) for complete documentation.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
# Quick setup
git clone https://github.com/oscura-re/oscura.git
cd oscura
uv sync --all-extras
./scripts/setup/install-hooks.sh

# Run tests
./scripts/test.sh

# Quality checks
./scripts/check.sh
```python

---

## Documentation

### Getting Started

- **[Quick Start Guide](docs/guides/quick-start.md)** - Begin here
- **[Demos](demos/)** - Working examples for every feature
- **[Migration Guide](docs/migration/v0-to-v1.md)** - Upgrade from older versions

### User Guides

- **[Black-Box Protocol Analysis](docs/guides/blackbox-analysis.md)** - Unknown protocol reverse engineering
- **[Hardware Acquisition](docs/guides/hardware-acquisition.md)** - Direct hardware integration (Phase 2)
- **[Side-Channel Analysis](docs/guides/side-channel-analysis.md)** - Power/timing/EM attacks
- **[Workflows](docs/guides/workflows.md)** - Complete analysis workflows

### API Reference

- **[API Reference](docs/api/)** - Complete API documentation
- **[Session Management](docs/api/session-management.md)** - Interactive analysis sessions
- **[CLI Reference](docs/cli.md)** - Command line usage

### Development

- **[Architecture](docs/architecture/)** - Design principles and patterns
- **[Testing Guide](docs/testing/)** - Test suite architecture
- **[CHANGELOG](CHANGELOG.md)** - Version history

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use Oscura in research:

```bibtex
@software{oscura,
  title = {Oscura: Signal Reverse Engineering Toolkit},
  author = {Oscura Contributors},
  year = {2026},
  url = {https://github.com/oscura-re/oscura}
}
```python

Machine-readable: [CITATION.cff](CITATION.cff)

---

## Support

- **[GitHub Issues](https://github.com/oscura-re/oscura/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/oscura-re/oscura/discussions)** - Questions and community

---

**Oscura** • Reverse engineer any system from captured waveforms
