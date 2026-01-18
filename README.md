# Oscura

**Unified hardware reverse engineering framework. Extract all information from any system through signals and data.**

[![PyPI version](https://badge.fury.io/py/oscura.svg)](https://badge.fury.io/py/oscura)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/oscura-re/oscura/workflows/CI/badge.svg)](https://github.com/oscura-re/oscura/actions)

---

## What is Oscura?

Oscura is a hardware reverse engineering framework for security researchers, right-to-repair advocates, defense analysts, and commercial intelligence teams. From oscilloscope captures to complete system understanding.

**Reverse Engineering**: Unknown protocol discovery • State machine extraction • CRC/checksum recovery • Proprietary device replication • Security vulnerability analysis

**Signal Analysis**: IEEE-compliant measurements (181/1241/1459/2414) • Comprehensive protocol decoding (16+ protocols) • Spectral analysis • Timing characterization

**Built For**: Exploitation • Replication • Defense analysis • Commercial intelligence • Right-to-repair

---

## Installation

```bash
# Using uv (recommended)
uv pip install oscura

# Or with pip
pip install oscura

# Development install
git clone https://github.com/oscura-re/oscura.git
cd oscura
uv sync --all-extras
./scripts/setup/install-hooks.sh
```

---

## Quick Start

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
```

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
python demos/data_generation/generate_all_demo_data.py

# Run a demo
uv run python demos/01_waveform_analysis/comprehensive_wfm_analysis.py
```

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
# Analyze a waveform
oscura analyze capture.wfm

# Decode protocol
oscura decode capture.wfm --protocol uart --baud 115200

# Generate report
oscura report capture.wfm -o report.pdf

# Convert formats
oscura convert input.wfm output.csv
```

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
```

---

## Documentation

- **[Demos](demos/)** - Start here (working examples)
- **[API Reference](docs/api/)** - Complete API documentation
- **[CLI Reference](docs/cli.md)** - Command line usage
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
```

Machine-readable: [CITATION.cff](CITATION.cff)

---

## Support

- **[GitHub Issues](https://github.com/oscura-re/oscura/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/oscura-re/oscura/discussions)** - Questions and community

---

**Oscura** • Reverse engineer any system from captured waveforms
