# Oscura Demos

**Production-ready demonstrations of Oscura's capabilities for reverse engineering analog and digital signals.**

Demos are the primary documentation for Oscura. Each category has a comprehensive README explaining concepts with working code examples.

---

## First Time Setup

Demo data files are NOT tracked in git. Generate them before running demos:

```bash
# Generate ALL demo data (~220 MB, takes 2-3 minutes)
uv run python demos/generate_all_demo_data.py

# Or generate specific demos only
uv run python demos/generate_all_demo_data.py --demos 01,02,05
```

**Why not in git?**

- All demo data is 100% synthetic and reproducible
- Reduces repository size by 220 MB
- Faster clones for all users

---

## Demo Categories

### Data Loading & I/O

| Category                                      | Description           | Key Capabilities                        |
| --------------------------------------------- | --------------------- | --------------------------------------- |
| [01_waveform_analysis](01_waveform_analysis/) | Oscilloscope captures | Tektronix, Rigol, LeCroy WFM loading    |
| [02_file_format_io](02_file_format_io/)       | All formats           | VCD, CSV, HDF5, NPZ, custom binary      |
| [03_custom_daq](03_custom_daq/)               | Custom DAQ streaming  | Large file handling, streaming patterns |

### Protocol Decoding

| Category                                            | Description            | Key Capabilities                |
| --------------------------------------------------- | ---------------------- | ------------------------------- |
| [04_serial_protocols](04_serial_protocols/)         | UART, SPI, I2C, JTAG   | Auto-baud, multi-mode, ACK/NACK |
| [05_protocol_decoding](05_protocol_decoding/)       | Comprehensive decoding | All protocol types              |
| [06_udp_packet_analysis](06_udp_packet_analysis/)   | UDP/network packets    | PCAP analysis, protocol stats   |
| [08_automotive_protocols](08_automotive_protocols/) | CAN, LIN, FlexRay      | DBC, J1939, transport layer     |

### Signal Analysis

| Category                                          | Description    | Key Capabilities                   |
| ------------------------------------------------- | -------------- | ---------------------------------- |
| [10_timing_measurements](10_timing_measurements/) | Rise/fall time | IEEE 181-2011, duty cycle          |
| [11_mixed_signal](11_mixed_signal/)               | Mixed-signal   | Analog + digital combined          |
| [12_spectral_compliance](12_spectral_compliance/) | FFT, THD, SNR  | IEEE 1241-2010, ENOB, SFDR         |
| [13_jitter_analysis](13_jitter_analysis/)         | TIE, RJ/DJ     | IEEE 2414-2020, eye diagrams       |
| [14_power_analysis](14_power_analysis/)           | Power quality  | IEEE 1459-2010, efficiency, ripple |
| [15_signal_integrity](15_signal_integrity/)       | TDR, S-params  | Eye metrics, crosstalk             |

### Inference & RE

| Category                                                        | Description         | Key Capabilities                       |
| --------------------------------------------------------------- | ------------------- | -------------------------------------- |
| [07_protocol_inference](07_protocol_inference/)                 | Reverse engineering | CRC reverse, state machines, Wireshark |
| [17_signal_reverse_engineering](17_signal_reverse_engineering/) | Signal RE           | Unknown signal analysis                |
| [18_advanced_inference](18_advanced_inference/)                 | ML techniques       | Bayesian, L\*, Protocol DSL            |

### Domain-Specific

| Category                                | Description        | Key Capabilities                |
| --------------------------------------- | ------------------ | ------------------------------- |
| [09_automotive](09_automotive/)         | OBD-II, UDS, J1939 | 54+ PIDs, 17 UDS services, DTCs |
| [16_emc_compliance](16_emc_compliance/) | EMC/EMI testing    | CISPR, FCC, MIL-STD limit masks |

### Workflows

| Category                                        | Description | Key Capabilities           |
| ----------------------------------------------- | ----------- | -------------------------- |
| [19_complete_workflows](19_complete_workflows/) | End-to-end  | Full RE pipelines, reports |

---

## Quick Start

### Run Your First Demo

```bash
# 1. Generate demo data
uv run python demos/generate_all_demo_data.py --demos 01

# 2. Run a demo
uv run python demos/01_waveform_analysis/comprehensive_wfm_analysis.py
```

### Common Demo Commands

```bash
# Analyze waveform
uv run python demos/12_spectral_compliance/comprehensive_spectral_demo.py

# Decode UART/JTAG
uv run python demos/04_serial_protocols/jtag_demo.py

# Automotive CAN analysis
uv run python demos/09_automotive/comprehensive_automotive_demo.py

# EMC compliance check
uv run python demos/16_emc_compliance/comprehensive_emc_demo.py
```

---

## When to Use Which Demo

| Your Task                         | Use This Demo                                                   |
| --------------------------------- | --------------------------------------------------------------- |
| Load oscilloscope captures        | [01_waveform_analysis](01_waveform_analysis/)                   |
| Load custom binary formats        | [02_file_format_io](02_file_format_io/)                         |
| Stream large DAQ files            | [03_custom_daq](03_custom_daq/)                                 |
| Decode UART/SPI/I2C/JTAG          | [04_serial_protocols](04_serial_protocols/)                     |
| Comprehensive protocol decoding   | [05_protocol_decoding](05_protocol_decoding/)                   |
| Analyze network packets           | [06_udp_packet_analysis](06_udp_packet_analysis/)               |
| Reverse engineer unknown protocol | [07_protocol_inference](07_protocol_inference/)                 |
| Analyze CAN/LIN/FlexRay           | [08_automotive_protocols](08_automotive_protocols/)             |
| OBD-II/UDS diagnostics            | [09_automotive](09_automotive/)                                 |
| Rise/fall time measurement        | [10_timing_measurements](10_timing_measurements/)               |
| Mixed analog+digital analysis     | [11_mixed_signal](11_mixed_signal/)                             |
| Audio/ADC spectral analysis       | [12_spectral_compliance](12_spectral_compliance/)               |
| Jitter and eye diagrams           | [13_jitter_analysis](13_jitter_analysis/)                       |
| Power supply characterization     | [14_power_analysis](14_power_analysis/)                         |
| High-speed signal integrity       | [15_signal_integrity](15_signal_integrity/)                     |
| EMC pre-compliance testing        | [16_emc_compliance](16_emc_compliance/)                         |
| Reverse engineer unknown signal   | [17_signal_reverse_engineering](17_signal_reverse_engineering/) |
| Advanced ML inference             | [18_advanced_inference](18_advanced_inference/)                 |
| Complete RE workflow              | [19_complete_workflows](19_complete_workflows/)                 |

---

## Standards Compliance

Oscura demos validate against industry standards:

| Standard        | Demo Category           | Coverage |
| --------------- | ----------------------- | -------- |
| IEEE 181-2011   | 10_timing_measurements  | Full     |
| IEEE 1241-2010  | 12_spectral_compliance  | Full     |
| IEEE 1459-2010  | 14_power_analysis       | Full     |
| IEEE 2414-2020  | 13_jitter_analysis      | Full     |
| ISO 11898 (CAN) | 08_automotive_protocols | Full     |
| ISO 14229 (UDS) | 09_automotive           | Full     |
| CISPR 32        | 16_emc_compliance       | Full     |
| FCC Part 15     | 16_emc_compliance       | Full     |

---

## Demo Validation

All demos include self-validation and are tested in CI:

```bash
# List all available demos
uv run python demos/validate_all_demos.py --list

# Validate all demos
uv run python demos/validate_all_demos.py

# Check specific category
uv run python demos/validate_all_demos.py --category serial
```

---

## Contributing

To add a new demo:

1. Create demo in appropriate category (or create new numbered category)
2. Update category README.md with demo details
3. Add to `validate_all_demos.py` for CI validation
4. Update this main README.md

**Demo Quality Standards**:

- Production-ready code with error handling
- Self-validating outputs (assertions)
- Comprehensive README in each category
- Test data included or generatable

---

**Last Updated**: 2026-01-16
**Status**: Production-ready demonstrations
