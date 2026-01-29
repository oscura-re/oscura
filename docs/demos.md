# Demos - Working Examples

All Oscura demos are **working Python scripts** with comprehensive READMEs. They serve as both documentation and validation.

→ **[Browse all demos on GitHub](https://github.com/oscura-re/oscura/tree/main/demos)**

---

## Demo Categories

### 🎯 **Beginner-Friendly**

Start here if you're new to Oscura:

| Demo                                                                                               | Description                         | Key Features                     |
| -------------------------------------------------------------------------------------------------- | ----------------------------------- | -------------------------------- |
| [01 - Waveform Analysis](https://github.com/oscura-re/oscura/tree/main/demos/01_waveform_analysis) | Load and analyze oscilloscope files | File loading, basic measurements |
| [02 - File Formats](https://github.com/oscura-re/oscura/tree/main/demos/02_file_format_io)         | Work with multiple file formats     | CSV, HDF5, MATLAB, VCD           |
| [04 - Serial Protocols](https://github.com/oscura-re/oscura/tree/main/demos/04_serial_protocols)   | Decode UART, SPI, I2C, JTAG         | Basic protocol decoding          |

### 🔧 **Intermediate**

More complex workflows and analysis:

| Demo                                                                                               | Description                            | Key Features               |
| -------------------------------------------------------------------------------------------------- | -------------------------------------- | -------------------------- |
| [03 - Custom DAQ](https://github.com/oscura-re/oscura/tree/main/demos/03_custom_daq)               | Memory-efficient large file processing | Streaming, chunking        |
| [05 - Protocol Decoding](https://github.com/oscura-re/oscura/tree/main/demos/05_protocol_decoding) | Comprehensive multi-protocol decode    | Auto-detection, validation |
| [06 - UDP Analysis](https://github.com/oscura-re/oscura/tree/main/demos/06_udp_packet_analysis)    | Network packet analysis                | PCAP, payload parsing      |
| [10 - Timing](https://github.com/oscura-re/oscura/tree/main/demos/10_timing_measurements)          | IEEE 181 pulse measurements            | Rise/fall time, slew rate  |
| [11 - Mixed Signal](https://github.com/oscura-re/oscura/tree/main/demos/11_mixed_signal)           | Analog + digital analysis              | Clock recovery, jitter     |

### 🚀 **Advanced**

Complex reverse engineering and compliance:

| Demo                                                                                                     | Description                       | Key Features              |
| -------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------- |
| [07 - Protocol Inference](https://github.com/oscura-re/oscura/tree/main/demos/07_protocol_inference)     | CRC recovery, pattern recognition | Protocol RE workflows     |
| [08 - Automotive Protocols](https://github.com/oscura-re/oscura/tree/main/demos/08_automotive_protocols) | CAN, LIN, FlexRay                 | Multi-protocol automotive |
| [09 - Automotive](https://github.com/oscura-re/oscura/tree/main/demos/09_automotive)                     | OBD-II, UDS, J1939 diagnostics    | Advanced automotive       |
| [17 - Signal RE](https://github.com/oscura-re/oscura/tree/main/demos/17_signal_reverse_engineering)      | Complete RE workflow              | Unknown signal analysis   |
| [18 - Advanced Inference](https://github.com/oscura-re/oscura/tree/main/demos/18_advanced_inference)     | ML-based inference                | State machine learning    |

### ✅ **Compliance & Standards**

IEEE and EMC validation:

| Demo                                                                                                   | Description                            | Standard       |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------- | -------------- |
| [12 - Spectral Compliance](https://github.com/oscura-re/oscura/tree/main/demos/12_spectral_compliance) | FFT, THD, SNR, SINAD, ENOB             | IEEE 1241-2010 |
| [13 - Jitter Analysis](https://github.com/oscura-re/oscura/tree/main/demos/13_jitter_analysis)         | TIE, RJ/DJ decomposition, eye diagrams | IEEE 2414-2020 |
| [14 - Power Analysis](https://github.com/oscura-re/oscura/tree/main/demos/14_power_analysis)           | Power quality, harmonics               | IEEE 1459-2010 |
| [15 - Signal Integrity](https://github.com/oscura-re/oscura/tree/main/demos/15_signal_integrity)       | TDR, S-parameters                      | IEEE 181-2011  |
| [16 - EMC Compliance](https://github.com/oscura-re/oscura/tree/main/demos/16_emc_compliance)           | CISPR 32, IEC 61000 testing            | CISPR 16       |

### 🏁 **Complete Workflows**

End-to-end production examples:

| Demo                                                                                                 | Description                | Use Case             |
| ---------------------------------------------------------------------------------------------------- | -------------------------- | -------------------- |
| [19 - Complete Workflows](https://github.com/oscura-re/oscura/tree/main/demos/19_complete_workflows) | Production-ready pipelines | Real-world workflows |

---

## How Demos Work

Each demo includes:

1. **README.md** - Comprehensive documentation
2. **Working Python script(s)** - Production-ready code
3. **Expected outputs** - What success looks like
4. **Self-validation** - Built-in tests

### Running a Demo

```bash
# Navigate to demo directory
cd demos/01_waveform_analysis

# Run the demo
uv run python comprehensive_wfm_analysis.py

# Most demos work with test data by default
# Or provide your own file:
uv run python comprehensive_wfm_analysis.py --wfm-file your_file.wfm
```

---

## Demo Statistics

- **19 categories** with 33+ comprehensive demos covering all major features
- **Learning paths** organized by skill level: Beginner (2-4h), Intermediate (6-10h), Advanced (12-20h), Expert (20-40h)
- **All validated** via automated checker (BaseDemo pattern)
- **Self-testing** - demos include validation suite
- **Synthetic data** - no external files needed

---

## Finding the Right Demo

### By Use Case

- **"I have an oscilloscope file"** → [Demo 01](https://github.com/oscura-re/oscura/tree/main/demos/01_waveform_analysis)
- **"I need to decode UART/SPI"** → [Demo 04](https://github.com/oscura-re/oscura/tree/main/demos/04_serial_protocols)
- **"Unknown signal to reverse engineer"** → [Demo 17](https://github.com/oscura-re/oscura/tree/main/demos/17_signal_reverse_engineering)
- **"CAN bus / automotive"** → [Demo 09](https://github.com/oscura-re/oscura/tree/main/demos/09_automotive)
- **"Need IEEE compliance"** → [Demo 12](https://github.com/oscura-re/oscura/tree/main/demos/12_spectral_compliance)

### By File Format

- **Tektronix .wfm** → [Demo 01](https://github.com/oscura-re/oscura/tree/main/demos/01_waveform_analysis)
- **Sigrok .sr / VCD** → [Demo 02](https://github.com/oscura-re/oscura/tree/main/demos/02_file_format_io)
- **PCAP network** → [Demo 06](https://github.com/oscura-re/oscura/tree/main/demos/06_udp_packet_analysis)
- **CAN (BLF/ASC)** → [Demo 08](https://github.com/oscura-re/oscura/tree/main/demos/08_automotive_protocols)

### By Protocol

- **UART, SPI, I2C** → [Demo 04](https://github.com/oscura-re/oscura/tree/main/demos/04_serial_protocols)
- **CAN, LIN, FlexRay** → [Demo 08](https://github.com/oscura-re/oscura/tree/main/demos/08_automotive_protocols)
- **OBD-II, J1939, UDS** → [Demo 09](https://github.com/oscura-re/oscura/tree/main/demos/09_automotive)
- **Unknown protocol** → [Demo 17](https://github.com/oscura-re/oscura/tree/main/demos/17_signal_reverse_engineering)

---

## Next Steps

- **Try a demo**: Pick one matching your use case
- **Read the guide**: Each README explains concepts
- **Modify the code**: Demos are designed to be adapted
- **Check API docs**: [API Reference](api/index.md) for deep dives
