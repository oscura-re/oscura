# Oscura Demonstrations

Comprehensive demonstrations of all Oscura capabilities organized by complexity and application domain.

**Total Demos**: 33 core demonstrations + 3 complete workflows + utilities
**Start Here**: `01_waveform_analysis/comprehensive_wfm_analysis.py`
**Installation**: See main [README](../README.md) for Oscura installation

## Overview

All demonstrations follow the `BaseDemo` pattern with:
- Self-contained synthetic data generation
- Automatic validation checks
- Formatted console output
- Working code ready to adapt

---

## Quick Start

```bash
# Run your first demo
python demos/01_waveform_analysis/comprehensive_wfm_analysis.py

# Run with validation
python demos/01_waveform_analysis/comprehensive_wfm_analysis.py --validate

# Validate all demos (requires all dependencies)
python demos/validate_all_demos.py
```

---

## Category Index

| Category | Demos | Description | Level |
|----------|-------|-------------|-------|
| [01 - Waveform Analysis](#01---waveform-analysis) | 2 | Basic oscilloscope file loading and measurements | Beginner |
| [02 - File Format I/O](#02---file-format-io) | 1 | VCD, CSV, HDF5 format handling | Beginner |
| [03 - Custom DAQ](#03---custom-daq) | 3 | Memory-efficient large file processing | Intermediate |
| [04 - Serial Protocols](#04---serial-protocols) | 6 | Manchester, I2S, JTAG, OneWire, SWD, USB | Intermediate |
| [05 - Protocol Decoding](#05---protocol-decoding) | 1 | Comprehensive multi-protocol decoder | Intermediate |
| [06 - UDP Packet Analysis](#06---udp-packet-analysis) | 1 | PCAP and network packet analysis | Intermediate |
| [07 - Protocol Inference](#07---protocol-inference) | 3 | CRC recovery, state machines, Wireshark dissectors | Advanced |
| [08 - Automotive Protocols](#08---automotive-protocols) | 2 | FlexRay, LIN protocol analysis | Advanced |
| [09 - Automotive](#09---automotive) | 1 | OBD-II, UDS, J1939 diagnostics | Advanced |
| [10 - Timing Measurements](#10---timing-measurements) | 1 | IEEE 181 pulse measurements | Intermediate |
| [11 - Mixed Signal](#11---mixed-signal) | 1 | Analog + digital correlation | Intermediate |
| [12 - Spectral Compliance](#12---spectral-compliance) | 1 | FFT, THD, SNR, SINAD, ENOB (IEEE 1241) | Advanced |
| [13 - Jitter Analysis](#13---jitter-analysis) | 2 | TIE, RJ/DJ, bathtub curves (IEEE 2414) | Advanced |
| [14 - Power Analysis](#14---power-analysis) | 2 | DC/DC efficiency, ripple analysis (IEEE 1459) | Advanced |
| [15 - Signal Integrity](#15---signal-integrity) | 3 | TDR, S-parameters, timing analysis (IEEE 181) | Advanced |
| [16 - EMC Compliance](#16---emc-compliance) | 1 | CISPR 32, IEC 61000 testing | Expert |
| [17 - Signal RE](#17---signal-reverse-engineering) | 3 | Complete unknown signal RE workflow | Expert |
| [18 - Advanced Inference](#18---advanced-inference) | 3 | ML-based, Bayesian, DSL approaches | Expert |
| [19 - Complete Workflows](#19---complete-workflows) | 3 | End-to-end production pipelines | Expert |

---

## Learning Paths

### Beginner Path (2-4 hours)

**Objective**: Learn basic Oscura operations and file handling

1. **01_waveform_analysis/comprehensive_wfm_analysis.py** (30 min)
   - Load oscilloscope files
   - Basic measurements (amplitude, frequency, duty cycle)
   - Understanding WaveformTrace objects

2. **02_file_format_io/vcd_loader_demo.py** (15 min)
   - VCD file format handling
   - Multi-format data export

3. **01_waveform_analysis/all_output_formats.py** (20 min)
   - CSV, JSON, HDF5, MATLAB export formats
   - Data interchange patterns

4. **10_timing_measurements/ieee_181_pulse_demo.py** (30 min)
   - IEEE 181-2011 pulse measurements
   - Rise/fall time, overshoot, slew rate

5. **04_serial_protocols/manchester_demo.py** (30 min)
   - First protocol decoding example
   - Understanding encoding schemes

**Outcome**: Comfortable loading files, running basic measurements, exporting results

---

### Intermediate Path (6-10 hours)

**Prerequisites**: Complete Beginner Path

**Objective**: Master protocol decoding and analysis workflows

1. **05_protocol_decoding/comprehensive_protocol_demo.py** (1 hour)
   - UART, SPI, I2C decoding
   - Auto-detection capabilities
   - Multi-protocol handling

2. **04_serial_protocols/** (2 hours total)
   - **i2s_demo.py** - Audio serial interface
   - **jtag_demo.py** - JTAG boundary scan
   - **onewire_demo.py** - 1-Wire protocol
   - **swd_demo.py** - ARM Serial Wire Debug
   - **usb_demo.py** - USB packet decoding

3. **06_udp_packet_analysis/comprehensive_udp_analysis.py** (45 min)
   - PCAP file loading
   - Network protocol analysis
   - Payload parsing

4. **11_mixed_signal/comprehensive_mixed_signal_demo.py** (1 hour)
   - Clock recovery from noisy signals
   - Analog/digital correlation
   - Jitter measurements

5. **03_custom_daq/** (2 hours total)
   - **simple_loader.py** - Basic custom format loader
   - **chunked_loader.py** - Memory-efficient processing
   - **optimal_streaming_loader.py** - High-performance streaming

**Outcome**: Proficient with protocol decoding, custom loaders, mixed-signal analysis

---

### Advanced Path (12-20 hours)

**Prerequisites**: Complete Intermediate Path

**Objective**: Reverse engineering, inference, standards compliance

1. **07_protocol_inference/** (3 hours total)
   - **crc_reverse_demo.py** - CRC specification recovery
   - **state_machine_learning.py** - RPNI algorithm for state machines
   - **wireshark_dissector_demo.py** - Auto-generate Lua dissectors

2. **08_automotive_protocols/** + **09_automotive/** (3 hours total)
   - **flexray_demo.py** - FlexRay protocol analysis
   - **lin_demo.py** - LIN bus decoding
   - **comprehensive_automotive_demo.py** - OBD-II, UDS, J1939

3. **12_spectral_compliance/comprehensive_spectral_demo.py** (1.5 hours)
   - IEEE 1241-2010 ADC testing
   - FFT, THD, SNR, SINAD, ENOB, SFDR
   - Spurious-Free Dynamic Range

4. **13_jitter_analysis/** (2 hours total)
   - **ddj_dcd_demo.py** - Data-dependent jitter, duty cycle distortion
   - **bathtub_curve_demo.py** - Eye diagram BER analysis

5. **14_power_analysis/** (2 hours total)
   - **dcdc_efficiency_demo.py** - DC/DC converter characterization
   - **ripple_analysis_demo.py** - Power supply quality (IEEE 1459)

6. **15_signal_integrity/** (2.5 hours total)
   - **tdr_impedance_demo.py** - Time-domain reflectometry
   - **sparams_demo.py** - S-parameter extraction
   - **setup_hold_timing_demo.py** - Digital timing validation

7. **16_emc_compliance/comprehensive_emc_demo.py** (2 hours)
   - CISPR 32 conducted/radiated emissions
   - IEC 61000-4-x immunity testing
   - EMI spectrum analysis

**Outcome**: Expert-level analysis, standards compliance, automotive protocols

---

### Expert Path (20-40 hours)

**Prerequisites**: Complete Advanced Path

**Objective**: Complete reverse engineering workflows and ML-based inference

1. **17_signal_reverse_engineering/** (8 hours total)
   - **comprehensive_re.py** - Full RE workflow on unknown signal
   - **exploratory_analysis.py** - Statistical signal characterization
   - **reverse_engineer_tool.py** - Complete tool integration

2. **18_advanced_inference/** (6 hours total)
   - **bayesian_inference_demo.py** - Probabilistic protocol inference
   - **active_learning_demo.py** - ML-assisted field detection
   - **protocol_dsl_demo.py** - Domain-specific language generation

3. **19_complete_workflows/** (6 hours total)
   - **unknown_signal_workflow.py** - Capture to dissector pipeline
   - **network_analysis_workflow.py** - Full network protocol RE
   - **automotive_full_workflow.py** - CAN to DBC generation

**Outcome**: Master-level capabilities, production workflow automation

---

## Demo Catalog

### 01 - Waveform Analysis

**Level**: Beginner | **Demos**: 2

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_wfm_analysis.py | Complete waveform analysis workflow | Load files, measurements, statistics, plots | 30 min |
| all_output_formats.py | Export demonstration | CSV, JSON, HDF5, MATLAB, VCD formats | 20 min |

**Key Capabilities**: File loading, amplitude/frequency/duty cycle measurements, multi-format export

---

### 02 - File Format I/O

**Level**: Beginner | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| vcd_loader_demo.py | VCD file format handling | Load VCD, parse digital signals, export | 15 min |

**Key Capabilities**: VCD format parsing, digital signal extraction, format conversion

---

### 03 - Custom DAQ

**Level**: Intermediate | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| simple_loader.py | Basic custom format loader | Implement custom file format parser | 30 min |
| chunked_loader.py | Memory-efficient chunked processing | Handle TB-scale files with limited RAM | 45 min |
| optimal_streaming_loader.py | High-performance streaming | Real-time data ingestion, zero-copy | 1 hour |

**Key Capabilities**: Custom format loaders, memory-mapped files, streaming processing

---

### 04 - Serial Protocols

**Level**: Intermediate | **Demos**: 6

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| manchester_demo.py | Manchester encoding/decoding | Self-clocking codes, bit recovery | 30 min |
| i2s_demo.py | I2S audio serial interface | Audio data extraction, multi-channel | 30 min |
| jtag_demo.py | JTAG boundary scan analysis | TAP state machine, IR/DR capture | 45 min |
| onewire_demo.py | 1-Wire protocol decoding | Temperature sensors, ROM codes | 30 min |
| swd_demo.py | ARM Serial Wire Debug | Memory access, debug registers | 45 min |
| usb_demo.py | USB packet decoding | Control, bulk, interrupt transfers | 1 hour |

**Key Capabilities**: Advanced protocol decoders, encoding schemes, hardware debug interfaces

---

### 05 - Protocol Decoding

**Level**: Intermediate | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_protocol_demo.py | Multi-protocol decoder | UART, SPI, I2C auto-detection and decode | 1 hour |

**Key Capabilities**: Auto-detection, multi-protocol analysis, validation

---

### 06 - UDP Packet Analysis

**Level**: Intermediate | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_udp_analysis.py | PCAP network analysis | Load PCAP, parse UDP, payload analysis | 45 min |

**Key Capabilities**: PCAP loading, network protocol stack, payload parsing

---

### 07 - Protocol Inference

**Level**: Advanced | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| crc_reverse_demo.py | CRC specification recovery | Polynomial, init, XOR-out detection | 1 hour |
| state_machine_learning.py | RPNI state machine extraction | Passive observation learning | 1.5 hours |
| wireshark_dissector_demo.py | Auto-generate Wireshark dissectors | Protocol spec to Lua dissector | 1 hour |

**Key Capabilities**: CRC recovery, state machine inference, dissector generation

---

### 08 - Automotive Protocols

**Level**: Advanced | **Demos**: 2

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| flexray_demo.py | FlexRay protocol analysis | Time-triggered automotive networking | 1.5 hours |
| lin_demo.py | LIN bus decoding | Low-speed automotive serial | 1 hour |

**Key Capabilities**: Automotive protocols, time-triggered systems, bus analysis

---

### 09 - Automotive

**Level**: Advanced | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_automotive_demo.py | OBD-II, UDS, J1939 diagnostics | Vehicle diagnostics, DTC reading | 2 hours |

**Key Capabilities**: OBD-II diagnostics, UDS services, J1939 parameter groups numbers

---

### 10 - Timing Measurements

**Level**: Intermediate | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| ieee_181_pulse_demo.py | IEEE 181-2011 pulse measurements | Rise/fall time, overshoot, slew rate | 30 min |

**Key Capabilities**: IEEE 181-2011 compliance, pulse parameter extraction

---

### 11 - Mixed Signal

**Level**: Intermediate | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_mixed_signal_demo.py | Analog + digital correlation | Clock recovery, jitter, crosstalk | 1 hour |

**Key Capabilities**: Mixed-signal analysis, clock recovery, analog/digital correlation

---

### 12 - Spectral Compliance

**Level**: Advanced | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_spectral_demo.py | IEEE 1241-2010 ADC testing | FFT, THD, SNR, SINAD, ENOB, SFDR | 1.5 hours |

**Key Capabilities**: IEEE 1241-2010, spectral analysis, ADC characterization

---

### 13 - Jitter Analysis

**Level**: Advanced | **Demos**: 2

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| ddj_dcd_demo.py | Data-dependent jitter analysis | DDJ, DCD, pattern-dependent effects | 1 hour |
| bathtub_curve_demo.py | Eye diagram BER analysis | TIE, bathtub curves, RJ/DJ decomposition | 1 hour |

**Key Capabilities**: IEEE 2414-2020, jitter decomposition, eye diagram analysis

---

### 14 - Power Analysis

**Level**: Advanced | **Demos**: 2

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| dcdc_efficiency_demo.py | DC/DC converter characterization | Efficiency vs load, switching losses | 1 hour |
| ripple_analysis_demo.py | Power supply quality | Ripple, harmonics, power factor (IEEE 1459) | 1 hour |

**Key Capabilities**: IEEE 1459-2010, power quality, converter efficiency

---

### 15 - Signal Integrity

**Level**: Advanced | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| tdr_impedance_demo.py | Time-domain reflectometry | Impedance discontinuities, trace faults | 1 hour |
| sparams_demo.py | S-parameter extraction | Return loss, insertion loss, crosstalk | 1 hour |
| setup_hold_timing_demo.py | Digital timing validation | Setup/hold violations, clock-to-Q | 30 min |

**Key Capabilities**: IEEE 181-2011, TDR, S-parameters, digital timing

---

### 16 - EMC Compliance

**Level**: Expert | **Demos**: 1

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_emc_demo.py | CISPR 32, IEC 61000 testing | Conducted/radiated emissions, immunity | 2 hours |

**Key Capabilities**: CISPR 16/32, IEC 61000-4-x, EMI/EMC compliance testing

---

### 17 - Signal Reverse Engineering

**Level**: Expert | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| comprehensive_re.py | Complete unknown signal RE workflow | Full capture to documentation pipeline | 3 hours |
| exploratory_analysis.py | Statistical signal characterization | Pattern discovery, field detection | 2 hours |
| reverse_engineer_tool.py | Complete tool integration | Multi-tool workflow automation | 3 hours |

**Key Capabilities**: End-to-end RE workflows, hypothesis tracking, confidence scoring

---

### 18 - Advanced Inference

**Level**: Expert | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| bayesian_inference_demo.py | Probabilistic protocol inference | Bayesian field detection, uncertainty | 2 hours |
| active_learning_demo.py | ML-assisted protocol learning | Feature extraction, classification | 2 hours |
| protocol_dsl_demo.py | Domain-specific language generation | Protocol grammar extraction | 2 hours |

**Key Capabilities**: ML/AI-based inference, active learning, DSL generation

---

### 19 - Complete Workflows

**Level**: Expert | **Demos**: 3

| Demo | Description | Capabilities | Time |
|------|-------------|--------------|------|
| unknown_signal_workflow.py | Capture to Wireshark dissector | Full unknown protocol workflow | 2 hours |
| network_analysis_workflow.py | Network protocol RE pipeline | PCAP to protocol specification | 2 hours |
| automotive_full_workflow.py | CAN to DBC generation | Raw CAN to production DBC file | 2 hours |

**Key Capabilities**: Production workflows, export automation, tool chaining

---

## Search by Capability

### Protocol Decoding
- **UART**: 05_protocol_decoding/comprehensive_protocol_demo.py
- **SPI**: 05_protocol_decoding/comprehensive_protocol_demo.py
- **I2C**: 05_protocol_decoding/comprehensive_protocol_demo.py
- **CAN**: 08_automotive_protocols/, 09_automotive/, 19_complete_workflows/automotive_full_workflow.py
- **LIN**: 08_automotive_protocols/lin_demo.py
- **FlexRay**: 08_automotive_protocols/flexray_demo.py
- **Manchester**: 04_serial_protocols/manchester_demo.py
- **I2S**: 04_serial_protocols/i2s_demo.py
- **JTAG**: 04_serial_protocols/jtag_demo.py
- **1-Wire**: 04_serial_protocols/onewire_demo.py
- **SWD**: 04_serial_protocols/swd_demo.py
- **USB**: 04_serial_protocols/usb_demo.py

### File Format Loading
- **VCD**: 02_file_format_io/vcd_loader_demo.py
- **CSV**: 01_waveform_analysis/all_output_formats.py
- **HDF5**: 01_waveform_analysis/all_output_formats.py
- **MATLAB**: 01_waveform_analysis/all_output_formats.py
- **PCAP**: 06_udp_packet_analysis/comprehensive_udp_analysis.py
- **Custom**: 03_custom_daq/simple_loader.py

### Reverse Engineering
- **CRC Recovery**: 07_protocol_inference/crc_reverse_demo.py
- **State Machines**: 07_protocol_inference/state_machine_learning.py
- **Wireshark Dissectors**: 07_protocol_inference/wireshark_dissector_demo.py
- **Unknown Signals**: 17_signal_reverse_engineering/comprehensive_re.py
- **Bayesian Inference**: 18_advanced_inference/bayesian_inference_demo.py
- **Active Learning**: 18_advanced_inference/active_learning_demo.py
- **Protocol DSL**: 18_advanced_inference/protocol_dsl_demo.py

### Signal Analysis
- **Waveform**: 01_waveform_analysis/comprehensive_wfm_analysis.py
- **Spectral**: 12_spectral_compliance/comprehensive_spectral_demo.py
- **Jitter**: 13_jitter_analysis/ddj_dcd_demo.py, 13_jitter_analysis/bathtub_curve_demo.py
- **Power**: 14_power_analysis/dcdc_efficiency_demo.py, 14_power_analysis/ripple_analysis_demo.py
- **Signal Integrity**: 15_signal_integrity/tdr_impedance_demo.py, 15_signal_integrity/sparams_demo.py
- **EMC**: 16_emc_compliance/comprehensive_emc_demo.py
- **Mixed Signal**: 11_mixed_signal/comprehensive_mixed_signal_demo.py
- **Timing**: 10_timing_measurements/ieee_181_pulse_demo.py, 15_signal_integrity/setup_hold_timing_demo.py

### Automotive
- **OBD-II**: 09_automotive/comprehensive_automotive_demo.py
- **UDS**: 09_automotive/comprehensive_automotive_demo.py
- **J1939**: 09_automotive/comprehensive_automotive_demo.py
- **CAN**: 08_automotive_protocols/, 09_automotive/, 19_complete_workflows/automotive_full_workflow.py
- **LIN**: 08_automotive_protocols/lin_demo.py
- **FlexRay**: 08_automotive_protocols/flexray_demo.py
- **DBC Generation**: 19_complete_workflows/automotive_full_workflow.py

### Performance
- **Memory-Efficient**: 03_custom_daq/chunked_loader.py
- **Streaming**: 03_custom_daq/optimal_streaming_loader.py
- **Large Files**: 03_custom_daq/chunked_loader.py

---

## Search by IEEE Standard

### IEEE 181-2011 (Pulse Measurement)
- **10_timing_measurements/ieee_181_pulse_demo.py** - Rise/fall time, overshoot, slew rate
- **15_signal_integrity/setup_hold_timing_demo.py** - Digital timing parameters
- **15_signal_integrity/tdr_impedance_demo.py** - Time-domain reflectometry

### IEEE 1241-2010 (ADC Testing)
- **12_spectral_compliance/comprehensive_spectral_demo.py** - SNR, SINAD, THD, SFDR, ENOB

### IEEE 1459-2010 (Power Measurement)
- **14_power_analysis/ripple_analysis_demo.py** - Active/reactive power, harmonics, power factor
- **14_power_analysis/dcdc_efficiency_demo.py** - Power supply characterization

### IEEE 2414-2020 (Jitter)
- **13_jitter_analysis/ddj_dcd_demo.py** - Data-dependent jitter, duty cycle distortion
- **13_jitter_analysis/bathtub_curve_demo.py** - TIE, period jitter, RJ/DJ decomposition

### CISPR/IEC (EMC)
- **16_emc_compliance/comprehensive_emc_demo.py** - CISPR 16/32, IEC 61000-4-x

---

## Running Demos

### Basic Usage

Each demo is self-contained and can be run directly:

```bash
# Navigate to demos directory
cd demos

# Run a specific demo
python 01_waveform_analysis/comprehensive_wfm_analysis.py

# Run with validation
python 01_waveform_analysis/comprehensive_wfm_analysis.py --validate

# Most demos accept --help for options
python 07_protocol_inference/crc_reverse_demo.py --help
```

### Running All Demos in a Category

```bash
# Run all waveform analysis demos
for demo in 01_waveform_analysis/*.py; do
    [ -f "$demo" ] && [ "$demo" != "*generate_demo_data.py" ] && python "$demo"
done
```

### Validation

```bash
# Validate all demos at once
python demos/validate_all_demos.py

# Check specific category
python demos/validate_all_demos.py --category 01_waveform_analysis
```

### Data Generation

Most demos generate synthetic data automatically. Some have separate data generation scripts:

```bash
# Generate all demo data upfront
python demos/generate_all_demo_data.py

# Generate data for specific category
python demos/01_waveform_analysis/generate_demo_data.py
```

---

## Demo Structure

All demonstrations follow the `BaseDemo` pattern:

```python
from demos.common import BaseDemo, ValidationSuite

class MyDemo(BaseDemo):
    name = "My Feature Demo"
    description = "Demonstrates feature X with Y"
    category = "protocols"

    def generate_data(self):
        """Generate or load test data."""
        # Synthetic data generation
        pass

    def run_analysis(self):
        """Perform analysis."""
        # Core demonstration logic
        pass

    def validate_results(self, suite: ValidationSuite):
        """Validate outputs."""
        suite.check_equal("Result", self.result, expected_value)
        suite.check_range("Metric", self.metric, 0.9, 1.1)

if __name__ == "__main__":
    MyDemo().main()
```

**Benefits**:
- Consistent structure across all demos
- Automatic error handling and reporting
- Built-in validation framework
- Formatted console output
- Easy to adapt for your use case

---

## Common Utilities

The `demos/common/` directory provides shared utilities:

| Module | Purpose |
|--------|---------|
| base_demo.py | BaseDemo abstract class |
| validation.py | ValidationSuite for self-testing |
| formatting.py | Console output formatting |
| plotting.py | Plotting utilities (future) |

All demos can leverage these utilities for consistency.

---

## Examples vs Demos

This repository contains both:

- **`examples/`** (6 files) - High-level workflow examples from README
  - side_channel_analysis_demo.py
  - ml_signal_classification_demo.py
  - wireshark_dissector_demo.py
  - dbc_generation_example.py
  - lin_analysis_example.py
  - web_dashboard_example.py

- **`demos/`** (33+ files) - Comprehensive feature demonstrations
  - Organized by complexity and domain
  - Follow BaseDemo pattern
  - Include validation and documentation
  - Cover every Oscura capability

**Use `examples/`** for quick overview of core workflows.
**Use `demos/`** for in-depth learning and feature exploration.

---

## Contributing Demos

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Demo contributions should:

1. Inherit from `BaseDemo`
2. Generate synthetic data (no external dependencies)
3. Include validation checks
4. Have clear documentation
5. Pass `./scripts/check.sh` quality checks
6. Update this README with demo entry

---

## Support

- **Issues**: [GitHub Issues](https://github.com/oscura-re/oscura/issues)
- **Discussions**: [GitHub Discussions](https://github.com/oscura-re/oscura/discussions)
- **Documentation**: [Main README](../README.md) | [API Docs](../docs/api/)

---

**Oscura** - _Illuminate what others obscure._
