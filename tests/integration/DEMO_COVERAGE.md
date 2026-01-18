# Integration Test Coverage by Demos

**Purpose**: Map which demos validate which integration test scenarios to prevent redundancy.

**Last Updated**: 2026-01-15
**Status**: Active

---

## Summary

This document maps Oscura demos to the integration test scenarios they cover. Use this to determine if a new integration test is needed or if a demo already validates the workflow.

**Key Principle**: Demos ARE integration tests. Only write integration tests for edge cases NOT covered by demos.

---

## Demo 01: Waveform Analysis

**Location**: `demos/01_waveform_analysis/comprehensive_wfm_analysis.py`

**Purpose**: Comprehensive waveform loading, analysis, and export

### Replaces These Integration Tests

❌ DELETED:

- `test_wfm_to_analysis.py::test_wfm_load_to_fft` - Basic WFM→FFT workflow
- `test_wfm_to_analysis.py::test_wfm_load_to_measurements` - Basic measurements
- `test_wfm_to_analysis.py::test_wfm_to_psd` - Power spectral density
- `test_wfm_to_analysis.py::test_wfm_to_thd` - Total harmonic distortion

### Validated by Demo

- ✅ WFM loading from Tektronix files (.wfm format)
- ✅ FFT computation with peak detection
- ✅ Spectral analysis (PSD, THD, SINAD, SNR)
- ✅ Time domain measurements (RMS, amplitude, frequency)
- ✅ All export formats:
  - NumPy (.npz)
  - CSV (.csv)
  - HDF5 (.h5)
  - MATLAB (.mat)
  - JSON (.json)
  - HTML report (.html)

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Malformed WFM file handling
- Truncated WFM files
- Very large WFM files (>1 GB) - memory management
- Corrupted header recovery
- Multi-channel WFM files with missing channels

**Test Location**: `tests/integration/test_wfm_to_analysis.py` (kept edge cases only)

---

## Demo 02: Custom DAQ Loading

**Location**: `demos/02_custom_daq/`

**Files**:

- `simple_loader.py` - Load all data into memory
- `optimal_streaming_loader.py` - Constant memory streaming
- `chunked_loader.py` - Chunk-based statistics

**Purpose**: Demonstrate configurable loader patterns for custom data acquisition formats

### Replaces These Integration Tests

❌ DELETED:

- `test_config_driven.py` - Partial overlap (~200 LOC of basic YAML loading)

### Validated by Demo

- ✅ YAML configuration parsing
- ✅ Custom format loading (CSV-like DAQ data)
- ✅ Three loading patterns:
  1. Simple (load all)
  2. Optimal streaming (constant memory)
  3. Chunked processing (statistics)
- ✅ Memory management with large files
- ✅ Real-time statistics computation

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Invalid YAML configuration handling
- Malformed custom format data
- Validation rule enforcement
- Complex multi-format configurations

**Test Location**: `tests/integration/test_config_driven.py` (kept validation tests)

---

## Demo 03: UDP Packet Analysis

**Location**: `demos/03_udp_packet_analysis/`

**Purpose**: Reverse engineering network protocols from packet captures

### Replaces These Integration Tests

❌ CONSIDERED FOR DELETION:

- `test_pcap_to_inference.py` - Partial overlap with packet analysis

### Validated by Demo

- ✅ PCAP file loading
- ✅ UDP packet extraction
- ✅ Packet field inference (headers, payloads, checksums)
- ✅ Protocol pattern detection
- ✅ Wireshark dissector generation

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Malformed PCAP files
- Fragmented UDP packets
- Large packet captures (>1 GB)
- Non-UDP protocols (TCP, ICMP)

**Test Location**: `tests/integration/test_pcap_to_inference.py` (review for consolidation)

---

## Demo 04: Signal Reverse Engineering

**Location**: `demos/04_signal_reverse_engineering/`

**Files**:

- `comprehensive_re.py` - Complete RE workflow
- `reverse_engineer_tool.py` - Interactive RE tool
- `exploratory_analysis.py` - Exploratory workflow

**Purpose**: Reverse engineer unknown signals from waveforms

### Replaces These Integration Tests

✅ VALIDATES:

- Signal characterization workflows
- Pattern detection and correlation
- State machine learning
- Timing analysis

### Validated by Demo

- ✅ Signal loading from multiple formats
- ✅ Statistical analysis (distribution, entropy)
- ✅ Pattern detection (repeating sequences)
- ✅ Correlation analysis (timing relationships)
- ✅ State machine inference
- ✅ Interactive visualization

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Very noisy signals (low SNR)
- Complex multi-signal correlation
- Long-duration signals (>1 hour)

**Test Location**: None needed - demo comprehensive

---

## Demo 05: Protocol Decoding

**Location**: `demos/05_protocol_decoding/comprehensive_protocol_demo.py`

**Purpose**: Decode 16+ transport protocols from digital waveforms

### Replaces These Integration Tests

❌ DELETED/MERGED:

- `test_end_to_end_workflows.py::test_uart_complete_workflow` - UART decoding
- `test_binary_to_protocol.py` - Partial overlap with protocol decoding
- `test_module_interactions.py` - Protocol decoder interactions

### Validated by Demo

- ✅ UART decoding (8N1, configurable baud rates)
- ✅ SPI decoding (CPOL/CPHA modes)
- ✅ I2C decoding (7-bit/10-bit addressing)
- ✅ CAN bus decoding (standard/extended IDs)
- ✅ Multi-frame message assembly
- ✅ CRC validation
- ✅ Frame error detection

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Protocol decoder error handling (framing errors)
- CRC failure recovery
- Cross-protocol analysis (UART + I2C simultaneous)
- Very noisy signals (bit errors)

**Test Location**: `tests/integration/test_binary_to_protocol.py` (consolidate to edge cases)

---

## Demo 06: Spectral Compliance

**Location**: `demos/06_spectral_compliance/`

**Purpose**: IEEE-compliant spectral analysis and measurements

### Replaces These Integration Tests

✅ VALIDATES:

- FFT computation workflows
- Spectral measurements (THD, SNR, SINAD)
- IEEE compliance validation

### Validated by Demo

- ✅ IEEE 181-2011 compliant measurements
- ✅ IEEE 1241-2010 ADC characterization
- ✅ FFT with windowing (Hann, Hamming, Blackman)
- ✅ THD+N measurements
- ✅ SNR, SINAD, ENOB calculations
- ✅ Compliance reporting

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Non-IEEE compliant signals
- Edge cases for specific IEEE standards
- Large-scale batch compliance testing

**Test Location**: None needed - demo comprehensive

---

## Demo 07: Mixed-Signal Analysis

**Location**: `demos/07_mixed_signal/comprehensive_mixed_signal_demo.py`

**Purpose**: Analyze mixed analog/digital signals

### Replaces These Integration Tests

✅ VALIDATES:

- Digital signal analysis workflows
- Eye diagram generation
- Jitter measurements

### Validated by Demo

- ✅ Eye diagram generation
- ✅ Jitter analysis (TIE, period jitter)
- ✅ Signal integrity measurements
- ✅ Digital edge detection
- ✅ Timing correlation

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Very high-speed signals (>10 Gbps)
- Complex modulation schemes
- Multi-channel digital buses

**Test Location**: None needed - demo comprehensive

---

## Demo 08: Automotive Analysis

**Location**: `demos/08_automotive/comprehensive_automotive_demo.py`

**Purpose**: Automotive protocol analysis (CAN, OBD-II, UDS)

### Replaces These Integration Tests

✅ VALIDATES:

- CAN bus decoding workflows
- OBD-II message parsing
- UDS diagnostic protocols

### Validated by Demo

- ✅ CAN bus decoding (standard/extended frames)
- ✅ OBD-II PID decoding
- ✅ UDS diagnostic service decoding
- ✅ Multi-frame message assembly
- ✅ Real vehicle data analysis

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- CAN error frames
- Bus-off recovery
- FlexRay protocol support
- LIN protocol support

**Test Location**: None needed for CAN/OBD - demo comprehensive

---

## Demo 09: EMC Compliance

**Location**: `demos/09_emc_compliance/comprehensive_emc_demo.py`

**Purpose**: Electromagnetic compatibility testing and compliance

### Replaces These Integration Tests

✅ VALIDATES:

- EMC measurement workflows
- Emissions analysis
- Compliance checking

### Validated by Demo

- ✅ Conducted emissions measurements
- ✅ Radiated emissions analysis
- ✅ Harmonic analysis
- ✅ Power quality measurements
- ✅ IEEE 1459 power analysis
- ✅ Compliance reporting (FCC Part 15, CISPR)

### Still Needs Integration Test

⚠️ Edge cases NOT in demo:

- Near-field EMC measurements
- Time-domain EMC analysis
- Complex test setups (multiple probes)

**Test Location**: None needed - demo comprehensive

---

## Integration Test Consolidation Summary

### Tests DELETED (covered by demos)

|Test File|LOC|Reason|Demo Coverage|
|---|---|---|---|
|`test_chunked_consistency.py`|434|Tests NumPy FFT, not Oscura|N/A (vendor lib)|
|`test_wfm_to_analysis.py` (partial)|236|Basic WFM workflows|Demo 01, 06|

**Total Deleted**: 670 LOC

### Tests MERGED/CONSOLIDATED

|Original Files|New File|LOC Reduction|Reason|
|---|---|---|---|
|`test_end_to_end_workflows.py` + `test_module_interactions.py`|`test_integration_workflows.py`|~1,006|Overlapping workflows|

**Total Reduction Target**: 2,221 LOC (47%)

### Tests KEPT (edge cases only)

|Test File|LOC|Reason|Edge Cases|
|---|---|---|---|
|`test_wfm_to_analysis.py`|231|Edge case handling|Malformed files, memory limits|
|`test_config_driven.py`|~450|Validation rules|Complex configs, error handling|
|`test_real_captures.py`|352|Vendor quirks|Tektronix format edge cases|

---

## Decision Matrix: New Test Needed?

Use this flowchart when considering a new integration test:

```
Does a demo cover this workflow?
  │
  ├─ YES → Is it an edge case not in demo?
  │        │
  │        ├─ YES → Integration test ✅
  │        └─ NO → Don't create test ❌ (redundant)
  │
  └─ NO → Is it a basic workflow?
           │
           ├─ YES → Add to demo first, then test edge cases
           └─ NO → Integration test ✅ (if 2+ modules)
```

---

## Maintenance

**When adding new demos**:

1. Update this document with demo coverage
2. Review existing integration tests for redundancy
3. Delete redundant tests
4. Keep only edge cases NOT in demo

**When adding new integration tests**:

1. Check this document for demo coverage
2. Verify edge case NOT in demo
3. Update this document if test adds unique coverage
4. Follow TEST_CHARTER.md quality gate

---

## References

- **Test Charter**: `TEST_CHARTER.md` - Integration test scope definition
- **Test Optimization Plan**: `TEST_SUITE_OPTIMIZATION_PLAN.md` - Detailed consolidation plan
- **Validation Matrix**: `demos/VALIDATION_MATRIX.md` - Demo self-validation status
- **Testing Strategy**: `tests/README.md` - Complete testing approach

---

**Maintain this document**: Update when demos are added/modified or integration tests change.
