# Choosing the Right Features

Guide to selecting the appropriate Oscura APIs for your use case.

## Quick Reference

| I need to... | Use this API | Demo |
|---|---|---|
| Load a file | [Loader API](../api/loader.md) | [File Format Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/02_file_format_io) |
| Measure waveforms | [Analysis API](../api/analysis.md) | [Waveform Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/01_waveform_analysis) |
| Decode protocols | [Analysis API](../api/analysis.md) | [Protocol Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/04_serial_protocols) |
| Analyze power | [Power API](../api/power-analysis.md) | [Power Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/14_power_analysis) |
| Test compliance | [EMC API](../api/emc-compliance.md) | [Compliance Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/12_spectral_compliance) |
| Reverse engineer | [Workflows API](../api/workflows.md) | [RE Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/17_signal_reverse_engineering) |
| Build pipelines | [Pipelines API](../api/pipelines.md) | [Workflows Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/19_complete_workflows) |
| Export results | [Export API](../api/export.md) | [Export Demo](https://github.com/lair-click-bats/oscura/tree/main/demos/01_waveform_analysis) |

---

## By Domain

### 🔌 **Electronics / Hardware**

**Analog circuits** (audio, power, RF):

- [Power Analysis API](../api/power-analysis.md) - AC/DC, ripple, efficiency
- [Component Analysis API](../api/component-analysis.md) - TDR, impedance, capacitance

**Digital circuits** (logic, protocols):

- [Analysis API](../api/analysis.md) - Protocol decoders (UART, SPI, I2C)
- Auto-decode with `osc.auto_decode(trace)`

**Mixed-signal**:

- Both analog and digital analyzers work together
- Demo: [Mixed Signal](https://github.com/lair-click-bats/oscura/tree/main/demos/11_mixed_signal)

---

## By Task

### 📊 **Measurement & Analysis**

**Basic measurements** (amplitude, frequency, rise time):

```python
osc.analyze(trace)  # Returns dict of measurements
```

→ [Analysis API](../api/analysis.md)

**Spectral analysis** (FFT, THD, SNR, ENOB):

```python
osc.quick_spectral(trace)  # Returns IEEE 1241 metrics
```

→ [Analysis API](../api/analysis.md)

**Power analysis** (AC/DC, ripple, PF):

```python
osc.power_analysis(trace)
```

→ [Power API](../api/power-analysis.md)

---

### 🔍 **Protocol Work**

**Known protocols** (UART, SPI, I2C, CAN):

```python
osc.auto_decode(trace)  # Auto-detect and decode
```

→ [Analysis API](../api/analysis.md)

**Unknown protocols** (reverse engineering):

```python
osc.workflows.reverse_engineer_signal(trace)
```

→ [Workflows API](../api/workflows.md)

**Automotive** (CAN, OBD-II, J1939):

- Special decoders for automotive protocols
- Demo: [Automotive](https://github.com/lair-click-bats/oscura/tree/main/demos/09_automotive)

---

### ✅ **Compliance & Standards**

**IEEE 1241-2010** (ADC testing):

- `quick_spectral()` provides THD, SNR, SINAD, ENOB
- Demo: [Spectral Compliance](https://github.com/lair-click-bats/oscura/tree/main/demos/12_spectral_compliance)

**IEEE 2414-2020** (Jitter):

- Jitter analysis functions in [Analysis API](../api/analysis.md)
- Demo: [Jitter Analysis](https://github.com/lair-click-bats/oscura/tree/main/demos/13_jitter_analysis)

**CISPR 16 / IEC 61000** (EMC):

- [EMC Compliance API](../api/emc-compliance.md)
- Demo: [EMC Testing](https://github.com/lair-click-bats/oscura/tree/main/demos/16_emc_compliance)

---

## By File Format

| Format | Loader | Demo |
|---|---|---|
| Tektronix .wfm | `osc.load_wfm()` | [Waveform](https://github.com/lair-click-bats/oscura/tree/main/demos/01_waveform_analysis) |
| Sigrok .sr | `osc.load_sigrok()` | [File Format](https://github.com/lair-click-bats/oscura/tree/main/demos/02_file_format_io) |
| VCD | `osc.load_vcd()` | [File Format](https://github.com/lair-click-bats/oscura/tree/main/demos/02_file_format_io) |
| PCAP | `osc.load_pcap()` | [UDP Analysis](https://github.com/lair-click-bats/oscura/tree/main/demos/06_udp_packet_analysis) |
| CAN (BLF/ASC) | `osc.load_can()` | [Automotive](https://github.com/lair-click-bats/oscura/tree/main/demos/08_automotive_protocols) |
| CSV/HDF5/NumPy | `osc.load()` | [File Format](https://github.com/lair-click-bats/oscura/tree/main/demos/02_file_format_io) |

**Auto-detect**: `osc.load(filename)` detects format automatically

Full list: [Loader API](../api/loader.md)

---

## By Experience Level

### 🟢 **Beginner** - High-level convenience functions

**Start here**:

```python
import oscura as osc

# Load and analyze in 3 lines
trace = osc.load("file.wfm")
result = osc.quick_spectral(trace)
print(f"SNR: {result.snr_db} dB")
```

**APIs to use**:

- `osc.load()` - Auto-detect file format
- `osc.analyze()` - All basic measurements
- `osc.quick_spectral()` - Spectral analysis
- `osc.auto_decode()` - Protocol detection

**Demos**: [01](https://github.com/lair-click-bats/oscura/tree/main/demos/01_waveform_analysis), [04](https://github.com/lair-click-bats/oscura/tree/main/demos/04_serial_protocols), [12](https://github.com/lair-click-bats/oscura/tree/main/demos/12_spectral_compliance)

---

### 🟡 **Intermediate** - Specific analyzers

**More control**:

```python
from oscura.analyzers import spectral, protocols

# Use specific analyzers
fft_result = spectral.compute_fft(trace)
uart_frames = protocols.decode_uart(trace, baud_rate=115200)
```

**APIs to use**:

- Individual analyzer modules
- Specific protocol decoders
- Custom parameters

**Demos**: [05](https://github.com/lair-click-bats/oscura/tree/main/demos/05_protocol_decoding), [08](https://github.com/lair-click-bats/oscura/tree/main/demos/08_automotive_protocols), [13](https://github.com/lair-click-bats/oscura/tree/main/demos/13_jitter_analysis)

---

### 🔴 **Advanced** - Low-level APIs and pipelines

**Full control**:

```python
from oscura.loaders.configurable import ConfigurableLoader
from oscura.core.pipeline import Pipeline

# Build custom pipelines
loader = ConfigurableLoader(chunk_size=1000000)
pipeline = Pipeline([filter_step, analyze_step, export_step])
```

**APIs to use**:

- [Expert API](../api/expert-api.md) - Low-level access
- [Pipelines API](../api/pipelines.md) - Custom workflows
- [Session Management](../api/session-management.md) - Multi-step analysis

**Demos**: [03](https://github.com/lair-click-bats/oscura/tree/main/demos/03_custom_daq), [19](https://github.com/lair-click-bats/oscura/tree/main/demos/19_complete_workflows)

---

## Decision Tree

```
What's your goal?

├─ Analyze a single file
│  ├─ Oscilloscope → Analysis API
│  ├─ Logic analyzer → Analysis API (protocol decoders)
│  └─ Network capture → Analysis API (PCAP)
│
├─ Process many files
│  └─ Pipelines API + batch processing
│
├─ Reverse engineer unknown signal
│  └─ Workflows API (reverse_engineer_signal)
│
├─ Check compliance
│  ├─ IEEE standards → quick_spectral()
│  └─ EMC → EMC Compliance API
│
└─ Build custom workflow
   └─ Expert API + Pipelines API
```

---

## Next Steps

- **Try it**: Start with [Quick Start](quick-start.md)
- **Learn patterns**: Read [Common Workflows](workflows.md)
- **Understand design**: See [Core Concepts](concepts.md)
- **Deep dive**: Browse [API Reference](../api/index.md)
