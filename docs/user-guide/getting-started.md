# Getting Started with Oscura

Oscura is a comprehensive hardware reverse engineering framework that automates complete workflows from oscilloscope captures to protocol dissectors. This guide will help you get started quickly.

## Installation

### Prerequisites

- **Python:** 3.12 or higher
- **Operating System:** Linux, macOS, or Windows
- **Recommended:** 8GB RAM minimum, 16GB+ for large captures

### Option 1: PyPI Install (Stable Release)

```bash
pip install oscura
```

This installs the stable release from PyPI with core dependencies.

### Option 2: Development Install (Recommended)

For the latest features and full development environment:

```bash
# Clone repository
git clone https://github.com/oscura-re/oscura.git
cd oscura

# Run automated setup
./scripts/setup.sh
```

The setup script automatically:

- Installs Python dependencies via `uv`
- Sets up pre-commit hooks
- Configures development tools (ruff, mypy, pytest)
- Validates installation

### Option 3: Install with Extras

```bash
# Install with specific feature sets
pip install oscura[reporting]  # HTML/PDF report generation
pip install oscura[ml]          # Machine learning features
pip install oscura[all]         # All optional features
```

### Verify Installation

```bash
# Check Oscura version
python -m oscura --version

# Run basic health check
python -c "import oscura; print(oscura.__version__)"

# Run test suite
./scripts/test.sh --fast
```

## First Steps

### 1. Load a Waveform File

Oscura supports multiple oscilloscope formats:

```python
from oscura.loaders import load_waveform

# Auto-detect format (supports: WFM, WAV, CSV, VCD, etc.)
waveform = load("capture.wfm")

# Access waveform data
print(f"Sample rate: {waveform.sample_rate} Hz")
print(f"Duration: {waveform.duration} seconds")
print(f"Channels: {len(waveform.channels)}")
```

**Supported formats:**

- Tektronix (.wfm, .isf)
- Rigol (.wfm)
- LeCroy (.trc, .wvs)
- Siglent (.bin)
- Generic CSV/WAV/VCD

### 2. Decode a Known Protocol

For protocols with known specifications:

```python
from oscura.analyzers.protocols import UARTDecoder

# Configure decoder
decoder = UARTDecoder(
    baud_rate=115200,
    data_bits=8,
    parity='N',
    stop_bits=1
)

# Decode waveform
messages = decoder.decode(waveform)

# Display results
for msg in messages:
    print(f"{msg.timestamp:.6f}s: {msg.data.hex()} ({msg.data.decode()})")
```

**Built-in protocol decoders:**

- Serial: UART, SPI, I2C, 1-Wire, JTAG, SWD
- Automotive: CAN, CAN-FD, LIN, FlexRay
- Industrial: Modbus, PROFIBUS
- Others: USB, Manchester, HDLC

### 3. Reverse Engineer Unknown Protocol

Oscura's differential analysis automatically infers protocol structure:

```python
from oscura.sessions import BlackBoxSession

# Create analysis session
session = BlackBoxSession(name="Unknown Device RE")

# Add captures from different device states
session.add_recording("idle", "captures/idle.bin")
session.add_recording("button_press", "captures/button.bin")
session.add_recording("sensor_triggered", "captures/sensor.bin")

# Differential analysis - compare states
diff = session.compare("idle", "button_press")
print(f"Detected {len(diff.changed_fields)} changing fields")

# Automatic protocol specification inference
spec = session.generate_protocol_spec()

# Export Wireshark dissector
session.export_results("dissector", "protocol.lua")

# Generate report
session.export_results("report", "analysis_report.html")
```

**What differential analysis finds:**

- Message structure and field boundaries
- Static vs dynamic fields
- Checksums and CRCs (auto-recovery)
- Sequence numbers and counters
- Encrypted/compressed regions (entropy analysis)

### 4. Analyze CAN Bus Traffic

Automotive and industrial protocol analysis:

```python
from oscura.sessions import CANSession

# Create CAN analysis session
session = CANSession(
    bitrate=500000,
    fd_mode=False  # Use True for CAN-FD
)

# Load capture file (BLF, ASC, PCAP, etc.)
session.load("vehicle_capture.blf")

# Automatic signal extraction
signals = session.extract_signals()

# Generate DBC file (no manual signal definition needed)
session.export_dbc("vehicle_protocol.dbc")

# Analyze message patterns
patterns = session.find_patterns()
print(f"Found {len(patterns.periodic_messages)} periodic messages")
print(f"Found {len(patterns.event_driven_messages)} event-driven messages")
```

### 5. Generate Protocol Artifacts

Export inferred protocols to multiple formats:

```python
from oscura.export import ExportManager

# Create exporter
exporter = ExportManager(session)

# Wireshark dissector (Lua)
exporter.wireshark_dissector(
    output="protocol.lua",
    protocol_name="custom_proto",
    validate=True  # Validate Lua syntax
)

# Scapy layer (Python)
exporter.scapy_layer(
    output="protocol_layer.py",
    base_class="Packet"
)

# Kaitai Struct (multi-language parser)
exporter.kaitai_struct(
    output="protocol.ksy"
)

# C/C++ header
exporter.c_header(
    output="protocol.h",
    include_guards=True
)

# Documentation
exporter.markdown_spec(
    output="PROTOCOL_SPEC.md",
    include_examples=True
)
```

## Common Workflows

### Signal Quality Analysis

```python
from oscura.workflows import signal_integrity_analysis

# Comprehensive signal quality check
results = signal_integrity_analysis(
    waveform=waveform,
    checks=[
        "rise_time",
        "fall_time",
        "overshoot",
        "undershoot",
        "jitter",
        "eye_diagram"
    ]
)

# Check compliance
if results.meets_spec("USB 2.0"):
    print("Signal meets USB 2.0 specification")
else:
    print(f"Issues: {results.violations}")
```

### Power Analysis / Side-Channel

```python
from oscura.workflows import power_analysis

# Correlation Power Analysis
results = power_analysis(
    traces_file="power_traces.npy",
    plaintexts_file="plaintexts.npy",
    algorithm="AES-128",
    target_byte=0
)

# Display key recovery results
print(f"Recovered key byte: 0x{results.key_byte:02x}")
print(f"Confidence: {results.confidence:.2%}")
print(f"Traces required: {results.traces_needed}")
```

### Batch Processing

```python
# NOTE: Use workflows or manual iteration in v0.6
# from oscura.workflows import batch_analyze

# Process multiple captures in parallel
results = batch_analyze(
    captures_dir="./captures/",
    pattern="*.wfm",
    analysis_func=lambda w: UARTDecoder(115200).decode(w),
    parallel=True,
    num_workers=4
)

# Aggregate results
summary = results.aggregate(
    metrics=["message_count", "error_rate", "unique_patterns"]
)
```

## Configuration

### Project Configuration File

Create `oscura.yaml` in your project directory:

```yaml
# oscura.yaml - Project configuration
project:
  name: "My RE Project"
  version: "1.0.0"

defaults:
  # Default protocol settings
  uart:
    baud_rate: 115200
    data_bits: 8
    parity: 'N'
    stop_bits: 1

  can:
    bitrate: 500000
    fd_mode: false

  # Analysis settings
  analysis:
    confidence_threshold: 0.8
    auto_crc_recovery: true
    detect_encryption: true

  # Export settings
  export:
    validate_dissectors: true
    include_test_data: true
    format: "both"  # wireshark + scapy

# Custom protocol definitions
protocols:
  my_device:
    base: uart
    baud_rate: 9600
    message_format:
      header: 0xAA55
      length: 2
      payload: variable
      checksum: crc16_ccitt
```

Load configuration:

```python
from oscura.core.config import load_config

config = load_config("oscura.yaml")
session = BlackBoxSession(config=config)
```

### Environment Variables

```bash
# Control Oscura behavior via environment
export OSCURA_LOG_LEVEL=DEBUG
export OSCURA_CACHE_DIR=/tmp/oscura_cache
export OSCURA_MAX_WORKERS=8
export OSCURA_ENABLE_GPU=true
```

## Next Steps

Now that you have Oscura installed and understand the basics:

1. **[Common Workflows](workflows.md)** - Detailed workflow examples
2. **[Tutorials](../tutorials/)** - Step-by-step guides for specific tasks
3. **[API Reference](../api/)** - Complete API documentation
4. **[Protocol Catalog](../protocols/)** - Supported protocols and coverage

## Getting Help

- **Documentation:** [https://oscura-re.github.io/oscura](https://oscura-re.github.io/oscura)
- **GitHub Issues:** [https://github.com/oscura-re/oscura/issues](https://github.com/oscura-re/oscura/issues)
- **FAQ:** [Frequently Asked Questions](../faq/index.md)
- **Examples:** See `examples/` directory in repository

## Troubleshooting

### Common Issues

**Import Error: "No module named oscura"**

```bash
# Ensure Oscura is installed
pip install oscura

# Or use development install
pip install -e .
```

**Memory Error with Large Files**

```bash
# Use streaming mode for large captures
from oscura.loaders import load_waveform_streaming

stream = load_waveform_streaming("large_file.wfm", chunk_size=1000000)
for chunk in stream:
    process(chunk)
```

**Decoder Not Finding Protocol**

```python
# Increase search range for auto-detection
decoder = UARTDecoder.auto_detect(
    waveform,
    baud_rate_range=(9600, 921600),
    confidence_threshold=0.7  # Lower threshold
)
```

See [Error Codes Reference](../error-codes.md) for complete error documentation.
