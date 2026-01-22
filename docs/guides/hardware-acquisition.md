# Hardware Acquisition Guide

**Version**: 0.4.0
**Last Updated**: 2026-01-20
**Status**: Phase 2 Placeholder (Future Implementation)

Guide to acquiring signal data directly from hardware devices using Oscura's unified Source protocol.

---

## Overview

Oscura's hardware acquisition layer provides direct integration with:

- **SocketCAN**: Linux CAN interfaces for automotive/industrial protocols
- **Saleae Logic**: Logic analyzer integration for digital signal capture
- **PyVISA**: Oscilloscope and test equipment support

**Current Status**: Hardware acquisition is a Phase 2 feature (placeholder in v0.3.0). This guide documents the planned API and setup procedures.

---

## Quick Start (Future v1.0)

### Basic Hardware Acquisition

```python
from oscura.acquisition import HardwareSource

# SocketCAN - Linux CAN interface
can = HardwareSource.socketcan("can0", bitrate=500000)
trace = can.read()

# Saleae Logic Analyzer
logic = HardwareSource.saleae()
logic.configure(sample_rate=1e6, duration=10)
trace = logic.read()

# PyVISA Oscilloscope
scope = HardwareSource.visa("USB0::0x0699::0x0401::INSTR")
scope.configure(channels=[1, 2], timebase=1e-6)
trace = scope.read()
```

### Streaming from Hardware

```python
# Continuous acquisition
with HardwareSource.socketcan("can0") as source:
    for chunk in source.stream(chunk_size=1000):
        # Process each chunk
        analyze(chunk)

        # Break on condition
        if stop_condition:
            break
```

---

## SocketCAN Integration (Phase 2)

### Setup

**Prerequisites**:

- Linux with SocketCAN kernel module
- `can-utils` package
- `python-can` library

**Installation**:

```bash
# Install system packages
sudo apt-get install can-utils

# Install Python dependencies
pip install oscura[automotive]  # Includes python-can
```

**Hardware Setup**:

```bash
# Setup virtual CAN for testing
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Setup real CAN interface
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Verify interface
ip link show can0
```

---

### Basic Usage

```python
from oscura.acquisition import HardwareSource

# Create SocketCAN source
can = HardwareSource.socketcan("can0", bitrate=500000)

# Read messages (one-shot)
messages = can.read()
print(f"Captured {len(messages.data)} CAN messages")

# Stream messages (continuous)
with HardwareSource.socketcan("can0") as source:
    for chunk in source.stream(duration=60):
        # Process each chunk
        for msg in chunk.messages:
            print(f"ID: 0x{msg.arbitration_id:X}, Data: {msg.data.hex()}")
```

---

### Advanced Configuration

```python
# Custom filters
can = HardwareSource.socketcan(
    interface="can0",
    bitrate=500000,
    filters=[
        {"can_id": 0x123, "can_mask": 0x7FF},  # Standard ID
        {"can_id": 0x80000456, "can_mask": 0x1FFFFFFF},  # Extended ID
    ]
)

# CAN-FD support
canfd = HardwareSource.socketcan(
    interface="can0",
    bitrate=500000,
    data_bitrate=2000000,  # CAN-FD data phase
    fd=True
)

# Error handling
can = HardwareSource.socketcan(
    interface="can0",
    receive_own_messages=True,
    bus_errors=True  # Capture bus error frames
)
```

---

### Integration with Sessions

```python
from oscura.automotive.can import CANSession
from oscura.acquisition import HardwareSource

# Create CAN analysis session
session = CANSession(name="Vehicle Analysis")

# Live capture from CAN bus
session.add_recording("idle", HardwareSource.socketcan("can0"))
session.add_recording("accelerate", HardwareSource.socketcan("can0"))

# Analyze traffic
analysis = session.analyze()
print(f"Total messages: {analysis['inventory']['total_messages']}")

# Compare recordings
diff = session.compare("idle", "accelerate")
print(f"Changed IDs: {len(diff.details['changed_ids'])}")

# Export DBC file
session.export_dbc("vehicle.dbc")
```

---

### Virtual CAN for Testing

```python
# Setup virtual CAN interface
import subprocess

subprocess.run(["sudo", "modprobe", "vcan"])
subprocess.run(["sudo", "ip", "link", "add", "dev", "vcan0", "type", "vcan"])
subprocess.run(["sudo", "ip", "link", "set", "up", "vcan0"])

# Use in tests
can = HardwareSource.socketcan("vcan0")

# Send test messages
from can import Message
msg = Message(arbitration_id=0x123, data=[0x01, 0x02, 0x03])
can.bus.send(msg)

# Receive
trace = can.read()
```

---

## Saleae Logic Integration (Phase 2)

### Setup

**Prerequisites**:

- Saleae Logic Analyzer hardware
- Saleae Logic 2 software installed
- `saleae` Python library

**Installation**:

```bash
# Download Saleae Logic 2
# https://www.saleae.com/downloads/

# Install Python library
pip install oscura[saleae]
```

---

### Basic Usage

```python
from oscura.acquisition import HardwareSource

# Auto-detect Saleae device
logic = HardwareSource.saleae()

# Configure channels
logic.configure(
    digital_channels=[0, 1, 2, 3],  # D0-D3
    analog_channels=[0],             # A0
    sample_rate=1e6,                 # 1 MS/s
    duration=10                      # 10 seconds
)

# Capture
trace = logic.read()
print(f"Captured {len(trace.data)} samples")

# Access channels
ch0 = trace.channels["D0"]
ch1 = trace.channels["D1"]
```

---

### Protocol Analysis

```python
from oscura.acquisition import HardwareSource
from oscura.analyzers.protocols import UARTDecoder

# Capture serial data
logic = HardwareSource.saleae()
logic.configure(
    digital_channels=[0],  # UART TX on D0
    sample_rate=10e6,      # 10 MS/s for 115200 baud
    duration=5
)

trace = logic.read()

# Decode UART
decoder = UARTDecoder(baud_rate=115200)
frames = decoder.decode(trace.channels["D0"])

for frame in frames:
    print(f"Data: 0x{frame.data:02X}, Valid: {frame.valid}")
```

---

### Streaming Mode

```python
# Continuous capture with real-time analysis
with HardwareSource.saleae() as source:
    source.configure(digital_channels=[0, 1, 2, 3], sample_rate=1e6)

    for chunk in source.stream(duration=60):
        # Analyze each chunk in real-time
        results = analyze_chunk(chunk)

        # Save interesting events
        if results.anomaly_detected:
            chunk.save(f"anomaly_{chunk.timestamp}.bin")
```

---

### Advanced Configuration

```python
logic = HardwareSource.saleae()

# High-speed digital capture
logic.configure(
    digital_channels=list(range(16)),  # All 16 channels
    sample_rate=100e6,                 # 100 MS/s
    buffer_size=1e9                    # 1 GB buffer
)

# Mixed analog/digital
logic.configure(
    digital_channels=[0, 1, 2],
    analog_channels=[0, 1],
    digital_sample_rate=10e6,
    analog_sample_rate=1e6,
    analog_voltage_range=5.0
)

# Triggered capture
logic.configure_trigger(
    channel=0,
    edge="rising",
    voltage_threshold=2.5
)
```

---

## PyVISA Integration (Phase 3)

### Setup

**Prerequisites**:

- VISA-compatible instrument (oscilloscope, spectrum analyzer, etc.)
- NI-VISA or VISA backend installed
- `pyvisa` and `pyvisa-py` libraries

**Installation**:

```bash
# Install NI-VISA (recommended) or use pyvisa-py
# NI-VISA: https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html

# Install Python libraries
pip install oscura[visa]
```

---

### Basic Usage

```python
from oscura.acquisition import HardwareSource

# Auto-detect instruments
instruments = HardwareSource.visa_list()
for addr in instruments:
    print(f"Found: {addr}")

# Connect to oscilloscope
scope = HardwareSource.visa("USB0::0x0699::0x0401::INSTR")

# Configure
scope.configure(
    channels=[1, 2],
    timebase=1e-6,      # 1 μs/div
    voltage_range=5.0,  # ±5V
    sample_rate=1e9     # 1 GS/s
)

# Capture
trace = scope.read()
```

---

### Oscilloscope Control

```python
# Advanced oscilloscope configuration
scope = HardwareSource.visa("TCPIP::192.168.1.100::INSTR")

# Setup channels
scope.set_channel(1, scale=1.0, offset=0.0, coupling="DC")
scope.set_channel(2, scale=2.0, offset=0.0, coupling="AC")

# Setup timebase
scope.set_timebase(scale=1e-6, position=0.0)

# Setup trigger
scope.set_trigger(
    source="CH1",
    level=1.5,
    edge="rising",
    mode="normal"
)

# Acquire
scope.single()  # Single shot
trace = scope.read_waveform(channel=1)

# Continuous acquisition
scope.run()
for i in range(100):
    trace = scope.read_waveform(channel=1)
    analyze(trace)
scope.stop()
```

---

### Spectrum Analyzer

```python
# Connect to spectrum analyzer
sa = HardwareSource.visa("GPIB0::18::INSTR")

# Configure
sa.configure(
    center_freq=2.4e9,   # 2.4 GHz
    span=100e6,          # 100 MHz
    rbw=1e6,             # 1 MHz resolution bandwidth
    vbw=1e6              # 1 MHz video bandwidth
)

# Capture spectrum
spectrum = sa.read()

# Analyze
from oscura.analyzers.spectral import find_peaks
peaks = find_peaks(spectrum)
for peak in peaks:
    print(f"Peak at {peak.frequency/1e6:.1f} MHz: {peak.amplitude:.1f} dBm")
```

---

## Unified Source Pattern

All hardware sources implement the `Source` protocol for consistency.

### Polymorphic Usage

```python
from oscura.acquisition import Source

def analyze_from_any_source(source: Source):
    """Works with files, hardware, or synthetic sources."""
    trace = source.read()
    return analyze(trace)

# Works identically with all sources
from oscura.acquisition import FileSource, HardwareSource, SyntheticSource

analyze_from_any_source(FileSource("capture.wfm"))
analyze_from_any_source(HardwareSource.socketcan("can0"))
analyze_from_any_source(SyntheticSource(builder))
```

---

### Context Manager Pattern

```python
# Automatic resource cleanup
with HardwareSource.socketcan("can0") as can:
    trace = can.read()
    # CAN interface automatically closed

# Manual resource management
can = HardwareSource.socketcan("can0")
try:
    trace = can.read()
finally:
    can.close()
```

---

### Streaming Pattern

```python
# Consistent streaming across all sources
def stream_analyze(source: Source, duration: float):
    with source:
        for chunk in source.stream(duration=duration):
            results = analyze(chunk)
            yield results

# Works with any source
for results in stream_analyze(HardwareSource.socketcan("can0"), 60):
    print(results)
```

---

## Troubleshooting

### SocketCAN Issues

**Issue**: "Network is down"

```bash
# Solution: Bring interface up
sudo ip link set up can0

# Verify
ip link show can0
```

**Issue**: "No such device"

```bash
# Solution: Load kernel module
sudo modprobe can
sudo modprobe can_raw

# For virtual CAN
sudo modprobe vcan
```

**Issue**: "Bus-off state"

```bash
# Solution: Restart interface
sudo ip link set can0 type can restart-ms 100
sudo ip link set down can0
sudo ip link set up can0
```

---

### Saleae Issues

**Issue**: "Device not found"

```python
# Solution: Check USB connection and permissions
import saleae

# List devices
devices = saleae.list_devices()
print(f"Found devices: {devices}")

# Try specific device
logic = HardwareSource.saleae(device_id=devices[0])
```

**Issue**: "Sample rate too high"

```python
# Solution: Reduce sample rate or channel count
logic.configure(
    digital_channels=[0, 1],  # Fewer channels
    sample_rate=1e6           # Lower rate
)
```

---

### PyVISA Issues

**Issue**: "VISA library not found"

```bash
# Solution: Install NI-VISA or configure pyvisa-py
pip install pyvisa-py

# Set backend
export PYVISA_LIBRARY="@py"
```

**Issue**: "Instrument not responding"

```python
# Solution: Increase timeout
scope = HardwareSource.visa("USB0::0x0699::INSTR")
scope.timeout = 10000  # 10 seconds

# Test connection
print(scope.query("*IDN?"))
```

---

## Best Practices

### 1. Resource Management

**Always use context managers**:

```python
# Good
with HardwareSource.socketcan("can0") as source:
    trace = source.read()

# Bad
source = HardwareSource.socketcan("can0")
trace = source.read()
# Resource leak if exception occurs
```

---

### 2. Error Handling

```python
from oscura.core.exceptions import AcquisitionError

try:
    with HardwareSource.socketcan("can0") as source:
        trace = source.read()
except AcquisitionError as e:
    print(f"Acquisition failed: {e}")
    # Fallback to file source
    trace = FileSource("fallback.blf").read()
```

---

### 3. Buffering for Performance

```python
# Use appropriate buffer sizes
with HardwareSource.socketcan("can0", buffer_size=10000) as source:
    # Large buffer reduces USB overhead
    trace = source.read()
```

---

### 4. Validation

```python
# Validate captured data
trace = source.read()

if len(trace.data) == 0:
    raise ValueError("No data captured - check hardware connection")

if trace.sample_rate < expected_rate * 0.9:
    print(f"WARNING: Sample rate lower than expected")
```

---

## Example: Complete Hardware Workflow

```python
from oscura.acquisition import HardwareSource
from oscura.automotive.can import CANSession

# Setup
print("Phase 1: Hardware Setup")
can_source = HardwareSource.socketcan("can0", bitrate=500000)

# Phase 2: Create Analysis Session
print("Phase 2: Creating CAN Session")
session = CANSession(name="Vehicle Reverse Engineering")

# Phase 3: Baseline Capture
print("Phase 3: Capturing Baseline")
input("Ensure vehicle is idle, then press Enter...")
session.add_recording("idle", can_source)

# Phase 4: Stimulus Captures
print("Phase 4: Capturing Stimuli")
input("Press brake pedal and hit Enter...")
session.add_recording("brake", can_source)

input("Press accelerator and hit Enter...")
session.add_recording("accelerate", can_source)

# Phase 5: Analysis
print("Phase 5: Analyzing Traffic")
analysis = session.analyze()
print(f"Total messages: {analysis['inventory']['total_messages']}")
print(f"Unique IDs: {len(analysis['inventory']['message_ids'])}")

# Phase 6: Differential Analysis
print("Phase 6: Differential Analysis")
diff_brake = session.compare("idle", "brake")
diff_accel = session.compare("idle", "accelerate")

print(f"Brake - Changed IDs: {len(diff_brake.details['changed_ids'])}")
print(f"Accel - Changed IDs: {len(diff_accel.details['changed_ids'])}")

# Phase 7: Signal Discovery
print("Phase 7: Signal Discovery")
# Use discovery tools to identify signals
# (See automotive demos for detailed examples)

# Phase 8: Export
print("Phase 8: Export Results")
session.export_dbc("vehicle_protocol.dbc")
print("Analysis complete! DBC file exported.")
```

---

## Related Documentation

- [Black-Box Analysis Guide](blackbox-analysis.md) - Protocol reverse engineering
- [Session Management API](../api/session-management.md) - Session API reference
- [Automotive Protocols Demo](../../demos/08_automotive_protocols/) - CAN examples
- [Migration Guide](../migration/v0-to-v1.md) - Version migration

---

## Implementation Status

| Feature | Status | Version |
|---------|--------|---------|
| Source Protocol | ✅ Complete | v0.3.0 |
| FileSource | ✅ Complete | v0.3.0 |
| SyntheticSource | ✅ Complete | v0.3.0 |
| SocketCAN | 🚧 Phase 2 | v1.0 (planned) |
| Saleae Logic | 🚧 Phase 2 | v1.0 (planned) |
| PyVISA | 🚧 Phase 3 | v1.1 (planned) |

**Legend**:

- ✅ Complete and tested
- 🚧 Planned, API documented
- ⏳ In development

---

**Note**: This guide documents the planned API for Phase 2 hardware integration. Placeholder implementations exist in v0.3.0 but will raise `NotImplementedError`. Full implementation targeted for v1.0.
