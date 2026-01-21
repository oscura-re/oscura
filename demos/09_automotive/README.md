# Automotive Protocol Analysis Demo

**Comprehensive demonstration** of Oscura's automotive reverse engineering and protocol analysis capabilities across ALL major vehicle communication protocols.

---

## 🚗 Overview

This demo showcases Oscura's complete automotive protocol toolkit for security research, ECU debugging, and vehicle bus analysis. Covers 7 major automotive protocols with real-world use cases.

### Protocols Supported

|Protocol|Speed|Description|Use Cases|
|---|---|---|---|
|**CAN 2.0**|125-1000 kbps|Controller Area Network (11/29-bit)|Powertrain, body, chassis|
|**CAN-FD**|2-8 Mbps|Flexible Data Rate CAN|ADAS, high-bandwidth sensors|
|**LIN 2.0**|2.4-19.2 kbps|Local Interconnect Network|Body control, HVAC, seats|
|**FlexRay**|10 Mbps|Time-triggered dual-channel|X-by-wire, safety-critical systems|
|**OBD-II**|Over CAN/K|On-Board Diagnostics|Emissions, diagnostics, testing|
|**UDS**|Over CAN|Unified Diagnostic Services (ISO14229)|ECU programming, security access|
|**J1939**|250 kbps|Heavy-duty vehicle protocol|Trucks, buses, construction|

## 🚀 Quick Start

### Run All Demos

```bash
# Full demonstration (all protocols)
python demos/08_automotive/comprehensive_automotive_demo.py

# Run specific demo
python demos/08_automotive/comprehensive_automotive_demo.py --demo can-re

# List available demos
python demos/08_automotive/comprehensive_automotive_demo.py --list
```

### Available Demos

```bash
--demo all       # Run all demonstrations (default)
--demo can-dbc   # CAN with DBC integration
--demo can-re    # CAN reverse engineering
--demo obd2      # OBD-II diagnostics
--demo uds       # UDS diagnostic services
--demo j1939     # J1939 heavy vehicle protocol
--demo lin       # LIN single-wire protocol
--demo flexray   # FlexRay time-triggered protocol
```

---

## 📋 Capabilities Demonstrated

### 1. CAN Bus Analysis with DBC Integration

**Features**:

- Load DBC database files (standard automotive signal definition)
- Decode CAN messages to engineering values
- Extract signals: RPM, speed, temperature, etc.
- Message inventory and frequency analysis
- Bus utilization calculation

**Python API**:

```python
import cantools
from oscura.automotive.can import CANSession
from oscura.acquisition import FileSource

# Load DBC database
db = cantools.database.load_file("vehicle.dbc")

# Create session and load CAN data
session = CANSession(name="Vehicle Analysis")
session.add_recording("capture", FileSource("capture.blf"))

# Analyze traffic
analysis = session.analyze()
print(f"Total messages: {analysis['inventory']['total_messages']}")

# Decode messages (access via session recordings)
for recording_name, messages in session._recordings.items():
    for msg in messages:
        decoded_msg = db.get_message_by_name("Engine_Status")
        if decoded_msg.frame_id == msg.arbitration_id:
            signals = decoded_msg.decode(msg.data)
            rpm = signals['engine_rpm']
            print(f"RPM: {rpm}")
```

**Use Cases**:

- Vehicle diagnostics and monitoring
- Signal extraction for analysis
- Data logging with known protocols
- Compliance testing

---

### 2. CAN Bus Reverse Engineering

**Features**:

- Message inventory generation (ID, frequency, length)
- Byte entropy analysis (constant vs. variable)
- Signal hypothesis testing (encoding, scale, offset)
- Statistical validation (min/max, rate of change)
- Discovery documentation with evidence tracking
- DBC file generation from discovered signals

**Workflow**:

```python
from oscura.automotive.can import CANSession
from oscura.acquisition import FileSource
from oscura.automotive.can.discovery import DiscoveryDocument
from oscura.automotive.dbc import DBCGenerator

# Create session and load unknown CAN traffic
session = CANSession(name="Unknown Protocol RE")
session.add_recording("unknown", FileSource("unknown.blf"))

# Analyze traffic
analysis = session.analyze()
print(f"Unique IDs: {len(analysis['inventory']['message_ids'])}")

# Generate message inventory
inventory = session.inventory()
print(inventory)

# Analyze specific message
msg = session.message(0x280)
msg_analysis = msg.analyze()

# Test hypothesis
hypothesis = msg.test_hypothesis(
    signal_name="engine_rpm",
    start_byte=2,
    bit_length=16,
    byte_order="big_endian",
    scale=0.25,
    unit="rpm",
)

# Document and export
if hypothesis.is_valid:
    doc = DiscoveryDocument()
    # ... add discoveries ...
    DBCGenerator.generate(doc, "discovered.dbc")
```

**Use Cases**:

- Aftermarket integrations (reverse engineering OEM protocols)
- Automotive security research
- Custom ECU development
- Protocol documentation

---

### 3. OBD-II Diagnostic Decoding

**Features**:

- Mode 01: Live data (RPM, speed, coolant temp, etc.)
- Mode 03: Diagnostic Trouble Codes (DTCs)
- Mode 09: Vehicle information (VIN, calibration ID)
- Multi-frame message handling
- DTC database lookup (200+ powertrain/chassis/body codes)

**Example**:

```python
from oscura.automotive.obd import OBD2Decoder
from oscura.automotive.dtc import DTCDatabase

# Decode OBD-II message
service = OBD2Decoder.decode_service(can_message)
print(service)  # "Mode 01 PID 0x0C: Engine RPM"

# Look up DTC
info = DTCDatabase.lookup("P0420")
print(f"{info.code}: {info.description}")
# P0420: Catalyst System Efficiency Below Threshold (Bank 1)
```

**Use Cases**:

- Vehicle diagnostics and troubleshooting
- Emissions testing
- Fleet monitoring
- DIY automotive repair

---

### 4. UDS (ISO 14229) Diagnostic Services

**Features**:

- Service 0x10: Diagnostic Session Control
- Service 0x22: Read Data By Identifier
- Service 0x27: Security Access (seed/key)
- Service 0x2E: Write Data By Identifier
- Service 0x31: Routine Control
- Service 0x34/36/37: Memory programming
- Service 0x11: ECU Reset

**Security Access Example**:

```python
from oscura.automotive.uds import UDSDecoder

# Request seed
decoded = UDSDecoder.decode_service(request_msg)
# "SecurityAccess RequestSeed (0x01)"

# Receive seed, calculate key, send key
decoded = UDSDecoder.decode_service(key_msg)
# "SecurityAccess SendKey (0x02)"
```

**Use Cases**:

- ECU programming and flashing
- Security research (seed/key algorithms)
- Factory diagnostics and testing
- Automotive penetration testing

---

### 5. J1939 Heavy Vehicle Protocol

**Features**:

- 29-bit identifier parsing (Priority, PGN, Source Address)
- Parameter Group Number (PGN) decoding
- Multi-packet message reassembly
- Standard PGN database (engine, transmission, brakes)
- Fleet telemetry extraction

**Example**:

```python
from oscura.automotive.j1939 import J1939Decoder

# Decode J1939 message
decoded = J1939Decoder.decode_message(can_message)
print(f"PGN: {decoded.pgn} ({decoded.pgn_name})")
print(f"Priority: {decoded.priority}")
print(f"Source: 0x{decoded.source_address:02X}")
```

**Use Cases**:

- Heavy-duty vehicle diagnostics
- Fleet telematics
- Construction equipment monitoring
- Agricultural machinery

---

### 6. LIN Single-Wire Protocol

**Features**:

- LIN 2.0 frame decoding
- Master/slave communication analysis
- Enhanced checksum validation
- Low-speed body control applications

**Frame Structure**:

```
Break → Sync → Protected ID → Data[0-8] → Checksum
```

**Use Cases**:

- Body control modules (doors, windows, mirrors)
- HVAC systems
- Seat control
- Lighting systems

**Note**: Requires analog waveform capture (use Oscura's UART decoder with LIN parameters).

---

### 7. FlexRay Time-Triggered Protocol

**Features**:

- Dual-channel redundancy
- Static segment (TDMA slots)
- Dynamic segment (mini-slots)
- 10 Mbps per channel
- Cycle-based communication

**Applications**:

- Brake-by-wire, steer-by-wire
- Active safety systems (ADAS)
- Suspension control
- Next-generation powertrain

**Note**: Requires specialized hardware for capture.

---

## 📊 Demo Data Files

Generate realistic demo data:

```bash
python demos/08_automotive/generate_demo_data.py
```

Generated files:

|File|Size|Description|
|---|---|---|
|`can_bus_normal_traffic.mf4`|~10 MB|CAN 2.0B with engine/body messages|
|`can_fd_high_speed.mf4`|~8 MB|CAN-FD high-throughput data|
|`lin_body_control.wfm`|~2 MB|LIN 2.0 @ 19.2 kbps|
|`obd2_diagnostic_session.pcap`|~1 MB|OBD-II diagnostic sequence|
|`uds_security_sequence.mf4`|~3 MB|UDS security access + memory read|
|`demo_signals.dbc`|~5 KB|Sample DBC with common signals|

## 🎯 Use Cases

### Automotive Security Research

- Reverse engineer proprietary CAN protocols
- Discover security vulnerabilities
- Analyze authentication mechanisms
- Test ECU hardening

### ECU Debugging & Development

- Monitor vehicle bus traffic in real-time
- Validate new ECU implementations
- Debug intermittent communication issues
- Compliance testing (ISO 11898, ISO 14229, etc.)

### Aftermarket Integration

- Reverse engineer OEM signals for custom integrations
- Create DBC files for third-party tools
- Monitor vehicle state for custom applications
- Interface with factory systems

### Fleet Monitoring & Telemetry

- Extract vehicle data for analytics
- Monitor heavy-duty vehicle health
- Predictive maintenance
- Driver behavior analysis

### Vintage Computing & Restoration

- Document legacy vehicle protocols
- Create replacement ECUs
- Retrofit modern features
- Preserve automotive heritage

---

## 🔧 Python API Reference

### CANSession

```python
from oscura.automotive.can import CANSession
from oscura.acquisition import FileSource

# Create session
session = CANSession(name="Vehicle Analysis")

# Load from various formats using unified Source protocol
session.add_recording("baseline", FileSource("capture.blf"))  # BLF
session.add_recording("test1", FileSource("capture.asc"))  # ASC
session.add_recording("test2", FileSource("capture.csv"))  # CSV

# Analyze traffic
analysis = session.analyze()
print(f"Total messages: {analysis['inventory']['total_messages']}")
print(f"Unique IDs: {len(analysis['inventory']['message_ids'])}")

# Message inventory
inventory = session.inventory()  # pandas DataFrame

# Analyze specific message
msg = session.message(0x280)
msg_analysis = msg.analyze()  # Message statistics

# Get unique IDs
ids = session.unique_ids()  # Set of all CAN IDs

# Compare recordings
diff = session.compare("baseline", "test1")
print(f"Changed IDs: {len(diff.details['changed_ids'])}")

# Export DBC file
session.export_dbc("discovered.dbc")
```

### Hypothesis Testing

```python
# Test signal encoding hypothesis
hypothesis = msg.test_hypothesis(
    signal_name="vehicle_speed",
    start_byte=0,
    bit_length=16,
    byte_order="big_endian",
    scale=0.01,
    offset=0,
    unit="km/h",
    expected_min=0,
    expected_max=250,
)

print(f"Valid: {hypothesis.is_valid}")
print(f"Confidence: {hypothesis.confidence}")
```

### DBC Generation

```python
from oscura.automotive.dbc import DBCGenerator

# Create discovery document
doc = DiscoveryDocument()
doc.vehicle.make = "Tesla"
doc.vehicle.model = "Model 3"

# Add discovered messages and signals
# ...

# Generate DBC file
DBCGenerator.generate(
    doc,
    output_path="discovered.dbc",
    min_confidence=0.8,  # Only include high-confidence signals
)
```

### OBD-II Decoding

```python
from oscura.automotive.obd import OBD2Decoder

# Decode service
service = OBD2Decoder.decode_service(can_message)

# Common PIDs
# 0x0C: Engine RPM
# 0x0D: Vehicle Speed
# 0x05: Coolant Temperature
# 0x0F: Intake Air Temperature
```

### UDS Decoding

```python
from oscura.automotive.uds import UDSDecoder

# Decode diagnostic service
service = UDSDecoder.decode_service(can_message)

# Common services:
# 0x10: Diagnostic Session Control
# 0x22: Read Data By Identifier
# 0x27: Security Access
# 0x2E: Write Data By Identifier
```

---

## 📚 Related Documentation

- **Main Demos**: `demos/README.md`
- **Protocol Decoding**: `demos/05_protocol_decoding/`
- **Signal RE**: `demos/04_signal_reverse_engineering/`
- **Examples**: `examples/automotive/`
- **API Reference**: `docs/api/automotive/`

---

## 🔗 Standards References

|Standard|Title|Coverage|
|---|---|---|
|ISO 11898-1|CAN data link layer and physical layer|CAN 2.0|
|ISO 11898-7|CAN FD data link layer|CAN-FD|
|ISO 14229|Unified Diagnostic Services (UDS)|UDS|
|ISO 15765-4|Diagnostic communication over CAN|OBD-II over CAN|
|ISO 17987|Local Interconnect Network (LIN)|LIN 2.0+|
|SAE J1939|Recommended Practice for Vehicle Network|J1939|
|FlexRay Consortium|FlexRay Communications System Protocol|FlexRay|

## 💡 Tips & Best Practices

### Reverse Engineering Workflow

1. **Capture**: Record CAN traffic during specific vehicle events
2. **Inventory**: Generate message inventory to identify periodic vs. event-driven
3. **Focus**: Isolate messages that change during stimulus (throttle, brake, etc.)
4. **Hypothesize**: Test encoding assumptions (byte order, scale, offset)
5. **Validate**: Verify hypothesis across multiple captures
6. **Document**: Create discovery document with evidence
7. **Export**: Generate DBC for integration with other tools

### Security Considerations

- **ECU Security Access**: Always test on isolated bench setups, never on production vehicles
- **Seed/Key Algorithms**: Document but do not publish proprietary security algorithms
- **Responsible Disclosure**: Report vulnerabilities to manufacturers before public disclosure
- **Safety-Critical Systems**: Never test on brake, steering, or airbag systems without proper safety measures

### Performance Tips

- **Large Captures**: Use streaming loaders for multi-GB files
- **Filtering**: Filter by CAN ID early to reduce memory usage
- **Chunking**: Process in time windows for real-time analysis
- **Caching**: Save discovery documents to avoid re-analysis

---

## 🐛 Common Issues

### Issue: "cantools not found"

**Solution**:

```bash
uv sync --all-extras
```

### Issue: "Cannot load MF4 file"

**Solution**: Install asammdf:

```bash
uv sync --all-extras
```

### Issue: "No frames decoded"

**Solution**: Check CAN bitrate, verify data format (standard vs. extended IDs)

### Issue: "DBC generation fails"

**Solution**: Ensure discovery document has messages with confidence ≥ min_confidence

---

## 🚀 Advanced Features

### Stimulus-Response Analysis

```python
from oscura.automotive.can.stimulus_response import find_correlations

# Find messages correlated with stimulus
correlations = find_correlations(
    session,
    stimulus_signal="throttle_position",
    min_correlation=0.7,
)
```

### State Machine Learning

```python
from oscura.automotive.can.state_machine import learn_state_machine

# Learn state machine from CAN traffic
fsm = learn_state_machine(
    session,
    message_id=0x280,
    byte_index=0,
)
```

### CRC Detection

```python
from oscura.crc import detect_crc_polynomial

# Detect CRC algorithm
result = detect_crc_polynomial(
    messages,
    data_slice=slice(0, 7),  # Data bytes
    crc_slice=slice(7, 8),   # CRC byte
)
```

---

## 📖 Example Outputs

### Message Inventory

```
CAN ID  | Count | Freq (Hz) | Period (ms) | Length | Entropy
--------|-------|-----------|-------------|--------|--------
0x0C0   |  1000 |     100.0 |       10.0  |     8  |  0.234
0x280   |  1000 |     100.0 |       10.0  |     8  |  0.456
0x300   |   500 |      50.0 |       20.0  |     8  |  0.123
0x400   |   200 |      20.0 |       50.0  |     8  |  0.789
```

### Signal Hypothesis Test

```
Signal: engine_rpm
  Start Byte: 2
  Length: 16 bits
  Byte Order: big_endian
  Scale: 0.25
  Offset: 0
  Unit: rpm

  ✓ Valid: True
  Confidence: 0.95
  Value Range: 800.0 - 2000.0 rpm
  Evidence:
    - Statistical analysis: monotonic increase
    - Value range matches expectations
    - No outliers detected
```

### Generated DBC File

```dbc
VERSION ""

NS_ :
  NS_DESC_
  CM_
  BA_DEF_
  BA_

BS_:

BO_ 640 Engine_Status: 8 Vector__XXX
 SG_ engine_rpm : 16|16@1+ (0.25,0) [0|8000] "rpm" Vector__XXX

BA_DEF_ "BusType" STRING;
BA_DEF_ BO_ "GenMsgCycleTime" INT 0 0;
BA_ "BusType" "CAN";
BA_ "GenMsgCycleTime" BO_ 640 10;
```

---

**Last Updated**: 2026-01-15
**Status**: Production-ready
**Maintainer**: Oscura Team
**License**: MIT
