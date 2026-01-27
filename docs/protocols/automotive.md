# Automotive Protocols

Oscura provides comprehensive support for automotive communication protocols used in modern vehicles.

---

## CAN and CAN-FD

### Overview

Controller Area Network (CAN) is the dominant protocol in automotive systems, used for engine management, body control, infotainment, and more. CAN-FD (Flexible Data-rate) extends CAN with larger payloads and higher data rates.

**Standards:**

- CAN 2.0A (11-bit identifiers)
- CAN 2.0B (29-bit extended identifiers)
- CAN-FD (ISO 11898-1:2015)

### Features

- Full CAN 2.0A/B and CAN-FD support
- Automatic signal extraction and boundary detection
- DBC file generation (no manual signal definition)
- Multiplexed message handling
- Counter and checksum detection
- Bus load and timing analysis
- State machine extraction

### Basic Usage

```python
from oscura.sessions import CANSession

# Create CAN session
session = CANSession(
    bitrate=500000,      # 500 kbps
    fd_mode=False,       # Set True for CAN-FD
    sample_point=0.75    # 75% sample point
)

# Load capture (supports BLF, ASC, PCAP, LOG)
session.load("vehicle_capture.blf")

# Basic statistics
stats = session.get_statistics()
print(f"Messages: {stats.message_count:,}")
print(f"Bus load: {stats.bus_load:.1%}")
print(f"Error frames: {stats.error_count}")
```

### CAN-FD Specific Features

```python
# CAN-FD session
session = CANSession(
    bitrate=500000,          # Arbitration bitrate
    data_bitrate=2000000,    # Data phase bitrate (CAN-FD)
    fd_mode=True,
    brs=True                 # Bit Rate Switching
)

# Analyze CAN-FD specific features
fd_analysis = session.analyze_fd_features()

print(f"FD frames: {fd_analysis.fd_frame_count}")
print(f"BRS frames: {fd_analysis.brs_frame_count}")
print(f"Max payload: {fd_analysis.max_dlc} bytes")
print(f"Avg data rate: {fd_analysis.avg_data_rate / 1e6:.2f} Mbps")
```

### Signal Extraction

```python
# Automatic signal boundary detection
signals = session.extract_signals()

for signal in signals:
    print(f"{signal.name}:")
    print(f"  CAN ID: 0x{signal.can_id:03X}")
    print(f"  Start bit: {signal.start_bit}")
    print(f"  Length: {signal.length} bits")
    print(f"  Byte order: {signal.byte_order}")
    print(f"  Range: {signal.min_value} - {signal.max_value}")
```

### DBC Generation

```python
# Export DBC file with all inferred signals
session.export_dbc(
    output="vehicle.dbc",
    version="1.0",
    include_comments=True,
    include_value_tables=True,
    validation=True
)
```

### Advanced Analysis

**Message Pattern Detection:**

```python
patterns = session.find_patterns()

# Periodic messages
for can_id, period in patterns.periodic_messages.items():
    print(f"0x{can_id:03X}: {period:.1f} ms period")

# Event-driven messages
for can_id in patterns.event_driven_messages:
    print(f"0x{can_id:03X}: sporadic")

# Multiplexed messages
for can_id, mux_info in patterns.multiplexed_messages.items():
    print(f"0x{can_id:03X}: {len(mux_info)} multiplexor values")
```

**Security Analysis:**

```python
security = session.security_analysis()

print(f"No checksum: {len(security.no_checksum)} messages")
print(f"No counter: {len(security.no_counter)} messages")
print(f"Replay vulnerable: {security.replay_vulnerable}")
```

### Performance

- **Decoding:** 200 MB/s
- **Signal extraction:** 1M messages/sec
- **DBC generation:** <5 seconds for typical vehicle capture

---

## LIN

### Overview

Local Interconnect Network (LIN) is a low-cost serial protocol for automotive body control (windows, mirrors, seats, etc.).

**Standards:**

- LIN 1.x
- LIN 2.x (ISO 17987)

### Features

- Full LIN 1.x and 2.x support
- Master/slave frame detection
- Checksum validation (classic and enhanced)
- Break field detection
- Schedule table extraction

### Basic Usage

```python
from oscura.analyzers.protocols import LINDecoder

# Create LIN decoder
decoder = LINDecoder(
    baud_rate=19200,        # Common: 9600, 19200
    version="2.0",
    checksum_type="enhanced"
)

# Decode waveform
messages = decoder.decode(waveform)

for msg in messages:
    print(f"ID: 0x{msg.frame_id:02X}")
    print(f"  Data: {msg.data.hex()}")
    print(f"  Checksum: {'valid' if msg.checksum_valid else 'INVALID'}")
```

### Auto-Detection

```python
# Auto-detect LIN parameters
params = LINDecoder.auto_detect(waveform)

print(f"Baud rate: {params.baud_rate}")
print(f"Version: {params.version}")
print(f"Checksum: {params.checksum_type}")
```

### Schedule Table Extraction

```python
from oscura.sessions import LINSession

session = LINSession(baud_rate=19200)
session.load("lin_capture.vcd")

# Extract communication schedule
schedule = session.extract_schedule()

for entry in schedule:
    print(f"{entry.frame_id:02X}: every {entry.period:.1f} ms")
```

### LDF Generation

```python
# Export LIN Description File
session.export_ldf("lighting_control.ldf")
```

---

## FlexRay

### Overview

FlexRay is a high-speed, deterministic protocol for safety-critical automotive applications (steering, braking, suspension).

**Standards:**

- FlexRay 2.1 (ISO 17458)

### Features

- Static and dynamic segment support
- Dual-channel support
- Cycle-based scheduling
- Startup sequence analysis
- Symbol decoding (partial)

### Basic Usage

```python
from oscura.analyzers.protocols import FlexRayDecoder

decoder = FlexRayDecoder(
    bitrate=10_000_000,     # 10 Mbps
    channels=2,              # Dual channel
    cycle_length=5.0         # 5ms cycle
)

messages = decoder.decode(waveform)

for msg in messages:
    print(f"Slot: {msg.slot_id}")
    print(f"  Channel: {msg.channel}")
    print(f"  Cycle: {msg.cycle}")
    print(f"  Segment: {msg.segment}")  # static or dynamic
    print(f"  Data: {msg.payload.hex()}")
```

### Cluster Analysis

```python
from oscura.sessions import FlexRaySession

session = FlexRaySession()
session.load("flexray_capture.blf")

# Analyze cluster configuration
cluster = session.analyze_cluster()

print(f"Static slots: {cluster.static_slot_count}")
print(f"Dynamic slots: {cluster.dynamic_slot_count}")
print(f"Cycle length: {cluster.cycle_length} ms")
print(f"Nodes detected: {len(cluster.nodes)}")
```

### FIBEX Export

```python
# Export FIBEX (FlexRay Interface Specification)
session.export_fibex("cluster_config.xml")
```

### Limitations

- Symbol encoding not fully implemented
- Startup frame analysis partial
- AUTOSAR integration planned for v0.6.0

---

## UDS (Unified Diagnostic Services)

### Overview

UDS (ISO 14229) is the standard diagnostic protocol for automotive ECUs, used for reading DTCs, flashing firmware, and security access.

**Standards:**

- ISO 14229-1 (Application layer)
- ISO 15765-2 (Transport over CAN)

### Features

- All UDS services (0x10-0x3E, 0x83-0x87)
- Session management
- Security access tracking
- DTC handling
- Memory read/write
- Routine control

### Basic Usage

```python
from oscura.analyzers.protocols import UDSDecoder

# UDS over CAN (ISO-TP)
decoder = UDSDecoder(
    transport="can",
    request_id=0x7E0,
    response_id=0x7E8
)

# Load CAN capture with UDS traffic
messages = decoder.decode_from_can(can_capture)

for msg in messages:
    print(f"Service: {msg.service_name} (0x{msg.sid:02X})")
    print(f"  Direction: {msg.direction}")
    print(f"  Data: {msg.data.hex()}")

    if msg.is_negative_response:
        print(f"  NRC: {msg.nrc_name} (0x{msg.nrc:02X})")
```

### Service-Specific Analysis

**Diagnostic Session Control (0x10):**

```python
sessions = decoder.find_service(0x10)

for session in sessions:
    print(f"Session type: {session.session_type}")
    print(f"  0x01: Default")
    print(f"  0x02: Programming")
    print(f"  0x03: Extended")
```

**Read Data By Identifier (0x22):**

```python
read_requests = decoder.find_service(0x22)

for req in read_requests:
    print(f"DID: 0x{req.did:04X}")
    print(f"  Response: {req.response_data.hex()}")
    print(f"  Length: {len(req.response_data)} bytes")
```

**Security Access (0x27):**

```python
security_analysis = decoder.analyze_security_access()

print(f"Security levels: {security_analysis.levels}")
print(f"Seed-key pairs: {len(security_analysis.seed_key_pairs)}")

# Attempt to infer security algorithm
if security_analysis.algorithm_detected:
    print(f"Algorithm: {security_analysis.algorithm}")
```

**Request Download (0x34) / Transfer Data (0x36):**

```python
downloads = decoder.find_firmware_downloads()

for download in downloads:
    print(f"Memory address: 0x{download.address:X}")
    print(f"  Size: {download.size} bytes")
    print(f"  Blocks: {len(download.blocks)}")

    # Extract firmware image
    firmware = download.reconstruct_firmware()
    with open("extracted_firmware.bin", "wb") as f:
        f.write(firmware)
```

### Security Analysis

```python
from oscura.sessions import UDSSession

session = UDSSession()
session.load("diagnostic_capture.blf")

# Security assessment
security = session.security_assessment()

print(f"\nSecurity Findings:")
print(f"  Unprotected services: {len(security.unprotected_services)}")
print(f"  Weak seed-key: {security.weak_seed_key}")
print(f"  Seed reuse: {security.seed_reuse}")
print(f"  Timing attacks possible: {security.timing_attack_vulnerable}")

# Export security report
security.export_report("uds_security_assessment.html")
```

### Integration with CAN Analysis

```python
from oscura.sessions import CANSession

# Load CAN capture
can_session = CANSession(bitrate=500000)
can_session.load("vehicle_diagnostics.blf")

# Enable UDS decoder
can_session.enable_uds_analysis(
    request_ids=[0x7E0, 0x7E1, 0x7E2],
    response_ids=[0x7E8, 0x7E9, 0x7EA]
)

# Analyze both CAN signals and UDS diagnostics
combined_analysis = can_session.analyze_all()

print(f"CAN messages: {combined_analysis.can_message_count}")
print(f"UDS transactions: {combined_analysis.uds_transaction_count}")
```

---

## Comparison Table

| Feature | CAN/CAN-FD | LIN | FlexRay | UDS |
|---------|------------|-----|---------|-----|
| **Bitrate** | 125K-8M | 9.6K-20K | 2.5M-10M | N/A (over CAN) |
| **Max Payload** | 8/64 bytes | 8 bytes | 254 bytes | 4095 bytes |
| **Deterministic** | No | No | Yes | N/A |
| **Redundancy** | No | No | Dual channel | No |
| **Typical Use** | General | Body control | Safety | Diagnostics |
| **Auto-detect** | ✓ | ✓ | Partial | ✓ |
| **DBC/LDF Export** | ✓ | ✓ | FIBEX | N/A |

---

## Examples

### Complete Vehicle RE Workflow

```python
from oscura.workflows import automotive_re_workflow

# Complete automotive RE in one function
results = automotive_re_workflow(
    can_capture="drive_cycle.blf",
    lin_capture="body_control.vcd",
    uds_capture="diagnostics.blf",
    export_dir="output/"
)

# Artifacts created:
# - output/vehicle_can.dbc
# - output/body_control.ldf
# - output/diagnostics_report.html
# - output/security_assessment.pdf
```

### OBD-II Analysis

```python
from oscura.integrations import obd2_analysis

# Analyze OBD-II capture
obd = obd2_analysis(
    capture_file="obd2_session.blf",
    protocol="ISO 15765-4"  # CAN
)

# Extract data
dtcs = obd.get_dtcs()
pids = obd.get_supported_pids()
freeze_frames = obd.get_freeze_frames()

print(f"DTCs: {[f'{dtc.code} - {dtc.description}' for dtc in dtcs]}")
```

---

## Best Practices

### Capture Guidelines

1. **Sample rate:** At least 10x the bitrate
2. **Duration:** Capture full operational cycle
3. **State variation:** Include idle, active, error states
4. **Document conditions:** Note speed, RPM, temperature, etc.

### Analysis Strategy

1. **Start with CAN:** Most common automotive protocol
2. **Identify periodic messages:** Usually sensor data
3. **Correlate with events:** Match signals to user actions
4. **Check security:** Look for authentication, counters, checksums
5. **Validate findings:** Replay messages, observe vehicle response

### Security Considerations

1. **Never test on moving vehicle:** Safety-critical systems
2. **Disconnect safety systems:** Before injection testing
3. **Document everything:** Legal/safety liability
4. **Responsible disclosure:** Report vulnerabilities properly

---

## See Also

- [Tutorial: CAN Bus Analysis](../tutorials/can-bus-analysis.md)
- [Tutorial: UDS Diagnostics](../tutorials/uds-diagnostics.md)
- [API: CAN Session](../api/sessions/can.md)
- [API: UDS Decoder](../api/protocols/uds.md)
- [FAQ: Automotive Protocols](../faq/automotive.md)
