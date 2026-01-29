# Tutorial: Automotive CAN Bus Analysis and DBC Generation

Learn how to reverse engineer CAN bus traffic and automatically generate DBC files without manual signal definition.

**Scenario:** Analyzing vehicle CAN bus to understand message structures and generate a DBC file for use in automotive tools.

**Time Required:** 45-60 minutes
**Difficulty:** Intermediate
**Prerequisites:** Basic understanding of CAN protocol, access to vehicle CAN bus

---

## Part 1: Equipment and Capture

### Required Equipment

- CAN bus interface (e.g., PCAN-USB, Kvaser Leaf, SocketCAN)
- OBD-II adapter or direct CAN connection
- Vehicle or CAN bus simulator

### Capture CAN Traffic

Using Vector CANoe, Kvaser CANKing, or `candump`:

```bash
# Linux with SocketCAN
candump -l can0

# Creates: candump-YYYY-MM-DD_HHMMSS.log

# Or use Oscura's built-in capture (requires SocketCAN)
python -m oscura.capture can --interface can0 --duration 60 --output vehicle_capture.blf
```

### Load Existing Capture

Oscura supports multiple CAN formats:

```python
from oscura.sessions import CANSession

# Create CAN session
session = CANSession(
    bitrate=500000,      # 500 kbps (common for automotive)
    fd_mode=False,       # Standard CAN (not CAN-FD)
    name="Vehicle Analysis"
)

# Load capture file
# Supported formats: BLF, ASC, LOG, PCAP, CSV
session.load("vehicle_capture.blf")

# Basic statistics
stats = session.get_statistics()
print(f"Capture Statistics:")
print(f"  Duration: {stats.duration:.2f} seconds")
print(f"  Total messages: {stats.message_count:,}")
print(f"  Unique CAN IDs: {len(stats.unique_ids)}")
print(f"  Bus load: {stats.bus_load:.1%}")
print(f"  Error frames: {stats.error_count}")
```

**Expected Output:**

```
Capture Statistics:
  Duration: 120.45 seconds
  Total messages: 24,532
  Unique CAN IDs: 42
  Bus load: 32.5%
  Error frames: 0
```

---

## Part 2: Message Classification

### Identify Message Patterns

```python
# Automatic pattern detection
patterns = session.find_patterns()

print(f"\n=== Periodic Messages (Fixed Rate) ===")
for can_id, period in sorted(patterns.periodic_messages.items()):
    print(f"ID 0x{can_id:03X}: {period:.1f} ms period")

print(f"\n=== Event-Driven Messages ===")
for can_id in sorted(patterns.event_driven_messages.keys()):
    count = patterns.event_driven_messages[can_id]
    print(f"ID 0x{can_id:03X}: {count} occurrences (sporadic)")

print(f"\n=== Multiplexed Messages ===")
for can_id in sorted(patterns.multiplexed_messages.keys()):
    multiplexors = patterns.multiplexed_messages[can_id]
    print(f"ID 0x{can_id:03X}: {len(multiplexors)} multiplexor values")
```

**Expected Output:**

```
=== Periodic Messages (Fixed Rate) ===
ID 0x0C0: 10.0 ms period
ID 0x0D0: 10.0 ms period
ID 0x120: 20.0 ms period
ID 0x130: 20.0 ms period
ID 0x180: 100.0 ms period
ID 0x280: 100.0 ms period
ID 0x300: 1000.0 ms period

=== Event-Driven Messages ===
ID 0x400: 15 occurrences (sporadic)
ID 0x410: 8 occurrences (sporadic)

=== Multiplexed Messages ===
ID 0x200: 4 multiplexor values
```

### Analyze Specific CAN ID

```python
# Deep dive into specific message
msg_id = 0x0C0

analysis = session.analyze_message(msg_id)

print(f"\n=== Analysis of CAN ID 0x{msg_id:03X} ===")
print(f"Message count: {analysis.count}")
print(f"DLC: {analysis.dlc} bytes")
print(f"Period: {analysis.period:.2f} ms")
print(f"Jitter: {analysis.jitter:.2f} ms")

print(f"\nByte-level analysis:")
for byte_idx in range(analysis.dlc):
    byte_stats = analysis.byte_statistics[byte_idx]
    print(f"  Byte {byte_idx}:")
    print(f"    Range: 0x{byte_stats.min:02X} - 0x{byte_stats.max:02X}")
    print(f"    Unique values: {byte_stats.unique_count}")
    print(f"    Entropy: {byte_stats.entropy:.2f} bits")
    print(f"    Classification: {byte_stats.classification}")
```

**Expected Output:**

```
=== Analysis of CAN ID 0x0C0 ===
Message count: 12045
DLC: 8 bytes
Period: 10.01 ms
Jitter: 0.15 ms

Byte-level analysis:
  Byte 0:
    Range: 0x00 - 0xFF
    Unique values: 256
    Entropy: 7.98 bits
    Classification: counter
  Byte 1:
    Range: 0x00 - 0x64
    Unique values: 101
    Entropy: 6.65 bits
    Classification: data
  Byte 2:
    Range: 0x80 - 0xFF
    Unique values: 128
    Entropy: 6.99 bits
    Classification: data
  ...
```

---

## Part 3: Signal Extraction

### Automatic Signal Boundary Detection

```python
# Extract all signals from all messages
signals = session.extract_signals()

print(f"\n=== Extracted Signals ({len(signals)}) ===\n")

for signal in signals[:10]:  # Show first 10
    print(f"Signal: {signal.name}")
    print(f"  CAN ID: 0x{signal.can_id:03X}")
    print(f"  Start bit: {signal.start_bit}")
    print(f"  Length: {signal.length} bits")
    print(f"  Byte order: {signal.byte_order}")
    print(f"  Value type: {signal.value_type}")
    print(f"  Range: {signal.min_value:.2f} - {signal.max_value:.2f}")
    print(f"  Resolution: {signal.resolution}")
    if signal.unit:
        print(f"  Unit: {signal.unit}")
    print()
```

**Expected Output:**

```
=== Extracted Signals (127) ===

Signal: MSG_0C0_Counter
  CAN ID: 0x0C0
  Start bit: 0
  Length: 8 bits
  Byte order: little_endian
  Value type: unsigned
  Range: 0.00 - 255.00
  Resolution: 1.0

Signal: MSG_0C0_Data_1
  CAN ID: 0x0C0
  Start bit: 8
  Length: 16 bits
  Byte order: little_endian
  Value type: unsigned
  Range: 0.00 - 6400.00
  Resolution: 0.01
  Unit: percent

Signal: MSG_0C0_Data_2
  CAN ID: 0x0C0
  Start bit: 24
  Length: 16 bits
  Byte order: little_endian
  Value type: signed
  Range: -128.00 - 127.00
  Resolution: 1.0
  Unit: degrees_celsius
...
```

### Correlate Signals with Vehicle Behavior

```python
# Add event markers (from manual observations)
session.add_event_marker(15.2, "brake_pressed")
session.add_event_marker(18.5, "accelerator_pressed")
session.add_event_marker(25.3, "left_turn_signal")
session.add_event_marker(30.1, "gear_shift_d_to_r")

# Find signals that correlate with events
correlations = session.correlate_with_events()

print(f"\n=== Signal-Event Correlations ===\n")

for event_time, event_name in session.events:
    print(f"Event: {event_name} @ {event_time:.1f}s")

    if event_name in correlations:
        for signal, confidence in correlations[event_name]:
            print(f"  → {signal.name} (confidence: {confidence:.2%})")
    print()
```

**Expected Output:**

```
=== Signal-Event Correlations ===

Event: brake_pressed @ 15.2s
  → MSG_0C0_Data_1 (confidence: 95.3%)
  → MSG_120_BrakePressure (confidence: 98.7%)

Event: accelerator_pressed @ 18.5s
  → MSG_0C0_Data_3 (confidence: 92.1%)
  → MSG_130_ThrottlePosition (confidence: 97.4%)

Event: left_turn_signal @ 25.3s
  → MSG_400_TurnSignal (confidence: 99.2%)

Event: gear_shift_d_to_r @ 30.1s
  → MSG_280_GearPosition (confidence: 96.8%)
```

### Identify Special Field Types

```python
# Automatically identify counters, checksums, etc.
special_fields = session.identify_special_fields()

print(f"\n=== Special Fields ===\n")

print("Counters:")
for field in special_fields.counters:
    print(f"  {field.signal_name} (ID 0x{field.can_id:03X}): "
          f"modulo {field.modulo}")

print("\nChecksums:")
for field in special_fields.checksums:
    print(f"  {field.signal_name} (ID 0x{field.can_id:03X}): "
          f"{field.algorithm}")

print("\nAlive/Heartbeat counters:")
for field in special_fields.alive_counters:
    print(f"  {field.signal_name} (ID 0x{field.can_id:03X})")
```

**Expected Output:**

```
=== Special Fields ===

Counters:
  MSG_0C0_Counter (ID 0x0C0): modulo 256
  MSG_0D0_Counter (ID 0x0D0): modulo 256
  MSG_120_Counter (ID 0x120): modulo 16

Checksums:
  MSG_0C0_CRC (ID 0x0C0): CRC-8
  MSG_0D0_CRC (ID 0x0D0): CRC-8
  MSG_200_Checksum (ID 0x200): XOR

Alive/Heartbeat counters:
  MSG_300_Alive (ID 0x300)
```

---

## Part 4: Signal Naming and Annotation

### Auto-Generate Signal Names

```python
# Oscura generates names based on patterns
# You can add semantic meaning based on domain knowledge

# Rename signals based on correlation analysis
session.rename_signal("MSG_0C0_Data_1", "BrakePressure_Percent")
session.rename_signal("MSG_0C0_Data_2", "EngineTemp_Celsius")
session.rename_signal("MSG_130_Data_1", "ThrottlePosition_Percent")
session.rename_signal("MSG_400_Data_0", "TurnSignal_Status")

# Add units and descriptions
session.annotate_signal(
    "BrakePressure_Percent",
    unit="%",
    description="Brake pedal pressure",
    min_physical=0.0,
    max_physical=100.0
)

session.annotate_signal(
    "EngineTemp_Celsius",
    unit="°C",
    description="Engine coolant temperature",
    min_physical=-40.0,
    max_physical=150.0,
    offset=-40.0  # Raw value 0 = -40°C
)

# Add enum values
session.annotate_signal(
    "TurnSignal_Status",
    value_table={
        0: "OFF",
        1: "LEFT",
        2: "RIGHT",
        3: "HAZARD"
    }
)
```

### Bulk Annotation from CSV

```python
# Load annotations from CSV file
# CSV format: can_id, signal_name, unit, description, min, max, offset, scale
session.load_annotations("signal_annotations.csv")
```

Example `signal_annotations.csv`:

```csv
can_id,signal_name,unit,description,min,max,offset,scale
0x0C0,BrakePressure_Percent,%,Brake pedal pressure,0,100,0,0.01
0x0C0,EngineTemp_Celsius,°C,Engine coolant temperature,-40,150,-40,1
0x130,ThrottlePosition_Percent,%,Accelerator pedal position,0,100,0,0.1
0x280,VehicleSpeed_Kph,km/h,Vehicle speed from wheel sensors,0,255,0,1
```

---

## Part 5: DBC File Generation

### Generate DBC File Automatically

```python
# Export DBC file with all inferred signals
session.export_dbc(
    output="vehicle_protocol.dbc",
    version="1.0",
    include_comments=True,
    include_value_tables=True,
    validation=True  # Validate DBC syntax
)

print("DBC file generated: vehicle_protocol.dbc")
```

**Generated DBC file** (`vehicle_protocol.dbc`):

```dbc
VERSION ""

NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGtype_
    BA_SGtype_
    SIG_TYPE_REF_
    VAL_TABLE_
    SIG_GROUP_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    BA_DEF_REL_
    BA_REL_
    BA_SGtype_REL_
    SG_MUL_VAL_

BS_:

BU_: ECU_Engine ECU_Transmission ECU_ABS ECU_Instrument

BO_ 192 MSG_0C0: 8 ECU_Engine
 SG_ MSG_0C0_Counter : 0|8@1+ (1,0) [0|255] "" Vector__XXX
 SG_ BrakePressure_Percent : 8|16@1+ (0.01,0) [0|100] "%" Vector__XXX
 SG_ EngineTemp_Celsius : 24|16@1- (1,-40) [-40|150] "°C" Vector__XXX
 SG_ MSG_0C0_CRC : 56|8@1+ (1,0) [0|255] "" Vector__XXX

BO_ 208 MSG_0D0: 8 ECU_Transmission
 SG_ MSG_0D0_Counter : 0|8@1+ (1,0) [0|255] "" Vector__XXX
 SG_ GearPosition : 8|8@1+ (1,0) [0|7] "" Vector__XXX
 SG_ MSG_0D0_CRC : 56|8@1+ (1,0) [0|255] "" Vector__XXX

BO_ 304 MSG_130: 8 ECU_Engine
 SG_ ThrottlePosition_Percent : 0|16@1+ (0.1,0) [0|100] "%" Vector__XXX

BO_ 640 MSG_280: 8 ECU_Instrument
 SG_ VehicleSpeed_Kph : 0|8@1+ (1,0) [0|255] "km/h" Vector__XXX
 SG_ MSG_280_Alive : 8|4@1+ (1,0) [0|15] "" Vector__XXX

BO_ 1024 MSG_400: 2 ECU_Instrument
 SG_ TurnSignal_Status : 0|2@1+ (1,0) [0|3] "" Vector__XXX

CM_ SG_ 192 BrakePressure_Percent "Brake pedal pressure";
CM_ SG_ 192 EngineTemp_Celsius "Engine coolant temperature";
CM_ SG_ 304 ThrottlePosition_Percent "Accelerator pedal position";
CM_ SG_ 640 VehicleSpeed_Kph "Vehicle speed from wheel sensors";

BA_DEF_ SG_ "GenSigStartValue" FLOAT 0 1000000;
BA_DEF_DEF_ "GenSigStartValue" 0;

VAL_ 1024 TurnSignal_Status 0 "OFF" 1 "LEFT" 2 "RIGHT" 3 "HAZARD";
VAL_ 208 GearPosition 0 "P" 1 "R" 2 "N" 3 "D" 4 "D1" 5 "D2" 6 "D3";
```

### Validate DBC File

```python
# Validate generated DBC
validation_result = session.validate_dbc("vehicle_protocol.dbc")

if validation_result.is_valid:
    print("✓ DBC file is valid")
else:
    print("✗ DBC validation errors:")
    for error in validation_result.errors:
        print(f"  - {error}")

# Test DBC in Python
from cantools import database

db = database.load_file("vehicle_protocol.dbc")

# Decode a CAN message
msg = db.get_message_by_name("MSG_0C0")
data = bytes.fromhex("0A1234FF7890AB56")
decoded = msg.decode(data)

print(f"\nDecoded message:")
for signal_name, value in decoded.items():
    print(f"  {signal_name}: {value}")
```

**Expected Output:**

```
✓ DBC file is valid

Decoded message:
  MSG_0C0_Counter: 10
  BrakePressure_Percent: 46.76
  EngineTemp_Celsius: 65.0
  MSG_0C0_CRC: 86
```

---

## Part 6: Advanced Analysis

### Detect Message Dependencies

```python
# Find messages that always appear together
dependencies = session.find_message_dependencies()

print(f"\n=== Message Dependencies ===\n")

for msg_id, dependent_ids in dependencies.items():
    print(f"ID 0x{msg_id:03X} always appears with:")
    for dep_id in dependent_ids:
        print(f"  → 0x{dep_id:03X}")
```

### State Machine Extraction

```python
# Infer state machine from message sequences
state_machine = session.extract_state_machine()

print(f"\n=== Vehicle State Machine ===\n")
print(f"States: {len(state_machine.states)}")

for state in state_machine.states:
    print(f"\nState: {state.name}")
    print(f"  Entry messages: {[f'0x{m:03X}' for m in state.entry_messages]}")
    print(f"  Active messages: {[f'0x{m:03X}' for m in state.active_messages]}")

print(f"\nTransitions: {len(state_machine.transitions)}")

for transition in state_machine.transitions:
    print(f"  {transition.from_state} → {transition.to_state}: "
          f"trigger=0x{transition.trigger_message:03X}")
```

### Security Analysis

```python
# Check for security issues
security_report = session.security_analysis()

print(f"\n=== Security Analysis ===\n")

print(f"Findings:")
print(f"  Messages without checksums: {len(security_report.no_checksum)}")
print(f"  Messages without counters: {len(security_report.no_counter)}")
print(f"  Potential replay attacks: {security_report.replay_vulnerable}")
print(f"  No authentication: {security_report.no_authentication}")

if security_report.no_checksum:
    print(f"\nMessages vulnerable to corruption:")
    for msg_id in security_report.no_checksum[:5]:
        print(f"  ID 0x{msg_id:03X}")
```

---

## Part 7: Export to Other Formats

### Export to KCD (CANdb++)

```python
# Export to KCD format for use with CANdb++
session.export_kcd("vehicle_protocol.kcd")
```

### Export to ARXML (AUTOSAR)

```python
# Export to AUTOSAR XML
session.export_arxml("vehicle_protocol.arxml", version="4.3.0")
```

### Export to Wireshark Dissector

```python
# Generate Wireshark dissector for CAN traffic analysis
session.export_wireshark_dissector(
    output="can_vehicle.lua",
    protocol_name="vehicle_can"
)
```

### Export to Python/Scapy

```python
# Generate Python code for packet crafting
session.export_scapy_definitions("vehicle_can.py")

# Use generated code
from vehicle_can import MSG_0C0

# Create message
msg = MSG_0C0(
    MSG_0C0_Counter=42,
    BrakePressure_Percent=75.5,
    EngineTemp_Celsius=90.0
)

print(f"CAN message: {bytes(msg).hex()}")
```

---

## Part 8: Replay and Testing

### Replay CAN Messages

```python
from oscura.validation import CANReplay
import can

# Connect to CAN interface
bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)

# Replay captured traffic
replay = CANReplay(
    source_file="vehicle_capture.blf",
    target_bus=bus,
    real_time=True,  # Maintain original timing
    loop=False
)

# Start replay
replay.start()
print("Replaying CAN messages...")

# Monitor responses
monitor = replay.monitor_responses(timeout=10.0)
print(f"Received {len(monitor.responses)} responses")
```

### Send Test Messages

```python
# Send specific test message
from vehicle_can import MSG_0C0

# Create test message
test_msg = MSG_0C0(
    MSG_0C0_Counter=0,
    BrakePressure_Percent=50.0,
    EngineTemp_Celsius=80.0
)

# Send via CAN bus
can_msg = can.Message(
    arbitration_id=0x0C0,
    data=bytes(test_msg),
    is_extended_id=False
)

bus.send(can_msg)
print(f"Sent test message ID 0x0C0")

# Wait for response
response = bus.recv(timeout=1.0)
if response:
    print(f"Response ID 0x{response.arbitration_id:03X}: {response.data.hex()}")
```

---

## Summary

**What You Accomplished:**

1. Loaded and analyzed CAN bus capture
2. Classified messages (periodic, event-driven, multiplexed)
3. Automatically extracted signal boundaries
4. Correlated signals with vehicle behavior
5. Identified counters, checksums, and special fields
6. Annotated signals with semantic meaning
7. Generated DBC file automatically
8. Exported to multiple formats (KCD, ARXML, Wireshark)
9. Validated and tested generated artifacts

**Artifacts Created:**

- `vehicle_protocol.dbc` - CAN database file
- `vehicle_protocol.kcd` - CANdb++ format
- `vehicle_protocol.arxml` - AUTOSAR format
- `can_vehicle.lua` - Wireshark dissector
- `vehicle_can.py` - Python/Scapy definitions

**Time Saved:**

- **Manual DBC creation:** 10-20 hours (decode signals, test, document)
- **With Oscura:** 45-60 minutes (mostly annotation and validation)

---

## Next Steps

### Advanced Topics

1. **CAN-FD analysis:** Extended frames, higher bitrates
2. **Multi-network analysis:** Combine multiple CAN buses
3. **Diagnostic protocols:** UDS (ISO 14229) over CAN
4. **Simulation:** Use DBC with CARLA or other simulators

### Related Tutorials

- [UDS Diagnostic Protocol](uds-diagnostics.md)
- [Multi-Protocol Correlation](multi-protocol-analysis.md)
- [Vehicle Security Testing](automotive-security.md)

### Additional Resources

- [CAN Protocol Decoder API](../api/protocols/can.md)
- [CAN Session Guide](../api/sessions/can.md)
- [DBC Export Documentation](../api/export/dbc.md)
- [Signal Extraction Algorithm](../developer-guide/algorithms/signal-extraction.md)

---

## Troubleshooting

**Problem:** Signal boundaries incorrectly detected

**Solution:**

```python
# Manual signal definition if auto-detection fails
session.define_signal_manually(
    can_id=0x0C0,
    name="CustomSignal",
    start_bit=16,
    length=12,
    byte_order="little_endian",
    value_type="unsigned",
    scale=0.1,
    offset=-100.0
)
```

**Problem:** Too many false positive correlations

**Solution:**

```python
# Increase correlation threshold
correlations = session.correlate_with_events(
    min_confidence=0.95,  # Require 95% confidence
    time_window=0.5        # ±500ms around event
)
```

**Problem:** DBC validation fails

**Solution:**

```python
# Export with strict validation disabled for manual review
session.export_dbc(
    output="vehicle_protocol.dbc",
    strict_validation=False,
    include_warnings_as_comments=True
)
```
