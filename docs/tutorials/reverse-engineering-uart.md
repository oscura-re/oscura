# Tutorial: Reverse Engineering Unknown UART Protocol

This tutorial demonstrates complete reverse engineering of an unknown UART-based protocol using Oscura's differential analysis capabilities.

**Scenario:** You've connected to debug pins on an IoT device and captured UART traffic, but the protocol is undocumented.

**Time Required:** 30-45 minutes
**Difficulty:** Intermediate
**Prerequisites:** Basic Python knowledge, oscilloscope or logic analyzer

---

## Part 1: Capture Collection

### Equipment Setup

```
IoT Device TX -----> Logic Analyzer CH0 (RX)
IoT Device RX <----- Logic Analyzer CH1 (TX)
           GND ----- Logic Analyzer GND
```

### Capture Different Device States

Collect captures showing different behaviors:

1. **Power-on sequence:** Device boot messages
2. **Idle state:** Normal operation, no user interaction
3. **Button press:** User triggers some action
4. **Sensor event:** Temperature sensor triggers alert
5. **Configuration change:** Modify device settings

```bash
# Capture files
captures/
  ├── 01_power_on.wfm       # Boot sequence
  ├── 02_idle.wfm           # 30 seconds of idle
  ├── 03_button_press.wfm   # User button pressed
  ├── 04_temp_alert.wfm     # Temperature exceeded threshold
  └── 05_config_change.wfm  # WiFi SSID changed
```

---

## Part 2: Initial Analysis - Find UART Parameters

### Auto-Detect Baud Rate

```python
import oscura as osc
from oscura.analyzers.protocols import UARTDecoder

# Load capture
waveform = osc.load("captures/01_power_on.wfm")

# Auto-detect UART parameters
params = UARTDecoder.auto_detect(
    waveform=waveform,
    channel=0,  # TX from device
    baud_rate_range=(9600, 921600),
    common_rates_first=True  # Try 9600, 115200, etc. first
)

print(f"Detected UART parameters:")
print(f"  Baud rate: {params.baud_rate}")
print(f"  Data bits: {params.data_bits}")
print(f"  Parity: {params.parity}")
print(f"  Stop bits: {params.stop_bits}")
print(f"  Confidence: {params.confidence:.2%}")
```

**Expected Output:**

```
Detected UART parameters:
  Baud rate: 115200
  Data bits: 8
  Parity: N
  Stop bits: 1
  Confidence: 98.5%
```

### Decode Boot Messages

```python
# Create decoder with detected parameters
decoder = UARTDecoder(
    baud_rate=115200,
    data_bits=8,
    parity='N',
    stop_bits=1
)

# Decode messages
messages = decoder.decode(waveform)

# Display first 10 messages
print(f"\nDecoded {len(messages)} messages:")
for i, msg in enumerate(messages[:10]):
    timestamp = msg.timestamp
    data_hex = msg.data.hex()

    # Try ASCII decode
    try:
        data_ascii = msg.data.decode('ascii')
        print(f"{i+1}. [{timestamp:.6f}s] {data_hex} | {data_ascii}")
    except:
        print(f"{i+1}. [{timestamp:.6f}s] {data_hex} | (non-ASCII)")
```

**Expected Output:**

```
Decoded 42 messages:
1. [0.001234s] aa5501000042 | (non-ASCII)
2. [0.052100s] aa5502001043 | (non-ASCII)
3. [0.104520s] aa5503002044 | (non-ASCII)
4. [0.156780s] aa5504003045 | (non-ASCII)
5. [0.208901s] aa5505004046 | (non-ASCII)
...
```

**Observation:** Messages appear binary (not ASCII), have consistent structure starting with `aa55`.

---

## Part 3: Differential Analysis

### Set Up Black Box Session

```python
from oscura.sessions import BlackBoxSession

# Create analysis session
session = BlackBoxSession(
    name="IoT Device UART RE",
    description="Reverse engineering proprietary IoT protocol"
)

# Add all captures
session.add_recording("power_on", "captures/01_power_on.wfm")
session.add_recording("idle", "captures/02_idle.wfm")
session.add_recording("button_press", "captures/03_button_press.wfm")
session.add_recording("temp_alert", "captures/04_temp_alert.wfm")
session.add_recording("config_change", "captures/05_config_change.wfm")

# Configure UART decoder for session
session.set_decoder(UARTDecoder(baud_rate=115200))
```

### Compare Idle vs Button Press

```python
# Differential analysis
diff = session.compare("idle", "button_press")

print(f"\nDifferential Analysis Results:")
print(f"  Messages in idle: {diff.count_baseline}")
print(f"  Messages in button_press: {diff.count_compare}")
print(f"  New messages: {diff.new_messages}")
print(f"  Changed fields: {len(diff.changed_fields)}")

# Show what changed
print(f"\nStatic header (unchanged):")
print(f"  {diff.static_prefix.hex()}")

print(f"\nChanging bytes:")
for field in diff.changed_fields:
    print(f"  Byte {field.offset}: "
          f"idle={field.baseline_value:02x}, "
          f"button={field.compare_value:02x}, "
          f"delta={field.compare_value - field.baseline_value:+d}")
```

**Expected Output:**

```
Differential Analysis Results:
  Messages in idle: 120
  Messages in button_press: 135
  New messages: 15
  Changed fields: 3

Static header (unchanged):
  aa55

Changing bytes:
  Byte 2: idle=01, button=03, delta=+2
  Byte 5: idle=00, button=64, delta=+100
  Byte 7: idle=42, button=a6, delta=+100
```

**Hypothesis:** Byte 2 might be message type (01=status, 03=event). Byte 5 changes significantly (0x00 → 0x64 = 100 decimal). Byte 7 changes by same amount - likely checksum!

### Automated Pattern Detection

```python
# Let Oscura detect patterns automatically
patterns = session.find_patterns()

print(f"\nDetected Patterns:")

# Sequential counters
if patterns.counters:
    print(f"\nCounter fields:")
    for counter in patterns.counters:
        print(f"  Byte {counter.offset}: "
              f"range={counter.min_value}-{counter.max_value}, "
              f"increment={counter.increment}")

# Constant fields
if patterns.constants:
    print(f"\nConstant fields (likely magic numbers/version):")
    for const in patterns.constants:
        print(f"  Bytes {const.offset}-{const.offset+const.length}: "
              f"{const.value.hex()}")

# Entropy analysis (detect encrypted/compressed regions)
if patterns.high_entropy_regions:
    print(f"\nHigh entropy regions (possible crypto/compression):")
    for region in patterns.high_entropy_regions:
        print(f"  Bytes {region.start}-{region.end}: "
              f"entropy={region.entropy:.2f} bits")
```

**Expected Output:**

```
Detected Patterns:

Counter fields:
  Byte 3: range=0-255, increment=1

Constant fields (likely magic numbers/version):
  Bytes 0-1: aa55
  Byte 4: 00

High entropy regions (possible crypto/compression):
  (none detected)
```

---

## Part 4: CRC/Checksum Recovery

### Automatic CRC Detection

```python
# Oscura automatically tests common CRC algorithms
crc_results = session.recover_checksum()

if crc_results.found:
    print(f"\nChecksum recovered!")
    print(f"  Type: {crc_results.algorithm}")
    print(f"  Polynomial: 0x{crc_results.polynomial:X}")
    print(f"  Initial value: 0x{crc_results.init:X}")
    print(f"  XOR out: 0x{crc_results.xor_out:X}")
    print(f"  Reflected: {crc_results.reflected}")
    print(f"  Checksum offset: byte {crc_results.checksum_offset}")
    print(f"  Checksum length: {crc_results.checksum_length} bytes")
    print(f"  Confidence: {crc_results.confidence:.2%}")

    # Validate on all messages
    validation = crc_results.validate_all_messages()
    print(f"  Validation: {validation.passed}/{validation.total} messages")
else:
    print("No standard CRC found. May use custom checksum.")
```

**Expected Output:**

```
Checksum recovered!
  Type: CRC-8
  Polynomial: 0x07
  Initial value: 0x00
  XOR out: 0x00
  Reflected: False
  Checksum offset: byte 7
  Checksum length: 1 bytes
  Confidence: 100.0%
  Validation: 275/275 messages
```

### Manual CRC Verification (Optional)

```python
from oscura.inference.crc_reverse import CRCReverser

# If auto-detection fails, manual search
reverser = CRCReverser()

# Provide message-checksum pairs
samples = [
    (bytes.fromhex("aa5501000000"), 0x42),  # message → checksum
    (bytes.fromhex("aa5502001000"), 0x43),
    (bytes.fromhex("aa5503002000"), 0x44),
]

# Search for CRC parameters
crc_params = reverser.find_crc(samples)

if crc_params:
    print(f"Found CRC: {crc_params}")
```

---

## Part 5: Protocol Specification Generation

### Generate Complete Protocol Spec

```python
# Automatic protocol structure inference
spec = session.generate_protocol_spec()

print(f"\nInferred Protocol Specification:")
print(f"{'='*60}")

print(f"\nMessage Structure ({spec.message_length} bytes):")
print(f"  Magic number: {spec.header_pattern.hex()} (2 bytes)")

for field in spec.fields:
    print(f"\n  Field: {field.name}")
    print(f"    Offset: {field.offset}")
    print(f"    Size: {field.size} bytes")
    print(f"    Type: {field.inferred_type}")

    if field.inferred_type == "enum":
        print(f"    Values: {field.possible_values}")
    elif field.inferred_type == "counter":
        print(f"    Range: {field.min_value} - {field.max_value}")
    elif field.inferred_type == "checksum":
        print(f"    Algorithm: {field.checksum_algorithm}")

# Export specification to markdown
spec.export_markdown("PROTOCOL_SPEC.md")
```

**Expected Output:**

```
Inferred Protocol Specification:
============================================================

Message Structure (8 bytes):
  Magic number: aa55 (2 bytes)

  Field: magic
    Offset: 0
    Size: 2 bytes
    Type: constant

  Field: message_type
    Offset: 2
    Size: 1 bytes
    Type: enum
    Values: {0x01: 'status', 0x02: 'response', 0x03: 'event'}

  Field: sequence
    Offset: 3
    Size: 1 bytes
    Type: counter
    Range: 0 - 255

  Field: version
    Offset: 4
    Size: 1 bytes
    Type: constant

  Field: payload
    Offset: 5
    Size: 2 bytes
    Type: data

  Field: checksum
    Offset: 7
    Size: 1 bytes
    Type: checksum
    Algorithm: CRC-8 (poly=0x07)
```

### Add Semantic Meaning (Manual Annotation)

```python
# Add your domain knowledge to improve spec
session.annotate_field(
    field_name="payload",
    description="Temperature value in 0.1°C units",
    unit="decidegrees_celsius",
    example="0x0064 = 10.0°C"
)

session.annotate_field(
    field_name="message_type",
    values={
        0x01: "Periodic status heartbeat",
        0x02: "Response to command",
        0x03: "Asynchronous event notification"
    }
)

# Re-export with annotations
spec_annotated = session.generate_protocol_spec()
spec_annotated.export_markdown("PROTOCOL_SPEC_ANNOTATED.md")
```

---

## Part 6: Export Artifacts

### Generate Wireshark Dissector

```python
# Export Lua dissector for Wireshark
session.export_results(
    format="dissector",
    output="iot_device_proto.lua",
    protocol_name="iot_device",
    protocol_description="IoT Device Proprietary Protocol",
    validate=True  # Validate Lua syntax
)

print("Wireshark dissector generated: iot_device_proto.lua")
print("\nTo use in Wireshark:")
print("  1. Copy to ~/.local/lib/wireshark/plugins/")
print("  2. Restart Wireshark")
print("  3. Apply filter: iot_device")
```

**Generated Lua dissector** (`iot_device_proto.lua`):

```lua
-- Auto-generated by Oscura
-- Protocol: IoT Device Proprietary Protocol

local iot_device_proto = Proto("iot_device", "IoT Device Protocol")

-- Field definitions
local f_magic = ProtoField.uint16("iot_device.magic", "Magic", base.HEX)
local f_msg_type = ProtoField.uint8("iot_device.type", "Message Type", base.HEX)
local f_sequence = ProtoField.uint8("iot_device.seq", "Sequence", base.DEC)
local f_version = ProtoField.uint8("iot_device.version", "Version", base.HEX)
local f_payload = ProtoField.uint16("iot_device.payload", "Payload", base.DEC)
local f_checksum = ProtoField.uint8("iot_device.checksum", "CRC-8", base.HEX)

iot_device_proto.fields = {f_magic, f_msg_type, f_sequence, f_version, f_payload, f_checksum}

-- Message type lookup
local msg_types = {
    [0x01] = "Status",
    [0x02] = "Response",
    [0x03] = "Event"
}

function iot_device_proto.dissector(buffer, pinfo, tree)
    if buffer:len() < 8 then return end

    pinfo.cols.protocol = "IoT Device"

    local subtree = tree:add(iot_device_proto, buffer(), "IoT Device Protocol")

    -- Parse fields
    subtree:add(f_magic, buffer(0,2))
    local msg_type = buffer(2,1):uint()
    subtree:add(f_msg_type, buffer(2,1)):append_text(" (" .. (msg_types[msg_type] or "Unknown") .. ")")
    subtree:add(f_sequence, buffer(3,1))
    subtree:add(f_version, buffer(4,1))

    -- Temperature payload (convert to °C)
    local temp_raw = buffer(5,2):uint()
    local temp_c = temp_raw / 10.0
    subtree:add(f_payload, buffer(5,2)):append_text(" (" .. temp_c .. " °C)")

    -- Checksum with validation
    local calc_crc = crc8(buffer(0,7):bytes())
    local actual_crc = buffer(7,1):uint()
    local crc_item = subtree:add(f_checksum, buffer(7,1))
    if calc_crc == actual_crc then
        crc_item:append_text(" [correct]")
    else
        crc_item:append_text(" [incorrect, should be " .. string.format("0x%02x", calc_crc) .. "]")
    end

    -- Update info column
    pinfo.cols.info = msg_types[msg_type] or "Unknown"
end

-- CRC-8 implementation
function crc8(data)
    local crc = 0
    for i = 0, data:len()-1 do
        crc = crc ~ data:get_index(i)
        for j = 0, 7 do
            if (crc & 0x80) ~= 0 then
                crc = ((crc << 1) ~ 0x07) & 0xFF
            else
                crc = (crc << 1) & 0xFF
            end
        end
    end
    return crc
end

-- Register dissector
local uart_table = DissectorTable.get("wtap_encap")
uart_table:add(wtap.USER0, iot_device_proto)
```

### Generate Scapy Layer for Testing

```python
# Export Python Scapy layer for packet crafting
session.export_results(
    format="scapy",
    output="iot_device_layer.py"
)

print("Scapy layer generated: iot_device_layer.py")
```

**Generated Scapy layer** (`iot_device_layer.py`):

```python
from scapy.all import *

class IoTDeviceProto(Packet):
    """IoT Device Proprietary Protocol"""
    name = "IoTDevice"

    fields_desc = [
        XShortField("magic", 0xAA55),
        ByteEnumField("msg_type", 1, {1: "Status", 2: "Response", 3: "Event"}),
        ByteField("sequence", 0),
        ByteField("version", 0),
        ShortField("payload", 0),
        XByteField("checksum", 0)
    ]

    def post_build(self, pkt, pay):
        """Auto-calculate checksum"""
        if self.checksum == 0:
            crc = self._calc_crc8(pkt[:-1])
            pkt = pkt[:-1] + bytes([crc])
        return pkt + pay

    @staticmethod
    def _calc_crc8(data):
        """CRC-8 calculation (poly=0x07)"""
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

# Bind to UART
bind_layers(UART, IoTDeviceProto)

# Usage example
if __name__ == "__main__":
    # Create packet
    pkt = IoTDeviceProto(msg_type=1, sequence=42, payload=250)

    # Checksum auto-calculated
    print(pkt.show())
    print(f"Raw bytes: {bytes(pkt).hex()}")

    # Parse packet
    parsed = IoTDeviceProto(bytes.fromhex("aa5501002a00fa5c"))
    print(f"Parsed payload: {parsed.payload} (temp: {parsed.payload/10.0}°C)")
```

### Test Scapy Layer

```python
# Test the generated Scapy layer
from iot_device_layer import IoTDeviceProto

# Create status message
status = IoTDeviceProto(
    msg_type=1,
    sequence=10,
    version=0,
    payload=235  # 23.5°C
)

# Checksum auto-calculated
print(f"Crafted message: {bytes(status).hex()}")
# Output: aa55010a00eb**  (* = auto checksum)

# Parse received message
received = IoTDeviceProto(bytes.fromhex("aa5503140001f4a2"))
print(f"Message type: {received.msg_type}")  # 3 = Event
print(f"Sequence: {received.sequence}")      # 20
print(f"Temperature: {received.payload/10.0}°C")  # 50.0°C
```

---

## Part 7: Validation and Testing

### Replay Messages to Device

```python
from oscura.validation import replay_messages
import serial

# Connect to device
ser = serial.Serial('/dev/ttyUSB0', baudrate=115200)

# Replay known-good messages
test_messages = [
    bytes.fromhex("aa5501000000"),  # Status query
    bytes.fromhex("aa5502001000"),  # Read temperature
]

results = replay_messages(
    transport=ser,
    messages=test_messages,
    expect_response=True,
    timeout=1.0
)

for i, result in enumerate(results):
    print(f"\nMessage {i+1}:")
    print(f"  Sent: {test_messages[i].hex()}")
    print(f"  Response: {result.response.hex() if result.response else 'None'}")
    print(f"  Status: {result.status}")
```

### Fuzzing (Find Edge Cases)

```python
from oscura.testing import protocol_fuzzer

# Fuzz the protocol (send malformed messages)
fuzzer = protocol_fuzzer.ProtocolFuzzer(
    spec=spec,
    transport=ser,
    strategy="grammar_aware"  # Respects protocol structure
)

# Run 100 fuzz tests
results = fuzzer.run(iterations=100)

print(f"\nFuzzing results:")
print(f"  Tests run: {results.test_count}")
print(f"  Crashes: {results.crash_count}")
print(f"  Hangs: {results.hang_count}")
print(f"  Invalid responses: {results.invalid_response_count}")

# Save crash reproducers
for crash in results.crashes:
    print(f"\nCrash trigger: {crash.test_case.hex()}")
    crash.save("crashes/crash_{i}.bin")
```

---

## Part 8: Documentation

### Generate Comprehensive Report

```python
# Generate HTML report with all findings
session.export_results(
    format="report",
    output="uart_re_report.html",
    include=[
        "summary",
        "captures",
        "differential_analysis",
        "protocol_spec",
        "crc_recovery",
        "hypotheses",
        "test_vectors",
        "recommendations"
    ]
)

print("Report generated: uart_re_report.html")
```

The HTML report includes:

- Executive summary
- Capture inventory
- Differential analysis results with visualizations
- Complete protocol specification
- CRC/checksum recovery details
- Hypothesis testing audit trail
- Test message examples
- Security observations
- Recommendations for further analysis

---

## Summary

**What You Accomplished:**

1. Captured UART traffic in multiple device states
2. Auto-detected UART parameters (115200 baud, 8N1)
3. Performed differential analysis to identify changing fields
4. Automatically recovered CRC-8 checksum algorithm
5. Generated complete protocol specification
6. Exported Wireshark dissector (Lua)
7. Exported Scapy layer for packet crafting (Python)
8. Validated findings with message replay
9. Documented entire reverse engineering process

**Artifacts Created:**

- `PROTOCOL_SPEC.md` - Complete protocol documentation
- `iot_device_proto.lua` - Wireshark dissector
- `iot_device_layer.py` - Scapy packet definition
- `uart_re_report.html` - Comprehensive analysis report

**Time Saved:**

- **Without Oscura:** 4-8 hours (manual hex analysis, CRC brute force, dissector coding)
- **With Oscura:** 30-45 minutes (mostly annotation and validation)

---

## Next Steps

### Advanced Analysis

1. **State machine extraction:** Map message sequences to device states
2. **Command injection:** Test device responses to crafted messages
3. **Security analysis:** Look for authentication, replay protection
4. **Firmware correlation:** Match protocol to firmware strings/functions

### Related Tutorials

- [CAN Bus Analysis Tutorial](can-bus-analysis.md)

### Additional Resources

See the source code documentation for details on UART protocol decoding, BlackBox session management, CRC recovery, and Wireshark dissector export.

---

## Troubleshooting

**Problem:** Auto-detection fails to find baud rate

**Solution:**

```python
# Manually specify if auto-detection fails
decoder = UARTDecoder(
    baud_rate=115200,  # Try common rates: 9600, 19200, 38400, 57600, 115200
    tolerance=0.05      # Allow 5% baud rate mismatch
)
```

**Problem:** CRC recovery fails

**Solution:**

```python
# Expand search space
crc_results = session.recover_checksum(
    algorithms=["crc8", "crc16", "crc32", "sum", "xor"],
    custom_polynomials=[0x07, 0x31, 0x9B, 0x1D],  # Try custom polynomials
    brute_force=True  # Last resort: brute force search
)
```

**Problem:** Too many false positives in differential analysis

**Solution:**

```python
# Increase confidence threshold
diff = session.compare(
    "idle",
    "button_press",
    confidence_threshold=0.9,  # Require 90% confidence
    min_occurrences=3  # Field must change consistently
)
```
