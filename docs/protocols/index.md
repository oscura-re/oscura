# Protocol Catalog

Oscura supports 16+ built-in protocol decoders across automotive, industrial, IoT, and embedded system domains. This catalog provides comprehensive coverage information for each protocol.

## Overview by Category

| Category | Protocols | Coverage |
|----------|-----------|----------|
| **Automotive** | CAN, CAN-FD, LIN, FlexRay, UDS | Full |
| **Serial** | UART, SPI, I2C, 1-Wire | Full |
| **Debug** | JTAG, SWD | Full |
| **Industrial** | Modbus RTU, PROFIBUS | Partial |
| **Encoding** | Manchester, HDLC | Full |
| **USB** | USB 2.0/3.0 (protocol layer) | Partial |
| **Audio** | I2S | Full |

## Quick Reference

### By Use Case

**Reverse Engineering Unknown Protocol:**

- Start with [BlackBox Session](../api/sessions/blackbox.md)
- Use differential analysis
- Auto-detect protocol family

**Analyzing Known Protocol:**

- Use specific decoder (UART, CAN, etc.)
- See individual protocol documentation

**Automotive Analysis:**

- [CAN/CAN-FD](automotive.md#can-and-can-fd)
- [LIN](automotive.md#lin)
- [FlexRay](automotive.md#flexray)
- [UDS Diagnostics](automotive.md#uds)

**Embedded Debugging:**

- [UART](serial.md#uart)
- [SPI](serial.md#spi)
- [I2C](serial.md#i2c)
- [JTAG](debug.md#jtag)
- [SWD](debug.md#swd)

**Industrial Control:**

- [Modbus RTU](industrial.md#modbus-rtu)
- [PROFIBUS](industrial.md#profibus)

---

## Protocol Pages

Detailed documentation for each protocol family:

- **[Automotive Protocols](automotive.md)** - CAN, CAN-FD, LIN, FlexRay, UDS
- **[Serial Protocols](serial.md)** - UART, SPI, I2C, 1-Wire
- **[Debug Protocols](debug.md)** - JTAG, SWD
- **[Industrial Protocols](industrial.md)** - Modbus, PROFIBUS
- **[Encoding Schemes](encoding.md)** - Manchester, HDLC

---

## Coverage Legend

- **Full:** Complete specification support, all features implemented
- **Partial:** Core features implemented, some advanced features pending
- **Experimental:** Early implementation, API may change

---

## Protocol Support Matrix

### Automotive Protocols

| Protocol | Decoding | Encoding | Auto-Detect | DBC Export | Analysis |
|----------|----------|----------|-------------|------------|----------|
| CAN 2.0A/B | ✓ | ✓ | ✓ | ✓ | ✓ |
| CAN-FD | ✓ | ✓ | ✓ | ✓ | ✓ |
| LIN 1.x/2.x | ✓ | ✓ | ✓ | ✓ | ✓ |
| FlexRay | ✓ | ✓ | Partial | Partial | ✓ |
| UDS (ISO 14229) | ✓ | ✓ | ✓ | N/A | ✓ |

### Serial Protocols

| Protocol | Decoding | Encoding | Auto-Detect | Parameter Detection | Error Detection |
|----------|----------|----------|-------------|---------------------|-----------------|
| UART | ✓ | ✓ | ✓ | ✓ | ✓ |
| SPI | ✓ | ✓ | ✓ | ✓ | ✓ |
| I2C | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1-Wire | ✓ | ✓ | ✓ | ✓ | ✓ |

### Debug Protocols

| Protocol | Decoding | Encoding | Auto-Detect | Commands | Analysis |
|----------|----------|----------|-------------|----------|----------|
| JTAG | ✓ | ✓ | ✓ | Partial | ✓ |
| SWD | ✓ | ✓ | ✓ | Partial | ✓ |

### Industrial Protocols

| Protocol | Decoding | Encoding | Auto-Detect | Function Codes | Analysis |
|----------|----------|----------|-------------|----------------|----------|
| Modbus RTU | ✓ | ✓ | ✓ | ✓ | ✓ |
| PROFIBUS | ✓ | Partial | ✓ | Partial | ✓ |

---

## Auto-Detection Capabilities

Oscura can automatically detect many protocols from raw waveforms:

```python
from oscura.analyzers import auto_detect_protocol

# Automatic protocol detection
result = auto_detect_protocol(waveform)

print(f"Detected protocol: {result.protocol}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Parameters: {result.parameters}")
```

**Supported auto-detection:**

- **UART:** Baud rate, data bits, parity, stop bits
- **SPI:** Clock polarity, phase, bit order, mode
- **I2C:** Clock speed, addressing mode
- **CAN:** Bitrate, sample point, identifier format
- **LIN:** Baud rate, protocol version
- **Manchester:** Encoding variant, data rate

---

## Adding New Protocol Decoders

Oscura provides base classes for implementing custom protocol decoders:

```python
from oscura.analyzers.protocols import ProtocolDecoder

class MyCustomProtocol(ProtocolDecoder):
    """Custom protocol decoder."""

    def __init__(self, **params):
        super().__init__(name="my_custom_proto")
        # Initialize decoder parameters

    def decode(self, waveform):
        """Decode waveform and return messages."""
        # Implement decoding logic
        pass

    def encode(self, messages):
        """Encode messages to waveform."""
        # Implement encoding logic
        pass

    @classmethod
    def auto_detect(cls, waveform):
        """Auto-detect protocol parameters."""
        # Implement auto-detection
        pass
```

See [Developer Guide: Custom Protocol Decoders](../developer-guide/custom-protocols.md) for complete documentation.

---

## Protocol-Specific Examples

### Example: UART Decoding

```python
from oscura.analyzers.protocols import UARTDecoder

# Auto-detect UART parameters
params = UARTDecoder.auto_detect(waveform)

# Create decoder
decoder = UARTDecoder(
    baud_rate=params.baud_rate,
    data_bits=params.data_bits,
    parity=params.parity,
    stop_bits=params.stop_bits
)

# Decode messages
messages = decoder.decode(waveform)
```

### Example: CAN Analysis

```python
from oscura.sessions import CANSession

# Load and analyze CAN capture
session = CANSession(bitrate=500000)
session.load("capture.blf")

# Extract signals automatically
signals = session.extract_signals()

# Export DBC file
session.export_dbc("protocol.dbc")
```

### Example: Unknown Protocol

```python
from oscura.sessions import BlackBoxSession

# Create RE session
session = BlackBoxSession()

# Add captures from different states
session.add_recording("idle", "idle.bin")
session.add_recording("active", "active.bin")

# Differential analysis
diff = session.compare("idle", "active")

# Generate protocol spec
spec = session.generate_protocol_spec()

# Export artifacts
session.export_results("dissector", "protocol.lua")
```

---

## Performance Characteristics

Typical decoding performance on modern hardware:

| Protocol | Throughput | Latency (Real-time) | Notes |
|----------|------------|---------------------|-------|
| UART | 100 MB/s | <1ms | CPU-bound |
| SPI | 150 MB/s | <1ms | CPU-bound |
| I2C | 80 MB/s | <2ms | State machine overhead |
| CAN | 200 MB/s | <500μs | Optimized for automotive |
| FlexRay | 120 MB/s | <2ms | Complex framing |

_Benchmarked on Intel i7-9700K @ 3.6GHz_

---

## Known Limitations

### Current Limitations

1. **USB 3.0:** SuperSpeed decoding experimental
2. **FlexRay:** Symbol encoding not fully implemented
3. **PROFIBUS:** DP-V2 features partial
4. **CAN-FD:** BRS (Bit Rate Switching) timing analysis approximate

### Planned Improvements (v0.6.0+)

- BLE (Bluetooth Low Energy) decoder
- Zigbee/802.15.4 decoder
- LoRaWAN decoder
- Enhanced USB 3.0 support
- SOME/IP (Automotive Ethernet)

---

## Contributing Protocol Decoders

We welcome contributions of new protocol decoders! See:

- [Contributing Guide](../contributing.md)
- [Developer Guide: Protocol Decoders](../developer-guide/protocol-decoders.md)
- [Protocol Decoder Template](https://github.com/oscura-re/oscura/tree/main/templates/protocol_decoder.py)

---

## See Also

- [API Reference: Protocol Decoders](../api/protocols/)
- [Tutorial: Reverse Engineering Unknown UART](../tutorials/reverse-engineering-uart.md)
- [Tutorial: CAN Bus Analysis](../tutorials/can-bus-analysis.md)
- [FAQ: Protocol Detection](../faq/protocol-detection.md)
