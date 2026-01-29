# Protocol Decoding Demos

Demonstrates **comprehensive protocol decoding** for serial, automotive, and debug protocols using Oscura's extensive decoder library.

---

## Files in This Demo

1. **`comprehensive_protocol_demo.py`** ⭐ **COMPREHENSIVE**
   - UART, SPI, I2C decoding examples
   - Automatic baud rate detection
   - Protocol auto-detection
   - Multi-protocol analysis
   - Error detection and reporting

---

## Quick Start

### 1. Decode UART Communication

```bash
# Using synthetic signal (demonstration)
python demos/05_protocol_decoding/comprehensive_protocol_demo.py

# Using your capture file
python demos/05_protocol_decoding/comprehensive_protocol_demo.py \
    --file uart_capture.wfm \
    --protocol uart

# Auto-detect baud rate
python demos/05_protocol_decoding/comprehensive_protocol_demo.py \
    --file uart_capture.wfm \
    --protocol uart \
    --verbose
```

### 2. Auto-Detect Protocol

```bash
python demos/05_protocol_decoding/comprehensive_protocol_demo.py \
    --file mystery_protocol.wfm \
    --auto-detect
```

### 3. Analyze All Protocols

```bash
python demos/05_protocol_decoding/comprehensive_protocol_demo.py \
    --file multi_protocol.wfm \
    --all-protocols
```

---

## Supported Protocols

**Serial Protocols**:

- UART (RS-232, RS-485)
- SPI (all CPOL/CPHA modes)
- I2C (7-bit and 10-bit addressing)

**Automotive Protocols**:

- CAN (11-bit and 29-bit identifiers)
- CAN-FD (Flexible Data Rate)
- LIN (Local Interconnect Network)
- FlexRay

**Debug Protocols**:

- JTAG (Joint Test Action Group)
- SWD (Serial Wire Debug)

**Other Protocols**:

- 1-Wire
- I2S (Inter-IC Sound)
- Manchester encoding
- HDLC (High-Level Data Link Control)
- USB (Low/Full/High Speed)

---

## Features Demonstrated

### ✅ Automatic Baud Rate Detection

Tries common baud rates: 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600 bps

### ✅ Protocol Auto-Detection

Analyzes signal characteristics to identify protocol:

- Idle level (high/low)
- Transition patterns
- Clock detection
- Frame structure

### ✅ Comprehensive Frame Decoding

Extracts:

- Data bytes
- Parity bits
- Error flags (framing, parity)
- Timestamps
- ASCII representation

### ✅ Multi-Protocol Support

Handles captures with multiple protocols on different channels

---

## Python API Usage

```python
import oscura as osc

# Load capture
signal = osc.load("uart_capture.wfm")

# Decode UART
frames = osc.decode_uart(
    signal,
    baudrate=115200,
    data_bits=8,
    parity="none",
    stop_bits=1,
)

# Process frames
for frame in frames:
    print(f"Data: {frame.data.hex()}, Time: {frame.timestamp}")

# Decode SPI (multi-channel)
spi_frames = osc.decode_spi(
    signal,
    clock_channel="SCLK",
    data_channel="MOSI",
    cs_channel="CS",
    cpol=0,
    cpha=0,
)

# Decode I2C
i2c_frames = osc.decode_i2c(
    signal,
    scl_channel="SCL",
    sda_channel="SDA",
)

# Decode CAN
can_frames = osc.decode_can(
    signal,
    baudrate=500000,
)
```

---

## Common Issues

### Issue: "No frames decoded"

**Solution**: Check baud rate, signal polarity, and threshold levels

### Issue: "Framing errors"

**Solution**: Verify correct protocol parameters (data bits, parity, stop bits)

### Issue: "Multi-channel required"

**Solution**: SPI and I2C need multiple channels (clock + data)

---

## Related Documentation

- **Main demos**: `demos/README.md`
- **Signal RE**: `demos/04_signal_reverse_engineering/`
- **Examples**: `examples/04_protocol_decoding/`

---

**Last Updated**: 2026-01-15
**Status**: Production-ready
