# Serial Protocols

> UART, SPI, I2C, 1-Wire decoding with auto-parameter detection

**Oscura Version**: 0.1.2 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers serial protocol decoding for UART, SPI, I2C, and 1-Wire interfaces. Oscura provides auto-baud detection, multi-mode support, and comprehensive frame decoding.

### Key Capabilities

- UART/RS-232 decoding with auto-baud
- SPI mode detection and decoding (CPOL/CPHA)
- I2C address/data extraction (7/10-bit)
- 1-Wire protocol decoding
- Multi-protocol correlation

### Protocol Support

|Protocol|Features|
|---|---|
|UART|Auto-baud, 5-9 data bits, parity, 1-2 stop bits|
|SPI|CPOL/CPHA modes 0-3, MSB/LSB first, multi-slave|
|I2C|Standard/Fast/Fast+, 7/10-bit addressing, ACK/NACK|
|1-Wire|Standard/Overdrive, ROM commands, family codes|

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 03`)

### 30-Second Example

```python
from oscura.analyzers.protocols.uart import decode_uart, detect_baud_rate

# Auto-detect baud rate
baud = detect_baud_rate(signal, sample_rate=1e6)
print(f"Detected baud: {baud}")

# Decode UART frames
frames = decode_uart(signal, sample_rate=1e6, baud_rate=115200)
for frame in frames:
    print(f"Data: 0x{frame.data:02X}")
```

---

## Demo Scripts

|Script|Purpose|Complexity|
|---|---|---|
|`uart_decoding.py`|UART with auto-baud detection|Basic|
|`spi_analysis.py`|SPI mode detection, MOSI/MISO|Basic|
|`i2c_transaction.py`|I2C address scanning|Intermediate|
|`one_wire_devices.py`|1-Wire device enumeration|Intermediate|
|`multi_protocol.py`|Correlating multiple protocols|Advanced|

## Related Demos

- [04_automotive_protocols](../04_automotive_protocols/) - CAN, LIN, FlexRay
- [05_debug_protocols](../05_debug_protocols/) - JTAG, SWD
- [11_protocol_inference](../11_protocol_inference/) - Auto protocol detection

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
