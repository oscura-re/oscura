# Automotive Protocols

> CAN, CAN-FD, LIN, FlexRay transport layer analysis

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers automotive protocol transport layer decoding for CAN, CAN-FD, LIN, and FlexRay. Oscura provides DBC file integration and J1939 transport layer support.

### Key Capabilities

- CAN 2.0B frame decoding (11/29-bit IDs)
- CAN-FD support with BRS
- LIN protocol decoding (v1.x/2.x)
- FlexRay static/dynamic segments
- DBC file integration
- J1939 transport layer

### Standards Compliance

| Standard  | Coverage | Notes                  |
| --------- | -------- | ---------------------- |
| ISO 11898 | Full     | CAN/CAN-FD             |
| ISO 17987 | Full     | LIN                    |
| ISO 17458 | Full     | FlexRay                |
| SAE J1939 | Full     | Heavy-duty vehicle CAN |

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .[automotive]`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 04`)

### 30-Second Example

```python
from oscura.analyzers.protocols.can import decode_can

frames = decode_can(signal=can_signal, sample_rate=10e6, bit_rate=500000)
for frame in frames:
    id_type = "EXT" if frame.extended_id else "STD"
    print(f"ID: 0x{frame.arbitration_id:X} ({id_type}), Data: {frame.data.hex()}")
```

---

## Demo Scripts

| Script               | Purpose                      | Complexity   |
| -------------------- | ---------------------------- | ------------ |
| `can_bus_basics.py`  | CAN frame decoding           | Basic        |
| `can_fd_demo.py`     | CAN-FD extended frames       | Intermediate |
| `lin_single_wire.py` | LIN protocol analysis        | Intermediate |
| `flexray_intro.py`   | FlexRay time-triggered       | Advanced     |
| `dbc_integration.py` | DBC file for signal decoding | Intermediate |
| `j1939_transport.py` | Heavy vehicle J1939          | Advanced     |

## Related Demos

- [12_automotive_diagnostics](../12_automotive_diagnostics/) - OBD-II, UDS, J1939 diagnostics
- [03_serial_protocols](../03_serial_protocols/) - UART, SPI, I2C
- [11_protocol_inference](../11_protocol_inference/) - Unknown protocol detection

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
