# Protocol Inference

> State machine learning, CRC reverse engineering, message format inference, Wireshark dissector generation

**Oscura Version**: 0.3.0 | **Last Updated**: 2026-01-16 | **Status**: Production

---

## Overview

This demo category covers protocol reverse engineering and inference capabilities. Oscura provides state machine learning (RPNI), CRC polynomial recovery, message structure discovery, and Wireshark Lua dissector generation.

### Key Capabilities

- Auto protocol detection
- Baud rate recovery
- Logic family detection (TTL/CMOS/LVDS)
- State machine learning (RPNI algorithm)
- CRC polynomial reverse engineering
- Message format/field boundary inference
- Wireshark Lua dissector generation
- Sequence alignment

---

## Quick Start

### Prerequisites

- Oscura installed (`uv pip install -e .`)
- Demo data generated (`python demos/generate_all_demo_data.py --demos 11`)

### 30-Second Example

```python
from oscura.inference.crc_reverse import reverse_crc, STANDARD_CRCS

# CRC reverse engineering
messages = [b"\x01\x02\x03", b"\x04\x05\x06", b"\x07\x08\x09"]
crcs = [0x1234, 0x5678, 0x9ABC]

result = reverse_crc(messages, crcs, width=16)
print(f"Polynomial: 0x{result.polynomial:04X}")
print(f"Init value: 0x{result.init:04X}")
print(f"Match: {result.name if result.name else 'Custom'}")
```

---

## Demo Scripts

| Script                      | Purpose                         | Complexity   |
| --------------------------- | ------------------------------- | ------------ |
| `auto_protocol_detect.py`   | Unknown protocol identification | Basic        |
| `baud_rate_recovery.py`     | Baud rate from signal           | Basic        |
| `logic_level_detect.py`     | TTL/CMOS/LVDS identification    | Basic        |
| `state_machine_learning.py` | RPNI automaton inference        | Advanced     |
| `crc_reverse.py`            | CRC polynomial recovery         | Intermediate |
| `message_format.py`         | Field boundary detection        | Intermediate |
| `wireshark_dissector.py`    | Lua dissector generation        | Advanced     |

## Related Demos

- [03_serial_protocols](../03_serial_protocols/) - Known protocol decoding
- [14_advanced_inference](../14_advanced_inference/) - Advanced ML techniques
- [15_complete_workflows](../15_complete_workflows/) - Full RE workflows

---

## Validation

This demo includes self-validation. All examples are tested in CI via `demos/validate_all_demos.py`.
