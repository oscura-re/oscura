# Remaining Protocol Decoding Demos - Implementation Guide

This document provides implementation templates for the remaining 7 protocol demos in Category 03.

## Template Structure

Each demo should follow this pattern (see completed demos for examples):

```python
"""Protocol Name: Description

Demonstrates:
- oscura.decode_protocol() - Protocol features

IEEE Standards: [Standard references]
Related Demos: [List related demos]
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from demos.common import BaseDemo, ValidationSuite
from oscura import decode_protocol
from oscura.core.types import DigitalTrace, TraceMetadata


class ProtocolDemo(BaseDemo):
    def __init__(self) -> None:
        super().__init__(
            name="protocol_name",
            description="Description",
            capabilities=["oscura.decode_protocol"],
            ieee_standards=["Standard"],
            related_demos=["Related demos"],
        )

    def generate_test_data(self) -> dict:
        """Generate synthetic signals."""
        # Implementation
        pass

    def run_demonstration(self, data: dict) -> dict:
        """Decode signals."""
        # Implementation
        pass

    def validate(self, results: dict) -> bool:
        """Validate results."""
        suite = ValidationSuite("Protocol Validation")
        # Add checks
        suite.print_summary()
        return suite.passed


if __name__ == "__main__":
    demo = ProtocolDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
```

## 05_can_fd.py

**Source Reference**: `demonstrations/03_protocol_decoding/02_automotive_protocols.py` (lines 326-481)

**Key Implementation Points**:

- Dual bitrate: nominal_bitrate=500000, data_bitrate=2000000
- Extended payload: up to 64 bytes
- BRS and FDF flags in control field
- DLC to length code conversion (_dlc_to_length_code helper)
- CRC17 for CAN-FD (17 bits instead of 15)

**Signal Generation**:

```python
def _generate_can_fd_signal(
    self,
    frame_id: int,
    is_extended: bool,
    data: bytes,  # Up to 64 bytes
    nominal_bitrate: int,
    data_bitrate: int,
    sample_rate: float,
) -> DigitalTrace:
    # Use two different bit times:
    # - nominal phase: arbitration, control
    # - data phase: payload, CRC (faster)
```

**Test Configurations**:

1. Short payload (8 bytes) with BRS
2. Medium payload (16 bytes)
3. Long payload (64 bytes)
4. Mixed standard/extended IDs

## 06_lin.py

**Source References**:

- `demonstrations/03_protocol_decoding/02_automotive_protocols.py` (lines 483-623)
- `demonstrations/05_domain_specific/automotive_lin.py` (enhanced automotive features)

**Key Implementation Points**:

- Break field: 13 dominant bits
- Sync field: 0x55 pattern
- Protected ID (PID): 6-bit ID + 2 parity bits
- Checksum: Classic vs Enhanced (LIN 2.x)
- Typical baudrate: 19200 bps

**Signal Generation**:

```python
# Break field (13 bits low)
signal.extend([0] * (13 * samples_per_bit))

# Sync byte (0x55)
for bit in [0,1,0,1,0,1,0,1]:  # LSB first
    signal.extend([bit] * samples_per_bit)

# Protected ID with parity
pid = frame_id & 0x3F
p0 = parity_bit_0(pid)
p1 = parity_bit_1(pid)
pid_with_parity = pid | (p0 << 6) | (p1 << 7)
```

**Test Configurations**:

1. Standard frame (ID < 0x3C)
2. Diagnostic frame (ID = 0x3C or 0x3D)
3. Reserved frame (ID > 0x3D)

## 07_flexray.py

**Source References**:

- `demonstrations/03_protocol_decoding/02_automotive_protocols.py` (lines 626-778)
- `demonstrations/05_domain_specific/automotive_flexray.py`

**Key Implementation Points**:

- Differential signaling: BP (Bus Plus) and BM (Bus Minus)
- Idle state: BP=0, BM=1
- Bit encoding: Recessive (BP=0, BM=1), Dominant (BP=1, BM=0)
- Frame header: sync pattern, slot ID, payload length
- Typical bitrate: 10 Mbps
- Static and dynamic segments

**Signal Generation**:

```python
def _generate_flexray_signals(
    self,
    slot_id: int,
    data: bytes,
    bitrate: int,
    sample_rate: float,
) -> tuple[DigitalTrace, DigitalTrace]:  # Returns (BP, BM)
    # Generate complementary BP/BM signals
    # Idle: BP=0, BM=1
    # Bit 0: BP=1, BM=0
    # Bit 1: BP=0, BM=1
```

**Test Configurations**:

1. Static segment frame
2. Dynamic segment frame
3. Startup frame
4. Sync frame

## 09_swd.py

**Source Reference**: `demonstrations/03_protocol_decoding/swd.py`

**Key Implementation Points**:

- 2-wire protocol: SWDCLK, SWDIO
- Packet structure: Start + APnDP + RnW + Addr[2:3] + Parity + Stop + Park + ACK[0:2] + Data[0:31] + Parity
- ACK responses: OK (0b001), WAIT (0b010), FAULT (0b100)
- Turnaround cycles between host and target
- Parity bit for data integrity

**Signal Generation**:

```python
def _generate_swd_transaction(
    self,
    is_ap: bool,  # True for AP, False for DP
    is_read: bool,
    address: int,  # 2-bit address
    data: int,  # 32-bit data
    ack: int,  # ACK value (1=OK, 2=WAIT, 4=FAULT)
    clock_freq: float,
    sample_rate: float,
) -> tuple[DigitalTrace, DigitalTrace]:  # Returns (SWDCLK, SWDIO)
```

**Test Configurations**:

1. DP register read (IDCODE)
2. AP register read (CSW)
3. AP register write
4. WAIT response handling

## 10_i2s.py

**Source Reference**: `demonstrations/03_protocol_decoding/i2s.py`

**Key Implementation Points**:

- 3-wire interface: SCK (bit clock), WS (word select), SD (serial data)
- WS signal: 0=Left channel, 1=Right channel
- Standard I2S: Data MSB 1 BCK cycle after WS change
- Left-justified: Data MSB at WS change
- Right-justified: Data LSB at WS change
- Common sample rates: 44.1 kHz, 48 kHz, 96 kHz
- Bit depths: 16, 24, 32-bit

**Signal Generation**:

```python
def _generate_i2s_signals(
    self,
    left_samples: list[int],
    right_samples: list[int],
    audio_sample_rate: int,
    bit_depth: int,
    mode: str,  # "standard", "left_justified", "right_justified"
    capture_sample_rate: float,
) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace]:  # (SCK, WS, SD)
```

**Test Configurations**:

1. Standard mode, 16-bit, 48 kHz
2. Left-justified, 24-bit, 96 kHz
3. Right-justified, 32-bit, 44.1 kHz
4. Stereo sine wave test pattern

## 11_usb.py

**Source Reference**: `demonstrations/03_protocol_decoding/usb.py`

**Important Note**: File has `# SKIP_VALIDATION` comment due to PID validation issues in decoder.

**Key Implementation Points**:

- Low-Speed USB: 1.5 Mbps
- Differential signaling: D+ and D-
- NRZI encoding: 0 = transition, 1 = no transition
- Bit stuffing: After six consecutive 1s, insert 0
- Packet types:
  - Token: OUT, IN, SOF, SETUP
  - Data: DATA0, DATA1
  - Handshake: ACK, NAK, STALL
- SYNC pattern: 0x80 after NRZI decoding
- CRC5 for token packets, CRC16 for data packets

**Signal Generation**:

```python
def _generate_usb_packet(
    self,
    packet_type: str,  # "OUT", "IN", "DATA0", "ACK", etc.
    address: int,  # 7-bit device address
    endpoint: int,  # 4-bit endpoint
    data: bytes,  # For data packets
    sample_rate: float,
) -> tuple[DigitalTrace, DigitalTrace]:  # (D+, D-)
    # NRZI encode
    # Add bit stuffing
    # Apply differential encoding
```

**Test Configurations**:

1. SETUP token
2. DATA0 packet with payload
3. ACK handshake
4. IN token + DATA1 response

**Known Issues**: Decoder may have PID validation issues - validation should be lenient.

## 12_comprehensive_protocols.py

**Source Reference**: `demonstrations/03_protocol_decoding/protocol_comprehensive.py`

**Key Implementation Points**:

- Combines UART, SPI, I2C decodings in single demo
- Uses SignalBuilder for synthetic data generation
- Demonstrates auto protocol detection
- Multi-channel synchronization
- Complete workflow from generation to validation

**Structure**:

```python
def generate_test_data(self) -> dict:
    """Generate all protocol signals."""
    self._generate_uart()
    self._generate_spi()
    self._generate_i2c()
    # Optional: CAN, others
    return {}

def run_demonstration(self, data: dict) -> dict:
    """Decode all protocols."""
    uart_results = self._decode_uart()
    spi_results = self._decode_spi()
    i2c_results = self._decode_i2c()
    return {"uart": uart_results, "spi": spi_results, "i2c": i2c_results}
```

**Test Configurations**:

1. Concurrent UART + SPI + I2C
2. Sequential protocol switching
3. Auto-detection mode
4. Multi-device bus (I2C with multiple addresses)

## Implementation Priority

Recommended order based on complexity:

1. **06_lin.py** - Similar to UART, straightforward UART-like framing
2. **05_can_fd.py** - Extends 04_can_basic.py, reuse CAN generation logic
3. **09_swd.py** - Similar to JTAG (08_jtag.py), 2-wire instead of 4-wire
4. **10_i2s.py** - Straightforward 3-wire serial with clock/data/select
5. **07_flexray.py** - More complex differential signaling
6. **11_usb.py** - Complex NRZI + bit stuffing + CRC (note validation issues)
7. **12_comprehensive_protocols.py** - Combines existing demos

## Testing Commands

After implementation, test each demo:

```bash
# Syntax validation
python -m py_compile demos/03_protocol_decoding/05_can_fd.py

# Execution test
python demos/03_protocol_decoding/05_can_fd.py

# Verbose output
python demos/03_protocol_decoding/05_can_fd.py --verbose
```

## Common Pitfalls

1. **Synthetic Signal Limitations**: Decoder may not detect packets on perfectly synthesized signals - validation should be lenient (warning instead of error).

2. **Timing Precision**: Ensure samples_per_bit calculations account for floating point precision.

3. **Edge Cases**: Test empty data, maximum data, boundary conditions.

4. **Import Paths**: Always use `from demos.common import BaseDemo, ValidationSuite`.

5. **Metadata**: Include all required fields: capabilities, ieee_standards, related_demos.

## Update Checklist

When implementing a demo:

- [ ] Copy template structure
- [ ] Implement signal generation with correct protocol timing
- [ ] Add decoder invocation with proper parameters
- [ ] Create 3-4 test configurations
- [ ] Add ValidationSuite checks
- [ ] Update ieee_standards and related_demos metadata
- [ ] Syntax validate with py_compile
- [ ] Test execution
- [ ] Update MIGRATION_STATUS_03-05.md
- [ ] Update 03_protocol_decoding/README.md

---

**Note**: All source files remain in `demonstrations/` and `demos/` for reference. Do not delete source files until migration is complete and validated.
