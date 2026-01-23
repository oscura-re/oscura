# Protocol Decoding

**Master communication protocol decoding for serial, automotive, debug, and parallel bus protocols.**

This section contains 6 demonstrations covering 20+ industry-standard protocols including UART, SPI, I2C, CAN, USB, JTAG, and more. Learn how to extract, decode, and validate protocol packets from captured signals.

---

## Prerequisites

Before running these demonstrations, ensure you have:

- **Completed Getting Started** - Finish `00_getting_started/` first
- **Completed Basic Analysis** - Finish `02_basic_analysis/` recommended
- **Python 3.12+** - Oscura requires Python 3.12 or higher
- **Oscura installed** - Install with `pip install oscura` or `uv add oscura`
- **Protocol knowledge** - Basic understanding of serial/parallel communication helps
- **Digital signal basics** - Understanding logic levels and timing

---

## Demonstrations

| Demo      | File                         | Time        | Difficulty               | Topics                                      |
| --------- | ---------------------------- | ----------- | ------------------------ | ------------------------------------------- |
| **01**    | `01_serial_comprehensive.py` | 20 min      | Beginner                 | UART, SPI, I2C, 1-Wire                      |
| **02**    | `02_automotive_protocols.py` | 20 min      | Intermediate             | CAN, CAN-FD, LIN, FlexRay                   |
| **03**    | `03_debug_protocols.py`      | 15 min      | Intermediate             | JTAG, SWD, USB                              |
| **04**    | `04_parallel_bus.py`         | 15 min      | Advanced                 | IEEE-488 (GPIB), Centronics, ISA            |
| **05**    | `05_encoded_protocols.py`    | 15 min      | Advanced                 | Manchester, I2S, HDLC                       |
| **06**    | `06_auto_detection.py`       | 15 min      | Advanced                 | Protocol auto-detection, baud rate recovery |
| **Total** |                              | **100 min** | **Beginner to Advanced** | **20+ protocol decoders**                   |

---

## Learning Path

These demonstrations are designed to be completed **in order**. Each builds on concepts from the previous one:

```
01_serial_comprehensive.py → 02_automotive_protocols.py → 03_debug_protocols.py
        ↓                              ↓                           ↓
  Serial protocols             Automotive buses            Debug interfaces
  UART, SPI, I2C, 1-Wire      CAN, CAN-FD, LIN, FlexRay   JTAG, SWD, USB
  Baud rates, parity          Arbitration, error frames    TAP states, packets
        ↓                              ↓                           ↓
04_parallel_bus.py → 05_encoded_protocols.py → 06_auto_detection.py
        ↓                      ↓                          ↓
  Parallel buses           Encoded protocols        Auto-detection
  GPIB, Centronics        Manchester, I2S, HDLC    Protocol inference
  Multi-line data         Bit stuffing, encoding   Baud rate recovery
```

### Recommended Time

**Beginner path** (40 min): Demos 01, 02
**Intermediate path** (70 min): Demos 01-03, 06
**Advanced path** (100 min): All demos

---

## Key Concepts

### What You'll Learn

**Serial Protocols** (Demo 01):

- UART: Baud rate, parity, stop bits, framing
- SPI: Clock polarity/phase (CPOL/CPHA), chip select
- I2C: Start/stop conditions, ACK/NACK, addressing
- 1-Wire: Master/slave timing, search ROM

**Automotive Protocols** (Demo 02):

- CAN: Standard/extended frames, arbitration, stuffing
- CAN-FD: Dual bitrate, flexible data length
- LIN: Master/slave, sync break, checksum
- FlexRay: Deterministic communication, dual channel

**Debug Protocols** (Demo 03):

- JTAG: TAP state machine, IR/DR scan chains
- SWD: Debug Port (DP) and Access Port (AP)
- USB: Low/Full Speed, NRZI encoding, packets

**Parallel Bus** (Demo 04):

- IEEE-488 (GPIB): Instrument control bus
- Centronics: Parallel printer interface
- ISA: Address/data bus transactions

**Encoded Protocols** (Demo 05):

- Manchester: Self-clocking, differential encoding
- I2S: Audio bus, stereo sample extraction
- HDLC: Telecom framing, bit stuffing

**Auto-Detection** (Demo 06):

- Protocol type inference from signal characteristics
- Automatic baud rate recovery
- Multi-protocol testing on unknown signals
- Confidence scoring for detected protocols

---

## Running Demonstrations

### Option 1: Run Individual Demo

Run a single demo to focus on specific protocols:

```bash
# From the project root
python demonstrations/03_protocol_decoding/01_serial_comprehensive.py

# Or from the demo directory
cd demonstrations/03_protocol_decoding
python 01_serial_comprehensive.py
```

Expected output: Decoded packets with validation against expected data.

### Option 2: Run All Protocol Decoding Demos

Run all six demos in sequence:

```bash
# From the project root
for demo in demonstrations/03_protocol_decoding/*.py; do
    python "$demo"
done
```

### Option 3: Validate All Demonstrations

Validate all demonstrations in the project:

```bash
# From the project root
python demonstrations/validate_all.py
```

This runs all demonstrations and reports coverage, validation status, and failures.

---

## What You'll Learn

### Protocol Standards

**Serial Communication**:

- RS-232/RS-485 (UART) specifications
- SPI timing and mode configurations
- I2C bus arbitration and multi-master
- 1-Wire device enumeration

**Automotive Standards**:

- ISO 11898-1 (CAN specification)
- ISO 17458 (CAN-FD specification)
- ISO 17987 (LIN specification)
- ISO 17458-4 (FlexRay specification)

**Debug Standards**:

- IEEE 1149.1 (JTAG/Boundary Scan)
- ARM Debug Interface v5 (SWD)
- USB 2.0 Low/Full Speed specification

**Parallel Standards**:

- IEEE 488 (GPIB instrument bus)
- Centronics parallel printer standard
- ISA bus specification

### Decoding Techniques

**Timing Analysis**:

- Bit timing and baud rate detection
- Clock recovery from data
- Edge alignment and synchronization
- Setup/hold time validation

**Frame Extraction**:

- Start/stop bit detection
- Frame boundary identification
- Packet length determination
- Checksum/CRC validation

**State Machine Tracking**:

- JTAG TAP state transitions
- I2C bus state (start, data, stop, ACK)
- CAN arbitration and error states
- FlexRay schedule tracking

**Error Detection**:

- Framing errors (UART)
- CRC errors (CAN, FlexRay)
- Parity errors (UART, parallel)
- Bit stuffing violations (CAN, HDLC)

### Advanced Capabilities

**Auto-Detection**:

- Signal pattern recognition
- Baud rate inference from edge timing
- Protocol fingerprinting
- Confidence scoring

**Multi-Protocol**:

- Simultaneous multi-channel decoding
- Protocol tunneling (e.g., CAN over USB)
- Mixed analog/digital analysis
- Cross-protocol correlation

**Performance**:

- Efficient packet extraction
- Streaming decode for long captures
- Minimal memory footprint
- Real-time decode capabilities

---

## Common Issues and Solutions

### "Baud rate detection failed"

**Solution**: Insufficient edges or noisy signal:

1. Ensure capture has sufficient bit transitions
2. Check signal integrity (clean edges, adequate SNR)
3. Manually specify baud rate if auto-detection fails
4. Use longer capture window for better statistics

### "Framing errors in UART decode"

**Solution**: Timing mismatch or incorrect configuration:

1. Verify baud rate matches transmitter
2. Check data bits, parity, stop bits configuration
3. Validate sample rate is 10x baud rate minimum
4. Look for clock drift in long captures

### "CAN frames show CRC errors"

**Solution**: Bit stuffing or signal integrity issues:

1. Verify CAN high/low voltage levels
2. Check for proper bus termination
3. Validate bit timing and sample point
4. Look for noise during arbitration

### "I2C decode misses ACK/NACK"

**Solution**: Threshold or timing issues:

1. Adjust logic threshold for proper HIGH/LOW detection
2. Check for clock stretching by slave devices
3. Verify SDA setup/hold times meet I2C spec
4. Look for bus capacitance affecting edges

### "Auto-detection returns wrong protocol"

**Solution**: Ambiguous signal characteristics:

1. Manually specify protocol if known
2. Check confidence scores in detection results
3. Provide more context (baud rate, bus type)
4. Use longer capture for better fingerprinting

---

## Next Steps: Where to Go After Protocol Decoding

### If You Want to...

| Goal                                       | Next Demo                                         | Path                           |
| ------------------------------------------ | ------------------------------------------------- | ------------------------------ |
| Analyze automotive diagnostics             | `05_domain_specific/01_automotive_diagnostics.py` | Protocol → OBD-II/J1939/UDS    |
| Perform advanced signal integrity analysis | `04_advanced_analysis/03_signal_integrity.py`     | Protocol → SI analysis         |
| Build complete RE workflow                 | `16_complete_workflows/01_protocol_discovery.py`  | Protocol → Full workflow       |
| Detect unknown protocol patterns           | `04_advanced_analysis/05_pattern_discovery.py`    | Protocol → Pattern recognition |
| Analyze jitter and timing                  | `04_advanced_analysis/01_jitter_analysis.py`      | Protocol → Timing quality      |

### Recommended Learning Sequence

1. **Master Protocol Decoding** (this section)
   - Understand protocol structures
   - Extract and validate packets
   - Handle error conditions

2. **Explore Domain-Specific Applications** (05_domain_specific/)
   - Automotive diagnostics (OBD-II, J1939, UDS)
   - Side-channel analysis
   - Vintage logic family detection

3. **Advanced Analysis** (04_advanced_analysis/)
   - Signal integrity for protocol signals
   - Jitter analysis for timing validation
   - Pattern discovery for unknown protocols

4. **Complete Workflows** (16_complete_workflows/)
   - End-to-end reverse engineering
   - Protocol discovery from scratch
   - Multi-protocol analysis

---

## Tips for Learning

### Start with Known Protocols

Begin with simple serial protocols before tackling complex automotive buses:

```python
from oscura import decode_uart

# Start with UART - simplest protocol
packets = decode_uart(trace, baud_rate=9600, data_bits=8, parity='N', stop_bits=1)

for packet in packets:
    print(f"Data: 0x{packet.data:02X}")
```

### Validate Decoded Data

Always verify decoded data against known reference:

```python
# Generate known UART transmission
expected_bytes = [0x41, 0x42, 0x43]  # "ABC"
trace = generate_uart_signal(expected_bytes, baud_rate=9600)

# Decode and validate
packets = decode_uart(trace, baud_rate=9600)
decoded_bytes = [p.data for p in packets]

assert decoded_bytes == expected_bytes
```

### Understand Protocol Timing

Each protocol has specific timing requirements:

```python
# UART: Bit time = 1 / baud_rate
bit_time = 1.0 / 9600  # 104.17 μs

# SPI: Clock frequency determines bit rate
spi_bit_time = 1.0 / spi_clock_freq

# I2C: Standard mode = 100 kHz, Fast mode = 400 kHz
i2c_clock = 100_000  # Hz
```

### Visualize Protocol Timing

Plot decoded packets with timing information:

```python
import matplotlib.pyplot as plt

packets = decode_uart(trace, baud_rate=9600)

# Plot packet positions
for i, packet in enumerate(packets):
    plt.axvline(packet.timestamp, color='red', alpha=0.5)
    plt.text(packet.timestamp, 0, f"0x{packet.data:02X}")

plt.plot(trace.time(), trace.data)
plt.show()
```

### Combine with Analysis

Protocol decoding works best with signal analysis:

```python
# 1. Filter noise
filtered = low_pass(trace, cutoff=baud_rate * 5)

# 2. Detect edges
edges = find_rising_edges(filtered, threshold=0.5)

# 3. Decode protocol
packets = decode_uart(filtered, baud_rate=9600)

# 4. Validate timing
for packet in packets:
    validate_uart_timing(packet, baud_rate=9600)
```

---

## Understanding the Framework

### Decoder API

**Simple Decoding**:

```python
from oscura import decode_uart, decode_spi, decode_i2c, decode_can

# UART decoding
uart_packets = decode_uart(trace, baud_rate=9600, parity='N')

# SPI decoding
spi_packets = decode_spi(clk_trace, data_trace, mode=0)

# I2C decoding
i2c_packets = decode_i2c(sda_trace, scl_trace)

# CAN decoding
can_frames = decode_can(trace, bitrate=500000)
```

**Auto-Detection**:

```python
from oscura import detect_protocol

# Automatic protocol detection
result = detect_protocol(trace)
print(f"Detected: {result.protocol} (confidence: {result.confidence})")

if result.confidence > 0.8:
    packets = result.decode()
```

**Multi-Channel Decoding**:

```python
from oscura import load_all_channels, decode_spi

# Load multi-channel capture
channels = load_all_channels("logic_capture.sr")

# Decode SPI from multiple channels
packets = decode_spi(
    clk=channels["CLK"],
    mosi=channels["MOSI"],
    miso=channels["MISO"],
    cs=channels["CS"]
)
```

### Packet Structure

**All decoded packets share common structure**:

```python
class ProtocolPacket:
    timestamp: float      # Packet start time (seconds)
    data: bytes          # Decoded payload
    address: int | None  # Address (if applicable)
    status: str          # 'valid', 'error', 'warning'
    errors: list[str]    # Error descriptions
    metadata: dict       # Protocol-specific info
```

### Protocol-Specific Features

**UART**:

```python
uart_packet.parity_error   # Parity check failed
uart_packet.framing_error  # Stop bit missing
uart_packet.break_detected # Break condition
```

**CAN**:

```python
can_frame.identifier       # CAN ID (11-bit or 29-bit)
can_frame.extended        # Extended ID flag
can_frame.rtr             # Remote transmission request
can_frame.error_frame     # Error frame detected
can_frame.stuffing_error  # Bit stuffing violation
```

**I2C**:

```python
i2c_packet.address        # 7-bit or 10-bit address
i2c_packet.read_write     # 'read' or 'write'
i2c_packet.ack            # ACK/NACK status
i2c_packet.repeated_start # Repeated start condition
```

---

## Resources

### In This Repository

- **`src/oscura/analyzers/protocols/`** - Protocol decoder implementations
- **`tests/integration/protocols/`** - Protocol test cases
- **`examples/protocols/`** - Real-world protocol examples

### External Resources

- **[CAN Specification](https://www.can-cia.org/)** - ISO 11898-1 details
- **[I2C Specification](https://www.nxp.com/docs/en/user-guide/UM10204.pdf)** - Philips/NXP I2C standard
- **[SPI Overview](https://www.analog.com/en/analog-dialogue/articles/introduction-to-spi-interface.html)** - Analog Devices tutorial
- **[JTAG/Boundary Scan](https://www.xjtag.com/)** - IEEE 1149.1 resources
- **[USB Specification](https://www.usb.org/documents)** - USB 2.0 standard

### Protocol Analyzers

Compare Oscura results with commercial tools:

- **Saleae Logic** - Multi-protocol logic analyzer
- **Total Phase** - Beagle I2C/SPI analyzers
- **CANalyzer** - Vector automotive protocol analyzer

### Getting Help

1. Check protocol specifications for authoritative timing
2. Review demo docstrings for decoder usage examples
3. Examine source code in `src/oscura/analyzers/protocols/`
4. Test with known reference signals
5. Validate against commercial protocol analyzers

---

## Summary

The Protocol Decoding section covers:

| Demo                    | Focus             | Outcome                             |
| ----------------------- | ----------------- | ----------------------------------- |
| 01_serial_comprehensive | Serial protocols  | Decode UART, SPI, I2C, 1-Wire       |
| 02_automotive_protocols | Automotive buses  | Decode CAN, CAN-FD, LIN, FlexRay    |
| 03_debug_protocols      | Debug interfaces  | Decode JTAG, SWD, USB               |
| 04_parallel_bus         | Parallel buses    | Decode GPIB, Centronics, ISA        |
| 05_encoded_protocols    | Encoded protocols | Decode Manchester, I2S, HDLC        |
| 06_auto_detection       | Auto-detection    | Infer protocols, recover baud rates |

After completing these six 100-minute demonstrations, you'll understand:

- How to decode 20+ industry-standard protocols
- Protocol timing analysis and validation
- Error detection and handling
- Auto-detection and baud rate recovery
- Multi-channel synchronized decoding
- Protocol-specific features and edge cases

**Ready to start?** Run this to begin with serial protocols:

```bash
python demonstrations/03_protocol_decoding/01_serial_comprehensive.py
```

Happy decoding!
