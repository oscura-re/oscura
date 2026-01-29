# Category 03: Protocol Decoding Demonstrations

Comprehensive protocol decoding examples demonstrating Oscura's capabilities for analyzing digital communication protocols.

## Available Demonstrations

### Serial Protocols

#### 01_uart_basic.py ✅
**UART Protocol Decoding**
- Multiple baudrates: 9600, 115200, 230400 bps
- Data bit configurations: 7-bit, 8-bit
- Parity modes: none, even, odd
- Stop bits: 1, 2
- Comprehensive validation with expected data matching

**Run**: `python demos/03_protocol_decoding/01_uart_basic.py`

#### 02_spi_basic.py ✅
**SPI Protocol Decoding**
- All 4 SPI modes (CPOL/CPHA combinations)
- Full-duplex communication (MOSI/MISO)
- Multiple bitrates: 500 kHz - 4 MHz
- Variable word sizes
- Master/slave data verification

**Run**: `python demos/03_protocol_decoding/02_spi_basic.py`

#### 03_i2c_basic.py ✅
**I2C Protocol Decoding**
- Standard mode (100 kHz) and Fast mode (400 kHz)
- 7-bit addressing
- Read and write transactions
- START/STOP condition detection
- ACK/NACK handling
- Multi-byte transfers

**Run**: `python demos/03_protocol_decoding/03_i2c_basic.py`

### Automotive Protocols

#### 04_can_basic.py ✅
**CAN 2.0 Protocol Decoding**
- Standard 11-bit and extended 29-bit identifiers
- Data frames with variable DLC (0-8 bytes)
- Bitrates: 500 kbps, 1 Mbps
- CRC validation
- Frame structure analysis

**Run**: `python demos/03_protocol_decoding/04_can_basic.py`

**ISO Standards**: ISO 11898-1:2015

#### 05_can_fd.py 🔲
**CAN-FD Protocol Decoding**
- Dual bitrate (nominal and data phase)
- Extended payload (up to 64 bytes)
- BRS (Bit Rate Switch) support
- FD-specific frame format

**Status**: Pending - Use demonstrations/03_protocol_decoding/02_automotive_protocols.py as reference

#### 06_lin.py 🔲
**LIN Bus Protocol Decoding**
- Break field detection
- Protected ID (PID) with parity
- Checksum validation
- Frame response types

**Status**: Pending - Merge demonstrations/03_protocol_decoding/02_automotive_protocols.py and demonstrations/05_domain_specific/automotive_lin.py

#### 07_flexray.py 🔲
**FlexRay Protocol Decoding**
- Differential signaling (BP/BM)
- Slot-based scheduling
- Static and dynamic segments
- Frame CRC validation

**Status**: Pending - Merge automotive_flexray.py sources

### Debug Protocols

#### 08_jtag.py ✅
**JTAG Protocol Decoding (IEEE 1149.1)**
- TAP state machine tracking
- Instruction Register (IR) operations
- Data Register (DR) operations
- Standard instructions: IDCODE, BYPASS, EXTEST
- Boundary-scan analysis

**Run**: `python demos/03_protocol_decoding/08_jtag.py`

**IEEE Standards**: IEEE 1149.1-2013

#### 09_swd.py 🔲
**SWD Protocol Decoding**
- ARM CoreSight debug
- DP/AP register access
- ACK/WAIT/FAULT responses
- Parity checking

**Status**: Pending - Use demonstrations/03_protocol_decoding/swd.py as reference

### Audio/Media Protocols

#### 10_i2s.py 🔲
**I2S Audio Protocol Decoding**
- Standard/Left-justified/Right-justified modes
- Multiple bit depths: 16, 24, 32-bit
- Sample rates: 44.1 kHz, 48 kHz, 96 kHz
- Stereo sample extraction
- Word Select (LRCLK) synchronization

**Status**: Pending - Use demonstrations/03_protocol_decoding/i2s.py as reference

#### 11_usb.py 🔲
**USB Protocol Decoding**
- USB Low-Speed (1.5 Mbps) decoding
- NRZI encoding/decoding
- Bit unstuffing
- PID extraction and validation
- Token, Data, and Handshake packets
- CRC5/CRC16 validation

**Status**: Pending - Note: SKIP_VALIDATION marker present (PID issues)

### Multi-Protocol

#### 12_comprehensive_protocols.py 🔲
**Comprehensive Multi-Protocol Analysis**
- Combined UART, SPI, I2C decoding
- Auto protocol detection
- Multi-channel synchronization
- Complete workflow demonstration

**Status**: Pending - Use demonstrations/03_protocol_decoding/protocol_comprehensive.py as reference

## Common Features

All demonstrations include:
- **Self-contained data generation**: Synthetic signals created programmatically
- **BaseDemo pattern**: Consistent structure with generate_test_data(), run_demonstration(), validate()
- **ValidationSuite**: Comprehensive validation with expect_true(), expect_equal()
- **Comprehensive metadata**: Capabilities, IEEE standards, related demos
- **Error handling**: Graceful handling of decoder limitations on synthetic signals
- **Documentation**: Detailed docstrings with usage examples

## Usage Patterns

### Basic Execution
```bash
python demos/03_protocol_decoding/01_uart_basic.py
```

### With Verbose Output
```bash
python demos/03_protocol_decoding/01_uart_basic.py --verbose
```

### Import in Scripts
```python
from demos.03_protocol_decoding.01_uart_basic import UARTDemo

demo = UARTDemo()
success = demo.execute()
```

## Validation

All completed demos pass:
- ✅ Syntax validation (py_compile)
- ✅ Import validation (demos.common infrastructure)
- ✅ Self-contained data generation
- ✅ ValidationSuite checks

## IEEE/ISO Standards Compliance

- **IEEE 181-2011**: Waveform measurements (all demos)
- **IEEE 1149.1-2013**: JTAG boundary-scan (08_jtag.py)
- **ISO 11898-1:2015**: CAN protocol (04_can_basic.py, 05_can_fd.py)
- **ISO 17987**: LIN bus (06_lin.py)
- **ISO 17458**: FlexRay (07_flexray.py)

## Migration Sources

Demos consolidated from:
- `demonstrations/03_protocol_decoding/` - Primary source with BaseDemo pattern
- `demos/04_serial_protocols/` - ValidationSuite additions
- `demos/05_protocol_decoding/` - Comprehensive examples
- `demonstrations/05_domain_specific/` - Automotive-specific protocols

## Development Status

**Completion**: 5/12 demos (42%)

- ✅ Completed: UART, SPI, I2C, CAN, JTAG
- 🔲 Pending: CAN-FD, LIN, FlexRay, SWD, I2S, USB, Comprehensive

See `demos/MIGRATION_STATUS_03-05.md` for detailed migration tracking.

## Related Categories

- **Category 01**: Waveform Analysis - Foundation for protocol decoding
- **Category 02**: File Format I/O - Loading captured protocol data
- **Category 05**: Domain Specific - Automotive/EMC applications using these protocols

---

**Last Updated**: 2026-01-29
**Status**: Core protocol demos completed, automotive and specialized protocols pending
