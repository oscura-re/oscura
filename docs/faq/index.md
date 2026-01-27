# Frequently Asked Questions

Common questions about Oscura and hardware reverse engineering workflows.

---

## General

### What is Oscura?

Oscura is a comprehensive hardware reverse engineering framework that automates workflows from oscilloscope captures to protocol dissectors. It integrates specialized tools (sigrok, scipy, ChipWhisperer) into unified Python workflows, eliminating manual file conversions and tool-hopping.

### What makes Oscura different from other tools?

**Workflow automation:** Complete end-to-end workflows in Python, not fragmented toolchains

**Automatic inference:** CRC recovery, signal extraction, protocol structure detection without manual analysis

**Multiple domains:** Covers automotive, IoT, industrial, embedded - not limited to one protocol family

**Hypothesis tracking:** Scientific approach with confidence scoring and audit trails

### Is Oscura free?

Yes, Oscura is open source (MIT license). Free for commercial and non-commercial use.

---

## Installation

### What Python version is required?

Python 3.12 or higher. We use modern Python features (type hints, structural pattern matching, etc.).

### Installation fails with "No module named 'oscura'"

**Solution:**

```bash
# Ensure you're in the right environment
pip install oscura

# Or for development
git clone https://github.com/oscura-re/oscura.git
cd oscura
./scripts/setup.sh
```

### Can I use Oscura on Windows?

Yes, but Linux/macOS recommended for best compatibility. Windows Subsystem for Linux (WSL2) works well.

### What dependencies are required?

Core dependencies:

- numpy, scipy (signal processing)
- matplotlib (visualization)
- pandas (data analysis)

Optional:

- cantools (CAN DBC files)
- pytest (testing)
- mkdocs (documentation)

See `pyproject.toml` for complete list.

---

## Capabilities

### What file formats are supported?

**Oscilloscope formats:**

- Tektronix: .wfm, .isf
- Rigol: .wfm
- LeCroy: .trc, .wvs
- Siglent: .bin

**Generic formats:**

- WAV (audio)
- CSV (comma-separated)
- VCD (Verilog)
- HDF5

**CAN formats:**

- BLF (Vector)
- ASC (Vector)
- LOG (SocketCAN)
- PCAP/PCAPNG

**Side-channel:**

- ChipWhisperer (.npy)

### What protocols can Oscura decode?

16+ built-in decoders:

**Serial:** UART, SPI, I2C, 1-Wire

**Automotive:** CAN, CAN-FD, LIN, FlexRay, UDS

**Debug:** JTAG, SWD

**Industrial:** Modbus RTU, PROFIBUS

**Encoding:** Manchester, HDLC

**Others:** USB (partial), I2S

See [Protocol Catalog](../protocols/) for details.

### Can Oscura reverse engineer unknown protocols?

Yes! This is a core feature. Use `BlackBoxSession` for differential analysis:

```python
session = BlackBoxSession()
session.add_recording("idle", "idle.bin")
session.add_recording("active", "active.bin")

# Automatic protocol inference
spec = session.generate_protocol_spec()
session.export_results("dissector", "protocol.lua")
```

Features:

- Automatic field boundary detection
- CRC/checksum recovery
- Counter and sequence number identification
- Entropy analysis (detect encryption/compression)
- State machine extraction

### Can Oscura generate Wireshark dissectors?

Yes, automatically:

```python
session.export_results("dissector", "protocol.lua")
```

Also generates:

- Scapy layers (Python)
- Kaitai Struct (multi-language)
- C/C++ headers
- DBC files (CAN)

### Does Oscura support side-channel analysis?

Yes, power analysis features:

```python
from oscura.workflows import power_analysis

result = power_analysis(
    traces="power_traces.npy",
    plaintexts="plaintexts.npy",
    algorithm="AES-128"
)
```

Supports:

- CPA (Correlation Power Analysis)
- DPA (Differential Power Analysis)
- Template attacks
- Trace filtering/alignment

---

## Usage

### How do I load a waveform file?

```python
from oscura.loaders import load_waveform

# Auto-detect format
waveform = load("capture.wfm")

# Or specify format
waveform = load("capture.csv", format="csv")
```

### How do I decode a UART signal?

```python
from oscura.analyzers.protocols import UARTDecoder

# Auto-detect parameters
params = UARTDecoder.auto_detect(waveform)

# Create decoder
decoder = UARTDecoder(**params)

# Decode messages
messages = decoder.decode(waveform)
```

### How do I analyze CAN bus traffic?

```python
from oscura.sessions import CANSession

session = CANSession(bitrate=500000)
session.load("vehicle.blf")

# Extract signals automatically
signals = session.extract_signals()

# Generate DBC file
session.export_dbc("vehicle.dbc")
```

### My capture file is too large (out of memory)

Use streaming mode:

```python
from oscura.loaders import load_waveform_streaming

stream = load_waveform_streaming("huge.wfm", chunk_size=1_000_000)
for chunk in stream:
    process(chunk)
```

Or memory-mapped I/O:

```python
from oscura.io import MemoryMappedWaveform

waveform = MemoryMappedWaveform("huge.bin")
# Data accessed on-demand
```

### How accurate is CRC recovery?

Very high accuracy for standard CRCs:

- CRC-8: ~99% success rate
- CRC-16: ~98% success rate
- CRC-32: ~97% success rate

Requires at least 10-15 message-checksum pairs for reliable detection.

For custom polynomials, provide samples:

```python
crc_result = session.recover_checksum(
    algorithms=["crc8", "crc16", "crc32"],
    custom_polynomials=[0x07, 0x31, 0x9B],
    min_samples=20
)
```

---

## Performance

### How fast is Oscura?

Typical performance on modern hardware (i7-9700K):

| Operation | Throughput | Notes |
|-----------|------------|-------|
| UART decode | 100 MB/s | CPU-bound |
| CAN decode | 200 MB/s | Optimized |
| FFT analysis | 50 MS/s | NumPy FFT |
| CRC brute force | 1M trials/s | Parallelizable |

### Can I use multiple CPU cores?

Yes, batch processing uses parallel execution:

```python
# NOTE: Use workflows or manual iteration in v0.6
# from oscura.workflows import batch_analyze

results = batch_analyze(
    files=file_list,
    analysis_func=my_analysis,
    parallel=True,
    num_workers=8  # 8 cores
)
```

### Is GPU acceleration supported?

Experimental support for CUDA:

```python
from oscura.performance import enable_gpu

enable_gpu()  # Use GPU for FFT, correlation, CPA
```

Requires: NVIDIA GPU with CUDA toolkit

---

## Troubleshooting

### Protocol auto-detection fails

**Problem:** `UARTDecoder.auto_detect()` returns None

**Solutions:**

1. **Increase search range:**

```python
params = UARTDecoder.auto_detect(
    waveform,
    baud_rate_range=(1200, 921600),  # Wider range
    tolerance=0.05                    # Allow 5% mismatch
)
```

1. **Lower confidence threshold:**

```python
params = UARTDecoder.auto_detect(
    waveform,
    confidence_threshold=0.7  # Accept 70% confidence
)
```

1. **Specify known parameters:**

```python
decoder = UARTDecoder(
    baud_rate=115200,  # Known value
    data_bits=8,
    parity='N',
    stop_bits=1
)
```

### CRC recovery returns no results

**Problem:** `session.recover_checksum()` doesn't find CRC

**Solutions:**

1. **Provide more samples:**
   - Need 10-15 minimum, 20-30 recommended
   - Ensure diversity (different messages)

2. **Expand search space:**

```python
crc_result = session.recover_checksum(
    algorithms=["crc8", "crc16", "crc32", "sum", "xor"],
    custom_polynomials=[0x07, 0x31, 0x9B, 0x1D],
    brute_force=True  # Try all combinations (slow)
)
```

1. **Check if actually has checksum:**
   - Not all protocols use CRC
   - May use simple XOR or sum

### Differential analysis shows too many changes

**Problem:** `session.compare()` shows everything changing

**Solutions:**

1. **Increase confidence threshold:**

```python
diff = session.compare(
    "idle",
    "active",
    confidence_threshold=0.9,  # Require 90% confidence
    min_occurrences=3          # Must change consistently
)
```

1. **Align messages:**
   - Ensure captures are time-aligned
   - Remove transient startup/shutdown

2. **Filter noise:**
   - Use longer captures (more samples)
   - Apply low-pass filter to waveform

### Generated Wireshark dissector doesn't work

**Problem:** Dissector loads but doesn't decode

**Solutions:**

1. **Validate dissector:**

```python
session.export_results(
    "dissector",
    "protocol.lua",
    validate=True  # Check Lua syntax
)
```

1. **Check installation:**

```bash
# Copy to correct directory
cp protocol.lua ~/.local/lib/wireshark/plugins/

# Restart Wireshark
```

1. **Test with sample PCAP:**

```python
session.export_test_pcap("test.pcap")
# Open test.pcap in Wireshark
```

### Import error: "No module named 'tm_data_types'"

**Problem:** Tektronix WFM loader missing dependency

**Solution:**

```bash
pip install tm-data-types
```

Or install all extras:

```bash
pip install oscura[all]
```

---

## Best Practices

### How much data should I capture?

**Minimum:**

- 10-20 complete message transactions
- Include headers, payloads, checksums

**Recommended:**

- Multiple operational states (idle, active, error)
- 30-60 seconds per state
- Full operational cycle

**For differential analysis:**

- Multiple captures of same state (verify consistency)
- Systematic variation (change one variable at a time)

### What sample rate should I use?

**Rule of thumb:** 10x the signal frequency

Examples:

- 115200 baud UART → 1.152 MHz minimum, 10 MHz recommended
- 500 kbps CAN → 5 MHz minimum, 50 MHz recommended
- 10 Mbps FlexRay → 100 MHz minimum, 1 GHz recommended

### How do I validate my findings?

1. **Cross-validation:**
   - Test on multiple captures
   - Verify consistency across sessions

2. **Replay testing:**

```python
from oscura.validation import replay_messages

results = replay_messages(
    transport=serial_port,
    messages=crafted_messages,
    expect_response=True
)
```

1. **Hypothesis testing:**
   - Form hypothesis about field meaning
   - Test against all captures
   - Document confidence score

2. **Peer review:**
   - Export comprehensive report
   - Share with colleagues
   - Reproduce findings independently

### How should I organize my RE project?

Recommended structure:

```
project/
├── captures/
│   ├── raw/               # Original captures
│   ├── idle/              # Idle state captures
│   ├── active/            # Active state captures
│   └── edge_cases/        # Error conditions, etc.
├── scripts/
│   ├── 01_initial_analysis.py
│   ├── 02_differential.py
│   ├── 03_crc_recovery.py
│   └── 04_export_artifacts.py
├── output/
│   ├── dissectors/
│   ├── reports/
│   └── dbc_files/
├── oscura.yaml            # Configuration
└── README.md              # Project documentation
```

---

## Legal and Ethical

### Is reverse engineering legal?

**Depends on jurisdiction and context:**

**Generally legal:**

- Interoperability (EU Directive 2009/24/EC)
- Security research (responsible disclosure)
- Right to repair
- Academic research

**May be illegal:**

- Bypassing DRM/copy protection (DMCA 1201)
- Violating terms of service
- Unauthorized access to systems
- Patent infringement

**Always:**

- Check local laws
- Consult legal counsel
- Obtain permission when possible
- Follow responsible disclosure

### Can I use Oscura for security research?

Yes, but responsibly:

1. **Legal authorization:** Only analyze systems you own or have permission to test
2. **Responsible disclosure:** Report vulnerabilities to manufacturers before public disclosure
3. **No harm:** Never deploy attacks against production systems
4. **Document everything:** Maintain audit trail for legal protection

### Can I publish my findings?

**Best practices:**

1. **Notify vendor first:** 90-day disclosure timeline
2. **Remove sensitive data:** Redact proprietary details
3. **Focus on concepts:** Teach principles, not exploits
4. **Follow ethics:** CVD (Coordinated Vulnerability Disclosure)

---

## Getting Help

### Where can I get help?

1. **Documentation:** [oscura-re.github.io/oscura](https://oscura-re.github.io/oscura)
2. **GitHub Issues:** [github.com/oscura-re/oscura/issues](https://github.com/oscura-re/oscura/issues)
3. **Examples:** `examples/` directory in repository
4. **Tutorials:** Step-by-step guides in docs

### How do I report a bug?

1. Check [existing issues](https://github.com/oscura-re/oscura/issues)
2. Create new issue with:
   - Oscura version (`python -m oscura --version`)
   - Python version
   - Operating system
   - Minimal reproducer
   - Expected vs actual behavior

### How can I contribute?

See [Contributing Guide](../contributing.md)

Ways to contribute:

- Bug reports
- Protocol decoders
- Documentation
- Example scripts
- Test cases
- Performance improvements

---

## See Also

- [Getting Started Guide](../user-guide/getting-started.md)
- [API Reference](../api/)
- [Tutorials](../tutorials/)
- [Protocol Catalog](../protocols/)
- [Contributing Guide](../contributing.md)
