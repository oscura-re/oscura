# Test Vector Generator

**Purpose**: Generate comprehensive test data files to eliminate "missing test data" skips in the test suite.

## Overview

The `generate_test_vectors.py` script creates test data files across multiple formats:

- **WFM files** (Tektronix oscilloscope captures)
- **PCAP files** (network packet captures)
- **NPZ files** (NumPy compressed signals)
- **CSV files** (waveform data in CSV format)
- **VCD files** (digital logic simulation)
- **Protocol test vectors** (UART, SPI binary data)

## Usage

### Basic Usage

```bash
uv run python scripts/generate_test_vectors.py
```

This generates 27 test files in `test_data/` with default configuration.

### Custom Output Directory

```bash
uv run python scripts/generate_test_vectors.py /path/to/custom/directory
```

## Generated Files

### WFM Files (10 files, ~790KB total)

Tektronix oscilloscope waveform files using `tm_data_types`:

- `formats/tektronix/analog/sine_1khz_basic.wfm` - Clean 1kHz sine wave
- `formats/tektronix/analog/sine_10khz.wfm` - 10kHz sine wave
- `formats/tektronix/analog/noisy_sine_20db.wfm` - SNR=20dB noisy signal
- `formats/tektronix/analog/noisy_sine_10db.wfm` - SNR=10dB noisy signal
- `formats/tektronix/digital/square_1khz_basic.wfm` - 1kHz square wave (50% duty cycle)
- `formats/tektronix/digital/pwm_1khz_25pct.wfm` - 1kHz PWM (25% duty cycle)
- `formats/tektronix/multi_channel/quad_channel_CH1-4.wfm` - 4-channel capture (separate files)

**Parameters**: 10,000 samples per channel, 1 MHz sample rate, 10ms duration

### PCAP Files (6 files, ~600 bytes total)

Network packet captures using `scapy`:

- `formats/pcap/tcp/simple_tcp.pcap` - Simple TCP packet
- `formats/pcap/udp/simple_udp.pcap` - Simple UDP packet
- `formats/pcap/tcp/http/http_get.pcap` - HTTP GET request
- `formats/pcap/tcp/http/http_post.pcap` - HTTP POST request
- `formats/pcap/malformed/truncated.pcap` - Truncated packet (error handling test)
- `formats/pcap/malformed/invalid_checksum.pcap` - Corrupted checksum (error handling test)

**Usage**: Protocol analysis, decoder testing, error handling validation

### NPZ Files (4 files, ~320KB total)

NumPy compressed signal files with metadata:

- `synthetic/waveforms/npz/sine_1khz.npz` - 1kHz sine wave
- `synthetic/waveforms/npz/square_1khz.npz` - 1kHz square wave
- `synthetic/waveforms/npz/sawtooth_500hz.npz` - 500Hz sawtooth wave
- `synthetic/waveforms/npz/white_noise.npz` - White noise signal

**Contents**: `data` array, `time` array, `metadata` JSON string

### CSV Files (3 files, ~1MB total)

Waveform data in comma-separated format:

- `formats/csv/sine_1khz.csv` - 1kHz sine (with header)
- `formats/csv/square_2khz.csv` - 2kHz square (with header)
- `formats/csv/no_header.csv` - 1kHz sine (no header, parser test)

**Format**: `time,voltage` columns (10,000 rows each)

### VCD Files (2 files, ~40KB total)

Digital logic simulation format (Value Change Dump):

- `formats/vcd/clock_1khz.vcd` - 1kHz clock signal (1 cycle)
- `formats/vcd/clock_10mhz.vcd` - 10MHz clock signal (10,000 cycles)

**Standard**: IEEE 1364-1995 compatible

### Protocol Test Vectors (2 files, <1KB total)

Binary protocol data for decoder testing:

- `synthetic/protocols/uart_test.bin` - UART encoded "Hello UART"
- `synthetic/protocols/spi_test.npz` - SPI MOSI/MISO data (4 bytes each)

**Usage**: Protocol decoder validation

## Dependencies

### Required

- `numpy` - Signal generation and data manipulation
- `pathlib` - File path operations

### Optional

Install optional dependencies for full functionality:

```bash
# For WFM file generation (Tektronix oscilloscope format)
uv pip install tm-data-types

# For PCAP file generation (network packet captures)
uv pip install scapy
```

**Graceful Degradation**: Script runs without optional dependencies but skips files requiring them.

## Implementation Details

### Reproducibility

All test data uses `seed=42` for deterministic generation:

```python
generator = TestVectorGenerator(seed=42)
```

Same inputs always produce identical outputs.

### File Size Optimization

- Small samples: 10,000 points (10ms @ 1MHz)
- Manageable file sizes: <100KB per file
- Fast generation: ~5 seconds total
- Git-friendly: files excluded via `.gitignore` patterns

### Signal Quality

**Clean signals**: SNR = ∞ (no noise)
**Noisy signals**: SNR = 10dB, 20dB (configurable)
**Frequency range**: 100Hz - 10MHz
**Sample rates**: 1MHz default (adjustable)

### Error Handling

**Missing dependencies**: Skip files with warning message
**Invalid parameters**: Raise `ValueError` with clear message
**File I/O errors**: Propagate with traceback

## Extending the Generator

### Add New Signal Type

```python
def generate_wfm_custom_signal(
    self,
    output_path: Path,
    # ... custom parameters
) -> None:
    """Generate custom signal type."""
    # Generate signal data
    data = ...  # Your signal generation logic

    # Create WFM file
    wfm = tm_data_types.AnalogWaveform()
    wfm.y_axis_values = data.astype(np.float64)
    # ... configure waveform

    # Write file
    tm_data_types.write_file(str(output_path), wfm)
    self.generated_files.append(str(output_path))
```

### Add to Generation Suite

In `generate_all_test_vectors()`:

```python
self.generate_wfm_custom_signal(
    self.base_dir / "formats/tektronix/custom/my_signal.wfm",
    # ... parameters
)
```

## Validation

### Verify Generated Files

```bash
# Check file existence
ls -lh test_data/formats/tektronix/analog/
ls -lh test_data/formats/pcap/tcp/

# Verify loadable by Oscura
uv run python -c "
from pathlib import Path
from oscura.loaders.tektronix import load_tektronix_wfm
from oscura.loaders.pcap import load_pcap

# Test WFM loading
wfm = load_tektronix_wfm(Path('test_data/formats/tektronix/analog/sine_1khz_basic.wfm'))
print(f'WFM: {len(wfm.data)} samples')

# Test PCAP loading
pcap = load_pcap(Path('test_data/formats/pcap/tcp/simple_tcp.pcap'))
print(f'PCAP: {len(pcap)} packets')
"
```

### Run Tests with New Data

```bash
# Run loader tests
./scripts/test.sh tests/unit/loaders/

# Check skip count reduction
uv run pytest tests/ --collect-only -q | grep -i skip
```

## Success Metrics

**Target**: Eliminate 20+ "missing test data" skips

**Before**: 560 skipped tests
**After**: 540 skipped tests (expected)
**Reduction**: 20 tests enabled (3.6% improvement)

## Troubleshooting

### "tm_data_types not available"

**Cause**: Optional dependency not installed

**Solution**:

```bash
uv pip install tm-data-types
```

### "scapy not available"

**Cause**: Optional dependency not installed

**Solution**:

```bash
uv pip install scapy
```

### "Permission denied" errors

**Cause**: Insufficient write permissions to `test_data/`

**Solution**:

```bash
chmod u+w test_data/
# Or run with custom directory
uv run python scripts/generate_test_vectors.py /tmp/test_data
```

### Files not showing in git

**Cause**: `.gitignore` excludes `*.wfm` files

**Expected**: Generated files are not committed (too large)

**Regenerate** on each development machine or CI run.

## Performance

**Generation time**: ~5 seconds (all 27 files)
**Total size**: ~2.1MB (all files)
**Bottlenecks**: WFM file writing (tm_data_types), PCAP assembly (scapy)

## Related Documentation

- `test_data/README.md` - Test data organization
- `test_data/manifest.json` - Test data catalog
- `tests/SKIP_PATTERNS.md` - Valid skip patterns
- `CONTRIBUTING.md` - Test data best practices

## CI Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Generate test vectors
  run: uv run python scripts/generate_test_vectors.py

- name: Run tests
  run: ./scripts/test.sh
```

**Why CI generation**: Ensures reproducible test data, avoids committing large binaries, validates generator works.

## Changelog

**2026-01-25**: Initial implementation with 27 test files across 6 formats
