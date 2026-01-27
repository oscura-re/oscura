# Technical Debt Documentation

**Last Updated**: 2026-01-26
**Version**: 0.6.0
**Priority**: HIGH - Follow-up work required for v0.7.0

This document tracks remaining technical debt that will be addressed in follow-up pull requests. The debt was identified during the v0.6.0 quality optimization sprint and represents systematic patterns that need refactoring.

---

## Table of Contents

- [1. Catch-All Exception Handlers (190+ Remaining)](#1-catch-all-exception-handlers)
- [2. Missing Test Data (230+ Tests)](#2-missing-test-data)
- [3. Implementation Roadmap](#3-implementation-roadmap)
- [4. Priority Matrix](#4-priority-matrix)

---

## 1. Catch-All Exception Handlers

### Problem Statement

**Count**: 190+ occurrences across 34 test files
**Pattern**: `except Exception as e: pytest.skip()`
**Priority**: HIGH
**Estimated Effort**: 12-16 hours

Catch-all exception handlers mask real bugs by treating all exceptions as "missing dependencies" or "unsupported features." Only `ImportError` should skip tests; implementation bugs (`ValueError`, `TypeError`, `AttributeError`) should fail tests to provide actionable feedback.

### Why It's Problematic

```python
# CURRENT PROBLEMATIC PATTERN
try:
    result = process_signal(data)
    assert result > 0
except Exception as e:
    pytest.skip(f"Processing not available: {e}")
```

**Issues:**

- Masks `ValueError` from incorrect arguments → Should FAIL
- Masks `TypeError` from API misuse → Should FAIL
- Masks `AttributeError` from refactoring bugs → Should FAIL
- Only `ImportError` (missing optional dependency) should skip

### Affected Files (34 Files)

**High Priority (Most Occurrences)**:

- `tests/unit/loaders/test_tektronix.py` (26 occurrences)
- `tests/unit/analyzers/digital/test_dsp.py` (26 occurrences)
- `tests/integration/test_multi_format_pipelines.py` (11 occurrences)
- `tests/integration/test_export_roundtrips.py` (11 occurrences)
- `tests/integration/test_complete_workflows.py` (11 occurrences)
- `tests/integration/test_integration_workflows.py` (11 occurrences)
- `tests/integration/test_protocol_analysis_workflows.py` (11 occurrences)
- `tests/unit/loaders/test_tektronix_loader.py` (11 occurrences)

**Medium Priority (5-10 Occurrences)**:

- `tests/integration/test_complex_scenarios.py` (10 occurrences)
- `tests/integration/test_database_workflows.py` (10 occurrences)
- `tests/integration/test_binary_to_protocol.py` (8 occurrences)
- `tests/integration/test_config_driven.py` (5 occurrences)
- `tests/validation/test_reverse_engineering.py` (5 occurrences)

**Lower Priority (1-4 Occurrences)**:

- 21 additional files with 1-4 occurrences each

### Fix Pattern

**BEFORE** (Masks All Errors):

```python
def test_signal_processing(self):
    try:
        from oscura.analyzers import spectral
        result = spectral.analyze(data)
        assert result["snr"] > 20
    except Exception as e:
        pytest.skip(f"Spectral analysis not available: {e}")
```

**AFTER** (Only Skip on ImportError):

```python
def test_signal_processing(self):
    pytest.importorskip("scipy")  # Skip if scipy not installed

    from oscura.analyzers import spectral

    # Let real bugs fail the test
    result = spectral.analyze(data)
    assert result["snr"] > 20
    # ValueError/TypeError/AttributeError will properly fail test
```

### Refactoring Strategy

**Phase 1: Import-Only Skips** (4 hours)

- Replace `except Exception` with `pytest.importorskip()` for optional dependencies
- Move imports outside try-except blocks
- Let implementation errors fail properly

**Phase 2: Explicit Exception Types** (4 hours)

- Identify legitimate runtime skips (missing test data, platform-specific)
- Use `except ImportError` for dependency checks
- Use `except FileNotFoundError` for missing test data
- Remove catch-all handlers

**Phase 3: Validation** (4 hours)

- Run full test suite with `-v` flag
- Verify real bugs surface as failures, not skips
- Update skip reasons to be actionable
- Document remaining legitimate skips in `tests/SKIP_PATTERNS.md`

### Example Refactoring

**File**: `tests/unit/analyzers/digital/test_dsp.py` (26 occurrences)

```python
# BEFORE (Line 455)
try:
    result = calculate_noise_margin(signal, thresholds)
    assert result["high_margin"] > 0.3
except Exception as e:
    pytest.skip(f"Noise margin test skipped: {e}")

# AFTER
# No try-except needed - let real bugs fail
result = calculate_noise_margin(signal, thresholds)
assert result["high_margin"] > 0.3
# If calculate_noise_margin has a bug, test will fail with traceback
```

### Validation Checklist

After refactoring each file:

- [ ] Optional dependencies use `pytest.importorskip()`
- [ ] No `except Exception` handlers remain
- [ ] Implementation bugs fail with clear traceback
- [ ] Only legitimate skips for missing data/platform issues
- [ ] Skip reasons are actionable and specific

---

## 2. Missing Test Data

### Problem Statement

**Count**: 230+ test skips due to missing data
**Priority**: MEDIUM
**Estimated Effort**: 6-8 hours (generation scripts) + 2-3 hours (acquisition)

Tests skip because required test data files don't exist in the repository. This includes WFM oscilloscope captures, PCAP network traces, synthetic signal files, and ground truth data for validation.

### Categories

#### 2.1 WFM Files (19 Tests)

**Location**: `tests/unit/loaders/test_tektronix*.py`, `tests/integration/test_wfm_loading.py`

**Missing Files**:

- `golden_analog.wfm` - Reference analog waveform
- `digital_waveform.wfm` - Digital logic capture
- `iq_waveform.wfm` - I/Q modulated signal
- Multi-channel WFM files (2-4 channels)
- Large WFM files (>10MB) for performance testing

**Skip Pattern**:

```python
if not wfm_files:
    pytest.skip("No WFM files available")

if not golden.exists():
    pytest.skip("golden_analog.wfm not found")
```

**Acquisition Strategy**:

**Option A: Generate Synthetic WFM** (4 hours)

```python
# Create script: scripts/test-data/generate_wfm_files.py
from oscura.export import export_wfm
import numpy as np

def generate_golden_analog():
    """Generate golden_analog.wfm reference file."""
    t = np.linspace(0, 1e-3, 10000)
    signal = np.sin(2 * np.pi * 1000 * t)  # 1kHz sine wave

    metadata = {
        "sample_rate": 10e6,
        "channel": "CH1",
        "vertical_scale": 1.0,
        "vertical_offset": 0.0,
    }

    export_wfm(signal, "test_data/formats/tektronix/golden_analog.wfm", metadata)

if __name__ == "__main__":
    generate_golden_analog()
    generate_digital_waveform()
    generate_iq_waveform()
```

**Option B: Acquire Real Captures** (3 hours + equipment access)

- Use Tektronix oscilloscope (MDO3000/4000 series)
- Capture reference signals (1kHz sine, square waves, UART traffic)
- Export as .wfm format
- Place in `test_data/formats/tektronix/analog/single_channel/`

**Recommended**: Option A (no hardware dependency)

#### 2.2 Pattern Detection Data (8 Tests)

**Location**: `tests/unit/analyzers/patterns/test_pattern_detection.py`

**Missing Files**:

- `periodic_pattern.bin` - Repeating byte sequence
- `repeating_sequence.bin` - Known motif patterns
- `anomaly_pattern.bin` - Signal with outliers

**Skip Pattern**:

```python
pytest.skip("Periodic pattern file not available")
pytest.skip("Repeating sequence file not available")
pytest.skip("Anomaly pattern file not available")
```

**Generation Script** (1 hour):

```python
# Create script: scripts/test-data/generate_pattern_files.py
import numpy as np
from pathlib import Path

def generate_periodic_pattern():
    """Generate periodic_pattern.bin with known period."""
    # 100 repetitions of 8-byte pattern
    pattern = np.array([0xAA, 0x55, 0xFF, 0x00, 0x12, 0x34, 0x56, 0x78], dtype=np.uint8)
    data = np.tile(pattern, 100)

    output = Path("test_data/patterns/periodic_pattern.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(output)

def generate_repeating_sequence():
    """Generate repeating_sequence.bin with multiple motifs."""
    motif1 = np.array([0x01, 0x02, 0x03], dtype=np.uint8)
    motif2 = np.array([0x0A, 0x0B], dtype=np.uint8)

    # Random arrangement of motifs
    data = []
    for _ in range(50):
        data.extend(motif1)
        data.extend(motif2)

    output = Path("test_data/patterns/repeating_sequence.bin")
    np.array(data, dtype=np.uint8).tofile(output)

def generate_anomaly_pattern():
    """Generate anomaly_pattern.bin with outliers."""
    # Normal distribution with few outliers
    normal = np.random.normal(128, 20, 990).astype(np.uint8)
    outliers = np.array([255, 0, 255, 0, 255, 0, 255, 0, 255, 0], dtype=np.uint8)
    data = np.concatenate([normal, outliers])

    output = Path("test_data/patterns/anomaly_pattern.bin")
    data.tofile(output)

if __name__ == "__main__":
    generate_periodic_pattern()
    generate_repeating_sequence()
    generate_anomaly_pattern()
```

#### 2.3 Entropy Test Data (5 Tests)

**Location**: `tests/unit/analyzers/statistical/test_entropy.py`

**Missing Files**:

- `low_entropy.bin` - Repetitive data (entropy < 2.0)
- `text_entropy.bin` - ASCII text (entropy ~4.5)
- `high_entropy.bin` - Random data (entropy > 7.8)
- `compressed_high_entropy.bin` - Compressed data

**Generation Script** (30 minutes):

```python
# Create script: scripts/test-data/generate_entropy_files.py
import numpy as np
from pathlib import Path

def generate_low_entropy():
    """Generate low_entropy.bin - repetitive data."""
    # Repeat 0xAA 1000 times (entropy ~0.0)
    data = np.full(1000, 0xAA, dtype=np.uint8)

    output = Path("test_data/entropy/low_entropy.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(output)

def generate_text_entropy():
    """Generate text_entropy.bin - English text."""
    text = "The quick brown fox jumps over the lazy dog. " * 100
    data = np.frombuffer(text.encode("ascii"), dtype=np.uint8)

    output = Path("test_data/entropy/text_entropy.bin")
    data.tofile(output)

def generate_high_entropy():
    """Generate high_entropy.bin - cryptographically random."""
    # Use secure random (entropy ~7.8-8.0)
    data = np.random.default_rng().bytes(1000)

    output = Path("test_data/entropy/high_entropy.bin")
    np.frombuffer(data, dtype=np.uint8).tofile(output)

if __name__ == "__main__":
    generate_low_entropy()
    generate_text_entropy()
    generate_high_entropy()
```

#### 2.4 Synthetic Signals (3 Tests)

**Location**: `tests/validation/test_synthetic_signals.py`

**Missing Files**:

- `1mhz_square.npz` - 1MHz square wave for clock recovery
- `uart_9600_8n1.npz` - UART signal with known baud rate
- Ground truth JSON files with expected measurements

**Generation Script** (2 hours - includes validation):

```python
# Create script: scripts/test-data/generate_synthetic_signals.py
import numpy as np
from pathlib import Path
import json

def generate_1mhz_square():
    """Generate 1MHz square wave with ground truth."""
    sample_rate = 100e6  # 100 MSPS
    duration = 1e-3  # 1ms
    freq = 1e6  # 1MHz

    t = np.arange(0, duration, 1/sample_rate)
    signal = np.sign(np.sin(2 * np.pi * freq * t))

    # Save signal
    output = Path("test_data/synthetic/1mhz_square.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, signal=signal, time=t, sample_rate=sample_rate)

    # Save ground truth
    ground_truth = {
        "frequency": float(freq),
        "period": 1.0 / freq,
        "duty_cycle": 0.5,
        "edge_count": len(np.where(np.diff(signal) != 0)[0]),
    }

    gt_file = output.parent / "1mhz_square_ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(ground_truth, f, indent=2)

def generate_uart_signal():
    """Generate UART 9600 baud 8N1 signal."""
    sample_rate = 1e6  # 1 MSPS
    baud_rate = 9600
    bit_duration = 1.0 / baud_rate

    # Encode "Hello" (0x48 0x65 0x6C 0x6C 0x6F)
    message = [0x48, 0x65, 0x6C, 0x6C, 0x6F]

    signal = []
    for byte_val in message:
        # Start bit (0)
        signal.extend([0] * int(bit_duration * sample_rate))

        # Data bits (LSB first)
        for i in range(8):
            bit = (byte_val >> i) & 1
            signal.extend([bit] * int(bit_duration * sample_rate))

        # Stop bit (1)
        signal.extend([1] * int(bit_duration * sample_rate))

    signal = np.array(signal, dtype=np.float64)
    t = np.arange(len(signal)) / sample_rate

    # Save signal
    output = Path("test_data/synthetic/uart_9600_8n1.npz")
    np.savez(output, signal=signal, time=t, sample_rate=sample_rate)

    # Save ground truth
    ground_truth = {
        "baud_rate": baud_rate,
        "format": "8N1",
        "message": "Hello",
        "bytes": [hex(b) for b in message],
    }

    gt_file = output.parent / "uart_9600_8n1_ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(ground_truth, f, indent=2)

if __name__ == "__main__":
    generate_1mhz_square()
    generate_uart_signal()
```

#### 2.5 PCAP Network Captures (8 Tests)

**Location**: `tests/unit/loaders/test_pcap_loader.py`, `tests/integration/test_pcap_to_inference.py`

**Missing Files**:

- `http_capture.pcap` - HTTP GET/POST requests
- `modbus_tcp.pcap` - Modbus TCP traffic
- `dns_queries.pcap` - DNS query/response pairs

**Acquisition Strategy**:

**Option A: Generate with Scapy** (2 hours)

```python
# Create script: scripts/test-data/generate_pcap_files.py
from scapy.all import IP, TCP, UDP, DNS, Raw, wrpcap
from scapy.layers.http import HTTP, HTTPRequest

def generate_http_capture():
    """Generate HTTP GET request PCAP."""
    packets = [
        # TCP handshake
        IP(dst="192.168.1.100")/TCP(dport=80, flags="S"),
        IP(src="192.168.1.100")/TCP(sport=80, flags="SA"),
        IP(dst="192.168.1.100")/TCP(dport=80, flags="A"),

        # HTTP GET
        IP(dst="192.168.1.100")/TCP(dport=80, flags="PA")/Raw(load=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"),
    ]

    output = Path("test_data/pcap/http_capture.pcap")
    output.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(output), packets)

def generate_modbus_capture():
    """Generate Modbus TCP PCAP."""
    # Modbus read holding registers (function code 03)
    modbus_request = bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function code (read holding registers)
        0x00, 0x00,  # Start address
        0x00, 0x0A,  # Quantity
    ])

    packets = [
        IP(dst="192.168.1.100")/TCP(dport=502, flags="PA")/Raw(load=modbus_request),
    ]

    output = Path("test_data/pcap/modbus_tcp.pcap")
    wrpcap(str(output), packets)

if __name__ == "__main__":
    generate_http_capture()
    generate_modbus_capture()
```

**Option B: Use tcpdump** (30 minutes + network access)

```bash
# Capture real HTTP traffic
tcpdump -i eth0 -w test_data/pcap/http_capture.pcap 'tcp port 80' -c 100

# Capture Modbus traffic (requires Modbus device)
tcpdump -i eth0 -w test_data/pcap/modbus_tcp.pcap 'tcp port 502' -c 50
```

**Recommended**: Option A (reproducible, no network dependency)

#### 2.6 Ground Truth Files (40+ Tests)

**Location**: `tests/validation/test_protocol_messages.py`, `tests/validation/test_synthetic_packets.py`

**Missing Files**:

- `ground_truth/*.json` - Expected field values for protocol messages
- `checksums/*.json` - CRC/checksum validation data

**Generation Strategy** (2 hours):

Create `scripts/test-data/generate_ground_truth.py`:

```python
import json
from pathlib import Path

def generate_protocol_ground_truth():
    """Generate ground truth for protocol message tests."""
    ground_truth = {
        "64b": {
            "header": {"type": 0x01, "length": 64, "checksum": 0xABCD},
            "payload": [0x12, 0x34, 0x56, 0x78],
            "footer": {"crc16": 0x1234},
        },
        "128b": {
            "header": {"type": 0x02, "length": 128, "checksum": 0xDEF0},
            "payload": list(range(16)),
            "footer": {"crc32": 0x12345678},
        },
    }

    output = Path("test_data/ground_truth/protocol_messages.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(ground_truth, f, indent=2)

if __name__ == "__main__":
    generate_protocol_ground_truth()
```

### Validation Checklist

After generating test data:

- [ ] All generation scripts run without errors
- [ ] Test data files exist in correct directories
- [ ] File sizes are appropriate (not too large for git)
- [ ] Ground truth files match generated data
- [ ] Tests pass when data files are present
- [ ] Skip messages removed from passing tests

---

## 3. Implementation Roadmap

### Sprint 1: Critical Fixes (Week 1)

**Focus**: High-impact files with most occurrences

**Tasks**:

1. Refactor `test_tektronix.py` (26 exception handlers)
2. Refactor `test_dsp.py` (26 exception handlers)
3. Generate WFM test files (19 tests unblocked)
4. **Deliverable**: 50+ cleaner tests, 19 tests unblocked

**GitHub Issue**: `#TBD-SPRINT1` - "Refactor Catch-All Handlers in Loader Tests"

### Sprint 2: Integration Tests (Week 2)

**Focus**: Integration test files with 10+ occurrences each

**Tasks**:

1. Refactor 7 integration test files (11 occurrences each)
2. Generate pattern detection test data
3. Generate entropy test files
4. **Deliverable**: 77+ cleaner tests, 13 tests unblocked

**GitHub Issue**: `#TBD-SPRINT2` - "Clean Integration Test Exception Handling"

### Sprint 3: Synthetic Signals (Week 3)

**Focus**: Validation tests requiring synthetic signals

**Tasks**:

1. Generate 1MHz square wave + ground truth
2. Generate UART signal + ground truth
3. Generate PCAP files for network tests
4. **Deliverable**: 11 validation tests unblocked

**GitHub Issue**: `#TBD-SPRINT3` - "Generate Synthetic Signal Test Data"

### Sprint 4: Remaining Files (Week 4)

**Focus**: Low-priority files with 1-4 occurrences

**Tasks**:

1. Refactor remaining 21 test files
2. Generate remaining ground truth files
3. Update `tests/SKIP_PATTERNS.md` documentation
4. **Deliverable**: 100% clean exception handling, all test data present

**GitHub Issue**: `#TBD-SPRINT4` - "Complete Test Data Generation"

### Milestones

**Milestone v0.7.0-alpha** (End of Week 2):

- 127+ tests with clean exception handling
- WFM, pattern, entropy test data generated
- 50% of technical debt resolved

**Milestone v0.7.0-beta** (End of Week 4):

- All 190+ catch-all handlers refactored
- All synthetic test data generated
- 100% technical debt resolved
- Documentation updated

**Milestone v0.7.0-release** (Week 5):

- Full test suite passes with all data
- Zero catch-all exception handlers
- Quality metrics: 90%+ coverage, 0 linting errors

---

## 4. Priority Matrix

### High Priority (Do First)

| Item | Impact | Effort | Files | Tests Affected |
|------|--------|--------|-------|----------------|
| Refactor `test_tektronix.py` | HIGH | 2h | 1 | 26 |
| Refactor `test_dsp.py` | HIGH | 2h | 1 | 26 |
| Generate WFM files | HIGH | 4h | N/A | 19 |
| Refactor integration tests | HIGH | 4h | 7 | 77 |

**Total**: 12 hours, 148 tests improved

### Medium Priority (Do Second)

| Item | Impact | Effort | Files | Tests Affected |
|------|--------|--------|-------|----------------|
| Generate pattern data | MEDIUM | 1h | N/A | 8 |
| Generate entropy data | MEDIUM | 0.5h | N/A | 5 |
| Generate synthetic signals | MEDIUM | 2h | N/A | 3 |
| Refactor medium-priority files | MEDIUM | 3h | 5 | 42 |

**Total**: 6.5 hours, 58 tests improved

### Low Priority (Do Last)

| Item | Impact | Effort | Files | Tests Affected |
|------|--------|--------|-------|----------------|
| Generate PCAP files | LOW | 2h | N/A | 8 |
| Generate ground truth | LOW | 2h | N/A | 40 |
| Refactor low-priority files | LOW | 3h | 21 | 24 |
| Update documentation | LOW | 1h | 1 | N/A |

**Total**: 8 hours, 72 tests improved

---

## Summary Statistics

**Total Technical Debt**:

- 190 catch-all exception handlers
- 230+ tests skipped due to missing data
- 34 test files requiring refactoring
- Estimated 26-32 hours total effort

**Expected Outcomes**:

- 100% clean exception handling (all 190 handlers refactored)
- 230+ tests enabled with proper data
- Test quality improved by 60% (actionable failures vs masked errors)
- Test coverage increased by 5-8% (enabled tests contribute to coverage)

**Success Criteria**:

- [ ] Zero `except Exception` handlers in test suite
- [ ] All test data generation scripts documented and working
- [ ] Test skip rate reduced from 15% to <5%
- [ ] All skips have actionable reasons documented
- [ ] CI/CD passes with full test suite enabled

---

## References

- **SKIP_PATTERNS.md**: Documents legitimate skip patterns
- **SKIP_INVENTORY.md**: Complete inventory of all skipped tests
- **SKIP_DOCUMENTATION.md**: Standards for documenting test skips
- **pyproject.toml**: Test configuration and dependencies
- **scripts/test-data/**: Test data generation scripts (to be created)

---

## Contributing

When addressing technical debt:

1. **Create GitHub Issue** for each sprint
2. **Branch Naming**: `fix/tech-debt-sprint-N`
3. **Commit Messages**: `fix(tests): refactor catch-all handlers in test_X.py`
4. **Update This Doc**: Mark items as complete with PR links
5. **Add Tests**: Ensure refactored tests still pass
6. **Review Checklist**: Use validation checklists above

---

## Questions?

Contact the maintainers or open a GitHub discussion for clarification on technical debt priorities.
