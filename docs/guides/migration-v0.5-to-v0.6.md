# Migration Guide: v0.5.1 → v0.6.0

This guide helps you upgrade from Oscura v0.5.1 to v0.6.0.

## Summary

v0.6.0 is a **major feature release** adding 48 new capabilities across hardware reverse engineering, protocol analysis, performance optimization, and infrastructure improvements. Most changes are **backward compatible**, with a few minor breaking changes detailed below.

---

## Quick Upgrade

```bash
# Update via pip
pip install --upgrade oscura

# Or with uv
uv pip install --upgrade oscura
```

**Verification:**

```python
import oscura
print(oscura.__version__)  # Should show 0.6.0
```

---

## Breaking Changes

### 1. Plugin System Removed

**What changed:** The incomplete plugin system (`oscura plugins` CLI command) has been removed from v0.6.0.

**Impact:** If you were using `/oscura plugins` commands, they will no longer work.

**Migration:**

- The plugin system was non-functional in v0.5.1 (directory didn't exist)
- Full plugin support is planned for v0.8.0
- For now, use extension points directly via the API

**Before (v0.5.1 - didn't work):**

```bash
oscura plugins list  # Command existed but failed
```

**After (v0.6.0):**

```python
# Use extensibility API directly
from oscura.extensibility.extensions import get_registry

registry = get_registry()
# Register custom decoders/analyzers directly
```

### 2. DBC Generator Consolidation

**What changed:** Duplicate DBC generator implementations consolidated into single module.

**Impact:** Import paths updated for cleaner API.

**Migration:**

**Before (v0.5.1):**

```python
from oscura.automotive.dbc.generator import DBCGenerator  # Old location
```

**After (v0.6.0):**

```python
from oscura.automotive.can.dbc_generator import DBCGenerator  # New canonical location
# OR use backward-compatible wrapper
from oscura.automotive.dbc import generate, generate_from_session  # Still works
```

### 3. Payload Analysis Import Update

**What changed:** `compute_similarity()` function moved to canonical location.

**Impact:** Tests or code importing this function need updated imports.

**Migration:**

**Before (v0.5.1):**

```python
from oscura.analyzers.packet.payload import compute_similarity
```

**After (v0.6.0):**

```python
from oscura.analyzers.packet.payload_analysis import compute_similarity
```

---

## New Features

### Phase 3: Enhanced Analysis (10 Features)

#### 1. ML Signal Classification

```python
from oscura.analyzers.ml.signal_classifier import MLSignalClassifier

classifier = MLSignalClassifier()
classifier.train(training_data)  # Optional
result = classifier.classify(unknown_signal, sample_rate=10000)
print(f"Signal type: {result.predicted_class} (confidence: {result.confidence})")
```

#### 2. Firmware Pattern Recognition

```python
from oscura.firmware.pattern_recognition import FirmwarePatternRecognizer

recognizer = FirmwarePatternRecognizer()
with open("firmware.bin", "rb") as f:
    firmware = f.read()

analysis = recognizer.analyze(firmware)
print(f"Architecture: {analysis.detected_architecture}")
print(f"Functions found: {len(analysis.functions)}")
print(f"String tables: {len(analysis.string_tables)}")
```

#### 3. Hardware Abstraction Layer Detection

```python
from oscura.hardware.hal_detector import HALDetector

detector = HALDetector()
result = detector.detect(firmware_binary)
print(f"HAL framework: {result.detected_hal}")
print(f"Peripherals: {result.peripherals}")
```

#### 4. Side-Channel Attack Detection

```python
from oscura.security.side_channel_detector import SideChannelDetector

detector = SideChannelDetector()
report = detector.analyze(power_traces, plaintexts)
for vuln in report.vulnerabilities:
    print(f"{vuln.type}: {vuln.severity} - {vuln.description}")
```

### Phase 4: Validation & Testing (5 Features)

#### 5. Protocol Grammar Validator

```python
from oscura.validation.grammar_validator import ProtocolGrammarValidator

validator = ProtocolGrammarValidator()
report = validator.validate(protocol_spec)
if report.has_errors():
    for error in report.errors:
        print(f"{error.severity}: {error.message}")
```

#### 6. Compliance Test Generator

```python
from oscura.validation.compliance_tests import ComplianceTestGenerator

generator = ComplianceTestGenerator(protocol_spec)
test_cases = generator.generate_all()
for test in test_cases:
    result = test.execute(implementation)
    print(f"{test.name}: {result.status}")
```

#### 7. Fuzzer Integration

```python
from oscura.validation.fuzzer import ProtocolFuzzer

fuzzer = ProtocolFuzzer(protocol_spec)
for test_case in fuzzer.generate_test_cases(count=1000):
    response = send_to_device(test_case.mutated_message)
    if fuzzer.is_crash(response):
        fuzzer.save_crash(test_case)
```

#### 8. Hardware-in-Loop Testing

```python
from oscura.validation.hil_testing import HILTester, HILConfig

config = HILConfig(interface="serial", port="/dev/ttyUSB0", baud_rate=115200)
tester = HILTester(config)

with tester:
    report = tester.run_tests([
        {"name": "Test 1", "send": b"\x01\x02", "expected": b"\x03\x04"},
        {"name": "Test 2", "send": b"\x05\x06", "expected": b"\x07\x08"},
    ])

print(f"Passed: {report.passed_count}/{report.total_count}")
```

#### 9. Regression Test Suite

```python
from oscura.validation.regression_suite import RegressionTestSuite

suite = RegressionTestSuite("uart_decoder")
suite.register_test("decode_basic", my_decode_function, args=(test_signal,))
suite.capture_baseline("decode_basic")  # Save expected output

# Later, after code changes
report = suite.run_all()
if report.regressions_found:
    print(f"Regressions: {len(report.regressions_found)}")
```

### Phase 5: Performance & Integration (10 Features)

#### 10. Performance Profiling

```python
from oscura.performance.profiling import PerformanceProfiler

with PerformanceProfiler() as profiler:
    # Your code here
    process_large_dataset()

profiler.export_html("profile_report.html")
print(f"Hotspots: {profiler.result.hotspots[:5]}")
```

#### 11. Memory Optimization

```python
from oscura.performance.memory_optimizer import MemoryOptimizer

optimizer = MemoryOptimizer(memory_limit_mb=500)

# Load huge file without OOM
data = optimizer.load_optimized("huge_capture.bin")

# Process in chunks
for chunk in optimizer.create_stream_processor(data, chunk_size=1000000):
    process_chunk(chunk)
```

#### 12. Parallel Processing

```python
from oscura.performance.parallel import ParallelProcessor

processor = ParallelProcessor(num_workers=8)
results = processor.map(analyze_trace, list_of_traces)
print(f"Speedup: {processor.result.speedup}x")
```

#### 13. Caching Layer

```python
from oscura.performance.caching import CacheManager, CachePolicy

cache = CacheManager(policy=CachePolicy(ttl=3600, max_size_mb=100))

@cache.cached(ttl=600)
def expensive_fft(signal):
    return np.fft.fft(signal)  # Cached for 10 minutes
```

#### 14. Database Backend

```python
from oscura.storage import DatabaseBackend, DatabaseConfig

# SQLite (default)
db = DatabaseBackend(DatabaseConfig(url="sqlite:///analysis.db"))

# Create project and session
project_id = db.create_project("CAN Bus RE", "Automotive analysis")
session_id = db.create_session(project_id, "can")

# Store protocols and messages
protocol_id = db.store_protocol(session_id, "UDS", spec_json, confidence=0.95)
db.store_message(protocol_id, timestamp=1.5, data=b"\x02\x10\x01", decoded=fields)

# Query historical data
protocols = db.find_protocols(name_pattern="UDS%", min_confidence=0.8)
```

#### 15. REST API Server

```python
from oscura.api.rest_server import RESTAPIServer

# Start server
server = RESTAPIServer(host="0.0.0.0", port=8000)
server.start()

# Access at http://localhost:8000/docs for Swagger UI
```

**API Endpoints:**

- `POST /api/v1/analyze` - Upload and analyze captures
- `GET /api/v1/sessions` - List analysis sessions
- `GET /api/v1/protocols` - List discovered protocols
- `POST /api/v1/export/{format}` - Export results

#### 16. Enhanced CLI

```bash
# New analyze subcommand
oscura analyze signal.vcd --protocol auto --export wireshark

# New visualize subcommand
oscura visualize signal.wfm --interactive

# New benchmark subcommand
oscura benchmark --iterations 100
```

---

## Deprecations

None in v0.6.0. All APIs remain backward compatible except for the removed plugin system.

---

## Known Issues

### Portability Validation Warnings

Some audit output files contain hardcoded project names. This is a documentation-only issue and doesn't affect functionality.

---

## Performance Improvements

- **Parallel processing:** Up to 8x speedup on multi-core systems
- **Memory optimization:** Handle captures 10x larger with streaming
- **Caching:** FFT and correlation operations cached for repeated use
- **Database queries:** Indexed queries for fast historical lookup

---

## Recommended Actions After Upgrade

1. **Update imports** if using DBC generator or payload analysis directly
2. **Run tests** to verify compatibility: `./scripts/test.sh`
3. **Try new features** like ML classification and fuzzing
4. **Enable caching** for frequently-run analyses
5. **Consider database backend** for persistent storage

---

## Getting Help

- **Documentation:** https://oscura.readthedocs.io
- **Issues:** https://github.com/yourusername/oscura/issues
- **FAQ:** docs/faq/index.md
- **Examples:** demos/ directory

---

## Rollback

If you encounter issues, rollback to v0.5.1:

```bash
pip install oscura==0.5.1
```

Report issues at https://github.com/yourusername/oscura/issues

---

**Updated:** 2026-01-25  
**Version:** 0.6.0
