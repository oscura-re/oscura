# Test Suite Guide

**Purpose**: Complete reference for understanding and working with Oscura's test suite
**Audience**: Contributors, maintainers, CI/CD engineers
**Status**: v1.0 - 2026-01-15

---

## Overview

Oscura maintains a comprehensive test suite with **18,324 tests** across unit, integration, performance, and compliance testing. The suite achieves **0.00% skip rate** (ZERO permanent skips - all tests passing) and uses optimal pytest configuration for fast, reliable execution.

### Quick Stats

| Metric             | Value               | Status             |
| ------------------ | ------------------- | ------------------ |
| **Total Tests**    | 18,324              | ✅ All functional  |
| **Skip Rate**      | 0.00%               | ✅ ZERO skips      |
| **Pass Rate**      | 100%                | ✅ Perfect         |
| **Execution Time** | 8-10 min (parallel) | ✅ Optimal         |
| **Coverage**       | >80%                | ✅ Meets threshold |
| **Dependencies**   | All installed       | ✅ Complete        |

---

## Test Organization

### Directory Structure

```
tests/
├── unit/                    # 18,000+ unit tests (fast, isolated)
│   ├── analyzers/          # Signal analysis tests
│   ├── loaders/            # File format loading tests
│   ├── protocols/          # Protocol decoder tests
│   ├── visualization/      # Plotting and display tests
│   ├── core/               # Core functionality tests
│   └── ...
├── integration/            # Integration workflow tests
│   ├── test_integration_workflows.py
│   ├── test_wfm_loading.py
│   ├── TEST_CHARTER.md
│   └── DEMO_COVERAGE.md
├── performance/            # Performance benchmarks
│   └── test_benchmarks.py
├── stress/                 # Stress and load tests
│   └── test_performance.py
├── compliance/             # Standards compliance tests
├── fixtures/               # Shared test fixtures
│   └── signal_builders.py  # SignalBuilder infrastructure
└── conftest.py             # Global pytest configuration

demos/                      # Working demonstrations with validation
```

### Test Types

| Type            | Count   | Purpose                        | Execution Time |
| --------------- | ------- | ------------------------------ | -------------- |
| **Unit**        | ~18,000 | Fast, isolated, no I/O         | <1s each       |
| **Integration** | ~300    | Multi-component workflows      | 1-5s each      |
| **Performance** | ~20     | Performance benchmarks         | 5-30s each     |
| **Compliance**  | ~50     | IEEE/JEDEC standard validation | Variable       |
| **Stress**      | ~10     | High-load, memory-intensive    | 10-60s each    |

---

## Pytest Markers

### Primary Level Markers

```python
@pytest.mark.unit              # Unit tests (fast, isolated, no I/O)
@pytest.mark.integration       # Integration tests (slower, multiple components)
@pytest.mark.performance       # Performance benchmark tests
@pytest.mark.stress            # Stress tests (high load, memory intensive)
@pytest.mark.compliance        # Standards compliance tests
@pytest.mark.validation        # Ground truth validation tests
```

### Domain Markers (Top-Level)

```python
@pytest.mark.analyzer          # Analyzer module tests
@pytest.mark.loader            # Loader module tests
@pytest.mark.inference         # Protocol inference tests
@pytest.mark.exporter          # Exporter module tests
@pytest.mark.core              # Core functionality tests
@pytest.mark.visualization     # Visualization tests
@pytest.mark.workflow          # Workflow-specific tests
@pytest.mark.automotive        # Automotive/CAN bus tests
```

### Subdomain Markers (Specific)

```python
@pytest.mark.digital           # Digital signal analysis
@pytest.mark.spectral          # Spectral analysis
@pytest.mark.statistical       # Statistical analysis
@pytest.mark.protocol          # Protocol analysis
@pytest.mark.pattern           # Pattern recognition
@pytest.mark.power             # Power analysis
@pytest.mark.jitter            # Jitter measurement
@pytest.mark.eye               # Eye diagram analysis
@pytest.mark.packet            # Packet analysis
```

### Special Markers

```python
@pytest.mark.slow              # Tests taking >1 second
@pytest.mark.memory_intensive  # Tests requiring >100MB memory
@pytest.mark.requires_data     # Tests needing test_data directory
@pytest.mark.requires_optional # Tests needing optional dependencies
@pytest.mark.hypothesis        # Property-based tests (Hypothesis)
@pytest.mark.requirement("ID") # Link test to requirement ID
```

### Usage Examples

```python
# Unit test for digital analyzer with requirement traceability
@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DIG-001")
def test_logic_threshold_detection():
    pass

# Performance test that should skip in CI
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Performance tests unreliable in CI"
)
def test_large_file_loading_performance():
    pass

# Integration test requiring optional dependencies
@pytest.mark.integration
@pytest.mark.requires_optional
def test_can_analysis_workflow():
    pass
```

---

## Running Tests

### Quick Commands

```bash
# Run all tests (optimal configuration)
./scripts/test.sh

# Run fast tests only (no coverage)
./scripts/test.sh --fast

# Run specific test file
uv run pytest tests/unit/analyzers/test_digital.py

# Run tests matching pattern
uv run pytest -k "test_uart"

# Run tests with specific marker
uv run pytest -m "digital"

# Run tests excluding slow tests
uv run pytest -m "not slow"
```

### Advanced Usage

```bash
# Run with custom worker count
uv run pytest -n 4 tests/

# Run with coverage report
./scripts/run_coverage.sh

# Run performance benchmarks only
uv run pytest -m "performance" --benchmark-only

# Run integration tests with verbose output
uv run pytest tests/integration/ -v

# Run tests and stop at first failure
uv run pytest -x tests/

# Run tests with detailed failure output
uv run pytest tests/ --tb=long
```

### CI/CD Usage

```bash
# Full CI validation (use in pre-push hooks)
./scripts/pre-push.sh --full

# Run all quality checks
./scripts/check.sh

# Auto-fix linting issues
./scripts/fix.sh
```

---

## Test Configuration

### pyproject.toml Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

addopts = [
    "-ra",                    # Show summary of all test outcomes
    "--strict-markers",       # Error on unknown markers
    "--strict-config",        # Error on config mistakes
]

filterwarnings = [
    "error",                  # Treat warnings as errors
    "ignore::DeprecationWarning",  # Ignore deprecation warnings
]

timeout = 60                  # Default test timeout (seconds)
timeout_method = "thread"     # Use thread-based timeouts
console_output_style = "progress"
```

### Optimal Execution Settings

**Worker Count** (auto-detected by `scripts/test.sh`):

- Local: `min(cpu_count - 1, 6)` workers
- CI: 4 workers
- Low memory: 2 workers (systems <16GB RAM)

**Distribution Strategy**:

- `--dist loadscope` - Groups tests by scope (5-10% faster than loadfile)
- Keeps fixtures scoped together for efficiency

**Timeouts**:

- Default: 60 seconds per test
- Thread-based (not signal-based) for compatibility
- Set in pyproject.toml, overridable with `@pytest.mark.timeout()`

---

## Test Fixtures

### SignalBuilder Infrastructure

Located in `tests/fixtures/signal_builders.py` (404 LOC), provides 20+ standardized signal generation methods:

```python
# Import SignalBuilder
from tests.fixtures.signal_builders import SignalBuilder

# Use in tests via fixture
def test_signal_analysis(signal_builder):
    # Generate common signals
    square = signal_builder.square_wave(frequency=1000, duty_cycle=0.5)
    sine = signal_builder.sine_wave(frequency=1000, amplitude=1.0)
    noisy = signal_builder.noisy_signal(signal, snr_db=20)

    # Generate protocol signals
    uart = signal_builder.uart_signal(data=[0x55, 0xAA], baud_rate=9600)
    spi = signal_builder.spi_signal(mosi_data=[0x12, 0x34])

    # Generate specialized signals
    jitter = signal_builder.jittery_clock(frequency=1000, jitter_std=1e-9)
    chirp = signal_builder.chirp_signal(f0=100, f1=1000)
```

### Available Signal Types

**Basic Waveforms**:

- `square_wave()` - Square wave with configurable duty cycle
- `sine_wave()` - Pure sine wave
- `triangle_wave()` - Triangle wave
- `sawtooth_wave()` - Sawtooth wave
- `pwm_signal()` - PWM with variable duty cycle

**Protocol Signals**:

- `uart_signal()` - UART serial data
- `spi_signal()` - SPI master/slave
- `i2c_signal()` - I2C bus communication
- `can_signal()` - CAN bus messages

**Analysis Signals**:

- `noisy_signal()` - Add noise to any signal
- `jittery_clock()` - Clock with jitter
- `multi_tone_signal()` - Multiple frequency components
- `chirp_signal()` - Frequency sweep
- `eye_diagram_signal()` - Eye diagram test pattern

**Statistical**:

- `repeating_pattern_signal()` - Repeating data pattern
- `anomaly_signal()` - Signal with anomalies
- `statistical_test_signals()` - Signals for statistical testing

### Global Fixtures (conftest.py)

```python
# Automatically available in all tests
@pytest.fixture(scope="module")
def signal_builder():
    """Provides SignalBuilder instance."""
    return SignalBuilder(sample_rate=1e6, duration=0.01)

@pytest.fixture
def temp_dir(tmp_path):
    """Provides temporary directory."""
    return tmp_path

@pytest.fixture
def test_data_dir():
    """Provides test data directory path."""
    return Path(__file__).parent / "test_data"
```

---

## Skip Policy

### Current Status

- **Total skips**: 0 (0.00% of 18,324 tests)
- **ZERO permanent skips** - All previously skipped tests have been fixed
- **Test suite in IDEAL STATE** - 100% of tests passing

### Previously Fixed Skips

#### 1. Async Decorator Support ✅ FIXED

**File**: `tests/unit/core/test_correlation.py`
**Issue**: Decorator didn't support async functions (implementation limitation)
**Fixed**: Added async function detection and dual wrapper paths (commit b2f5795)
**Status**: ✅ Implemented and passing

#### 2. Chunked FFT Correlation ✅ FIXED

**File**: `tests/unit/analyzers/statistical/test_chunked_corr.py`
**Issue**: Known issue in chunked FFT correlation algorithm
**Fixed**: Replaced buggy custom implementation with scipy.signal.correlate() (commit b2f5795)
**Status**: ✅ Implemented and passing

### Conditional Skips (Runtime)

Tests may skip at runtime for legitimate reasons:

- **Test data unavailable** - Optional test files not present
- **OS limitations** - Symlinks, path length, etc.
- **Optional dependencies** - matplotlib, PyWavelets (all installed in full setup)

These are proper runtime checks and do NOT indicate test suite issues.

### When to Add Skips

**DO add skip**:

- Known bug in implementation (with issue link)
- Design limitation (with explanation)
- Optional test data not available
- OS-specific limitation

**DO NOT add skip**:

- Test is flaky (fix the test)
- Test is slow (use @pytest.mark.slow instead)
- Test needs refactoring (refactor it)
- Mock is complex (simplify or delete test)

---

## Performance Tests

### Guidelines

1. **Mark appropriately**:

   ```python
   @pytest.mark.performance
   @pytest.mark.slow
   ```

2. **Skip in CI** (timing unreliable):

   ```python
   @pytest.mark.skipif(
       os.getenv("CI") == "true",
       reason="Performance tests unreliable in CI"
   )
   ```

3. **Use relative comparisons**, not absolute times:

   ```python
   # BAD - environment-dependent
   assert elapsed < 5.0

   # GOOD - relative comparison
   assert optimized_time < baseline_time * 0.5
   ```

4. **Document performance expectations**:

   ```python
   def test_large_file_loading():
       """Test that 100MB file loads in reasonable time.

       Baseline: ~2s on standard hardware
       Expected: <5s with 2x safety margin
       """
   ```

### Running Benchmarks

```bash
# Run pytest-benchmark suite
uv run pytest tests/performance/ --benchmark-only

# Save benchmark results
uv run pytest tests/performance/ --benchmark-only --benchmark-json=results.json

# Compare against baseline
uv run pytest tests/performance/ --benchmark-only --benchmark-compare

# Run performance tests (skip in CI)
uv run pytest -m "performance" --ignore=tests/performance/
```

---

## Dependencies

### Core Dependencies

Always installed:

- pytest, pytest-xdist (parallel execution)
- pytest-timeout (timeout handling)
- numpy, scipy, pandas (data processing)

### Optional Dependencies

Install with `uv sync --all-extras`:

- **matplotlib** - Visualization tests (253 tests)
- **PyWavelets** - Wavelet analysis tests (28 tests)
- **PyYAML** - YAML configuration tests (15 tests)
- **openpyxl** - Excel export tests
- **networkx** - Graph analysis tests
- **RigolWFM** - Rigol oscilloscope tests
- **asammdf** - MDF file tests
- **cantools** - CAN database tests
- **scapy** - PCAP analysis tests
- **h5py** - HDF5 file tests
- **jupyter** - Notebook tests
- **python-pptx** - PowerPoint export tests

**Installation**:

```bash
# Install all dependencies (REQUIRED for zero skips)
uv sync --all-extras

# Verify installation
uv pip list | grep -iE "matplotlib|pywavelets|pyyaml"
```

---

## Common Patterns

### Writing New Tests

```python
# Good test structure
@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SPEC-001")
def test_feature_basic_functionality(signal_builder):
    """Test basic functionality of feature X.

    Tests that feature X correctly processes input Y
    and produces expected output Z.
    """
    # Arrange - Set up test data
    signal = signal_builder.sine_wave(frequency=1000)

    # Act - Execute the function
    result = analyze_signal(signal)

    # Assert - Verify results
    assert result.frequency == pytest.approx(1000, rel=0.01)
    assert result.amplitude > 0
```

### Testing Exceptions

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        process_signal(sample_rate=-1)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("frequency,expected", [
    (100, 100.0),
    (1000, 1000.0),
    (10000, 10000.0),
])
def test_frequency_detection(signal_builder, frequency, expected):
    """Test frequency detection across range."""
    signal = signal_builder.sine_wave(frequency=frequency)
    detected = detect_frequency(signal)
    assert detected == pytest.approx(expected, rel=0.01)
```

---

## Best Practices

### Test Design

1. **Keep tests focused** - One concept per test
2. **Use descriptive names** - `test_uart_detects_framing_error_on_invalid_stop_bit`
3. **Arrange-Act-Assert** - Clear test structure
4. **Test behavior, not implementation** - Test what, not how
5. **Use fixtures** - DRY principle for test data

### Test Performance

1. **Use module-scoped fixtures** for expensive setup
2. **Generate minimal test data** - Just enough to test
3. **Avoid file I/O** - Use in-memory data when possible
4. **Mark slow tests** - Use @pytest.mark.slow
5. **Run in parallel** - Use `./scripts/test.sh`

### Test Maintenance

1. **Delete unused tests** - Don't keep "just in case"
2. **Update docstrings** - Document what test validates
3. **Link to requirements** - Use @pytest.mark.requirement
4. **Fix flaky tests** - Don't ignore intermittent failures
5. **Review skips regularly** - Ensure all skips still valid

---

## Troubleshooting

### Common Issues

#### Tests Skipping Unexpectedly

```bash
# Check why tests are skipping
uv run pytest tests/ -v --tb=short -ra | grep SKIP

# Check for missing dependencies
uv pip list | grep -iE "matplotlib|pywavelets|pyyaml"

# Reinstall all dependencies
uv sync --all-extras
```

#### Tests Timing Out

```bash
# Increase timeout for specific test
@pytest.mark.timeout(120)
def test_large_file_processing():
    pass

# Disable timeout for debugging
@pytest.mark.timeout(0)
def test_complex_analysis():
    pass
```

#### Parallel Execution Issues

```bash
# Run tests sequentially
uv run pytest tests/ -n 0

# Run with fewer workers
uv run pytest tests/ -n 2

# Check for shared state issues
uv run pytest tests/ --lf  # Run last-failed tests
```

#### Memory Issues

```bash
# Run with memory profiling
uv run pytest tests/ --memray

# Limit worker count
uv run pytest tests/ -n 2

# Run specific test file
uv run pytest tests/unit/specific_test.py
```

---

## CI/CD Integration

### GitHub Actions Configuration

Tests run in CI via `.github/workflows/tests-chunked.yml`:

- **Sharding**: 8 parallel chunks for speed
- **Matrix**: Python 3.12, 3.13
- **Timeout**: 60 minutes per chunk
- **Coverage**: Uploaded to Codecov
- **Artifacts**: Test results retained 30 days

### Pre-Push Validation

```bash
# Run full CI-equivalent validation
./scripts/pre-push.sh --full

# Includes:
# - All unit tests
# - Integration tests
# - Linting (ruff)
# - Type checking (mypy)
# - Shell checking (shellcheck)
# - YAML linting (yamllint)
# - Coverage >80%
```

---

## Resources

### Documentation

- `docs/testing/index.md` - Testing strategy overview
- `tests/integration/TEST_CHARTER.md` - Integration test charter
- `tests/integration/DEMO_COVERAGE.md` - Demo coverage matrix
- `CONTRIBUTING.md` - Contributing guidelines
- `CLAUDE.md` - Project context for AI assistants

### Scripts

- `./scripts/test.sh` - Primary test execution (SSOT)
- `./scripts/test.sh --fast` - Quick tests without coverage
- `./scripts/check.sh` - All quality checks
- `./scripts/fix.sh` - Auto-fix linting issues
- `./scripts/pre-push.sh --full` - Full validation

### External Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-xdist](https://pytest-xdist.readthedocs.io/) - Parallel execution
- [pytest-timeout](https://pypi.org/project/pytest-timeout/) - Timeout handling
- [Hypothesis](https://hypothesis.readthedocs.io/) - Property-based testing

---

## Changelog

**v1.1 (2026-01-15)**: Zero skips achievement

- Updated to reflect 0 permanent skips (18,324 tests, 100% passing)
- Documented async decorator and chunked FFT fixes
- All previously skipped tests now implemented and passing

**v1.0 (2026-01-15)**: Initial comprehensive guide

- Documented all tests and markers
- Added pytest configuration details
- Documented SignalBuilder infrastructure
- Added skip policy and best practices
- Added troubleshooting guide

---

**Maintained by**: Oscura Maintainers
**Last Updated**: 2026-01-15
**Status**: ✅ Complete and up-to-date
