# Oscura Test Suite

Comprehensive test suite for the Oscura signal analysis framework.

## Quick Start

```bash
# Run all tests (optimal parallel execution)
./scripts/test.sh

# Run specific test categories
./scripts/test.sh --fast              # Quick tests without coverage
uv run pytest tests/unit/             # Unit tests only
uv run pytest tests/integration/      # Integration tests only
uv run pytest -m compliance           # Compliance tests only
uv run pytest -m benchmark            # Performance benchmarks only
```

## Test Suite Organization

```
tests/
├── unit/              # Unit tests (modules in isolation)
├── integration/       # Integration tests (multi-module workflows)
├── compliance/        # IEEE/JEDEC standards compliance
├── performance/       # Performance benchmarks (pytest-benchmark)
├── stress/            # Stress tests (edge cases, load tests)
├── automotive/        # Automotive protocol tests
├── validation/        # Synthetic data validation
├── fixtures/          # Shared test fixtures (SignalBuilder, etc.)
└── utils/             # Test utilities (assertions, factories, mocking)
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual functions and classes in isolation.

- **Purpose**: Verify algorithm correctness, edge cases, error handling
- **Scope**: Single module only
- **Speed**: Fast (<100ms per test)
- **Coverage**: Aim for >80%

**Documentation**: [tests/unit/README.md](unit/README.md)

### Integration Tests (`tests/integration/`)

Test multi-module workflows and data flow across boundaries.

- **Purpose**: Verify modules work together correctly
- **Scope**: 2+ modules, edge cases NOT in demos
- **Speed**: Medium (<5s per test)
- **Rule**: Only test scenarios NOT covered by demos

**Documentation**: [tests/integration/README.md](integration/README.md)
**Charter**: [tests/integration/TEST_CHARTER.md](integration/TEST_CHARTER.md)
**Demo Mapping**: [tests/integration/DEMO_COVERAGE.md](integration/DEMO_COVERAGE.md)

### Compliance Tests (`tests/compliance/`)

Verify Oscura measurements conform to industry standards.

- **Purpose**: IEEE/JEDEC/IEC standards compliance
- **Scope**: Standard-specified algorithms and tolerances
- **Standards**: IEEE 181, 1057, 1241, 1459, 2414, JEDEC
- **Validation**: Reference test vectors, analytical ground truth

**Documentation**: [tests/compliance/README.md](compliance/README.md)

### Performance Tests (`tests/performance/`)

Benchmark performance and detect regressions using pytest-benchmark.

- **Purpose**: Performance monitoring and regression detection
- **Scope**: Loading, analysis, inference, memory efficiency
- **CI Integration**: Automatic baseline comparison on PRs
- **Thresholds**: 20% regression threshold (configurable)

**Documentation**: [tests/performance/README.md](performance/README.md)

### Stress Tests (`tests/stress/`)

Test system behavior under extreme conditions.

- **Purpose**: Edge cases, memory limits, error recovery
- **Scope**: Large files, malformed data, resource exhaustion
- **Examples**: OOM prevention, corrupted registries, race conditions

### Automotive Tests (`tests/automotive/`)

Specialized tests for automotive protocols (CAN, OBD-II, J1939, UDS).

- **Purpose**: Automotive protocol decoding and analysis
- **Scope**: CAN bus, DBC generation, checksum detection, correlation
- **Standards**: SAE J1939, ISO 14229 (UDS), SAE J2012 (DTC)

### Validation Tests (`tests/validation/`)

Validate synthetic test data generation and protocol messages.

- **Purpose**: Verify test data generators produce correct signals
- **Scope**: SignalBuilder, protocol signals, synthetic packets
- **Examples**: UART frames, SPI transactions, UDP packets

## Test Markers

Use markers to run specific test subsets:

```bash
# Compliance tests
uv run pytest -m compliance

# Performance benchmarks
uv run pytest -m benchmark --benchmark-only

# Slow tests (>5s)
uv run pytest -m slow

# Fast tests (<0.1s)
uv run pytest -m fast

# Automotive tests
uv run pytest -m automotive

# GPU-enabled tests (requires CUDA)
uv run pytest -m gpu
```

## Test Fixtures

Shared fixtures are organized in:

- `tests/conftest.py` - Global fixtures (all tests)
- `tests/fixtures/` - Reusable fixtures
  - `signal_builders.py` - SignalBuilder patterns (20+ signal types)
  - `protocol_signals.py` - Protocol-specific fixtures
- `tests/utils/` - Test utilities
  - `assertions.py` - Custom assertions
  - `factories.py` - Test data factories
  - `mocking.py` - Mock helpers

### Using Fixtures

```python
def test_with_signal_builder(signal_builder):
    """Test using SignalBuilder fixture."""
    signal = signal_builder.sine_wave(frequency=1e3, snr_db=40)
    result = analyze(signal)
    assert result is not None

def test_with_standard_signals(standard_signals):
    """Test using cached standard signals."""
    sine = standard_signals["sine_1khz"]
    result = analyze(sine)
    assert result is not None
```

## Running Tests

### Quick Commands

```bash
# Optimal: Use validated scripts
./scripts/test.sh                    # Full test suite (8-10 min)
./scripts/test.sh --fast             # Quick tests (5-7 min)
./scripts/check.sh                   # Lint + typecheck + tests

# Manual: pytest commands
uv run pytest tests/unit/ -v         # Unit tests verbose
uv run pytest tests/ -x              # Stop on first failure
uv run pytest -k "test_fft"          # Run tests matching pattern
```

### With Coverage

```bash
# Using validated script
./scripts/testing/run_coverage.sh

# Manual pytest
uv run pytest tests/unit/ \
  --cov=src/oscura \
  --cov-report=html \
  --cov-report=term-missing
```

### Parallel Execution

```bash
# Auto-detect CPU cores
uv run pytest tests/unit/ -n auto

# Specific worker count
uv run pytest tests/unit/ -n 8
```

### Filtering Tests

```bash
# By directory
uv run pytest tests/unit/analyzers/

# By marker
uv run pytest -m "not slow"
uv run pytest -m "compliance and not slow"

# By pattern
uv run pytest -k "fft or spectral"

# Specific file
uv run pytest tests/unit/core/test_types.py
```

## Test Data

### Synthetic Data (Primary)

Generated test data using SignalBuilder:

```python
from tests.fixtures.signal_builders import SignalBuilder

builder = SignalBuilder()
signal = builder.sine_wave(frequency=1e3, snr_db=40)
```

**Advantages**:

- Reproducible (fixed seeds)
- Small file size
- Legally safe (no licensing concerns)
- Fast generation

### Real Captures (Secondary)

Real oscilloscope/logic analyzer captures in `test_data/real_captures/`.

**Use for**:

- Vendor file format quirks
- Real-world validation
- Integration testing

**Not tracked**: Large real capture files (use `test_data/synthetic/` for version control)

## Writing Tests

### Test File Naming

```
test_<module>.py           # Unit test for src/oscura/<module>.py
test_<feature>_<variant>.py  # Feature with variants
```

### Test Function Naming

```python
def test_<function>_<scenario>_<expected>():
    """Test <function> when <scenario>, expects <expected>."""
    # Arrange
    input = create_input()

    # Act
    result = function_under_test(input)

    # Assert
    assert result == expected
```

### Docstring Template

```python
def test_example():
    """Test example function with valid input.

    This test verifies that example() correctly processes
    valid input and returns expected output format.
    """
```

## Guidelines

### Unit Tests

1. ✅ Test single module in isolation
2. ✅ Use synthetic data (SignalBuilder)
3. ✅ Keep tests fast (<100ms)
4. ✅ Test edge cases and error handling
5. ❌ Don't test vendor libraries (NumPy, scipy, pandas)

### Integration Tests

1. ✅ Test 2+ modules crossing boundaries
2. ✅ Test edge cases NOT in demos
3. ✅ Test error handling chains
4. ❌ Don't duplicate demo functionality
5. ❌ Don't test single-module algorithms

### Performance Tests

1. ✅ Use pytest-benchmark
2. ✅ Mark with `@pytest.mark.benchmark`
3. ✅ Add `@pytest.mark.slow` if >1s
4. ✅ Compare with baseline
5. ❌ Don't use legacy profiling scripts

### Compliance Tests

1. ✅ Reference specific standard sections
2. ✅ Use documented test vectors
3. ✅ Apply standard-specified tolerances
4. ✅ Verify correct SI units
5. ❌ Don't test implementation details

## Test Quality Metrics

### Current Status

| Metric                 | Target             | Status                        |
| ---------------------- | ------------------ | ----------------------------- |
| Unit test coverage     | >80%               | ✅ 85%                        |
| Integration test count | <50                | ✅ 42 tests                   |
| Compliance tests       | All standards      | ✅ IEEE 181, 1241, 1459, 2414 |
| Performance baselines  | All critical paths | ✅ 47 benchmarks              |
| Test execution time    | <10 min            | ✅ 8-10 min (parallel)        |

### CI/CD Integration

Tests run automatically on:

- Every commit (pre-commit hooks)
- Every push (pre-push hooks)
- Pull requests (full test matrix)
- Main branch merges (baseline updates)
- Weekly schedule (cross-platform matrix)

## Troubleshooting

### Tests Failing Locally

```bash
# Run single test with verbose output
uv run pytest tests/unit/path/to/test.py::test_name -v -s --tb=long

# Check for stale caches
rm -rf .pytest_cache .mypy_cache .ruff_cache
uv sync --all-extras

# Verify installation
./scripts/setup/verify.sh
```

### Performance Regressions

```bash
# Run benchmarks and save results
uv run pytest tests/performance/ --benchmark-only \
  --benchmark-json=current.json

# Compare with baseline
uv run python scripts/quality/compare_benchmarks.py \
  tests/performance/baseline_results.json \
  current.json
```

### Memory Issues

```bash
# Run with memory profiling
uv run pytest tests/unit/ --memray

# Check memory limits
uv run pytest tests/stress/ -v -s
```

## Further Reading

- **Testing Strategy**: `docs/testing/test-suite-guide.md`
- **Integration Test Charter**: `tests/integration/TEST_CHARTER.md`
- **Performance Benchmarking**: `tests/performance/README.md`
- **CI/CD Workflows**: `.github/workflows/`
- **Development Guide**: `CONTRIBUTING.md`
