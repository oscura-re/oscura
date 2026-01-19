# Testing Guide

> **Version**: 0.1.0 | **Last Updated**: 2026-01-17

Comprehensive guide to running and writing tests for Oscura using our modern testing infrastructure.

## Testing Philosophy

Oscura follows these testing principles:

- **Fast feedback loops** - Use incremental testing (testmon) during development
- **High isolation** - Tests run independently in any order (enforced by pytest-randomly)
- **Reliable execution** - Automatic retry for flaky tests (pytest-rerunfailures)
- **Resource efficiency** - Memory profiling (pytest-memray) and smart parallelization (pytest-split)
- **Quality gates** - Strict marker validation and diff coverage enforcement

## Quick Reference

### Common Commands

```bash
# Fast incremental testing (only run tests affected by changes)
pytest --testmon

# Run all unit tests with optimal parallelization
pytest tests/unit -n auto

# Run with coverage
pytest tests/unit --cov=src/oscura --cov-report=term

# Run specific module
pytest tests/unit/analyzers -v

# Debug single test in VS Code
# Use "Debug Current Test File" launch configuration (F5)

# Check test isolation
python scripts/testing/check_test_isolation.py --sample 15

# Profile memory usage
pytest tests/unit --memray --most-allocations=10
```

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── analyzers/    # Analyzer tests
│   ├── loaders/      # Loader tests
│   ├── protocols/    # Protocol tests
│   └── ...
├── integration/       # Integration tests
├── compliance/        # IEEE/JEDEC compliance tests
├── performance/       # Benchmark tests
└── conftest.py       # Shared fixtures
```

## New Testing Tools

### 1. pytest-testmon - Incremental Testing

**Purpose**: Run only tests affected by code changes for faster feedback during development.

**Basic Usage**:

```bash
# Enable testmon (tracks which tests need rerunning)
pytest --testmon

# Disable testmon for full run
pytest --no-testmon

# Clear testmon cache and start fresh
pytest --testmon-nocache
```

**How it works**: Testmon tracks code dependencies and only runs tests affected by your changes. Perfect for TDD and rapid iteration.

**Best practices**:

- Use `--testmon` for local development
- Use `--no-testmon` in CI (full coverage needed)
- Clear cache after major refactoring: `pytest --testmon-nocache`

### 2. pytest-randomly - Randomized Test Order

**Purpose**: Detect test order dependencies by randomizing execution order.

**Basic Usage**:

```bash
# Randomize test order (enabled by default)
pytest tests/unit

# Use specific seed to reproduce order
pytest --randomly-seed=12345

# Repeat last run's order
pytest --randomly-seed=last

# Disable randomization for debugging
pytest --randomly-dont-shuffle
```

**How it works**: Tests run in random order each time. If tests fail only in certain orders, they have hidden dependencies.

**CI Integration**: Weekly randomized runs detect order dependencies automatically (see `.github/workflows/test-quality.yml`).

### 3. pytest-rerunfailures - Automatic Retry

**Purpose**: Automatically retry flaky tests to distinguish transient failures from real bugs.

**Basic Usage**:

```bash
# Retry failed tests up to 2 times with 1 second delay
pytest --reruns 2 --reruns-delay 1

# Retry only specific markers
pytest -m "integration" --reruns 3

# Skip retry for specific test
@pytest.mark.no_rerun
def test_critical_no_retry():
    pass
```

**How it works**: Failed tests are automatically rerun. If they pass on retry, marked as flaky. Helps identify intermittent issues.

**CI Integration**: All CI test runs use `--reruns 2 --reruns-delay 1` by default.

### 4. pytest-split - Duration-Based Test Sharding

**Purpose**: Split tests into balanced groups based on duration for parallel CI execution.

**Basic Usage**:

```bash
# Split tests into 8 groups, run group 1
pytest --splits 8 --group 1

# Split tests into 4 groups, run group 3
pytest --splits 4 --group 3

# Store timing data for better splits
pytest --store-durations

# Use stored durations for balancing
pytest --splits 8 --group 1 --splitting-algorithm duration_based_chunks
```

**How it works**: Tests are distributed across groups to balance execution time. Uses historical duration data for optimal balancing.

**CI Integration**: CI uses test sharding for balanced parallel execution across matrix jobs.

### 5. pytest-memray - Memory Profiling

**Purpose**: Profile memory usage to detect leaks and optimize memory consumption.

**Basic Usage**:

```bash
# Enable memory profiling
pytest --memray

# Show top 10 memory allocations
pytest --memray --most-allocations=10

# Save memory profiles to directory
pytest --memray --memray-bin-path=./memray-results/

# Profile specific test
pytest tests/unit/analyzers/test_dsp.py --memray
```

**Using Memory Markers**:

```python
import pytest

@pytest.mark.limit_memory("100 MB")
def test_bounded_memory():
    """Test will fail if peak memory exceeds 100MB."""
    signal = np.random.randn(1_000_000)
    process(signal)

@pytest.mark.limit_leaks("1 MB")
def test_no_memory_leaks():
    """Test will fail if any call stack leaks >1MB."""
    for _ in range(100):
        process_data()
```

**Best practices**:

- Use `@pytest.mark.limit_memory()` for memory-intensive tests
- Profile regularly during development to catch leaks early
- Check memory usage before marking tests as `memory_intensive`

### 6. diff-cover - PR Coverage Enforcement

**Purpose**: Enforce coverage requirements on changed lines in pull requests.

**Basic Usage**:

```bash
# Run tests with coverage
pytest --cov=src/oscura --cov-report=xml

# Check coverage on changed lines (vs main branch)
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --fail-under=80 \
  --html-report=diff-coverage.html
```

**How it works**: Analyzes git diff to find changed lines and checks their test coverage. Enforces 80% coverage threshold.

**CI Integration**: Automatic diff coverage checks on all PRs with comment posted to PR.

## CI/CD Features

### Test Retry Policy

All CI test runs automatically retry failed tests:

```bash
pytest --reruns 2 --reruns-delay 1
```

**Configuration**:

- Maximum 2 retries per failed test
- 1 second delay between retries
- Helps distinguish flaky tests from real failures

### Weekly Randomized Test Runs

Every Sunday at 4 AM UTC, CI runs tests with randomized order:

```bash
pytest --randomly-seed=auto -n 4
```

**Purpose**: Detect test order dependencies that could hide bugs.

**What happens on failure**:

- GitHub issue automatically created with label `test-order-dependency`
- Workflow artifacts contain failure details
- Issue includes instructions for local reproduction

### Strict Isolation Check Enforcement

Every PR validates test isolation:

```bash
python scripts/testing/check_test_isolation.py --sample 15
```

**Configuration**:

- Samples 15 random test files
- Runs each test file twice in different orders
- Fails CI if results differ (indicates shared state)

**Rationale**: Prevents test pollution where tests affect each other.

### Diff Coverage Enforcement

All PRs require 80% coverage on changed lines:

```bash
diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

**Features**:

- Automatic PR comment with coverage report
- HTML report artifact for detailed analysis
- Markdown report for easy reading

**Rationale**: Ensures new code is well-tested without requiring 100% overall coverage.

### Benchmark Regression Detection

PRs automatically compare performance to main branch:

```bash
python scripts/compare_benchmarks.py \
  baseline.json \
  current.json \
  --threshold 15
```

**Configuration**:

- 15% regression threshold (stricter than before)
- Automatic PR comment with benchmark comparison
- Fails CI if regression exceeds threshold

**What gets compared**:

- Mean execution time
- Memory usage
- Number of iterations per second

## VS Code Integration

### Updated Settings

Oscura includes optimized VS Code settings in `.vscode/settings.json`:

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "--tb=short",
    "-v",
    "--strict-markers",
    "--strict-config"
  ],
  "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

**Features**:

- Automatic test discovery on save
- Strict marker validation
- Short traceback format for readability

### Debug Configurations

Use `.vscode/launch.json` for debugging (press F5):

**Available configurations**:

1. **Debug Current Test File** - Debug the test file currently open
2. **Debug Single Test (under cursor)** - Debug specific test under cursor
3. **Debug All Unit Tests** - Debug entire unit test suite
4. **Debug Integration Tests** - Debug integration tests
5. **Debug Fast Tests Only** - Debug only fast tests (skip slow/memory-intensive)
6. **Debug Marked Tests** - Debug tests with specific marker (prompts for marker name)
7. **Debug with Coverage** - Debug with coverage report generation

**Quick debugging workflow**:

1. Set breakpoints by clicking left of line numbers
2. Open test file in editor
3. Press **F5** to launch "Debug Current Test File"
4. Use **F10** (step over), **F11** (step into), **Shift+F11** (step out)
5. Inspect variables in Debug Console

**Debugging specific test**:

1. Place cursor on test name
2. Select test name
3. Press **F5** and choose "Debug Single Test (under cursor)"
4. Or right-click test name and select "Debug Test"

## Local Development Workflow

### Fast Feedback Loop with Testmon

```bash
# First run: Run all tests and build dependency graph
pytest --testmon

# Make code changes in src/oscura/analyzers/digital.py

# Second run: Only runs tests affected by digital.py changes
pytest --testmon  # Much faster!

# Make more changes...

# Continue using testmon for rapid iteration
pytest --testmon
```

**When to clear cache**:

- After major refactoring
- When testmon seems to miss changed tests
- After merging main branch

```bash
pytest --testmon-nocache  # Clears cache and rebuilds
```

### Running Specific Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only fast tests (skip slow and memory-intensive)
pytest -m "unit and not slow and not memory_intensive"

# Run analyzer tests only
pytest -m analyzer

# Run protocol decoder tests
pytest -m protocol

# Run digital signal analysis tests
pytest -m digital

# Combine markers
pytest -m "analyzer and digital and not slow"
```

**List all available markers**:

```bash
pytest --markers
```

### Debugging Tests in VS Code

**Method 1: Using launch configurations (recommended)**

1. Open test file in editor
2. Press **F5** (or Run > Start Debugging)
3. Select "Debug Current Test File"
4. Set breakpoints as needed
5. Use debugger controls to step through code

**Method 2: Using Test Explorer**

1. Open Test Explorer (Testing icon in sidebar)
2. Right-click test
3. Select "Debug Test"

**Method 3: Using inline code lens**

1. Enable code lens in settings
2. Click "Debug" link above test function
3. Debugger launches automatically

**Debugging tips**:

- Use **F9** to toggle breakpoints
- Use **Debug Console** to evaluate expressions
- Use **Watch** panel to monitor variables
- Use **Call Stack** to see execution path

### Checking Test Isolation Locally

Before pushing changes, verify test isolation:

```bash
# Check isolation for 15 random test files
python scripts/testing/check_test_isolation.py --sample 15

# Check specific test file
python scripts/testing/check_test_isolation.py tests/unit/analyzers/test_digital.py

# Check all test files (slow)
python scripts/testing/check_test_isolation.py --all
```

**What it checks**:

- Runs each test file twice in different orders
- Compares results - should be identical
- Reports any differences (indicates shared state)

**If isolation check fails**:

1. Review the reported differences
2. Look for global state, class variables, or module-level caches
3. Use fixtures to isolate test data
4. Ensure cleanup with `yield` fixtures or `autouse` fixtures

## Running Tests

### Unit Tests

```bash
# All unit tests with optimal parallelization
pytest tests/unit -n auto

# Specific file
pytest tests/unit/test_loaders.py -v

# Specific test
pytest tests/unit/test_loaders.py::TestCSVLoader::test_basic -v

# Pattern matching
pytest tests/unit -k "frequency" -v

# With incremental testing (testmon)
pytest tests/unit --testmon
```

### With Coverage

```bash
# Terminal report
pytest tests/unit --cov=src/oscura --cov-report=term

# HTML report
pytest tests/unit --cov=src/oscura --cov-report=html
open htmlcov/index.html

# Specific module coverage
pytest tests/unit/analyzers --cov=src/oscura/analyzers

# With diff coverage (vs main branch)
pytest --cov=src/oscura --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main
```

### Parallel Execution

```bash
# Use 4 workers
pytest tests/unit -n 4

# Auto-detect CPU count
pytest tests/unit -n auto

# With memory isolation
pytest tests/unit -n 4 --dist loadfile

# Limit worker restarts (prevent memory leaks)
pytest tests/unit -n 4 --max-worker-restart=2
```

### Filtering Tests

```bash
# By marker
pytest -m "not slow"
pytest -m "slow"
pytest -m "analyzer and digital"

# By keyword
pytest -k "uart or spi"

# Skip slow tests (default in VS Code)
pytest tests/unit -m "not slow and not memory_intensive"

# Run only failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff
```

## Test Markers

Oscura uses a comprehensive marker system. All markers are registered in `pyproject.toml`.

### Test Level Markers

```python
@pytest.mark.unit
def test_basic():
    """Unit test (fast, isolated)."""
    pass

@pytest.mark.integration
def test_workflow():
    """Integration test (multiple components)."""
    pass

@pytest.mark.slow
def test_long_running():
    """Test taking >1 second."""
    pass

@pytest.mark.memory_intensive
def test_large_data():
    """Test using >100MB memory."""
    pass
```

### Domain Markers

```python
@pytest.mark.analyzer
def test_analyzer():
    """Analyzer module test."""
    pass

@pytest.mark.loader
def test_loader():
    """Loader module test."""
    pass

@pytest.mark.protocol
def test_protocol():
    """Protocol decoder test."""
    pass

@pytest.mark.digital
def test_digital_signal():
    """Digital signal analysis test."""
    pass
```

### Marker Validation

Oscura enforces marker correctness:

```bash
# Validate markers (strict mode)
python scripts/testing/validate_test_markers.py --strict

# Auto-fix missing markers
python scripts/testing/validate_test_markers.py --fix

# Show marker distribution
python scripts/testing/validate_test_markers.py
```

**CI Integration**: Marker validation runs on every PR.

## Writing Tests

### Test Structure

```python
"""Test module docstring.

Tests for oscura.analyzers.spectral module.
"""
import pytest
import numpy as np
import oscura as osc


class TestFFT:
    """Tests for FFT computation."""

    def test_basic_fft(self):
        """Test basic FFT computation."""
        # Arrange
        trace = generate_sine_wave(frequency=1e6)

        # Act
        spectrum = osc.compute_fft(trace)

        # Assert
        peak_idx = spectrum.magnitude_db.argmax()
        assert abs(spectrum.frequencies[peak_idx] - 1e6) < 1000

    def test_empty_input(self):
        """Test FFT with empty input raises error."""
        with pytest.raises(ValueError):
            osc.compute_fft([])

    @pytest.mark.slow
    def test_large_fft(self):
        """Test FFT with large dataset."""
        trace = generate_sine_wave(num_samples=10_000_000)
        spectrum = osc.compute_fft(trace)
        assert len(spectrum.frequencies) > 0
```

### Fixtures

```python
@pytest.fixture
def sine_trace():
    """Generate a 1 MHz sine wave trace."""
    from oscura.testing import generate_sine_wave
    return generate_sine_wave(frequency=1e6, sample_rate=100e6)

@pytest.fixture(scope="session")
def large_trace(tmp_path_factory):
    """Generate large trace once per session."""
    path = tmp_path_factory.mktemp("data") / "large.npz"
    trace = generate_large_trace()
    np.savez(path, data=trace.data)
    return path

def test_with_sine(sine_trace):
    """Test using sine_trace fixture."""
    freq = osc.measure_frequency(sine_trace)
    assert abs(freq - 1e6) < 1000
```

### Parametrized Tests

```python
@pytest.mark.parametrize("baud_rate", [9600, 19200, 115200, 921600])
def test_uart_baud_rates(baud_rate):
    """Test UART decoder with various baud rates."""
    trace = generate_uart_signal(baud_rate=baud_rate)
    decoder = UARTDecoder(baud_rate=baud_rate)
    messages = decoder.decode(trace)
    assert len(messages) > 0

@pytest.mark.parametrize("low,high", [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
])
def test_rise_time_thresholds(low, high):
    """Test rise time with various thresholds."""
    trace = generate_pulse()
    rise = osc.measure_rise_time(trace, low=low, high=high)
    assert not np.isnan(rise)
```

## CRITICAL: Anti-Patterns to Avoid

### NEVER Use `pytest.main()` in Test Files

**This will cause terminal crashes due to fork bomb:**

```python
# DO NOT DO THIS - WILL CRASH YOUR TERMINAL
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Why it's dangerous**:

- Creates recursive pytest invocations
- Spawns exponentially growing processes
- Exhausts system resources
- Crashes terminal/system

**Correct approach**:

```bash
# Run tests from command line
pytest tests/unit/test_example.py -v
```

Or use VS Code debug configurations (F5).

## Memory Management

### Prevent Memory Leaks

```python
@pytest.fixture(autouse=True, scope="function")
def memory_cleanup():
    """Force garbage collection after each test."""
    yield
    import gc
    gc.collect()

@pytest.mark.memory_intensive
@pytest.mark.limit_memory("100 MB")
def test_large_dataset():
    """Test with large data."""
    data = generate_large_dataset()
    result = process(data)
    del data  # Explicit cleanup
    assert result is not None
```

### Test Timeouts

```python
# Default timeout: 60 seconds (in pyproject.toml)

@pytest.mark.timeout(120)
def test_slow_operation():
    """Test needing extended timeout."""
    pass
```

## Debugging Tests

### Verbose Output

```bash
# Show full output
pytest tests/unit/test_file.py -v

# Show print statements
pytest tests/unit/test_file.py -s

# Show local variables on failure
pytest tests/unit/test_file.py -l

# Drop into debugger on failure
pytest tests/unit/test_file.py --pdb

# Show why tests were skipped
pytest tests/unit/test_file.py -rs

# Show all test outcomes
pytest tests/unit/test_file.py -ra
```

### Using pdb

```python
def test_debug():
    """Test with debugger."""
    result = function()
    import pdb; pdb.set_trace()  # Breakpoint
    assert result == expected
```

Run with:

```bash
pytest tests/unit/test_file.py -s  # -s to see pdb output
```

## Coverage Requirements

- **Minimum**: 70% (enforced by Codecov)
- **Target**: 80%+
- **Critical modules**: 90%+ required
- **PR diff coverage**: 80% (enforced by diff-cover)

## Common Issues

### Tests Hang

**Solution**: Run with timeout

```bash
pytest tests/unit -v --timeout=10
```

### Memory Errors

**Solution**: Use smaller datasets or profile memory

```bash
# Profile memory
pytest --memray --most-allocations=10

# Use smaller test data
@pytest.fixture
def small_data():
    return np.random.randn(100)  # Not 10_000_000
```

### Import Errors

**Solution**: Use uv

```bash
uv run pytest tests/unit
```

### Parallel Test Failures

**Solution**: Check isolation and use fixtures

```bash
# Check isolation
python scripts/testing/check_test_isolation.py --sample 15

# Use fixtures for isolation
@pytest.fixture
def isolated_data(tmp_path):
    """Each test gets its own data."""
    return tmp_path / "data.txt"
```

### Testmon Not Detecting Changes

**Solution**: Clear cache and rebuild

```bash
pytest --testmon-nocache
```

### Randomized Test Failures

**Solution**: Reproduce with same seed

```bash
# Tests failed with seed 12345
pytest --randomly-seed=12345

# Or use last seed
pytest --randomly-seed=last

# Disable randomization for debugging
pytest --randomly-dont-shuffle
```

## Checklist Before Committing

- [ ] No `pytest.main()` calls in test files
- [ ] Tests pass: `pytest tests/unit -x`
- [ ] Tests pass with randomized order: `pytest --randomly-seed=auto`
- [ ] Test isolation verified: `python scripts/testing/check_test_isolation.py --sample 5`
- [ ] Coverage maintained: `pytest --cov=src/oscura`
- [ ] Markers validated: `python scripts/testing/validate_test_markers.py --strict`
- [ ] No infinite loops or timeouts
- [ ] Memory-intensive tests marked with `@pytest.mark.memory_intensive`
- [ ] Slow tests marked with `@pytest.mark.slow`
- [ ] Tests are isolated (no shared state)
- [ ] Docstrings explain what's being tested

## See Also

- [OOM Prevention Guide](./oom-prevention.md) - Getting started with testing
- [OOM Prevention](./oom-prevention.md) - Preventing out-of-memory issues
- [Hypothesis Testing Guide](https://github.com/oscura-re/oscura/blob/main/tests/HYPOTHESIS_TESTING_GUIDE.md) - Property-based testing
- [Contributing Guide](https://github.com/oscura-re/oscura/blob/main/CONTRIBUTING.md) - Contribution guidelines
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-testmon Documentation](https://testmon.org/)
- [pytest-randomly Documentation](https://github.com/pytest-dev/pytest-randomly)
- [pytest-memray Documentation](https://pytest-memray.readthedocs.io/)

---

**Need help?** See the [OOM Prevention Guide](./oom-prevention.md) for a beginner-friendly introduction.
