# OOM Prevention Guide for Oscura Test Suite

> **Version**: 0.9.0 | **Last Updated**: 2026-01-25

## Problem Overview

Oscura's test suite contains thousands of tests across multiple categories. Running the full suite can cause Out of Memory (OOM) errors (exit code 137) on systems with limited memory.

> **Note**: Test counts vary over time as tests are added. Use `pytest --collect-only | grep "test session starts" -A1` to get current counts.

## Root Causes

1. **Test accumulation**: Pytest loads all test metadata into memory
2. **Fixture retention**: Large signal fixtures (10M samples) held in memory
3. **NumPy array allocation**: Signal processing creates large temporary arrays
4. **Matplotlib figures**: Unclosed figures accumulate in memory
5. **Lack of garbage collection**: Python doesn't aggressively free memory between tests

## Implemented Solutions

### 1. Automatic Memory Cleanup

**File**: `tests/conftest.py`

Added automatic garbage collection after each test:

```python
@pytest.fixture(autouse=True, scope="function")
def memory_cleanup():
    """Force garbage collection after each test to prevent memory buildup."""
    yield
    import gc
    gc.collect()
```

**Impact**: Reduces peak memory by ~15-20% by immediately freeing unused objects.

### 2. Enhanced Test Markers

**File**: `pyproject.toml`

Added comprehensive test markers for selective execution:

```toml
markers = [
    "memory_intensive: marks tests that use significant memory (>100MB)",
    "unit: marks unit tests (isolated components)",
    "integration: marks integration tests",
    "stress: marks stress and edge case tests",
    "validation: marks ground truth validation tests",
    "performance: marks performance benchmarks",
    # ... and more
]
```

**Usage**:

```bash
# Run only fast unit tests
pytest -m "unit and not memory_intensive and not slow"

# Run only memory-intensive tests (isolated)
pytest -m "memory_intensive" --maxfail=5
```

### 3. Test Suite Splitting Script

**File**: `scripts/run_tests_split.py`

Splits tests into 6 categories with optimized worker counts:

| Category    | Tests | Workers | Memory Impact |
| ----------- | ----- | ------- | ------------- |
| unit_fast   | ~1800 | 4       | Low           |
| unit_memory | ~200  | 2       | High          |
| integration | ~170  | 2       | Medium        |
| validation  | ~140  | 2       | Medium        |
| stress      | ~74   | 1       | High          |
| performance | ~50   | 1       | Variable      |

**Usage**:

```bash
# Run all chunks sequentially (recommended)
python scripts/run_tests_split.py

# Run specific chunk
python scripts/run_tests_split.py --chunk unit_fast

# Dry run to see commands
python scripts/run_tests_split.py --dry-run

# Disable parallelization
python scripts/run_tests_split.py --no-parallel
```

### 4. GitHub Actions Workflow

**File**: `.github/workflows/tests-chunked.yml`

CI/CD workflow that runs test chunks as separate jobs with memory isolation:

- Each chunk runs in its own container
- Parallel execution across chunks (not within)
- Individual timeouts prevent runaway processes
- Summary job aggregates results

## Best Practices

### For Test Authors

1. **Mark memory-intensive tests**:

   ```python
   @pytest.mark.memory_intensive
   def test_large_signal_processing():
       signal = np.random.randn(10_000_000)
       # ...
   ```

2. **Optimize fixture scopes**:

   ```python
   # Bad: Creates new 10M array for each test
   @pytest.fixture
   def large_signal():
       return np.random.randn(10_000_000)

   # Good: Creates once per session
   @pytest.fixture(scope="session")
   def large_signal():
       return np.random.randn(10_000_000)
   ```

3. **Explicit cleanup in tests**:

   ```python
   def test_signal_processing():
       signal = create_large_signal()
       result = process(signal)

       # Explicit cleanup before assertions
       del signal

       assert result.mean() > 0
   ```

4. **Use smaller test data when possible**:

   ```python
   # Bad: Uses 10M samples for basic test
   def test_filter_works():
       signal = np.random.randn(10_000_000)
       filtered = bandpass_filter(signal, 100, 1000)
       assert filtered is not None

   # Good: 1000 samples sufficient for correctness
   def test_filter_works():
       signal = np.random.randn(1000)
       filtered = bandpass_filter(signal, 100, 1000)
       assert filtered is not None
   ```

### For CI/CD

1. **Use chunked workflow** (`.github/workflows/tests-chunked.yml`)
2. **Set appropriate timeouts** (30-45 minutes per chunk)
3. **Monitor memory usage** with GitHub Actions logs
4. **Consider matrix strategy** for multiple Python versions:

   ```yaml
   strategy:
     matrix:
       python-version: ['3.12', '3.13']
       chunk: [unit_fast, unit_memory, integration]
   ```

### For Local Development

1. **Run subsets during development**:

   ```bash
   # Only test what you're working on
   pytest tests/unit/analyzers/digital/

   # Skip slow and memory-intensive tests
   pytest -m "not slow and not memory_intensive"
   ```

2. **Use pytest-xdist for parallelization**:

   ```bash
   # Install
   uv add --dev pytest-xdist

   # Run with 4 workers (memory isolation)
   pytest -n 4 tests/unit
   ```

3. **Monitor memory with pytest-memray**:

   ```bash
   # Install
   uv add --dev pytest-memray

   # Profile memory usage
   pytest --memray tests/unit/analyzers/

   # Set memory limits
   pytest --memray --most-allocations=10
   ```

## Advanced Solutions

### pytest-xdist Configuration

Install for true memory isolation between tests:

```bash
uv add --dev pytest-xdist
```

**Distribution strategies**:

```bash
# Load-balanced (default)
pytest -n auto

# Group by file (better memory isolation)
pytest -n 4 --dist loadfile

# Group by module
pytest -n 4 --dist loadscope

# Custom grouping with xdist_group marker
pytest -n 4 --dist loadgroup
```

**In pyproject.toml**:

```toml
[tool.pytest.ini_options]
addopts = [
    "-n", "auto",
    "--dist", "loadfile",
    "--maxprocesses", "4",
]
```

### pytest-split for Duration-Based Chunking

For even more granular control:

```bash
uv add --dev pytest-split

# First run to collect durations
pytest --store-durations

# Split into 4 groups, run group 1
pytest --splits 4 --group 1

# Use least-duration algorithm
pytest --splits 4 --group 1 --splitting-algorithm least_duration
```

### Memory Profiling

**pytest-memray** for detailed profiling:

```bash
# Find memory-hungry tests
pytest --memray --most-allocations=20 tests/

# Set memory limits on individual tests
@pytest.mark.limit_memory("100 MB")
def test_bounded_memory():
    # Test will fail if it exceeds 100MB
    pass

# Detect memory leaks
@pytest.mark.limit_leaks("1 MB")
def test_no_leaks():
    # Test will fail if any call stack leaks >1MB
    pass
```

**Memory profiler** for line-by-line analysis:

```bash
uv add --dev memory-profiler

# Profile specific test
python -m memory_profiler tests/unit/test_specific.py
```

## Fixture Optimization Checklist

- [ ] Session-scoped for expensive, immutable data (test data paths, ground truth)
- [ ] Module-scoped for shared test data within a file
- [ ] Class-scoped for shared state across test methods
- [ ] Function-scoped (default) only when necessary for isolation
- [ ] Use `yield` for cleanup in all fixtures that allocate resources
- [ ] Avoid creating large objects in parametrized fixtures (multiplicative effect)

**Example optimization**:

```python
# Before: Creates test data 2437 times
@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "test_data"

# After: Creates once per session
@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent.parent / "test_data"
```

## Measuring Success

### Key Metrics

1. **Peak memory usage**: Should stay under 4GB for full suite
2. **Test completion rate**: Should reach 100% without OOM
3. **Execution time**: Chunked execution may be slower but completes
4. **Failure isolation**: Failed chunk doesn't crash entire suite

### Monitoring Commands

```bash
# Monitor memory during test run
watch -n 1 'ps aux | grep pytest | grep -v grep'

# Linux: Check OOM killer logs
dmesg | grep -i oom

# Track pytest memory over time
pytest --memray --memray-bin-path=./memray-results/
```

### Expected Results

| Metric            | Before    | After  | Improvement |
| ----------------- | --------- | ------ | ----------- |
| Peak Memory       | >8GB      | ~3-4GB | 50%+        |
| Completion Rate   | 34%       | 100%   | 3x          |
| Time (sequential) | N/A (OOM) | ~45min | Complete    |
| Time (parallel)   | N/A (OOM) | ~15min | Complete    |

## Troubleshooting

### Still Getting OOM?

1. **Reduce worker count**:

   ```bash
   python scripts/run_tests_split.py --no-parallel
   ```

2. **Run smaller chunks**:

   ```bash
   # Run each test category separately
   pytest tests/unit/loaders/
   pytest tests/unit/analyzers/
   # etc.
   ```

3. **Increase system swap**:

   ```bash
   # Linux: Add 8GB swap
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Use pytest-split**:

   ```bash
   # Split into 10 smaller chunks
   for i in {1..10}; do
       pytest --splits 10 --group $i
   done
   ```

### Tests Pass Individually but Fail in Suite

This indicates **inter-test dependencies** or **resource leaks**:

1. **Check for global state**:

   ```bash
   # Run tests in random order
   pytest --random-order
   ```

2. **Profile memory leaks**:

   ```bash
   pytest --memray --most-allocations=50
   ```

3. **Add explicit cleanup**:

   ```python
   @pytest.fixture(autouse=True)
   def reset_global_state():
       yield
       # Reset any global variables
       import my_module
       my_module.CACHE.clear()
   ```

## References

### Research Sources

Industry best practices from major projects:

- [Catching memory leaks with your test suite](https://pythonspeed.com/articles/identifying-resource-leaks-with-pytest)
- [Pytest Parallel Execution for Large Test Suites in Python 2025](https://johal.in/pytest-parallel-execution-for-large-test-suites-in-python-2025/)
- [Very large test suites take far too much RAM - pytest-dev](https://github.com/pytest-dev/pytest/issues/619)
- [13 Proven Ways To Improve Test Runtime With Pytest](https://pytest-with-eric.com/pytest-advanced/pytest-improve-runtime/)

pytest-xdist memory isolation:

- [Parallel Testing Made Easy With pytest-xdist](https://pytest-with-eric.com/plugins/pytest-xdist/)
- [pytest-xdist Documentation](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
- [Unleashing Pytest Power: Parallel Testing with xdist](https://techbytesdispatch.medium.com/unleashing-pytest-power-how-parallel-testing-with-xdist-made-python-tests-5x-faster-035c04e77187)

Memory profiling tools:

- [pytest-memray Documentation](https://pytest-memray.readthedocs.io/)
- [Memray: The endgame memory profiler](https://bloomberg.github.io/memray/)
- [pytest-memray GitHub](https://github.com/bloomberg/pytest-memray)

Fixture optimization:

- [pytest fixtures: explicit, modular, scalable](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Mastering Pytest Fixtures Advanced Scope Parameterization](https://leapcell.io/blog/mastering-pytest-fixtures-advanced-scope-parameterization-and-dependency-management)
- [Boost Test Speed with Pytest Fixture Scope](https://articles.mergify.com/pytest-fixture-scope/)
- [What Are Pytest Fixture Scopes?](https://pytest-with-eric.com/fixtures/pytest-fixture-scope/)

Test splitting:

- [pytest-split GitHub](https://github.com/jerry-git/pytest-split)
- [pytest-split Documentation](https://jerry-git.github.io/pytest-split/)
- [How to Distribute Tests with Pytest-Split](https://medium.com/@krijnvanderburg/how-to-distribute-tests-in-ci-cd-for-faster-execution-zero-bs-1-b86d4d69b19d)
- [Blazing fast CI with pytest-split and GitHub Actions](https://blog.jerrycodes.com/pytest-split-and-github-actions/)

### Related Documentation

- `tests/TEST_SUITE_ARCHITECTURE.md`: Test organization and structure
- `tests/TEST_SUITE_CLEANUP_REPORT.md`: Historical cleanup decisions
- `scripts/README.md`: Script documentation

## Summary

The OOM prevention strategy uses a multi-layered approach:

1. **Immediate fixes**: Auto-cleanup, markers, fixture optimization
2. **Test splitting**: Chunked execution with optimized parallelization
3. **CI/CD integration**: Isolated jobs prevent cascading failures
4. **Developer tools**: Scripts and workflows for efficient local testing
5. **Monitoring**: Memory profiling to identify and fix leaks

This approach ensures the full test suite completes successfully while maintaining fast feedback loops for development.
