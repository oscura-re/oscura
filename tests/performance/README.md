# Oscura Performance Testing

This directory contains performance benchmarks using pytest-benchmark for the Oscura signal analysis framework.

## Quick Start

```bash
# Run all benchmarks
uv run pytest tests/performance/test_benchmarks.py --benchmark-only

# Save results and compare with baseline
uv run pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-json=results.json

uv run python scripts/compare_benchmarks.py \
  tests/performance/baseline_results.json \
  results.json
```

## Migration Notice

**NEW (pytest-benchmark)**: Performance tests are now integrated with pytest
**LEGACY**: `comprehensive_profiling.py` and `visualize_results.py` are deprecated

Use pytest-benchmark for all new performance testing.

## Directory Structure

```
tests/performance/
├── test_benchmarks.py          # Main pytest-benchmark test suite
├── conftest.py                 # Benchmark fixtures
├── baseline_results.json       # Baseline for regression detection
├── comprehensive_profiling.py  # DEPRECATED: Legacy profiling script
├── visualize_results.py        # DEPRECATED: Legacy visualization
├── results/                    # Legacy profiling results
└── test_data/                  # Legacy test datasets
```

## Benchmark Test Suite

The main test suite (`test_benchmarks.py`) includes:

- **LoaderBenchmarks**: Binary file loading at various sizes
- **AnalyzerBenchmarks**: Edge detection, FFT, statistics, moving average
- **InferenceBenchmarks**: Message format inference, state machine learning
- **MemoryBenchmarks**: Memory overhead and efficiency
- **ComplexityBenchmarks**: Algorithm O(n) behavior
- **ScalabilityBenchmarks**: Concurrent operations

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
uv run pytest tests/performance/test_benchmarks.py --benchmark-only

# Run specific benchmark class
uv run pytest tests/performance/test_benchmarks.py::TestLoaderBenchmarks --benchmark-only

# Run with verbose output
uv run pytest tests/performance/test_benchmarks.py --benchmark-only -v
```

### Advanced Options

```bash
# Save results to JSON
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-json=results.json

# Custom rounds and warmup
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-min-rounds=10 \
  --benchmark-warmup=on

# Sort by different metrics
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-sort=min  # or max, mean, stddev

# Generate histogram
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-histogram=histogram
```

### Filtering Tests

```bash
# Exclude slow tests
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  -m "not slow"

# Run only memory benchmarks
pytest tests/performance/test_benchmarks.py::TestMemoryBenchmarks \
  --benchmark-only

# Run specific parametrized size
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  -k "signal_length-1000"
```

## CI Integration

Benchmarks run automatically in CI via `.github/workflows/benchmarks.yml`:

- **Pull Requests**: Run on changes to `src/oscura/**` or `tests/performance/**`
- **Main Branch**: Update baseline on every push
- **Weekly Schedule**: Full matrix across OS and Python versions
- **Manual**: Workflow dispatch available

### Regression Detection

CI automatically:

1. Downloads baseline from main branch
2. Runs current benchmarks
3. Compares results using `scripts/compare_benchmarks.py`
4. Posts comparison as PR comment
5. Fails if regressions >20% detected

## Comparison and Regression Testing

### Using the Comparison Script

```bash
# Compare two benchmark runs
uv run python scripts/compare_benchmarks.py baseline.json current.json

# Custom threshold (default: 20%)
uv run python scripts/compare_benchmarks.py baseline.json current.json --threshold 15

# JSON output for automation
uv run python scripts/compare_benchmarks.py baseline.json current.json --json
```

### Interpreting Results

- **Regressions**: Tests >threshold% slower (⚠️)
- **Improvements**: Tests >5% faster (✅)
- **No change**: Tests within ±5% (~)

**Exit codes**:

- `0`: No significant regressions
- `1`: Regressions >threshold detected

## Adding New Benchmarks

### Basic Benchmark

```python
import pytest
from oscura.analyzers import my_analyzer

pytestmark = [pytest.mark.performance, pytest.mark.benchmark]

def test_my_analyzer_performance(benchmark):
    """Benchmark my_analyzer function."""
    data = generate_test_data()
    result = benchmark(my_analyzer, data)
    assert result is not None
```

### Parametrized Benchmark

```python
@pytest.mark.parametrize("size", [1000, 10000, 100000])
def test_scalability(benchmark, size):
    """Test performance at different scales."""
    data = np.random.randn(size)
    result = benchmark(my_function, data)
    assert result is not None
```

### Memory-Aware Benchmark

```python
def test_memory_efficiency(benchmark, memory_monitor):
    """Benchmark with memory tracking."""
    with memory_monitor() as monitor:
        result = benchmark(my_function, large_data)
    assert monitor.peak_mb < 500  # Max 500MB
```

## Performance Thresholds

### Configuration (pyproject.toml)

```toml
benchmark_min_rounds = 5
benchmark_max_time = 1.0
benchmark_warmup = true
benchmark_disable_gc = true
```

### Regression Thresholds

- **5%**: Noise threshold (changes <5% ignored)
- **20%**: Default regression threshold (CI fails)
- **Custom**: Override with `--threshold` flag

### Memory Thresholds (conftest.py)

- `load_file`: 100 MB
- `analyze_signal`: 200 MB
- `infer_protocol`: 500 MB

## Best Practices

### Writing Benchmarks

1. **Mark tests**: Use `@pytest.mark.benchmark` and `@pytest.mark.performance`
2. **Add slow marker**: Use `@pytest.mark.slow` for tests >1 second
3. **Parametrize sizes**: Test scalability with multiple input sizes
4. **Stable data**: Use fixed seeds for reproducible results
5. **Include assertions**: Basic correctness checks

### Running Benchmarks

1. **Use `--benchmark-only`**: Skip regular tests
2. **Filter with `-k`**: Run specific benchmarks
3. **Stable environment**: Close other applications
4. **Compare with baseline**: Always check for regressions
5. **Full suite before commit**: Ensure no regressions

### CI/CD

1. **Automatic baseline**: Let CI update baseline on main
2. **Review regressions**: Check comparison in PR comments
3. **Don't ignore failures**: Investigate benchmark failures
4. **Weekly matrix**: Monitor cross-platform performance

## Troubleshooting

### Benchmarks Too Slow

```bash
pytest --benchmark-only \
  --benchmark-min-rounds=3 \
  --benchmark-max-time=0.5
```

### Unstable Results

```bash
pytest --benchmark-only \
  --benchmark-min-rounds=10 \
  --benchmark-disable-gc
```

### Missing Baseline

```bash
pytest tests/performance/test_benchmarks.py \
  --benchmark-only \
  --benchmark-json=tests/performance/baseline_results.json
```

## Migration from Legacy Scripts

**NEW**: `test_benchmarks.py` (pytest-benchmark)
**LEGACY**: `comprehensive_profiling.py`, `visualize_results.py` (deprecated)

**Advantages**:

- Integrated with pytest test suite
- Automatic regression detection
- CI integration
- Better reporting

## Further Reading

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [Oscura Testing Guide](../../docs/testing/test-suite-guide.md)
- [Test Suite Overview](../README.md)
