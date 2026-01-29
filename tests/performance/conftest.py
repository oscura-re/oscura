"""Performance test fixtures.

This module provides fixtures for performance and benchmark tests:
- Timing utilities
- Performance threshold fixtures
- Large dataset generators
- Resource monitoring fixtures
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# =============================================================================
# Timing Utilities
# =============================================================================


@pytest.fixture
def benchmark_timer():
    """Simple timing fixture for benchmarks.

    Returns:
        Context manager that measures elapsed time.

    Example:
        >>> with benchmark_timer() as timer:
        ...     # code to benchmark
        ...     pass
        >>> print(f"Elapsed: {timer.elapsed:.3f}s")
    """

    class Timer:
        """Context manager for timing code execution."""

        def __init__(self):
            self.start: float = 0.0
            self.end: float = 0.0
            self.elapsed: float = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            self.elapsed = self.end - self.start

    return Timer


@pytest.fixture
def measure_time():
    """Factory for measuring function execution time.

    Returns:
        Function that measures execution time and returns result + time.

    Example:
        >>> result, elapsed = measure_time(my_function, arg1, arg2)
    """

    def _measure(func, *args, **kwargs):
        """Measure function execution time.

        Args:
            func: Function to measure.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Tuple of (result, elapsed_time).
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return _measure


# =============================================================================
# Performance Thresholds
# =============================================================================


@pytest.fixture
def performance_thresholds() -> dict[str, float]:
    """Performance threshold configurations.

    Returns:
        Dictionary mapping operation to max time in seconds.
    """
    return {
        "load_file": 1.0,  # Load a typical file
        "load_large_file": 5.0,  # Load >100 MB file
        "analyze_signal": 2.0,  # Analyze 1M samples
        "infer_protocol": 5.0,  # Infer protocol from 1000 messages
        "render_plot": 1.0,  # Render visualization
        "save_figure": 2.0,  # Save figure to disk
        "export_data": 3.0,  # Export processed data
    }


@pytest.fixture
def memory_thresholds() -> dict[str, float]:
    """Memory usage thresholds.

    Returns:
        Dictionary mapping operation to max memory in MB.
    """
    return {
        "load_file": 100,  # Max memory for file loading
        "analyze_signal": 200,  # Max memory for signal analysis
        "infer_protocol": 500,  # Max memory for protocol inference
        "render_plot": 150,  # Max memory for plotting
        "total_test": 1000,  # Max total memory per test
    }


# =============================================================================
# Large Dataset Generators
# =============================================================================
# NOTE: large_signal and very_large_signal fixtures are available from root conftest.py


@pytest.fixture
def large_binary_data(scope="session") -> bytes:
    """Generate large binary data for loader performance testing (10 MB).

    Returns:
        10 MB of random binary data.
    """
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=10_000_000, dtype=np.uint8)
    return data.tobytes()


@pytest.fixture
def large_message_set(scope="session") -> list[bytes]:
    """Generate large set of protocol messages for inference testing.

    Returns:
        List of 10,000 protocol messages.
    """
    messages = []
    for i in range(10_000):
        # Variable-length messages
        payload_len = (i % 200) + 10
        header = b"\xaa\x55"
        seq = i.to_bytes(2, "big")
        length = payload_len.to_bytes(1, "big")
        payload = bytes([(i + j) % 256 for j in range(payload_len)])
        checksum = (sum(payload) & 0xFF).to_bytes(1, "big")
        messages.append(header + seq + length + payload + checksum)
    return messages


# =============================================================================
# Profiling Fixtures
# =============================================================================


@pytest.fixture
def profile_config() -> dict[str, Any]:
    """Configuration for profiling tests.

    Returns:
        Dictionary with profiling settings.
    """
    return {
        "enable_cprofile": True,
        "enable_memory_profiler": True,
        "enable_line_profiler": False,  # Requires line_profiler package
        "output_dir": "performance/results",
        "save_stats": True,
    }


# =============================================================================
# Iteration Counts
# =============================================================================


@pytest.fixture
def benchmark_iterations() -> dict[str, int]:
    """Number of iterations for benchmark tests.

    Returns:
        Dictionary mapping test type to iteration count.
    """
    return {
        "quick": 10,  # Quick smoke test
        "normal": 100,  # Normal benchmarking
        "thorough": 1000,  # Thorough benchmarking
        "stress": 10000,  # Stress testing
    }


# =============================================================================
# Resource Monitoring
# =============================================================================


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during test execution.

    Returns:
        Context manager that tracks peak memory usage.

    Example:
        >>> with memory_monitor() as monitor:
        ...     # code to monitor
        ...     pass
        >>> print(f"Peak memory: {monitor.peak_mb:.2f} MB")
    """
    import tracemalloc

    class MemoryMonitor:
        """Context manager for memory monitoring."""

        def __init__(self):
            self.start_memory: int = 0
            self.peak_memory: int = 0
            self.current_memory: int = 0

        @property
        def peak_mb(self) -> float:
            """Peak memory usage in MB."""
            return self.peak_memory / (1024 * 1024)

        @property
        def current_mb(self) -> float:
            """Current memory usage in MB."""
            return self.current_memory / (1024 * 1024)

        def __enter__(self):
            tracemalloc.start()
            self.start_memory = tracemalloc.get_traced_memory()[0]
            return self

        def __exit__(self, *args):
            self.current_memory, self.peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

    return MemoryMonitor


# =============================================================================
# Comparison Utilities
# =============================================================================


@pytest.fixture
def performance_comparison():
    """Compare performance of multiple implementations.

    Returns:
        Function that runs and compares multiple implementations.

    Example:
        >>> results = performance_comparison(
        ...     {'impl1': func1, 'impl2': func2},
        ...     args=(data,),
        ...     iterations=100
        ... )
    """

    def _compare(
        implementations: dict[str, callable],
        args: tuple = (),
        kwargs: dict | None = None,
        iterations: int = 10,
    ) -> dict[str, dict[str, float]]:
        """Compare performance of multiple implementations.

        Args:
            implementations: Dict mapping name to function.
            args: Positional arguments for functions.
            kwargs: Keyword arguments for functions.
            iterations: Number of iterations per implementation.

        Returns:
            Dict mapping name to performance metrics.
        """
        if kwargs is None:
            kwargs = {}

        results = {}
        for name, func in implementations.items():
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "median": np.median(times),
            }

        return results

    return _compare


# =============================================================================
# Scalability Testing
# =============================================================================


@pytest.fixture
def scalability_test_sizes() -> list[int]:
    """Test data sizes for scalability testing.

    Returns:
        List of data sizes from small to large.
    """
    return [
        100,  # Tiny
        1_000,  # Small
        10_000,  # Medium
        100_000,  # Large
        1_000_000,  # Very large
    ]


@pytest.fixture
def scalability_data_generator():
    """Generate data at various scales for scalability testing.

    Returns:
        Function that generates data of specified size.
    """

    def _generate(size: int, data_type: str = "signal") -> Any:
        """Generate test data of specified size.

        Args:
            size: Number of samples.
            data_type: Type of data ('signal', 'binary', 'messages').

        Returns:
            Generated data.
        """
        rng = np.random.default_rng(42)

        if data_type == "signal":
            return rng.standard_normal(size)
        elif data_type == "binary":
            return rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        elif data_type == "messages":
            messages = []
            for i in range(size):
                payload_len = (i % 50) + 10
                msg = b"\xaa\x55" + bytes([payload_len]) + bytes(range(payload_len))
                messages.append(msg)
            return messages
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    return _generate


# =============================================================================
# Regression Testing
# =============================================================================


@pytest.fixture
def performance_baseline() -> dict[str, float]:
    """Baseline performance metrics for regression testing.

    Returns:
        Dictionary with baseline times in seconds.
    """
    return {
        "load_wfm_file": 0.5,
        "load_pcap_file": 0.3,
        "edge_detection_1M_samples": 0.2,
        "fft_1M_samples": 0.1,
        "protocol_inference_1k_messages": 2.0,
        "render_waveform_plot": 0.5,
    }


# =============================================================================
# pytest-benchmark Fixtures
# =============================================================================


@pytest.fixture
def benchmark_config() -> dict[str, Any]:
    """Configuration for pytest-benchmark tests.

    Returns:
        Dictionary with benchmark configuration.
    """
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "min_time": 0.000005,
        "warmup": True,
        "disable_gc": True,
        "timer": "perf_counter",
    }


@pytest.fixture
def benchmark_baseline():
    """Load baseline benchmark results for comparison.

    Returns:
        Dictionary with baseline benchmark results, or None if not found.
    """
    baseline_file = Path(__file__).parent / "baseline_results.json"
    if baseline_file.exists():
        import json

        with open(baseline_file) as f:
            return json.load(f)
    return None


@pytest.fixture
def assert_benchmark_regression(benchmark_baseline):
    """Assert that benchmark does not regress beyond threshold.

    Returns:
        Function that checks benchmark against baseline.

    Example:
        >>> def test_my_func(benchmark, assert_benchmark_regression):
        ...     result = benchmark(my_func, arg)
        ...     assert_benchmark_regression("test_my_func", result, threshold=1.2)
    """

    def _assert_regression(test_name: str, result: Any, threshold: float = 1.2):
        """Check if benchmark regressed.

        Args:
            test_name: Name of the test.
            result: Benchmark result object.
            threshold: Maximum allowed slowdown factor (e.g., 1.2 = 20% slower).

        Raises:
            AssertionError: If regression detected.
        """
        if not benchmark_baseline:
            # No baseline available, skip check
            return

        benchmarks = {b["name"]: b for b in benchmark_baseline.get("benchmarks", [])}
        if test_name not in benchmarks:
            # Test not in baseline, skip check
            return

        baseline_mean = benchmarks[test_name]["stats"]["mean"]
        current_mean = result.stats["mean"]

        if current_mean > baseline_mean * threshold:
            slowdown = (current_mean / baseline_mean - 1) * 100
            raise AssertionError(
                f"Performance regression detected: {test_name} is {slowdown:.1f}% slower "
                f"(baseline: {baseline_mean:.6f}s, current: {current_mean:.6f}s, "
                f"threshold: {threshold:.0%})"
            )

    return _assert_regression


@pytest.fixture
def regression_tolerance() -> float:
    """Acceptable performance regression tolerance.

    Returns:
        Maximum acceptable slowdown factor (e.g., 1.1 = 10% slower).
    """
    return 1.1  # Allow 10% regression


# =============================================================================
# Warmup Utilities
# =============================================================================


@pytest.fixture
def warmup_function():
    """Warm up function by running it several times.

    Returns:
        Function that warms up another function.
    """

    def _warmup(func, args: tuple = (), kwargs: dict | None = None, iterations: int = 3):
        """Warm up function to eliminate cold-start effects.

        Args:
            func: Function to warm up.
            args: Positional arguments.
            kwargs: Keyword arguments.
            iterations: Number of warmup iterations.
        """
        if kwargs is None:
            kwargs = {}

        for _ in range(iterations):
            func(*args, **kwargs)

    return _warmup


# =============================================================================
# Statistical Analysis
# =============================================================================


@pytest.fixture
def performance_statistics():
    """Calculate statistical measures from timing results.

    Returns:
        Function that computes statistics from timing data.
    """

    def _stats(times: list[float]) -> dict[str, float]:
        """Calculate statistics from timing measurements.

        Args:
            times: List of timing measurements.

        Returns:
            Dictionary with statistical measures.
        """
        times_array = np.array(times)
        return {
            "mean": np.mean(times_array),
            "std": np.std(times_array),
            "min": np.min(times_array),
            "max": np.max(times_array),
            "median": np.median(times_array),
            "p95": np.percentile(times_array, 95),
            "p99": np.percentile(times_array, 99),
            "cv": np.std(times_array) / np.mean(times_array),  # Coefficient of variation
        }

    return _stats
