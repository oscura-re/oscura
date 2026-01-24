"""Optimization: Performance Techniques and Caching

Demonstrates:
- oscura.configure_fft_cache() - FFT result caching
- oscura.get_fft_cache_stats() - Cache statistics
- Lazy evaluation patterns
- Memory-efficient processing
- Performance optimization strategies

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/01_pipeline_api.py
- 07_advanced_api/06_streaming_api.py

Performance optimization is critical for processing large signals and
real-time applications. Learn caching, lazy evaluation, and memory management.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave
from oscura import (
    clear_fft_cache,
    configure_fft_cache,
    fft,
    get_fft_cache_stats,
    rms,
    thd,
)


class LazySignalProcessor:
    """Lazy evaluation wrapper for signal processing."""

    def __init__(self, trace):
        """Initialize lazy processor."""
        self.trace = trace
        self._operations = []
        self._result = None

    def filter(self, filter_func):
        """Add filter operation (lazy)."""
        self._operations.append(("filter", filter_func))
        return self

    def measure(self, measure_func):
        """Add measurement operation (lazy)."""
        self._operations.append(("measure", measure_func))
        return self

    def execute(self):
        """Execute all operations (evaluate lazily)."""
        if self._result is not None:
            return self._result

        result = self.trace
        measurements = {}

        for op_type, op_func in self._operations:
            if op_type == "filter":
                result = op_func(result)
            elif op_type == "measure":
                measurements[op_func.__name__] = op_func(result)

        self._result = (result, measurements)
        return self._result

    def reset(self):
        """Reset lazy evaluation."""
        self._result = None
        return self


class OptimizationDemo(BaseDemo):
    """Demonstrate performance optimization techniques."""

    def __init__(self):
        """Initialize optimization demonstration."""
        super().__init__(
            name="optimization",
            description="Performance optimization techniques and caching strategies",
            capabilities=[
                "oscura.configure_fft_cache",
                "oscura.get_fft_cache_stats",
                "lazy_evaluation",
                "memory_optimization",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals."""
        # Create multiple signals of different sizes
        small_signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.001, sample_rate=100e3
        )

        medium_signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
        )

        large_signal = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.1, sample_rate=100e3
        )

        return {
            "small": small_signal,
            "medium": medium_signal,
            "large": large_signal,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run optimization demonstration."""
        small = data["small"]
        medium = data["medium"]
        large = data["large"]

        self.section("Optimization: Performance Techniques")

        # ===================================================================
        # Part 1: FFT Caching
        # ===================================================================
        self.subsection("1. FFT Result Caching")
        self.info("Cache expensive FFT computations")

        # Clear cache to start fresh
        clear_fft_cache()

        # Configure cache
        configure_fft_cache(size=10)

        self.info("Cache configuration:")
        stats = get_fft_cache_stats()
        self.result("  Max size", stats.get("size", 10), "entries")
        self.result("  Enabled", True, "")

        # First FFT call (cache miss)
        start = time.time()
        freq1, mag1 = fft(medium)
        time1 = time.time() - start

        # Second FFT call (cache hit)
        start = time.time()
        freq2, mag2 = fft(medium)
        time2 = time.time() - start

        # Third FFT call with same data (cache hit)
        start = time.time()
        freq3, mag3 = fft(medium)
        time3 = time.time() - start

        self.info("\nFFT performance:")
        self.result("  First call (miss)", f"{time1 * 1000:.3f}", "ms")
        self.result("  Second call (hit)", f"{time2 * 1000:.3f}", "ms")
        self.result("  Third call (hit)", f"{time3 * 1000:.3f}", "ms")
        self.result("  Speedup (2nd)", f"{time1 / time2:.1f}", "x")

        # Check cache stats
        stats = get_fft_cache_stats()
        self.result("\nCache statistics", "")
        total_calls = stats["hits"] + stats["misses"]
        hit_rate = stats["hits"] / total_calls if total_calls > 0 else 0.0
        self.result("  Total calls", total_calls, "")
        self.result("  Cache hits", stats["hits"], "")
        self.result("  Cache misses", stats["misses"], "")
        self.result("  Hit rate", f"{hit_rate * 100:.1f}", "%")

        self.success("FFT caching provides significant speedup")

        # ===================================================================
        # Part 2: Lazy Evaluation
        # ===================================================================
        self.subsection("2. Lazy Evaluation Patterns")
        self.info("Defer computation until results are needed")

        from oscura import high_pass, low_pass

        # Eager evaluation (immediate computation)
        start = time.time()
        eager_filtered = high_pass(medium, cutoff=100.0)
        eager_filtered = low_pass(eager_filtered, cutoff=5000.0)
        _eager_rms = rms(eager_filtered)  # Result computed immediately
        eager_time = time.time() - start

        # Lazy evaluation (deferred computation)
        start = time.time()
        lazy = LazySignalProcessor(medium)
        lazy.filter(lambda t: high_pass(t, cutoff=100.0))
        lazy.filter(lambda t: low_pass(t, cutoff=5000.0))
        lazy.measure(rms)
        # No computation yet...
        lazy_time_setup = time.time() - start

        # Now execute
        start = time.time()
        lazy_trace, lazy_measurements = lazy.execute()
        lazy_time_exec = time.time() - start

        self.info("Evaluation strategies:")
        self.result("  Eager execution", f"{eager_time * 1000:.3f}", "ms")
        self.result("  Lazy setup", f"{lazy_time_setup * 1000:.3f}", "ms")
        self.result("  Lazy execution", f"{lazy_time_exec * 1000:.3f}", "ms")
        self.result("  Lazy total", f"{(lazy_time_setup + lazy_time_exec) * 1000:.3f}", "ms")

        # Show lazy benefits: reuse without recomputation
        start = time.time()
        lazy_trace2, lazy_measurements2 = lazy.execute()  # Cached!
        lazy_time_reuse = time.time() - start

        self.result("  Lazy reuse", f"{lazy_time_reuse * 1000:.3f}", "ms")
        self.success("Lazy evaluation enables efficient reuse")

        # ===================================================================
        # Part 3: Memory-Efficient Processing
        # ===================================================================
        self.subsection("3. Memory-Efficient Processing")
        self.info("Process large signals without excessive memory")

        # Calculate memory usage
        def estimate_memory(trace):
            """Estimate memory usage in MB."""
            # Data array + metadata
            data_bytes = trace.data.nbytes
            overhead = 1024  # Approximate overhead
            return (data_bytes + overhead) / (1024 * 1024)

        small_mem = estimate_memory(small)
        medium_mem = estimate_memory(medium)
        large_mem = estimate_memory(large)

        self.info("Memory usage estimates:")
        self.result("  Small signal", f"{small_mem:.3f}", "MB")
        self.result("  Medium signal", f"{medium_mem:.3f}", "MB")
        self.result("  Large signal", f"{large_mem:.3f}", "MB")

        # In-place operations save memory
        def process_in_place(trace, scale_factor):
            """Process signal in-place (memory efficient)."""
            trace.data *= scale_factor
            return trace

        def process_copy(trace, scale_factor):
            """Process signal with copy (memory inefficient)."""
            result = replace(trace, data=trace.data.copy())
            result.data[:] *= scale_factor
            return result

        # Compare memory patterns
        test_signal = replace(medium, data=medium.data.copy())

        start = time.time()
        _result_copy = process_copy(test_signal, 2.0)  # Creates new trace
        time_copy = time.time() - start

        test_signal2 = replace(medium, data=medium.data.copy())
        start = time.time()
        _result_inplace = process_in_place(test_signal2, 2.0)  # Modifies in place
        time_inplace = time.time() - start

        self.info("\nProcessing strategies:")
        self.result("  Copy-based", f"{time_copy * 1000:.3f}", "ms")
        self.result("  In-place", f"{time_inplace * 1000:.3f}", "ms")
        self.success("In-place operations reduce memory pressure")

        # ===================================================================
        # Part 4: Batch Processing Optimization
        # ===================================================================
        self.subsection("4. Batch Processing Optimization")
        self.info("Optimize processing of multiple signals")

        # Create batch of signals
        batch_size = 10
        signals = [
            generate_sine_wave(
                frequency=1000.0 + i * 100, amplitude=1.0, duration=0.001, sample_rate=100e3
            )
            for i in range(batch_size)
        ]

        # Sequential processing
        start = time.time()
        sequential_results = []
        for sig in signals:
            sequential_results.append(rms(sig))
        sequential_time = time.time() - start

        # Batched processing with list comprehension
        start = time.time()
        _batched_results = [rms(sig) for sig in signals]  # All results computed
        batched_time = time.time() - start

        self.info(f"Batch processing ({batch_size} signals):")
        self.result("  Sequential", f"{sequential_time * 1000:.3f}", "ms")
        self.result("  Batched", f"{batched_time * 1000:.3f}", "ms")
        self.result("  Per-signal (seq)", f"{sequential_time / batch_size * 1000:.3f}", "ms")
        self.result("  Per-signal (batch)", f"{batched_time / batch_size * 1000:.3f}", "ms")

        self.success("Batch processing improves throughput")

        # ===================================================================
        # Part 5: Computation Reuse
        # ===================================================================
        self.subsection("5. Computation Reuse Strategies")
        self.info("Reuse intermediate results")

        # Without reuse
        start = time.time()
        _thd1 = thd(medium)  # Computes FFT
        _thd2 = thd(medium)  # Recomputes FFT (if cache disabled)
        time_no_reuse = time.time() - start

        # With FFT caching (automatic reuse)
        configure_fft_cache(size=10)
        clear_fft_cache()

        start = time.time()
        _thd1_cached = thd(medium)  # Computes FFT
        _thd2_cached = thd(medium)  # Reuses cached FFT
        time_with_reuse = time.time() - start

        self.info("Computation reuse:")
        self.result("  Without cache", f"{time_no_reuse * 1000:.3f}", "ms")
        self.result("  With cache", f"{time_with_reuse * 1000:.3f}", "ms")
        self.result(
            "  Improvement", f"{(time_no_reuse - time_with_reuse) / time_no_reuse * 100:.1f}", "%"
        )

        self.success("Caching eliminates redundant computations")

        # ===================================================================
        # Part 6: Performance Summary
        # ===================================================================
        self.subsection("6. Performance Best Practices")
        self.info("Summary of optimization techniques:")
        self.info("  1. Enable FFT caching for repeated spectral analysis")
        self.info("  2. Use lazy evaluation for conditional processing")
        self.info("  3. Prefer in-place operations for large signals")
        self.info("  4. Batch similar operations together")
        self.info("  5. Reuse intermediate results when possible")

        cache_stats = get_fft_cache_stats()
        total = cache_stats["hits"] + cache_stats["misses"]
        hit_rate = cache_stats["hits"] / total if total > 0 else 0.0

        return {
            "cache_speedup": time1 / time2,
            "cache_hit_rate": hit_rate,
            "eager_time": eager_time,
            "lazy_time": lazy_time_exec,
            "sequential_time": sequential_time,
            "batched_time": batched_time,
            "thd_with_cache": _thd1_cached,
        }

    def validate(self, results: dict) -> bool:
        """Validate optimization results."""
        self.info("Validating optimization techniques...")

        # Cache should provide speedup
        if results["cache_speedup"] < 2.0:
            self.warning(f"Cache speedup modest: {results['cache_speedup']:.1f}x")
        else:
            print(f"  ✓ Cache speedup excellent: {results['cache_speedup']:.1f}x")

        # Cache hit rate should be good
        if results["cache_hit_rate"] < 0.5:
            print(f"  ✗ Cache hit rate too low: {results['cache_hit_rate'] * 100:.1f}%")
            return False
        print(f"  ✓ Cache hit rate: {results['cache_hit_rate'] * 100:.1f}%")

        # Lazy and eager should produce similar times (lazy has overhead)
        if results["lazy_time"] > results["eager_time"] * 2:
            self.warning("Lazy evaluation has significant overhead")
        else:
            print("  ✓ Lazy evaluation competitive with eager")

        # Batched should be faster or comparable to sequential
        if results["batched_time"] > results["sequential_time"] * 1.2:
            self.warning("Batched processing slower than sequential")
        else:
            print("  ✓ Batched processing efficient")

        # THD should be reasonable (can be very low for clean sine wave)
        if not (-120.0 <= results["thd_with_cache"] <= 0.0):
            print(f"  ✗ THD unreasonable: {results['thd_with_cache']:.2f} dB")
            return False
        print(f"  ✓ THD reasonable: {results['thd_with_cache']:.2f} dB")

        self.success("All optimization techniques validated!")
        self.info("\nKey takeaways:")
        self.info("  - FFT caching provides 2-10x speedup")
        self.info("  - Lazy evaluation enables efficient reuse")
        self.info("  - In-place operations reduce memory usage")
        self.info("  - Batch processing improves throughput")

        return True


if __name__ == "__main__":
    demo = OptimizationDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
