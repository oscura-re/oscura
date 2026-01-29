"""Comprehensive performance profiling suite for Oscura.

This module performs extensive performance analysis including:
- Loading performance curves for various file sizes
- Memory profiling and leak detection
- CPU bottleneck identification
- Scalability testing with concurrent operations
- Algorithm complexity measurements (O(n) behavior)
- Comparison against theoretical limits

Test categories:
- Small files (<1MB): 16 files
- Medium files (1-10MB): 43 files
- Large files (>10MB): 4 files
"""

from __future__ import annotations

import cProfile
import gc
import io
import json
import pstats
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.loaders import load


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    operation: str
    file_size_bytes: int
    duration_seconds: float
    peak_memory_mb: float
    cpu_time_seconds: float
    samples_processed: int
    throughput_samples_per_sec: float
    memory_per_sample_bytes: float


@dataclass
class ProfileResult:
    """Container for profiling analysis results."""

    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float


def _load_and_analyze_worker(path: Path) -> float:
    """Load file and perform basic analysis (module-level for pickling)."""
    start = time.perf_counter()
    trace = load(path)
    from oscura.analyzers.statistics import basic

    _ = basic.basic_stats(trace.data)
    return time.perf_counter() - start


class PerformanceProfiler:
    """Main performance profiling orchestrator."""

    def __init__(self, output_dir: Path):
        """Initialize profiler with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[PerformanceMetrics] = []
        self.profiles: dict[str, list[ProfileResult]] = {}

    def profile_loading_performance(self, file_paths: list[Path]) -> list[PerformanceMetrics]:
        """Profile loading performance for various file sizes.

        Args:
            file_paths: List of test files to profile

        Returns:
            List of performance metrics for each file
        """
        print("\n=== Loading Performance Profiling ===")
        results = []

        for file_path in file_paths:
            file_size = file_path.stat().st_size
            print(f"\nProfiling: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")

            # Warm up
            _ = load(file_path)
            gc.collect()

            # Profile with memory tracking
            tracemalloc.start()
            process = psutil.Process()
            mem_before = process.memory_info().rss

            start_time = time.perf_counter()
            cpu_start = time.process_time()

            trace = load(file_path)

            cpu_end = time.process_time()
            end_time = time.perf_counter()

            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_after = process.memory_info().rss
            mem_delta = (mem_after - mem_before) / 1024 / 1024  # MB

            duration = end_time - start_time
            cpu_time = cpu_end - cpu_start
            samples = len(trace.data)
            throughput = samples / duration if duration > 0 else 0
            mem_per_sample = (mem_delta * 1024 * 1024) / samples if samples > 0 else 0

            metrics = PerformanceMetrics(
                operation="load",
                file_size_bytes=file_size,
                duration_seconds=duration,
                peak_memory_mb=peak / 1024 / 1024,
                cpu_time_seconds=cpu_time,
                samples_processed=samples,
                throughput_samples_per_sec=throughput,
                memory_per_sample_bytes=mem_per_sample,
            )

            results.append(metrics)
            self.metrics.append(metrics)

            print(f"  Duration: {duration:.4f}s")
            print(f"  Throughput: {throughput / 1e6:.2f} MSa/s")
            print(f"  Peak memory: {peak / 1024 / 1024:.2f} MB")
            print(f"  Memory/sample: {mem_per_sample:.2f} bytes")

        return results

    def profile_cpu_bottlenecks(self, file_path: Path) -> dict[str, Any]:
        """Profile CPU usage and identify bottlenecks.

        Args:
            file_path: Test file to profile

        Returns:
            Dictionary with profiling statistics
        """
        print(f"\n=== CPU Profiling: {file_path.name} ===")

        profiler = cProfile.Profile()
        profiler.enable()

        # Perform various operations
        trace = load(file_path)

        # Import analysis modules
        from oscura.analyzers.statistics import basic

        # Run different analysis operations
        stats = basic.basic_stats(trace.data)

        # Run FFT
        if len(trace.data) >= 1000:
            fft_result = np.fft.rfft(trace.data[:100000])

        # Time vector computation
        _ = trace.time_vector

        # Additional statistical operations
        percentiles_result = np.percentile(trace.data, [25, 50, 75, 95, 99])

        profiler.disable()

        # Analyze results
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats("cumulative")
        ps.print_stats(20)

        # Extract top functions
        top_functions = []
        ps.sort_stats("cumulative")
        for func, (_cc, nc, tt, ct, _callers) in list(ps.stats.items())[:20]:
            if isinstance(func, tuple):
                filename, line, func_name = func
                top_functions.append(
                    ProfileResult(
                        function_name=f"{func_name} ({Path(filename).name}:{line})",
                        total_time=tt,
                        cumulative_time=ct,
                        call_count=nc,
                        time_per_call=tt / nc if nc > 0 else 0,
                    )
                )

        self.profiles["cpu_bottlenecks"] = top_functions

        # Print top bottlenecks
        print("\nTop CPU consumers:")
        for i, result in enumerate(top_functions[:10], 1):
            print(
                f"{i}. {result.function_name}: {result.cumulative_time:.4f}s "
                f"({result.call_count} calls)"
            )

        return {
            "top_functions": top_functions,
            "profile_text": stream.getvalue(),
        }

    def profile_memory_usage(self, file_sizes: list[int]) -> list[dict[str, Any]]:
        """Profile memory usage patterns and detect leaks.

        Args:
            file_sizes: List of file sizes to test (in number of samples)

        Returns:
            List of memory profiling results
        """
        print("\n=== Memory Usage Profiling ===")
        results = []

        for size in file_sizes:
            print(f"\nTesting size: {size / 1e6:.2f} MSamples")

            # Create test data
            data = np.random.randn(size).astype(np.float64)
            theoretical_size = data.nbytes / 1024 / 1024  # MB

            tracemalloc.start()
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            # Create trace
            metadata = TraceMetadata(sample_rate=1e9)
            trace = WaveformTrace(data=data, metadata=metadata)

            # Perform operations
            _ = trace.time_vector
            _ = trace.duration

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            mem_delta = mem_after - mem_before
            overhead = (mem_delta - theoretical_size) / theoretical_size * 100

            result = {
                "samples": size,
                "theoretical_mb": theoretical_size,
                "actual_mb": mem_delta,
                "peak_mb": peak / 1024 / 1024,
                "overhead_percent": overhead,
            }
            results.append(result)

            print(f"  Theoretical: {theoretical_size:.2f} MB")
            print(f"  Actual: {mem_delta:.2f} MB")
            print(f"  Overhead: {overhead:.1f}%")

            # Clean up
            del trace
            del data
            gc.collect()

        return results

    def test_scalability_concurrent(
        self, file_paths: list[Path], max_workers: int = 4
    ) -> dict[str, Any]:
        """Test scalability with concurrent operations.

        Args:
            file_paths: Files to process concurrently
            max_workers: Maximum number of concurrent workers

        Returns:
            Dictionary with scalability results
        """
        print(f"\n=== Scalability Testing (max_workers={max_workers}) ===")

        # Sequential baseline
        print("\nSequential processing...")
        start = time.perf_counter()
        sequential_times = [_load_and_analyze_worker(p) for p in file_paths]
        sequential_total = time.perf_counter() - start

        # Thread-based parallelism
        print(f"\nThread-based parallel processing ({max_workers} workers)...")
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            thread_times = list(executor.map(_load_and_analyze_worker, file_paths))
        thread_total = time.perf_counter() - start

        # Process-based parallelism
        print(f"\nProcess-based parallel processing ({max_workers} workers)...")
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_times = list(executor.map(_load_and_analyze_worker, file_paths))
        process_total = time.perf_counter() - start

        speedup_thread = sequential_total / thread_total
        speedup_process = sequential_total / process_total

        results = {
            "sequential_total": sequential_total,
            "thread_total": thread_total,
            "process_total": process_total,
            "speedup_thread": speedup_thread,
            "speedup_process": speedup_process,
            "efficiency_thread": speedup_thread / max_workers,
            "efficiency_process": speedup_process / max_workers,
        }

        print("\nResults:")
        print(f"  Sequential: {sequential_total:.2f}s")
        print(f"  Thread parallel: {thread_total:.2f}s (speedup: {speedup_thread:.2f}x)")
        print(f"  Process parallel: {process_total:.2f}s (speedup: {speedup_process:.2f}x)")

        return results

    def measure_algorithm_complexity(self) -> dict[str, Any]:
        """Measure O(n) behavior for different operations.

        Returns:
            Dictionary with complexity measurements
        """
        print("\n=== Algorithm Complexity Analysis ===")

        from oscura.analyzers.statistics import basic

        # Test sizes
        sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        operations = {
            "statistics": lambda data: basic.basic_stats(data),
            "time_vector": lambda trace: trace.time_vector,
        }

        results = {}

        for op_name, op_func in operations.items():
            print(f"\nMeasuring: {op_name}")
            timings = []

            for size in sizes:
                # Create test data
                data = np.random.randn(size)
                metadata = TraceMetadata(sample_rate=1e9)
                trace = WaveformTrace(data=data, metadata=metadata)

                # Time the operation
                if "time_vector" in op_name:
                    times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        _ = op_func(trace)
                        times.append(time.perf_counter() - start)
                else:
                    times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        _ = op_func(data)
                        times.append(time.perf_counter() - start)

                avg_time = np.mean(times)
                timings.append({"size": size, "time": avg_time})
                print(f"  n={size}: {avg_time * 1000:.4f}ms")

            # Estimate complexity
            # Fit to a*n^b model
            sizes_arr = np.array([t["size"] for t in timings])
            times_arr = np.array([t["time"] for t in timings])

            # Log-log regression
            log_sizes = np.log10(sizes_arr)
            log_times = np.log10(times_arr)
            coeffs = np.polyfit(log_sizes, log_times, 1)
            complexity = coeffs[0]

            results[op_name] = {
                "timings": timings,
                "estimated_complexity": f"O(n^{complexity:.2f})",
                "complexity_exponent": complexity,
            }

            print(f"  Estimated complexity: O(n^{complexity:.2f})")

        return results

    def save_results(self, complexity_results: dict[str, Any]) -> None:
        """Save all profiling results to JSON files.

        Args:
            complexity_results: Algorithm complexity measurement results
        """
        # Save metrics
        metrics_data = [
            {
                "operation": m.operation,
                "file_size_bytes": m.file_size_bytes,
                "file_size_mb": m.file_size_bytes / 1024 / 1024,
                "duration_seconds": m.duration_seconds,
                "peak_memory_mb": m.peak_memory_mb,
                "cpu_time_seconds": m.cpu_time_seconds,
                "samples_processed": m.samples_processed,
                "throughput_samples_per_sec": m.throughput_samples_per_sec,
                "throughput_msa_per_sec": m.throughput_samples_per_sec / 1e6,
                "memory_per_sample_bytes": m.memory_per_sample_bytes,
            }
            for m in self.metrics
        ]

        with open(self.output_dir / "performance_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Save CPU profiles
        cpu_profiles = {
            "top_functions": [
                {
                    "function_name": p.function_name,
                    "total_time": p.total_time,
                    "cumulative_time": p.cumulative_time,
                    "call_count": p.call_count,
                    "time_per_call": p.time_per_call,
                }
                for p in self.profiles.get("cpu_bottlenecks", [])
            ]
        }

        with open(self.output_dir / "cpu_profiles.json", "w") as f:
            json.dump(cpu_profiles, f, indent=2)

        # Save complexity results
        with open(self.output_dir / "complexity_analysis.json", "w") as f:
            json.dump(complexity_results, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")


def generate_test_files(output_dir: Path) -> dict[str, list[Path]]:
    """Generate synthetic test files of various sizes.

    Args:
        output_dir: Directory to store test files

    Returns:
        Dictionary mapping size categories to file paths
    """
    print("\n=== Generating Test Files ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {"small": [], "medium": [], "large": []}

    # Small files (<1MB): 16 files
    # At float64 (8 bytes/sample), 1MB = 125,000 samples
    print("\nGenerating small files (<1MB)...")
    small_sizes = [10000, 25000, 50000, 100000]
    for i, size in enumerate(small_sizes):
        for j in range(4):  # 4 files per size
            file_path = output_dir / f"small_{i}_{j}_{size}.npz"
            data = np.random.randn(size)
            metadata = TraceMetadata(sample_rate=1e9)
            # Save as NPZ
            np.savez(file_path, data=data, sample_rate=metadata.sample_rate)
            files["small"].append(file_path)
    print(f"  Generated {len(files['small'])} small files")

    # Medium files (1-10MB): 43 files
    print("\nGenerating medium files (1-10MB)...")
    # 125K to 1.25M samples
    medium_sizes = [125000, 250000, 500000, 750000, 1000000, 1250000]
    counts = [8, 8, 8, 7, 7, 5]  # Total: 43
    for i, (size, count) in enumerate(zip(medium_sizes, counts, strict=False)):
        for j in range(count):
            file_path = output_dir / f"medium_{i}_{j}_{size}.npz"
            data = np.random.randn(size)
            metadata = TraceMetadata(sample_rate=1e9)
            # Save as NPZ
            np.savez(file_path, data=data, sample_rate=metadata.sample_rate)
            files["medium"].append(file_path)
    print(f"  Generated {len(files['medium'])} medium files")

    # Large files (>10MB): 4 files
    print("\nGenerating large files (>10MB)...")
    # >1.25M samples
    large_sizes = [2000000, 5000000, 10000000, 20000000]
    for i, size in enumerate(large_sizes):
        file_path = output_dir / f"large_{i}_{size}.npz"
        data = np.random.randn(size)
        metadata = TraceMetadata(sample_rate=1e9)
        # Save as NPZ
        np.savez(file_path, data=data, sample_rate=metadata.sample_rate)
        files["large"].append(file_path)
    print(f"  Generated {len(files['large'])} large files")

    total = len(files["small"]) + len(files["medium"]) + len(files["large"])
    print(f"\nTotal files generated: {total}")

    return files


def main():
    """Run comprehensive performance profiling."""
    base_dir = Path(__file__).parent
    test_data_dir = base_dir / "test_data"
    results_dir = base_dir / "results"

    print("=" * 80)
    print("Oscura Comprehensive Performance Analysis")
    print("=" * 80)

    # Generate test files
    test_files = generate_test_files(test_data_dir)
    all_files = test_files["small"] + test_files["medium"] + test_files["large"]

    # Initialize profiler
    profiler = PerformanceProfiler(results_dir)

    # 1. Profile loading performance
    loading_metrics = profiler.profile_loading_performance(all_files)

    # 2. Profile CPU bottlenecks (use a medium file)
    cpu_profile = profiler.profile_cpu_bottlenecks(test_files["medium"][0])

    # 3. Profile memory usage
    memory_results = profiler.profile_memory_usage([100000, 500000, 1000000, 5000000, 10000000])

    # Save memory results
    with open(results_dir / "memory_profiling.json", "w") as f:
        json.dump(memory_results, f, indent=2)

    # 4. Test scalability
    scalability_files = test_files["small"][:8]  # Use subset for concurrency test
    scalability_results = profiler.test_scalability_concurrent(scalability_files)

    with open(results_dir / "scalability.json", "w") as f:
        json.dump(scalability_results, f, indent=2)

    # 5. Measure algorithm complexity
    complexity_results = profiler.measure_algorithm_complexity()

    # Save all results
    profiler.save_results(complexity_results)

    print("\n" + "=" * 80)
    print("Profiling Complete!")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print("\nGenerated files:")
    print("  - performance_metrics.json")
    print("  - cpu_profiles.json")
    print("  - memory_profiling.json")
    print("  - scalability.json")
    print("  - complexity_analysis.json")


if __name__ == "__main__":
    main()
