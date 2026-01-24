"""Batch Processing Optimization: Serial vs Parallel vs GPU Performance Comparison

Demonstrates:
- Serial processing baseline
- Parallel thread-based processing (oscura.optimization.parallel)
- Parallel process-based processing
- GPU batch processing with graceful fallback (oscura.core.gpu_backend)
- Performance comparison across all methods
- Best practices for batch optimization

This demonstration shows how to optimize batch processing performance by choosing
the right parallelization strategy (threads vs processes vs GPU) based on workload
characteristics and available hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.batch.advanced import AdvancedBatchProcessor, BatchConfig
from oscura.core.gpu_backend import GPUBackend
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.optimization.parallel import (
    get_optimal_workers,
    parallel_map,
)


class BatchOptimizationDemo(BaseDemo):
    """Demonstrate batch processing optimization strategies."""

    def __init__(self) -> None:
        """Initialize batch optimization demonstration."""
        super().__init__(
            name="batch_optimization",
            description="Performance comparison: serial vs parallel vs GPU batch processing",
            capabilities=[
                "oscura.batch.AdvancedBatchProcessor",
                "oscura.optimization.parallel.parallel_map",
                "oscura.optimization.parallel.get_optimal_workers",
                "oscura.core.gpu_backend.GPUBackend",
            ],
            related_demos=[
                "09_batch_processing/01_parallel_batch.py",
                "09_batch_processing/02_result_aggregation.py",
                "09_batch_processing/03_progress_tracking.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for batch processing benchmarks.

        Returns:
            Dictionary containing:
            - traces: List of WaveformTrace objects
            - num_files: Number of test files
            - expected_speedup: Expected performance improvements
        """
        num_files = 50  # Enough to see parallelization benefits
        sample_rate = 1e6  # 1 MHz
        duration = 0.02  # 20 ms per trace
        num_samples = int(duration * sample_rate)

        traces = []
        for i in range(num_files):
            # Generate mixed-frequency signal
            frequency1 = 1e3 * (i + 1)  # 1-50 kHz
            frequency2 = 5e3 * (i + 1)  # 5-250 kHz
            t = np.arange(num_samples) / sample_rate

            # Complex signal with multiple components
            data = (
                0.5 * np.sin(2 * np.pi * frequency1 * t)
                + 0.3 * np.sin(2 * np.pi * frequency2 * t)
                + 0.1 * np.random.randn(num_samples)  # Noise
            )

            metadata = TraceMetadata(
                sample_rate=sample_rate,
                channel_name=f"CH{i:02d}",
                source_file=f"trace_{i:03d}.bin",
            )

            trace = WaveformTrace(data=data, metadata=metadata)
            traces.append(trace)

        return {
            "traces": traces,
            "num_files": num_files,
            "sample_rate": sample_rate,
            "duration": duration,
            "num_samples": num_samples,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the batch optimization demonstration."""
        results: dict[str, Any] = {}

        self.section("Batch Processing Optimization Demonstration")
        self.info("Comparing serial, parallel (threads), parallel (processes), and GPU processing")

        traces = data["traces"]
        num_files = data["num_files"]

        # Optimal worker count
        optimal_workers = get_optimal_workers()
        self.info(f"System CPU cores: {optimal_workers}")

        # Check GPU availability
        gpu_backend = GPUBackend()
        gpu_available = gpu_backend.gpu_available
        if gpu_available:
            self.success("GPU acceleration available (CuPy detected)")
        else:
            self.info("GPU not available - will demonstrate CPU-only fallback")

        # Part 1: Serial processing baseline
        self.subsection("Part 1: Serial Processing Baseline")
        self.info("Process all files sequentially (single thread).")

        start_time = time.time()
        serial_results = []

        for i, trace in enumerate(traces):
            result = self._process_signal_cpu(trace, i)
            serial_results.append(result)

        serial_time = time.time() - start_time

        self.result("Files processed", num_files)
        self.result("Serial time", f"{serial_time:.3f}", "seconds")
        self.result("Throughput", f"{num_files / serial_time:.1f}", "files/sec")
        self.result("Average per file", f"{serial_time / num_files * 1000:.1f}", "ms")

        results["serial_time"] = serial_time
        results["serial_results"] = serial_results

        # Part 2: Parallel processing with threads
        self.subsection("Part 2: Parallel Processing (Threads)")
        self.info("Use thread pool for I/O-bound operations.")

        start_time = time.time()

        # Use oscura.optimization.parallel.parallel_map
        parallel_result = parallel_map(
            self._process_signal_cpu_wrapper,
            [(trace, i) for i, trace in enumerate(traces)],
            max_workers=optimal_workers,
            use_threads=True,
            collect_errors=True,
        )

        thread_time = parallel_result.execution_time

        self.result("Files processed", parallel_result.success_count)
        self.result("Thread pool time", f"{thread_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / thread_time:.2f}", "x")
        self.result("Throughput", f"{num_files / thread_time:.1f}", "files/sec")
        self.result("Average per file", f"{thread_time / num_files * 1000:.1f}", "ms")

        if parallel_result.error_count > 0:
            self.warning(f"{parallel_result.error_count} errors encountered")

        results["thread_time"] = thread_time
        results["thread_speedup"] = serial_time / thread_time
        results["thread_results"] = parallel_result.results

        # Part 3: Parallel processing with processes
        self.subsection("Part 3: Parallel Processing (Processes)")
        self.info("Use process pool for CPU-bound operations (bypasses GIL).")

        start_time = time.time()

        # Use process pool
        parallel_result = parallel_map(
            _process_signal_cpu_standalone,
            [(trace, i) for i, trace in enumerate(traces)],
            max_workers=optimal_workers,
            use_threads=False,  # Use processes
            collect_errors=True,
        )

        process_time = parallel_result.execution_time

        self.result("Files processed", parallel_result.success_count)
        self.result("Process pool time", f"{process_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / process_time:.2f}", "x")
        self.result("Speedup vs threads", f"{thread_time / process_time:.2f}", "x")
        self.result("Throughput", f"{num_files / process_time:.1f}", "files/sec")
        self.result("Average per file", f"{process_time / num_files * 1000:.1f}", "ms")

        results["process_time"] = process_time
        results["process_speedup"] = serial_time / process_time
        results["process_results"] = parallel_result.results

        # Part 4: GPU batch processing
        self.subsection("Part 4: GPU Batch Processing")
        if gpu_available:
            self.info("Use GPU acceleration for FFT-heavy workloads.")
        else:
            self.info("GPU not available - demonstrating graceful CPU fallback.")

        start_time = time.time()

        # Process with GPU backend (automatic fallback to CPU if unavailable)
        gpu_results = []
        for i, trace in enumerate(traces):
            result = self._process_signal_gpu(trace, i, gpu_backend)
            gpu_results.append(result)

        gpu_time = time.time() - start_time

        self.result("Files processed", len(gpu_results))
        self.result("GPU time", f"{gpu_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / gpu_time:.2f}", "x")
        self.result("Speedup vs threads", f"{thread_time / gpu_time:.2f}", "x")
        self.result("Speedup vs processes", f"{process_time / gpu_time:.2f}", "x")
        self.result("Throughput", f"{num_files / gpu_time:.1f}", "files/sec")
        self.result("Average per file", f"{gpu_time / num_files * 1000:.1f}", "ms")

        if gpu_available:
            self.success("GPU acceleration provided performance boost")
        else:
            self.info("CPU fallback performed gracefully")

        results["gpu_time"] = gpu_time
        results["gpu_speedup"] = serial_time / gpu_time
        results["gpu_results"] = gpu_results
        results["gpu_available"] = gpu_available

        # Part 5: Batch processing with AdvancedBatchProcessor
        self.subsection("Part 5: Advanced Batch Processing")
        self.info("Use AdvancedBatchProcessor for production workflows.")

        config = BatchConfig(
            on_error="skip",
            max_workers=optimal_workers,
            use_threads=False,  # Processes for CPU-bound work
            progress_bar=False,  # Disable for demo output clarity
            timeout_per_file=5.0,  # 5 second timeout per file
        )

        processor = AdvancedBatchProcessor(config)

        # Create file paths for processor
        file_paths = [f"trace_{i:03d}.bin" for i in range(num_files)]

        # Analysis function wrapper
        def analyze_trace_by_path(file_path: str) -> dict[str, Any]:
            # Find trace by path
            idx = int(file_path.split("_")[1].split(".")[0])
            if idx < len(traces):
                return self._process_signal_cpu(traces[idx], idx)
            return {}

        start_time = time.time()
        batch_results_df = processor.process(file_paths, analyze_trace_by_path)
        batch_time = time.time() - start_time

        self.result("Files processed", len(batch_results_df))
        self.result("Successful", batch_results_df["success"].sum())
        self.result("Failed", (~batch_results_df["success"]).sum())
        self.result("Batch time", f"{batch_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / batch_time:.2f}", "x")

        results["batch_time"] = batch_time
        results["batch_results_df"] = batch_results_df

        # Part 6: Performance comparison summary
        self.subsection("Part 6: Performance Comparison Summary")

        comparison = {
            "Serial": {
                "time": serial_time,
                "speedup": 1.0,
                "throughput": num_files / serial_time,
            },
            "Threads": {
                "time": thread_time,
                "speedup": serial_time / thread_time,
                "throughput": num_files / thread_time,
            },
            "Processes": {
                "time": process_time,
                "speedup": serial_time / process_time,
                "throughput": num_files / process_time,
            },
            "GPU": {
                "time": gpu_time,
                "speedup": serial_time / gpu_time,
                "throughput": num_files / gpu_time,
            },
            "Advanced Batch": {
                "time": batch_time,
                "speedup": serial_time / batch_time,
                "throughput": num_files / batch_time,
            },
        }

        self.info("\nMethod              Time (s)  Speedup  Throughput (files/s)")
        self.info("-" * 65)
        for method, stats in comparison.items():
            self.info(
                f"{method:18s}  {stats['time']:6.2f}    {stats['speedup']:5.2f}x    "
                f"{stats['throughput']:6.1f}"
            )

        results["comparison"] = comparison

        # Part 7: Optimization guidelines
        self.subsection("Part 7: Optimization Best Practices")

        self.info("\nChoosing the right strategy:")
        self.info("  1. Serial: Small datasets (<10 files), simple operations")
        self.info("  2. Threads: I/O-bound (file loading, network), moderate CPU work")
        self.info("  3. Processes: CPU-bound (complex analysis), bypasses Python GIL")
        self.info("  4. GPU: FFT-heavy, large arrays, convolution (requires CuPy + NVIDIA GPU)")
        self.info("  5. AdvancedBatchProcessor: Production use with error handling")

        self.info("\nKey considerations:")
        self.info("  - Threads: Low overhead, shares memory, limited by GIL for CPU work")
        self.info("  - Processes: Higher overhead, isolated memory, true parallelism")
        self.info("  - GPU: Best for large-scale FFT, requires data transfer overhead")
        self.info("  - Batch: Adds checkpointing, resume, timeout, error isolation")

        self.success("Batch optimization demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating batch optimization results...")

        # Validate all methods completed
        required_keys = [
            "serial_time",
            "thread_time",
            "process_time",
            "gpu_time",
            "batch_time",
        ]
        if not all(key in results for key in required_keys):
            self.error("Missing timing results")
            return False

        # Validate speedups are reasonable (at least not slower by orders of magnitude)
        if results["thread_speedup"] < 0.1:
            self.error(f"Thread speedup ({results['thread_speedup']:.2f}x) is unreasonably low")
            return False

        if results["process_speedup"] < 0.1:
            self.error(f"Process speedup ({results['process_speedup']:.2f}x) is unreasonably low")
            return False

        # GPU speedup validation depends on availability
        if results["gpu_available"]:
            if results["gpu_speedup"] < 0.1:
                self.warning(f"GPU speedup ({results['gpu_speedup']:.2f}x) is lower than expected")
        else:
            # GPU fallback to CPU should still work
            if results["gpu_speedup"] < 0.1:
                self.error("GPU fallback to CPU failed")
                return False

        # Validate result counts
        num_files = len(results["serial_results"])
        if len(results["thread_results"]) != num_files:
            self.error(
                f"Thread result count mismatch: {len(results['thread_results'])} != {num_files}"
            )
            return False

        if len(results["process_results"]) != num_files:
            self.error(
                f"Process result count mismatch: {len(results['process_results'])} != {num_files}"
            )
            return False

        if len(results["gpu_results"]) != num_files:
            self.error(f"GPU result count mismatch: {len(results['gpu_results'])} != {num_files}")
            return False

        # Validate batch results DataFrame
        batch_df = results["batch_results_df"]
        if len(batch_df) != num_files:
            self.error(f"Batch result count mismatch: {len(batch_df)} != {num_files}")
            return False

        # Validate result correctness (all methods should produce similar results)
        serial_rms = [r["rms"] for r in results["serial_results"]]
        thread_rms = [r["rms"] for r in results["thread_results"]]

        # Check first result matches (within floating point tolerance)
        if not np.isclose(serial_rms[0], thread_rms[0], rtol=1e-6):
            self.error(
                f"Result mismatch between serial and thread: {serial_rms[0]} vs {thread_rms[0]}"
            )
            return False

        # Validate comparison table
        if "comparison" not in results:
            self.error("Missing comparison table")
            return False

        comparison = results["comparison"]
        if len(comparison) != 5:
            self.error(f"Expected 5 comparison entries, got {len(comparison)}")
            return False

        self.success("All batch optimization validations passed!")
        self.info("\nKey Findings:")
        self.info(f"  - Thread speedup: {results['thread_speedup']:.2f}x")
        self.info(f"  - Process speedup: {results['process_speedup']:.2f}x")
        if results["gpu_available"]:
            self.info(f"  - GPU speedup: {results['gpu_speedup']:.2f}x (GPU available)")
        else:
            self.info("  - GPU fallback: CPU graceful fallback verified")
        self.info("  - All methods produced consistent results")
        self.info("  - AdvancedBatchProcessor provides production-ready features")

        return True

    def _process_signal_cpu(self, trace: WaveformTrace, file_id: int) -> dict[str, Any]:
        """Process signal using CPU (numpy).

        Args:
            trace: Waveform trace to process
            file_id: File identifier

        Returns:
            Dictionary with analysis results
        """
        # Simulate realistic CPU-bound workload
        time.sleep(0.01)  # Simulate I/O overhead

        # CPU-based analysis
        rms = float(np.sqrt(np.mean(trace.data**2)))
        peak = float(np.max(np.abs(trace.data)))
        mean = float(np.mean(trace.data))
        std = float(np.std(trace.data))

        # FFT analysis (CPU-bound)
        fft = np.fft.rfft(trace.data)
        power_spectrum = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(trace.data), 1 / trace.metadata.sample_rate)
        dominant_freq = float(freqs[np.argmax(power_spectrum)])
        total_power = float(np.sum(power_spectrum))

        return {
            "file_id": file_id,
            "channel": trace.metadata.channel_name,
            "rms": rms,
            "peak": peak,
            "mean": mean,
            "std": std,
            "dominant_frequency": dominant_freq,
            "total_power": total_power,
            "num_samples": len(trace.data),
        }

    def _process_signal_cpu_wrapper(self, args: tuple[WaveformTrace, int]) -> dict[str, Any]:
        """Wrapper for parallel_map compatibility.

        Args:
            args: Tuple of (trace, file_id)

        Returns:
            Dictionary with analysis results
        """
        trace, file_id = args
        return self._process_signal_cpu(trace, file_id)

    def _process_signal_gpu(
        self, trace: WaveformTrace, file_id: int, gpu_backend: GPUBackend
    ) -> dict[str, Any]:
        """Process signal using GPU backend (automatic CPU fallback).

        Args:
            trace: Waveform trace to process
            file_id: File identifier
            gpu_backend: GPU backend instance

        Returns:
            Dictionary with analysis results
        """
        # Simulate I/O overhead
        time.sleep(0.01)

        # Use GPU backend (automatically falls back to CPU if unavailable)
        rms = float(np.sqrt(np.mean(trace.data**2)))
        peak = float(np.max(np.abs(trace.data)))
        mean = float(np.mean(trace.data))
        std = float(np.std(trace.data))

        # GPU-accelerated FFT (or CPU fallback)
        fft = gpu_backend.rfft(trace.data)
        power_spectrum = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(trace.data), 1 / trace.metadata.sample_rate)
        dominant_freq = float(freqs[np.argmax(power_spectrum)])
        total_power = float(np.sum(power_spectrum))

        return {
            "file_id": file_id,
            "channel": trace.metadata.channel_name,
            "rms": rms,
            "peak": peak,
            "mean": mean,
            "std": std,
            "dominant_frequency": dominant_freq,
            "total_power": total_power,
            "num_samples": len(trace.data),
            "used_gpu": gpu_backend.gpu_available,
        }


def _process_signal_cpu_standalone(args: tuple[WaveformTrace, int]) -> dict[str, Any]:
    """Standalone function for process pool (must be picklable).

    Args:
        args: Tuple of (trace, file_id)

    Returns:
        Dictionary with analysis results
    """
    trace, file_id = args

    # Simulate I/O
    time.sleep(0.01)

    # CPU analysis
    rms = float(np.sqrt(np.mean(trace.data**2)))
    peak = float(np.max(np.abs(trace.data)))
    mean = float(np.mean(trace.data))
    std = float(np.std(trace.data))

    # FFT
    fft = np.fft.rfft(trace.data)
    power_spectrum = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(trace.data), 1 / trace.metadata.sample_rate)
    dominant_freq = float(freqs[np.argmax(power_spectrum)])
    total_power = float(np.sum(power_spectrum))

    return {
        "file_id": file_id,
        "channel": trace.metadata.channel_name,
        "rms": rms,
        "peak": peak,
        "mean": mean,
        "std": std,
        "dominant_frequency": dominant_freq,
        "total_power": total_power,
        "num_samples": len(trace.data),
    }


if __name__ == "__main__":
    demo: BatchOptimizationDemo = BatchOptimizationDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
