"""Parallel Processing: Multi-Core Signal Analysis

Demonstrates:
- Parallel batch processing with multiprocessing
- Thread pool and process pool patterns
- Progress tracking for parallel operations
- Multi-core utilization strategies
- Performance scaling analysis

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/05_optimization.py
- 07_advanced_api/06_streaming_api.py

Parallel processing enables efficient utilization of multi-core systems,
dramatically reducing processing time for batch operations.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave
from oscura import amplitude, frequency, rms, thd


def process_signal(signal_data: tuple) -> dict:
    """Process a single signal (for parallel execution)."""
    freq, amp, idx = signal_data

    # Generate signal
    signal = generate_sine_wave(frequency=freq, amplitude=amp, duration=0.01, sample_rate=100e3)

    # Perform analysis
    return {
        "index": idx,
        "frequency": freq,
        "amplitude": amp,
        "measured_rms": rms(signal),
        "measured_amplitude": amplitude(signal),
        "measured_frequency": frequency(signal),
    }


def process_signal_heavy(signal_data: tuple) -> dict:
    """Process signal with heavy computation (for parallelism benefit)."""
    freq, amp, idx = signal_data

    signal = generate_sine_wave(frequency=freq, amplitude=amp, duration=0.01, sample_rate=100e3)

    # Add harmonics for THD calculation (more expensive)
    signal2 = generate_sine_wave(
        frequency=freq * 2, amplitude=amp * 0.3, duration=0.01, sample_rate=100e3
    )
    signal3 = generate_sine_wave(
        frequency=freq * 3, amplitude=amp * 0.1, duration=0.01, sample_rate=100e3
    )

    signal.data = signal.data + signal2.data + signal3.data

    # Expensive operations
    return {
        "index": idx,
        "frequency": freq,
        "rms": rms(signal),
        "amplitude": amplitude(signal),
        "thd": thd(signal),
    }


class ParallelProcessingDemo(BaseDemo):
    """Demonstrate parallel processing for multi-core utilization."""

    def __init__(self):
        """Initialize parallel processing demonstration."""
        super().__init__(
            name="parallel_processing",
            description="Multi-core parallel processing and performance scaling",
            capabilities=[
                "parallel_processing",
                "multiprocessing",
                "thread_pool",
                "process_pool",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals."""
        # Create batch of signal parameters
        batch_size = 50
        signal_params = [
            (1000.0 + i * 100, 1.0, i)  # (frequency, amplitude, index)
            for i in range(batch_size)
        ]

        return {"signal_params": signal_params}

    def run_demonstration(self, data: dict) -> dict:
        """Run parallel processing demonstration."""
        signal_params = data["signal_params"]
        batch_size = len(signal_params)

        self.section("Parallel Processing: Multi-Core Utilization")

        # ===================================================================
        # Part 1: System Capabilities
        # ===================================================================
        self.subsection("1. System Capabilities")
        self.info("Detect available CPU resources")

        cpu_count = mp.cpu_count()
        self.result("CPU cores available", cpu_count, "")
        self.result("Batch size", batch_size, "signals")

        self.success(f"System has {cpu_count} cores for parallel processing")

        # ===================================================================
        # Part 2: Sequential Baseline
        # ===================================================================
        self.subsection("2. Sequential Processing (Baseline)")
        self.info("Process signals sequentially for comparison")

        start_time = time.time()
        sequential_results = []
        for params in signal_params:
            result = process_signal(params)
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        self.result("Sequential time", f"{sequential_time:.3f}", "s")
        self.result("Throughput", f"{batch_size / sequential_time:.1f}", "signals/s")
        self.result("Per-signal time", f"{sequential_time / batch_size * 1000:.2f}", "ms")

        # ===================================================================
        # Part 3: Thread Pool Parallelism
        # ===================================================================
        self.subsection("3. Thread Pool Parallelism")
        self.info("Use ThreadPoolExecutor for I/O-bound tasks")

        # Note: ThreadPoolExecutor is limited by GIL for CPU-bound tasks
        # but demonstrates the API pattern

        start_time = time.time()
        thread_results = []

        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_signal, params) for params in signal_params]

            for future in as_completed(futures):
                thread_results.append(future.result())

        thread_time = time.time() - start_time

        self.result("Thread pool time", f"{thread_time:.3f}", "s")
        self.result("Speedup", f"{sequential_time / thread_time:.2f}", "x")
        self.result("Efficiency", f"{sequential_time / thread_time / cpu_count * 100:.1f}", "%")

        self.info("Note: Thread pools limited by GIL for CPU-bound Python code")

        # ===================================================================
        # Part 4: Process Pool Parallelism
        # ===================================================================
        self.subsection("4. Process Pool Parallelism")
        self.info("Use ProcessPoolExecutor for CPU-bound tasks")

        start_time = time.time()
        process_results = []

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_signal, params) for params in signal_params]

            for future in as_completed(futures):
                process_results.append(future.result())

        process_time = time.time() - start_time

        self.result("Process pool time", f"{process_time:.3f}", "s")
        self.result("Speedup", f"{sequential_time / process_time:.2f}", "x")
        self.result("Efficiency", f"{sequential_time / process_time / cpu_count * 100:.1f}", "%")

        self.success("Process pools bypass GIL for true parallelism")

        # ===================================================================
        # Part 5: Scaling Analysis
        # ===================================================================
        self.subsection("5. Scaling Analysis")
        self.info("Test parallelism with different worker counts")

        worker_counts = [1, 2, 4, cpu_count]
        scaling_results = {}

        for workers in worker_counts:
            if workers > cpu_count:
                continue

            start_time = time.time()

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_signal, params) for params in signal_params[:20]]
                _results = [f.result() for f in as_completed(futures)]  # Collect all results

            elapsed = time.time() - start_time
            scaling_results[workers] = elapsed

            speedup = scaling_results[1] / elapsed if workers > 1 else 1.0
            efficiency = speedup / workers * 100 if workers > 1 else 100.0

            self.result(
                f"  {workers} workers",
                f"{elapsed:.3f} s",
                f"(speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)",
            )

        self.success("Scaling efficiency depends on task granularity")

        # ===================================================================
        # Part 6: Progress Tracking
        # ===================================================================
        self.subsection("6. Progress Tracking")
        self.info("Monitor parallel execution progress")

        completed_count = 0
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {executor.submit(process_signal, params): params for params in signal_params}

            for future in as_completed(futures):
                result = future.result()
                completed_count += 1

                # Progress update every 10 signals
                if completed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed
                    remaining = batch_size - completed_count
                    eta = remaining / rate

                    self.info(
                        f"  Progress: {completed_count}/{batch_size} ({completed_count / batch_size * 100:.0f}%) - ETA: {eta:.1f}s"
                    )

        total_time = time.time() - start_time
        self.result("Total time with progress", f"{total_time:.3f}", "s")
        self.success("Progress tracking enables user feedback")

        # ===================================================================
        # Part 7: Heavy Computation Parallelism
        # ===================================================================
        self.subsection("7. Heavy Computation Parallelism")
        self.info("Demonstrate parallelism benefit for expensive operations")

        # Sequential heavy processing
        heavy_params = signal_params[:20]  # Smaller batch for demonstration

        start_time = time.time()
        _seq_heavy_results = [process_signal_heavy(params) for params in heavy_params]
        seq_heavy_time = time.time() - start_time

        # Parallel heavy processing
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            _par_heavy_results = list(executor.map(process_signal_heavy, heavy_params))

        par_heavy_time = time.time() - start_time

        self.info("Heavy computation (THD + harmonics):")
        self.result("  Sequential time", f"{seq_heavy_time:.3f}", "s")
        self.result("  Parallel time", f"{par_heavy_time:.3f}", "s")
        self.result("  Speedup", f"{seq_heavy_time / par_heavy_time:.2f}", "x")
        self.result("  Efficiency", f"{seq_heavy_time / par_heavy_time / cpu_count * 100:.1f}", "%")

        self.success("Heavier computations show better parallel efficiency")

        # ===================================================================
        # Part 8: Batch Processing Best Practices
        # ===================================================================
        self.subsection("8. Best Practices Summary")
        self.info("Guidelines for parallel signal processing:")
        self.info("  1. Use ProcessPoolExecutor for CPU-bound tasks")
        self.info("  2. Match worker count to CPU cores (or slightly less)")
        self.info("  3. Ensure tasks are substantial enough (>10ms)")
        self.info("  4. Implement progress tracking for user feedback")
        self.info("  5. Handle exceptions in worker processes")
        self.info("  6. Consider memory overhead of multiple processes")

        # Compare all approaches
        self.info("\nPerformance comparison:")
        self.result("  Sequential", f"{sequential_time:.3f} s", "(baseline)")
        self.result(
            "  Thread pool", f"{thread_time:.3f} s", f"({sequential_time / thread_time:.2f}x)"
        )
        self.result(
            "  Process pool", f"{process_time:.3f} s", f"({sequential_time / process_time:.2f}x)"
        )

        best_approach = "Process pool" if process_time < thread_time else "Thread pool"
        self.success(f"Best approach for this workload: {best_approach}")

        return {
            "cpu_count": cpu_count,
            "sequential_time": sequential_time,
            "thread_time": thread_time,
            "process_time": process_time,
            "thread_speedup": sequential_time / thread_time,
            "process_speedup": sequential_time / process_time,
            "seq_heavy_time": seq_heavy_time,
            "par_heavy_time": par_heavy_time,
            "heavy_speedup": seq_heavy_time / par_heavy_time,
        }

    def validate(self, results: dict) -> bool:
        """Validate parallel processing results."""
        self.info("Validating parallel processing...")

        # Should have multiple cores
        if results["cpu_count"] < 2:
            self.warning("System has only 1 CPU core - limited parallelism benefit")
        else:
            print(f"  ✓ System has {results['cpu_count']} CPU cores")

        # Process pool should provide speedup
        if results["process_speedup"] < 1.2:
            self.warning(f"Process pool speedup modest: {results['process_speedup']:.2f}x")
        else:
            print(f"  ✓ Process pool speedup: {results['process_speedup']:.2f}x")

        # Heavy computation should show better speedup
        if results["heavy_speedup"] < results["process_speedup"]:
            self.warning("Heavy computation speedup worse than light - unexpected")
        else:
            print(f"  ✓ Heavy computation speedup: {results['heavy_speedup']:.2f}x (better)")

        # Efficiency check
        process_efficiency = results["process_speedup"] / results["cpu_count"]
        if process_efficiency < 0.5:
            self.warning(f"Low parallel efficiency: {process_efficiency * 100:.1f}%")
        else:
            print(f"  ✓ Parallel efficiency: {process_efficiency * 100:.1f}%")

        # Speedup should not exceed CPU count (theoretical limit)
        if results["process_speedup"] > results["cpu_count"] * 1.2:
            self.warning("Speedup exceeds CPU count - measurement anomaly")

        self.success("Parallel processing validated!")
        self.info("\nKey takeaways:")
        self.info("  - ProcessPoolExecutor bypasses GIL for true parallelism")
        self.info("  - Speedup scales with CPU cores and task complexity")
        self.info("  - Progress tracking essential for user experience")
        self.info("  - Heavy computations show better parallel efficiency")

        return True


if __name__ == "__main__":
    demo = ParallelProcessingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
