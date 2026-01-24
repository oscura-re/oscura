"""Parallel Batch Processing: Multi-file Processing with Thread and Process Pools

Demonstrates:
- Multi-file parallel processing
- Thread pool vs process pool comparison
- Progress tracking across files
- Result aggregation from parallel workers
- Error handling in parallel contexts

This demonstration shows how to process multiple signal files in parallel
using both thread and process pools, with proper progress tracking and
result aggregation.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace


class ParallelBatchDemo(BaseDemo):
    """Demonstrate parallel batch processing with thread and process pools."""

    def __init__(self) -> None:
        """Initialize parallel batch processing demonstration."""
        super().__init__(
            name="parallel_batch",
            description="Process multiple signal files in parallel with thread/process pools",
            capabilities=[
                "concurrent.futures.ThreadPoolExecutor",
                "concurrent.futures.ProcessPoolExecutor",
                "batch.parallel_processing",
                "batch.progress_tracking",
            ],
            related_demos=[
                "09_batch_processing/02_result_aggregation.py",
                "09_batch_processing/03_progress_tracking.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate multiple synthetic signals for batch processing.

        Returns:
            Dictionary containing:
            - traces: List of WaveformTrace objects
            - num_files: Number of files to process
            - expected_results: Expected processing results
        """
        num_files = 20
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        num_samples = int(duration * sample_rate)

        traces = []
        for i in range(num_files):
            # Generate different frequency sine waves
            frequency = 1e3 * (i + 1)  # 1 kHz to 20 kHz
            t = np.arange(num_samples) / sample_rate
            data = np.sin(2 * np.pi * frequency * t)

            metadata = TraceMetadata(
                sample_rate=sample_rate,
                channel_name=f"CH{i:02d}",
                source_file=f"signal_{i:03d}.wfm",
            )

            trace = WaveformTrace(data=data, metadata=metadata)
            traces.append(trace)

        return {
            "traces": traces,
            "num_files": num_files,
            "sample_rate": sample_rate,
            "duration": duration,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the parallel batch processing demonstration."""
        results: dict[str, Any] = {}

        self.section("Parallel Batch Processing Demonstration")
        self.info("Processing multiple signals in parallel with thread and process pools")

        traces = data["traces"]
        num_files = data["num_files"]

        # Part 1: Serial processing baseline
        self.subsection("Part 1: Serial Processing Baseline")
        self.info("Process all files serially to establish baseline performance.")

        start_time = time.time()
        serial_results = []

        for i, trace in enumerate(traces):
            result = self._process_signal(trace, i)
            serial_results.append(result)

        serial_time = time.time() - start_time

        self.result("Files processed", num_files)
        self.result("Serial processing time", f"{serial_time:.3f}", "seconds")
        self.result("Average time per file", f"{serial_time / num_files * 1000:.1f}", "ms")

        results["serial_time"] = serial_time
        results["serial_results"] = serial_results

        # Part 2: Thread pool processing
        self.subsection("Part 2: Thread Pool Processing")
        self.info("Process files using ThreadPoolExecutor (good for I/O bound tasks).")

        start_time = time.time()
        thread_results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_signal, trace, i): i for i, trace in enumerate(traces)
            }

            # Collect results as they complete
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                thread_results.append(result)

                if completed % 5 == 0:
                    self.info(f"  Progress: {completed}/{num_files} files completed")

        thread_time = time.time() - start_time

        self.result("Files processed", num_files)
        self.result("Thread pool time", f"{thread_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / thread_time:.2f}", "x")
        self.result("Average time per file", f"{thread_time / num_files * 1000:.1f}", "ms")

        results["thread_time"] = thread_time
        results["thread_results"] = thread_results
        results["thread_speedup"] = serial_time / thread_time

        # Part 3: Process pool processing
        self.subsection("Part 3: Process Pool Processing")
        self.info("Process files using ProcessPoolExecutor (good for CPU-bound tasks).")

        start_time = time.time()
        process_results = []

        with ProcessPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_signal_standalone, trace, i): i
                for i, trace in enumerate(traces)
            }

            # Collect results with progress tracking
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                process_results.append(result)
                completed += 1

                if completed % 5 == 0:
                    self.info(f"  Progress: {completed}/{num_files} files completed")

        process_time = time.time() - start_time

        self.result("Files processed", num_files)
        self.result("Process pool time", f"{process_time:.3f}", "seconds")
        self.result("Speedup vs serial", f"{serial_time / process_time:.2f}", "x")
        self.result("Average time per file", f"{process_time / num_files * 1000:.1f}", "ms")

        results["process_time"] = process_time
        results["process_results"] = process_results
        results["process_speedup"] = serial_time / process_time

        # Part 4: Result aggregation
        self.subsection("Part 4: Result Aggregation")
        self.info("Aggregate results from all parallel workers.")

        # Aggregate statistics across all results
        all_rms_values = [r["rms"] for r in process_results]
        all_peak_values = [r["peak"] for r in process_results]
        all_frequencies = [r["dominant_frequency"] for r in process_results]

        aggregated = {
            "total_files": len(process_results),
            "mean_rms": float(np.mean(all_rms_values)),
            "std_rms": float(np.std(all_rms_values)),
            "mean_peak": float(np.mean(all_peak_values)),
            "std_peak": float(np.std(all_peak_values)),
            "frequency_range": (float(np.min(all_frequencies)), float(np.max(all_frequencies))),
        }

        self.result("Total files processed", aggregated["total_files"])
        self.result("Mean RMS", f"{aggregated['mean_rms']:.6f}", "V")
        self.result("Std RMS", f"{aggregated['std_rms']:.6f}", "V")
        self.result("Mean peak", f"{aggregated['mean_peak']:.6f}", "V")
        self.result("Std peak", f"{aggregated['std_peak']:.6f}", "V")
        self.result(
            "Frequency range",
            f"{aggregated['frequency_range'][0]:.1f} - {aggregated['frequency_range'][1]:.1f}",
            "Hz",
        )

        results["aggregated"] = aggregated

        # Part 5: Error handling
        self.subsection("Part 5: Error Handling in Parallel Processing")
        self.info("Demonstrate error handling with graceful degradation.")

        # Create a mix of good and bad traces
        error_traces = traces[:5]  # Good traces
        error_traces.append(None)  # Bad trace (will cause error)

        errors_encountered = 0
        successful_results = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._process_signal_with_error_handling, trace, i): i
                for i, trace in enumerate(error_traces)
            }

            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "error":
                    errors_encountered += 1
                    self.warning(f"Error processing file {result['file_id']}: {result['error']}")
                else:
                    successful_results.append(result)

        self.result("Total files attempted", len(error_traces))
        self.result("Successful", len(successful_results))
        self.result("Errors", errors_encountered)

        results["error_handling"] = {
            "total": len(error_traces),
            "successful": len(successful_results),
            "errors": errors_encountered,
        }

        self.success("Parallel batch processing demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating parallel batch processing...")

        # Validate timing results exist
        if not all(key in results for key in ["serial_time", "thread_time", "process_time"]):
            self.error("Missing timing results")
            return False

        # Validate speedups are reasonable
        if results["thread_speedup"] < 0.5:
            self.warning(
                f"Thread speedup ({results['thread_speedup']:.2f}x) is lower than expected"
            )

        if results["process_speedup"] < 0.5:
            self.warning(
                f"Process speedup ({results['process_speedup']:.2f}x) is lower than expected"
            )

        # Validate result counts
        serial_count = len(results["serial_results"])
        thread_count = len(results["thread_results"])
        process_count = len(results["process_results"])

        if not (serial_count == thread_count == process_count):
            self.error(
                f"Result counts don't match: serial={serial_count}, "
                f"thread={thread_count}, process={process_count}"
            )
            return False

        # Validate aggregated results
        if "aggregated" not in results:
            self.error("Missing aggregated results")
            return False

        agg = results["aggregated"]
        if agg["total_files"] != process_count:
            self.error(f"Aggregated count mismatch: {agg['total_files']} != {process_count}")
            return False

        # Validate error handling
        if "error_handling" not in results:
            self.error("Missing error handling results")
            return False

        error_stats = results["error_handling"]
        if error_stats["errors"] != 1:
            self.error(f"Expected 1 error, got {error_stats['errors']}")
            return False

        self.success("All parallel batch processing validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - ThreadPoolExecutor: Good for I/O-bound tasks (file loading)")
        self.info("  - ProcessPoolExecutor: Good for CPU-bound tasks (signal processing)")
        self.info("  - Use as_completed() for real-time progress tracking")
        self.info("  - Always implement error handling for robust batch processing")
        self.info("  - Aggregate results after parallel processing completes")

        return True

    def _process_signal(self, trace: WaveformTrace, file_id: int) -> dict[str, Any]:
        """Process a single signal (CPU-bound simulation).

        Args:
            trace: Waveform trace to process
            file_id: File identifier

        Returns:
            Dictionary with processing results
        """
        # Simulate CPU-bound processing
        time.sleep(0.05)  # Simulate processing time

        # Calculate measurements
        rms = float(np.sqrt(np.mean(trace.data**2)))
        peak = float(np.max(np.abs(trace.data)))

        # Simple FFT for dominant frequency
        fft = np.fft.rfft(trace.data)
        freqs = np.fft.rfftfreq(len(trace.data), 1 / trace.metadata.sample_rate)
        dominant_freq = float(freqs[np.argmax(np.abs(fft))])

        return {
            "file_id": file_id,
            "channel": trace.metadata.channel_name,
            "rms": rms,
            "peak": peak,
            "dominant_frequency": dominant_freq,
            "num_samples": len(trace.data),
        }

    def _process_signal_with_error_handling(
        self, trace: WaveformTrace | None, file_id: int
    ) -> dict[str, Any]:
        """Process signal with error handling.

        Args:
            trace: Waveform trace to process (may be None)
            file_id: File identifier

        Returns:
            Dictionary with processing results or error info
        """
        try:
            if trace is None:
                raise ValueError("Invalid trace: None")

            return {
                "status": "success",
                "file_id": file_id,
                **self._process_signal(trace, file_id),
            }

        except Exception as e:
            return {
                "status": "error",
                "file_id": file_id,
                "error": str(e),
            }


def _process_signal_standalone(trace: WaveformTrace, file_id: int) -> dict[str, Any]:
    """Standalone function for process pool (must be picklable).

    Args:
        trace: Waveform trace to process
        file_id: File identifier

    Returns:
        Dictionary with processing results
    """
    # Simulate CPU-bound processing
    time.sleep(0.05)

    rms = float(np.sqrt(np.mean(trace.data**2)))
    peak = float(np.max(np.abs(trace.data)))

    # Simple FFT
    fft = np.fft.rfft(trace.data)
    freqs = np.fft.rfftfreq(len(trace.data), 1 / trace.metadata.sample_rate)
    dominant_freq = float(freqs[np.argmax(np.abs(fft))])

    return {
        "file_id": file_id,
        "channel": trace.metadata.channel_name,
        "rms": rms,
        "peak": peak,
        "dominant_frequency": dominant_freq,
        "num_samples": len(trace.data),
    }


if __name__ == "__main__":
    demo: ParallelBatchDemo = ParallelBatchDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
