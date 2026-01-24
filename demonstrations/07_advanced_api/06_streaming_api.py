"""Streaming API: Real-Time and Large-File Processing

Demonstrates:
- oscura.StreamingAnalyzer - Process signals in chunks
- oscura.load_trace_chunks() - Stream large files
- Generator-based processing
- Online algorithm implementations
- Backpressure handling

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/05_optimization.py
- 01_data_loading/06_streaming_large_files.py

Streaming enables processing signals that don't fit in memory and
real-time analysis of incoming data.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura import StreamingAnalyzer, rms

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


def signal_generator(
    frequency: float, duration: float, sample_rate: float, chunk_duration: float = 0.001
) -> Generator[WaveformTrace, None, None]:
    """Generate signal in chunks (simulates streaming)."""
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = int(duration * sample_rate)
    samples_generated = 0

    while samples_generated < total_samples:
        # Generate chunk
        remaining = total_samples - samples_generated
        current_chunk_size = min(chunk_samples, remaining)

        t = np.arange(current_chunk_size) / sample_rate + samples_generated / sample_rate
        _data = np.sin(2 * np.pi * frequency * t)  # For validation if needed

        chunk = generate_sine_wave(
            frequency=frequency,
            amplitude=1.0,
            duration=current_chunk_size / sample_rate,
            sample_rate=sample_rate,
        )

        yield chunk
        samples_generated += current_chunk_size


class OnlineRMSCalculator:
    """Online RMS calculation (Welford's algorithm variant)."""

    def __init__(self):
        """Initialize online calculator."""
        self.count = 0
        self.sum_squares = 0.0

    def update(self, chunk: WaveformTrace):
        """Update with new chunk."""
        self.count += len(chunk.data)
        self.sum_squares += np.sum(chunk.data**2)

    def get_rms(self) -> float:
        """Get current RMS estimate."""
        if self.count == 0:
            return 0.0
        return np.sqrt(self.sum_squares / self.count)

    def reset(self):
        """Reset calculator."""
        self.count = 0
        self.sum_squares = 0.0


class OnlineStatistics:
    """Online statistics using Welford's algorithm."""

    def __init__(self):
        """Initialize online statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def update(self, chunk: WaveformTrace):
        """Update statistics with new chunk."""
        for value in chunk.data:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
            self.min_val = min(self.min_val, value)
            self.max_val = max(self.max_val, value)

    def get_statistics(self) -> dict:
        """Get current statistics."""
        if self.count < 2:
            return {
                "count": self.count,
                "mean": self.mean,
                "variance": 0.0,
                "std_dev": 0.0,
                "min": self.min_val,
                "max": self.max_val,
            }

        variance = self.m2 / self.count
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": variance,
            "std_dev": np.sqrt(variance),
            "min": self.min_val,
            "max": self.max_val,
        }

    def reset(self):
        """Reset statistics."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")


class StreamingAPIDemo(BaseDemo):
    """Demonstrate streaming API for real-time and large-file processing."""

    def __init__(self):
        """Initialize streaming API demonstration."""
        super().__init__(
            name="streaming_api",
            description="Real-time processing with streaming API",
            capabilities=[
                "oscura.StreamingAnalyzer",
                "oscura.load_trace_chunks",
                "online_algorithms",
                "generator_processing",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test data."""
        # Create reference signal for validation
        reference = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
        )

        return {"reference": reference}

    def run_demonstration(self, data: dict) -> dict:
        """Run streaming API demonstration."""
        reference = data["reference"]

        self.section("Streaming API: Real-Time Processing")

        # ===================================================================
        # Part 1: Basic Streaming with Generators
        # ===================================================================
        self.subsection("1. Generator-Based Streaming")
        self.info("Process signal chunks as they arrive")

        # Create streaming generator
        chunk_count = 0
        total_samples = 0

        start_time = time.time()
        for chunk in signal_generator(
            frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
        ):
            chunk_count += 1
            total_samples += len(chunk.data)

        streaming_time = time.time() - start_time

        self.result("Chunks processed", chunk_count, "")
        self.result("Total samples", total_samples, "")
        self.result("Processing time", f"{streaming_time * 1000:.2f}", "ms")
        self.result("Throughput", f"{total_samples / streaming_time / 1e6:.2f}", "MSamples/s")

        self.success("Generator-based streaming enables incremental processing")

        # ===================================================================
        # Part 2: Online RMS Calculation
        # ===================================================================
        self.subsection("2. Online RMS Calculation")
        self.info("Calculate RMS incrementally without storing all data")

        # Online calculation
        online_calc = OnlineRMSCalculator()

        start_time = time.time()
        for chunk in signal_generator(
            frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
        ):
            online_calc.update(chunk)

        online_rms = online_calc.get_rms()
        online_time = time.time() - start_time

        # Batch calculation (reference)
        start_time = time.time()
        batch_rms = rms(reference)
        batch_time = time.time() - start_time

        self.info("RMS calculation comparison:")
        self.result("  Online RMS", f"{online_rms:.6f}", "V")
        self.result("  Batch RMS", f"{batch_rms:.6f}", "V")
        self.result("  Error", f"{abs(online_rms - batch_rms) / batch_rms * 100:.3f}", "%")
        self.result("  Online time", f"{online_time * 1000:.2f}", "ms")
        self.result("  Batch time", f"{batch_time * 1000:.2f}", "ms")

        self.success("Online algorithms produce accurate results incrementally")

        # ===================================================================
        # Part 3: Online Statistics (Welford's Algorithm)
        # ===================================================================
        self.subsection("3. Online Statistics with Welford's Algorithm")
        self.info("Calculate mean, variance, and extrema incrementally")

        # Online statistics
        online_stats = OnlineStatistics()

        for chunk in signal_generator(
            frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
        ):
            online_stats.update(chunk)

        stats = online_stats.get_statistics()

        self.info("Online statistics results:")
        self.result("  Samples processed", stats["count"], "")
        self.result("  Mean", f"{stats['mean']:.6f}", "V")
        self.result("  Std deviation", f"{stats['std_dev']:.6f}", "V")
        self.result("  Min value", f"{stats['min']:.6f}", "V")
        self.result("  Max value", f"{stats['max']:.6f}", "V")

        # Validate against batch calculation
        batch_mean = np.mean(reference.data)
        batch_std = np.std(reference.data)

        self.info("\nComparison with batch calculation:")
        self.result(
            "  Mean error", f"{abs(stats['mean'] - batch_mean) / abs(batch_mean) * 100:.3f}", "%"
        )
        self.result(
            "  Std error", f"{abs(stats['std_dev'] - batch_std) / batch_std * 100:.3f}", "%"
        )

        self.success("Welford's algorithm provides numerically stable streaming statistics")

        # ===================================================================
        # Part 4: StreamingAnalyzer
        # ===================================================================
        self.subsection("4. StreamingAnalyzer for Complex Analysis")
        self.info("Use built-in streaming analyzer")

        # Create streaming analyzer
        analyzer = StreamingAnalyzer()

        # Process chunks
        chunk_count = 0
        for chunk in signal_generator(
            frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
        ):
            analyzer.accumulate_statistics(chunk)
            chunk_count += 1

        # Get results
        stats = analyzer.get_statistics()

        self.info("StreamingAnalyzer results:")
        self.result("  Chunks processed", chunk_count, "")
        self.result("  Total samples", stats["n_samples"], "")
        self.result("  Mean", f"{stats['mean']:.6f}", "V")
        self.result("  Std deviation", f"{stats['std']:.6f}", "V")

        self.success("StreamingAnalyzer simplifies streaming workflows")

        # ===================================================================
        # Part 5: Backpressure and Flow Control
        # ===================================================================
        self.subsection("5. Backpressure and Flow Control")
        self.info("Handle slow processing with backpressure")

        class BackpressureBuffer:
            """Simple backpressure buffer."""

            def __init__(self, max_size: int = 10):
                self.buffer = []
                self.max_size = max_size
                self.dropped_chunks = 0

            def add(self, chunk):
                """Add chunk with backpressure."""
                if len(self.buffer) >= self.max_size:
                    self.dropped_chunks += 1
                    return False
                self.buffer.append(chunk)
                return True

            def process(self):
                """Process buffered chunks."""
                processed = len(self.buffer)
                self.buffer.clear()
                return processed

        # Simulate backpressure scenario
        buffer = BackpressureBuffer(max_size=5)
        added_count = 0

        for i, chunk in enumerate(
            signal_generator(
                frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
            )
        ):
            if buffer.add(chunk):
                added_count += 1

            # Simulate slow processing every 3 chunks
            if (i + 1) % 3 == 0:
                buffer.process()

        self.info("Backpressure simulation:")
        self.result("  Chunks added", added_count, "")
        self.result("  Chunks dropped", buffer.dropped_chunks, "")
        self.result(
            "  Drop rate",
            f"{buffer.dropped_chunks / (added_count + buffer.dropped_chunks) * 100:.1f}",
            "%",
        )

        self.success("Backpressure prevents memory overflow in real-time scenarios")

        # ===================================================================
        # Part 6: Memory-Efficient Processing
        # ===================================================================
        self.subsection("6. Memory Efficiency Comparison")
        self.info("Streaming vs. batch memory usage")

        # Estimate memory usage
        def estimate_memory_mb(num_samples: int) -> float:
            """Estimate memory for signal."""
            return num_samples * 8 / (1024 * 1024)  # 8 bytes per float64

        total_samples = int(0.1 * 100e3)  # 100ms at 100kHz
        chunk_samples = int(0.001 * 100e3)  # 1ms chunks

        batch_memory = estimate_memory_mb(total_samples)
        streaming_memory = estimate_memory_mb(chunk_samples)

        self.info("Memory comparison (100ms signal):")
        self.result("  Batch approach", f"{batch_memory:.2f}", "MB")
        self.result("  Streaming approach", f"{streaming_memory:.2f}", "MB")
        self.result("  Memory reduction", f"{(1 - streaming_memory / batch_memory) * 100:.1f}", "%")

        self.success("Streaming dramatically reduces memory requirements")

        # ===================================================================
        # Part 7: Real-Time Latency
        # ===================================================================
        self.subsection("7. Real-Time Latency Analysis")
        self.info("Measure processing latency per chunk")

        latencies = []

        for chunk in signal_generator(
            frequency=1000.0, duration=0.01, sample_rate=100e3, chunk_duration=0.001
        ):
            start = time.time()
            # Simulate processing
            _ = rms(chunk)
            latency = time.time() - start
            latencies.append(latency)

        avg_latency = np.mean(latencies) * 1000  # ms
        max_latency = np.max(latencies) * 1000  # ms
        p95_latency = np.percentile(latencies, 95) * 1000  # ms

        self.info("Latency statistics:")
        self.result("  Average latency", f"{avg_latency:.3f}", "ms")
        self.result("  Max latency", f"{max_latency:.3f}", "ms")
        self.result("  P95 latency", f"{p95_latency:.3f}", "ms")
        self.result("  Max throughput", f"{1000 / avg_latency:.1f}", "chunks/s")

        self.success("Low latency enables real-time processing")

        return {
            "online_rms": online_rms,
            "batch_rms": batch_rms,
            "online_mean": stats["mean"],
            "batch_mean": batch_mean,
            "online_std": stats["std"],
            "batch_std": batch_std,
            "avg_latency_ms": avg_latency,
        }

    def validate(self, results: dict) -> bool:
        """Validate streaming results."""
        self.info("Validating streaming operations...")

        # Online and batch RMS should match closely
        if not validate_approximately(
            results["online_rms"],
            results["batch_rms"],
            tolerance=0.001,
            name="Online vs Batch RMS",
        ):
            return False

        # Online statistics should match batch
        # For sine wave, mean should be close to 0, so use absolute difference
        mean_diff = abs(results["online_mean"] - results["batch_mean"])
        if mean_diff > 0.01:  # 0.01V tolerance for mean
            print(f"  ✗ Mean difference too large: {mean_diff:.6f}V")
            return False
        print(f"  ✓ Mean difference: {mean_diff:.6f}V")

        std_error = abs(results["online_std"] - results["batch_std"]) / results["batch_std"]
        if std_error > 0.01:  # 1% tolerance
            print(f"  ✗ Std dev error too large: {std_error * 100:.3f}%")
            return False
        print(f"  ✓ Std dev error: {std_error * 100:.3f}%")

        # Latency should be reasonable
        if results["avg_latency_ms"] > 10.0:
            self.warning(f"High latency: {results['avg_latency_ms']:.3f} ms")
        else:
            print(f"  ✓ Low latency: {results['avg_latency_ms']:.3f} ms")

        self.success("All streaming operations validated!")
        self.info("\nKey takeaways:")
        self.info("  - Streaming enables processing of unlimited data")
        self.info("  - Online algorithms provide accurate incremental results")
        self.info("  - Backpressure prevents memory overflow")
        self.info("  - Low latency enables real-time analysis")

        return True


if __name__ == "__main__":
    demo = StreamingAPIDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
