"""Load and stress tests for real-time streaming module.

Tests real-time streaming performance with high data rates, large buffers,
long-running streams, and concurrent operations.

Coverage:
- High-throughput streaming (1M+ samples/sec)
- Large buffer management (100K+ samples)
- Long-duration streaming (sustained operation)
- Concurrent write/read operations
- Memory stability under load
- Performance benchmarking
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from oscura.streaming.realtime import (
    RealtimeAnalyzer,
    RealtimeBuffer,
    RealtimeConfig,
    RealtimeSource,
    RealtimeStream,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = [pytest.mark.stress, pytest.mark.slow]


class HighThroughputSource(RealtimeSource):
    """High-throughput test data source."""

    def __init__(self, chunk_size: int = 10000, freq: float = 1000.0, sample_rate: float = 1e6):
        """Initialize high-throughput source.

        Args:
            chunk_size: Number of samples per acquisition.
            freq: Frequency of generated sine wave in Hz.
            sample_rate: Sample rate in Hz.
        """
        self.chunk_size = chunk_size
        self.freq = freq
        self.sample_rate = sample_rate
        self.phase = 0.0

    def acquire(self) -> NDArray[np.float64]:
        """Acquire next chunk of samples.

        Returns:
            Array of samples.
        """
        t = np.arange(self.chunk_size) / self.sample_rate + self.phase
        data = np.sin(2 * np.pi * self.freq * t)
        self.phase = t[-1]
        return data


class BurstySource(RealtimeSource):
    """Source with bursty data patterns."""

    def __init__(self, burst_size: int = 50000, idle_count: int = 10):
        """Initialize bursty source.

        Args:
            burst_size: Size of each data burst.
            idle_count: Number of idle acquisitions between bursts.
        """
        self.burst_size = burst_size
        self.idle_count = idle_count
        self.counter = 0

    def acquire(self) -> NDArray[np.float64]:
        """Acquire with bursty pattern.

        Returns:
            Large burst or small idle chunk.
        """
        self.counter += 1
        if self.counter % (self.idle_count + 1) == 0:
            # Big burst
            return np.random.randn(self.burst_size)
        else:
            # Small idle chunk
            return np.random.randn(10)


@pytest.mark.stress
@pytest.mark.slow
class TestRealtimeBufferLoad:
    """Load tests for RealtimeBuffer with high throughput."""

    def test_buffer_high_throughput_write(self) -> None:
        """Test buffer can handle high write throughput (1M+ samples/sec)."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100000, chunk_size=10000)
        buffer = RealtimeBuffer(config)

        n_samples = 1_000_000
        chunk_size = 10000
        n_chunks = n_samples // chunk_size

        start_time = time.time()

        for _ in range(n_chunks):
            data = np.random.randn(chunk_size)
            buffer.write(data)

        elapsed = time.time() - start_time

        # Should write at least 500K samples/sec
        throughput = n_samples / elapsed
        assert throughput > 500_000, f"Write throughput too low: {throughput:.0f} samples/sec"

        stats = buffer.get_stats()
        assert stats["total_samples"] == n_samples

    def test_buffer_high_throughput_read(self) -> None:
        """Test buffer can handle high read throughput."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100000, chunk_size=10000)
        buffer = RealtimeBuffer(config)

        # Pre-fill buffer
        total_samples = 500_000
        chunk_size = 10000
        for _ in range(total_samples // chunk_size):
            buffer.write(np.random.randn(chunk_size))

        # Read back
        start_time = time.time()
        samples_read = 0

        while samples_read < total_samples:
            data = buffer.read(chunk_size, timeout=1.0)
            samples_read += len(data)

        elapsed = time.time() - start_time

        # Should read at least 500K samples/sec
        throughput = samples_read / elapsed
        assert throughput > 500_000, f"Read throughput too low: {throughput:.0f} samples/sec"

    def test_buffer_concurrent_read_write(self) -> None:
        """Test buffer handles concurrent read/write operations."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=50000, chunk_size=5000)
        buffer = RealtimeBuffer(config)

        write_count = [0]
        read_count = [0]
        errors = []
        stop_event = threading.Event()

        def writer():
            """Writer thread."""
            try:
                while not stop_event.is_set() and write_count[0] < 100:
                    data = np.random.randn(5000)
                    buffer.write(data)
                    write_count[0] += 1
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            """Reader thread."""
            try:
                while not stop_event.is_set() and read_count[0] < 100:
                    try:
                        data = buffer.read(5000, timeout=0.5)
                        if len(data) > 0:
                            read_count[0] += 1
                    except TimeoutError:
                        pass  # Expected when buffer is empty
            except Exception as e:
                errors.append(("reader", e))

        # Start threads
        write_thread = threading.Thread(target=writer)
        read_thread = threading.Thread(target=reader)

        write_thread.start()
        time.sleep(0.1)  # Let writer get ahead
        read_thread.start()

        # Run for limited time
        time.sleep(2.0)
        stop_event.set()

        write_thread.join(timeout=2.0)
        read_thread.join(timeout=2.0)

        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Should have successful operations
        assert write_count[0] > 50, f"Too few writes: {write_count[0]}"
        assert read_count[0] > 30, f"Too few reads: {read_count[0]}"

    def test_buffer_large_capacity(self) -> None:
        """Test buffer with very large capacity (100K+ samples)."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100_000, chunk_size=10000)
        buffer = RealtimeBuffer(config)

        # Fill to capacity
        total_written = 0
        chunk_size = 10000

        while total_written < config.buffer_size:
            data = np.random.randn(chunk_size)
            buffer.write(data)
            total_written += chunk_size

        stats = buffer.get_stats()
        assert stats["available"] == config.buffer_size

        # Read it all back
        data = buffer.read(config.buffer_size, timeout=1.0)
        assert len(data) == config.buffer_size

    def test_buffer_overflow_handling(self) -> None:
        """Test buffer correctly handles overflow conditions."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=1000)
        buffer = RealtimeBuffer(config)

        # Write more than capacity
        overflow_amount = 5000
        total_write = config.buffer_size + overflow_amount

        for i in range(0, total_write, 1000):
            buffer.write(np.ones(1000) * i)

        stats = buffer.get_stats()

        # Buffer should be at capacity
        assert stats["available"] == config.buffer_size

        # Overflow should be tracked
        assert stats["overflow_count"] >= overflow_amount

        # Total samples written
        assert stats["total_samples"] == total_write


@pytest.mark.stress
@pytest.mark.slow
class TestRealtimeStreamLoad:
    """Load tests for RealtimeStream with sustained operation."""

    def test_stream_sustained_operation(self) -> None:
        """Test stream can operate continuously for extended period."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=50000, chunk_size=5000)
        source = HighThroughputSource(chunk_size=5000, freq=1000.0, sample_rate=1e6)
        stream = RealtimeStream(config, source)

        stream.start()

        chunks_received = 0
        start_time = time.time()
        target_duration = 3.0  # 3 seconds of sustained operation

        for chunk in stream.iter_chunks():
            chunks_received += 1

            # Check data quality
            assert len(chunk.data) > 0
            assert not np.any(np.isnan(chunk.data))

            # Stop after target duration
            if time.time() - start_time > target_duration:
                stream.stop()
                break

        elapsed = time.time() - start_time

        # Should receive chunks continuously
        assert chunks_received > 500, f"Too few chunks: {chunks_received}"
        assert elapsed >= target_duration, f"Duration too short: {elapsed:.2f}s"

    def test_stream_high_data_rate(self) -> None:
        """Test stream with high data rate (10K samples/chunk)."""
        config = RealtimeConfig(sample_rate=10e6, buffer_size=100000, chunk_size=10000)
        source = HighThroughputSource(chunk_size=10000, freq=10000.0, sample_rate=10e6)
        stream = RealtimeStream(config, source)

        stream.start()

        total_samples = 0
        start_time = time.time()

        for chunk in stream.iter_chunks():
            total_samples += len(chunk.data)

            if total_samples >= 1_000_000:  # 1M samples
                stream.stop()
                break

        elapsed = time.time() - start_time

        # Should process at least 500K samples/sec
        throughput = total_samples / elapsed
        assert throughput > 500_000, f"Throughput too low: {throughput:.0f} samples/sec"

    def test_stream_bursty_data(self) -> None:
        """Test stream handles bursty data patterns."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100000, chunk_size=5000)
        source = BurstySource(burst_size=50000, idle_count=10)
        stream = RealtimeStream(config, source)

        stream.start()

        chunks = []
        chunk_sizes = []

        for chunk in stream.iter_chunks():
            chunks.append(chunk)
            chunk_sizes.append(len(chunk.data))

            if len(chunks) >= 50:
                stream.stop()
                break

        # Should handle bursts without errors
        assert len(chunks) >= 50
        assert max(chunk_sizes) > 0
        assert min(chunk_sizes) > 0


@pytest.mark.stress
@pytest.mark.slow
class TestRealtimeAnalyzerLoad:
    """Load tests for RealtimeAnalyzer with rolling statistics."""

    def test_analyzer_large_window(self) -> None:
        """Test analyzer with very large rolling window (50K+ samples)."""
        config = RealtimeConfig(
            sample_rate=1e6, buffer_size=100000, chunk_size=5000, window_size=50000
        )
        buffer = RealtimeBuffer(config)
        analyzer = RealtimeAnalyzer(buffer, window_size=50000)

        # Write enough data to fill window
        for _ in range(15):  # 75K samples
            data = np.random.randn(5000)
            buffer.write(data)

        # Compute statistics
        stats = analyzer.compute_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_analyzer_continuous_updates(self) -> None:
        """Test analyzer can handle continuous statistical updates."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=50000, chunk_size=5000)
        buffer = RealtimeBuffer(config)
        analyzer = RealtimeAnalyzer(buffer)

        stats_history = []

        for i in range(100):
            # Write data
            data = np.sin(2 * np.pi * 1000 * np.arange(5000) / 1e6) + np.random.randn(5000) * 0.1
            buffer.write(data)

            # Compute stats every 10 iterations
            if i % 10 == 0:
                stats = analyzer.compute_statistics()
                stats_history.append(stats)

        # Should have multiple statistical snapshots
        assert len(stats_history) >= 10

        # Stats should be reasonable for sine + noise
        final_stats = stats_history[-1]
        assert -0.5 < final_stats["mean"] < 0.5
        assert 0.3 < final_stats["std"] < 1.2

    def test_analyzer_reset_and_restart(self) -> None:
        """Test analyzer can be reset and restarted."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=20000, chunk_size=2000)
        buffer = RealtimeBuffer(config)
        analyzer = RealtimeAnalyzer(buffer)

        # First run
        for _ in range(10):
            buffer.write(np.random.randn(2000))

        stats1 = analyzer.compute_statistics()

        # Clear and restart
        buffer.clear()

        for _ in range(10):
            buffer.write(np.random.randn(2000) + 5.0)  # Different mean

        stats2 = analyzer.compute_statistics()

        # Second run should reflect new data (higher mean)
        assert stats2["mean"] > stats1["mean"] + 3.0


@pytest.mark.stress
@pytest.mark.slow
class TestRealtimeMemoryStability:
    """Memory stability tests for real-time streaming."""

    def test_buffer_memory_stability(self) -> None:
        """Test buffer doesn't leak memory over many operations."""
        import gc
        import sys

        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=1000)
        buffer = RealtimeBuffer(config)

        # Measure initial memory
        gc.collect()
        data = np.random.randn(1000)
        initial_size = sys.getsizeof(buffer._buffer)

        # Perform many write/clear cycles
        for _ in range(1000):
            buffer.write(data)
            if buffer.get_available() >= 5000:
                _ = buffer.read(5000, timeout=0.1)

        # Memory should not grow significantly
        gc.collect()
        final_size = sys.getsizeof(buffer._buffer)

        # Allow some growth but not excessive (< 2x)
        growth = final_size / max(initial_size, 1)
        assert growth < 2.0, f"Memory grew {growth:.1f}x"

    def test_stream_memory_stability(self) -> None:
        """Test stream doesn't leak memory during sustained operation."""
        import gc
        import sys

        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=1000)
        source = HighThroughputSource(chunk_size=1000)
        stream = RealtimeStream(config, source)

        gc.collect()
        initial_buffer_size = sys.getsizeof(stream._buffer._buffer)

        stream.start()

        chunks = 0
        for chunk in stream.iter_chunks():
            chunks += 1
            if chunks >= 500:
                stream.stop()
                break

        gc.collect()
        final_buffer_size = sys.getsizeof(stream._buffer._buffer)

        # Buffer size should be stable
        growth = final_buffer_size / max(initial_buffer_size, 1)
        assert growth < 2.0, f"Buffer grew {growth:.1f}x"


@pytest.mark.stress
@pytest.mark.slow
class TestRealtimePerformance:
    """Performance benchmarks for real-time streaming."""

    def test_buffer_write_performance(self) -> None:
        """Benchmark buffer write performance."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100000, chunk_size=10000)
        buffer = RealtimeBuffer(config)

        n_iterations = 1000
        chunk_size = 1000

        data = np.random.randn(chunk_size)

        start_time = time.time()
        for _ in range(n_iterations):
            buffer.write(data)

        elapsed = time.time() - start_time

        ops_per_sec = n_iterations / elapsed
        samples_per_sec = (n_iterations * chunk_size) / elapsed

        # Should achieve high write rate
        assert ops_per_sec > 10_000, f"Write ops/sec too low: {ops_per_sec:.0f}"
        assert samples_per_sec > 1_000_000, f"Write samples/sec too low: {samples_per_sec:.0f}"

    def test_buffer_read_performance(self) -> None:
        """Benchmark buffer read performance."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100000, chunk_size=10000)
        buffer = RealtimeBuffer(config)

        # Pre-fill buffer
        for _ in range(100):
            buffer.write(np.random.randn(1000))

        n_iterations = 500
        chunk_size = 100

        start_time = time.time()
        for _ in range(n_iterations):
            try:
                buffer.read(chunk_size, timeout=0.1)
            except TimeoutError:
                pass

        elapsed = time.time() - start_time

        ops_per_sec = n_iterations / elapsed

        # Should achieve high read rate
        assert ops_per_sec > 1_000, f"Read ops/sec too low: {ops_per_sec:.0f}"

    def test_analyzer_statistics_performance(self) -> None:
        """Benchmark analyzer statistics computation."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=50000, chunk_size=5000)
        buffer = RealtimeBuffer(config)
        analyzer = RealtimeAnalyzer(buffer)

        # Fill buffer
        for _ in range(10):
            buffer.write(np.random.randn(5000))

        n_iterations = 1000

        start_time = time.time()
        for _ in range(n_iterations):
            analyzer.compute_statistics()

        elapsed = time.time() - start_time

        ops_per_sec = n_iterations / elapsed

        # Should compute stats quickly
        assert ops_per_sec > 100, f"Stats computation too slow: {ops_per_sec:.0f} ops/sec"
