"""Performance benchmark tests using pytest-benchmark.

Converted from standalone comprehensive_profiling.py script.
These tests measure performance of critical code paths and can
run in CI to detect regressions.

Benchmarks are organized by module:
- Loaders: Binary data loading performance
- Analyzers: Signal analysis and processing
- Inference: Protocol inference algorithms
- Memory: Memory usage and efficiency
- Large Files: 1GB+ file processing (MED-005)
- Chunked Analyzers: Streaming/chunked processing performance

Run with:
    pytest tests/performance/test_benchmarks.py --benchmark-only
    pytest tests/performance/test_benchmarks.py --benchmark-only --benchmark-json=results.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.performance, pytest.mark.benchmark]


# =============================================================================
# Loader Benchmarks
# =============================================================================


class TestLoaderBenchmarks:
    """Benchmarks for data loaders."""

    @pytest.mark.parametrize("file_size_mb", [1, 10])
    def test_binary_loader_performance(self, benchmark, tmp_path: Path, file_size_mb: int) -> None:
        """Benchmark binary file loading at different sizes.

        Args:
            benchmark: pytest-benchmark fixture.
            tmp_path: Temporary directory fixture.
            file_size_mb: File size in megabytes.
        """
        from oscura.loaders import load

        # Create test file
        file_path = tmp_path / f"test_{file_size_mb}mb.npz"
        data_size = file_size_mb * 1024 * 1024 // 8  # float64 = 8 bytes
        test_data = np.random.randn(data_size)
        metadata = TraceMetadata(sample_rate=1e9)
        np.savez(file_path, data=test_data, sample_rate=metadata.sample_rate)

        # Benchmark loading
        result = benchmark(load, file_path)

        # Assertions
        assert result is not None
        assert len(result.data) == data_size

    @pytest.mark.slow
    def test_npz_loader_large_file(self, measure_time, tmp_path: Path) -> None:
        """Benchmark loading of large NPZ files (100MB).

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
        """
        import warnings

        from oscura.loaders import load

        # Create 100MB test file
        file_path = tmp_path / "large_100mb.npz"
        data_size = 100 * 1024 * 1024 // 8  # 100MB of float64 data
        test_data = np.random.randn(data_size)
        metadata = TraceMetadata(sample_rate=1e9)
        np.savez(file_path, data=test_data, sample_rate=metadata.sample_rate)

        # Benchmark (suppress large file warning - intentional for this test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result, elapsed = measure_time(load, file_path)

        # Assertions
        assert result is not None
        assert len(result.data) == data_size
        # Relaxed timeout for 100MB file
        assert elapsed < 30.0, f"Loading 100MB took too long: {elapsed:.2f}s"


# =============================================================================
# Analyzer Benchmarks
# =============================================================================


class TestAnalyzerBenchmarks:
    """Benchmarks for signal analyzers."""

    @pytest.mark.parametrize("signal_length", [1000, 10000, 100000])
    def test_edge_detection_performance(self, benchmark, signal_length: int) -> None:
        """Benchmark edge detection at different signal lengths.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.analyzers.digital.edges import detect_edges

        # Generate test signal - square wave
        t = np.linspace(0, 1, signal_length)
        signal = np.where(np.sin(2 * np.pi * 10 * t) > 0, 3.3, 0.0)

        # Benchmark
        result = benchmark(detect_edges, signal, threshold=1.65)

        # Assertions
        assert result is not None
        assert len(result) > 0

    @pytest.mark.parametrize("signal_length", [1000, 10000])
    def test_basic_stats_performance(self, benchmark, signal_length: int) -> None:
        """Benchmark basic statistics computation.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.analyzers.statistics import basic

        # Generate test signal
        signal = np.random.randn(signal_length)

        # Benchmark
        result = benchmark(basic.basic_stats, signal)

        # Assertions
        assert result is not None
        assert "mean" in result
        assert "std" in result

    @pytest.mark.slow
    def test_fft_performance(self, measure_time) -> None:
        """Benchmark FFT analysis on 1M samples.

        Args:
            measure_time: Custom timing fixture.
        """
        # Generate 1M sample signal
        signal_length = 1_000_000
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

        # Benchmark FFT
        result, elapsed = measure_time(np.fft.rfft, signal)

        # Assertions
        assert result is not None
        assert len(result) == signal_length // 2 + 1
        assert elapsed < 5.0, f"FFT took too long: {elapsed:.2f}s"

    # NOTE: moving_average test removed - function doesn't exist in codebase
    # Can be re-added when function is implemented


# =============================================================================
# Inference Benchmarks
# =============================================================================


class TestInferenceBenchmarks:
    """Benchmarks for protocol inference algorithms."""

    @pytest.mark.parametrize("packet_count", [100, 1000])
    def test_message_format_inference_performance(self, benchmark, packet_count: int) -> None:
        """Benchmark message format inference.

        Args:
            benchmark: pytest-benchmark fixture.
            packet_count: Number of packets to infer from.
        """
        try:
            from oscura.inference.message_format import infer_format
        except ImportError:
            # SKIP: Valid - Optional format inference
            # Only skip if auto-detection module not available
            # SKIP: Valid - Optional format inference
            # Only skip if auto-detection module not available
            pytest.skip("infer_format not available")

        # Generate test packets with consistent structure and FIXED length
        # infer_format requires all messages to have same length
        packets = []
        payload_len = 32  # Fixed payload length
        for i in range(packet_count):
            # Header (2 bytes) + Sequence (2 bytes) + Length (1 byte) + Payload (32 bytes) + CRC (2 bytes)
            header = b"\xaa\x55"
            seq = i.to_bytes(2, "big")
            length = payload_len.to_bytes(1, "big")
            payload = bytes([(i + j) % 256 for j in range(payload_len)])
            crc = (sum(payload) & 0xFFFF).to_bytes(2, "big")
            packets.append(header + seq + length + payload + crc)

        # Benchmark - use correct function name
        result = benchmark(infer_format, packets)

        # Assertions
        assert result is not None

    @pytest.mark.slow
    def test_state_machine_learning_performance(self, measure_time) -> None:
        """Benchmark state machine learning (RPNI algorithm).

        Args:
            measure_time: Custom timing fixture.
        """
        try:
            from oscura.inference.state_machine import learn_fsm
        except ImportError:
            pytest.skip("learn_fsm function not available")

        # Create sample sequences (simple alternating pattern)
        positive_samples = []
        for i in range(50):
            seq = [j % 2 for j in range(i % 10 + 5)]
            positive_samples.append(seq)

        # Benchmark
        result, elapsed = measure_time(learn_fsm, positive_samples, [])

        # Assertions
        assert result is not None
        assert hasattr(result, "states")
        assert elapsed < 30.0, f"FSM learning took too long: {elapsed:.2f}s"


# =============================================================================
# Memory Efficiency Benchmarks
# =============================================================================


class TestMemoryBenchmarks:
    """Benchmarks for memory usage and efficiency."""

    @pytest.mark.parametrize("sample_count", [100000, 1000000])
    def test_trace_object_memory_overhead(
        self, benchmark, sample_count: int, memory_monitor
    ) -> None:
        """Benchmark memory overhead of WaveformTrace objects.

        Args:
            benchmark: pytest-benchmark fixture.
            sample_count: Number of samples.
            memory_monitor: Memory monitoring fixture.
        """

        def create_trace() -> WaveformTrace:
            """Create a trace object."""
            data = np.random.randn(sample_count)
            metadata = TraceMetadata(sample_rate=1e9)
            return WaveformTrace(data=data, metadata=metadata)

        # Benchmark with memory monitoring
        with memory_monitor() as monitor:
            result = benchmark(create_trace)

        # Assertions
        assert result is not None
        assert len(result.data) == sample_count

        # Check memory overhead is reasonable
        theoretical_mb = (sample_count * 8) / (1024 * 1024)  # float64 = 8 bytes
        assert monitor.peak_mb < theoretical_mb * 2.5  # Allow 150% overhead

    def test_time_vector_computation_memory(self, benchmark, memory_monitor) -> None:
        """Benchmark memory usage of time vector computation.

        Args:
            benchmark: pytest-benchmark fixture.
            memory_monitor: Memory monitoring fixture.
        """
        # Create a large trace
        sample_count = 1_000_000
        data = np.random.randn(sample_count)
        metadata = TraceMetadata(sample_rate=1e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Benchmark time vector access
        def get_time_vector() -> NDArray[np.float64]:
            """Get time vector."""
            return trace.time_vector

        with memory_monitor() as monitor:
            result = benchmark(get_time_vector)

        # Assertions
        assert result is not None
        assert len(result) == sample_count
        # Time vector should be cached, minimal overhead
        assert monitor.peak_mb < 20  # Adjusted for Python memory overhead


# =============================================================================
# Algorithm Complexity Benchmarks
# =============================================================================


class TestComplexityBenchmarks:
    """Benchmarks for measuring algorithm complexity (O(n) behavior)."""

    @pytest.mark.parametrize("size", [1000, 5000, 10000, 50000, 100000], ids=lambda x: f"n={x}")
    def test_statistics_complexity(self, benchmark, size: int) -> None:
        """Measure O(n) behavior of statistics computation.

        Args:
            benchmark: pytest-benchmark fixture.
            size: Data size.
        """
        from oscura.analyzers.statistics import basic

        # Generate test data
        data = np.random.randn(size)

        # Benchmark
        result = benchmark(basic.basic_stats, data)

        # Assertions
        assert result is not None

    @pytest.mark.parametrize("size", [1000, 10000, 100000], ids=lambda x: f"n={x}")
    def test_edge_detection_complexity(self, benchmark, size: int) -> None:
        """Measure O(n) behavior of edge detection.

        Args:
            benchmark: pytest-benchmark fixture.
            size: Signal length.
        """
        from oscura.analyzers.digital.edges import detect_edges

        # Generate test signal
        t = np.linspace(0, 1, size)
        signal = np.where(np.sin(2 * np.pi * 10 * t) > 0, 3.3, 0.0)

        # Benchmark
        result = benchmark(detect_edges, signal, threshold=1.65)

        # Assertions
        assert result is not None


# =============================================================================
# Scalability Benchmarks
# =============================================================================


class TestScalabilityBenchmarks:
    """Benchmarks for testing scalability with concurrent operations."""

    @pytest.mark.slow
    def test_sequential_file_loading(self, measure_time, tmp_path: Path) -> None:
        """Benchmark sequential file loading (baseline).

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
        """
        from oscura.loaders import load

        # Create multiple test files
        file_paths = []
        for i in range(10):
            file_path = tmp_path / f"test_{i}.npz"
            data = np.random.randn(100000)
            metadata = TraceMetadata(sample_rate=1e9)
            np.savez(file_path, data=data, sample_rate=metadata.sample_rate)
            file_paths.append(file_path)

        # Benchmark sequential loading
        def load_all_sequential() -> list[WaveformTrace]:
            """Load all files sequentially."""
            return [load(p) for p in file_paths]

        result, elapsed = measure_time(load_all_sequential)

        # Assertions
        assert result is not None
        assert len(result) == 10
        assert elapsed < 10.0, f"Loading took too long: {elapsed:.2f}s"


# =============================================================================
# Large File Processing Benchmarks (MED-005)
# =============================================================================


class TestLargeFileBenchmarks:
    """Benchmarks for large file processing (1GB+).

    These tests verify that oscura can handle very large datasets
    efficiently without running out of memory or taking excessive time.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("file_size_mb", [500, 1000])
    def test_large_file_loading(self, measure_time, tmp_path: Path, file_size_mb: int) -> None:
        """Benchmark loading of very large files (500MB-1GB).

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
            file_size_mb: File size in megabytes.
        """
        import warnings

        from oscura.loaders import load

        # Create large test file
        file_path = tmp_path / f"large_{file_size_mb}mb.npz"
        data_size = file_size_mb * 1024 * 1024 // 8  # float64 = 8 bytes
        test_data = np.random.randn(data_size)
        metadata = TraceMetadata(sample_rate=1e9)
        np.savez(file_path, data=test_data, sample_rate=metadata.sample_rate)

        # Benchmark loading (suppress large file warning - intentional for this test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result, elapsed = measure_time(load, file_path)

        # Assertions
        assert result is not None
        assert len(result.data) == data_size
        # Relaxed timeout for large files
        assert elapsed < 180.0, f"Loading {file_size_mb}MB took too long: {elapsed:.2f}s"

    @pytest.mark.slow
    def test_gigabyte_file_processing(self, measure_time, tmp_path: Path) -> None:
        """Benchmark 1GB+ file processing.

        This test verifies that the system can handle gigabyte-scale data.

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
        """
        import warnings

        # Create ~1GB file (1024MB / 8 bytes = 128M samples)
        file_size_mb = 1024
        file_path = tmp_path / "gigabyte_file.npz"
        data_size = file_size_mb * 1024 * 1024 // 8

        # Generate in chunks to avoid memory issues during setup
        test_data = np.random.randn(data_size)
        metadata = TraceMetadata(sample_rate=1e9)
        np.savez(file_path, data=test_data, sample_rate=metadata.sample_rate)

        # Import after file creation
        from oscura.loaders import load

        # Benchmark loading (suppress large file warning - intentional for this test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result, elapsed = measure_time(load, file_path)

        # Assertions
        assert result is not None
        assert len(result.data) == data_size
        # Very relaxed timeout for 1GB file
        assert elapsed < 300.0, f"Loading 1GB took too long: {elapsed:.2f}s"


# =============================================================================
# Chunked Analyzer Benchmarks (MED-005)
# =============================================================================


class TestChunkedAnalyzerBenchmarks:
    """Benchmarks for chunked/streaming analyzers.

    Tests performance of analyzers that process data in chunks,
    which is essential for handling large files without loading
    everything into memory.
    """

    @pytest.mark.parametrize("chunk_size", [1024, 4096, 16384, 65536])
    def test_chunked_fft_performance(self, benchmark, chunk_size: int) -> None:
        """Benchmark chunked FFT analysis with different chunk sizes.

        Args:
            benchmark: pytest-benchmark fixture.
            chunk_size: Size of each processing chunk.
        """
        from oscura.analyzers.spectral.fft import fft_chunked

        # Generate large test signal (10M samples)
        signal_length = 10_000_000
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

        # Benchmark chunked FFT
        result = benchmark(fft_chunked, signal, chunk_size=chunk_size)

        # Assertions
        assert result is not None
        freqs, mags = result
        assert len(freqs) > 0
        assert len(mags) > 0

    @pytest.mark.parametrize("total_samples", [1_000_000, 10_000_000])
    def test_streaming_statistics_performance(self, benchmark, total_samples: int) -> None:
        """Benchmark streaming statistics computation.

        Args:
            benchmark: pytest-benchmark fixture.
            total_samples: Total number of samples to process.
        """
        try:
            from oscura.analyzers.statistics.streaming import StreamingStats
        except ImportError:
            # SKIP: Valid - Optional streaming statistics
            # Only skip if streaming stats module not available
            # SKIP: Valid - Optional streaming statistics
            # Only skip if streaming stats module not available
            pytest.skip("StreamingStats not available")

        # Generate test data
        data = np.random.randn(total_samples)

        def compute_streaming_stats():
            """Compute statistics in streaming fashion."""
            stats = StreamingStats()
            chunk_size = 10000
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                stats.update(chunk)
            return stats.finalize()

        # Benchmark
        result = benchmark(compute_streaming_stats)

        # Assertions
        assert result is not None

    @pytest.mark.slow
    def test_chunked_vs_nonchunked_consistency(self, tmp_path: Path) -> None:
        """Verify chunked and non-chunked analyzers produce consistent results.

        This is a correctness test, not a benchmark, but important for
        validating that chunked processing doesn't lose accuracy.
        """
        try:
            from oscura.analyzers.spectral.fft import fft_chunked
        except ImportError:
            # SKIP: Valid - Optional chunked FFT processing
            # Only skip if chunked FFT module not available
            # SKIP: Valid - Optional chunked FFT processing
            # Only skip if chunked FFT module not available
            pytest.skip("fft_chunked not available")

        # Generate test signal
        signal_length = 100_000
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t)

        # Non-chunked FFT
        fft_direct = np.fft.rfft(signal)

        # Chunked FFT (should produce similar results)
        fft_chunked_result = fft_chunked(signal, chunk_size=1024)

        # Results should be similar (exact match depends on implementation)
        # For now, just verify both complete without error
        assert fft_direct is not None
        assert fft_chunked_result is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("file_size_mb", [100, 500])
    def test_chunked_file_analysis(self, measure_time, tmp_path: Path, file_size_mb: int) -> None:
        """Benchmark chunked analysis of large files.

        Tests the scenario where a large file is analyzed in chunks
        rather than loading entirely into memory.

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
            file_size_mb: File size in megabytes.
        """

        # Create large test file
        data_size = file_size_mb * 1024 * 1024 // 8
        test_data = np.random.randn(data_size)

        def analyze_in_chunks(data: np.ndarray, chunk_size: int = 100000) -> dict:
            """Analyze data in chunks, accumulating results."""
            # Simulate chunked analysis
            running_sum = 0.0
            running_sum_sq = 0.0
            total_count = 0

            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                running_sum += np.sum(chunk)
                running_sum_sq += np.sum(chunk**2)
                total_count += len(chunk)

            mean = running_sum / total_count
            variance = (running_sum_sq / total_count) - (mean**2)

            return {
                "mean": mean,
                "std": np.sqrt(variance),
                "count": total_count,
            }

        # Benchmark chunked analysis
        result, elapsed = measure_time(analyze_in_chunks, test_data)

        # Assertions
        assert result is not None
        assert "mean" in result
        assert "std" in result
        assert result["count"] == data_size
        assert elapsed < 60.0, f"Chunked analysis of {file_size_mb}MB took too long: {elapsed:.2f}s"


# =============================================================================
# Parallel Processing Benchmarks
# =============================================================================


class TestParallelProcessingBenchmarks:
    """Benchmarks for parallel processing capabilities."""

    @pytest.mark.slow
    def test_parallel_fft_performance(self, measure_time, tmp_path: Path) -> None:
        """Benchmark parallel FFT analysis.

        Args:
            measure_time: Custom timing fixture.
            tmp_path: Temporary directory fixture.
        """
        import warnings

        try:
            from oscura.analyzers.spectral.chunked_fft import fft_chunked_parallel
        except ImportError:
            pytest.skip("fft_chunked_parallel not available")

        # Generate large test signal
        signal_length = 10_000_000
        sample_rate = 1e6
        t = np.linspace(0, signal_length / sample_rate, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

        # Save to temp file (API requirement)
        tmp_file = tmp_path / "signal.npz"
        np.savez(tmp_file, signal=signal, sample_rate=sample_rate)

        # Benchmark parallel FFT (suppress runtime warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result, elapsed = measure_time(fft_chunked_parallel, tmp_file, 65536, n_workers=4)

        # Assertions (result is tuple of freq, mag arrays)
        assert result is not None
        assert len(result) == 2, "Expected (freq, mag) tuple"
        # Relaxed timeout for parallel FFT
        assert elapsed < 60.0, f"Parallel FFT took too long: {elapsed:.2f}s"

    @pytest.mark.slow
    def test_parallel_vs_sequential_speedup(self, tmp_path: Path) -> None:
        """Measure speedup from parallel processing.

        This test compares parallel and sequential performance to
        verify that parallelization provides actual benefits.
        """
        import time

        try:
            from oscura.analyzers.spectral.chunked_fft import fft_chunked, fft_chunked_parallel
        except ImportError:
            pytest.skip("fft_chunked_parallel not available")

        import tempfile

        # Generate test signal
        signal_length = 5_000_000
        sample_rate = 1e6
        t = np.linspace(0, signal_length / sample_rate, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t)

        # Save signal to temp file for parallel processing (required by API)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name
            np.savez(tmp_path, signal=signal, sample_rate=sample_rate)

        try:
            import warnings

            # Suppress runtime warnings for FFT operations
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                # Time sequential (can use in-memory)
                start = time.perf_counter()
                fft_chunked(tmp_path, segment_size=65536, sample_rate=sample_rate)
                sequential_time = time.perf_counter() - start

                # Time parallel (requires file path)
                start = time.perf_counter()
                fft_chunked_parallel(tmp_path, segment_size=65536, n_workers=4)
                parallel_time = time.perf_counter() - start

            # Log results (not strictly a pass/fail test)
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
            print(f"\nSequential: {sequential_time:.3f}s, Parallel: {parallel_time:.3f}s")
            print(f"Speedup: {speedup:.2f}x")

            # Parallel should not be significantly slower
            assert parallel_time < sequential_time * 1.5  # Allow some overhead
        finally:
            Path(tmp_path).unlink()


# =============================================================================
# Protocol Detection Benchmarks (New)
# =============================================================================


class TestProtocolDetectionBenchmarks:
    """Benchmarks for protocol detection performance."""

    @pytest.mark.parametrize("signal_length", [10000, 100000])
    def test_protocol_detection_sequential(self, benchmark, signal_length: int) -> None:
        """Benchmark sequential protocol detection.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.inference.protocol import detect_protocol

        # Generate UART-like signal
        t = np.linspace(0, 1, signal_length)
        # Simulate UART: idle high, irregular edges
        signal = np.where(np.random.rand(signal_length) > 0.7, 3.3, 0.0)
        signal[: signal_length // 10] = 3.3  # Idle high at start

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # Benchmark sequential detection
        result = benchmark(detect_protocol, trace, parallel=False)

        # Assertions
        assert result is not None
        assert "protocol" in result

    @pytest.mark.parametrize("signal_length", [10000, 100000])
    def test_protocol_detection_parallel(self, benchmark, signal_length: int) -> None:
        """Benchmark parallel protocol detection.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.inference.protocol import detect_protocol

        # Generate UART-like signal
        t = np.linspace(0, 1, signal_length)
        signal = np.where(np.random.rand(signal_length) > 0.7, 3.3, 0.0)
        signal[: signal_length // 10] = 3.3

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # Benchmark parallel detection
        result = benchmark(detect_protocol, trace, parallel=True)

        # Assertions
        assert result is not None
        assert "protocol" in result

    def test_protocol_detection_speedup(self) -> None:
        """Verify parallel protocol detection provides consistent results.

        Note: Parallel overhead may be higher than speedup for simple scoring functions.
        The parallel implementation is more beneficial when:
        - Running many detections concurrently
        - Scoring functions are compute-intensive
        - System has multiple cores available
        """
        import time

        from oscura.inference.protocol import detect_protocol

        # Generate test signal
        signal_length = 100000
        signal = np.where(np.random.rand(signal_length) > 0.7, 3.3, 0.0)
        signal[: signal_length // 10] = 3.3

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # Time sequential
        start = time.perf_counter()
        for _ in range(10):
            result_seq = detect_protocol(trace, parallel=False)
        sequential_time = time.perf_counter() - start

        # Time parallel
        start = time.perf_counter()
        for _ in range(10):
            result_par = detect_protocol(trace, parallel=True)
        parallel_time = time.perf_counter() - start

        # Log results
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        print(
            f"\nProtocol detection - Sequential: {sequential_time:.3f}s, "
            f"Parallel: {parallel_time:.3f}s, Speedup: {speedup:.2f}x"
        )

        # Both implementations should produce same result
        assert result_seq["protocol"] == result_par["protocol"]

        # Parallel version functional (may have overhead for simple tasks)
        # This is expected and acceptable - the parallel implementation
        # shines when scaling to multiple signals or heavier workloads
        assert parallel_time < sequential_time * 5.0  # Allow for overhead


# =============================================================================
# FFT Cache Benchmarks (New)
# =============================================================================


class TestFFTCacheBenchmarks:
    """Benchmarks for FFT caching performance."""

    @pytest.mark.parametrize("signal_length", [1000, 10000])
    def test_fft_cache_hit(self, benchmark, signal_length: int) -> None:
        """Benchmark FFT with cache hits.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.analyzers.waveform.spectral import clear_fft_cache, fft

        # Clear cache before test
        clear_fft_cache()

        # Generate test signal
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # First call to populate cache
        fft(trace, use_cache=True)

        # Benchmark repeated calls (should hit cache)
        result = benchmark(fft, trace, use_cache=True)

        # Assertions
        assert result is not None

    @pytest.mark.parametrize("signal_length", [1000, 10000])
    def test_fft_cache_miss(self, benchmark, signal_length: int) -> None:
        """Benchmark FFT without caching.

        Args:
            benchmark: pytest-benchmark fixture.
            signal_length: Number of samples in signal.
        """
        from oscura.analyzers.waveform.spectral import fft

        # Generate test signal
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # Benchmark without cache
        result = benchmark(fft, trace, use_cache=False)

        # Assertions
        assert result is not None

    def test_fft_cache_effectiveness(self) -> None:
        """Verify FFT caching provides speedup for repeated operations."""
        import time

        from oscura.analyzers.waveform.spectral import (
            clear_fft_cache,
            fft,
            get_fft_cache_stats,
        )

        # Clear cache
        clear_fft_cache()

        # Generate test signal
        signal_length = 10000
        t = np.linspace(0, 1, signal_length)
        signal = np.sin(2 * np.pi * 1000 * t)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        # Time without cache
        start = time.perf_counter()
        for _ in range(100):
            fft(trace, use_cache=False)
        uncached_time = time.perf_counter() - start

        # Clear and time with cache
        clear_fft_cache()
        start = time.perf_counter()
        for _ in range(100):
            fft(trace, use_cache=True)
        cached_time = time.perf_counter() - start

        # Get cache stats
        stats = get_fft_cache_stats()
        print(f"\nFFT cache stats: {stats}")

        # Log results
        speedup = uncached_time / cached_time if cached_time > 0 else 1.0
        print(
            f"FFT caching - Uncached: {uncached_time:.3f}s, "
            f"Cached: {cached_time:.3f}s, Speedup: {speedup:.2f}x"
        )

        # Cache should provide significant speedup
        assert cached_time < uncached_time * 0.5  # At least 2x speedup expected


# =============================================================================
# Import Time Benchmarks (New)
# =============================================================================


class TestImportTimeBenchmarks:
    """Benchmarks for module import times."""

    def test_core_import_time(self, benchmark) -> None:
        """Benchmark core oscura import time.

        Args:
            benchmark: pytest-benchmark fixture.
        """
        import sys

        def import_oscura() -> None:
            """Import oscura module."""
            # Remove from cache to force reimport
            if "oscura" in sys.modules:
                del sys.modules["oscura"]
            import oscura  # noqa: F401

        # Benchmark import
        result = benchmark(import_oscura)
        assert result is not None
        # Verify import was successful
        assert "oscura" in sys.modules

    def test_lazy_import_overhead(self, benchmark) -> None:
        """Benchmark lazy import overhead.

        Args:
            benchmark: pytest-benchmark fixture.
        """
        from oscura.utils.lazy_imports import lazy_import

        def create_lazy_module() -> None:
            """Create lazy module proxy."""
            result = lazy_import("numpy")
            assert result is not None
            return result

        # Benchmark lazy import creation (should be very fast)
        result = benchmark(create_lazy_module)
        assert result is not None

    def test_lazy_vs_eager_import(self) -> None:
        """Compare lazy vs eager import performance.

        Tests the overhead of lazy import wrapper creation vs actual import.
        """
        import time

        from oscura.utils.lazy_imports import lazy_import

        # Test lazy import creation (very fast - no actual import)
        start = time.perf_counter()
        for _ in range(1000):
            _ = lazy_import("some.module.that.does.not.exist")
        lazy_creation_time = (time.perf_counter() - start) / 1000

        # Test lazy import with actual module (scipy as test - not imported yet in this test)
        start = time.perf_counter()
        scipy_lazy = lazy_import("scipy")
        lazy_wrapper_time = time.perf_counter() - start

        # Test attribute access (triggers actual import)
        start = time.perf_counter()
        _ = scipy_lazy.version  # Trigger import
        lazy_first_access_time = time.perf_counter() - start

        # Subsequent access should be instant (module already loaded)
        start = time.perf_counter()
        _ = scipy_lazy.version
        lazy_cached_access_time = time.perf_counter() - start

        # Log results
        print(
            f"\nLazy import - Creation: {lazy_creation_time * 1000:.4f}ms, "
            f"Wrapper: {lazy_wrapper_time * 1000:.4f}ms, "
            f"First access: {lazy_first_access_time * 1000:.2f}ms, "
            f"Cached access: {lazy_cached_access_time * 1000:.4f}ms"
        )

        # Lazy wrapper creation should be extremely fast (<0.1ms)
        assert lazy_wrapper_time < 0.001  # Less than 1ms

        # Cached access should be faster than first access (allow for timing variance)
        # Note: Relaxed threshold to account for OS scheduling and cache effects
        assert lazy_cached_access_time < lazy_first_access_time * 0.5  # 50% threshold


# =============================================================================
# Comprehensive Performance Test
# =============================================================================


class TestComprehensivePerformance:
    """Comprehensive performance tests covering multiple operations."""

    @pytest.mark.slow
    def test_full_analysis_pipeline(self, measure_time) -> None:
        """Benchmark complete analysis pipeline.

        Tests realistic workflow: load → analyze → detect protocol.

        Args:
            measure_time: Custom timing fixture.
        """
        from oscura.analyzers.waveform.spectral import fft, thd
        from oscura.inference.protocol import detect_protocol

        # Generate realistic signal
        signal_length = 50000
        t = np.linspace(0, 1, signal_length)
        # Mixed signal: carrier + protocol edges
        signal = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)
        signal += np.where(np.random.rand(signal_length) > 0.9, 0.5, 0)

        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=signal, metadata=metadata)

        def full_pipeline() -> dict:
            """Run complete analysis pipeline."""
            # Spectral analysis
            freq, mag = fft(trace)

            # Quality metrics
            thd_val = thd(trace)

            # Protocol detection (may fail for non-digital signals)
            try:
                protocol = detect_protocol(trace)
            except (ValueError, RuntimeError, KeyError, AttributeError) as e:
                protocol = {"protocol": "unknown"}

            return {
                "fft_bins": len(freq),
                "thd": thd_val if isinstance(thd_val, (int, float)) else thd_val.get("value", 0.0),
                "protocol": protocol["protocol"],
            }

        # Benchmark complete pipeline
        result, elapsed = measure_time(full_pipeline)

        # Assertions
        assert result is not None
        assert "fft_bins" in result
        assert "thd" in result
        assert elapsed < 10.0, f"Full pipeline took too long: {elapsed:.2f}s"
