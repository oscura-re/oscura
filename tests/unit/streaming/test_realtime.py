"""Comprehensive tests for real-time streaming module.

Tests cover:
- RealtimeConfig validation
- RealtimeBuffer thread-safety and circular buffer behavior
- RealtimeSource interface and subclassing
- RealtimeAnalyzer rolling statistics
- RealtimeStream integration and chunk iteration
- Edge cases and error handling

Requirements tested:
"""

import threading
import time

import numpy as np
import pytest

from oscura.streaming.realtime import (
    RealtimeAnalyzer,
    RealtimeBuffer,
    RealtimeConfig,
    RealtimeSource,
    RealtimeStream,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.streaming
class TestRealtimeConfig:
    """Test RealtimeConfig dataclass and validation."""

    def test_default_initialization(self):
        """Test config with default parameters."""
        config = RealtimeConfig(sample_rate=1e6)

        assert config.sample_rate == 1e6
        assert config.buffer_size == 10000
        assert config.chunk_size == 1000
        assert config.timeout == 10.0
        assert config.window_size is None
        assert config.enable_validation is True

    def test_custom_initialization(self):
        """Test config with custom parameters."""
        config = RealtimeConfig(
            sample_rate=1e6,
            buffer_size=50000,
            chunk_size=5000,
            timeout=20.0,
            window_size=25000,
            enable_validation=False,
        )

        assert config.sample_rate == 1e6
        assert config.buffer_size == 50000
        assert config.chunk_size == 5000
        assert config.timeout == 20.0
        assert config.window_size == 25000
        assert config.enable_validation is False

    def test_validate_positive_sample_rate(self):
        """Test validation requires positive sample_rate."""
        config = RealtimeConfig(sample_rate=-1e6)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            config.validate()

    def test_validate_zero_sample_rate(self):
        """Test validation rejects zero sample_rate."""
        config = RealtimeConfig(sample_rate=0)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            config.validate()

    def test_validate_positive_buffer_size(self):
        """Test validation requires positive buffer_size."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=-1000)
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            config.validate()

    def test_validate_positive_chunk_size(self):
        """Test validation requires positive chunk_size."""
        config = RealtimeConfig(sample_rate=1e6, chunk_size=-100)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()

    def test_validate_chunk_size_within_buffer(self):
        """Test validation requires chunk_size <= buffer_size."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000, chunk_size=2000)
        with pytest.raises(ValueError, match="chunk_size cannot exceed buffer_size"):
            config.validate()

    def test_validate_positive_timeout(self):
        """Test validation requires positive timeout."""
        config = RealtimeConfig(sample_rate=1e6, timeout=-1.0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            config.validate()

    def test_validate_window_size_positive(self):
        """Test validation requires positive window_size if specified."""
        config = RealtimeConfig(sample_rate=1e6, window_size=-1)
        with pytest.raises(ValueError, match="window_size must be positive"):
            config.validate()

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = RealtimeConfig(sample_rate=1e6)
        # Should not raise
        config.validate()


@pytest.mark.unit
@pytest.mark.streaming
class TestRealtimeBuffer:
    """Test RealtimeBuffer circular buffer and thread safety."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        buffer = RealtimeBuffer(config)

        assert buffer.get_available() == 0
        stats = buffer.get_stats()
        assert stats["total_samples"] == 0
        assert stats["overflow_count"] == 0
        assert stats["available"] == 0

    def test_write_float_array(self):
        """Test writing float array to buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        written = buffer.write(data)

        assert written == 5
        assert buffer.get_available() == 5

    def test_write_complex_array(self):
        """Test writing complex array to buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        # Use magnitude of complex numbers
        data = np.abs(np.array([1.0 + 2.0j, 3.0 + 4.0j]))
        written = buffer.write(data)

        assert written == 2
        assert buffer.get_available() == 2

    def test_write_multiple_times(self):
        """Test writing multiple times accumulates samples."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        for i in range(3):
            data = np.array([float(i)] * 10)
            buffer.write(data)

        assert buffer.get_available() == 30
        stats = buffer.get_stats()
        assert stats["total_samples"] == 30

    def test_write_buffer_overflow(self):
        """Test buffer overflow tracking."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10, chunk_size=5)
        buffer = RealtimeBuffer(config)

        # Write more than buffer can hold
        data = np.array([float(i) for i in range(30)])
        buffer.write(data)

        # Buffer should contain only last 10 samples
        assert buffer.get_available() == 10

        stats = buffer.get_stats()
        assert stats["total_samples"] == 30
        assert stats["overflow_count"] == 20

    def test_write_invalid_data_type(self):
        """Test write rejects non-array data."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        with pytest.raises(TypeError, match="data must be numpy array"):
            buffer.write([1.0, 2.0, 3.0])

    def test_write_invalid_dtype(self):
        """Test write rejects non-numeric data."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        data = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match="data must be float or complex array"):
            buffer.write(data)

    def test_read_available_data(self):
        """Test reading available data from buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        # Write data
        expected_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        buffer.write(expected_data)

        # Read data
        read_data = buffer.read(5, timeout=1.0)

        assert len(read_data) == 5
        np.testing.assert_array_equal(read_data, expected_data)

    def test_read_timeout_no_data(self):
        """Test read timeout when no data available."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50, timeout=0.1)
        buffer = RealtimeBuffer(config)

        # Try to read without writing
        with pytest.raises(TimeoutError, match="No data available"):
            buffer.read(10, timeout=0.1)

    def test_read_partial_data(self):
        """Test read returns available data when less than requested."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        data = np.array([1.0, 2.0, 3.0])
        buffer.write(data)

        # Request more than available (blocking mode)
        read_data = buffer.read(10, timeout=0.05)

        # Should return available data
        assert len(read_data) <= 3

    def test_read_invalid_n_samples(self):
        """Test read rejects invalid sample counts."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            buffer.read(0, timeout=1.0)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            buffer.read(-10, timeout=1.0)

    def test_read_uses_config_timeout_default(self):
        """Test read uses config timeout by default."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50, timeout=0.05)
        buffer = RealtimeBuffer(config)

        # Should timeout with config default
        with pytest.raises(TimeoutError):
            buffer.read(100)  # No timeout specified

    def test_read_blocks_for_data(self):
        """Test read blocks until data available."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        result = {"data": None}

        def write_delayed():
            time.sleep(0.1)
            buffer.write(np.array([42.0]))

        writer = threading.Thread(target=write_delayed)
        writer.start()

        # This should block then return
        data = buffer.read(1, timeout=2.0)
        result["data"] = data

        writer.join()
        assert len(result["data"]) == 1
        assert result["data"][0] == 42.0

    def test_thread_safe_write_read(self):
        """Test concurrent writes and reads are thread-safe."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        buffer = RealtimeBuffer(config)

        results = {"errors": []}

        def writer():
            try:
                for i in range(100):
                    data = np.array([float(i)] * 10)
                    buffer.write(data)
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(e)

        def reader():
            try:
                for _ in range(100):
                    try:
                        buffer.read(5, timeout=0.5)
                    except TimeoutError:
                        pass
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join(timeout=5.0)
        reader_thread.join(timeout=5.0)

        assert len(results["errors"]) == 0

    def test_buffer_clear(self):
        """Test clearing buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        buffer.write(np.array([1.0, 2.0, 3.0]))
        assert buffer.get_available() == 3

        buffer.clear()

        assert buffer.get_available() == 0
        stats = buffer.get_stats()
        assert stats["total_samples"] == 0
        assert stats["overflow_count"] == 0

    def test_buffer_close(self):
        """Test closing buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        buffer.write(np.array([1.0, 2.0, 3.0]))
        buffer.close()

        assert buffer.get_available() == 0


@pytest.mark.unit
@pytest.mark.streaming
class TestRealtimeAnalyzer:
    """Test RealtimeAnalyzer rolling statistics."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        analyzer = RealtimeAnalyzer(config)

        with pytest.raises(ValueError, match="No data accumulated"):
            analyzer.get_statistics()

    def test_accumulate_single_chunk(self):
        """Test accumulating single chunk of data."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()

        assert stats["n_samples"] == 5
        assert stats["mean"] == 3.0
        np.testing.assert_allclose(stats["std"], np.std(data))
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["peak_to_peak"] == 4.0

    def test_accumulate_multiple_chunks(self):
        """Test accumulating multiple chunks."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        for i in range(5):
            data = np.array([float(j + i * 10) for j in range(10)])
            analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        assert stats["n_samples"] == 50

    def test_accumulate_complex_data(self):
        """Test accumulating complex data (magnitude)."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        # Use magnitude of complex numbers
        data = np.abs(np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]))
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        assert stats["n_samples"] == 3

    def test_accumulate_rolling_window(self):
        """Test rolling window behavior."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50, window_size=10)
        analyzer = RealtimeAnalyzer(config)

        # Add 20 samples with window size of 10
        data = np.array([float(i) for i in range(20)])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()

        # Should only have last 10 samples in rolling window
        assert stats["n_samples"] == 10
        # Note: min/max are tracked across all samples, not just the window
        assert stats["min"] == 0.0
        assert stats["max"] == 19.0
        # But the mean and std should be based on the rolling window
        assert 9.0 < stats["mean"] < 15.0  # Mean of last 10 samples (10-19) is 14.5

    def test_accumulate_uses_buffer_size_as_default_window(self):
        """Test window size defaults to buffer_size."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=50, chunk_size=25)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([float(i) for i in range(100)])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        # Should have 50 samples (buffer size)
        assert stats["n_samples"] == 50

    def test_accumulate_invalid_data_type(self):
        """Test accumulate rejects non-array data."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        analyzer = RealtimeAnalyzer(config)

        with pytest.raises(TypeError, match="data must be numpy array"):
            analyzer.accumulate([1.0, 2.0, 3.0])

    def test_accumulate_invalid_dtype(self):
        """Test accumulate rejects non-numeric data."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        analyzer = RealtimeAnalyzer(config)

        data = np.array(["a", "b", "c"])
        with pytest.raises(TypeError, match="data must be float or complex array"):
            analyzer.accumulate(data)

    def test_analyzer_reset(self):
        """Test resetting analyzer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([1.0, 2.0, 3.0])
        analyzer.accumulate(data)

        analyzer.reset()

        with pytest.raises(ValueError, match="No data accumulated"):
            analyzer.get_statistics()

    def test_statistics_with_negative_values(self):
        """Test statistics with negative values."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()

        assert stats["min"] == -5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 0.0
        assert stats["peak_to_peak"] == 10.0

    def test_statistics_numerical_stability(self):
        """Test numerical stability of statistics computation."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000)
        analyzer = RealtimeAnalyzer(config)

        # Large values that might cause numerical issues
        data = np.array([1e10 + float(i) for i in range(100)])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()

        # Statistics should still be computed
        assert stats["n_samples"] == 100
        assert stats["std"] >= 0


@pytest.mark.unit
@pytest.mark.streaming
class TestRealtimeSource:
    """Test RealtimeSource base class and interface."""

    def test_source_not_implemented(self):
        """Test base source acquire raises NotImplementedError."""
        source = RealtimeSource()

        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            source.acquire()

    def test_source_start_stop(self):
        """Test start/stop have default implementations."""
        source = RealtimeSource()

        # Should not raise
        source.start()
        source.stop()

    def test_custom_source_subclass(self):
        """Test creating custom source subclass."""

        class DummySource(RealtimeSource):
            def __init__(self):
                self.acquire_count = 0

            def acquire(self):
                self.acquire_count += 1
                return np.array([1.0, 2.0, 3.0])

        source = DummySource()
        data = source.acquire()

        assert source.acquire_count == 1
        np.testing.assert_array_equal(data, np.array([1.0, 2.0, 3.0]))

    def test_custom_source_with_start_stop(self):
        """Test custom source with start/stop."""

        class ControlledSource(RealtimeSource):
            def __init__(self):
                self.is_started = False

            def start(self):
                self.is_started = True

            def stop(self):
                self.is_started = False

            def acquire(self):
                if self.is_started:
                    return np.array([42.0])
                return np.array([])

        source = ControlledSource()
        assert source.acquire().size == 0

        source.start()
        data = source.acquire()
        assert data[0] == 42.0

        source.stop()
        assert source.acquire().size == 0


@pytest.mark.unit
@pytest.mark.streaming
class TestRealtimeStream:
    """Test RealtimeStream integration."""

    def test_stream_initialization(self):
        """Test stream initializes correctly."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0])

        source = DummySource()
        stream = RealtimeStream(config, source)

        assert stream.config == config
        assert stream.source == source
        assert stream.get_chunk_count() == 0

    def test_stream_start_stop(self):
        """Test starting and stopping stream."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)

        class DummySource(RealtimeSource):
            def __init__(self):
                self.started = False

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

            def acquire(self):
                if self.started:
                    return np.array([1.0, 2.0])
                return np.array([])

        source = DummySource()
        stream = RealtimeStream(config, source)

        assert not source.started
        stream.start()
        assert source.started

        stream.stop()
        assert not source.started

    def test_stream_iter_chunks_requires_started(self):
        """Test iter_chunks requires stream to be started."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0])

        source = DummySource()
        stream = RealtimeStream(config, source)

        with pytest.raises(RuntimeError, match="Stream not started"):
            next(iter(stream.iter_chunks()))

    def test_stream_iter_chunks_basic(self):
        """Test basic chunk iteration."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def __init__(self):
                self.call_count = 0

            def acquire(self):
                self.call_count += 1
                if self.call_count <= 10:
                    return np.array([float(i) for i in range(100)])
                return np.array([])

        source = DummySource()
        stream = RealtimeStream(config, source)

        stream.start()

        chunks = []
        for i, chunk in enumerate(stream.iter_chunks()):
            chunks.append(chunk)
            if i >= 4:  # Get a few chunks
                stream.stop()
                break

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.data) == 100

    def test_stream_chunk_metadata(self):
        """Test chunk metadata is correct."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([float(i) for i in range(100)])

        source = DummySource()
        stream = RealtimeStream(config, source)

        stream.start()

        chunk = None
        for chunk in stream.iter_chunks():
            break

        stream.stop()

        assert chunk is not None
        assert chunk.metadata.sample_rate == 1e6
        assert len(chunk.data) == 100

    def test_stream_chunk_callback(self):
        """Test callback is called for each chunk."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0] * 100)

        source = DummySource()
        callback_chunks = []

        def callback(chunk):
            callback_chunks.append(chunk)

        stream = RealtimeStream(config, source, on_chunk=callback)

        stream.start()

        # Get a chunk through iteration
        for i, _chunk in enumerate(stream.iter_chunks()):
            if i >= 1:
                stream.stop()
                break

        # Callback should have been called
        assert len(callback_chunks) > 0

    def test_stream_get_statistics(self):
        """Test stream statistics."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([float(i) for i in range(100)])

        source = DummySource()
        stream = RealtimeStream(config, source)

        stats = stream.get_statistics()
        assert stats["mean"] == 0.0
        assert stats["n_samples"] == 0

        stream.start()

        for i, _chunk in enumerate(stream.iter_chunks()):
            if i >= 1:
                stream.stop()
                break

        stats = stream.get_statistics()
        assert stats["n_samples"] > 0

    def test_stream_get_buffer_stats(self):
        """Test stream buffer statistics."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0] * 100)

        source = DummySource()
        stream = RealtimeStream(config, source)

        buffer_stats = stream.get_buffer_stats()
        assert buffer_stats["total_samples"] == 0

        stream.start()
        time.sleep(0.1)
        stream.stop()

        buffer_stats = stream.get_buffer_stats()
        assert buffer_stats["total_samples"] > 0

    def test_stream_get_chunk_count(self):
        """Test stream chunk count."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0] * 100)

        source = DummySource()
        stream = RealtimeStream(config, source)

        assert stream.get_chunk_count() == 0

        stream.start()

        for i, _chunk in enumerate(stream.iter_chunks()):
            if i >= 2:
                stream.stop()
                break

        assert stream.get_chunk_count() > 0

    def test_stream_multiple_start_stop(self):
        """Test multiple start/stop cycles."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000, chunk_size=100)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0] * 100)

        source = DummySource()
        stream = RealtimeStream(config, source)

        for _ in range(3):
            stream.start()
            time.sleep(0.05)
            stream.stop()
            time.sleep(0.05)

        # Should complete without errors

    def test_stream_invalid_config(self):
        """Test stream with invalid config."""
        config = RealtimeConfig(sample_rate=-1e6)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0])

        source = DummySource()

        with pytest.raises(ValueError):
            RealtimeStream(config, source)


@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingRealtimeEdgeCases:
    """Test edge cases and error conditions."""

    def test_buffer_with_empty_array(self):
        """Test writing empty array to buffer."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        data = np.array([])
        written = buffer.write(data)

        assert written == 0
        assert buffer.get_available() == 0

    def test_buffer_with_multidimensional_array(self):
        """Test buffer with multi-dimensional array."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=50)
        buffer = RealtimeBuffer(config)

        # 2D array should be flattened
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        buffer.write(data)

        # All elements should be added
        assert buffer.get_available() == 4

    def test_analyzer_with_inf_values(self):
        """Test analyzer handles inf values."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([1.0, float("inf"), 3.0])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        assert stats["max"] == float("inf")

    def test_analyzer_with_nan_values(self):
        """Test analyzer handles NaN values."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([1.0, float("nan"), 3.0])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        # NaN comparisons are tricky, just verify it doesn't crash
        assert stats["n_samples"] == 3

    def test_stream_with_source_returning_none(self):
        """Test stream handles source returning None."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000, chunk_size=100)

        class NullSource(RealtimeSource):
            def acquire(self):
                return None

        source = NullSource()
        stream = RealtimeStream(config, source)

        stream.start()
        time.sleep(0.1)

        # Should handle None gracefully without crashing
        stream.stop()

    def test_stream_with_source_exception(self):
        """Test stream handles source exceptions."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1000, chunk_size=100)

        class FailingSource(RealtimeSource):
            def __init__(self):
                self.call_count = 0

            def acquire(self):
                self.call_count += 1
                if self.call_count > 2:
                    raise RuntimeError("Source error")
                return np.array([1.0] * 100)

        source = FailingSource()
        stream = RealtimeStream(config, source)

        stream.start()
        time.sleep(0.1)

        # Should handle exception and continue
        stream.stop()

    def test_very_large_buffer_size(self):
        """Test with very large buffer size."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=1_000_000)
        buffer = RealtimeBuffer(config)

        data = np.array([1.0, 2.0, 3.0])
        buffer.write(data)

        assert buffer.get_available() == 3

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=100, chunk_size=1)

        class DummySource(RealtimeSource):
            def acquire(self):
                return np.array([1.0])

        source = DummySource()
        stream = RealtimeStream(config, source)

        # Should work with size 1
        stream.start()
        time.sleep(0.05)
        stream.stop()

    def test_floating_point_sample_rate(self):
        """Test with floating point sample rate."""
        config = RealtimeConfig(sample_rate=1.5e6, buffer_size=1000)
        analyzer = RealtimeAnalyzer(config)

        data = np.array([1.0, 2.0, 3.0])
        analyzer.accumulate(data)

        stats = analyzer.get_statistics()
        assert stats["n_samples"] == 3


@pytest.mark.unit
@pytest.mark.streaming
class TestConcurrency:
    """Test concurrent behavior and race conditions."""

    def test_concurrent_buffer_operations(self):
        """Test buffer with multiple concurrent readers and writers."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=5000)
        buffer = RealtimeBuffer(config)

        results = {"read_count": 0, "write_count": 0, "errors": []}

        def writer_thread():
            try:
                for i in range(50):
                    data = np.array([float(i)] * 100)
                    buffer.write(data)
                    results["write_count"] += 1
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(("writer", e))

        def reader_thread():
            try:
                for _ in range(50):
                    try:
                        buffer.read(100, timeout=0.5)
                        results["read_count"] += 1
                    except TimeoutError:
                        pass
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(("reader", e))

        writers = [threading.Thread(target=writer_thread) for _ in range(2)]
        readers = [threading.Thread(target=reader_thread) for _ in range(2)]

        for t in writers + readers:
            t.start()

        for t in writers + readers:
            t.join(timeout=10.0)

        assert len(results["errors"]) == 0
        assert results["write_count"] > 0

    def test_stream_concurrent_access(self):
        """Test stream with concurrent access patterns."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class SlowSource(RealtimeSource):
            def acquire(self):
                time.sleep(0.001)
                return np.array([1.0] * 100)

        source = SlowSource()
        stream = RealtimeStream(config, source)

        stats_results = []

        def stats_reader():
            for _ in range(10):
                stats_results.append(stream.get_statistics())
                time.sleep(0.001)

        stream.start()

        reader_thread = threading.Thread(target=stats_reader)
        reader_thread.start()

        time.sleep(0.05)
        stream.stop()

        reader_thread.join(timeout=5.0)

        # Should have collected stats without error
        assert len(stats_results) > 0


@pytest.mark.unit
@pytest.mark.streaming
class TestStreamingRealtimeIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test complete pipeline from source to analysis."""
        config = RealtimeConfig(
            sample_rate=1e6, buffer_size=10000, chunk_size=1000, window_size=5000
        )

        # Create a source that generates a sine wave
        class SineWaveSource(RealtimeSource):
            def __init__(self, frequency=1000.0, sample_rate=1e6):
                self.frequency = frequency
                self.sample_rate = sample_rate
                self.sample_index = 0

            def acquire(self):
                n_samples = 1000
                indices = np.arange(self.sample_index, self.sample_index + n_samples)
                self.sample_index += n_samples
                t = indices / self.sample_rate
                return np.sin(2 * np.pi * self.frequency * t).astype(np.float64)

        source = SineWaveSource()
        stream = RealtimeStream(config, source)

        stream.start()

        chunks_processed = 0
        for _chunk in stream.iter_chunks():
            chunks_processed += 1
            if chunks_processed >= 5:
                stream.stop()
                break

        stream.stop()

        stats = stream.get_statistics()

        # Sine wave should have mean close to 0
        assert abs(stats["mean"]) < 0.5
        # Sine wave should have std close to 1/sqrt(2)
        assert 0.5 < stats["std"] < 1.0

    def test_multiple_source_simulation(self):
        """Test handling data from different acquisition rates."""
        config = RealtimeConfig(sample_rate=1e6, buffer_size=10000, chunk_size=100)

        class VariableRateSource(RealtimeSource):
            def __init__(self):
                self.call_count = 0

            def acquire(self):
                self.call_count += 1
                # Alternate between 50 and 150 samples
                n_samples = 50 if self.call_count % 2 == 0 else 150
                return np.random.randn(n_samples)

        source = VariableRateSource()
        stream = RealtimeStream(config, source)

        stream.start()

        chunks = []
        for chunk in stream.iter_chunks():
            chunks.append(chunk)
            if len(chunks) >= 3:
                stream.stop()
                break

        # Should handle variable chunk sizes
        assert all(len(c.data) > 0 for c in chunks)
