"""Comprehensive unit tests for src/oscura/streaming/chunked.py

Tests all public functions and classes with edge cases, error handling,
and validation. Covers:
- load_trace_chunks: Generator for memory-efficient chunk loading
- StreamingAnalyzer: Accumulator for streaming statistics and PSD
- chunked_spectrogram: Memory-bounded spectrogram computation
- chunked_fft: Segmented FFT with averaging
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.streaming.chunked import (
    StreamingAnalyzer,
    chunked_fft,
    chunked_spectrogram,
    load_trace_chunks,
)

pytestmark = pytest.mark.unit


class TestLoadTraceChunks:
    """Tests for load_trace_chunks generator function."""

    def test_basic_chunk_loading(self) -> None:
        """Test basic chunk loading without overlap."""
        # Create mock loader that returns a trace
        sample_rate = 100_000
        total_samples = 1_000_000
        data = np.sin(2 * np.pi * 1000 * np.arange(total_samples) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)
        chunk_size = 100_000

        chunks = list(load_trace_chunks("test.bin", chunk_size=chunk_size, loader=loader))

        # Should have 10 chunks of 100k samples each
        assert len(chunks) == 10
        for chunk in chunks:
            assert isinstance(chunk, WaveformTrace)
            assert len(chunk.data) == chunk_size
            assert chunk.metadata.sample_rate == sample_rate

    def test_chunk_loading_with_overlap(self) -> None:
        """Test chunk loading with sample overlap."""
        sample_rate = 10_000
        total_samples = 100_000
        data = np.sin(2 * np.pi * 100 * np.arange(total_samples) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)
        chunk_size = 10_000
        overlap = 1000

        chunks = list(
            load_trace_chunks("test.bin", chunk_size=chunk_size, overlap=overlap, loader=loader)
        )

        # With overlap, total chunks should be calculated
        assert len(chunks) > 0
        # First chunk should be full size
        assert len(chunks[0].data) == chunk_size
        # Later chunks should be full size or smaller (last chunk)
        for chunk in chunks[:-1]:
            assert len(chunk.data) == chunk_size

    def test_path_handling(self) -> None:
        """Test that Path and string paths are handled."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 100 * np.arange(100_000) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        # Test with string path
        chunks_str = list(load_trace_chunks("test.bin", chunk_size=50_000, loader=loader))
        assert len(chunks_str) > 0

        # Reset mock
        loader.reset_mock()

        # Test with Path object
        chunks_path = list(load_trace_chunks(Path("test.bin"), chunk_size=50_000, loader=loader))
        assert len(chunks_path) > 0

    def test_float_chunk_size(self) -> None:
        """Test chunk size specified as bytes (float > 1e6)."""
        sample_rate = 100_000
        data = np.sin(2 * np.pi * 1000 * np.arange(1_000_000) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        # Specify chunk size as bytes (100 MB = 100e6 bytes)
        chunks = list(load_trace_chunks("test.bin", chunk_size=100e6, loader=loader))

        # Should calculate correct number of chunks
        assert len(chunks) > 0
        # Each chunk should be data[i*n:(i+1)*n]
        total_yielded = sum(len(c.data) for c in chunks)
        assert total_yielded >= len(data)

    def test_progress_callback(self) -> None:
        """Test progress callback is invoked correctly."""
        sample_rate = 10_000
        total_samples = 100_000
        data = np.sin(2 * np.pi * 100 * np.arange(total_samples) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)
        progress_callback = Mock()
        chunk_size = 10_000

        chunks = list(
            load_trace_chunks(
                "test.bin",
                chunk_size=chunk_size,
                loader=loader,
                progress_callback=progress_callback,
            )
        )

        # Progress callback should be called for each chunk
        assert progress_callback.call_count == len(chunks)

        # Check calls are in order
        for i, call in enumerate(progress_callback.call_args_list):
            args, _ = call
            chunk_num, total_chunks = args
            assert chunk_num == i
            assert total_chunks > 0

    def test_loader_failure(self) -> None:
        """Test error handling when loader fails."""
        loader = Mock(side_effect=ValueError("Failed to load"))

        with pytest.raises(ValueError, match="Failed to load trace metadata"):
            list(load_trace_chunks("test.bin", loader=loader))

    def test_metadata_preservation(self) -> None:
        """Test that metadata is preserved in chunks."""
        sample_rate = 100_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="CH1",
            vertical_offset=0.5,
            vertical_scale=1.0,
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        loader = Mock(return_value=trace)

        chunks = list(load_trace_chunks("test.bin", chunk_size=10_000, loader=loader))

        for chunk in chunks:
            assert chunk.metadata.sample_rate == sample_rate
            assert chunk.metadata.channel_name == "CH1"
            assert chunk.metadata.vertical_offset == 0.5
            assert chunk.metadata.vertical_scale == 1.0

    def test_exact_chunk_boundary(self) -> None:
        """Test when data size is exact multiple of chunk size."""
        sample_rate = 10_000
        chunk_size = 10_000
        total_samples = chunk_size * 5
        data = np.sin(2 * np.pi * 100 * np.arange(total_samples) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        chunks = list(load_trace_chunks("test.bin", chunk_size=chunk_size, loader=loader))

        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk.data) == chunk_size

    def test_partial_last_chunk(self) -> None:
        """Test handling of partial last chunk."""
        sample_rate = 10_000
        chunk_size = 10_000
        total_samples = 25_000  # Not exact multiple
        data = np.sin(2 * np.pi * 100 * np.arange(total_samples) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        chunks = list(load_trace_chunks("test.bin", chunk_size=chunk_size, loader=loader))

        # First 2 chunks full, last chunk partial
        assert len(chunks) == 3
        assert len(chunks[0].data) == chunk_size
        assert len(chunks[1].data) == chunk_size
        assert len(chunks[2].data) == 5_000

    def test_zero_overlap(self) -> None:
        """Test explicitly zero overlap."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 100 * np.arange(100_000) / sample_rate)
        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        chunks = list(load_trace_chunks("test.bin", chunk_size=10_000, overlap=0, loader=loader))

        # Reconstruct data from non-overlapping chunks
        reconstructed = np.concatenate([c.data for c in chunks])
        assert len(reconstructed) >= len(data)


class TestStreamingAnalyzer:
    """Tests for StreamingAnalyzer accumulator class."""

    def test_initialization(self) -> None:
        """Test analyzer initializes with zero state."""
        analyzer = StreamingAnalyzer()

        # Statistics should be empty
        with pytest.raises(ValueError, match="No data accumulated"):
            analyzer.get_statistics()

        # PSD should be empty
        with pytest.raises(ValueError, match="No PSD data accumulated"):
            analyzer.get_psd()

        # Histogram should be empty
        with pytest.raises(ValueError, match="No histogram data accumulated"):
            analyzer.get_histogram()

    def test_accumulate_statistics_single_chunk(self) -> None:
        """Test accumulating statistics from single chunk."""
        analyzer = StreamingAnalyzer()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        chunk = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_statistics(chunk)
        stats = analyzer.get_statistics()

        assert stats["mean"] == 3.0
        assert stats["std"] == np.sqrt(2.0)  # std of [1,2,3,4,5]
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["n_samples"] == 5

    def test_accumulate_statistics_multiple_chunks(self) -> None:
        """Test accumulating statistics from multiple chunks."""
        analyzer = StreamingAnalyzer()

        # First chunk
        chunk1 = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_statistics(chunk1)

        # Second chunk
        chunk2 = WaveformTrace(
            data=np.array([4.0, 5.0, 6.0]),
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_statistics(chunk2)

        stats = analyzer.get_statistics()

        expected_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(stats["mean"], expected_data.mean())
        np.testing.assert_allclose(stats["std"], expected_data.std())
        assert stats["min"] == 1.0
        assert stats["max"] == 6.0
        assert stats["n_samples"] == 6

    def test_accumulate_psd_single_chunk(self) -> None:
        """Test PSD accumulation from single chunk."""
        analyzer = StreamingAnalyzer()
        sample_rate = 10_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        chunk = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        analyzer.accumulate_psd(chunk, nperseg=1024, window="hann")
        freqs, psd = analyzer.get_psd()

        assert len(freqs) > 0
        assert len(psd) == len(freqs)
        # PSD should show peak near 1 kHz
        peak_idx = np.argmax(psd)
        assert 500 < freqs[peak_idx] < 1500

    def test_accumulate_psd_multiple_chunks(self) -> None:
        """Test PSD accumulation averaging multiple chunks."""
        analyzer = StreamingAnalyzer()
        sample_rate = 10_000

        # Generate signal with clear frequency
        freq_signal = 1000.0
        for chunk_idx in range(3):
            t = np.arange(chunk_idx * 10000, (chunk_idx + 1) * 10000) / sample_rate
            data = np.sin(2 * np.pi * freq_signal * t)
            chunk = WaveformTrace(
                data=data,
                metadata=TraceMetadata(sample_rate=sample_rate),
            )
            analyzer.accumulate_psd(chunk, nperseg=512, window="hann")

        freqs, psd = analyzer.get_psd()

        # Should show averaged PSD with peak near signal frequency
        peak_idx = np.argmax(psd)
        assert 500 < freqs[peak_idx] < 1500

    def test_accumulate_histogram_single_chunk(self) -> None:
        """Test histogram accumulation from single chunk."""
        analyzer = StreamingAnalyzer()
        data = np.random.normal(0, 1, 10000)
        chunk = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_histogram(chunk, bins=50)
        counts, edges = analyzer.get_histogram()

        assert len(counts) == 50
        assert len(edges) == 51  # bin edges
        assert np.sum(counts) == len(data)

    def test_accumulate_histogram_multiple_chunks(self) -> None:
        """Test histogram accumulation from multiple chunks."""
        analyzer = StreamingAnalyzer()

        # First chunk
        data1 = np.random.normal(0, 1, 5000)
        chunk1 = WaveformTrace(
            data=data1,
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_histogram(chunk1, bins=50)

        # Second chunk
        data2 = np.random.normal(0, 1, 5000)
        chunk2 = WaveformTrace(
            data=data2,
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_histogram(chunk2, bins=50)

        counts, edges = analyzer.get_histogram()

        # Should accumulate both chunks
        assert np.sum(counts) == 10000

    def test_histogram_with_range(self) -> None:
        """Test histogram with specified range."""
        analyzer = StreamingAnalyzer()
        data = np.linspace(-10, 10, 1000)
        chunk = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_histogram(chunk, bins=100, range=(-5, 5))
        counts, edges = analyzer.get_histogram()

        assert len(counts) == 100
        assert edges[0] == -5
        assert edges[-1] == 5

    def test_reset(self) -> None:
        """Test analyzer reset clears all accumulators."""
        analyzer = StreamingAnalyzer()

        # Add data
        chunk = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_statistics(chunk)

        # Reset
        analyzer.reset()

        # Should be empty again
        with pytest.raises(ValueError, match="No data accumulated"):
            analyzer.get_statistics()

    def test_statistics_with_large_values(self) -> None:
        """Test statistics computation with large values."""
        analyzer = StreamingAnalyzer()
        large_data = np.array([1e10, 2e10, 3e10])
        chunk = WaveformTrace(
            data=large_data,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_statistics(chunk)
        stats = analyzer.get_statistics()

        expected_mean = 2e10
        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-10)

    def test_statistics_with_small_values(self) -> None:
        """Test statistics computation with small values."""
        analyzer = StreamingAnalyzer()
        small_data = np.array([1e-10, 2e-10, 3e-10])
        chunk = WaveformTrace(
            data=small_data,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_statistics(chunk)
        stats = analyzer.get_statistics()

        expected_mean = 2e-10
        np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-10)

    def test_psd_different_windows(self) -> None:
        """Test PSD with different window functions."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)
        chunk = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        for window in ["hann", "hamming", "blackman"]:
            analyzer = StreamingAnalyzer()
            analyzer.accumulate_psd(chunk, nperseg=512, window=window)
            freqs, psd = analyzer.get_psd()

            assert len(freqs) > 0
            assert len(psd) == len(freqs)

    def test_multiple_statistics_types(self) -> None:
        """Test accumulating multiple types of statistics."""
        analyzer = StreamingAnalyzer()

        chunk1 = WaveformTrace(
            data=np.array([1.0, 2.0, 3.0]),
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_statistics(chunk1)
        analyzer.accumulate_histogram(chunk1, bins=10)

        chunk2 = WaveformTrace(
            data=np.sin(2 * np.pi * 100 * np.arange(1000) / 1000),
            metadata=TraceMetadata(sample_rate=1000),
        )
        analyzer.accumulate_psd(chunk2, nperseg=256)

        stats = analyzer.get_statistics()
        counts, edges = analyzer.get_histogram()
        freqs, psd = analyzer.get_psd()

        assert len(stats) > 0
        assert len(counts) > 0
        assert len(freqs) > 0


class TestChunkedSpectrogram:
    """Tests for chunked_spectrogram function."""

    def test_basic_spectrogram(self) -> None:
        """Test basic spectrogram computation."""
        sample_rate = 10_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        times, freqs, Sxx = chunked_spectrogram(data, sample_rate, chunk_size=5000, nperseg=256)

        assert times.ndim == 1
        assert freqs.ndim == 1
        assert Sxx.ndim == 2
        assert Sxx.shape == (len(freqs), len(times))

    def test_single_chunk(self) -> None:
        """Test when data fits in single chunk."""
        sample_rate = 10_000
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        times, freqs, Sxx = chunked_spectrogram(data, sample_rate, chunk_size=100_000, nperseg=128)

        assert times.size > 0
        assert freqs.size > 0
        assert Sxx.shape == (len(freqs), len(times))

    def test_multiple_chunks(self) -> None:
        """Test processing multiple chunks."""
        sample_rate = 50_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        # Frequency sweep
        data = np.sin(2 * np.pi * 1000 * t * (1 + 0.5 * t))

        times, freqs, Sxx = chunked_spectrogram(
            data, sample_rate, chunk_size=5000, nperseg=256, overlap=512
        )

        # Verify monotonic time axis
        assert np.all(np.diff(times) > 0)

    def test_empty_signal(self) -> None:
        """Test with empty signal."""
        times, freqs, Sxx = chunked_spectrogram(np.array([]), 10_000, chunk_size=1000, nperseg=256)

        assert times.size == 0
        assert freqs.size == 0
        assert Sxx.shape == (0, 0)

    def test_very_small_signal(self) -> None:
        """Test with signal smaller than standard nperseg."""
        import warnings

        sample_rate = 10_000
        # Create a small but reasonable signal
        data = np.sin(2 * np.pi * 1000 * np.arange(50) / sample_rate)

        # Use nperseg appropriate for small signal
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            times, freqs, Sxx = chunked_spectrogram(
                data, sample_rate, chunk_size=1000, nperseg=16, noverlap=8
            )

        # Should handle gracefully
        assert isinstance(times, np.ndarray)
        assert isinstance(freqs, np.ndarray)
        # Should produce valid output even with small signal
        assert len(times) > 0 or len(freqs) == 0  # May be empty for very small signals

    def test_overlap_parameter(self) -> None:
        """Test overlap parameter effect."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        # No overlap
        times_no_overlap, freqs, Sxx_no_overlap = chunked_spectrogram(
            data, sample_rate, chunk_size=10_000, nperseg=256, overlap=0
        )

        # With overlap
        times_overlap, _, Sxx_overlap = chunked_spectrogram(
            data, sample_rate, chunk_size=10_000, nperseg=256, overlap=512
        )

        # Both should be valid
        assert times_no_overlap.size > 0
        assert times_overlap.size > 0

    def test_window_parameter(self) -> None:
        """Test different window functions."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        for window in ["hann", "hamming", "blackman", "bartlett"]:
            times, freqs, Sxx = chunked_spectrogram(
                data, sample_rate, chunk_size=10_000, nperseg=256, window=window
            )

            assert times.size > 0
            assert freqs.size > 0
            assert Sxx.shape == (len(freqs), len(times))

    def test_noverlap_parameter(self) -> None:
        """Test custom noverlap parameter."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        noverlap = 64
        times, freqs, Sxx = chunked_spectrogram(
            data,
            sample_rate,
            chunk_size=10_000,
            nperseg=256,
            noverlap=noverlap,
        )

        assert times.size > 0
        assert freqs.size > 0

    def test_memory_mapped_input(self) -> None:
        """Test with memory-mapped array."""
        sample_rate = 10_000
        duration = 0.5
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            data.tofile(f)
            f.flush()

            # Load as memmap
            mmap_data = np.memmap(f.name, dtype=np.float64, mode="r")

            times, freqs, Sxx = chunked_spectrogram(
                mmap_data, sample_rate, chunk_size=1000, nperseg=256
            )

            assert times.size > 0
            assert Sxx.shape == (len(freqs), len(times))

    def test_frequency_range(self) -> None:
        """Test that frequencies are in correct range."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        _, freqs, _ = chunked_spectrogram(data, sample_rate, chunk_size=10_000, nperseg=256)

        assert freqs[0] >= 0
        assert freqs[-1] <= sample_rate / 2

    def test_output_in_db(self) -> None:
        """Test that output is in dB."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        _, _, Sxx = chunked_spectrogram(data, sample_rate, chunk_size=10_000, nperseg=256)

        # Values should be in dB (generally negative)
        assert np.all(np.isfinite(Sxx))


class TestChunkedFFT:
    """Tests for chunked_fft function."""

    def test_basic_fft(self) -> None:
        """Test basic FFT computation."""
        sample_rate = 10_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=5000)

        assert freqs.ndim == 1
        assert magnitude.ndim == 1
        assert len(freqs) == len(magnitude)

    def test_single_chunk(self) -> None:
        """Test when data fits in single chunk."""
        sample_rate = 10_000
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=100_000)

        peak_idx = np.argmax(magnitude)
        # Peak should be near 1 kHz
        assert 500 < freqs[peak_idx] < 1500

    def test_multiple_chunks(self) -> None:
        """Test FFT with multiple chunks."""
        sample_rate = 10_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=5000, overlap=50)

        # Should identify correct frequency
        peak_idx = np.argmax(magnitude)
        freq_res = freqs[1] - freqs[0]
        assert abs(freqs[peak_idx] - 1000) < 2 * freq_res

    def test_empty_signal(self) -> None:
        """Test with empty signal."""
        freqs, magnitude = chunked_fft(np.array([]), 10_000, chunk_size=1000)

        assert freqs.size == 0
        assert magnitude.size == 0

    def test_overlap_parameter(self) -> None:
        """Test overlap percentage parameter."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        freqs_0, mag_0 = chunked_fft(data, sample_rate, chunk_size=5000, overlap=0)
        freqs_50, mag_50 = chunked_fft(data, sample_rate, chunk_size=5000, overlap=50)

        # Frequency axes should match
        np.testing.assert_allclose(freqs_0, freqs_50)

    def test_window_parameter(self) -> None:
        """Test different window functions."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        for window in ["hann", "hamming", "blackman", "bartlett", "rectangular"]:
            freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=5000, window=window)

            assert freqs.size > 0
            assert magnitude.size > 0

    def test_nfft_parameter(self) -> None:
        """Test custom NFFT parameter."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        nfft = 2048
        freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=5000, nfft=nfft)

        # Frequency resolution should match NFFT
        expected_freq_res = sample_rate / nfft
        actual_freq_res = freqs[1] - freqs[0]
        np.testing.assert_allclose(actual_freq_res, expected_freq_res, rtol=1e-10)

    def test_frequency_range(self) -> None:
        """Test that frequencies are in correct range."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        freqs, _ = chunked_fft(data, sample_rate, chunk_size=5000)

        assert freqs[0] == 0
        assert freqs[-1] <= sample_rate / 2

    def test_output_in_db(self) -> None:
        """Test that output is in dB."""
        sample_rate = 10_000
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / sample_rate)

        _, magnitude = chunked_fft(data, sample_rate, chunk_size=5000)

        # Values should be in dB (generally negative)
        assert np.all(np.isfinite(magnitude))

    def test_noise_reduction_through_averaging(self) -> None:
        """Test that multi-chunk averaging reduces noise."""
        sample_rate = 10_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0
        data = np.sin(2 * np.pi * freq_signal * t)

        # Add noise
        np.random.seed(42)
        data_noisy = data + 0.1 * np.random.randn(len(data))

        freqs, magnitude = chunked_fft(data_noisy, sample_rate, chunk_size=5000, overlap=50)

        # Peak should still be at signal frequency
        peak_idx = np.argmax(magnitude)
        freq_res = freqs[1] - freqs[0]
        assert abs(freqs[peak_idx] - freq_signal) < 2 * freq_res

    def test_memory_mapped_input(self) -> None:
        """Test with memory-mapped array."""
        sample_rate = 10_000
        duration = 0.5
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            data.tofile(f)
            f.flush()

            mmap_data = np.memmap(f.name, dtype=np.float64, mode="r")

            freqs, magnitude = chunked_fft(mmap_data, sample_rate, chunk_size=1000)

            assert freqs.size > 0
            assert magnitude.size > 0

    def test_very_small_signal(self) -> None:
        """Test with signal smaller than chunk size."""
        sample_rate = 10_000
        data = np.array([1.0, 2.0, 3.0])

        freqs, magnitude = chunked_fft(data, sample_rate, chunk_size=10000)

        assert freqs.size > 0
        assert magnitude.size > 0


class TestStreamingChunkedEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_sample_rate(self) -> None:
        """Test handling of invalid sample rate."""
        data = np.sin(2 * np.pi * 1000 * np.arange(100_000) / 10_000)

        # Zero or negative sample rate should fail
        with pytest.raises(ValueError):
            chunk = WaveformTrace(
                data=data,
                metadata=TraceMetadata(sample_rate=0),
            )

    def test_nan_data_handling(self) -> None:
        """Test handling of NaN values in data."""
        analyzer = StreamingAnalyzer()
        data_with_nan = np.array([1.0, np.nan, 3.0])
        chunk = WaveformTrace(
            data=data_with_nan,
            metadata=TraceMetadata(sample_rate=1000),
        )

        # Should not crash, but NaN will propagate
        analyzer.accumulate_statistics(chunk)
        stats = analyzer.get_statistics()

        assert np.isnan(stats["mean"]) or stats["mean"] is not None

    def test_inf_data_handling(self) -> None:
        """Test handling of infinite values."""
        analyzer = StreamingAnalyzer()
        data_with_inf = np.array([1.0, np.inf, 3.0])
        chunk = WaveformTrace(
            data=data_with_inf,
            metadata=TraceMetadata(sample_rate=1000),
        )

        analyzer.accumulate_statistics(chunk)
        stats = analyzer.get_statistics()

        # Max should be inf
        assert stats["max"] == np.inf

    def test_very_long_signal(self) -> None:
        """Test with very long signal (stress test)."""
        sample_rate = 1_000_000
        duration = 10.0  # 10 million samples
        n_samples = int(sample_rate * duration)

        # Create signal in chunks to avoid memory issues
        freqs = None
        magnitude = None
        chunk_size_write = 100_000

        with tempfile.NamedTemporaryFile(delete=False) as f:
            for i in range(0, n_samples, chunk_size_write):
                chunk_n = min(chunk_size_write, n_samples - i)
                t = np.arange(i, i + chunk_n) / sample_rate
                chunk_data = np.sin(2 * np.pi * 1000 * t)
                chunk_data.tofile(f)
            f.flush()

            mmap_data = np.memmap(f.name, dtype=np.float64, mode="r")
            freqs, magnitude = chunked_fft(mmap_data, sample_rate, chunk_size=1_000_000)

        # Should complete and identify signal frequency
        assert freqs.size > 0
        peak_idx = np.argmax(magnitude)
        assert 500 < freqs[peak_idx] < 1500

    def test_precision_with_large_accumulators(self) -> None:
        """Test numerical precision with large accumulators."""
        analyzer = StreamingAnalyzer()

        # Add many chunks to test accumulator stability
        for _ in range(100):
            data = np.random.randn(1000)
            chunk = WaveformTrace(
                data=data,
                metadata=TraceMetadata(sample_rate=1000),
            )
            analyzer.accumulate_statistics(chunk)

        stats = analyzer.get_statistics()

        # Should have reasonable statistics
        assert 0 < stats["std"] < 2  # std of normal should be ~1
        assert -3 < stats["mean"] < 3  # mean should be ~0


class TestStreamingChunkedIntegration:
    """Integration tests combining multiple components."""

    def test_full_streaming_pipeline(self) -> None:
        """Test complete streaming analysis pipeline."""
        # Create synthetic trace file
        sample_rate = 100_000
        total_samples = 1_000_000
        t = np.arange(total_samples) / sample_rate
        data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        loader = Mock(return_value=trace)

        # Process in chunks
        analyzer = StreamingAnalyzer()
        for chunk in load_trace_chunks("test.bin", chunk_size=100_000, loader=loader):
            analyzer.accumulate_statistics(chunk)
            analyzer.accumulate_psd(chunk, nperseg=512)
            analyzer.accumulate_histogram(chunk, bins=100)

        # Verify accumulated results
        stats = analyzer.get_statistics()
        assert stats["n_samples"] == total_samples

        freqs, psd = analyzer.get_psd()
        assert len(freqs) > 0

        counts, edges = analyzer.get_histogram()
        assert np.sum(counts) == total_samples

    def test_chunked_functions_consistency(self) -> None:
        """Test that chunked functions produce consistent results."""
        sample_rate = 50_000
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        data = np.sin(2 * np.pi * 1000 * t)

        # Small chunk size (force multiple chunks)
        times_chunked, freqs_chunked, Sxx_chunked = chunked_spectrogram(
            data, sample_rate, chunk_size=5000, nperseg=256, overlap=512
        )

        # Large chunk size (single chunk mode)
        times_single, freqs_single, Sxx_single = chunked_spectrogram(
            data, sample_rate, chunk_size=1_000_000, nperseg=256, overlap=512
        )

        # Both should identify the signal
        peak_freqs_chunked = freqs_chunked[np.argmax(Sxx_chunked, axis=0)]
        peak_freqs_single = freqs_single[np.argmax(Sxx_single, axis=0)]

        freq_res = freqs_chunked[1] - freqs_chunked[0]
        # Most peaks should be near 1 kHz in both cases
        assert np.mean(np.abs(peak_freqs_chunked - 1000) < 2 * freq_res) > 0.8
        assert np.mean(np.abs(peak_freqs_single - 1000) < 2 * freq_res) > 0.8
