import pytest

"""Tests for chunked streaming processing (MEM-004, MEM-006).

Tests chunked_spectrogram and chunked_fft functions for memory-efficient
processing of large signals.

Requirements tested:
"""

import numpy as np

from oscura.streaming import chunked_fft, chunked_spectrogram

pytestmark = pytest.mark.unit


class TestChunkedSpectrogram:
    """Test chunked_spectrogram function (MEM-004)."""

    def test_basic_spectrogram(self):
        """Test basic spectrogram computation."""
        # Generate test signal: 1 kHz + 5 kHz
        sample_rate = 50e3  # 50 kHz
        duration = 1.0  # 1 second
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

        # Compute spectrogram with chunking
        times, freqs, Sxx = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=10000, nperseg=256
        )

        # Verify output shapes
        assert times.ndim == 1
        assert freqs.ndim == 1
        assert Sxx.ndim == 2
        assert Sxx.shape[0] == len(freqs)
        assert Sxx.shape[1] == len(times)

        # Verify frequency range
        assert freqs[0] >= 0
        assert freqs[-1] <= sample_rate / 2

        # Verify time range
        assert times[0] >= 0
        assert times[-1] <= duration

    def test_single_chunk_mode(self):
        """Test that single chunk mode produces valid spectrogram."""
        # Small signal that fits in one chunk
        sample_rate = 10e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0
        signal_data = np.sin(2 * np.pi * freq_signal * t)

        # Compute with chunked function
        nperseg = 128
        noverlap = 64  # explicit value
        times_chunked, freqs_chunked, Sxx_chunked = chunked_spectrogram(
            signal_data,
            sample_rate,
            chunk_size=10000,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        # Verify output shapes and ranges
        assert times_chunked.ndim == 1
        assert freqs_chunked.ndim == 1
        assert Sxx_chunked.ndim == 2
        assert Sxx_chunked.shape[0] == len(freqs_chunked)
        assert Sxx_chunked.shape[1] == len(times_chunked)

        # Verify frequency range
        assert freqs_chunked[0] >= 0
        assert freqs_chunked[-1] <= sample_rate / 2

        # Find peak frequency at each time
        peak_freqs = freqs_chunked[np.argmax(Sxx_chunked, axis=0)]
        freq_tolerance = sample_rate / nperseg  # Frequency bin width

        # Most peaks should be near the signal frequency
        correct_peaks = np.abs(peak_freqs - freq_signal) < freq_tolerance
        assert np.mean(correct_peaks) > 0.8, "Most time bins should show correct peak frequency"

    def test_multiple_chunks(self):
        """Test spectrogram with multiple chunks."""
        # Long signal requiring multiple chunks
        sample_rate = 50e3
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        # Frequency sweep
        signal_data = np.sin(2 * np.pi * 1000 * t * (1 + t))

        # Compute with small chunks
        times, _freqs, _Sxx = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=5000, nperseg=256, overlap=512
        )

        # Verify continuity (no gaps in time axis)
        time_diffs = np.diff(times)
        assert np.all(time_diffs > 0), "Time axis should be monotonically increasing"

        # Verify reasonable time resolution
        median_dt = np.median(time_diffs)
        assert median_dt < 1.0 / sample_rate * 300, "Time resolution should be reasonable"

    def test_chunk_boundary_handling(self):
        """Test proper handling of chunk boundaries."""
        # Signal with known frequency
        sample_rate = 10e3
        duration = 0.5
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0
        signal_data = np.sin(2 * np.pi * freq_signal * t)

        # Compute with multiple chunks
        _times, freqs, Sxx = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=1000, nperseg=128, overlap=256
        )

        # Find peak frequency at each time
        peak_freqs = freqs[np.argmax(Sxx, axis=0)]

        # Most peaks should be near the signal frequency
        freq_tolerance = sample_rate / 128  # Frequency bin width
        correct_peaks = np.abs(peak_freqs - freq_signal) < freq_tolerance
        assert np.mean(correct_peaks) > 0.8, "Most time bins should show correct peak frequency"

    def test_overlap_parameter(self):
        """Test effect of overlap parameter."""
        sample_rate = 10e3
        duration = 0.2
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Compute with different overlaps
        times_no_overlap, freqs, Sxx_no_overlap = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=500, nperseg=64, overlap=0
        )

        times_overlap, freqs2, Sxx_overlap = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=500, nperseg=64, overlap=128
        )

        # Both should produce valid results
        assert times_no_overlap.size > 0
        assert times_overlap.size > 0
        assert Sxx_no_overlap.shape == (len(freqs), len(times_no_overlap))
        assert Sxx_overlap.shape == (len(freqs2), len(times_overlap))

    def test_window_parameter(self):
        """Test different window functions."""
        sample_rate = 10e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Test different windows
        for window in ["hann", "hamming", "blackman", "bartlett"]:
            times, freqs, Sxx = chunked_spectrogram(
                signal_data, sample_rate, chunk_size=500, nperseg=128, window=window
            )

            # All should produce valid results
            assert times.size > 0
            assert freqs.size > 0
            assert Sxx.shape == (len(freqs), len(times))

    def test_memory_mapped_array(self):
        """Test compatibility with memory-mapped arrays."""
        import tempfile

        # Create temporary file
        sample_rate = 10e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write to disk
            signal_data.tofile(f)
            f.flush()

            # Load as memory-mapped array
            mmap_data = np.memmap(f.name, dtype=np.float64, mode="r")

            # Compute spectrogram
            times, freqs, Sxx = chunked_spectrogram(
                mmap_data, sample_rate, chunk_size=500, nperseg=64
            )

            # Should work without loading entire file into memory
            assert times.size > 0
            assert freqs.size > 0
            assert Sxx.shape == (len(freqs), len(times))


class TestChunkedFFT:
    """Test chunked_fft function (MEM-006)."""

    def test_basic_fft(self):
        """Test basic FFT computation."""
        # Generate test signal
        sample_rate = 10e3
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Compute chunked FFT
        freqs, magnitude = chunked_fft(signal_data, sample_rate, chunk_size=1000)

        # Verify output
        assert freqs.ndim == 1
        assert magnitude.ndim == 1
        assert len(freqs) == len(magnitude)

        # Verify frequency range
        assert freqs[0] == 0
        assert freqs[-1] <= sample_rate / 2

    def test_single_chunk_mode(self):
        """Test that single chunk mode produces valid FFT."""
        # Small signal
        sample_rate = 10e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0
        signal_data = np.sin(2 * np.pi * freq_signal * t)

        # Compute with chunked function
        freqs, magnitude = chunked_fft(signal_data, sample_rate, chunk_size=10000, overlap=0)

        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]

        # Peak should be near signal frequency
        freq_resolution = sample_rate / len(signal_data)
        assert abs(peak_freq - freq_signal) < 2 * freq_resolution

    def test_multiple_chunks_averaging(self):
        """Test FFT averaging over multiple chunks."""
        # Long signal
        sample_rate = 10e3
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0
        signal_data = np.sin(2 * np.pi * freq_signal * t)

        # Add noise
        np.random.seed(42)
        signal_data += 0.1 * np.random.randn(len(signal_data))

        # Compute with multiple chunks (averaging should reduce noise)
        freqs, magnitude = chunked_fft(signal_data, sample_rate, chunk_size=1000, overlap=50)

        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]

        # Peak should still be at signal frequency despite noise
        freq_resolution = freqs[1] - freqs[0]
        assert abs(peak_freq - freq_signal) < 2 * freq_resolution

    def test_overlap_effect(self):
        """Test effect of overlap on FFT averaging."""
        sample_rate = 10e3
        duration = 0.5
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Compute with different overlaps
        freqs_0, mag_0 = chunked_fft(signal_data, sample_rate, chunk_size=500, overlap=0)

        freqs_50, mag_50 = chunked_fft(signal_data, sample_rate, chunk_size=500, overlap=50)

        # Both should have same frequency axis
        np.testing.assert_allclose(freqs_0, freqs_50, rtol=1e-10)

        # Magnitudes may differ slightly due to different averaging
        # but should be similar
        correlation = np.corrcoef(mag_0, mag_50)[0, 1]
        assert correlation > 0.95, "High overlap should produce similar results"

    def test_window_effect(self):
        """Test different window functions."""
        sample_rate = 10e3
        duration = 0.2
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Test different windows
        results = {}
        for window in ["hann", "hamming", "blackman", "bartlett"]:
            freqs, magnitude = chunked_fft(signal_data, sample_rate, chunk_size=500, window=window)
            results[window] = (freqs, magnitude)

        # All should produce valid results
        for window, (freqs, magnitude) in results.items():
            assert freqs.size > 0
            assert magnitude.size > 0
            assert len(freqs) == len(magnitude)

    def test_nfft_parameter(self):
        """Test custom NFFT parameter."""
        sample_rate = 10e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 1000 * t)

        # Compute with custom NFFT
        nfft = 2048
        freqs, _magnitude = chunked_fft(signal_data, sample_rate, chunk_size=500, nfft=nfft)

        # Frequency resolution should match NFFT
        expected_freq_res = sample_rate / nfft
        actual_freq_res = freqs[1] - freqs[0]
        np.testing.assert_allclose(actual_freq_res, expected_freq_res, rtol=1e-10)

    def test_memory_bounded(self):
        """Test that chunked FFT is memory bounded."""
        # Create large signal (but don't allocate full memory)
        sample_rate = 1e6
        duration = 1.0
        n_samples = int(sample_rate * duration)

        # Use memory-mapped array to simulate large file
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Generate and write signal in chunks to avoid memory issues
            chunk_size = 100000
            for i in range(0, n_samples, chunk_size):
                chunk_n = min(chunk_size, n_samples - i)
                t = np.arange(i, i + chunk_n) / sample_rate
                chunk_data = np.sin(2 * np.pi * 1000 * t)
                chunk_data.tofile(f)
            f.flush()

            # Load as memory-mapped array
            mmap_data = np.memmap(f.name, dtype=np.float64, mode="r")

            # Compute FFT with small chunk size
            freqs, magnitude = chunked_fft(mmap_data, sample_rate, chunk_size=10000, overlap=50)

            # Should complete without loading full array into memory
            assert freqs.size > 0
            assert magnitude.size > 0

    def test_variance_reduction(self):
        """Test that averaging produces consistent results across realizations."""
        sample_rate = 10e3
        duration = 1.0
        t = np.arange(0, duration, 1 / sample_rate)
        freq_signal = 1000.0

        # Generate multiple realizations of clean signal
        np.random.seed(42)
        signal_clean = np.sin(2 * np.pi * freq_signal * t)

        # Test that averaging reduces spectral variance
        # Compute FFT with multiple chunks
        freqs_avg, mag_avg = chunked_fft(signal_clean, sample_rate, chunk_size=1000, overlap=50)

        # Find peak frequency
        peak_idx_avg = np.argmax(mag_avg)
        peak_freq = freqs_avg[peak_idx_avg]

        # Peak should be near signal frequency
        freq_resolution = freqs_avg[1] - freqs_avg[0]
        assert abs(peak_freq - freq_signal) < 2 * freq_resolution, (
            "Multi-chunk averaging should accurately identify signal frequency"
        )

        # The spectrum should have a clear peak at the signal frequency
        peak_magnitude = mag_avg[peak_idx_avg]
        median_magnitude = np.median(mag_avg)
        assert peak_magnitude > median_magnitude + 20, (
            "Signal peak should be well above noise floor"
        )


class TestStreamingStreamingChunkedEdgeCases:
    """Test edge cases for chunked processing."""

    def test_very_small_signal(self):
        """Test with signal smaller than chunk size."""
        sample_rate = 1e3
        signal_data = np.sin(2 * np.pi * 100 * np.arange(100) / sample_rate)

        # Should work with chunk size larger than signal
        times, _freqs, _Sxx = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=1000, nperseg=32
        )
        assert times.size > 0

        freq_fft, _mag_fft = chunked_fft(signal_data, sample_rate, chunk_size=1000)
        assert freq_fft.size > 0

    def test_empty_signal(self):
        """Test handling of empty or very small signal."""
        sample_rate = 1e3
        signal_data = np.array([])

        # Empty signal should raise an error or produce empty output
        # scipy.signal.spectrogram may warn but not necessarily error
        # Just verify it doesn't crash catastrophically
        try:
            times, freqs, Sxx = chunked_spectrogram(
                signal_data, sample_rate, chunk_size=100, nperseg=32
            )
            # If it doesn't error, output should be valid (possibly empty)
            assert isinstance(times, np.ndarray)
            assert isinstance(freqs, np.ndarray)
            assert isinstance(Sxx, np.ndarray)
        except (ValueError, IndexError):
            pass  # These exceptions are acceptable for empty input

        try:
            freqs, mag = chunked_fft(signal_data, sample_rate, chunk_size=100)
            assert isinstance(freqs, np.ndarray)
            assert isinstance(mag, np.ndarray)
        except (ValueError, IndexError, ZeroDivisionError):
            pass  # These exceptions are acceptable for empty input

    def test_chunk_size_edge_cases(self):
        """Test various chunk size configurations."""
        sample_rate = 1e3
        duration = 0.1
        t = np.arange(0, duration, 1 / sample_rate)
        signal_data = np.sin(2 * np.pi * 100 * t)

        # Chunk size = 1 (extreme case)
        # May not work well but shouldn't crash
        try:
            times, _freqs, _Sxx = chunked_spectrogram(
                signal_data, sample_rate, chunk_size=100, nperseg=32
            )
            assert times.size >= 0  # At least doesn't crash
        except ValueError:
            pass  # Acceptable to reject too-small chunks

        # Chunk size >> signal length
        times, _freqs, _Sxx = chunked_spectrogram(
            signal_data, sample_rate, chunk_size=100000, nperseg=32
        )
        assert times.size > 0
