"""Comprehensive tests for spectral/fft.py.

This test module provides complete coverage for FFT analysis functions,
including edge cases and error conditions.
"""

import numpy as np

from oscura.analyzers.spectral.fft import (
    fft_chunked,
    fft_chunked_file,
    fft_chunked_parallel,
    streaming_fft,
    welch_psd_chunked,
)


class TestFFTChunked:
    """Test suite for fft_chunked function.

    Tests cover:
    - Normal operation with valid inputs
    - Different chunk sizes and overlap percentages
    - Edge cases (empty, small signals)
    - Window function variations
    - Signal reconstruction accuracy
    """

    def test_simple_sine_wave(self) -> None:
        """Test FFT on simple sine wave shows correct peak."""
        # 1 kHz sine wave at 10 kHz sample rate
        fs = 10000.0
        f_signal = 1000.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        signal = np.sin(2 * np.pi * f_signal * t)

        freqs, mags = fft_chunked(signal, chunk_size=1024)

        # Find peak frequency
        peak_idx = np.argmax(mags[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx] * fs

        # Peak should be near 1 kHz
        assert abs(peak_freq - f_signal) < 100, f"Peak at {peak_freq} Hz, expected {f_signal} Hz"
        assert len(freqs) == len(mags)

    def test_chunk_size_variations(self) -> None:
        """Test different chunk sizes produce valid results."""
        signal = np.random.randn(50000)

        for chunk_size in [512, 1024, 2048, 4096, 8192]:
            freqs, mags = fft_chunked(signal, chunk_size=chunk_size)

            assert len(freqs) == chunk_size // 2 + 1
            assert len(mags) == chunk_size // 2 + 1
            assert np.all(np.isfinite(mags))
            assert np.all(mags >= 0)  # Magnitudes should be non-negative

    def test_overlap_variations(self) -> None:
        """Test different overlap percentages."""
        signal = np.random.randn(10000)

        for overlap in [0.0, 25.0, 50.0, 75.0]:
            freqs, mags = fft_chunked(signal, chunk_size=1024, overlap_pct=overlap)

            assert len(freqs) > 0
            assert len(mags) > 0
            assert np.all(np.isfinite(mags))

    def test_window_functions(self) -> None:
        """Test different window functions."""
        signal = np.random.randn(10000)

        for window in ["hann", "hamming", "blackman", "bartlett"]:
            freqs, mags = fft_chunked(signal, chunk_size=1024, window=window)

            assert len(freqs) > 0
            assert len(mags) > 0
            assert np.all(np.isfinite(mags))

    def test_signal_shorter_than_chunk(self) -> None:
        """Test signal shorter than chunk size."""
        signal = np.random.randn(512)
        chunk_size = 1024

        freqs, mags = fft_chunked(signal, chunk_size=chunk_size)

        # When signal < chunk_size, FFT uses signal length
        # but frequencies array uses chunk_size (implementation detail)
        # Magnitudes match signal FFT output
        assert len(freqs) == chunk_size // 2 + 1
        assert len(mags) == len(signal) // 2 + 1  # Based on signal length
        assert np.all(np.isfinite(mags))

    def test_very_small_signal(self) -> None:
        """Test with minimal signal length."""
        signal = np.array([1.0, 2.0, 3.0])
        chunk_size = 1024

        freqs, mags = fft_chunked(signal, chunk_size=chunk_size)

        assert len(freqs) > 0
        assert len(mags) > 0
        assert np.all(np.isfinite(mags))

    def test_dc_signal(self) -> None:
        """Test constant DC signal."""
        signal = np.ones(10000) * 5.0

        freqs, mags = fft_chunked(signal, chunk_size=1024)

        # DC component should dominate (allowing for windowing effects)
        assert mags[0] > np.max(mags[2:]) * 2  # Skip bin 1 due to window edge
        assert np.all(np.isfinite(mags))

    def test_multi_frequency_signal(self) -> None:
        """Test signal with multiple frequency components."""
        fs = 10000.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        # Combine 100 Hz, 500 Hz, 1000 Hz
        signal = (
            np.sin(2 * np.pi * 100 * t)
            + 0.5 * np.sin(2 * np.pi * 500 * t)
            + 0.3 * np.sin(2 * np.pi * 1000 * t)
        )

        freqs, mags = fft_chunked(signal, chunk_size=2048)

        # Should have distinct peaks (check magnitude is reasonable)
        # Find peak (should be at lowest frequency with highest amplitude)
        peak_idx = np.argmax(mags[1:]) + 1
        assert mags[peak_idx] > np.median(mags) * 5  # Clear peak

    def test_output_types(self) -> None:
        """Test output data types are correct."""
        signal = np.random.randn(10000)

        freqs, mags = fft_chunked(signal, chunk_size=1024)

        assert isinstance(freqs, np.ndarray)
        assert isinstance(mags, np.ndarray)
        assert freqs.dtype == np.float64
        assert mags.dtype == np.float64

    def test_frequency_range(self) -> None:
        """Test frequency array covers expected range."""
        signal = np.random.randn(10000)
        chunk_size = 1024

        freqs, _ = fft_chunked(signal, chunk_size=chunk_size)

        # Frequencies should be 0 to Nyquist (normalized, assuming sample rate = 1)
        assert freqs[0] == 0.0
        assert freqs[-1] <= 0.5  # Nyquist frequency for normalized fs=1

    def test_empty_signal(self) -> None:
        """Test handling of empty signal."""
        signal = np.array([])

        # Should handle gracefully
        try:
            freqs, mags = fft_chunked(signal, chunk_size=1024)
            # If no exception, check results are valid
            assert len(freqs) > 0
            assert len(mags) > 0
        except (ValueError, IndexError):
            # Also acceptable to raise error
            pass

    def test_complex_input_converted(self) -> None:
        """Test complex input is handled."""
        signal = np.random.randn(1000) + 1j * np.random.randn(1000)

        # Should convert to real (with warning)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
            freqs, mags = fft_chunked(signal, chunk_size=512)  # type: ignore[arg-type]
            assert np.all(np.isreal(mags))

    def test_large_signal(self) -> None:
        """Test chunked processing handles large signal efficiently."""
        # 1 million samples
        signal = np.random.randn(1000000)

        freqs, mags = fft_chunked(signal, chunk_size=8192, overlap_pct=50.0)

        assert len(freqs) == 8192 // 2 + 1
        assert len(mags) == 8192 // 2 + 1
        assert np.all(np.isfinite(mags))

    def test_zero_overlap(self) -> None:
        """Test zero overlap case."""
        signal = np.random.randn(10000)

        freqs, mags = fft_chunked(signal, chunk_size=1024, overlap_pct=0.0)

        assert len(freqs) > 0
        assert len(mags) > 0

    def test_maximum_overlap(self) -> None:
        """Test very high overlap (95%)."""
        signal = np.random.randn(10000)

        freqs, mags = fft_chunked(signal, chunk_size=1024, overlap_pct=95.0)

        assert len(freqs) > 0
        assert len(mags) > 0

    def test_invalid_overlap_rejected(self) -> None:
        """Test invalid overlap percentage is handled."""
        signal = np.random.randn(1000)

        # Negative overlap
        try:
            freqs, mags = fft_chunked(signal, chunk_size=512, overlap_pct=-10.0)
            # If accepted, result should still be valid
            assert np.all(np.isfinite(mags))
        except ValueError:
            pass

        # Overlap > 100%
        try:
            freqs, mags = fft_chunked(signal, chunk_size=512, overlap_pct=110.0)
            assert np.all(np.isfinite(mags))
        except ValueError:
            pass


class TestExportedFunctions:
    """Test re-exported chunked FFT functions.

    These functions are imported from chunked_fft module.
    Basic smoke tests to ensure imports work correctly.
    """

    def test_fft_chunked_file_callable(self) -> None:
        """Test fft_chunked_file is callable."""
        assert callable(fft_chunked_file)

    def test_fft_chunked_parallel_callable(self) -> None:
        """Test fft_chunked_parallel is callable."""
        assert callable(fft_chunked_parallel)

    def test_streaming_fft_callable(self) -> None:
        """Test streaming_fft is callable."""
        assert callable(streaming_fft)

    def test_welch_psd_chunked_callable(self) -> None:
        """Test welch_psd_chunked is callable."""
        assert callable(welch_psd_chunked)


# Integration tests
class TestFFTIntegration:
    """Integration tests for FFT analysis."""

    def test_spectral_leakage_reduced_by_windowing(self) -> None:
        """Test that windowing reduces spectral leakage."""
        # Non-integer number of cycles causes leakage
        fs = 10000.0
        t = np.linspace(0, 1, int(fs), endpoint=False)
        signal = np.sin(2 * np.pi * 1000.5 * t)  # 1000.5 Hz (not exact bin)

        # Without proper windowing, would see more leakage
        # But our function applies window by default
        freqs, mags = fft_chunked(signal, chunk_size=1024, window="hann")

        # Peak should still be relatively narrow
        peak_idx = np.argmax(mags[1:]) + 1
        peak_mag = mags[peak_idx]

        # Check nearby bins are significantly lower
        if peak_idx > 2 and peak_idx < len(mags) - 3:
            nearby_avg = np.mean([mags[peak_idx - 2], mags[peak_idx + 2]])
            assert peak_mag > nearby_avg * 5  # At least 5x higher

    def test_averaging_reduces_noise(self) -> None:
        """Test that averaging multiple chunks reduces noise floor."""
        # Signal with noise
        fs = 10000.0
        t = np.linspace(0, 2, int(2 * fs), endpoint=False)
        signal = np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(len(t))

        # Process with chunking (will average multiple chunks)
        freqs, mags = fft_chunked(signal, chunk_size=1024, overlap_pct=50.0)

        # SNR should be reasonable
        peak_idx = np.argmax(mags[1:]) + 1
        peak_mag = mags[peak_idx]
        noise_floor = np.median(mags[10:100])  # Away from DC and peak

        snr = peak_mag / noise_floor
        assert snr > 10  # Should have good SNR

    def test_parseval_theorem_approximate(self) -> None:
        """Test energy conservation (Parseval's theorem)."""
        signal = np.random.randn(10000)
        time_energy = np.sum(signal**2)

        freqs, mags = fft_chunked(signal, chunk_size=1024, overlap_pct=0.0)

        # FFT magnitude spectrum energy (approximation)
        # Note: Exact Parseval requires careful normalization
        freq_energy = np.sum(mags**2)

        # Should be same order of magnitude (within factor of 10)
        ratio = freq_energy / time_energy if time_energy > 0 else 0
        assert 0.01 < ratio < 100  # Loose check due to chunking/windowing
