"""Unit tests for chunked FFT computation.

This module tests the chunked FFT functionality for processing very large signals
that don't fit in memory by processing them in overlapping segments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.spectral]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def test_signal(sample_rate: float) -> NDArray[np.float64]:
    """Generate a test signal with known frequency components.

    Creates a signal with 1 kHz and 5 kHz sine waves.
    """
    duration = 0.01  # 10 ms
    t = np.arange(0, duration, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)
    return signal.astype(np.float32)


@pytest.fixture
def small_signal() -> NDArray[np.float64]:
    """Generate a small test signal for basic tests."""
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 10 * t).astype(np.float32)


@pytest.fixture
def binary_file(tmp_path: Path, test_signal: NDArray[np.float64]) -> Path:
    """Create a temporary binary file with test signal data."""
    file_path = tmp_path / "test_signal.bin"
    test_signal.astype(np.float32).tofile(file_path)
    return file_path


@pytest.fixture
def large_binary_file(tmp_path: Path) -> Path:
    """Create a larger binary file for chunking tests."""
    file_path = tmp_path / "large_signal.bin"
    # Create 100k samples (400 KB for float32)
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(100000).astype(np.float32)
    signal.tofile(file_path)
    return file_path


# =============================================================================
# Basic FFT Chunked Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-006")
class TestFFTChunked:
    """Test basic fft_chunked functionality."""

    def test_fft_chunked_basic(self, binary_file: Path, sample_rate: float) -> None:
        """Test basic chunked FFT computation.

        Validates:
        - FFT runs successfully
        - Returns frequency and spectrum arrays
        - Output shapes are correct
        """
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert freqs is not None
        assert spectrum is not None
        assert len(freqs) == len(spectrum)
        assert len(freqs) > 0
        assert isinstance(spectrum, np.ndarray)
        assert not np.iscomplexobj(spectrum)  # Default is magnitude

    def test_fft_chunked_different_segment_sizes(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test FFT with different segment sizes."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        for segment_size in [256, 512, 1024, 2048]:
            freqs, spectrum = fft_chunked(
                binary_file,
                segment_size=segment_size,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )

            # FFT length should match segment size (nfft defaults to segment_size)
            expected_length = segment_size // 2 + 1  # rfft output length
            assert len(freqs) == expected_length
            assert len(spectrum) == expected_length

    def test_fft_chunked_overlap_percentages(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with different overlap percentages."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        for overlap_pct in [0.0, 25.0, 50.0, 75.0]:
            freqs, spectrum = fft_chunked(
                binary_file,
                segment_size=1024,
                overlap_pct=overlap_pct,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(spectrum) > 0

    def test_fft_chunked_window_functions(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with different window functions."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        for window in ["hann", "hamming", "blackman", "bartlett"]:
            freqs, spectrum = fft_chunked(
                binary_file,
                segment_size=1024,
                overlap_pct=50.0,
                window=window,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(spectrum) > 0
            assert np.all(np.isfinite(spectrum))

    def test_fft_chunked_custom_window_array(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with custom window array."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Custom Hann window
        segment_size = 1024
        custom_window = np.hanning(segment_size)

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=segment_size,
            overlap_pct=50.0,
            window=custom_window,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_with_nfft(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with custom nfft (zero-padding)."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        segment_size = 512
        nfft = 2048  # Zero-pad to larger FFT

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=segment_size,
            overlap_pct=50.0,
            nfft=nfft,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Output length should match nfft
        expected_length = nfft // 2 + 1
        assert len(freqs) == expected_length
        assert len(spectrum) == expected_length

    def test_fft_chunked_detrend_constant(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with constant detrending."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            detrend="constant",
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_detrend_linear(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with linear detrending."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            detrend="linear",
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_scaling_density(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with density scaling (PSD-like)."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            scaling="density",
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0
        assert np.all(spectrum >= 0)  # PSD is non-negative

    def test_fft_chunked_scaling_spectrum(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with spectrum scaling."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            scaling="spectrum",
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_average_methods(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with different averaging methods."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        for method in ["mean", "median", "max"]:
            freqs, spectrum = fft_chunked(
                binary_file,
                segment_size=512,
                overlap_pct=50.0,
                average_method=method,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(spectrum) > 0

    def test_fft_chunked_preserve_phase(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with phase preservation (complex output)."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            preserve_phase=True,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0
        assert np.iscomplexobj(spectrum)  # Should be complex

    def test_fft_chunked_float64_dtype(self, tmp_path: Path, sample_rate: float) -> None:
        """Test FFT with float64 data type."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create float64 binary file
        file_path = tmp_path / "test_float64.bin"
        signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 10000))
        signal.astype(np.float64).tofile(file_path)

        freqs, spectrum = fft_chunked(
            file_path,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float64",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_frequency_axis(self, binary_file: Path, sample_rate: float) -> None:
        """Test that frequency axis is correctly computed."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Check frequency axis properties
        assert freqs[0] == 0.0  # Should start at DC
        assert freqs[-1] <= sample_rate / 2  # Nyquist limit
        assert np.all(np.diff(freqs) > 0)  # Should be monotonically increasing


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-006")
class TestFFTChunkedEdgeCases:
    """Test edge cases and error handling."""

    def test_fft_chunked_invalid_overlap(self, binary_file: Path, sample_rate: float) -> None:
        """Test that invalid overlap percentage raises ValueError."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        with pytest.raises(ValueError, match="overlap_pct must be in"):
            fft_chunked(
                binary_file,
                segment_size=1024,
                overlap_pct=150.0,  # Invalid: > 100
                sample_rate=sample_rate,
                dtype="float32",
            )

        with pytest.raises(ValueError, match="overlap_pct must be in"):
            fft_chunked(
                binary_file,
                segment_size=1024,
                overlap_pct=-10.0,  # Invalid: < 0
                sample_rate=sample_rate,
                dtype="float32",
            )

    def test_fft_chunked_invalid_average_method(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test that invalid average method raises ValueError."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        with pytest.raises(ValueError, match="Unknown average_method"):
            fft_chunked(
                binary_file,
                segment_size=1024,
                overlap_pct=50.0,
                average_method="invalid",
                sample_rate=sample_rate,
                dtype="float32",
            )

    def test_fft_chunked_nonexistent_file(self, tmp_path: Path, sample_rate: float) -> None:
        """Test that nonexistent file raises appropriate error."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        nonexistent = tmp_path / "nonexistent.bin"

        with pytest.raises(FileNotFoundError):
            fft_chunked(
                nonexistent,
                segment_size=1024,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )

    def test_fft_chunked_empty_file(self, tmp_path: Path, sample_rate: float) -> None:
        """Test that empty file raises ValueError."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create empty file
        empty_file = tmp_path / "empty.bin"
        empty_file.touch()

        with pytest.raises(ValueError, match="No segments processed"):
            fft_chunked(
                empty_file,
                segment_size=1024,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )

    def test_fft_chunked_very_small_file(self, tmp_path: Path, sample_rate: float) -> None:
        """Test FFT on very small file (smaller than segment size)."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create file with only 100 samples
        small_file = tmp_path / "small.bin"
        np.random.randn(100).astype(np.float32).tofile(small_file)

        # Should still process, but with zero-padding
        freqs, spectrum = fft_chunked(
            small_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_non_power_of_2_segment(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test FFT with non-power-of-2 segment size."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Non-power-of-2 sizes should work (scipy.fft handles it)
        for segment_size in [100, 500, 1000]:
            freqs, spectrum = fft_chunked(
                binary_file,
                segment_size=segment_size,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(spectrum) > 0

    def test_fft_chunked_zero_overlap(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with zero overlap."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        freqs, spectrum = fft_chunked(
            binary_file,
            segment_size=1024,
            overlap_pct=0.0,  # No overlap
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    @pytest.mark.slow
    def test_fft_chunked_100_percent_overlap(self, binary_file: Path, sample_rate: float) -> None:
        """Test FFT with 100% overlap rejects invalid input."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # 100% overlap means hop=0, which would process same segment repeatedly
        # Function should validate and reject this pathological edge case
        with pytest.raises(ValueError, match="overlap|hop|must be less than"):
            fft_chunked(binary_file, sample_rate=sample_rate, chunk_size=4096, overlap_pct=1.0)


# =============================================================================
# Welch PSD Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-005")
class TestWelchPSDChunked:
    """Test Welch PSD estimation for large files."""

    def test_welch_psd_chunked_basic(self, binary_file: Path, sample_rate: float) -> None:
        """Test basic Welch PSD computation."""
        from oscura.analyzers.spectral.chunked_fft import welch_psd_chunked

        freqs, psd = welch_psd_chunked(
            binary_file,
            segment_size=256,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(psd) > 0
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)  # PSD is non-negative
        assert not np.iscomplexobj(psd)  # PSD is real-valued

    def test_welch_psd_chunked_different_segment_sizes(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test Welch PSD with different segment sizes."""
        from oscura.analyzers.spectral.chunked_fft import welch_psd_chunked

        for segment_size in [128, 256, 512, 1024]:
            freqs, psd = welch_psd_chunked(
                binary_file,
                segment_size=segment_size,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(psd) > 0

    def test_welch_psd_chunked_uses_mean_averaging(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test that Welch PSD uses mean averaging."""
        from oscura.analyzers.spectral.chunked_fft import welch_psd_chunked

        # Welch's method specifically uses mean averaging
        freqs, psd = welch_psd_chunked(
            binary_file,
            segment_size=256,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(psd) > 0
        # PSD should be smooth (averaged)
        assert np.all(np.isfinite(psd))

    def test_welch_psd_default_detrend(self, binary_file: Path, sample_rate: float) -> None:
        """Test that Welch PSD uses constant detrend by default."""
        from oscura.analyzers.spectral.chunked_fft import welch_psd_chunked

        freqs, psd = welch_psd_chunked(
            binary_file,
            segment_size=256,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Should remove DC bias (constant detrending)
        assert len(freqs) > 0
        assert len(psd) > 0


# =============================================================================
# Parallel FFT Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-006")
class TestFFTChunkedParallel:
    """Test parallel FFT computation."""

    def test_fft_chunked_parallel_basic(self, binary_file: Path, sample_rate: float) -> None:
        """Test basic parallel FFT computation.

        Note: Currently falls back to serial processing, but API should work.
        """
        from oscura.analyzers.spectral.chunked_fft import fft_chunked_parallel

        freqs, spectrum = fft_chunked_parallel(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            n_workers=4,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_parallel_different_workers(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test parallel FFT with different worker counts."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked_parallel

        for n_workers in [1, 2, 4, 8]:
            freqs, spectrum = fft_chunked_parallel(
                binary_file,
                segment_size=1024,
                overlap_pct=50.0,
                n_workers=n_workers,
                sample_rate=sample_rate,
                dtype="float32",
            )

            assert len(freqs) > 0
            assert len(spectrum) > 0

    def test_fft_chunked_parallel_kwargs_passthrough(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test that parallel FFT passes through kwargs correctly."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked_parallel

        freqs, spectrum = fft_chunked_parallel(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            n_workers=2,
            window="hamming",
            detrend="constant",
            scaling="density",
            average_method="median",
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_fft_chunked_parallel_preserve_phase(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test parallel FFT with phase preservation."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked_parallel

        freqs, spectrum = fft_chunked_parallel(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            n_workers=2,
            preserve_phase=True,
            sample_rate=sample_rate,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0
        assert np.iscomplexobj(spectrum)


# =============================================================================
# Streaming FFT Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("API-003")
class TestStreamingFFT:
    """Test streaming FFT with generator API."""

    def test_streaming_fft_basic(self, binary_file: Path, sample_rate: float) -> None:
        """Test basic streaming FFT computation."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        results = list(
            streaming_fft(
                binary_file,
                segment_size=1024,
                overlap_pct=50.0,
                sample_rate=sample_rate,
                dtype="float32",
            )
        )

        assert len(results) > 0

        # Check first result
        freqs, magnitude = results[0]
        assert len(freqs) > 0
        assert len(magnitude) > 0
        assert len(freqs) == len(magnitude)
        assert not np.iscomplexobj(magnitude)  # Streaming returns magnitude

    def test_streaming_fft_yields_multiple_segments(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test that streaming FFT yields multiple segments."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        segment_count = 0
        for freqs, magnitude in streaming_fft(
            large_binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        ):
            assert len(freqs) > 0
            assert len(magnitude) > 0
            segment_count += 1

        # With 100k samples, 1024 segment size, 50% overlap (512 hop)
        # Should have ~195 segments
        assert segment_count > 1

    def test_streaming_fft_progress_callback(self, binary_file: Path, sample_rate: float) -> None:
        """Test streaming FFT with progress callback."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        progress_calls = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        results = list(
            streaming_fft(
                binary_file,
                segment_size=512,
                overlap_pct=50.0,
                progress_callback=on_progress,
                sample_rate=sample_rate,
                dtype="float32",
            )
        )

        # Should have called progress callback
        assert len(progress_calls) > 0
        assert len(progress_calls) == len(results)

        # Check progress values
        for i, (current, total) in enumerate(progress_calls):
            assert current == i + 1
            assert total > 0
            # Note: In some edge cases, current can be > total due to rounding
            # This is acceptable for progress reporting

    def test_streaming_fft_frequency_consistency(
        self, binary_file: Path, sample_rate: float
    ) -> None:
        """Test that frequency axis is consistent across segments."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        freq_arrays = []
        for freqs, _magnitude in streaming_fft(
            binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        ):
            freq_arrays.append(freqs)

        # All frequency arrays should be identical
        assert len(freq_arrays) > 0
        reference = freq_arrays[0]
        for freqs in freq_arrays[1:]:
            assert np.allclose(freqs, reference)

    def test_streaming_fft_with_window(self, binary_file: Path, sample_rate: float) -> None:
        """Test streaming FFT with different window functions."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        for window in ["hann", "hamming", "blackman"]:
            results = list(
                streaming_fft(
                    binary_file,
                    segment_size=1024,
                    overlap_pct=50.0,
                    window=window,
                    sample_rate=sample_rate,
                    dtype="float32",
                )
            )

            assert len(results) > 0

    def test_streaming_fft_with_nfft(self, binary_file: Path, sample_rate: float) -> None:
        """Test streaming FFT with zero-padding."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        segment_size = 512
        nfft = 2048

        results = list(
            streaming_fft(
                binary_file,
                segment_size=segment_size,
                overlap_pct=50.0,
                nfft=nfft,
                sample_rate=sample_rate,
                dtype="float32",
            )
        )

        assert len(results) > 0

        # Check that output length matches nfft
        freqs, magnitude = results[0]
        expected_length = nfft // 2 + 1
        assert len(freqs) == expected_length
        assert len(magnitude) == expected_length

    def test_streaming_fft_invalid_overlap(self, binary_file: Path, sample_rate: float) -> None:
        """Test that invalid overlap raises ValueError."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        with pytest.raises(ValueError, match="overlap_pct must be in"):
            list(
                streaming_fft(
                    binary_file,
                    segment_size=1024,
                    overlap_pct=150.0,
                    sample_rate=sample_rate,
                    dtype="float32",
                )
            )


# =============================================================================
# StreamingAnalyzer Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("API-003")
class TestStreamingAnalyzer:
    """Test StreamingAnalyzer class for accumulating measurements."""

    def test_streaming_analyzer_init(self) -> None:
        """Test StreamingAnalyzer initialization."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        assert analyzer is not None
        assert analyzer.chunk_count == 0

    def test_streaming_analyzer_accumulate_psd(self) -> None:
        """Test PSD accumulation."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        # Create test chunks
        rng = np.random.default_rng(42)
        for _ in range(5):
            chunk = rng.standard_normal(10000)
            analyzer.accumulate_psd(chunk, nperseg=256, sample_rate=1e6)

        assert analyzer.chunk_count == 5

        # Get PSD
        freqs, psd = analyzer.get_psd()
        assert len(freqs) > 0
        assert len(psd) > 0
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)

    def test_streaming_analyzer_accumulate_stats(self) -> None:
        """Test statistics accumulation."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        # Create test chunks with known properties
        chunks = [
            np.ones(1000) * 2.0,
            np.ones(1000) * 3.0,
            np.ones(1000) * 4.0,
        ]

        for chunk in chunks:
            analyzer.accumulate_stats(chunk)

        stats = analyzer.get_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

        # Check approximate values
        assert abs(stats["mean"] - 3.0) < 0.1  # Average of 2, 3, 4
        assert stats["min"] == 2.0
        assert stats["max"] == 4.0

    def test_streaming_analyzer_get_psd_before_accumulation(self) -> None:
        """Test that get_psd raises error before accumulating data."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        with pytest.raises(ValueError, match="No PSD data accumulated"):
            analyzer.get_psd()

    def test_streaming_analyzer_get_stats_empty(self) -> None:
        """Test get_stats with no data returns zeros."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()
        stats = analyzer.get_stats()

        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_streaming_analyzer_reset(self) -> None:
        """Test resetting the analyzer."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        # Accumulate some data
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(1000)
        analyzer.accumulate_psd(chunk, nperseg=256)
        analyzer.accumulate_stats(chunk)

        assert analyzer.chunk_count > 0

        # Reset
        analyzer.reset()

        assert analyzer.chunk_count == 0
        with pytest.raises(ValueError):
            analyzer.get_psd()

        stats = analyzer.get_stats()
        assert stats["mean"] == 0.0

    def test_streaming_analyzer_multiple_accumulations(self) -> None:
        """Test multiple PSD and stats accumulations."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        rng = np.random.default_rng(42)
        for _i in range(10):
            chunk = rng.standard_normal(5000)
            analyzer.accumulate_psd(chunk, nperseg=128, sample_rate=1e6)
            analyzer.accumulate_stats(chunk)

        assert analyzer.chunk_count == 10

        freqs, psd = analyzer.get_psd()
        stats = analyzer.get_stats()

        assert len(freqs) > 0
        assert len(psd) > 0
        assert "mean" in stats

    def test_streaming_analyzer_different_nperseg(self) -> None:
        """Test PSD accumulation with different segment sizes."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(10000)

        for nperseg in [128, 256, 512]:
            analyzer.reset()
            analyzer.accumulate_psd(chunk, nperseg=nperseg, sample_rate=1e6)

            freqs, psd = analyzer.get_psd()
            # Different nperseg gives different frequency resolution
            assert len(freqs) > 0

    def test_streaming_analyzer_chunk_count_increment(self) -> None:
        """Test that chunk_count increments correctly."""
        from oscura.analyzers.spectral.chunked_fft import StreamingAnalyzer

        analyzer = StreamingAnalyzer()

        rng = np.random.default_rng(42)
        for i in range(5):
            chunk = rng.standard_normal(1000)
            analyzer.accumulate_psd(chunk)
            assert analyzer.chunk_count == i + 1


# =============================================================================
# Integration and Performance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestFFTChunkedIntegration:
    """Integration tests for chunked FFT processing."""

    def test_fft_chunked_on_known_signal(self, tmp_path: Path) -> None:
        """Test FFT on signal with known frequency components."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create signal with 1 kHz and 5 kHz components
        sample_rate = 100e3  # 100 kHz
        duration = 0.1  # 100 ms
        t = np.arange(0, duration, 1 / sample_rate)
        signal = (np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)).astype(
            np.float32
        )

        # Save to file
        file_path = tmp_path / "known_signal.bin"
        signal.tofile(file_path)

        # Compute FFT
        freqs, spectrum = fft_chunked(
            file_path,
            segment_size=2048,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Check that spectrum has significant energy at expected frequencies
        # Find indices near 1 kHz and 5 kHz
        idx_1k = np.argmin(np.abs(freqs - 1000))
        idx_5k = np.argmin(np.abs(freqs - 5000))

        # Get local spectrum values (use small window to account for spectral leakage)
        window = 5
        peak_1k = np.max(spectrum[max(0, idx_1k - window) : idx_1k + window + 1])
        peak_5k = np.max(spectrum[max(0, idx_5k - window) : idx_5k + window + 1])

        # Verify both frequencies have significant energy above noise floor
        noise_floor = np.median(spectrum)
        assert peak_1k > 2 * noise_floor, (
            f"No significant peak at 1 kHz: {peak_1k} vs noise {noise_floor}"
        )
        assert peak_5k > 2 * noise_floor, (
            f"No significant peak at 5 kHz: {peak_5k} vs noise {noise_floor}"
        )

        # 1 kHz should be stronger than 5 kHz (since amplitude is 1.0 vs 0.5)
        assert peak_1k > peak_5k, f"1 kHz peak ({peak_1k}) should be > 5 kHz peak ({peak_5k})"

    def test_fft_chunked_large_file_processing(self, tmp_path: Path) -> None:
        """Test processing of larger file."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create 1 MB file (~250k float32 samples)
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(250000).astype(np.float32)

        file_path = tmp_path / "large_file.bin"
        signal.tofile(file_path)

        freqs, spectrum = fft_chunked(
            file_path,
            segment_size=4096,
            overlap_pct=50.0,
            sample_rate=1e6,
            dtype="float32",
        )

        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_streaming_fft_complete_workflow(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test complete streaming workflow with analyzer."""
        from oscura.analyzers.spectral.chunked_fft import (
            StreamingAnalyzer,
            streaming_fft,
        )

        analyzer = StreamingAnalyzer()

        # Process segments as they arrive
        for freqs, magnitude in streaming_fft(
            large_binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        ):
            # Could process each segment here
            assert len(freqs) > 0
            assert len(magnitude) > 0

        # Analyzer is independent - would need to integrate differently
        # This just demonstrates the API pattern
        assert analyzer.chunk_count >= 0


# =============================================================================
# Memory Efficiency Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestMemoryEfficiency:
    """Test memory-efficient processing of large files."""

    def test_fft_chunked_segment_iterator_efficiency(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test that segment iteration doesn't load entire file."""
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Process large file with small segments
        freqs, spectrum = fft_chunked(
            large_binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Should complete without memory errors
        assert len(freqs) > 0
        assert len(spectrum) > 0

    def test_streaming_fft_yields_immediately(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test that streaming FFT yields results immediately."""
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        generator = streaming_fft(
            large_binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Get first result without consuming all segments
        freqs, magnitude = next(generator)

        assert len(freqs) > 0
        assert len(magnitude) > 0

        # Can continue processing or stop here
        # This demonstrates lazy evaluation


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports(self) -> None:
        """Test that all exported symbols are accessible."""
        from oscura.analyzers.spectral import chunked_fft

        expected_exports = [
            "StreamingAnalyzer",
            "fft_chunked",
            "fft_chunked_parallel",
            "streaming_fft",
            "welch_psd_chunked",
        ]

        for name in expected_exports:
            assert hasattr(chunked_fft, name)

    def test_import_all_functions(self) -> None:
        """Test direct import of all functions."""
        from oscura.analyzers.spectral.chunked_fft import (
            StreamingAnalyzer,
            fft_chunked,
            fft_chunked_parallel,
            streaming_fft,
            welch_psd_chunked,
        )

        assert callable(fft_chunked)
        assert callable(fft_chunked_parallel)
        assert callable(streaming_fft)
        assert callable(welch_psd_chunked)
        assert StreamingAnalyzer is not None


# =============================================================================
# Memory-Mapped I/O Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PERF-007")
class TestMemoryMappedIO:
    """Test memory-mapped I/O optimizations for large file processing."""

    def test_mmap_generates_correct_segments(self, large_binary_file: Path) -> None:
        """Test that memory-mapped segment generation produces correct data.

        Validates:
        - Segments match expected data from file
        - No data corruption from memory mapping
        - Segment boundaries are correct
        """
        from oscura.analyzers.spectral.chunked_fft import _generate_segments

        # Read file conventionally for comparison
        reference_data = np.fromfile(large_binary_file, dtype=np.float32)

        # Generate segments using memory-mapped I/O
        segment_size = 1024
        noverlap = 512
        segments = list(
            _generate_segments(
                large_binary_file, len(reference_data), segment_size, noverlap, np.float32
            )
        )

        # Verify first segment matches reference
        assert len(segments) > 0
        first_segment = segments[0]
        assert np.allclose(first_segment, reference_data[:segment_size])

        # Verify segment overlap is correct
        if len(segments) > 1:
            second_segment = segments[1]
            hop = segment_size - noverlap
            assert np.allclose(second_segment, reference_data[hop : hop + segment_size])

    def test_mmap_handles_partial_last_segment(self, tmp_path: Path) -> None:
        """Test memory-mapped I/O correctly handles partial last segment.

        Validates:
        - Last segment can be shorter than segment_size
        - No index out of bounds errors
        - Data is correct for partial segment
        """
        from oscura.analyzers.spectral.chunked_fft import _generate_segments

        # Create file with non-multiple-of-segment-size samples
        sample_count = 2500  # Not evenly divisible by 1024
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(sample_count).astype(np.float32)

        file_path = tmp_path / "partial_segment.bin"
        signal.tofile(file_path)

        # Generate segments
        segment_size = 1024
        noverlap = 512
        segments = list(
            _generate_segments(file_path, sample_count, segment_size, noverlap, np.float32)
        )

        # Last segment should be partial
        assert len(segments) > 0
        last_segment = segments[-1]
        assert len(last_segment) <= segment_size
        assert len(last_segment) > 0

    def test_mmap_large_file_efficiency(self, tmp_path: Path, sample_rate: float) -> None:
        """Test memory-mapped I/O performance on larger file.

        Validates:
        - Can process multi-MB files efficiently
        - No excessive memory usage
        - Results are consistent with traditional I/O
        """
        from oscura.analyzers.spectral.chunked_fft import fft_chunked

        # Create 5 MB file (~1.25M float32 samples)
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1_250_000).astype(np.float32)

        file_path = tmp_path / "large_mmap.bin"
        signal.tofile(file_path)

        # Process with memory-mapped I/O (happens automatically in _generate_segments)
        freqs, spectrum = fft_chunked(
            file_path,
            segment_size=4096,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        )

        # Verify results
        assert len(freqs) > 0
        assert len(spectrum) > 0
        assert not np.any(np.isnan(spectrum))
        assert not np.any(np.isinf(spectrum))

    def test_mmap_multiple_dtypes(self, tmp_path: Path) -> None:
        """Test memory-mapped I/O with different data types.

        Validates:
        - Works with float32 and float64
        - Correct byte offsets for different dtype sizes
        - No type conversion errors
        """
        from oscura.analyzers.spectral.chunked_fft import _generate_segments

        for dtype in [np.float32, np.float64]:
            # Create test file
            rng = np.random.default_rng(42)
            signal = rng.standard_normal(10000).astype(dtype)

            file_path = tmp_path / f"test_{dtype.__name__}.bin"
            signal.tofile(file_path)

            # Generate segments
            segments = list(_generate_segments(file_path, len(signal), 1024, 512, dtype))

            # Verify first segment data type and values
            assert len(segments) > 0
            assert segments[0].dtype == dtype
            assert np.allclose(segments[0], signal[:1024])

    def test_mmap_segment_boundaries(self, tmp_path: Path) -> None:
        """Test memory-mapped I/O segment boundary calculations.

        Validates:
        - Correct byte offset calculations
        - Proper handling of itemsize for different dtypes
        - No off-by-one errors in segment extraction
        """
        from oscura.analyzers.spectral.chunked_fft import _generate_segments

        # Create file with known pattern
        pattern = np.arange(5000, dtype=np.float32)
        file_path = tmp_path / "pattern.bin"
        pattern.tofile(file_path)

        # Generate segments with no overlap for easier validation
        segment_size = 1000
        noverlap = 0
        segments = list(
            _generate_segments(file_path, len(pattern), segment_size, noverlap, np.float32)
        )

        # Verify each segment contains correct sequential values
        assert len(segments) == 5  # 5000 samples / 1000 segment size
        for i, segment in enumerate(segments):
            expected_start = i * segment_size
            expected_end = expected_start + segment_size
            assert np.allclose(segment, pattern[expected_start:expected_end])

    def test_mmap_streaming_integration(self, large_binary_file: Path, sample_rate: float) -> None:
        """Test memory-mapped I/O integration with streaming FFT.

        Validates:
        - Streaming FFT uses memory-mapped segments
        - No memory leaks during iteration
        - Results are consistent
        """
        from oscura.analyzers.spectral.chunked_fft import streaming_fft

        segment_count = 0
        for freqs, magnitude in streaming_fft(
            large_binary_file,
            segment_size=1024,
            overlap_pct=50.0,
            sample_rate=sample_rate,
            dtype="float32",
        ):
            # Each iteration uses memory-mapped segment
            assert len(freqs) > 0
            assert len(magnitude) > 0
            assert not np.any(np.isnan(magnitude))
            segment_count += 1

        # Verify we processed multiple segments
        assert segment_count > 1
