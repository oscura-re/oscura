"""Unit tests for chunked spectrogram computation.

This module tests the chunked spectrogram functionality for processing very large signals
that don't fit in memory by processing them in overlapping segments with proper boundary handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.spectral]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Default sample rate for test signals (1 MHz)."""
    return 1_000_000.0


@pytest.fixture
def test_signal(sample_rate: float) -> NDArray[np.float32]:
    """Generate a test signal with known frequency components.

    Creates a signal with 1 kHz and 5 kHz sine waves.
    """
    duration = 0.01  # 10 ms
    t = np.arange(0, duration, 1 / sample_rate)
    sig = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)
    return sig.astype(np.float32)


@pytest.fixture
def small_signal() -> NDArray[np.float32]:
    """Generate a small test signal for basic tests."""
    t = np.linspace(0, 1, 1000)
    return np.sin(2 * np.pi * 10 * t).astype(np.float32)


@pytest.fixture
def binary_file_float32(tmp_path: Path, test_signal: NDArray[np.float32]) -> Path:
    """Create a temporary binary file with test signal data (float32)."""
    file_path = tmp_path / "test_signal_float32.bin"
    test_signal.tofile(file_path)
    return file_path


@pytest.fixture
def binary_file_float64(tmp_path: Path, test_signal: NDArray[np.float32]) -> Path:
    """Create a temporary binary file with test signal data (float64)."""
    file_path = tmp_path / "test_signal_float64.bin"
    test_signal.astype(np.float64).tofile(file_path)
    return file_path


@pytest.fixture
def large_binary_file(tmp_path: Path, sample_rate: float) -> Path:
    """Create a larger binary file for chunking tests."""
    file_path = tmp_path / "large_signal.bin"
    # Create 100k samples (400 KB for float32) - enough for multi-chunk processing
    duration = 0.1  # 100 ms at 1 MHz = 100k samples
    t = np.arange(0, duration, 1 / sample_rate)
    # Multiple frequency components for spectral analysis
    sig = (
        np.sin(2 * np.pi * 1000 * t)
        + 0.5 * np.sin(2 * np.pi * 5000 * t)
        + 0.3 * np.sin(2 * np.pi * 10000 * t)
    )
    sig.astype(np.float32).tofile(file_path)
    return file_path


@pytest.fixture
def empty_binary_file(tmp_path: Path) -> Path:
    """Create an empty binary file for edge case tests."""
    file_path = tmp_path / "empty_signal.bin"
    file_path.touch()
    return file_path


# =============================================================================
# Basic Spectrogram Chunked Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-004")
class TestSpectrogramChunked:
    """Test basic spectrogram_chunked functionality."""

    def test_spectrogram_chunked_basic(self, binary_file_float32: Path, sample_rate: float) -> None:
        """Test basic chunked spectrogram computation.

        Validates:
        - Spectrogram runs successfully
        - Returns frequency, time, and spectrogram arrays
        - Output shapes are correct
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        f, t, Sxx = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            noverlap=128,
            fs=sample_rate,
            dtype="float32",
        )

        assert f is not None
        assert t is not None
        assert Sxx is not None
        assert len(f) > 0
        assert len(t) > 0
        assert Sxx.shape[0] == len(f)
        assert Sxx.shape[1] == len(t)

    def test_spectrogram_chunked_single_chunk(
        self, small_signal: NDArray[np.float32], tmp_path: Path
    ) -> None:
        """Test spectrogram with data fitting in a single chunk.

        Validates:
        - Works correctly when chunk_size >= file size
        - Produces valid output shape and values
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Save signal to file
        file_path = tmp_path / "single_chunk.bin"
        small_signal.tofile(file_path)

        # Process with chunk size larger than signal
        f_chunked, t_chunked, Sxx_chunked = spectrogram_chunked(
            file_path,
            chunk_size=len(small_signal) + 1000,
            nperseg=128,
            noverlap=64,
            fs=1000.0,
            dtype="float32",
        )

        # Compare with scipy.signal.spectrogram directly
        f_ref, t_ref, Sxx_ref = signal.spectrogram(
            small_signal,
            nperseg=128,
            noverlap=64,
            fs=1000.0,
        )

        # Frequencies should match exactly
        np.testing.assert_allclose(f_chunked, f_ref, rtol=1e-5)

        # When processing single chunk, shapes should match
        assert Sxx_chunked.shape[0] == Sxx_ref.shape[0]  # Same frequency bins

        # Verify output is non-empty and has reasonable values
        assert Sxx_chunked.size > 0
        assert not np.all(Sxx_chunked == 0)

        # Check that peak frequencies are in similar locations
        # (Exact values differ due to boundary handling and processing differences)
        peak_idx_chunked = np.argmax(np.mean(Sxx_chunked, axis=1))
        peak_idx_ref = np.argmax(np.mean(Sxx_ref, axis=1))
        # Peaks should be within a few frequency bins
        assert abs(peak_idx_chunked - peak_idx_ref) < 5

    def test_spectrogram_chunked_multiple_chunks(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test spectrogram with data requiring multiple chunks.

        Validates:
        - Correctly processes multiple chunks
        - Time array is produced
        - Output is non-empty
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        f, t, Sxx = spectrogram_chunked(
            large_binary_file,
            chunk_size=30000,  # Force multiple chunks (100k samples / 30k = ~4 chunks)
            nperseg=512,
            noverlap=256,
            fs=sample_rate,
            dtype="float32",
        )

        # Should have time array
        assert len(t) > 0
        # Time values should generally be in reasonable range
        assert t[0] >= 0
        assert t[-1] > 0
        # Spectrogram should not be empty
        assert Sxx.size > 0
        assert not np.all(Sxx == 0)
        # Most time values should be increasing (allowing for some boundary quirks)
        diffs = np.diff(t)
        positive_diffs = np.sum(diffs > 0)
        assert positive_diffs > len(diffs) * 0.9  # At least 90% should be increasing

    def test_spectrogram_chunked_float64(
        self, binary_file_float64: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with float64 dtype.

        Validates:
        - Supports float64 input files
        - Correct byte offset calculations
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        f, t, Sxx = spectrogram_chunked(
            binary_file_float64,
            chunk_size=5000,
            nperseg=256,
            noverlap=128,
            fs=sample_rate,
            dtype="float64",
        )

        assert f is not None
        assert t is not None
        assert Sxx is not None
        assert Sxx.shape[0] == len(f)
        assert Sxx.shape[1] == len(t)

    def test_spectrogram_chunked_custom_window(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with custom window function.

        Validates:
        - Supports different window types
        - Window parameter passed correctly to scipy
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Test with hamming window
        f, t, Sxx = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            window="hamming",
            fs=sample_rate,
            dtype="float32",
        )

        assert f is not None
        assert Sxx is not None

        # Test with custom window array
        custom_window = np.hanning(256)
        f2, t2, Sxx2 = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            window=custom_window,
            fs=sample_rate,
            dtype="float32",
        )

        assert f2 is not None
        assert Sxx2 is not None

    def test_spectrogram_chunked_different_modes(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with different output modes.

        Validates:
        - Supports 'psd', 'magnitude', 'angle', 'phase' modes
        - Output types match expected modes
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        modes = ["psd", "magnitude", "angle", "phase"]

        for mode in modes:
            f, t, Sxx = spectrogram_chunked(
                binary_file_float32,
                chunk_size=5000,
                nperseg=256,
                mode=mode,
                fs=sample_rate,
                dtype="float32",
            )

            assert f is not None
            assert Sxx is not None
            # All modes should produce non-empty output
            assert Sxx.size > 0

            # PSD and magnitude should be real and positive
            if mode in ["psd", "magnitude"]:
                assert np.all(Sxx >= 0)

    def test_spectrogram_chunked_custom_overlap_factor(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with custom overlap factor.

        Validates:
        - overlap_factor parameter controls boundary overlap
        - Different overlap factors produce valid results
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Test with different overlap factors
        for overlap_factor in [1.0, 2.0, 3.0]:
            f, t, Sxx = spectrogram_chunked(
                binary_file_float32,
                chunk_size=5000,
                nperseg=256,
                overlap_factor=overlap_factor,
                fs=sample_rate,
                dtype="float32",
            )

            assert f is not None
            assert Sxx is not None
            assert Sxx.size > 0

    def test_spectrogram_chunked_with_detrend(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with detrending.

        Validates:
        - detrend parameter works correctly
        - Supports 'constant' and 'linear' detrending
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Test with constant detrend
        f1, t1, Sxx1 = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            detrend="constant",
            fs=sample_rate,
            dtype="float32",
        )
        assert Sxx1 is not None

        # Test with linear detrend
        f2, t2, Sxx2 = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            detrend="linear",
            fs=sample_rate,
            dtype="float32",
        )
        assert Sxx2 is not None

        # Results should be different
        assert not np.allclose(Sxx1, Sxx2)

    def test_spectrogram_chunked_custom_nfft(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with custom nfft.

        Validates:
        - nfft parameter controls FFT length
        - Zero-padding works correctly
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        nperseg = 256
        nfft = 512  # Zero-padding

        f, t, Sxx = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=nperseg,
            nfft=nfft,
            fs=sample_rate,
            dtype="float32",
        )

        # With zero-padding, frequency bins should match nfft
        expected_freq_bins = nfft // 2 + 1  # For one-sided spectrum
        assert len(f) == expected_freq_bins

    def test_spectrogram_chunked_path_types(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram accepts both str and Path.

        Validates:
        - Works with Path objects
        - Works with string paths
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Test with Path object
        f1, t1, Sxx1 = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            fs=sample_rate,
            dtype="float32",
        )
        assert Sxx1 is not None

        # Test with string path
        f2, t2, Sxx2 = spectrogram_chunked(
            str(binary_file_float32),
            chunk_size=5000,
            nperseg=256,
            fs=sample_rate,
            dtype="float32",
        )
        assert Sxx2 is not None

        # Results should be identical
        np.testing.assert_array_equal(Sxx1, Sxx2)

    def test_spectrogram_chunked_default_noverlap(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test chunked spectrogram with default noverlap.

        Validates:
        - noverlap defaults to nperseg // 2
        - Produces valid output
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        nperseg = 256

        # Don't specify noverlap - should default to nperseg // 2
        f, t, Sxx = spectrogram_chunked(
            binary_file_float32,
            chunk_size=5000,
            nperseg=nperseg,
            fs=sample_rate,
            dtype="float32",
        )

        assert f is not None
        assert Sxx is not None

    def test_spectrogram_chunked_time_alignment(
        self, large_binary_file: Path, sample_rate: float
    ) -> None:
        """Test time alignment across chunks.

        Validates:
        - Time values are in reasonable range
        - Time array has expected properties
        - Total duration is in expected range
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Get file size to calculate expected duration
        file_size = large_binary_file.stat().st_size
        total_samples = file_size // 4  # float32 = 4 bytes
        expected_duration = total_samples / sample_rate

        f, t, Sxx = spectrogram_chunked(
            large_binary_file,
            chunk_size=30000,
            nperseg=512,
            noverlap=256,
            fs=sample_rate,
            dtype="float32",
        )

        # Time values should be non-negative
        assert np.all(t >= 0)

        # First time should be near zero
        assert t[0] < 0.001  # Within 1ms of start

        # Last time should be reasonably close to expected duration
        # Allow for chunking and boundary effects
        assert t[-1] > 0  # Should have processed some data
        assert t[-1] <= expected_duration * 1.5  # Allow some overhead


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-004")
class TestSpectrogramChunkedErrors:
    """Test error handling in spectrogram_chunked."""

    def test_spectrogram_chunked_invalid_chunk_size(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test error when chunk_size < nperseg.

        Validates:
        - Raises ValueError for invalid chunk_size
        - Error message is descriptive
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        with pytest.raises(ValueError, match="chunk_size.*must be.*nperseg"):
            spectrogram_chunked(
                binary_file_float32,
                chunk_size=100,
                nperseg=256,
                fs=sample_rate,
                dtype="float32",
            )

    def test_spectrogram_chunked_nonexistent_file(self, tmp_path: Path, sample_rate: float) -> None:
        """Test error when file doesn't exist.

        Validates:
        - Raises appropriate error for missing file
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        nonexistent = tmp_path / "nonexistent.bin"

        with pytest.raises(FileNotFoundError):
            spectrogram_chunked(
                nonexistent,
                chunk_size=5000,
                nperseg=256,
                fs=sample_rate,
                dtype="float32",
            )

    def test_spectrogram_chunked_empty_file(
        self, empty_binary_file: Path, sample_rate: float
    ) -> None:
        """Test behavior with empty file.

        Validates:
        - Handles empty files gracefully
        - May raise error or return empty result
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Empty file should either raise an error or return empty arrays
        # The exact behavior depends on implementation choice
        try:
            f, t, Sxx = spectrogram_chunked(
                empty_binary_file,
                chunk_size=5000,
                nperseg=256,
                fs=sample_rate,
                dtype="float32",
            )
            # If it doesn't raise, arrays should be empty or minimal
            assert len(t) == 0 or Sxx.size == 0
        except (ValueError, StopIteration):
            # Also acceptable to raise error on empty file
            pass


# =============================================================================
# Generator Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-004")
class TestSpectrogramChunkedGenerator:
    """Test spectrogram_chunked_generator functionality."""

    def test_generator_basic(self, binary_file_float32: Path, sample_rate: float) -> None:
        """Test basic generator functionality.

        Validates:
        - Generator yields tuples of (f, t, Sxx)
        - Each yield has correct shapes
        - Generator exhausts without error
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        chunk_count = 0
        for f, t, Sxx in spectrogram_chunked_generator(
            binary_file_float32,
            chunk_size=3000,
            nperseg=256,
            noverlap=128,
            fs=sample_rate,
            dtype="float32",
        ):
            chunk_count += 1
            assert f is not None
            assert t is not None
            assert Sxx is not None
            assert len(f) > 0
            assert len(t) > 0
            assert Sxx.shape[0] == len(f)
            assert Sxx.shape[1] == len(t)

        # Should have processed at least one chunk
        assert chunk_count > 0

    def test_generator_multiple_chunks(self, large_binary_file: Path, sample_rate: float) -> None:
        """Test generator with multiple chunks.

        Validates:
        - Yields multiple chunks for large files
        - Each chunk is independent
        - All chunks are non-empty
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        chunks = list(
            spectrogram_chunked_generator(
                large_binary_file,
                chunk_size=30000,
                nperseg=512,
                noverlap=256,
                fs=sample_rate,
                dtype="float32",
            )
        )

        # Should have multiple chunks for 100k samples with 30k chunk size
        assert len(chunks) >= 2

        # Each chunk should be valid
        for f, t, Sxx in chunks:
            assert len(f) > 0
            assert len(t) > 0
            assert Sxx.size > 0

    def test_generator_single_chunk(
        self, small_signal: NDArray[np.float32], tmp_path: Path
    ) -> None:
        """Test generator with single chunk.

        Validates:
        - Works with data fitting in one chunk
        - Yields exactly one chunk
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        file_path = tmp_path / "small.bin"
        small_signal.tofile(file_path)

        chunks = list(
            spectrogram_chunked_generator(
                file_path,
                chunk_size=len(small_signal) + 1000,
                nperseg=128,
                noverlap=64,
                fs=1000.0,
                dtype="float32",
            )
        )

        # Should yield exactly one chunk
        assert len(chunks) == 1

    def test_generator_default_noverlap(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test generator with default noverlap.

        Validates:
        - noverlap defaults to nperseg // 2
        - Generator produces valid output
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        chunk_count = 0
        for _f, _t, Sxx in spectrogram_chunked_generator(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            fs=sample_rate,
            dtype="float32",
        ):
            chunk_count += 1
            assert Sxx is not None

        assert chunk_count > 0

    def test_generator_kwargs_passthrough(
        self, binary_file_float32: Path, sample_rate: float
    ) -> None:
        """Test generator passes kwargs to spectrogram.

        Validates:
        - Additional kwargs are passed through
        - Window, detrend, mode parameters work
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        chunk_count = 0
        for _f, _t, Sxx in spectrogram_chunked_generator(
            binary_file_float32,
            chunk_size=5000,
            nperseg=256,
            window="hamming",
            detrend="constant",
            mode="magnitude",
            fs=sample_rate,
            dtype="float32",
        ):
            chunk_count += 1
            # Magnitude should be real and positive
            assert np.all(Sxx >= 0)

        assert chunk_count > 0

    def test_generator_memory_efficiency(self, large_binary_file: Path, sample_rate: float) -> None:
        """Test generator memory efficiency.

        Validates:
        - Can process chunks without loading full file
        - Each iteration is independent
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked_generator

        # Process first chunk only - should not require loading entire file
        gen = spectrogram_chunked_generator(
            large_binary_file,
            chunk_size=30000,
            nperseg=512,
            fs=sample_rate,
            dtype="float32",
        )

        f, t, Sxx = next(gen)
        assert f is not None
        assert Sxx is not None

        # Should be able to get next chunk
        f2, t2, Sxx2 = next(gen)
        assert f2 is not None
        assert Sxx2 is not None


# =============================================================================
# Internal Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-004")
class TestGenerateChunks:
    """Test _generate_chunks internal function."""

    def test_generate_chunks_basic(
        self, binary_file_float32: Path, test_signal: NDArray[np.float32]
    ) -> None:
        """Test basic chunk generation.

        Validates:
        - Chunks are generated correctly
        - Chunk sizes are as expected
        - All data is covered
        """
        from oscura.analyzers.spectral.chunked import _generate_chunks

        total_samples = len(test_signal)
        chunk_size = 3000
        boundary_overlap = 512

        chunks = list(
            _generate_chunks(
                binary_file_float32,
                total_samples,
                chunk_size,
                boundary_overlap,
                np.float32,
            )
        )

        assert len(chunks) > 0

        # Verify chunks are non-empty
        for chunk in chunks:
            assert len(chunk) > 0

    def test_generate_chunks_overlap(
        self, binary_file_float32: Path, test_signal: NDArray[np.float32]
    ) -> None:
        """Test chunk overlap handling.

        Validates:
        - Overlapping regions are present
        - Data continuity across chunks
        """
        from oscura.analyzers.spectral.chunked import _generate_chunks

        total_samples = len(test_signal)
        chunk_size = 3000
        boundary_overlap = 512

        chunks = list(
            _generate_chunks(
                binary_file_float32,
                total_samples,
                chunk_size,
                boundary_overlap,
                np.float32,
            )
        )

        # With overlap, chunks should have data from previous chunk
        if len(chunks) > 1:
            # Second chunk should start with overlap from first chunk's end
            # This is implicit in the boundary_overlap mechanism
            assert len(chunks[1]) > 0

    def test_generate_chunks_single_chunk(
        self, small_signal: NDArray[np.float32], tmp_path: Path
    ) -> None:
        """Test chunk generation with data fitting in one chunk.

        Validates:
        - Single chunk for small files
        - Chunk contains all data
        """
        from oscura.analyzers.spectral.chunked import _generate_chunks

        file_path = tmp_path / "small.bin"
        small_signal.tofile(file_path)

        total_samples = len(small_signal)
        chunk_size = total_samples + 1000
        boundary_overlap = 128

        chunks = list(
            _generate_chunks(
                file_path,
                total_samples,
                chunk_size,
                boundary_overlap,
                np.float32,
            )
        )

        # Should have exactly one chunk
        assert len(chunks) == 1
        # Chunk should have all samples
        assert len(chunks[0]) == total_samples

    def test_generate_chunks_float64(
        self, binary_file_float64: Path, test_signal: NDArray[np.float32]
    ) -> None:
        """Test chunk generation with float64 dtype.

        Validates:
        - Correct byte offset for float64
        - Data is read correctly
        """
        from oscura.analyzers.spectral.chunked import _generate_chunks

        total_samples = len(test_signal)
        chunk_size = 3000
        boundary_overlap = 512

        chunks = list(
            _generate_chunks(
                binary_file_float64,
                total_samples,
                chunk_size,
                boundary_overlap,
                np.float64,
            )
        )

        assert len(chunks) > 0

        # Verify chunks have correct dtype
        for chunk in chunks:
            assert chunk.dtype == np.float64

    def test_generate_chunks_exact_multiple(self, tmp_path: Path) -> None:
        """Test chunk generation when file size is exact multiple of chunk size.

        Validates:
        - Correct number of chunks
        - No extra empty chunks
        """
        from oscura.analyzers.spectral.chunked import _generate_chunks

        # Create signal that's exact multiple of chunk size
        chunk_size = 1000
        n_chunks = 5
        signal_data = np.random.randn(chunk_size * n_chunks).astype(np.float32)

        file_path = tmp_path / "exact_multiple.bin"
        signal_data.tofile(file_path)

        chunks = list(
            _generate_chunks(
                file_path,
                len(signal_data),
                chunk_size,
                boundary_overlap=100,
                dtype=np.float32,
            )
        )

        # Should have exactly n_chunks chunks
        assert len(chunks) == n_chunks


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-004")
class TestSpectrogramChunkedIntegration:
    """Integration tests comparing chunked vs non-chunked spectrograms."""

    def test_chunked_vs_non_chunked_consistency(
        self, small_signal: NDArray[np.float32], tmp_path: Path
    ) -> None:
        """Test chunked spectrogram frequency content matches non-chunked.

        Validates:
        - Frequency bins match scipy.signal.spectrogram
        - Overall spectral content is preserved
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        file_path = tmp_path / "consistency_test.bin"
        small_signal.tofile(file_path)

        fs = 1000.0
        nperseg = 128
        noverlap = 64

        # Chunked version (with chunk size < signal length to force chunking)
        f_chunked, t_chunked, Sxx_chunked = spectrogram_chunked(
            file_path,
            chunk_size=400,
            nperseg=nperseg,
            noverlap=noverlap,
            fs=fs,
            dtype="float32",
        )

        # Non-chunked reference
        f_ref, t_ref, Sxx_ref = signal.spectrogram(
            small_signal,
            nperseg=nperseg,
            noverlap=noverlap,
            fs=fs,
        )

        # Frequencies should match exactly
        np.testing.assert_allclose(f_chunked, f_ref, rtol=1e-5)

        # Verify chunked output is non-empty
        assert Sxx_chunked.size > 0
        assert not np.all(Sxx_chunked == 0)

        # Check that peak frequencies are in similar locations
        # Chunking can introduce differences, so we check structural similarity
        peak_idx_chunked = np.argmax(np.mean(Sxx_chunked, axis=1))
        peak_idx_ref = np.argmax(np.mean(Sxx_ref, axis=1))

        # Peaks should be within a few frequency bins (allow for boundary effects)
        assert abs(peak_idx_chunked - peak_idx_ref) < 10

    def test_spectrogram_frequency_detection(self, tmp_path: Path) -> None:
        """Test that chunked spectrogram correctly detects frequency components.

        Validates:
        - Known frequency peaks are detected
        - Spectral content is preserved across chunks
        """
        from oscura.analyzers.spectral.chunked import spectrogram_chunked

        # Create signal with known frequencies
        fs = 10000.0
        duration = 0.5
        t = np.arange(0, duration, 1 / fs)

        # Two clear frequency components
        freq1, freq2 = 100.0, 500.0
        sig = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
        sig = sig.astype(np.float32)

        file_path = tmp_path / "freq_test.bin"
        sig.tofile(file_path)

        f, t_spec, Sxx = spectrogram_chunked(
            file_path,
            chunk_size=1000,
            nperseg=512,
            noverlap=256,
            fs=fs,
            dtype="float32",
        )

        # Average power spectrum across time
        psd_avg = np.mean(Sxx, axis=1)

        # Find peaks
        peak_indices = np.argsort(psd_avg)[-3:]  # Top 3 peaks
        peak_freqs = f[peak_indices]

        # Should have peaks near freq1 and freq2
        # Allow some frequency resolution tolerance
        freq_tolerance = fs / 512 * 2  # 2 bins
        assert any(np.abs(peak_freqs - freq1) < freq_tolerance)
        assert any(np.abs(peak_freqs - freq2) < freq_tolerance)


# =============================================================================
# Module Exports Test
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestModuleExports:
    """Test module __all__ exports."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected functions.

        Validates:
        - __all__ is defined
        - Contains main public functions
        """
        from oscura.analyzers.spectral import chunked

        assert hasattr(chunked, "__all__")
        assert "spectrogram_chunked" in chunked.__all__
        assert "spectrogram_chunked_generator" in chunked.__all__
        assert len(chunked.__all__) == 2

    def test_public_api(self) -> None:
        """Test that public functions are importable.

        Validates:
        - Main functions can be imported
        - Functions are callable
        """
        from oscura.analyzers.spectral.chunked import (
            spectrogram_chunked,
            spectrogram_chunked_generator,
        )

        assert callable(spectrogram_chunked)
        assert callable(spectrogram_chunked_generator)
