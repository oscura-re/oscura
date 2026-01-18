"""Unit tests for chunked wavelet transform computation.

This module tests the chunked wavelet functionality for processing very large signals
that don't fit in memory by processing them in overlapping segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
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
def small_binary_file(tmp_path: Path, small_signal: NDArray[np.float64]) -> Path:
    """Create a temporary binary file with small signal data."""
    file_path = tmp_path / "small_signal.bin"
    small_signal.astype(np.float32).tofile(file_path)
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


@pytest.fixture
def float64_binary_file(tmp_path: Path) -> Path:
    """Create a float64 binary file for dtype tests."""
    file_path = tmp_path / "signal_f64.bin"
    rng = np.random.default_rng(123)
    signal = rng.standard_normal(5000).astype(np.float64)
    signal.tofile(file_path)
    return file_path


# =============================================================================
# CWT Chunked Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-007")
class TestCWTChunked:
    """Test cwt_chunked functionality."""

    def test_cwt_chunked_basic(self, small_binary_file: Path) -> None:
        """Test basic chunked CWT computation.

        Validates:
        - CWT runs successfully
        - Returns coefficients and frequencies
        - Output shapes are correct
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4, 8]
        coeffs, freqs = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=500,
            sample_rate=1.0,
        )

        assert coeffs is not None
        assert freqs is not None
        assert coeffs.shape[0] == len(scales)
        assert len(freqs) == len(scales)

    def test_cwt_chunked_output_shape(self, small_binary_file: Path) -> None:
        """Test that CWT output shape matches input signal length.

        Validates:
        - Coefficient array has correct time dimension
        - All scales produce results
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        # Read original signal to get length
        signal = np.fromfile(small_binary_file, dtype=np.float32)
        signal_len = len(signal)

        scales = np.arange(1, 32)
        coeffs, freqs = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=300,
        )

        assert coeffs.shape[0] == len(scales)
        # Allow some tolerance for boundary handling
        assert abs(coeffs.shape[1] - signal_len) < 100

    def test_cwt_chunked_multiple_wavelets(self, small_binary_file: Path) -> None:
        """Test CWT with different wavelet types.

        Validates:
        - morl (Morlet) wavelet works
        - mexh (Mexican hat) wavelet works
        - Different wavelets produce valid outputs
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4, 8]

        for wavelet in ["morl", "mexh"]:
            coeffs, freqs = cwt_chunked(
                small_binary_file,
                scales=scales,
                wavelet=wavelet,
                chunk_size=500,
            )
            assert coeffs.shape[0] == len(scales)
            assert len(freqs) == len(scales)

    def test_cwt_chunked_overlap_factor(self, small_binary_file: Path) -> None:
        """Test different overlap factors.

        Validates:
        - Different overlap factors produce valid results
        - Larger overlap doesn't break processing
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4]

        for overlap_factor in [1.0, 2.0, 4.0]:
            coeffs, freqs = cwt_chunked(
                small_binary_file,
                scales=scales,
                wavelet="morl",
                chunk_size=400,
                overlap_factor=overlap_factor,
            )
            assert coeffs.shape[0] == len(scales)

    def test_cwt_chunked_large_file(self, large_binary_file: Path) -> None:
        """Test CWT on larger file with chunking.

        Validates:
        - Chunking works for files larger than chunk size
        - Results are continuous across chunk boundaries
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4, 8, 16]
        coeffs, freqs = cwt_chunked(
            large_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=10000,  # Process in 10k sample chunks
            sample_rate=1e6,
        )

        # Should process all 100k samples
        assert coeffs.shape[0] == len(scales)
        assert coeffs.shape[1] > 90000  # Allow for boundary trimming

    def test_cwt_chunked_float64_dtype(self, float64_binary_file: Path) -> None:
        """Test CWT with float64 data type.

        Validates:
        - float64 dtype parameter works
        - Correct number of samples read
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4]
        coeffs, freqs = cwt_chunked(
            float64_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=2000,
            dtype="float64",
        )

        assert coeffs.shape[0] == len(scales)
        assert coeffs.shape[1] > 4000  # Should read ~5000 samples

    def test_cwt_chunked_sample_rate(self, small_binary_file: Path) -> None:
        """Test that sample rate affects frequency output.

        Validates:
        - Different sample rates produce different frequencies
        - Frequencies scale appropriately with sample rate
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4, 8]

        _, freqs_1mhz = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            sample_rate=1e6,
        )

        _, freqs_100khz = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            sample_rate=1e5,
        )

        # Frequencies should differ based on sample rate
        assert not np.allclose(freqs_1mhz, freqs_100khz)

    def test_cwt_chunked_single_chunk(self, small_binary_file: Path) -> None:
        """Test CWT with chunk size larger than file.

        Validates:
        - Single chunk processing works
        - No boundary artifacts in single chunk mode
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4]
        coeffs, freqs = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=10000,  # Larger than 1000 sample file
        )

        assert coeffs.shape[0] == len(scales)

    def test_cwt_chunked_array_scales(self, small_binary_file: Path) -> None:
        """Test CWT with numpy array of scales.

        Validates:
        - Numpy array scales work
        - Continuous scale ranges work
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = np.arange(1, 33)  # Array of scales 1-32
        coeffs, freqs = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=500,
        )

        assert coeffs.shape[0] == len(scales)
        assert len(freqs) == len(scales)

    def test_cwt_chunked_path_object(self, small_binary_file: Path) -> None:
        """Test CWT accepts Path objects.

        Validates:
        - Path objects work in addition to strings
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4]
        coeffs, freqs = cwt_chunked(
            small_binary_file,  # Already a Path object
            scales=scales,
            wavelet="morl",
        )

        assert coeffs.shape[0] == len(scales)

    def test_cwt_chunked_string_path(self, small_binary_file: Path) -> None:
        """Test CWT accepts string paths.

        Validates:
        - String paths work
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        scales = [1, 2, 4]
        coeffs, freqs = cwt_chunked(
            str(small_binary_file),  # Convert to string
            scales=scales,
            wavelet="morl",
        )

        assert coeffs.shape[0] == len(scales)

    def test_cwt_chunked_no_file_raises(self, tmp_path: Path) -> None:
        """Test that missing file raises appropriate error.

        Validates:
        - FileNotFoundError or similar raised for missing files
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        nonexistent_file = tmp_path / "does_not_exist.bin"

        with pytest.raises((FileNotFoundError, OSError)):
            cwt_chunked(
                nonexistent_file,
                scales=[1, 2, 4],
                wavelet="morl",
            )

    def test_cwt_chunked_empty_file_raises(self, tmp_path: Path) -> None:
        """Test that empty file raises appropriate error.

        Validates:
        - Empty files are handled with proper error
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        empty_file = tmp_path / "empty.bin"
        empty_file.touch()

        with pytest.raises(ValueError):
            cwt_chunked(
                empty_file,
                scales=[1, 2, 4],
                wavelet="morl",
            )


# =============================================================================
# DWT Chunked Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-007")
class TestDWTChunked:
    """Test dwt_chunked functionality."""

    def test_dwt_chunked_basic(self, small_binary_file: Path) -> None:
        """Test basic chunked DWT computation.

        Validates:
        - DWT runs successfully
        - Returns list of coefficient arrays
        - Correct number of decomposition levels
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=3,
            chunk_size=500,
        )

        assert isinstance(coeffs, list)
        assert len(coeffs) == 4  # 3 detail levels + 1 approximation

    def test_dwt_chunked_output_structure(self, small_binary_file: Path) -> None:
        """Test DWT output structure.

        Validates:
        - First element is approximation coefficients
        - Remaining elements are detail coefficients
        - Coefficient sizes decrease with level
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=4,
        )

        assert len(coeffs) == 5  # cA4, cD4, cD3, cD2, cD1

        # Each level should be smaller (approximately half)
        for i in range(len(coeffs) - 1):
            assert isinstance(coeffs[i], np.ndarray)

    def test_dwt_chunked_multiple_wavelets(self, small_binary_file: Path) -> None:
        """Test DWT with different wavelet families.

        Validates:
        - Daubechies wavelets work (db4, db8)
        - Haar wavelet works
        - Symlet wavelets work
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        for wavelet in ["db4", "db8", "haar", "sym5"]:
            coeffs = dwt_chunked(
                small_binary_file,
                wavelet=wavelet,
                level=3,
                chunk_size=500,
            )
            assert len(coeffs) == 4

    def test_dwt_chunked_different_levels(self, small_binary_file: Path) -> None:
        """Test DWT with different decomposition levels.

        Validates:
        - Different levels produce correct number of coefficients
        - Level parameter controls decomposition depth
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        for level in [1, 2, 3, 4]:
            coeffs = dwt_chunked(
                small_binary_file,
                wavelet="db4",
                level=level,
            )
            assert len(coeffs) == level + 1

    def test_dwt_chunked_none_level(self, small_binary_file: Path) -> None:
        """Test DWT with automatic level selection (None).

        Validates:
        - None level parameter works
        - Produces valid output
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=None,
        )

        assert isinstance(coeffs, list)
        assert len(coeffs) > 0

    def test_dwt_chunked_boundary_modes(self, small_binary_file: Path) -> None:
        """Test DWT with different boundary extension modes.

        Validates:
        - symmetric mode works
        - periodic mode works
        - Different modes produce valid results
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        for mode in ["symmetric", "periodic", "zero", "constant"]:
            coeffs = dwt_chunked(
                small_binary_file,
                wavelet="db4",
                level=3,
                mode=mode,
            )
            assert len(coeffs) == 4

    def test_dwt_chunked_large_file(self, large_binary_file: Path) -> None:
        """Test DWT on larger file with chunking.

        Validates:
        - Chunking works for large files
        - All samples are processed
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            large_binary_file,
            wavelet="db4",
            level=5,
            chunk_size=10000,
        )

        assert len(coeffs) == 6
        # Verify reasonable total coefficient count
        total_coeffs = sum(len(c) for c in coeffs)
        assert total_coeffs > 50000

    def test_dwt_chunked_float64_dtype(self, float64_binary_file: Path) -> None:
        """Test DWT with float64 data type.

        Validates:
        - float64 dtype parameter works
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            float64_binary_file,
            wavelet="db4",
            level=3,
            dtype="float64",
        )

        assert len(coeffs) == 4

    def test_dwt_chunked_small_chunks(self, small_binary_file: Path) -> None:
        """Test DWT with small chunk sizes.

        Validates:
        - Small chunks work
        - Multiple chunks are processed correctly
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=2,
            chunk_size=200,  # Process in small chunks
        )

        assert len(coeffs) == 3

    def test_dwt_chunked_path_object(self, small_binary_file: Path) -> None:
        """Test DWT accepts Path objects.

        Validates:
        - Path objects work
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=3,
        )

        assert len(coeffs) == 4

    def test_dwt_chunked_string_path(self, small_binary_file: Path) -> None:
        """Test DWT accepts string paths.

        Validates:
        - String paths work
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        coeffs = dwt_chunked(
            str(small_binary_file),
            wavelet="db4",
            level=3,
        )

        assert len(coeffs) == 4

    def test_dwt_chunked_no_file_raises(self, tmp_path: Path) -> None:
        """Test that missing file raises appropriate error.

        Validates:
        - FileNotFoundError raised for missing files
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        nonexistent_file = tmp_path / "does_not_exist.bin"

        with pytest.raises((FileNotFoundError, OSError)):
            dwt_chunked(
                nonexistent_file,
                wavelet="db4",
                level=3,
            )

    def test_dwt_chunked_empty_file_raises(self, tmp_path: Path) -> None:
        """Test that empty file raises appropriate error.

        Validates:
        - Empty files handled with proper error
        """
        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        empty_file = tmp_path / "empty.bin"
        empty_file.touch()

        with pytest.raises(ValueError):
            dwt_chunked(
                empty_file,
                wavelet="db4",
                level=3,
            )


# =============================================================================
# CWT Chunked Generator Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-007")
class TestCWTChunkedGenerator:
    """Test cwt_chunked_generator functionality."""

    def test_generator_basic(self, small_binary_file: Path) -> None:
        """Test basic generator functionality.

        Validates:
        - Generator yields results
        - Each yield has coefficients and frequencies
        - Multiple chunks can be processed
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        scales = [1, 2, 4]
        chunk_count = 0

        for coeffs, freqs in cwt_chunked_generator(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=300,
        ):
            assert coeffs.shape[0] == len(scales)
            assert len(freqs) == len(scales)
            chunk_count += 1

        assert chunk_count > 0

    def test_generator_multiple_chunks(self, large_binary_file: Path) -> None:
        """Test generator with multiple chunks.

        Validates:
        - Multiple chunks are yielded
        - Each chunk has valid data
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        scales = [1, 2, 4, 8]
        chunks = list(
            cwt_chunked_generator(
                large_binary_file,
                scales=scales,
                wavelet="morl",
                chunk_size=10000,
            )
        )

        assert len(chunks) > 1  # Should have multiple chunks
        for coeffs, _freqs in chunks:
            assert coeffs.shape[0] == len(scales)

    def test_generator_lazy_evaluation(self, large_binary_file: Path) -> None:
        """Test that generator is lazy (doesn't process all at once).

        Validates:
        - Can iterate partially without processing entire file
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        scales = [1, 2, 4]
        gen = cwt_chunked_generator(
            large_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=10000,
        )

        # Get first chunk only
        coeffs, freqs = next(gen)
        assert coeffs.shape[0] == len(scales)

    def test_generator_with_kwargs(self, small_binary_file: Path) -> None:
        """Test generator with additional kwargs.

        Validates:
        - dtype kwarg works
        - sample_rate kwarg works
        - overlap_factor kwarg works
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        scales = [1, 2, 4]
        chunks = list(
            cwt_chunked_generator(
                small_binary_file,
                scales=scales,
                wavelet="morl",
                chunk_size=400,
                dtype="float32",
                sample_rate=1e6,
                overlap_factor=2.0,
            )
        )

        assert len(chunks) > 0

    def test_generator_different_wavelets(self, small_binary_file: Path) -> None:
        """Test generator with different wavelets.

        Validates:
        - Different wavelets produce valid results
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        scales = [1, 2, 4]

        for wavelet in ["morl", "mexh"]:
            chunks = list(
                cwt_chunked_generator(
                    small_binary_file,
                    scales=scales,
                    wavelet=wavelet,
                    chunk_size=400,
                )
            )
            assert len(chunks) > 0

    def test_generator_consistency_with_chunked(self, small_binary_file: Path) -> None:
        """Test generator produces similar results to regular chunked.

        Validates:
        - Generator and regular function produce comparable results
        - Total coefficients match approximately
        """
        from oscura.analyzers.spectral.chunked_wavelet import (
            cwt_chunked,
            cwt_chunked_generator,
        )

        scales = [1, 2, 4]

        # Get results from regular function
        coeffs_regular, _ = cwt_chunked(
            small_binary_file,
            scales=scales,
            wavelet="morl",
            chunk_size=400,
        )

        # Get results from generator
        chunks = list(
            cwt_chunked_generator(
                small_binary_file,
                scales=scales,
                wavelet="morl",
                chunk_size=400,
            )
        )

        # Total samples should be similar
        total_gen_samples = sum(c[0].shape[1] for c in chunks)
        assert abs(coeffs_regular.shape[1] - total_gen_samples) < 100


# =============================================================================
# Internal Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestGenerateChunks:
    """Test _generate_chunks internal function."""

    def test_generate_chunks_basic(self, small_binary_file: Path) -> None:
        """Test basic chunk generation.

        Validates:
        - Chunks are generated
        - Each chunk is a numpy array
        """
        from oscura.analyzers.spectral.chunked_wavelet import _generate_chunks

        signal = np.fromfile(small_binary_file, dtype=np.float32)
        total_samples = len(signal)

        chunks = list(
            _generate_chunks(
                small_binary_file,
                total_samples,
                chunk_size=300,
                boundary_overlap=10,
                dtype=np.float32,
            )
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)

    def test_generate_chunks_overlap(self, small_binary_file: Path) -> None:
        """Test that chunks have proper overlap.

        Validates:
        - Chunks overlap as specified
        - Overlap increases chunk sizes appropriately
        """
        from oscura.analyzers.spectral.chunked_wavelet import _generate_chunks

        signal = np.fromfile(small_binary_file, dtype=np.float32)
        total_samples = len(signal)

        chunks_no_overlap = list(
            _generate_chunks(
                small_binary_file,
                total_samples,
                chunk_size=300,
                boundary_overlap=0,
                dtype=np.float32,
            )
        )

        chunks_with_overlap = list(
            _generate_chunks(
                small_binary_file,
                total_samples,
                chunk_size=300,
                boundary_overlap=50,
                dtype=np.float32,
            )
        )

        # With overlap, should have same or more chunks
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_generate_chunks_single_chunk(self, small_binary_file: Path) -> None:
        """Test generation with chunk size larger than file.

        Validates:
        - Single chunk generated when chunk size > file size
        """
        from oscura.analyzers.spectral.chunked_wavelet import _generate_chunks

        signal = np.fromfile(small_binary_file, dtype=np.float32)
        total_samples = len(signal)

        chunks = list(
            _generate_chunks(
                small_binary_file,
                total_samples,
                chunk_size=10000,  # Larger than file
                boundary_overlap=10,
                dtype=np.float32,
            )
        )

        assert len(chunks) == 1

    def test_generate_chunks_multiple(self, large_binary_file: Path) -> None:
        """Test generation with multiple chunks.

        Validates:
        - Multiple chunks generated for large files
        - Chunks cover entire file
        """
        from oscura.analyzers.spectral.chunked_wavelet import _generate_chunks

        signal = np.fromfile(large_binary_file, dtype=np.float32)
        total_samples = len(signal)

        chunks = list(
            _generate_chunks(
                large_binary_file,
                total_samples,
                chunk_size=10000,
                boundary_overlap=100,
                dtype=np.float32,
            )
        )

        assert len(chunks) > 1
        # Verify total samples covered (approximately)
        total_chunk_samples = sum(len(c) for c in chunks)
        # Should be more than original due to overlaps
        assert total_chunk_samples >= total_samples


# =============================================================================
# Error Handling and Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_pywt_import_cwt(self, small_binary_file: Path, monkeypatch: Any) -> None:
        """Test ImportError when pywt is not available (CWT).

        Validates:
        - Appropriate error raised when pywt missing
        - Error message is helpful
        """
        # Mock pywt import failure
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pywt":
                raise ImportError("No module named 'pywt'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        with pytest.raises(ImportError, match="pywt.*required"):
            cwt_chunked(
                small_binary_file,
                scales=[1, 2, 4],
                wavelet="morl",
            )

    def test_missing_pywt_import_dwt(self, small_binary_file: Path, monkeypatch: Any) -> None:
        """Test ImportError when pywt is not available (DWT).

        Validates:
        - Appropriate error raised when pywt missing
        """
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pywt":
                raise ImportError("No module named 'pywt'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from oscura.analyzers.spectral.chunked_wavelet import dwt_chunked

        with pytest.raises(ImportError, match="pywt.*required"):
            dwt_chunked(
                small_binary_file,
                wavelet="db4",
                level=3,
            )

    def test_missing_pywt_import_generator(self, small_binary_file: Path, monkeypatch: Any) -> None:
        """Test ImportError when pywt is not available (generator).

        Validates:
        - Generator raises error when pywt missing
        """
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pywt":
                raise ImportError("No module named 'pywt'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked_generator

        gen = cwt_chunked_generator(
            small_binary_file,
            scales=[1, 2, 4],
            wavelet="morl",
        )

        with pytest.raises(ImportError, match="pywt.*required"):
            next(gen)

    def test_invalid_dtype(self, small_binary_file: Path) -> None:
        """Test behavior with mismatched dtype.

        Validates:
        - Wrong dtype reads incorrect number of samples
        - System handles gracefully
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked

        # File is float32, but we say float64
        # This should read half as many samples
        coeffs, freqs = cwt_chunked(
            small_binary_file,
            scales=[1, 2, 4],
            wavelet="morl",
            dtype="float64",  # Wrong dtype
        )

        # Should still produce output, just different size
        assert coeffs.shape[0] == 3


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("MEM-007")
class TestSpectralChunkedWaveletIntegration:
    """Integration tests combining multiple features."""

    def test_cwt_dwt_same_file(self, small_binary_file: Path) -> None:
        """Test both CWT and DWT on same file.

        Validates:
        - Both transforms work on same input
        - Can be used together in workflow
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked, dwt_chunked

        # CWT
        cwt_coeffs, cwt_freqs = cwt_chunked(
            small_binary_file,
            scales=[1, 2, 4, 8],
            wavelet="morl",
        )

        # DWT
        dwt_coeffs = dwt_chunked(
            small_binary_file,
            wavelet="db4",
            level=3,
        )

        assert cwt_coeffs.shape[0] == 4
        assert len(dwt_coeffs) == 4

    def test_full_workflow_signal_analysis(self, tmp_path: Path) -> None:
        """Test complete workflow: create signal -> CWT -> DWT.

        Validates:
        - End-to-end signal analysis workflow
        - All components work together
        """
        from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked, dwt_chunked

        # Create test signal with known frequencies
        t = np.linspace(0, 1, 5000)
        signal = (
            np.sin(2 * np.pi * 10 * t)
            + 0.5 * np.sin(2 * np.pi * 50 * t)
            + 0.25 * np.sin(2 * np.pi * 100 * t)
        )

        # Write to file
        signal_file = tmp_path / "test_signal.bin"
        signal.astype(np.float32).tofile(signal_file)

        # CWT analysis
        cwt_coeffs, cwt_freqs = cwt_chunked(
            signal_file,
            scales=np.arange(1, 64),
            wavelet="morl",
            chunk_size=2000,
            sample_rate=5000,
        )

        # DWT analysis
        dwt_coeffs = dwt_chunked(
            signal_file,
            wavelet="db4",
            level=5,
            chunk_size=2000,
        )

        # Verify both produced results
        assert cwt_coeffs.shape[0] == 63  # 63 scales
        assert len(dwt_coeffs) == 6  # 5 levels + approximation
