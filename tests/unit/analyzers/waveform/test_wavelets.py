"""Unit tests for wavelet transform analysis.

This module provides comprehensive tests for continuous and discrete wavelet
transforms, including chunked implementations for memory-efficient processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.waveform.wavelets import cwt, cwt_chunked, dwt, dwt_chunked

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Helper Functions
# =============================================================================


def make_sine_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a sine wave signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def make_chirp(
    f0: float,
    f1: float,
    sample_rate: float,
    duration: float,
) -> np.ndarray:
    """Generate a linear chirp signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    # Linear chirp
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    return np.sin(phase)


# =============================================================================
# Test CWT (In-Memory)
# =============================================================================


@pytest.mark.unit
class TestCWT:
    """Test Continuous Wavelet Transform."""

    def test_cwt_basic(self) -> None:
        """Test basic CWT computation."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(100, 1000, 1.0)
        scales = np.arange(1, 64)

        coefficients, frequencies = cwt(signal, scales, wavelet="morl")

        # Check shapes
        assert coefficients.shape == (len(scales), len(signal))
        assert len(frequencies) == len(scales)

    def test_cwt_morlet_wavelet(self) -> None:
        """Test CWT with Morlet wavelet."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 0.5)
        scales = np.arange(1, 32)

        coefficients, frequencies = cwt(signal, scales, wavelet="morl")

        # Coefficients should be complex for Morlet
        assert np.iscomplexobj(coefficients)

    def test_cwt_ricker_wavelet(self) -> None:
        """Test CWT with Ricker (Mexican hat) wavelet."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 0.5)
        scales = np.arange(1, 32)

        coefficients, frequencies = cwt(signal, scales, wavelet="mexh")

        # Should work with Ricker wavelet
        assert coefficients.shape[0] == len(scales)

    def test_cwt_sampling_period(self) -> None:
        """Test CWT with custom sampling period."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 0.5)
        scales = np.arange(1, 32)

        # Sampling period = 1/1000 = 0.001
        coefficients, frequencies = cwt(signal, scales, sampling_period=0.001)

        # Frequencies should be computed correctly
        assert np.all(frequencies > 0)

    def test_cwt_no_pywt(self, monkeypatch) -> None:
        """Test CWT error when PyWavelets not available."""
        # Mock PyWavelets as unavailable
        import sys

        monkeypatch.setitem(sys.modules, "pywt", None)

        # Force reimport to trigger the check
        from oscura.analyzers.waveform import wavelets

        monkeypatch.setattr(wavelets, "PYWT_AVAILABLE", False)

        signal = np.random.randn(100)
        scales = np.arange(1, 10)

        with pytest.raises(ImportError, match="PyWavelets"):
            wavelets.cwt(signal, scales)


# =============================================================================
# Test DWT (In-Memory)
# =============================================================================


@pytest.mark.unit
class TestDWT:
    """Test Discrete Wavelet Transform."""

    def test_dwt_basic(self) -> None:
        """Test basic DWT computation."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)

        coeffs = dwt(signal, wavelet="db4", level=3)

        # Should have 1 approximation + 3 detail levels
        assert len(coeffs) == 4  # [cA3, cD3, cD2, cD1]

    def test_dwt_different_wavelets(self) -> None:
        """Test DWT with different wavelet families."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)

        coeffs_db = dwt(signal, wavelet="db4", level=2)
        coeffs_sym = dwt(signal, wavelet="sym4", level=2)
        coeffs_coif = dwt(signal, wavelet="coif1", level=2)

        # All should produce valid decompositions
        assert len(coeffs_db) == 3  # [cA2, cD2, cD1]
        assert len(coeffs_sym) == 3
        assert len(coeffs_coif) == 3

    def test_dwt_auto_level(self) -> None:
        """Test DWT with automatic level selection."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)

        # Let pywt choose the level
        coeffs = dwt(signal, wavelet="db4", level=None)

        # Should have multiple levels
        assert len(coeffs) > 2

    def test_dwt_modes(self) -> None:
        """Test DWT with different signal extension modes."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 0.5)

        coeffs_sym = dwt(signal, wavelet="db4", level=2, mode="symmetric")
        coeffs_per = dwt(signal, wavelet="db4", level=2, mode="periodic")
        coeffs_zero = dwt(signal, wavelet="db4", level=2, mode="zero")

        # All should work
        assert len(coeffs_sym) > 0
        assert len(coeffs_per) > 0
        assert len(coeffs_zero) > 0


# =============================================================================
# Test Chunked CWT (MEM-007)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-007")
class TestCWTChunked:
    """Test chunked CWT for large files."""

    def test_cwt_chunked_basic(self, tmp_path: Path) -> None:
        """Test basic chunked CWT processing."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create temporary signal file
        signal = make_sine_wave(50, 1000, 2.0)  # 2000 samples
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        scales = np.arange(1, 32)

        # Process in chunks
        chunks_processed = 0
        for coeffs, scales_out in cwt_chunked(signal_file, scales, wavelet="morl", chunk_size=1000):
            chunks_processed += 1
            assert coeffs.shape[0] == len(scales)
            assert np.array_equal(scales_out, scales)

        # Should process 2 chunks (2000 samples / 1000 chunk_size)
        assert chunks_processed == 2

    def test_cwt_chunked_overlap(self, tmp_path: Path) -> None:
        """Test chunked CWT with overlap handling."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 3.0)  # 3000 samples
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        scales = np.arange(1, 16)

        results = []
        for coeffs, _ in cwt_chunked(
            signal_file, scales, wavelet="morl", chunk_size=1000, overlap_factor=2.0
        ):
            results.append(coeffs)

        # Should process multiple chunks
        assert len(results) > 0

    def test_cwt_chunked_file_not_found(self) -> None:
        """Test chunked CWT with non-existent file."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        scales = np.arange(1, 10)

        with pytest.raises(FileNotFoundError):
            list(cwt_chunked("/nonexistent/file.bin", scales))

    def test_cwt_chunked_dtype(self, tmp_path: Path) -> None:
        """Test chunked CWT with different data types."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create signal with float32
        signal = make_sine_wave(50, 1000, 1.0)
        signal_file = tmp_path / "signal_f32.bin"
        signal.astype(np.float32).tofile(signal_file)

        scales = np.arange(1, 16)

        # Process with float32
        chunks_processed = 0
        for coeffs, _ in cwt_chunked(
            signal_file, scales, wavelet="morl", chunk_size=500, dtype=np.float32
        ):
            chunks_processed += 1
            assert coeffs.dtype == np.float64  # Output is always float64

        assert chunks_processed > 0


# =============================================================================
# Test Chunked DWT (MEM-007)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-007")
class TestDWTChunked:
    """Test chunked DWT for large files."""

    def test_dwt_chunked_basic(self, tmp_path: Path) -> None:
        """Test basic chunked DWT processing."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create temporary signal file
        signal = make_sine_wave(50, 1000, 2.0)  # 2000 samples
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        # Process in chunks
        chunks_processed = 0
        for coeffs_dict in dwt_chunked(signal_file, wavelet="db4", level=3, chunk_size=1000):
            chunks_processed += 1

            # Check structure
            assert "cA3" in coeffs_dict
            assert "cD3" in coeffs_dict
            assert "cD2" in coeffs_dict
            assert "cD1" in coeffs_dict

        # Should process 2 chunks
        assert chunks_processed == 2

    def test_dwt_chunked_auto_level(self, tmp_path: Path) -> None:
        """Test chunked DWT with automatic level selection."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        # Let pywt choose level automatically
        chunks_processed = 0
        for coeffs_dict in dwt_chunked(signal_file, wavelet="db4", level=None, chunk_size=500):
            chunks_processed += 1
            # Should have approximation and details
            assert any(k.startswith("cA") for k in coeffs_dict)
            assert any(k.startswith("cD") for k in coeffs_dict)

        assert chunks_processed > 0

    def test_dwt_chunked_different_wavelets(self, tmp_path: Path) -> None:
        """Test chunked DWT with different wavelet families."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        # Test db4
        for coeffs in dwt_chunked(signal_file, wavelet="db4", level=2, chunk_size=500):
            assert "cA2" in coeffs
            break

        # Test sym4
        for coeffs in dwt_chunked(signal_file, wavelet="sym4", level=2, chunk_size=500):
            assert "cA2" in coeffs
            break

    def test_dwt_chunked_modes(self, tmp_path: Path) -> None:
        """Test chunked DWT with different signal extension modes."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(50, 1000, 1.0)
        signal_file = tmp_path / "signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        # Test different modes
        for mode in ["symmetric", "periodic", "zero"]:
            chunks_processed = 0
            for coeffs in dwt_chunked(
                signal_file, wavelet="db4", level=2, chunk_size=500, mode=mode
            ):
                chunks_processed += 1
                assert "cA2" in coeffs

            assert chunks_processed > 0

    def test_dwt_chunked_file_not_found(self) -> None:
        """Test chunked DWT with non-existent file."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        with pytest.raises(FileNotFoundError):
            list(dwt_chunked("/nonexistent/file.bin", wavelet="db4"))


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestWaveletIntegration:
    """Integration tests for wavelet transforms."""

    def test_chirp_detection_cwt(self) -> None:
        """Test CWT on chirp signal for time-frequency analysis."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Linear chirp from 10 Hz to 100 Hz
        signal = make_chirp(10, 100, 1000, 2.0)
        scales = np.arange(1, 64)

        coefficients, frequencies = cwt(signal, scales, wavelet="morl")

        # Magnitude should show frequency variation over time
        magnitude = np.abs(coefficients)

        # Check that we have time-frequency representation
        assert magnitude.shape == (len(scales), len(signal))

        # Maximum magnitude should move across scales (frequencies) over time
        max_scale_per_time = np.argmax(magnitude, axis=0)

        # Should show variation (not constant)
        assert np.std(max_scale_per_time) > 1.0

    def test_noise_filtering_dwt(self) -> None:
        """Test using DWT for noise filtering."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create clean signal + noise
        rng = np.random.default_rng(42)
        clean_signal = make_sine_wave(50, 1000, 1.0)
        noise = rng.normal(0, 0.1, len(clean_signal))
        noisy_signal = clean_signal + noise

        # Decompose
        coeffs = dwt(noisy_signal, wavelet="db4", level=3)

        # Should have multiple coefficient arrays
        assert len(coeffs) == 4

        # Coefficients should be numpy arrays
        assert all(isinstance(c, np.ndarray) for c in coeffs)

    def test_large_signal_processing(self, tmp_path: Path) -> None:
        """Test processing larger signal files."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create a larger signal (10000 samples)
        signal = make_sine_wave(50, 1000, 10.0)
        signal_file = tmp_path / "large_signal.bin"
        signal.astype(np.float64).tofile(signal_file)

        scales = np.arange(1, 32)

        # Process with CWT chunked
        total_samples = 0
        for coeffs, _ in cwt_chunked(signal_file, scales, chunk_size=2000):
            total_samples += coeffs.shape[1]

        # Should process all samples
        assert total_samples >= len(signal) - 100  # Allow for some edge trimming


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestWaveformWaveletsEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test processing empty file."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Create empty file
        empty_file = tmp_path / "empty.bin"
        empty_file.touch()

        scales = np.arange(1, 10)

        # Should handle gracefully
        chunks = list(cwt_chunked(empty_file, scales))
        assert len(chunks) == 0

    def test_tiny_signal(self) -> None:
        """Test DWT on very small signal."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # Very small signal
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        # Should work but with limited decomposition
        coeffs = dwt(signal, wavelet="db1", level=1)

        # Should have approximation and detail
        assert len(coeffs) == 2

    def test_constant_signal_cwt(self) -> None:
        """Test CWT on constant signal."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        signal = np.ones(1000)
        scales = np.arange(1, 32)

        coefficients, frequencies = cwt(signal, scales)

        # Should work but coefficients near zero
        assert coefficients.shape == (len(scales), len(signal))

    def test_high_frequency_signal(self) -> None:
        """Test wavelet analysis on high-frequency signal."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            pytest.skip("PyWavelets not installed")

        # High frequency relative to sample rate
        signal = make_sine_wave(400, 1000, 1.0)  # Nyquist is 500 Hz

        scales = np.arange(1, 32)

        # CWT should still work
        coefficients, frequencies = cwt(signal, scales)
        assert coefficients.shape[0] == len(scales)

        # DWT should also work
        coeffs = dwt(signal, wavelet="db4", level=2)
        assert len(coeffs) > 0
