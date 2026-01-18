"""Tests for GPU backend with automatic fallback.


Example:
    >>> pytest tests/unit/core/test_gpu_backend.py -v
"""

from __future__ import annotations

import os
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.core.gpu_backend import GPUBackend, gpu

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestGPUBackend:
    """Test suite for GPU backend with fallback behavior."""

    def test_gpu_backend_initialization(self) -> None:
        """Test GPU backend initializes correctly."""
        backend = GPUBackend()
        assert backend is not None
        assert isinstance(backend._force_cpu, bool)
        assert backend._initialized is False

    def test_force_cpu_mode(self) -> None:
        """Test that force_cpu=True disables GPU."""
        backend = GPUBackend(force_cpu=True)
        assert backend.gpu_available is False
        assert backend.using_gpu is False

    def test_environment_variable_disables_gpu(self) -> None:
        """Test OSCURA_USE_GPU=0 environment variable disables GPU."""
        with patch.dict(os.environ, {"OSCURA_USE_GPU": "0"}):
            backend = GPUBackend()
            assert backend.gpu_available is False

    def test_environment_variable_enables_gpu(self) -> None:
        """Test OSCURA_USE_GPU=1 tries to enable GPU if available."""
        with patch.dict(os.environ, {"OSCURA_USE_GPU": "1"}):
            backend = GPUBackend()
            # Result depends on whether CuPy is actually installed
            # Just verify it attempts to check
            _ = backend.gpu_available
            assert backend._initialized is True

    def test_gpu_check_lazy_initialization(self) -> None:
        """Test GPU availability check is lazy (only on first use)."""
        backend = GPUBackend()
        assert backend._initialized is False
        _ = backend.gpu_available  # Trigger check
        assert backend._initialized is True

    def test_module_level_singleton(self) -> None:
        """Test module-level gpu singleton exists."""
        assert gpu is not None
        assert isinstance(gpu, GPUBackend)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_fft_with_real_input(self, force_cpu: bool) -> None:
        """Test FFT with real-valued input."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.float64)

        result = backend.fft(signal)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.complex128
        assert len(result) == len(signal)
        # Verify result matches NumPy
        expected = np.fft.fft(signal)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_fft_with_complex_input(self, force_cpu: bool) -> None:
        """Test FFT with complex-valued input."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.complex128)
        signal += 1j * np.random.randn(1024)

        result = backend.fft(signal)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.complex128
        expected = np.fft.fft(signal)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_fft_with_zero_padding(self, force_cpu: bool) -> None:
        """Test FFT with zero-padding (n parameter)."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1000).astype(np.float64)

        result = backend.fft(signal, n=2048)

        assert len(result) == 2048
        expected = np.fft.fft(signal, n=2048)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_fft_normalization_modes(
        self, force_cpu: bool, norm: Literal["backward", "ortho", "forward"]
    ) -> None:
        """Test FFT with different normalization modes."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.float64)

        result = backend.fft(signal, norm=norm)

        expected = np.fft.fft(signal, norm=norm)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_ifft_basic(self, force_cpu: bool) -> None:
        """Test inverse FFT recovers original signal."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.float64)

        spectrum = backend.fft(signal)
        recovered = backend.ifft(spectrum)

        np.testing.assert_allclose(recovered.real, signal, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_rfft_basic(self, force_cpu: bool) -> None:
        """Test real FFT returns correct output size."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.float64)

        result = backend.rfft(signal)

        # Real FFT should return n//2 + 1 complex values
        assert len(result) == 1024 // 2 + 1
        assert result.dtype == np.complex128
        expected = np.fft.rfft(signal)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_irfft_basic(self, force_cpu: bool) -> None:
        """Test inverse real FFT recovers original signal."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1024).astype(np.float64)

        spectrum = backend.rfft(signal)
        recovered = backend.irfft(spectrum)

        assert recovered.dtype == np.float64
        np.testing.assert_allclose(recovered, signal, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_rfft_odd_length(self, force_cpu: bool) -> None:
        """Test real FFT with odd-length input."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(1023).astype(np.float64)

        result = backend.rfft(signal)
        recovered = backend.irfft(result, n=1023)

        assert len(result) == 1023 // 2 + 1
        np.testing.assert_allclose(recovered, signal, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_convolve_modes(self, force_cpu: bool, mode: Literal["full", "valid", "same"]) -> None:
        """Test convolution with different modes."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(100).astype(np.float64)
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

        result = backend.convolve(signal, kernel, mode=mode)

        expected = np.convolve(signal, kernel, mode=mode)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_convolve_smoothing(self, force_cpu: bool) -> None:
        """Test convolution for signal smoothing."""
        backend = GPUBackend(force_cpu=force_cpu)
        # Create noisy signal
        t = np.linspace(0, 1, 1000, dtype=np.float64)
        signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(1000)
        # Simple moving average kernel
        kernel = np.ones(5, dtype=np.float64) / 5

        smoothed = backend.convolve(signal, kernel, mode="same")

        # Smoothed signal should have less variance
        assert np.var(smoothed) < np.var(signal)

    @pytest.mark.parametrize("force_cpu", [True, False])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_correlate_modes(self, force_cpu: bool, mode: Literal["full", "valid", "same"]) -> None:
        """Test correlation with different modes."""
        backend = GPUBackend(force_cpu=force_cpu)
        signal = np.random.randn(100).astype(np.float64)
        template = np.random.randn(20).astype(np.float64)

        result = backend.correlate(signal, template, mode=mode)

        expected = np.correlate(signal, template, mode=mode)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_correlate_pattern_matching(self, force_cpu: bool) -> None:
        """Test correlation for pattern matching."""
        backend = GPUBackend(force_cpu=force_cpu)
        # Create signal with embedded pattern
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(1000).astype(np.float64)
        pattern = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        # Embed pattern at position 500
        signal[500:505] = pattern

        corr = backend.correlate(signal, pattern, mode="valid")

        # Correlation peak should be near position 500
        peak_idx = np.argmax(corr)
        assert 495 <= peak_idx <= 505

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_histogram_basic(self, force_cpu: bool) -> None:
        """Test histogram computation."""
        backend = GPUBackend(force_cpu=force_cpu)
        data = np.random.randn(10000).astype(np.float64)

        counts, edges = backend.histogram(data, bins=50)

        assert isinstance(counts, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert len(counts) == 50
        assert len(edges) == 51  # N bins -> N+1 edges
        assert counts.sum() == len(data)
        # Verify against NumPy
        exp_counts, exp_edges = np.histogram(data, bins=50)
        np.testing.assert_array_equal(counts, exp_counts)
        np.testing.assert_allclose(edges, exp_edges, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_histogram_with_range(self, force_cpu: bool) -> None:
        """Test histogram with specified range."""
        backend = GPUBackend(force_cpu=force_cpu)
        data = np.random.randn(10000).astype(np.float64)

        counts, edges = backend.histogram(data, bins=20, range=(-3.0, 3.0))

        assert len(counts) == 20
        assert edges[0] == -3.0
        assert edges[-1] == 3.0

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_histogram_density(self, force_cpu: bool) -> None:
        """Test histogram with density normalization."""
        backend = GPUBackend(force_cpu=force_cpu)
        data = np.random.randn(10000).astype(np.float64)

        counts, edges = backend.histogram(data, bins=50, density=True)

        # For density, integral should be approximately 1
        bin_widths = np.diff(edges)
        integral = np.sum(counts * bin_widths)
        np.testing.assert_allclose(integral, 1.0, rtol=0.1)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_dot_product_1d(self, force_cpu: bool) -> None:
        """Test dot product of 1D arrays."""
        backend = GPUBackend(force_cpu=force_cpu)
        a = np.random.randn(100).astype(np.float64)
        b = np.random.randn(100).astype(np.float64)

        result = backend.dot(a, b)

        expected = np.dot(a, b)
        assert isinstance(result, float | np.floating)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_dot_product_2d(self, force_cpu: bool) -> None:
        """Test dot product with 2D arrays."""
        backend = GPUBackend(force_cpu=force_cpu)
        a = np.random.randn(10, 20).astype(np.float64)
        b = np.random.randn(20, 30).astype(np.float64)

        result = backend.dot(a, b)

        expected = np.dot(a, b)
        assert result.shape == (10, 30)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_matmul_basic(self, force_cpu: bool) -> None:
        """Test matrix multiplication."""
        backend = GPUBackend(force_cpu=force_cpu)
        A = np.random.randn(50, 30).astype(np.float64)
        B = np.random.randn(30, 40).astype(np.float64)

        result = backend.matmul(A, B)

        expected = np.matmul(A, B)
        assert result.shape == (50, 40)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_matmul_vector(self, force_cpu: bool) -> None:
        """Test matrix-vector multiplication."""
        backend = GPUBackend(force_cpu=force_cpu)
        A = np.random.randn(100, 50).astype(np.float64)
        x = np.random.randn(50).astype(np.float64)

        result = backend.matmul(A, x)

        expected = np.matmul(A, x)
        assert result.shape == (100,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cpu_fallback_without_cupy(self) -> None:
        """Test graceful fallback when CuPy is not available."""
        # Mock import failure
        with patch.dict("sys.modules", {"cupy": None}):
            backend = GPUBackend()
            signal = np.random.randn(1024).astype(np.float64)

            # All operations should still work with NumPy
            result = backend.fft(signal)

            assert backend.gpu_available is False
            assert isinstance(result, np.ndarray)
            expected = np.fft.fft(signal)
            np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gpu_import_error_warning(self) -> None:
        """Test warning when GPU is accessible but fails initialization."""
        # Create a mock CuPy that raises during array creation
        mock_cp = MagicMock()
        mock_cp.array.side_effect = RuntimeError("CUDA not available")

        with patch.dict("sys.modules", {"cupy": mock_cp}):
            with pytest.warns(RuntimeWarning, match="GPU is not accessible"):
                backend = GPUBackend()
                _ = backend.gpu_available

            assert backend.gpu_available is False

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_empty_array_handling(self, force_cpu: bool) -> None:
        """Test handling of empty arrays."""
        backend = GPUBackend(force_cpu=force_cpu)
        empty = np.array([], dtype=np.float64)

        counts, _ = backend.histogram(empty, bins=10)

        assert len(counts) == 10
        assert counts.sum() == 0

    @pytest.mark.parametrize("force_cpu", [True, False])
    def test_dtype_preservation(self, force_cpu: bool) -> None:
        """Test that operations preserve appropriate dtypes."""
        backend = GPUBackend(force_cpu=force_cpu)

        # FFT: float64 -> complex128
        signal_f64 = np.random.randn(100).astype(np.float64)
        fft_result = backend.fft(signal_f64)
        assert fft_result.dtype == np.complex128

        # RFFT: float64 -> complex128
        rfft_result = backend.rfft(signal_f64)
        assert rfft_result.dtype == np.complex128

        # IRFFT: complex128 -> float64
        irfft_result = backend.irfft(rfft_result)
        assert irfft_result.dtype == np.float64

        # Convolution: float64 -> float64
        kernel = np.array([0.5, 0.5], dtype=np.float64)
        conv_result = backend.convolve(signal_f64, kernel)
        assert conv_result.dtype == np.float64


class TestGPUBackendIntegration:
    """Integration tests for GPU backend in realistic scenarios."""

    def test_spectral_analysis_workflow(self) -> None:
        """Test complete spectral analysis workflow."""
        # Simulate realistic signal processing pipeline
        backend = GPUBackend()

        # Generate test signal
        fs = 1000.0  # Sample rate
        t = np.arange(0, 1.0, 1 / fs)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        signal += 0.1 * np.random.randn(len(t))

        # Apply preprocessing: smoothing
        kernel = np.ones(5) / 5
        smoothed = backend.convolve(signal, kernel, mode="same")

        # Compute spectrum
        spectrum = backend.rfft(smoothed)

        # Verify result quality
        assert len(spectrum) == len(smoothed) // 2 + 1
        assert np.isfinite(spectrum).all()

    def test_pattern_detection_workflow(self) -> None:
        """Test pattern detection workflow using correlation."""
        backend = GPUBackend()

        # Create signal with known pattern
        signal = np.random.randn(10000).astype(np.float64)
        pattern = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)

        # Embed pattern at multiple locations
        signal[1000:1007] = pattern
        signal[5000:5007] = pattern

        # Find patterns
        correlation = backend.correlate(signal, pattern, mode="valid")

        # Should find two peaks
        threshold = 0.9 * np.max(correlation)
        peaks = np.where(correlation > threshold)[0]
        assert len(peaks) >= 2

    def test_statistical_analysis_workflow(self) -> None:
        """Test statistical analysis using histograms."""
        backend = GPUBackend()

        # Generate data from known distribution
        data = np.random.normal(loc=0.0, scale=2.0, size=100000)

        # Compute histogram
        counts, edges = backend.histogram(data, bins=100, density=True)

        # Verify histogram approximates normal distribution
        bin_centers = (edges[:-1] + edges[1:]) / 2
        # Should peak near 0 (allow wider tolerance for random variation)
        peak_idx = np.argmax(counts)
        assert -1.0 < bin_centers[peak_idx] < 1.0

    def test_matrix_computation_workflow(self) -> None:
        """Test matrix operations for pattern matching."""
        backend = GPUBackend()

        # Create data matrix and template matrix
        data_matrix = np.random.randn(100, 50).astype(np.float64)
        template = np.random.randn(50, 10).astype(np.float64)

        # Project data onto template space
        projection = backend.matmul(data_matrix, template)

        assert projection.shape == (100, 10)
        assert np.isfinite(projection).all()

    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_performance_with_varying_sizes(self, size: int) -> None:
        """Test backend handles different data sizes correctly."""
        backend = GPUBackend()

        signal = np.random.randn(size).astype(np.float64)

        # FFT
        spectrum = backend.rfft(signal)
        assert len(spectrum) == size // 2 + 1

        # Convolution
        kernel = np.ones(5) / 5
        smoothed = backend.convolve(signal, kernel, mode="same")
        assert len(smoothed) == size

        # Histogram
        counts, _ = backend.histogram(signal, bins=50)
        assert counts.sum() == size
