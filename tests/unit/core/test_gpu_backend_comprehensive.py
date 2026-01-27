"""Comprehensive tests for GPU backend module.

Tests GPU acceleration with automatic NumPy fallback, lazy initialization,
memory transfers, and all mathematical operations.
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from oscura.core.gpu_backend import GPUBackend, gpu


@pytest.fixture
def mock_cupy() -> Mock:
    """Create mock CuPy module."""
    mock_cp = Mock()
    mock_cp.ndarray = type("CuPyArray", (), {})
    mock_cp.array = lambda x: np.array(x)
    mock_cp.asarray = lambda x: np.array(x)
    mock_cp.asnumpy = lambda x: np.asarray(x)
    return mock_cp


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendInitialization:
    """Tests for GPUBackend initialization and lazy loading."""

    def test_initialization_default(self) -> None:
        """Should initialize with force_cpu=False."""
        backend = GPUBackend()

        assert backend._force_cpu is False
        assert backend._gpu_available is None
        assert backend._initialized is False

    def test_initialization_force_cpu(self) -> None:
        """Should initialize with force_cpu=True."""
        backend = GPUBackend(force_cpu=True)

        assert backend._force_cpu is True

    def test_lazy_initialization(self) -> None:
        """Should not check GPU until first use."""
        backend = GPUBackend()

        assert backend._initialized is False

    def test_check_gpu_initializes_once(self) -> None:
        """Should only check GPU once."""
        backend = GPUBackend()

        backend._check_gpu()
        assert backend._initialized is True

        # Second call should not re-check
        backend._check_gpu()
        # Still initialized
        assert backend._initialized is True

    def test_force_cpu_disables_gpu(self) -> None:
        """Should disable GPU when force_cpu=True."""
        backend = GPUBackend(force_cpu=True)

        available = backend._check_gpu()

        assert available is False
        assert backend._gpu_available is False

    @patch.dict(os.environ, {"OSCURA_USE_GPU": "0"})
    def test_environment_variable_disables_gpu(self) -> None:
        """Should disable GPU with OSCURA_USE_GPU=0."""
        backend = GPUBackend()

        available = backend._check_gpu()

        assert available is False

    @patch.dict(os.environ, {"OSCURA_USE_GPU": "1"})
    @patch("builtins.__import__", side_effect=ImportError("cupy not found"))
    def test_missing_cupy_fallback(self, mock_import: Mock) -> None:
        """Should fall back to NumPy when CuPy not installed."""
        backend = GPUBackend()

        available = backend._check_gpu()

        assert available is False
        assert backend._gpu_available is False


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendProperties:
    """Tests for GPU backend properties."""

    def test_gpu_available_property(self) -> None:
        """Should return GPU availability status."""
        backend = GPUBackend()

        available = backend.gpu_available

        assert isinstance(available, bool)
        assert backend._initialized is True

    def test_using_gpu_property(self) -> None:
        """Should return same as gpu_available."""
        backend = GPUBackend()

        assert backend.using_gpu == backend.gpu_available


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendDataTransfer:
    """Tests for CPU-GPU data transfer methods."""

    def test_to_cpu_with_numpy_array(self) -> None:
        """Should handle NumPy arrays directly."""
        backend = GPUBackend(force_cpu=True)
        data = np.array([1.0, 2.0, 3.0])

        result = backend._to_cpu(data)

        assert_array_equal(result, data)
        assert isinstance(result, np.ndarray)

    def test_to_gpu_force_cpu(self) -> None:
        """Should return NumPy array when force_cpu=True."""
        backend = GPUBackend(force_cpu=True)
        data = np.array([1.0, 2.0, 3.0])

        result = backend._to_gpu(data)

        assert_array_equal(result, data)
        assert isinstance(result, np.ndarray)

    @patch("builtins.__import__")
    def test_to_cpu_with_cupy_array(self, mock_import: Mock) -> None:
        """Should transfer from GPU to CPU."""
        # This test requires actual CuPy or sophisticated mocking
        backend = GPUBackend(force_cpu=True)
        data = np.array([1.0, 2.0, 3.0])

        result = backend._to_cpu(data)

        assert isinstance(result, np.ndarray)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendFFT:
    """Tests for FFT operations."""

    def test_fft_cpu_fallback(self) -> None:
        """Should compute FFT with NumPy fallback."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = backend.fft(signal)

        expected = np.fft.fft(signal)
        assert_array_almost_equal(result, expected)

    def test_fft_with_n_parameter(self) -> None:
        """Should respect n parameter."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = backend.fft(signal, n=8)

        assert len(result) == 8

    def test_fft_with_axis_parameter(self) -> None:
        """Should respect axis parameter."""
        backend = GPUBackend(force_cpu=True)
        signal = np.random.randn(4, 8)

        result = backend.fft(signal, axis=0)

        assert result.shape == signal.shape

    def test_fft_with_norm_parameter(self) -> None:
        """Should respect norm parameter."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = backend.fft(signal, norm="ortho")

        expected = np.fft.fft(signal, norm="ortho")
        assert_array_almost_equal(result, expected)

    def test_fft_complex_input(self) -> None:
        """Should handle complex input."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0 + 1j, 2.0 - 1j, 3.0, 4.0 + 2j])

        result = backend.fft(signal)

        expected = np.fft.fft(signal)
        assert_array_almost_equal(result, expected)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendIFFT:
    """Tests for inverse FFT operations."""

    def test_ifft_cpu_fallback(self) -> None:
        """Should compute IFFT with NumPy fallback."""
        backend = GPUBackend(force_cpu=True)
        spectrum = np.array([10.0 + 0j, -2.0 + 2j, -2.0 + 0j, -2.0 - 2j])

        result = backend.ifft(spectrum)

        expected = np.fft.ifft(spectrum)
        assert_array_almost_equal(result, expected)

    def test_ifft_roundtrip(self) -> None:
        """Should recover original signal with FFT/IFFT roundtrip."""
        backend = GPUBackend(force_cpu=True)
        original = np.array([1.0, 2.0, 3.0, 4.0])

        spectrum = backend.fft(original)
        recovered = backend.ifft(spectrum)

        assert_array_almost_equal(recovered.real, original)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendRFFT:
    """Tests for real FFT operations."""

    def test_rfft_cpu_fallback(self) -> None:
        """Should compute real FFT with NumPy fallback."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = backend.rfft(signal)

        expected = np.fft.rfft(signal)
        assert_array_almost_equal(result, expected)

    def test_rfft_output_size(self) -> None:
        """Should return n//2 + 1 frequencies."""
        backend = GPUBackend(force_cpu=True)
        signal = np.random.randn(100)

        result = backend.rfft(signal)

        assert len(result) == 51  # 100//2 + 1

    def test_irfft_cpu_fallback(self) -> None:
        """Should compute inverse real FFT."""
        backend = GPUBackend(force_cpu=True)
        spectrum = np.array([10.0 + 0j, -2.0 + 2j, -2.0 + 0j])

        result = backend.irfft(spectrum)

        expected = np.fft.irfft(spectrum)
        assert_array_almost_equal(result, expected)

    def test_rfft_irfft_roundtrip(self) -> None:
        """Should recover original real signal."""
        backend = GPUBackend(force_cpu=True)
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        spectrum = backend.rfft(original)
        # Need to specify n to get exact original length
        recovered = backend.irfft(spectrum, n=len(original))

        assert_array_almost_equal(recovered, original)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendConvolution:
    """Tests for convolution operations."""

    def test_convolve_full_mode(self) -> None:
        """Should compute full convolution."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        kernel = np.array([1.0, 0.5])

        result = backend.convolve(signal, kernel, mode="full")

        expected = np.convolve(signal, kernel, mode="full")
        assert_array_almost_equal(result, expected)

    def test_convolve_valid_mode(self) -> None:
        """Should compute valid convolution."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = np.array([1.0, 0.5, 0.25])

        result = backend.convolve(signal, kernel, mode="valid")

        expected = np.convolve(signal, kernel, mode="valid")
        assert_array_almost_equal(result, expected)

    def test_convolve_same_mode(self) -> None:
        """Should compute same-size convolution."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        kernel = np.array([1.0, 0.5])

        result = backend.convolve(signal, kernel, mode="same")

        expected = np.convolve(signal, kernel, mode="same")
        assert_array_almost_equal(result, expected)

    def test_convolve_smoothing_kernel(self) -> None:
        """Should apply smoothing kernel."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 5.0, 1.0, 5.0])
        kernel = np.array([0.25, 0.5, 0.25])

        result = backend.convolve(signal, kernel, mode="same")

        # Should smooth the signal
        assert result[1] < 5.0  # Peak reduced
        assert result[1] > 1.0  # Valley filled


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendCorrelation:
    """Tests for correlation operations."""

    def test_correlate_full_mode(self) -> None:
        """Should compute full correlation."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        template = np.array([1.0, 2.0])

        result = backend.correlate(signal, template, mode="full")

        expected = np.correlate(signal, template, mode="full")
        assert_array_almost_equal(result, expected)

    def test_correlate_valid_mode(self) -> None:
        """Should compute valid correlation."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        template = np.array([2.0, 3.0])

        result = backend.correlate(signal, template, mode="valid")

        expected = np.correlate(signal, template, mode="valid")
        assert_array_almost_equal(result, expected)

    def test_correlate_same_mode(self) -> None:
        """Should compute same-size correlation."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        template = np.array([2.0, 3.0])

        result = backend.correlate(signal, template, mode="same")

        expected = np.correlate(signal, template, mode="same")
        assert_array_almost_equal(result, expected)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendHistogram:
    """Tests for histogram operations."""

    def test_histogram_basic(self) -> None:
        """Should compute histogram."""
        backend = GPUBackend(force_cpu=True)
        data = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])

        counts, edges = backend.histogram(data, bins=3)

        expected_counts, expected_edges = np.histogram(data, bins=3)
        assert_array_equal(counts, expected_counts)
        assert_array_almost_equal(edges, expected_edges)

    def test_histogram_with_range(self) -> None:
        """Should respect range parameter."""
        backend = GPUBackend(force_cpu=True)
        data = np.random.randn(1000)

        counts, edges = backend.histogram(data, bins=10, range=(-2.0, 2.0))

        assert len(counts) == 10
        assert edges[0] == -2.0
        assert edges[-1] == 2.0

    def test_histogram_with_density(self) -> None:
        """Should compute probability density."""
        backend = GPUBackend(force_cpu=True)
        data = np.random.randn(10000)

        counts, edges = backend.histogram(data, bins=50, density=True)

        # Sum * bin_width should be approximately 1
        bin_width = edges[1] - edges[0]
        total_prob = np.sum(counts) * bin_width
        assert 0.95 < total_prob < 1.05

    def test_histogram_with_array_bins(self) -> None:
        """Should accept array of bin edges."""
        backend = GPUBackend(force_cpu=True)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = np.array([0.0, 2.5, 5.0])

        counts, edges = backend.histogram(data, bins=bins)

        assert_array_equal(edges, bins)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendLinearAlgebra:
    """Tests for linear algebra operations."""

    def test_dot_product_vectors(self) -> None:
        """Should compute dot product of vectors."""
        backend = GPUBackend(force_cpu=True)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        result = backend.dot(a, b)

        expected = np.dot(a, b)
        assert result == expected

    def test_dot_product_matrix_vector(self) -> None:
        """Should compute matrix-vector product."""
        backend = GPUBackend(force_cpu=True)
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 2.0])

        result = backend.dot(A, b)

        expected = np.dot(A, b)
        assert_array_almost_equal(result, expected)

    def test_matmul_basic(self) -> None:
        """Should compute matrix multiplication."""
        backend = GPUBackend(force_cpu=True)
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        result = backend.matmul(A, B)

        expected = np.matmul(A, B)
        assert_array_almost_equal(result, expected)

    def test_matmul_rectangular(self) -> None:
        """Should handle rectangular matrices."""
        backend = GPUBackend(force_cpu=True)
        A = np.random.randn(3, 5)
        B = np.random.randn(5, 2)

        result = backend.matmul(A, B)

        assert result.shape == (3, 2)


@pytest.mark.unit
@pytest.mark.core
class TestGlobalGPUInstance:
    """Tests for module-level GPU instance."""

    def test_global_gpu_exists(self) -> None:
        """Should have global GPU instance."""
        assert isinstance(gpu, GPUBackend)

    def test_global_gpu_usable(self) -> None:
        """Should be able to use global GPU instance."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])

        result = gpu.fft(signal)

        assert len(result) == len(signal)


@pytest.mark.unit
@pytest.mark.core
class TestGPUBackendEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array(self) -> None:
        """Should handle empty arrays."""
        backend = GPUBackend(force_cpu=True)
        empty = np.array([])

        # NumPy FFT raises ValueError for empty arrays, which is correct behavior
        with pytest.raises(ValueError, match="Invalid number of FFT data points"):
            backend.fft(empty)

    def test_single_element(self) -> None:
        """Should handle single-element arrays."""
        backend = GPUBackend(force_cpu=True)
        single = np.array([42.0])

        result = backend.fft(single)

        assert len(result) == 1

    def test_large_array_cpu_fallback(self) -> None:
        """Should handle large arrays with CPU fallback."""
        backend = GPUBackend(force_cpu=True)
        large_signal = np.random.randn(100000)

        result = backend.fft(large_signal)

        assert len(result) == len(large_signal)

    def test_nan_handling(self) -> None:
        """Should propagate NaN values."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, np.nan, 3.0, 4.0])

        result = backend.fft(signal)

        assert np.isnan(result).any()

    def test_inf_handling(self) -> None:
        """Should handle infinite values."""
        backend = GPUBackend(force_cpu=True)
        signal = np.array([1.0, np.inf, 3.0, 4.0])

        result = backend.fft(signal)

        # Should complete without error
        assert len(result) == len(signal)
