"""Tests for automatic backend selection.

This module tests the intelligent backend selector that chooses optimal
backends based on data size and system capabilities.
"""

from __future__ import annotations

import numpy as np

from oscura.core.backend_selector import (
    BackendCapabilities,
    BackendSelector,
    get_global_selector,
    get_system_capabilities,
    select_backend,
)


class TestBackendCapabilities:
    """Tests for BackendCapabilities dataclass."""

    def test_create_capabilities(self) -> None:
        """Should create BackendCapabilities object."""
        caps = BackendCapabilities(
            has_gpu=True,
            has_numba=True,
            has_dask=True,
            has_scipy=True,
            cpu_count=8,
            total_memory_gb=16.0,
            gpu_memory_gb=8.0,
        )

        assert caps.has_gpu is True
        assert caps.has_numba is True
        assert caps.cpu_count == 8
        assert caps.total_memory_gb == 16.0

    def test_minimal_capabilities(self) -> None:
        """Should work with minimal capabilities."""
        caps = BackendCapabilities(
            has_gpu=False,
            has_numba=False,
            has_dask=False,
            has_scipy=False,
            cpu_count=1,
            total_memory_gb=2.0,
            gpu_memory_gb=0.0,
        )

        assert caps.has_gpu is False
        assert caps.gpu_memory_gb == 0.0


class TestGetSystemCapabilities:
    """Tests for system capability detection."""

    def test_returns_capabilities_object(self) -> None:
        """Should return BackendCapabilities object."""
        caps = get_system_capabilities()

        assert isinstance(caps, BackendCapabilities)

    def test_has_required_fields(self) -> None:
        """Should have all required fields."""
        caps = get_system_capabilities()

        assert isinstance(caps.has_gpu, bool)
        assert isinstance(caps.has_numba, bool)
        assert isinstance(caps.has_dask, bool)
        assert isinstance(caps.has_scipy, bool)
        assert isinstance(caps.cpu_count, int)
        assert isinstance(caps.total_memory_gb, float)
        assert isinstance(caps.gpu_memory_gb, float)

    def test_reasonable_cpu_count(self) -> None:
        """CPU count should be reasonable."""
        caps = get_system_capabilities()

        assert caps.cpu_count >= 1
        assert caps.cpu_count <= 256  # Sanity check

    def test_reasonable_memory(self) -> None:
        """Memory should be positive."""
        caps = get_system_capabilities()

        assert caps.total_memory_gb > 0
        assert caps.gpu_memory_gb >= 0

    def test_gpu_memory_zero_without_gpu(self) -> None:
        """GPU memory should be 0 if no GPU."""
        caps = get_system_capabilities()

        if not caps.has_gpu:
            assert caps.gpu_memory_gb == 0.0


class TestBackendSelector:
    """Tests for BackendSelector class."""

    def test_initialization(self) -> None:
        """Should initialize with system capabilities."""
        selector = BackendSelector()

        assert hasattr(selector, "capabilities")
        assert isinstance(selector.capabilities, BackendCapabilities)

    def test_select_for_fft_small_data(self) -> None:
        """Should use scipy/numpy for small FFT."""
        selector = BackendSelector()

        # Small data: 10K samples
        backend = selector.select_for_fft(data_size=10_000)

        # Should be numpy or scipy (not gpu/dask)
        assert backend in ("numpy", "scipy")

    def test_select_for_fft_large_data_with_gpu(self) -> None:
        """Should use GPU for large FFT if available."""
        selector = BackendSelector()

        # Mock GPU available
        selector.capabilities.has_gpu = True

        # Large data: 50M samples
        backend = selector.select_for_fft(data_size=50_000_000)

        assert backend == "gpu"

    def test_select_for_fft_large_data_no_gpu(self) -> None:
        """Should use scipy for large FFT without GPU."""
        selector = BackendSelector()

        # Mock no GPU
        selector.capabilities.has_gpu = False
        selector.capabilities.has_scipy = True

        # Large data: 50M samples
        backend = selector.select_for_fft(data_size=50_000_000)

        assert backend == "scipy"

    def test_select_for_fft_huge_data_with_dask(self) -> None:
        """Should use Dask for huge datasets."""
        selector = BackendSelector()

        # Mock Dask available
        selector.capabilities.has_dask = True

        # Huge data: 200M samples
        backend = selector.select_for_fft(data_size=200_000_000)

        assert backend == "dask"

    def test_select_for_fft_fallback_to_numpy(self) -> None:
        """Should fall back to numpy if nothing else available."""
        selector = BackendSelector()

        # Mock no optional backends
        selector.capabilities.has_gpu = False
        selector.capabilities.has_scipy = False
        selector.capabilities.has_dask = False

        backend = selector.select_for_fft(data_size=1_000)

        assert backend == "numpy"

    def test_select_for_edge_detection_small(self) -> None:
        """Should use numpy for small edge detection."""
        selector = BackendSelector()

        backend = selector.select_for_edge_detection(data_size=10_000)

        assert backend == "numpy"

    def test_select_for_edge_detection_large_with_gpu(self) -> None:
        """Should use GPU for large edge detection if available."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True

        backend = selector.select_for_edge_detection(data_size=50_000_000)

        assert backend == "gpu"

    def test_select_for_edge_detection_with_hysteresis(self) -> None:
        """Should use numba for hysteresis with medium data."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_edge_detection(data_size=500_000, has_hysteresis=True)

        assert backend == "numba"

    def test_select_for_edge_detection_hysteresis_no_numba(self) -> None:
        """Should use numpy for hysteresis without numba."""
        selector = BackendSelector()
        selector.capabilities.has_numba = False

        backend = selector.select_for_edge_detection(data_size=500_000, has_hysteresis=True)

        assert backend == "numpy"

    def test_select_for_correlation_small(self) -> None:
        """Should use scipy for small correlation."""
        selector = BackendSelector()
        selector.capabilities.has_scipy = True

        backend = selector.select_for_correlation(
            signal1_size=10_000,
            signal2_size=100,
            mode="full",
        )

        assert backend == "scipy"

    def test_select_for_correlation_large_with_gpu(self) -> None:
        """Should use GPU for large correlation if available."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True

        backend = selector.select_for_correlation(
            signal1_size=20_000_000,
            signal2_size=100,
        )

        assert backend == "gpu"

    def test_select_for_correlation_memory_intensive(self) -> None:
        """Should use dask for memory-intensive correlation."""
        selector = BackendSelector()
        selector.capabilities.has_dask = True
        selector.capabilities.total_memory_gb = 1.0  # Very low memory (1GB)

        # Would use >50% RAM with large signals
        backend = selector.select_for_correlation(
            signal1_size=100_000_000,  # 100M samples
            signal2_size=100_000_000,  # Would need ~800MB just for input
            mode="full",
        )

        # Should use dask or scipy depending on memory calculation
        assert backend in ("dask", "scipy", "numpy")

    def test_select_for_protocol_decode_small(self) -> None:
        """Should use numpy for small protocol decode."""
        selector = BackendSelector()

        backend = selector.select_for_protocol_decode(
            data_size=100_000,
            protocol="uart",
        )

        assert backend == "numpy"

    def test_select_for_protocol_decode_large_with_numba(self) -> None:
        """Should use numba for large protocol decode."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_protocol_decode(
            data_size=5_000_000,
            protocol="spi",
        )

        assert backend == "numba"

    def test_select_for_pattern_matching_few_patterns(self) -> None:
        """Should use numpy for few patterns."""
        selector = BackendSelector()

        backend = selector.select_for_pattern_matching(
            data_size=1_000_000,
            pattern_count=5,
            approximate=False,
        )

        assert backend == "numpy"

    def test_select_for_pattern_matching_approximate(self) -> None:
        """Should use numpy/LSH for approximate matching."""
        selector = BackendSelector()

        backend = selector.select_for_pattern_matching(
            data_size=1_000_000,
            pattern_count=100,
            approximate=True,
        )

        assert backend == "numpy"  # LSH in numpy

    def test_select_for_pattern_matching_large(self) -> None:
        """Should use numba for large exact matching."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_pattern_matching(
            data_size=20_000_000,
            pattern_count=10,
            approximate=False,
        )

        assert backend == "numba"

    def test_estimate_correlation_output_full(self) -> None:
        """Should correctly estimate full correlation size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 100, "full")

        assert size == 1099  # 1000 + 100 - 1

    def test_estimate_correlation_output_valid(self) -> None:
        """Should correctly estimate valid correlation size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 100, "valid")

        assert size == 901  # max - min + 1

    def test_estimate_correlation_output_same(self) -> None:
        """Should correctly estimate same correlation size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 100, "same")

        assert size == 1000  # max(sizes)


class TestGlobalSelector:
    """Tests for global selector singleton."""

    def test_get_global_selector(self) -> None:
        """Should return BackendSelector instance."""
        selector = get_global_selector()

        assert isinstance(selector, BackendSelector)

    def test_singleton_behavior(self) -> None:
        """Should return same instance on repeated calls."""
        selector1 = get_global_selector()
        selector2 = get_global_selector()

        assert selector1 is selector2


class TestSelectBackendFunction:
    """Tests for select_backend convenience function."""

    def test_select_backend_fft(self) -> None:
        """Should select backend for FFT operation."""
        backend = select_backend("fft", data_size=10_000)

        assert backend in ("numpy", "scipy", "gpu", "dask")

    def test_select_backend_edge_detection(self) -> None:
        """Should select backend for edge detection."""
        backend = select_backend(
            "edge_detection",
            data_size=100_000,
            has_hysteresis=False,
        )

        assert backend in ("numpy", "numba", "gpu")

    def test_select_backend_correlation(self) -> None:
        """Should select backend for correlation."""
        backend = select_backend(
            "correlation",
            signal1_size=100_000,
            signal2_size=1_000,
            mode="full",
        )

        assert backend in ("numpy", "scipy", "gpu", "dask")

    def test_select_backend_protocol_decode(self) -> None:
        """Should select backend for protocol decode."""
        backend = select_backend(
            "protocol_decode",
            data_size=500_000,
            protocol="uart",
        )

        assert backend in ("numpy", "numba")

    def test_select_backend_pattern_matching(self) -> None:
        """Should select backend for pattern matching."""
        backend = select_backend(
            "pattern_matching",
            data_size=1_000_000,
            pattern_count=10,
            approximate=False,
        )

        assert backend in ("numpy", "numba")

    def test_select_backend_invalid_operation(self) -> None:
        """Should fall back to numpy for unknown operation."""
        backend = select_backend("unknown_operation")  # type: ignore[arg-type]

        assert backend == "numpy"

    def test_select_backend_with_dtype(self) -> None:
        """Should accept dtype parameter for FFT."""
        backend = select_backend(
            "fft",
            data_size=10_000,
            dtype=np.float32,
        )

        assert backend in ("numpy", "scipy", "gpu", "dask")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_data_size(self) -> None:
        """Should handle zero data size."""
        selector = BackendSelector()

        backend = selector.select_for_fft(data_size=0)

        assert backend in ("numpy", "scipy")

    def test_one_sample(self) -> None:
        """Should handle single sample."""
        selector = BackendSelector()

        backend = selector.select_for_fft(data_size=1)

        assert backend in ("numpy", "scipy")

    def test_boundary_sizes(self) -> None:
        """Should handle boundary sizes correctly."""
        selector = BackendSelector()

        # Just below threshold
        backend1 = selector.select_for_fft(data_size=9_999_999)
        # Just above threshold
        backend2 = selector.select_for_fft(data_size=10_000_001)

        # Behavior may differ at threshold
        assert backend1 in ("numpy", "scipy", "gpu")
        assert backend2 in ("numpy", "scipy", "gpu")

    def test_all_backends_disabled(self) -> None:
        """Should fall back to numpy when all backends disabled."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = False
        selector.capabilities.has_numba = False
        selector.capabilities.has_dask = False
        selector.capabilities.has_scipy = False

        backend = selector.select_for_fft(data_size=100_000_000)

        assert backend == "numpy"

    def test_negative_data_size(self) -> None:
        """Should handle negative size (treat as 0)."""
        selector = BackendSelector()

        # Negative size is invalid but shouldn't crash
        backend = selector.select_for_fft(data_size=-1000)

        assert backend in ("numpy", "scipy")

    def test_very_low_memory_system(self) -> None:
        """Should handle low memory systems."""
        selector = BackendSelector()
        selector.capabilities.total_memory_gb = 0.5  # 512MB

        # Should avoid memory-intensive operations
        backend = selector.select_for_correlation(
            signal1_size=1_000_000,
            signal2_size=1_000_000,
        )

        # Should prefer dask if available, else scipy
        assert backend in ("dask", "scipy", "numpy")


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_typical_oscilloscope_capture(self) -> None:
        """Should select appropriate backend for typical scope capture."""
        selector = BackendSelector()

        # Typical scope: 10M samples @ 1GS/s = 10ms capture
        backend = selector.select_for_fft(data_size=10_000_000)

        # Should use scipy or gpu depending on hardware
        assert backend in ("scipy", "gpu", "numpy")

    def test_long_term_monitoring(self) -> None:
        """Should use distributed backend for long captures."""
        selector = BackendSelector()
        selector.capabilities.has_dask = True

        # Long capture: 1B samples
        backend = selector.select_for_fft(data_size=1_000_000_000)

        assert backend == "dask"

    def test_uart_decode_typical(self) -> None:
        """Should select appropriate backend for UART decoding."""
        selector = BackendSelector()

        # Typical UART capture at 1MS/s, 1 second
        backend = selector.select_for_protocol_decode(
            data_size=1_000_000,
            protocol="uart",
        )

        assert backend in ("numpy", "numba")

    def test_gpu_accelerated_workflow(self) -> None:
        """Should use GPU when available for large operations."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True
        selector.capabilities.gpu_memory_gb = 8.0

        # Large FFT
        backend_fft = selector.select_for_fft(data_size=50_000_000)
        # Large correlation
        backend_corr = selector.select_for_correlation(
            signal1_size=20_000_000,
            signal2_size=1_000,
        )
        # Large edge detection
        backend_edge = selector.select_for_edge_detection(data_size=30_000_000)

        assert backend_fft == "gpu"
        assert backend_corr == "gpu"
        assert backend_edge == "gpu"
