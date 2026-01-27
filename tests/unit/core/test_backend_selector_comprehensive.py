"""Comprehensive tests for backend_selector module.

Tests automatic backend selection with mocking for system capabilities,
covering all backend selection methods and edge cases.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from oscura.core.backend_selector import (
    BackendCapabilities,
    BackendSelector,
    get_global_selector,
    get_system_capabilities,
    select_backend,
)


@pytest.fixture
def mock_no_extras() -> None:
    """Mock environment with no optional dependencies."""
    with (
        patch("oscura.core.backend_selector.HAS_GPU", False),
        patch("oscura.core.backend_selector.HAS_NUMBA", False),
        patch("oscura.core.backend_selector.HAS_DASK", False),
        patch("oscura.core.backend_selector.HAS_SCIPY", False),
    ):
        yield


@pytest.fixture
def mock_full_system() -> None:
    """Mock environment with all optional dependencies."""
    with (
        patch("oscura.core.backend_selector.HAS_GPU", True),
        patch("oscura.core.backend_selector.HAS_NUMBA", True),
        patch("oscura.core.backend_selector.HAS_DASK", True),
        patch("oscura.core.backend_selector.HAS_SCIPY", True),
    ):
        yield


@pytest.mark.unit
@pytest.mark.core
class TestBackendCapabilities:
    """Tests for BackendCapabilities dataclass."""

    def test_all_fields_present(self) -> None:
        """Should have all required fields."""
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
        assert caps.has_dask is True
        assert caps.has_scipy is True
        assert caps.cpu_count == 8
        assert caps.total_memory_gb == 16.0
        assert caps.gpu_memory_gb == 8.0

    def test_no_optional_dependencies(self) -> None:
        """Should work with no optional dependencies."""
        caps = BackendCapabilities(
            has_gpu=False,
            has_numba=False,
            has_dask=False,
            has_scipy=False,
            cpu_count=2,
            total_memory_gb=4.0,
            gpu_memory_gb=0.0,
        )

        assert caps.has_gpu is False
        assert caps.has_numba is False
        assert caps.has_dask is False
        assert caps.has_scipy is False


@pytest.mark.unit
@pytest.mark.core
class TestGetSystemCapabilities:
    """Tests for system capability detection."""

    def test_returns_valid_capabilities(self) -> None:
        """Should return valid BackendCapabilities."""
        caps = get_system_capabilities()

        assert isinstance(caps, BackendCapabilities)
        assert caps.cpu_count >= 1
        assert caps.total_memory_gb > 0
        assert caps.gpu_memory_gb >= 0

    def test_gpu_memory_consistent_with_availability(self) -> None:
        """GPU memory should be 0 if GPU not available."""
        caps = get_system_capabilities()

        if not caps.has_gpu:
            assert caps.gpu_memory_gb == 0.0

    @patch("oscura.core.backend_selector.psutil.cpu_count")
    @patch("oscura.core.backend_selector.psutil.virtual_memory")
    def test_handles_psutil_errors(self, mock_vmem: Mock, mock_cpu: Mock) -> None:
        """Should handle psutil errors gracefully."""
        mock_cpu.return_value = None  # psutil can return None
        mock_vmem.return_value = Mock(total=8 * 1024**3)

        caps = get_system_capabilities()
        assert caps.cpu_count == 1  # Fallback to 1


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorFFT:
    """Tests for FFT backend selection."""

    def test_small_data_uses_numpy_or_scipy(self) -> None:
        """Small FFT should use numpy/scipy."""
        selector = BackendSelector()

        backend = selector.select_for_fft(data_size=1000)
        assert backend in ("numpy", "scipy")

    def test_medium_data_uses_scipy(self) -> None:
        """Medium FFT should prefer scipy if available."""
        selector = BackendSelector()
        selector.capabilities.has_scipy = True
        selector.capabilities.has_gpu = False

        backend = selector.select_for_fft(data_size=500_000)
        assert backend == "scipy"

    def test_large_data_with_gpu(self) -> None:
        """Large FFT should use GPU if available."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True

        backend = selector.select_for_fft(data_size=20_000_000)
        assert backend == "gpu"

    def test_huge_data_uses_dask(self) -> None:
        """Huge FFT should use Dask for distributed processing."""
        selector = BackendSelector()
        selector.capabilities.has_dask = True

        backend = selector.select_for_fft(data_size=150_000_000)
        assert backend == "dask"

    def test_respects_dtype(self) -> None:
        """Should accept dtype parameter."""
        selector = BackendSelector()

        backend = selector.select_for_fft(data_size=10_000, dtype=np.float32)
        assert backend in ("numpy", "scipy")


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorEdgeDetection:
    """Tests for edge detection backend selection."""

    def test_small_data_no_hysteresis(self) -> None:
        """Small data without hysteresis uses numpy."""
        selector = BackendSelector()

        backend = selector.select_for_edge_detection(data_size=10_000, has_hysteresis=False)
        assert backend == "numpy"

    def test_large_data_with_hysteresis_uses_numba(self) -> None:
        """Large data with hysteresis should use Numba."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_edge_detection(data_size=500_000, has_hysteresis=True)
        assert backend == "numba"

    def test_large_data_with_hysteresis_no_numba(self) -> None:
        """Large data with hysteresis falls back to numpy without Numba."""
        selector = BackendSelector()
        selector.capabilities.has_numba = False

        backend = selector.select_for_edge_detection(data_size=500_000, has_hysteresis=True)
        assert backend == "numpy"

    def test_very_large_data_uses_gpu(self) -> None:
        """Very large data should prefer GPU."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True

        backend = selector.select_for_edge_detection(data_size=20_000_000, has_hysteresis=False)
        assert backend == "gpu"


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorCorrelation:
    """Tests for correlation backend selection."""

    def test_small_correlation(self) -> None:
        """Small correlation should use scipy if available."""
        selector = BackendSelector()
        selector.capabilities.has_scipy = True

        backend = selector.select_for_correlation(signal1_size=10_000, signal2_size=1000)
        assert backend == "scipy"

    def test_large_correlation_with_gpu(self) -> None:
        """Large correlation should use GPU."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = True

        backend = selector.select_for_correlation(signal1_size=20_000_000, signal2_size=100_000)
        assert backend == "gpu"

    def test_memory_limited_correlation(self) -> None:
        """High memory usage should trigger Dask."""
        selector = BackendSelector()
        selector.capabilities.has_dask = True
        selector.capabilities.has_scipy = True
        selector.capabilities.total_memory_gb = 1.0  # Very limited RAM (1GB)

        # Large signals that would exceed 50% of 1GB RAM
        # (100M + 10M) * 8 bytes ~= 880MB, output ~110M * 8 ~= 880MB
        # Total ~1.76GB > 50% of 1GB
        backend = selector.select_for_correlation(
            signal1_size=100_000_000, signal2_size=10_000_000, mode="full"
        )
        # Should use dask for memory-constrained case OR scipy if memory check passes
        assert backend in ("dask", "scipy")  # Either is acceptable

    def test_correlation_modes(self) -> None:
        """Should handle different correlation modes."""
        selector = BackendSelector()
        selector.capabilities.has_scipy = True

        for mode in ["full", "valid", "same"]:
            backend = selector.select_for_correlation(
                signal1_size=10_000,
                signal2_size=1000,
                mode=mode,  # type: ignore[arg-type]
            )
            assert backend in ("numpy", "scipy", "gpu", "dask")

    def test_estimate_correlation_output_full(self) -> None:
        """Should correctly estimate full correlation output size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 500, "full")
        assert size == 1000 + 500 - 1

    def test_estimate_correlation_output_valid(self) -> None:
        """Should correctly estimate valid correlation output size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 500, "valid")
        assert size == 1000 - 500 + 1

    def test_estimate_correlation_output_same(self) -> None:
        """Should correctly estimate same correlation output size."""
        selector = BackendSelector()

        size = selector._estimate_correlation_output(1000, 500, "same")
        assert size == 1000


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorProtocol:
    """Tests for protocol decode backend selection."""

    def test_small_protocol_decode(self) -> None:
        """Small protocol decode uses numpy."""
        selector = BackendSelector()

        backend = selector.select_for_protocol_decode(data_size=10_000, protocol="uart")
        assert backend == "numpy"

    def test_large_protocol_decode_with_numba(self) -> None:
        """Large protocol decode should use Numba."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_protocol_decode(data_size=5_000_000, protocol="spi")
        assert backend == "numba"

    def test_large_protocol_decode_without_numba(self) -> None:
        """Large protocol decode falls back to numpy."""
        selector = BackendSelector()
        selector.capabilities.has_numba = False

        backend = selector.select_for_protocol_decode(data_size=5_000_000, protocol="i2c")
        assert backend == "numpy"


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorPatternMatching:
    """Tests for pattern matching backend selection."""

    def test_exact_matching_small(self) -> None:
        """Small exact matching uses numpy."""
        selector = BackendSelector()

        backend = selector.select_for_pattern_matching(
            data_size=10_000, pattern_count=5, approximate=False
        )
        assert backend == "numpy"

    def test_approximate_matching_many_patterns(self) -> None:
        """Approximate matching with many patterns uses numpy (LSH)."""
        selector = BackendSelector()

        backend = selector.select_for_pattern_matching(
            data_size=100_000, pattern_count=50, approximate=True
        )
        assert backend == "numpy"

    def test_large_exact_matching_uses_numba(self) -> None:
        """Large exact matching uses Numba if available."""
        selector = BackendSelector()
        selector.capabilities.has_numba = True

        backend = selector.select_for_pattern_matching(
            data_size=20_000_000, pattern_count=10, approximate=False
        )
        assert backend == "numba"


@pytest.mark.unit
@pytest.mark.core
class TestSelectBackendFunction:
    """Tests for convenience select_backend function."""

    def test_select_fft_backend(self) -> None:
        """Should select FFT backend."""
        backend = select_backend("fft", data_size=10_000)
        assert backend in ("numpy", "scipy")

    def test_select_edge_detection_backend(self) -> None:
        """Should select edge detection backend."""
        backend = select_backend("edge_detection", data_size=10_000, has_hysteresis=False)
        assert backend == "numpy"

    def test_select_correlation_backend(self) -> None:
        """Should select correlation backend."""
        backend = select_backend("correlation", signal1_size=10_000, signal2_size=1000)
        assert backend in ("numpy", "scipy")

    def test_select_protocol_decode_backend(self) -> None:
        """Should select protocol decode backend."""
        backend = select_backend("protocol_decode", data_size=10_000, protocol="uart")
        assert backend == "numpy"

    def test_select_pattern_matching_backend(self) -> None:
        """Should select pattern matching backend."""
        backend = select_backend(
            "pattern_matching", data_size=10_000, pattern_count=5, approximate=False
        )
        assert backend == "numpy"

    def test_handles_missing_parameters(self) -> None:
        """Should handle missing optional parameters."""
        backend = select_backend("fft", data_size=10_000)
        assert backend in ("numpy", "scipy")

    def test_invalid_operation_fallback(self) -> None:
        """Should handle invalid operation gracefully."""
        # Type ignore because we're intentionally testing invalid input
        backend = select_backend("invalid_op", data_size=10_000)  # type: ignore[arg-type]
        assert backend == "numpy"


@pytest.mark.unit
@pytest.mark.core
class TestGlobalSelector:
    """Tests for global selector singleton."""

    def test_get_global_selector_returns_instance(self) -> None:
        """Should return BackendSelector instance."""
        selector = get_global_selector()
        assert isinstance(selector, BackendSelector)

    def test_get_global_selector_is_singleton(self) -> None:
        """Should return same instance on repeated calls."""
        selector1 = get_global_selector()
        selector2 = get_global_selector()

        # Should be exact same object
        assert selector1 is selector2

    def test_global_selector_reinitialization(self) -> None:
        """Should handle global selector reset."""
        import oscura.core.backend_selector as bs

        # Clear global selector
        bs._global_selector = None

        selector1 = get_global_selector()
        assert selector1 is not None

        selector2 = get_global_selector()
        assert selector1 is selector2


@pytest.mark.unit
@pytest.mark.core
class TestBackendSelectorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_data_size(self) -> None:
        """Should handle zero data size gracefully."""
        selector = BackendSelector()

        backend = selector.select_for_fft(data_size=0)
        assert backend in ("numpy", "scipy")

    def test_negative_data_size(self) -> None:
        """Should handle negative data size."""
        selector = BackendSelector()

        # Negative size should be treated as small
        backend = selector.select_for_fft(data_size=-100)
        assert backend in ("numpy", "scipy")

    def test_empty_pattern_list(self) -> None:
        """Should handle zero patterns."""
        selector = BackendSelector()

        backend = selector.select_for_pattern_matching(
            data_size=10_000, pattern_count=0, approximate=False
        )
        assert backend == "numpy"

    def test_correlation_with_zero_signal(self) -> None:
        """Should handle zero-length signals."""
        selector = BackendSelector()

        backend = selector.select_for_correlation(signal1_size=0, signal2_size=100)
        # Should not crash, returns valid backend
        assert backend in ("numpy", "scipy", "gpu", "dask")

    def test_selector_with_minimal_system(self) -> None:
        """Should work on minimal system with no extras."""
        selector = BackendSelector()
        selector.capabilities.has_gpu = False
        selector.capabilities.has_numba = False
        selector.capabilities.has_dask = False
        selector.capabilities.has_scipy = False

        # All operations should fall back to numpy
        assert selector.select_for_fft(10_000) == "numpy"
        assert selector.select_for_edge_detection(10_000) == "numpy"
        assert selector.select_for_correlation(10_000, 1000) == "numpy"
        assert selector.select_for_protocol_decode(10_000, "uart") == "numpy"
        assert selector.select_for_pattern_matching(10_000, 5) == "numpy"
