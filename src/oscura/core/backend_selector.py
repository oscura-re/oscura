"""Automatic backend selection for optimal performance.

This module provides intelligent backend selection based on data characteristics,
available hardware, and performance requirements. Automatically chooses between
NumPy, Numba, GPU (CuPy), and distributed (Dask) backends.

Usage:
    from oscura.core.backend_selector import BackendSelector, select_backend

    selector = BackendSelector()
    backend = selector.select_for_fft(signal_size=10_000_000)
    # Returns 'gpu' if available, else 'scipy'

Performance decision tree:
    - Small data (<100K): NumPy/SciPy
    - Medium data (100K-10M): Numba JIT
    - Large data (>10M): GPU if available, else Numba
    - Huge data (>1GB): Dask distributed

Example:
    >>> from oscura.core.backend_selector import select_backend
    >>> import numpy as np
    >>>
    >>> data = np.random.randn(50_000_000)
    >>> backend = select_backend('fft', data_size=len(data))
    >>> print(f"Selected backend: {backend}")  # 'gpu' or 'scipy'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import psutil

# Check available backends
try:
    from oscura.core.gpu_backend import gpu

    HAS_GPU = gpu.gpu_available
except (ImportError, AttributeError):
    HAS_GPU = False

try:
    import numba

    HAS_NUMBA = True
    del numba
except ImportError:
    HAS_NUMBA = False

try:
    import dask.array  # type: ignore[import-not-found]  # Optional dependency

    HAS_DASK = True
    del dask
except ImportError:
    HAS_DASK = False

try:
    import scipy.fft

    HAS_SCIPY = True
    del scipy
except ImportError:
    HAS_SCIPY = False


BackendType = Literal["numpy", "scipy", "numba", "gpu", "dask"]


@dataclass
class BackendCapabilities:
    """Available backend capabilities on this system.

    Attributes:
        has_gpu: Whether GPU (CuPy) is available.
        has_numba: Whether Numba JIT is available.
        has_dask: Whether Dask distributed is available.
        has_scipy: Whether SciPy is available.
        cpu_count: Number of CPU cores.
        total_memory_gb: Total system RAM in GB.
        gpu_memory_gb: GPU memory in GB (0 if no GPU).
    """

    has_gpu: bool
    has_numba: bool
    has_dask: bool
    has_scipy: bool
    cpu_count: int
    total_memory_gb: float
    gpu_memory_gb: float


def get_system_capabilities() -> BackendCapabilities:
    """Detect available backends and system resources.

    Returns:
        BackendCapabilities object with system information.

    Example:
        >>> caps = get_system_capabilities()
        >>> if caps.has_gpu:
        ...     print(f"GPU available with {caps.gpu_memory_gb:.1f} GB memory")
    """
    # CPU and memory info
    cpu_count = psutil.cpu_count(logical=False) or 1
    total_memory = psutil.virtual_memory().total
    total_memory_gb = total_memory / (1024**3)

    # GPU memory
    gpu_memory_gb = 0.0
    if HAS_GPU:
        try:
            from oscura.core.gpu_backend import gpu

            # Get GPU memory in bytes, convert to GB
            gpu_memory_gb = gpu.get_memory_info()[1] / (1024**3)  # type: ignore[attr-defined]
        except Exception:
            gpu_memory_gb = 0.0

    return BackendCapabilities(
        has_gpu=HAS_GPU,
        has_numba=HAS_NUMBA,
        has_dask=HAS_DASK,
        has_scipy=HAS_SCIPY,
        cpu_count=cpu_count,
        total_memory_gb=total_memory_gb,
        gpu_memory_gb=gpu_memory_gb,
    )


class BackendSelector:
    """Intelligent backend selector for optimal performance.

    This class analyzes data characteristics and system capabilities to
    automatically select the best backend for each operation.

    Example:
        >>> selector = BackendSelector()
        >>> # For FFT on 50M samples
        >>> backend = selector.select_for_fft(50_000_000)
        >>> # For edge detection with hysteresis
        >>> backend = selector.select_for_edge_detection(1_000_000, has_hysteresis=True)
    """

    def __init__(self) -> None:
        """Initialize backend selector with system capabilities."""
        self.capabilities = get_system_capabilities()

    def select_for_fft(
        self,
        data_size: int,
        dtype: type = np.float64,
    ) -> BackendType:
        """Select optimal backend for FFT operations.

        Args:
            data_size: Number of samples in signal.
            dtype: Data type (affects memory usage).

        Returns:
            BackendType: 'numpy', 'scipy', 'gpu', or 'dask'.

        Example:
            >>> selector = BackendSelector()
            >>> backend = selector.select_for_fft(10_000_000)
        """
        # Decision tree based on data size
        if data_size > 100_000_000 and self.capabilities.has_dask:
            # Huge data: use distributed
            return "dask"
        elif data_size > 10_000_000 and self.capabilities.has_gpu:
            # Large data + GPU: use GPU
            return "gpu"
        elif self.capabilities.has_scipy:
            # Use scipy.fft with workers (faster than numpy.fft)
            return "scipy"
        else:
            # Fallback to numpy
            return "numpy"

    def select_for_edge_detection(
        self,
        data_size: int,
        has_hysteresis: bool = False,
    ) -> BackendType:
        """Select optimal backend for edge detection.

        Args:
            data_size: Number of samples in signal.
            has_hysteresis: Whether hysteresis is used (affects vectorization).

        Returns:
            BackendType: 'numpy', 'numba', or 'gpu'.

        Example:
            >>> selector = BackendSelector()
            >>> backend = selector.select_for_edge_detection(5_000_000, has_hysteresis=True)
        """
        if data_size > 10_000_000 and self.capabilities.has_gpu:
            return "gpu"
        elif has_hysteresis and self.capabilities.has_numba and data_size > 100_000:
            return "numba"
        elif has_hysteresis:
            return "numpy"  # Actually uses Python for hysteresis state machine
        else:
            return "numpy"  # Vectorized without hysteresis

    def select_for_correlation(
        self,
        signal1_size: int,
        signal2_size: int,
        mode: Literal["full", "valid", "same"] = "full",
    ) -> BackendType:
        """Select optimal backend for correlation.

        Args:
            signal1_size: Size of first signal.
            signal2_size: Size of second signal.
            mode: Correlation mode.

        Returns:
            Backend name.

        Example:
            >>> selector = BackendSelector()
            >>> backend = selector.select_for_correlation(1_000_000, 10_000)
        """
        total_size = signal1_size + signal2_size
        output_size = self._estimate_correlation_output(signal1_size, signal2_size, mode)

        # Estimate memory
        total_memory_mb = (total_size + output_size) * 8 / (1024**2)

        if total_memory_mb > self.capabilities.total_memory_gb * 1024 * 0.5:
            # Would use >50% RAM: use chunked/streaming
            return "dask" if self.capabilities.has_dask else "numpy"
        elif signal1_size > 10_000_000 and self.capabilities.has_gpu:
            return "gpu"
        elif self.capabilities.has_scipy:
            return "scipy"
        else:
            return "numpy"

    def select_for_protocol_decode(
        self,
        data_size: int,
        protocol: str,
    ) -> BackendType:
        """Select optimal backend for protocol decoding.

        Args:
            data_size: Number of samples in signal.
            protocol: Protocol name (e.g., 'uart', 'spi', 'i2c').

        Returns:
            Backend name.

        Example:
            >>> selector = BackendSelector()
            >>> backend = selector.select_for_protocol_decode(5_000_000, 'uart')
        """
        # Protocol decoders use edge detection + state machines
        # Large signals benefit from Numba-compiled state machines
        if data_size > 1_000_000 and self.capabilities.has_numba:
            return "numba"
        else:
            return "numpy"

    def select_for_pattern_matching(
        self,
        data_size: int,
        pattern_count: int,
        approximate: bool = False,
    ) -> BackendType:
        """Select optimal backend for pattern matching.

        Args:
            data_size: Size of data to search.
            pattern_count: Number of patterns.
            approximate: Whether approximate matching is acceptable.

        Returns:
            Backend name.

        Example:
            >>> selector = BackendSelector()
            >>> backend = selector.select_for_pattern_matching(1_000_000, 100, approximate=True)
        """
        # For approximate matching with many patterns, LSH is best
        # Otherwise use standard string matching
        if approximate and pattern_count > 10:
            return "numpy"  # LSH implementation in NumPy
        elif data_size > 10_000_000:
            return "numba"
        else:
            return "numpy"

    def _estimate_correlation_output(
        self,
        size1: int,
        size2: int,
        mode: Literal["full", "valid", "same"],
    ) -> int:
        """Estimate output size of correlation.

        Args:
            size1: Size of first signal.
            size2: Size of second signal.
            mode: Correlation mode.

        Returns:
            Estimated output size in samples.
        """
        if mode == "full":
            return size1 + size2 - 1
        elif mode == "valid":
            return max(size1, size2) - min(size1, size2) + 1
        else:  # same
            return max(size1, size2)


# Global selector instance
_global_selector: BackendSelector | None = None


def get_global_selector() -> BackendSelector:
    """Get global backend selector instance (singleton).

    Returns:
        Global BackendSelector instance.

    Example:
        >>> selector = get_global_selector()
        >>> backend = selector.select_for_fft(1_000_000)
    """
    global _global_selector
    if _global_selector is None:
        _global_selector = BackendSelector()
    return _global_selector


def select_backend(
    operation: Literal[
        "fft", "edge_detection", "correlation", "protocol_decode", "pattern_matching"
    ],
    **kwargs: int | str | bool,
) -> BackendType:
    """Convenience function to select backend for an operation.

    Args:
        operation: Type of operation.
        **kwargs: Operation-specific parameters.

    Returns:
        Selected backend name.

    Example:
        >>> backend = select_backend('fft', data_size=10_000_000)
        >>> backend = select_backend('edge_detection', data_size=5_000_000, has_hysteresis=True)
        >>> backend = select_backend('correlation', signal1_size=1_000_000, signal2_size=10_000)
    """
    selector = get_global_selector()

    if operation == "fft":
        return selector.select_for_fft(
            data_size=int(kwargs.get("data_size", 0)),
            dtype=kwargs.get("dtype", np.float64),  # type: ignore[arg-type]
        )
    elif operation == "edge_detection":
        return selector.select_for_edge_detection(
            data_size=int(kwargs.get("data_size", 0)),
            has_hysteresis=bool(kwargs.get("has_hysteresis", False)),
        )
    elif operation == "correlation":
        return selector.select_for_correlation(
            signal1_size=int(kwargs.get("signal1_size", 0)),
            signal2_size=int(kwargs.get("signal2_size", 0)),
            mode=kwargs.get("mode", "full"),  # type: ignore[arg-type]
        )
    elif operation == "protocol_decode":
        return selector.select_for_protocol_decode(
            data_size=int(kwargs.get("data_size", 0)),
            protocol=str(kwargs.get("protocol", "")),
        )
    elif operation == "pattern_matching":
        return selector.select_for_pattern_matching(
            data_size=int(kwargs.get("data_size", 0)),
            pattern_count=int(kwargs.get("pattern_count", 0)),
            approximate=bool(kwargs.get("approximate", False)),
        )
    else:
        return "numpy"  # type: ignore[unreachable]


__all__ = [
    "BackendCapabilities",
    "BackendSelector",
    "BackendType",
    "get_global_selector",
    "get_system_capabilities",
    "select_backend",
]
