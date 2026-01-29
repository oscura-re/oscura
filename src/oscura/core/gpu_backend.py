"""GPU acceleration backend with automatic numpy fallback.

This module provides optional GPU acceleration using CuPy with seamless
fallback to NumPy when CuPy is unavailable or GPU processing is disabled.

The GPU backend is lazy-initialized and memory-safe, automatically transferring
data to/from GPU as needed. GPU usage can be controlled via the environment
variable OSCURA_USE_GPU (0 to disable, 1 to enable).


Example:
    >>> from oscura.core.gpu_backend import gpu
    >>> # Automatically uses GPU if available, numpy otherwise
    >>> freqs = gpu.fft(signal_data)
    >>>
    >>> # Force CPU-only operation
    >>> from oscura.core.gpu_backend import GPUBackend
    >>> cpu_only = GPUBackend(force_cpu=True)
    >>> freqs = cpu_only.fft(signal_data)

Configuration:
    Set OSCURA_USE_GPU environment variable to control GPU usage:
    - OSCURA_USE_GPU=0: Force CPU-only operation
    - OSCURA_USE_GPU=1: Enable GPU if available (default)

References:
    - CuPy documentation: https://docs.cupy.dev/
    - NumPy FFT module: https://numpy.org/doc/stable/reference/routines.fft.html
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GPUBackend:
    """Optional GPU acceleration with transparent numpy fallback.

    This class provides GPU-accelerated versions of common array operations
    with automatic fallback to NumPy when CuPy is unavailable or GPU is disabled.

    GPU availability is checked lazily on first use, and data is automatically
    transferred between CPU and GPU as needed for transparent operation.

    Args:
        force_cpu: If True, always use CPU (NumPy) even if GPU is available.
            Useful for testing or when GPU memory is limited.

    Attributes:
        gpu_available: True if CuPy is available and GPU is enabled.
        using_gpu: True if currently using GPU backend (may differ from
            gpu_available if lazy initialization hasn't occurred yet).

    Example:
        >>> backend = GPUBackend()
        >>> if backend.gpu_available:
        ...     print("Using GPU acceleration")
        >>> else:
        ...     print("Using CPU (NumPy) fallback")
        >>>
        >>> # All operations work identically regardless of backend
        >>> result = backend.fft(data)

    References:
        PERF-001 through PERF-004: GPU acceleration requirements
    """

    def __init__(self, force_cpu: bool = False) -> None:
        """Initialize GPU backend with optional CPU-only mode.

        Args:
            force_cpu: If True, never use GPU even if available.
        """
        self._force_cpu = force_cpu
        self._gpu_available: bool | None = None
        self._cp: Any = None  # CuPy module if available
        self._initialized = False

    def _check_gpu(self) -> bool:
        """Check if GPU/CuPy is available and should be used.

        This is called lazily on first operation to avoid import overhead
        when GPU is not needed.

        Returns:
            True if GPU should be used, False to fall back to NumPy.
        """
        if self._initialized:
            return self._gpu_available or False

        self._initialized = True

        # Check environment variable override
        use_gpu_env = os.environ.get("OSCURA_USE_GPU", "1")
        if use_gpu_env == "0" or self._force_cpu:
            self._gpu_available = False
            return False

        # Try to import CuPy
        try:
            import cupy as cp  # type: ignore[import-not-found]

            # Verify GPU is actually accessible
            try:
                # Try a simple operation to verify GPU works
                _ = cp.array([1.0])
                self._cp = cp
                self._gpu_available = True
                return True
            except Exception as e:
                warnings.warn(
                    f"CuPy is installed but GPU is not accessible: {e}. Falling back to NumPy.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._gpu_available = False
                return False
        except ImportError:
            # CuPy not installed - silent fallback
            self._gpu_available = False
            return False

    @property
    def gpu_available(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if CuPy is available and GPU can be used.
        """
        if not self._initialized:
            self._check_gpu()
        return self._gpu_available or False

    @property
    def using_gpu(self) -> bool:
        """Alias for gpu_available for backwards compatibility.

        Returns:
            True if currently using GPU backend.
        """
        return self.gpu_available

    def _to_cpu(self, array: Any) -> NDArray[Any]:
        """Transfer array from GPU to CPU if needed.

        Args:
            array: Array that may be on GPU or CPU.

        Returns:
            NumPy array on CPU.
        """
        if self.gpu_available and self._cp is not None:
            if isinstance(array, self._cp.ndarray):
                return self._cp.asnumpy(array)  # type: ignore[no-any-return]
        return np.asarray(array)

    def _to_gpu(self, array: NDArray[Any]) -> Any:
        """Transfer array from CPU to GPU if GPU is enabled.

        Args:
            array: NumPy array on CPU.

        Returns:
            CuPy array on GPU if GPU is available, otherwise NumPy array.
        """
        if self.gpu_available and self._cp is not None:
            return self._cp.asarray(array)
        return array

    def fft(
        self,
        data: NDArray[np.complex128] | NDArray[np.float64],
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complex128]:
        """GPU-accelerated FFT with automatic fallback to NumPy.

        Computes the one-dimensional discrete Fourier Transform using GPU
        if available, otherwise falls back to NumPy.

        Args:
            data: Input array (can be real or complex).
            n: Length of the transformed axis. If None, uses data.shape[axis].
            axis: Axis over which to compute the FFT.
            norm: Normalization mode ("backward", "ortho", or "forward").

        Returns:
            Complex-valued FFT of the input array (always NumPy array on CPU).

        Example:
            >>> signal = np.random.randn(1000)
            >>> spectrum = gpu.fft(signal)
            >>> # Result is always NumPy array, regardless of backend

        References:
            SPE-001: Standard FFT Computation
            PERF-001: GPU-accelerated FFT
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            result = self._cp.fft.fft(gpu_data, n=n, axis=axis, norm=norm)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.fft.fft(data, n=n, axis=axis, norm=norm)

    def ifft(
        self,
        data: NDArray[np.complex128],
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complex128]:
        """GPU-accelerated inverse FFT with automatic fallback.

        Computes the one-dimensional inverse discrete Fourier Transform.

        Args:
            data: Input complex array.
            n: Length of the transformed axis. If None, uses data.shape[axis].
            axis: Axis over which to compute the IFFT.
            norm: Normalization mode ("backward", "ortho", or "forward").

        Returns:
            Complex-valued IFFT of the input array (always NumPy array on CPU).

        Example:
            >>> spectrum = np.fft.fft(signal)
            >>> recovered = gpu.ifft(spectrum)

        References:
            SPE-001: Standard FFT Computation
            PERF-001: GPU-accelerated FFT
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            result = self._cp.fft.ifft(gpu_data, n=n, axis=axis, norm=norm)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.fft.ifft(data, n=n, axis=axis, norm=norm)

    def rfft(
        self,
        data: NDArray[np.float64],
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complex128]:
        """GPU-accelerated real FFT with automatic fallback.

        Computes the one-dimensional FFT of real-valued input, returning
        only the positive frequency components (memory efficient).

        Args:
            data: Input real-valued array.
            n: Length of the transformed axis. If None, uses data.shape[axis].
            axis: Axis over which to compute the FFT.
            norm: Normalization mode ("backward", "ortho", or "forward").

        Returns:
            Complex-valued FFT (positive frequencies only) on CPU.

        Example:
            >>> signal = np.random.randn(1000)
            >>> spectrum = gpu.rfft(signal)
            >>> # Result has length n//2 + 1

        References:
            SPE-001: Standard FFT Computation
            PERF-001: GPU-accelerated FFT
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            result = self._cp.fft.rfft(gpu_data, n=n, axis=axis, norm=norm)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.fft.rfft(data, n=n, axis=axis, norm=norm)

    def irfft(
        self,
        data: NDArray[np.complex128],
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.float64]:
        """GPU-accelerated inverse real FFT with automatic fallback.

        Computes the inverse FFT of rfft, returning real-valued output.

        Args:
            data: Input complex array (from rfft).
            n: Length of output. If None, uses (data.shape[axis] - 1) * 2.
            axis: Axis over which to compute the IFFT.
            norm: Normalization mode ("backward", "ortho", or "forward").

        Returns:
            Real-valued IFFT on CPU.

        Example:
            >>> spectrum = gpu.rfft(signal)
            >>> recovered = gpu.irfft(spectrum)

        References:
            SPE-001: Standard FFT Computation
            PERF-001: GPU-accelerated FFT
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            result = self._cp.fft.irfft(gpu_data, n=n, axis=axis, norm=norm)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.fft.irfft(data, n=n, axis=axis, norm=norm)

    def convolve(
        self,
        data: NDArray[np.float64],
        kernel: NDArray[np.float64],
        mode: Literal["full", "valid", "same"] = "full",
    ) -> NDArray[np.float64]:
        """GPU-accelerated convolution with automatic fallback.

        Computes the discrete linear convolution of data with kernel.
        Uses FFT-based convolution for efficiency on large arrays.

        Args:
            data: Input signal array.
            kernel: Convolution kernel (filter coefficients).
            mode: Convolution mode:
                - "full": Full convolution (length N + M - 1)
                - "valid": Only where data and kernel fully overlap
                - "same": Same length as data (centered)

        Returns:
            Convolved array on CPU.

        Example:
            >>> signal = np.random.randn(1000)
            >>> kernel = np.array([0.25, 0.5, 0.25])  # Simple smoothing
            >>> smoothed = gpu.convolve(signal, kernel, mode="same")

        References:
            PERF-002: GPU-accelerated convolution
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            gpu_kernel = self._to_gpu(kernel)
            result = self._cp.convolve(gpu_data, gpu_kernel, mode=mode)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.convolve(data, kernel, mode=mode)

    def correlate(
        self,
        a: NDArray[np.float64],
        v: NDArray[np.float64],
        mode: Literal["full", "valid", "same"] = "full",
    ) -> NDArray[np.float64]:
        """GPU-accelerated correlation with automatic fallback.

        Computes the cross-correlation of two 1-dimensional sequences.

        Args:
            a: First input sequence.
            v: Second input sequence.
            mode: Correlation mode ("full", "valid", or "same").

        Returns:
            Cross-correlation on CPU.

        Example:
            >>> signal = np.random.randn(1000)
            >>> template = signal[100:200]
            >>> corr = gpu.correlate(signal, template, mode="valid")
            >>> # Find best match location
            >>> match_idx = np.argmax(corr)

        References:
            PERF-003: GPU-accelerated pattern matching
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_a = self._to_gpu(a)
            gpu_v = self._to_gpu(v)
            result = self._cp.correlate(gpu_a, gpu_v, mode=mode)
            return self._to_cpu(result)
        else:
            # CPU fallback
            return np.correlate(a, v, mode=mode)

    def histogram(
        self,
        data: NDArray[np.float64],
        bins: int | NDArray[np.float64] = 10,
        range: tuple[float, float] | None = None,
        density: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """GPU-accelerated histogram with automatic fallback.

        Computes the histogram of a dataset, useful for statistical analysis
        and signal quality metrics.

        Args:
            data: Input data array.
            bins: Number of bins or array of bin edges.
            range: Lower and upper range of bins. If None, uses (data.min(), data.max()).
            density: If True, return probability density instead of counts.

        Returns:
            Tuple of (counts, bin_edges) both on CPU.

        Example:
            >>> signal = np.random.randn(10000)
            >>> counts, edges = gpu.histogram(signal, bins=100)
            >>> # Plot histogram
            >>> plt.bar(edges[:-1], counts, width=np.diff(edges))

        References:
            PERF-004: GPU-accelerated histogram computation
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_data = self._to_gpu(data)
            # Transfer bins to GPU if it's an array (not just int)
            gpu_bins = self._to_gpu(bins) if isinstance(bins, np.ndarray) else bins
            counts, edges = self._cp.histogram(
                gpu_data, bins=gpu_bins, range=range, density=density
            )
            return self._to_cpu(counts), self._to_cpu(edges)
        else:
            # CPU fallback
            return np.histogram(data, bins=bins, range=range, density=density)

    def dot(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64] | np.float64:
        """GPU-accelerated dot product with automatic fallback.

        Computes the dot product of two arrays, useful for correlation
        and pattern matching operations.

        Args:
            a: First array.
            b: Second array.

        Returns:
            Dot product on CPU (scalar or array depending on input dimensions).

        Example:
            >>> a = np.random.randn(1000)
            >>> b = np.random.randn(1000)
            >>> similarity = gpu.dot(a, b)

        References:
            PERF-003: GPU-accelerated matrix operations
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_a = self._to_gpu(a)
            gpu_b = self._to_gpu(b)
            result = self._cp.dot(gpu_a, gpu_b)
            return self._to_cpu(result)
        else:
            # CPU fallback - cast result to expected type
            from typing import cast

            return cast("NDArray[np.float64] | np.float64", np.dot(a, b))

    def matmul(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """GPU-accelerated matrix multiplication with automatic fallback.

        Computes the matrix product of two arrays.

        Args:
            a: First matrix.
            b: Second matrix.

        Returns:
            Matrix product on CPU.

        Example:
            >>> A = np.random.randn(100, 50)
            >>> B = np.random.randn(50, 100)
            >>> C = gpu.matmul(A, B)

        References:
            PERF-003: GPU-accelerated matrix operations
        """
        if self._check_gpu() and self._cp is not None:
            # GPU path
            gpu_a = self._to_gpu(a)
            gpu_b = self._to_gpu(b)
            result = self._cp.matmul(gpu_a, gpu_b)
            cpu_result = self._to_cpu(result)
            return cpu_result
        else:
            # CPU fallback - cast from Any to expected type
            result_cpu: NDArray[np.float64] = np.matmul(a, b)
            return result_cpu


# Module-level singleton for convenient access
gpu = GPUBackend()

__all__ = ["GPUBackend", "gpu"]
