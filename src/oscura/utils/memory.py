"""Memory management utilities for Oscura.

This module provides memory estimation, availability checking, and
OOM prevention for large signal processing operations.


Example:
    >>> from oscura.utils.memory import estimate_memory, check_memory_available
    >>> estimate = estimate_memory('fft', samples=1e9)
    >>> check = check_memory_available('spectrogram', samples=1e9, nperseg=4096)

References:
    Python psutil documentation
"""

from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MemoryEstimate:
    """Memory requirement estimate for an operation.

    Attributes:
        data: Memory for input data (bytes).
        intermediate: Memory for intermediate buffers (bytes).
        output: Memory for output data (bytes).
        total: Total memory required (bytes).
        operation: Operation name.
        parameters: Parameters used for estimate.
    """

    data: int
    intermediate: int
    output: int
    total: int
    operation: str
    parameters: dict  # type: ignore[type-arg]

    def __repr__(self) -> str:
        return (
            f"MemoryEstimate({self.operation}: "
            f"total={self.total / 1e9:.2f} GB, "
            f"data={self.data / 1e9:.2f} GB, "
            f"intermediate={self.intermediate / 1e9:.2f} GB, "
            f"output={self.output / 1e9:.2f} GB)"
        )


@dataclass
class MemoryCheck:
    """Result of memory availability check.

    Attributes:
        sufficient: True if enough memory is available.
        available: Available memory (bytes).
        required: Required memory (bytes).
        recommendation: Suggested action if insufficient.
    """

    sufficient: bool
    available: int
    required: int
    recommendation: str


class MemoryCheckError(Exception):
    """Exception raised when memory check fails.

    Attributes:
        required: Required memory in bytes.
        available: Available memory in bytes.
        recommendation: Suggested action.
    """

    def __init__(self, message: str, required: int, available: int, recommendation: str):
        super().__init__(message)
        self.required = required
        self.available = available
        self.recommendation = recommendation


def detect_wsl() -> bool:
    """Detect if running in Windows Subsystem for Linux.

    Returns:
        True if running in WSL.
    """
    try:
        with open("/proc/version") as f:
            version = f.read().lower()
            return "microsoft" in version or "wsl" in version
    except (FileNotFoundError, PermissionError):
        return False


def get_total_memory() -> int:
    """Get total system memory in bytes.

    Returns:
        Total physical memory in bytes.
    """
    try:
        import psutil

        return psutil.virtual_memory().total  # type: ignore[no-any-return]
    except ImportError:
        # Fallback without psutil
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Format: "MemTotal:       16384 kB"
                            return int(line.split()[1]) * 1024
            except (FileNotFoundError, PermissionError):
                pass
        # Default fallback: assume 8 GB
        return 8 * 1024 * 1024 * 1024


def get_available_memory() -> int:
    """Get available memory in bytes.

    Accounts for OS overhead and applies WSL conservative factor.

    Returns:
        Available memory in bytes.
    """
    # Get memory reserve from environment
    reserve_str = os.environ.get("TK_MEMORY_RESERVE", "0")
    try:
        if reserve_str.upper().endswith("GB"):
            reserve = int(float(reserve_str[:-2]) * 1e9)
        elif reserve_str.upper().endswith("MB"):
            reserve = int(float(reserve_str[:-2]) * 1e6)
        else:
            reserve = int(float(reserve_str))
    except ValueError:
        reserve = 0

    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        # Fallback without psutil
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            available = int(line.split()[1]) * 1024
                            break
                    else:
                        available = get_total_memory() // 2
            except (FileNotFoundError, PermissionError):
                available = get_total_memory() // 2
        else:
            available = get_total_memory() // 2

    # Apply WSL conservative factor
    if detect_wsl():
        available = int(available * 0.5)

    # Apply reserve
    available = max(0, available - reserve)

    return available  # type: ignore[no-any-return]


def get_swap_available() -> int:
    """Get available swap space in bytes.

    Returns:
        Available swap in bytes.
    """
    try:
        import psutil

        return psutil.swap_memory().free  # type: ignore[no-any-return]
    except ImportError:
        # Fallback
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("SwapFree:"):
                            return int(line.split()[1]) * 1024
            except (FileNotFoundError, PermissionError):
                pass
        return 0


def get_memory_pressure() -> float:
    """Get current memory utilization (0.0 to 1.0).

    Returns:
        Memory pressure as fraction of total memory used.
    """
    try:
        import psutil

        return psutil.virtual_memory().percent / 100.0  # type: ignore[no-any-return]
    except ImportError:
        total = get_total_memory()
        available = get_available_memory()
        return 1.0 - (available / total) if total > 0 else 0.5


def estimate_memory(
    operation: str,
    samples: int | float | None = None,
    *,
    nfft: int | None = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    dtype: str = "float64",
    channels: int = 1,
    **kwargs: Any,
) -> MemoryEstimate:
    """Estimate memory requirements for an operation.

    Args:
        operation: Operation name (fft, psd, spectrogram, eye_diagram, correlate, filter).
        samples: Number of samples (can be float for large values).
        nfft: FFT length (for fft, psd, spectrogram).
        nperseg: Segment length (for spectrogram, psd).
        noverlap: Overlap samples (for spectrogram).
        dtype: Data type (float32 or float64).
        channels: Number of channels.
        **kwargs: Additional operation-specific parameters.

    Returns:
        MemoryEstimate with memory requirements.

    Example:
        >>> estimate = estimate_memory('fft', samples=1e9, nfft=8192)
        >>> print(f"Required: {estimate.total / 1e9:.2f} GB")
    """
    bytes_per_sample = 4 if dtype == "float32" else 8
    samples = int(samples or 0)

    # Dispatch to operation-specific estimator (returns computed params)
    data_mem, intermediate_mem, output_mem, computed_params = _estimate_for_operation(
        operation, samples, bytes_per_sample, channels, nfft, nperseg, noverlap, kwargs
    )

    total_mem = data_mem + intermediate_mem + output_mem

    # Merge provided and computed parameters
    all_params = {
        "samples": samples,
        "dtype": dtype,
        "channels": channels,
        **computed_params,  # Include computed defaults
        **kwargs,
    }

    return MemoryEstimate(
        data=data_mem,
        intermediate=intermediate_mem,
        output=output_mem,
        total=total_mem,
        operation=operation,
        parameters=all_params,
    )


def _estimate_for_operation(
    operation: str,
    samples: int,
    bytes_per_sample: int,
    channels: int,
    nfft: int | None,
    nperseg: int | None,
    noverlap: int | None,
    kwargs: dict[str, Any],
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for specific operation.

    Returns:
        Tuple of (data_mem, intermediate_mem, output_mem, computed_params)
    """
    if operation == "fft":
        return _estimate_fft(samples, bytes_per_sample, channels, nfft)
    elif operation == "psd":
        return _estimate_psd(samples, bytes_per_sample, channels, nfft, nperseg)
    elif operation == "spectrogram":
        return _estimate_spectrogram(samples, bytes_per_sample, channels, nfft, nperseg, noverlap)
    elif operation == "eye_diagram":
        return _estimate_eye_diagram(samples, bytes_per_sample, channels, kwargs)
    elif operation == "correlate":
        return _estimate_correlate(samples, bytes_per_sample, channels)
    elif operation == "filter":
        return _estimate_filter(samples, bytes_per_sample, channels, kwargs)
    else:
        # Generic estimate
        data_mem = samples * bytes_per_sample * channels
        return data_mem, data_mem, data_mem, {}


def _estimate_fft(
    samples: int, bytes_per_sample: int, channels: int, nfft: int | None
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for FFT operation."""
    nfft = nfft or _next_power_of_2(samples)
    data_mem = samples * bytes_per_sample * channels
    intermediate_mem = nfft * bytes_per_sample * 2 * 2  # complex, work buffer
    output_mem = (nfft // 2 + 1) * bytes_per_sample * 2 * channels
    return data_mem, intermediate_mem, output_mem, {"nfft": nfft}


def _estimate_psd(
    samples: int, bytes_per_sample: int, channels: int, nfft: int | None, nperseg: int | None
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for PSD (Welch) operation."""
    nperseg = nperseg or 256
    nfft = nfft or nperseg
    data_mem = samples * bytes_per_sample * channels
    intermediate_mem = nperseg * bytes_per_sample * 2 + nfft * bytes_per_sample * 2
    output_mem = (nfft // 2 + 1) * bytes_per_sample * channels
    return data_mem, intermediate_mem, output_mem, {"nfft": nfft, "nperseg": nperseg}


def _estimate_spectrogram(
    samples: int,
    bytes_per_sample: int,
    channels: int,
    nfft: int | None,
    nperseg: int | None,
    noverlap: int | None,
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for spectrogram (STFT) operation."""
    nperseg = nperseg or 256
    noverlap = noverlap or nperseg // 2
    nfft = nfft or nperseg
    hop = nperseg - noverlap
    num_segments = max(1, (samples - noverlap) // hop)

    data_mem = samples * bytes_per_sample * channels
    intermediate_mem = nperseg * bytes_per_sample * 2 + nfft * bytes_per_sample * 2
    output_mem = (nfft // 2 + 1) * num_segments * bytes_per_sample * 2 * channels
    return (
        data_mem,
        intermediate_mem,
        output_mem,
        {"nfft": nfft, "nperseg": nperseg, "noverlap": noverlap},
    )


def _estimate_eye_diagram(
    samples: int, bytes_per_sample: int, channels: int, kwargs: dict[str, Any]
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for eye diagram generation."""
    samples_per_ui = kwargs.get("samples_per_ui", 100)
    num_uis = kwargs.get("num_uis", 1000)
    data_mem = samples * bytes_per_sample * channels
    intermediate_mem = samples_per_ui * num_uis * bytes_per_sample
    output_mem = samples_per_ui * num_uis * bytes_per_sample
    return data_mem, intermediate_mem, output_mem, {}


def _estimate_correlate(
    samples: int, bytes_per_sample: int, channels: int
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for correlation operation."""
    data_mem = samples * bytes_per_sample * 2 * channels  # Two signals
    nfft = _next_power_of_2(samples * 2)
    intermediate_mem = nfft * bytes_per_sample * 2 * 2  # Two FFTs
    output_mem = (samples * 2 - 1) * bytes_per_sample * channels
    return data_mem, intermediate_mem, output_mem, {"nfft": nfft}


def _estimate_filter(
    samples: int, bytes_per_sample: int, channels: int, kwargs: dict[str, Any]
) -> tuple[int, int, int, dict[str, Any]]:
    """Estimate memory for filter operation."""
    filter_order = kwargs.get("filter_order", 8)
    data_mem = samples * bytes_per_sample * channels
    intermediate_mem = (filter_order + samples) * bytes_per_sample
    output_mem = samples * bytes_per_sample * channels
    return data_mem, intermediate_mem, output_mem, {}


def check_memory_available(
    operation: str,
    samples: int | float | None = None,
    **kwargs: Any,
) -> MemoryCheck:
    """Check if sufficient memory is available for an operation.

    Args:
        operation: Operation name.
        samples: Number of samples.
        **kwargs: Additional parameters for estimate_memory.

    Returns:
        MemoryCheck with sufficiency status and recommendation.

    Example:
        >>> check = check_memory_available('spectrogram', samples=1e9, nperseg=4096)
        >>> if not check.sufficient:
        ...     print(check.recommendation)
    """
    estimate = estimate_memory(operation, samples, **kwargs)
    available = get_available_memory()

    sufficient = estimate.total <= available

    if sufficient:
        recommendation = "Memory sufficient for operation."
    else:
        # Generate recommendations
        ratio = estimate.total / available
        if ratio < 2:
            recommendation = (
                f"Need {estimate.total / 1e9:.1f} GB, have {available / 1e9:.1f} GB. "
                "Consider closing other applications or using chunked processing."
            )
        elif ratio < 10:
            recommendation = (
                f"Need {estimate.total / 1e9:.1f} GB, have {available / 1e9:.1f} GB. "
                f"Use chunked processing or downsample by {int(ratio)}x."
            )
        else:
            recommendation = (
                f"Need {estimate.total / 1e9:.1f} GB, have {available / 1e9:.1f} GB. "
                "Data too large for available memory. Use streaming/chunked processing "
                "or process a subset of the data."
            )

    return MemoryCheck(
        sufficient=sufficient,
        available=available,
        required=estimate.total,
        recommendation=recommendation,
    )


def require_memory(
    operation: str,
    samples: int | float | None = None,
    **kwargs: Any,
) -> None:
    """Raise exception if insufficient memory for operation.

    Args:
        operation: Operation name.
        samples: Number of samples.
        **kwargs: Additional parameters.

    Raises:
        MemoryCheckError: If insufficient memory.
    """
    check = check_memory_available(operation, samples, **kwargs)
    if not check.sufficient:
        raise MemoryCheckError(
            f"Insufficient memory for {operation}",
            required=check.required,
            available=check.available,
            recommendation=check.recommendation,
        )


def _next_power_of_2(n: int) -> int:
    """Return next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# Memory configuration
_max_memory: int | None = None


def set_max_memory(limit: int | str | None) -> None:
    """Set global memory limit for Oscura operations.

    Args:
        limit: Maximum memory in bytes, or string like "4GB", "512MB".

    Example:
        >>> set_max_memory("4GB")
        >>> set_max_memory(4 * 1024 * 1024 * 1024)
    """
    global _max_memory

    if limit is None:
        _max_memory = None
        return

    if isinstance(limit, str):
        limit = limit.upper().strip()
        if limit.endswith("GB"):
            _max_memory = int(float(limit[:-2]) * 1e9)
        elif limit.endswith("MB"):
            _max_memory = int(float(limit[:-2]) * 1e6)
        elif limit.endswith("KB"):
            _max_memory = int(float(limit[:-2]) * 1e3)
        else:
            _max_memory = int(float(limit))
    else:
        _max_memory = int(limit)


def get_max_memory() -> int:
    """Get the current memory limit.

    Returns:
        Memory limit in bytes (default: 80% of available).
    """
    if _max_memory is not None:
        return _max_memory

    # Check environment variable
    env_limit = os.environ.get("TK_MAX_MEMORY")
    if env_limit:
        set_max_memory(env_limit)
        if _max_memory is not None:
            return _max_memory  # type: ignore[unreachable]

    # Default: 80% of available
    return int(get_available_memory() * 0.8)


def gc_collect() -> int:
    """Force garbage collection.

    Returns:
        Number of unreachable objects collected.
    """
    return gc.collect()


def get_memory_info() -> dict[str, int]:
    """Get comprehensive memory information.

    Returns:
        Dictionary with memory statistics.
    """
    return {
        "total": get_total_memory(),
        "available": get_available_memory(),
        "swap_available": get_swap_available(),
        "max_memory": get_max_memory(),
        "pressure_pct": int(get_memory_pressure() * 100),
        "wsl": detect_wsl(),
    }


# ==========================================================================
# MEM-009, MEM-010, MEM-011: Memory Configuration & Limits
# ==========================================================================


@dataclass
class MemoryConfig:
    """Global memory configuration for Oscura operations.


    Attributes:
        max_memory: Global memory limit in bytes (None = 80% of available).
        warn_threshold: Warning threshold (0.0-1.0, default 0.7).
        critical_threshold: Critical threshold (0.0-1.0, default 0.9).
        auto_degrade: Automatically downsample if memory exceeded.
    """

    max_memory: int | None = None
    warn_threshold: float = 0.7
    critical_threshold: float = 0.9
    auto_degrade: bool = False

    def __post_init__(self) -> None:
        """Validate thresholds."""
        if not 0.0 <= self.warn_threshold <= 1.0:
            raise ValueError(f"warn_threshold must be 0.0-1.0, got {self.warn_threshold}")
        if not 0.0 <= self.critical_threshold <= 1.0:
            raise ValueError(f"critical_threshold must be 0.0-1.0, got {self.critical_threshold}")
        if self.warn_threshold >= self.critical_threshold:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) must be < critical_threshold "
                f"({self.critical_threshold})"
            )


# Global memory configuration instance
_memory_config = MemoryConfig()


def configure_memory(
    *,
    max_memory: int | str | None = None,
    warn_threshold: float | None = None,
    critical_threshold: float | None = None,
    auto_degrade: bool | None = None,
) -> None:
    """Configure global memory limits and thresholds.


    Args:
        max_memory: Maximum memory in bytes or string ("4GB", "512MB").
        warn_threshold: Warning threshold (0.0-1.0).
        critical_threshold: Critical threshold (0.0-1.0).
        auto_degrade: Enable automatic downsampling.

    Example:
        >>> configure_memory(max_memory="4GB", warn_threshold=0.7, critical_threshold=0.9)
        >>> configure_memory(auto_degrade=True)
    """
    global _memory_config

    if max_memory is not None:
        if isinstance(max_memory, str):
            # Parse string format
            limit_upper = max_memory.upper().strip()
            if limit_upper.endswith("GB"):
                _memory_config.max_memory = int(float(limit_upper[:-2]) * 1e9)
            elif limit_upper.endswith("MB"):
                _memory_config.max_memory = int(float(limit_upper[:-2]) * 1e6)
            elif limit_upper.endswith("KB"):
                _memory_config.max_memory = int(float(limit_upper[:-2]) * 1e3)
            else:
                _memory_config.max_memory = int(float(limit_upper))
        else:
            _memory_config.max_memory = int(max_memory)

    if warn_threshold is not None:
        _memory_config.warn_threshold = warn_threshold
    if critical_threshold is not None:
        _memory_config.critical_threshold = critical_threshold
    if auto_degrade is not None:
        _memory_config.auto_degrade = auto_degrade

    # Validate after updates
    _memory_config.__post_init__()


def get_memory_config() -> MemoryConfig:
    """Get current memory configuration.

    Returns:
        Current MemoryConfig instance.
    """
    return _memory_config


# ==========================================================================
# ==========================================================================


@dataclass
class DownsamplingRecommendation:
    """Recommendation for downsampling to fit memory constraints.

    Attributes:
        factor: Suggested downsampling factor (2, 4, 8, 16, etc.).
        required_memory: Memory required without downsampling (bytes).
        available_memory: Available memory (bytes).
        new_sample_rate: Effective sample rate after downsampling (Hz).
        message: Human-readable recommendation message.
    """

    factor: int
    required_memory: int
    available_memory: int
    new_sample_rate: float
    message: str


def suggest_downsampling(
    operation: str,
    samples: int | float,
    sample_rate: float,
    **kwargs: Any,
) -> DownsamplingRecommendation | None:
    """Suggest downsampling factor if operation would exceed memory limits.


    Args:
        operation: Operation name.
        samples: Number of samples.
        sample_rate: Current sample rate in Hz.
        **kwargs: Additional parameters for memory estimation.

    Returns:
        DownsamplingRecommendation if downsampling needed, None if sufficient memory.

    Example:
        >>> rec = suggest_downsampling('spectrogram', samples=1e9, sample_rate=1e9, nperseg=4096)
        >>> if rec:
        ...     print(f"Downsample by {rec.factor}x to {rec.new_sample_rate/1e6:.1f} MSa/s")
    """
    estimate = estimate_memory(operation, samples, **kwargs)
    available = get_available_memory()

    if estimate.total <= available:
        return None  # Sufficient memory

    # Calculate required downsampling factor
    ratio = estimate.total / available
    # Round up to next power of 2
    factor = 2 ** int(np.ceil(np.log2(ratio)))
    # Limit to reasonable factors
    factor = min(factor, 16)

    new_sample_rate = sample_rate / factor
    new_samples = int(samples) // factor

    # Re-estimate with downsampled size
    new_estimate = estimate_memory(operation, new_samples, **kwargs)

    message = (
        f"Insufficient memory for {operation}. "
        f"Need {estimate.total / 1e9:.1f} GB, have {available / 1e9:.1f} GB. "
        f"Recommend downsampling by {factor}x (new rate: {new_sample_rate / 1e6:.1f} MSa/s). "
        f"Estimated memory after downsampling: {new_estimate.total / 1e9:.2f} GB."
    )

    return DownsamplingRecommendation(
        factor=factor,
        required_memory=estimate.total,
        available_memory=available,
        new_sample_rate=new_sample_rate,
        message=message,
    )


# ==========================================================================
# ==========================================================================


class MemoryMonitor:
    """Context manager for monitoring memory usage and preventing OOM crashes.


    Attributes:
        operation: Name of the operation being monitored.
        max_memory: Maximum allowed memory (None = use global config).
        check_interval: How often to check memory (number of iterations).

    Example:
        >>> with MemoryMonitor('spectrogram', max_memory=4e9) as monitor:
        ...     for i in range(1000):
        ...         # Perform work
        ...         monitor.check(i)  # Check memory periodically
    """

    def __init__(
        self,
        operation: str,
        *,
        max_memory: int | str | None = None,
        check_interval: int = 100,
    ):
        self.operation = operation
        self.check_interval = check_interval
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
        self._iteration = 0

        # Parse max_memory
        if max_memory is None:
            self.max_memory = get_max_memory()
        elif isinstance(max_memory, str):
            limit_upper = max_memory.upper().strip()
            if limit_upper.endswith("GB"):
                self.max_memory = int(float(limit_upper[:-2]) * 1e9)
            elif limit_upper.endswith("MB"):
                self.max_memory = int(float(limit_upper[:-2]) * 1e6)
            else:
                self.max_memory = int(float(limit_upper))
        else:
            self.max_memory = int(max_memory)

    def __enter__(self) -> MemoryMonitor:
        """Enter context and record starting memory."""
        self.start_memory = self._get_process_memory()
        self.peak_memory = self.start_memory
        self.current_memory = self.start_memory
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility

    def check(self, iteration: int | None = None) -> None:
        """Check memory usage and raise error if limit approached.

        Args:
            iteration: Current iteration number (for periodic checking).

        Raises:
            MemoryError: If memory usage exceeds 95% of max_memory.
        """
        self._iteration += 1

        # Only check periodically
        if iteration is not None and iteration % self.check_interval != 0:
            return

        self.current_memory = self._get_process_memory()
        self.peak_memory = max(self.peak_memory, self.current_memory)

        # Check against available memory
        available = get_available_memory()
        critical_threshold = _memory_config.critical_threshold

        if available < self.max_memory * (1 - critical_threshold):
            raise MemoryError(
                f"Memory limit approached during {self.operation}. "
                f"Available: {available / 1e9:.2f} GB, "
                f"Limit: {self.max_memory / 1e9:.2f} GB. "
                f"Operation aborted to prevent system crash."
            )

    def _get_process_memory(self) -> int:
        """Get current process memory usage in bytes."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss  # type: ignore[no-any-return]
        except ImportError:
            # Fallback: use system available memory
            return get_total_memory() - get_available_memory()

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics for this monitoring session.

        Returns:
            Dictionary with start, current, and peak memory usage.

        Example:
            >>> with MemoryMonitor('fft') as monitor:
            ...     # ... do work ...
            ...     stats = monitor.get_stats()
            >>> print(f"Peak memory: {stats['peak'] / 1e6:.1f} MB")
        """
        return {
            "start": self.start_memory,
            "current": self.current_memory,
            "peak": self.peak_memory,
            "delta": self.peak_memory - self.start_memory,
        }


# ==========================================================================
# ==========================================================================


@dataclass
class ProgressInfo:
    """Progress information with memory metrics.


    Attributes:
        current: Current progress value.
        total: Total progress value.
        eta_seconds: Estimated time to completion in seconds.
        memory_used: Current memory usage in bytes.
        memory_peak: Peak memory usage since start in bytes.
        operation: Name of the operation.
    """

    current: int
    total: int
    eta_seconds: float
    memory_used: int
    memory_peak: int
    operation: str

    @property
    def percent(self) -> float:
        """Progress percentage (0.0-100.0)."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100.0

    def format_progress(self) -> str:
        """Format progress as human-readable string.

        Returns:
            Formatted string like "42.5% | 1.2 GB used | 2.1 GB peak | ETA 5s"
        """
        return (
            f"{self.percent:.1f}% | "
            f"{self.memory_used / 1e9:.2f} GB used | "
            f"{self.memory_peak / 1e9:.2f} GB peak | "
            f"ETA {self.eta_seconds:.0f}s"
        )
