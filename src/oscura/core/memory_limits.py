"""Per-operation memory limits for Oscura.

This module provides fine-grained memory control for individual operations
with automatic parameter adjustment to fit memory constraints.


Example:
    >>> from oscura.core.memory_limits import apply_memory_limit
    >>> params = apply_memory_limit('spectrogram', samples=1e9, max_memory='512MB')
    >>> print(f"Adjusted nperseg: {params['nperseg']}")

References:
    See oscura.config.memory for global memory configuration.
"""

import warnings
from typing import Any

from oscura.config.memory import get_memory_config
from oscura.utils.memory import estimate_memory


def parse_memory_limit(limit: int | str | None) -> int | None:
    """Parse memory limit from various formats.

    Args:
        limit: Memory limit as bytes (int), string ("4GB", "512MB"), or None.

    Returns:
        Memory limit in bytes, or None for no limit.

    Raises:
        ValueError: If format is invalid.

    Example:
        >>> parse_memory_limit("4GB")
        4000000000
        >>> parse_memory_limit(512 * 1024**2)
        536870912
        >>> parse_memory_limit(None) is None
        True
    """
    if limit is None:
        return None

    if isinstance(limit, str):
        limit_upper = limit.upper().strip()
        try:
            if limit_upper.endswith("GB"):
                return int(float(limit_upper[:-2]) * 1e9)
            elif limit_upper.endswith("MB"):
                return int(float(limit_upper[:-2]) * 1e6)
            elif limit_upper.endswith("KB"):
                return int(float(limit_upper[:-2]) * 1e3)
            elif limit_upper.endswith("GIB"):
                return int(float(limit_upper[:-3]) * 1024**3)
            elif limit_upper.endswith("MIB"):
                return int(float(limit_upper[:-3]) * 1024**2)
            elif limit_upper.endswith("KIB"):
                return int(float(limit_upper[:-3]) * 1024)
            else:
                return int(float(limit_upper))
        except ValueError as e:
            raise ValueError(f"Invalid memory limit format: {limit}") from e

    return int(limit)


def apply_memory_limit(
    operation: str,
    samples: int | float,
    *,
    max_memory: int | str | None = None,
    **params: Any,
) -> dict[str, Any]:
    """Apply memory limit and adjust parameters to fit.


    Args:
        operation: Operation name (fft, psd, spectrogram, etc.).
        samples: Number of samples to process.
        max_memory: Maximum memory limit (overrides global config if provided).
        **params: Operation parameters to adjust.

    Returns:
        Adjusted parameters dictionary that fits within memory limit.

    Example:
        >>> params = apply_memory_limit('spectrogram', samples=1e9, max_memory='512MB', nperseg=8192)
        >>> print(f"Adjusted to nperseg={params['nperseg']} to fit 512MB")

    Note:
        If parameters cannot be adjusted to fit memory, a warning is issued
        and the original parameters are returned.
    """
    # Parse memory limit
    limit_bytes = parse_memory_limit(max_memory)
    if limit_bytes is None:
        # Use global config
        config = get_memory_config()
        limit_bytes = config.max_memory
        if limit_bytes is None:
            # No limit, return params unchanged
            return params

    samples = int(samples)

    # Estimate with current parameters
    current_estimate = estimate_memory(operation, samples, **params)

    if current_estimate.total <= limit_bytes:
        # Already within limit
        return params

    # Need to adjust parameters
    adjusted_params = params.copy()

    if operation in ("fft", "psd"):
        # Reduce nfft if specified
        if "nfft" in adjusted_params:
            # Try reducing nfft
            original_nfft = adjusted_params["nfft"]
            # Binary search for suitable nfft
            nfft = _find_max_nfft(operation, samples, limit_bytes, **adjusted_params)
            if nfft < original_nfft:
                adjusted_params["nfft"] = nfft
                warnings.warn(
                    f"Reduced nfft from {original_nfft} to {nfft} to fit {limit_bytes / 1e6:.1f} MB limit",
                    UserWarning,
                    stacklevel=2,
                )

    elif operation == "spectrogram":
        # Adjust nperseg and/or nfft
        original_nperseg = adjusted_params.get("nperseg", 256)
        original_nfft = adjusted_params.get("nfft", original_nperseg)

        # Try reducing nperseg first
        nperseg = _find_max_nperseg(samples, limit_bytes, noverlap=adjusted_params.get("noverlap"))
        if nperseg < original_nperseg:
            adjusted_params["nperseg"] = nperseg
            # Adjust noverlap proportionally
            if "noverlap" in adjusted_params:
                overlap_ratio = adjusted_params["noverlap"] / original_nperseg
                adjusted_params["noverlap"] = int(nperseg * overlap_ratio)
            warnings.warn(
                f"Reduced nperseg from {original_nperseg} to {nperseg} to fit {limit_bytes / 1e6:.1f} MB limit",
                UserWarning,
                stacklevel=2,
            )

        # Also reduce nfft if needed
        if "nfft" in adjusted_params and adjusted_params["nfft"] > nperseg:
            adjusted_params["nfft"] = nperseg

    elif operation == "eye_diagram":
        # Reduce samples_per_ui or num_uis
        if "num_uis" in adjusted_params:
            original_num_uis = adjusted_params["num_uis"]
            # Calculate max num_uis that fits
            samples_per_ui = adjusted_params.get("samples_per_ui", 100)
            max_num_uis = _find_max_num_uis(limit_bytes, samples_per_ui)
            if max_num_uis < original_num_uis:
                adjusted_params["num_uis"] = max_num_uis
                warnings.warn(
                    f"Reduced num_uis from {original_num_uis} to {max_num_uis} to fit {limit_bytes / 1e6:.1f} MB limit",
                    UserWarning,
                    stacklevel=2,
                )

    # Verify final estimate
    final_estimate = estimate_memory(operation, samples, **adjusted_params)
    if final_estimate.total > limit_bytes:
        warnings.warn(
            f"Could not adjust parameters to fit {limit_bytes / 1e6:.1f} MB limit. "
            f"Operation requires {final_estimate.total / 1e6:.1f} MB. "
            "Consider using chunked processing or increasing memory limit.",
            UserWarning,
            stacklevel=2,
        )

    return adjusted_params


def _find_max_nfft(operation: str, samples: int, limit_bytes: int, **params: Any) -> int:
    """Binary search for maximum nfft that fits memory limit.

    Args:
        operation: Operation name.
        samples: Number of samples.
        limit_bytes: Memory limit in bytes.
        **params: Additional parameters.

    Returns:
        Maximum nfft that fits within limit.
    """
    min_nfft = 64
    max_nfft = params.get("nfft", 8192)

    # Binary search
    while min_nfft < max_nfft:
        mid_nfft = (min_nfft + max_nfft + 1) // 2
        test_params = {**params, "nfft": mid_nfft}
        estimate = estimate_memory(operation, samples, **test_params)

        if estimate.total <= limit_bytes:
            min_nfft = mid_nfft
        else:
            max_nfft = mid_nfft - 1

    return min_nfft


def _find_max_nperseg(samples: int, limit_bytes: int, noverlap: int | None = None) -> int:
    """Binary search for maximum nperseg that fits memory limit.

    Args:
        samples: Number of samples.
        limit_bytes: Memory limit in bytes.
        noverlap: Overlap samples (if specified).

    Returns:
        Maximum nperseg that fits within limit.
    """
    min_nperseg = 64
    max_nperseg = min(8192, samples // 4)

    # Binary search
    while min_nperseg < max_nperseg:
        mid_nperseg = (min_nperseg + max_nperseg + 1) // 2

        # Calculate memory for this nperseg
        hop = mid_nperseg - (noverlap or mid_nperseg // 2)
        num_segments = max(1, (samples - (noverlap or mid_nperseg // 2)) // hop)

        # Estimate memory
        bytes_per_sample = 8  # float64
        data_mem = samples * bytes_per_sample
        intermediate_mem = mid_nperseg * bytes_per_sample * 2 + mid_nperseg * bytes_per_sample * 2
        output_mem = (mid_nperseg // 2 + 1) * num_segments * bytes_per_sample * 2

        total_mem = data_mem + intermediate_mem + output_mem

        if total_mem <= limit_bytes:
            min_nperseg = mid_nperseg
        else:
            max_nperseg = mid_nperseg - 1

    return min_nperseg


def _find_max_num_uis(limit_bytes: int, samples_per_ui: int) -> int:
    """Find maximum num_uis that fits memory limit for eye diagrams.

    Args:
        limit_bytes: Memory limit in bytes.
        samples_per_ui: Samples per unit interval.

    Returns:
        Maximum num_uis that fits.
    """
    bytes_per_sample = 8  # float64
    # Eye diagram memory: samples_per_ui * num_uis * bytes_per_sample
    max_num_uis = limit_bytes // (samples_per_ui * bytes_per_sample * 2)
    return max(100, int(max_num_uis))  # At least 100 UIs


def get_operation_memory_limit(
    operation: str,
    max_memory: int | str | None = None,
) -> int:
    """Get effective memory limit for an operation.

    Args:
        operation: Operation name.
        max_memory: Override limit (or None for global config).

    Returns:
        Memory limit in bytes.

    Example:
        >>> limit = get_operation_memory_limit('spectrogram', max_memory='512MB')
        >>> print(f"Limit: {limit / 1e6:.1f} MB")
    """
    # Parse override
    limit_bytes = parse_memory_limit(max_memory)
    if limit_bytes is not None:
        return limit_bytes

    # Use global config
    config = get_memory_config()
    if config.max_memory is not None:
        return config.max_memory

    # Default: 80% of available
    from oscura.utils.memory import get_available_memory

    return int(get_available_memory() * 0.8)


def check_operation_fits(
    operation: str,
    samples: int | float,
    *,
    max_memory: int | str | None = None,
    **params: Any,
) -> bool:
    """Check if operation with given parameters fits within memory limit.

    Args:
        operation: Operation name.
        samples: Number of samples.
        max_memory: Memory limit (or None for global config).
        **params: Operation parameters.

    Returns:
        True if operation fits within limit.

    Example:
        >>> fits = check_operation_fits('fft', samples=1e9, max_memory='4GB', nfft=8192)
        >>> if not fits:
        ...     print("FFT too large for 4GB limit")
    """
    limit_bytes = get_operation_memory_limit(operation, max_memory)
    estimate = estimate_memory(operation, samples, **params)
    return estimate.total <= limit_bytes


__all__ = [
    "apply_memory_limit",
    "check_operation_fits",
    "get_operation_memory_limit",
    "parse_memory_limit",
]
