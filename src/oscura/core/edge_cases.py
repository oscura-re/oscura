"""Edge case handling utilities for Oscura.

This module provides utilities for gracefully handling edge cases including
empty inputs, single-sample traces, and NaN/Inf values.


Example:
    >>> from oscura.core.edge_cases import (
    ...     validate_signal,
    ...     handle_empty_trace,
    ...     sanitize_signal
    ... )
    >>> validated = validate_signal(signal, min_samples=10)
    >>> clean_signal = sanitize_signal(noisy_signal)

References:
    - IEEE 754 floating-point standard
    - NumPy NaN handling best practices
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmptyTraceError(Exception):
    """Exception raised when trace has no data.

    : Empty trace returns informative error, not crash.

    Example:
        >>> from oscura.core.edge_cases import EmptyTraceError
        >>> raise EmptyTraceError("Cannot analyze empty trace (0 samples)")

    References:
        EDGE-002: Graceful Empty/Short Signal Handling
    """

    def __init__(self, message: str = "Trace is empty (0 samples)") -> None:
        """Initialize EmptyTraceError.

        Args:
            message: Error message (default: "Trace is empty (0 samples)")
        """
        super().__init__(message)


class InsufficientSamplesError(Exception):
    """Exception raised when trace has insufficient samples.

    : Too-short trace warns and adapts.

    Attributes:
        required: Minimum samples required
        available: Actual samples available

    Example:
        >>> from oscura.core.edge_cases import InsufficientSamplesError
        >>> raise InsufficientSamplesError("Need at least 100 samples", 100, 10)

    References:
        EDGE-002: Graceful Empty/Short Signal Handling
    """

    def __init__(self, message: str, required: int, available: int) -> None:
        """Initialize InsufficientSamplesError.

        Args:
            message: Error message
            required: Minimum samples required
            available: Actual samples available
        """
        self.required = required
        self.available = available
        full_message = f"{message} (required: {required}, available: {available})"
        super().__init__(full_message)


def validate_signal(
    signal: NDArray[np.float64],
    *,
    min_samples: int = 1,
    allow_empty: bool = False,
    name: str = "signal",
) -> NDArray[np.float64]:
    """Validate signal array for basic requirements.

    : Empty trace returns informative error, not crash.
    Checks for empty arrays and minimum sample requirements.

    Args:
        signal: Input signal array
        min_samples: Minimum required samples (default: 1)
        allow_empty: Allow empty arrays (default: False)
        name: Signal name for error messages (default: "signal")

    Returns:
        Validated signal array

    Raises:
        EmptyTraceError: If signal is empty and allow_empty=False
        InsufficientSamplesError: If signal has fewer than min_samples
        ValueError: If signal is not 1D or has invalid shape

    Example:
        >>> import numpy as np
        >>> from oscura.core.edge_cases import validate_signal
        >>> signal = np.array([1.0, 2.0, 3.0])
        >>> validated = validate_signal(signal, min_samples=2)
        >>> # Empty signal raises error
        >>> validate_signal(np.array([]))  # Raises EmptyTraceError

    References:
        EDGE-002: Graceful Empty/Short Signal Handling
    """
    # Check if array
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(signal)}")

    # Check dimensions
    if signal.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {signal.ndim}D")

    # Check for empty
    n_samples = len(signal)
    if n_samples == 0:
        if allow_empty:
            return signal
        else:
            raise EmptyTraceError(f"{name} is empty (0 samples)")

    # Check minimum samples
    if n_samples < min_samples:
        raise InsufficientSamplesError(
            f"{name} has too few samples",
            required=min_samples,
            available=n_samples,
        )

    return signal


def handle_empty_trace(default_value: float = np.nan) -> NDArray[np.float64]:
    """Return a safe default for empty trace operations.

    : Empty trace returns informative error, not crash.
    Provides graceful fallback for operations on empty traces.

    Args:
        default_value: Default value to return (default: NaN)

    Returns:
        Single-element array with default value

    Example:
        >>> from oscura.core.edge_cases import handle_empty_trace
        >>> result = handle_empty_trace(0.0)
        >>> print(result)
        [0.]

    References:
        EDGE-002: Graceful Empty/Short Signal Handling
    """
    return np.array([default_value])


def check_single_sample(
    signal: NDArray[np.float64],
    operation: str = "operation",
) -> bool:
    """Check if signal has only one sample and warn.

    : Handle traces with 1 sample.
    Warns user that statistical operations may not be meaningful.

    Args:
        signal: Input signal
        operation: Operation name for warning message

    Returns:
        True if signal has only 1 sample

    Example:
        >>> import numpy as np
        >>> from oscura.core.edge_cases import check_single_sample
        >>> signal = np.array([42.0])
        >>> if check_single_sample(signal, "FFT"):
        ...     print("Cannot compute FFT on single sample")

    References:
        EDGE-002: Graceful Empty/Short Signal Handling
    """
    if len(signal) == 1:
        warnings.warn(
            f"Signal has only 1 sample. {operation} may not produce meaningful results.",
            UserWarning,
            stacklevel=2,
        )
        return True
    return False


def sanitize_signal(
    signal: NDArray[np.float64],
    *,
    replace_nan: float | str = "interpolate",
    replace_inf: float | str = "clip",
    warn: bool = True,
) -> NDArray[np.float64]:
    """Remove or replace NaN and Inf values in signal.

    : Handle NaN and Inf values gracefully.
    Cleans signal data for robust analysis.

    Args:
        signal: Input signal array
        replace_nan: How to handle NaN:
            - "interpolate": Linear interpolation (default)
            - "zero": Replace with 0
            - "remove": Remove samples (changes length)
            - float: Replace with specific value
        replace_inf: How to handle Inf:
            - "clip": Clip to min/max of finite values (default)
            - "zero": Replace with 0
            - "remove": Remove samples (changes length)
            - float: Replace with specific value
        warn: Issue warning if NaN/Inf found (default: True)

    Returns:
        Sanitized signal array

    Raises:
        ValueError: If replace_nan or replace_inf option is invalid.

    Example:
        >>> import numpy as np
        >>> from oscura.core.edge_cases import sanitize_signal
        >>> signal = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        >>> clean = sanitize_signal(signal)
        >>> print(clean)
        [1. 2. 3. 5. 5.]

    References:
        EDGE-003: NaN/Inf Handling
    """
    signal = signal.copy()  # Don't modify input
    n_nan = np.sum(np.isnan(signal))
    n_inf = np.sum(np.isinf(signal))

    # Warn if issues found
    if warn and (n_nan > 0 or n_inf > 0):
        warnings.warn(
            f"Signal contains {n_nan} NaN and {n_inf} Inf values. Applying sanitization.",
            UserWarning,
            stacklevel=2,
        )

    # Handle NaN
    if n_nan > 0:
        if replace_nan == "interpolate":
            signal = _interpolate_nan(signal)
        elif replace_nan == "zero":
            signal[np.isnan(signal)] = 0.0
        elif replace_nan == "remove":
            signal = signal[~np.isnan(signal)]
        elif isinstance(replace_nan, int | float):
            signal[np.isnan(signal)] = float(replace_nan)
        else:
            raise ValueError(f"Invalid replace_nan option: {replace_nan}")

    # Handle Inf
    if n_inf > 0:
        if replace_inf == "clip":
            finite_mask = np.isfinite(signal)
            if np.any(finite_mask):
                min_val = np.min(signal[finite_mask])
                max_val = np.max(signal[finite_mask])
                signal[signal == np.inf] = max_val
                signal[signal == -np.inf] = min_val
            else:
                signal[np.isinf(signal)] = 0.0
        elif replace_inf == "zero":
            signal[np.isinf(signal)] = 0.0
        elif replace_inf == "remove":
            signal = signal[~np.isinf(signal)]
        elif isinstance(replace_inf, int | float):
            signal[np.isinf(signal)] = float(replace_inf)
        else:
            raise ValueError(f"Invalid replace_inf option: {replace_inf}")

    return signal


def _interpolate_nan(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate NaN values using linear interpolation.

    Args:
        signal: Signal with NaN values

    Returns:
        Signal with NaN values interpolated

    References:
        EDGE-003: NaN/Inf Handling
    """
    # Find NaN locations
    nan_mask = np.isnan(signal)

    if not np.any(nan_mask):
        return signal

    # Get valid indices and values
    valid_mask = ~nan_mask
    if not np.any(valid_mask):
        # All NaN - replace with zeros
        return np.zeros_like(signal)

    valid_indices = np.where(valid_mask)[0]
    valid_values = signal[valid_mask]

    # Interpolate
    nan_indices = np.where(nan_mask)[0]
    interpolated = np.interp(nan_indices, valid_indices, valid_values)

    # Replace NaN with interpolated values
    result = signal.copy()
    result[nan_mask] = interpolated

    return result


def check_signal_quality(
    signal: NDArray[np.float64],
    *,
    clipping_threshold: float = 0.95,
    noise_floor_db: float = -60.0,
    dc_offset_max: float = 0.1,
) -> SignalQualityReport:
    """Check signal quality and detect common issues.

    : Detect clipping, noise floor, and DC offset problems.
    Analyzes signal for quality issues that may affect results.

    Args:
        signal: Input signal array
        clipping_threshold: Fraction of range for clipping detection (default: 0.95)
        noise_floor_db: Expected noise floor in dB (default: -60)
        dc_offset_max: Maximum acceptable DC offset (default: 0.1)

    Returns:
        SignalQualityReport with detected issues

    Example:
        >>> import numpy as np
        >>> from oscura.core.edge_cases import check_signal_quality
        >>> signal = np.random.randn(1000) + 0.5  # Signal with DC offset
        >>> quality = check_signal_quality(signal, dc_offset_max=0.1)
        >>> if quality.dc_offset_excessive:
        ...     print(f"DC offset: {quality.dc_offset:.3f}")

    References:
        EDGE-001: Signal Quality Warnings
    """
    # Calculate statistics
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))

    # Check for clipping
    signal_range = max_val - min_val
    clipping_detected = False
    clipping_percent = 0.0

    if signal_range > 0:
        # Count samples near limits
        upper_thresh = min_val + signal_range * clipping_threshold
        lower_thresh = min_val + signal_range * (1 - clipping_threshold)

        n_clipped = np.sum((signal >= upper_thresh) | (signal <= lower_thresh))
        clipping_percent = float(100.0 * n_clipped / len(signal))
        clipping_detected = clipping_percent > 1.0  # >1% clipping

    # Check noise floor (estimate SNR)
    if std_val > 0:
        snr_db = 20 * np.log10(abs(mean_val) / std_val) if abs(mean_val) > 0 else -np.inf
    else:
        snr_db = np.inf

    high_noise = snr_db < noise_floor_db

    # Check DC offset
    dc_offset = abs(mean_val)
    dc_offset_excessive = dc_offset > dc_offset_max

    return SignalQualityReport(
        clipping_detected=clipping_detected,
        clipping_percent=clipping_percent,
        adc_min=min_val,
        adc_max=max_val,
        high_noise=high_noise,
        noise_floor_db=float(snr_db),
        snr_db=float(snr_db),
        dc_offset_excessive=dc_offset_excessive,
        dc_offset=dc_offset,
    )


class SignalQualityReport:
    """Report of signal quality issues.

    : Warnings included in measurement results.

    Attributes:
        clipping_detected: Whether clipping was detected
        clipping_percent: Percentage of samples clipped
        adc_min: Minimum signal value
        adc_max: Maximum signal value
        high_noise: Whether noise floor is excessive
        noise_floor_db: Estimated noise floor in dB
        snr_db: Signal-to-noise ratio in dB
        dc_offset_excessive: Whether DC offset is excessive
        dc_offset: DC offset value

    Example:
        >>> from oscura.core.edge_cases import check_signal_quality
        >>> quality = check_signal_quality(signal)
        >>> print(quality.summary())

    References:
        EDGE-001: Signal Quality Warnings
    """

    def __init__(
        self,
        *,
        clipping_detected: bool = False,
        clipping_percent: float = 0.0,
        adc_min: float = 0.0,
        adc_max: float = 0.0,
        high_noise: bool = False,
        noise_floor_db: float = 0.0,
        snr_db: float = 0.0,
        dc_offset_excessive: bool = False,
        dc_offset: float = 0.0,
    ) -> None:
        """Initialize SignalQualityReport.

        Args:
            clipping_detected: Clipping detected flag
            clipping_percent: Percentage of clipped samples
            adc_min: Minimum signal value
            adc_max: Maximum signal value
            high_noise: High noise flag
            noise_floor_db: Noise floor in dB
            snr_db: Signal-to-noise ratio in dB
            dc_offset_excessive: Excessive DC offset flag
            dc_offset: DC offset value
        """
        self.clipping_detected = clipping_detected
        self.clipping_percent = clipping_percent
        self.adc_min = adc_min
        self.adc_max = adc_max
        self.high_noise = high_noise
        self.noise_floor_db = noise_floor_db
        self.snr_db = snr_db
        self.dc_offset_excessive = dc_offset_excessive
        self.dc_offset = dc_offset

    def has_issues(self) -> bool:
        """Check if any quality issues were detected.

        Returns:
            True if any issues found

        Example:
            >>> if quality.has_issues():
            ...     print(quality.summary())

        References:
            EDGE-001: Signal Quality Warnings
        """
        return self.clipping_detected or self.high_noise or self.dc_offset_excessive

    def summary(self) -> str:
        """Get text summary of quality issues.

        Returns:
            Summary string

        Example:
            >>> print(quality.summary())
            Signal Quality Report:
              ✓ No clipping detected
              ⚠ High noise floor: -45.2 dB
              ✓ DC offset within limits

        References:
            EDGE-001: Signal Quality Warnings
        """
        lines = ["Signal Quality Report:"]

        # Clipping
        if self.clipping_detected:
            lines.append(f"  ⚠ Clipping detected: {self.clipping_percent:.1f}% of samples")
            lines.append(f"    ADC range: {self.adc_min:.3f} to {self.adc_max:.3f}")
        else:
            lines.append("  ✓ No clipping detected")

        # Noise
        if self.high_noise:
            lines.append(f"  ⚠ High noise floor: {self.noise_floor_db:.1f} dB")
            lines.append(f"    SNR: {self.snr_db:.1f} dB")
        else:
            lines.append(f"  ✓ Noise floor acceptable (SNR: {self.snr_db:.1f} dB)")

        # DC offset
        if self.dc_offset_excessive:
            lines.append(f"  ⚠ DC offset: {self.dc_offset:.3f}")
        else:
            lines.append("  ✓ DC offset within limits")

        return "\n".join(lines)


__all__ = [
    "EmptyTraceError",
    "InsufficientSamplesError",
    "SignalQualityReport",
    "check_signal_quality",
    "check_single_sample",
    "handle_empty_trace",
    "sanitize_signal",
    "validate_signal",
]
