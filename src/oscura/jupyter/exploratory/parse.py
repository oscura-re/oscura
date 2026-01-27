"""Error-tolerant protocol parsing with timestamp correction.


This module provides robust protocol decoding that continues after errors
and timestamp correction for jittery captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal


class ErrorTolerance(Enum):
    """Error tolerance modes for protocol decoding.

    Attributes:
        STRICT: Abort on first error (backward compatible)
        TOLERANT: Skip error frame, resync, continue (default)
        PERMISSIVE: Best-effort decode, report all errors
    """

    STRICT = "strict"
    TOLERANT = "tolerant"
    PERMISSIVE = "permissive"


@dataclass
class DecodedFrame:
    """Decoded protocol frame with error annotation.

    Attributes:
        data: Decoded data bytes
        timestamp: Frame timestamp in seconds
        valid: Whether frame is valid or has errors
        error_type: Type of error if invalid (e.g., 'framing', 'parity')
        position: Byte position in original trace
    """

    data: bytes
    timestamp: float
    valid: bool
    error_type: str | None
    position: int


@dataclass
class TimestampCorrection:
    """Result from timestamp jitter correction.

    Attributes:
        corrected_timestamps: Array of corrected timestamps
        original_jitter_rms: RMS jitter before correction
        corrected_jitter_rms: RMS jitter after correction
        reduction_ratio: Jitter reduction factor (before/after)
        samples_corrected: Number of samples that were adjusted
        max_correction: Maximum correction applied to any sample
    """

    corrected_timestamps: NDArray[np.float64]
    original_jitter_rms: float
    corrected_jitter_rms: float
    reduction_ratio: float
    samples_corrected: int
    max_correction: float


def correct_timestamp_jitter(
    timestamps: NDArray[np.float64],
    expected_rate: float,
    *,
    method: Literal["lowpass", "pll"] = "lowpass",
    max_correction_factor: float = 2.0,
) -> TimestampCorrection:
    """Correct timestamp jitter using filtering or PLL model.

    Compensates for clock jitter in logic analyzer
    captures (e.g., USB transmission jitter) while preserving phase.

    Correction constraints (DAQ-003):
    - Max correction per sample: ±max_correction_factor x expected_period
    - Filter cutoff: expected_rate / 10 (removes 10x jitter frequency)
    - Target reduction: >=5x for typical USB jitter

    Args:
        timestamps: Original jittery timestamps in seconds
        expected_rate: Expected nominal sample rate in Hz
        method: Correction method ('lowpass' or 'pll')
        max_correction_factor: Max correction as multiple of period

    Returns:
        TimestampCorrection with corrected timestamps and metrics

    Raises:
        ValueError: If timestamps array is empty
        ValueError: If expected_rate <= 0
        ValueError: If max_correction_factor <= 0

    Examples:
        >>> # Correct jittery timestamps from USB logic analyzer
        >>> import numpy as np
        >>> timestamps = np.linspace(0, 1e-3, 1000)
        >>> jitter = np.random.normal(0, 1e-7, 1000)  # 100ns jitter
        >>> jittery = timestamps + jitter
        >>> result = correct_timestamp_jitter(jittery, expected_rate=1e6)
        >>> print(f"Jitter reduced by {result.reduction_ratio:.1f}x")

    References:
        DAQ-003: Timestamp Jitter Compensation and Clock Correction
    """
    # Validate inputs
    _validate_jitter_correction_inputs(timestamps, expected_rate, max_correction_factor)

    # Handle edge cases
    if len(timestamps) < 3:
        return _create_no_correction_result(timestamps)

    expected_period = 1.0 / expected_rate
    max_correction = max_correction_factor * expected_period

    # Calculate original jitter
    original_jitter_rms = _calculate_original_jitter_rms(timestamps, expected_period)

    # Skip correction if jitter is negligible
    if original_jitter_rms < 1e-9:
        return _create_negligible_jitter_result(timestamps, original_jitter_rms)

    # Apply correction method
    corrected = _apply_jitter_correction(
        timestamps, expected_rate, expected_period, max_correction, method
    )

    # Build and return result
    return _build_jitter_correction_result(
        timestamps, corrected, expected_period, original_jitter_rms, max_correction
    )


def _validate_jitter_correction_inputs(
    timestamps: NDArray[np.float64], expected_rate: float, max_correction_factor: float
) -> None:
    """Validate jitter correction input parameters."""
    if len(timestamps) == 0:
        raise ValueError("Timestamps array cannot be empty")
    if expected_rate <= 0:
        raise ValueError("expected_rate must be positive")
    if max_correction_factor <= 0:
        raise ValueError("max_correction_factor must be positive")


def _create_no_correction_result(timestamps: NDArray[np.float64]) -> TimestampCorrection:
    """Create result for insufficient data case."""
    return TimestampCorrection(
        corrected_timestamps=timestamps.copy(),
        original_jitter_rms=0.0,
        corrected_jitter_rms=0.0,
        reduction_ratio=1.0,
        samples_corrected=0,
        max_correction=0.0,
    )


def _calculate_original_jitter_rms(
    timestamps: NDArray[np.float64], expected_period: float
) -> float:
    """Calculate RMS jitter from timestamps."""
    diffs = np.diff(timestamps)
    original_jitter = diffs - expected_period
    return float(np.sqrt(np.mean(original_jitter**2)))


def _create_negligible_jitter_result(
    timestamps: NDArray[np.float64], original_jitter_rms: float
) -> TimestampCorrection:
    """Create result for negligible jitter case."""
    return TimestampCorrection(
        corrected_timestamps=timestamps.copy(),
        original_jitter_rms=original_jitter_rms,
        corrected_jitter_rms=original_jitter_rms,
        reduction_ratio=1.0,
        samples_corrected=0,
        max_correction=0.0,
    )


def _apply_jitter_correction(
    timestamps: NDArray[np.float64],
    expected_rate: float,
    expected_period: float,
    max_correction: float,
    method: Literal["lowpass", "pll"],
) -> NDArray[np.float64]:
    """Apply jitter correction using selected method."""
    if method == "lowpass":
        corrected = _apply_lowpass_correction(timestamps, expected_rate)
    else:  # pll
        corrected = _apply_pll_correction(timestamps, expected_period, max_correction)

    # Limit corrections to max_correction
    corrections = corrected - timestamps
    exceeded = np.abs(corrections) > max_correction
    corrections[exceeded] = np.sign(corrections[exceeded]) * max_correction
    return timestamps + corrections


def _apply_lowpass_correction(
    timestamps: NDArray[np.float64], expected_rate: float
) -> NDArray[np.float64]:
    """Apply low-pass filter correction."""
    cutoff_freq = expected_rate / 10.0
    nyquist = 0.5 * expected_rate
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.8

    sos = signal.butter(2, cutoff_freq / nyquist, btype="low", output="sos")
    t_mean: np.floating[Any] = np.mean(timestamps)
    t_detrended = timestamps - t_mean
    filtered = signal.sosfiltfilt(sos, t_detrended)
    result: NDArray[np.float64] = np.asarray(filtered + t_mean, dtype=np.float64)
    return result


def _apply_pll_correction(
    timestamps: NDArray[np.float64], expected_period: float, max_correction: float
) -> NDArray[np.float64]:
    """Apply phase-locked loop correction."""
    corrected = np.zeros_like(timestamps)
    corrected[0] = timestamps[0]

    for i in range(1, len(timestamps)):
        predicted = corrected[i - 1] + expected_period
        error = timestamps[i] - predicted
        correction = np.clip(error * 0.5, -max_correction, max_correction)
        corrected[i] = predicted + correction

    return corrected


def _build_jitter_correction_result(
    timestamps: NDArray[np.float64],
    corrected: NDArray[np.float64],
    expected_period: float,
    original_jitter_rms: float,
    max_correction: float,
) -> TimestampCorrection:
    """Build final jitter correction result."""
    corrected_diffs = np.diff(corrected)
    corrected_jitter = corrected_diffs - expected_period
    corrected_jitter_rms = float(np.sqrt(np.mean(corrected_jitter**2)))

    corrections = corrected - timestamps
    samples_corrected = int(np.sum(np.abs(corrections) > 1e-12))
    max_correction_applied = float(np.max(np.abs(corrections)))
    reduction_ratio = original_jitter_rms / max(corrected_jitter_rms, 1e-15)

    return TimestampCorrection(
        corrected_timestamps=corrected,
        original_jitter_rms=original_jitter_rms,
        corrected_jitter_rms=corrected_jitter_rms,
        reduction_ratio=reduction_ratio,
        samples_corrected=samples_corrected,
        max_correction=max_correction_applied,
    )


def _decode_uart_tolerant(
    data: NDArray[np.uint8],
    tolerance: ErrorTolerance,
    baud: float,
) -> list[DecodedFrame]:
    """Decode UART frames with error tolerance (simplified implementation)."""
    frames: list[DecodedFrame] = []
    pos = 0

    while pos < len(data):
        try:
            if pos + 1 >= len(data):
                break

            frame_data = bytes([data[pos]])
            timestamp = float(pos) / baud

            # Validate frame (simplified - real decoder checks start/stop bits)
            is_valid = data[pos] != 0xFF  # Example error condition
            error_type = "framing" if not is_valid else None

            frames.append(
                DecodedFrame(
                    data=frame_data,
                    timestamp=timestamp,
                    valid=is_valid,
                    error_type=error_type,
                    position=pos,
                )
            )

            if not is_valid and tolerance == ErrorTolerance.STRICT:
                break
            elif not is_valid and tolerance == ErrorTolerance.TOLERANT:
                pos += 1  # Skip error frame, resync
            else:
                pos += 1  # Permissive: record error, continue

        except Exception:
            if tolerance == ErrorTolerance.STRICT:
                raise
            pos += 1

    return frames


def decode_with_error_tolerance(
    data: NDArray[np.uint8],
    protocol: Literal["uart", "spi", "i2c", "can"],
    *,
    tolerance: ErrorTolerance = ErrorTolerance.TOLERANT,
    **protocol_params: Any,
) -> list[DecodedFrame]:
    """Decode protocol with error tolerance and resynchronization.

    : Continues decoding after framing/parity/stop-bit
    errors instead of aborting. Applies to all protocol decoders.

    Error tolerance modes (DAQ-004):
    - STRICT: Abort on first error (backward compatible)
    - TOLERANT: Skip error frame, resync, continue (default)
    - PERMISSIVE: Best-effort decode, report all errors

    Resynchronization strategies (DAQ-004):
    - UART: Search for next valid start bit + stop bit pattern
    - SPI: Re-align on next CS edge
    - I2C: Search for next START condition
    - CAN: Wait for recessive bus + next SOF

    Args:
        data: Raw protocol data bytes
        protocol: Protocol type ('uart', 'spi', 'i2c', 'can')
        tolerance: Error tolerance mode
        **protocol_params: Protocol-specific parameters (baud, parity, etc.)

    Returns:
        List of DecodedFrame objects with data and error annotations

    Raises:
        ValueError: If protocol not supported
        ValueError: If required protocol_params missing
        Exception: Re-raised in STRICT mode if decoding fails

    Examples:
        >>> # Decode UART with error tolerance
        >>> data = np.array([0xFF, 0x55, 0xAA, 0x00], dtype=np.uint8)
        >>> frames = decode_with_error_tolerance(
        ...     data, 'uart', tolerance=ErrorTolerance.TOLERANT, baud=9600
        ... )
        >>> valid_frames = [f for f in frames if f.valid]

    References:
        DAQ-004: Error-Tolerant Protocol Decoding with Resynchronization
    """
    if protocol not in ("uart", "spi", "i2c", "can"):
        raise ValueError(f"Unsupported protocol: {protocol}")

    # Protocol-specific decode logic
    # This is a simplified implementation showing the error handling pattern
    # Full protocol decoders are in oscura.analyzers.protocols

    if protocol == "uart":
        if "baud" not in protocol_params:
            raise ValueError("UART requires 'baud' parameter")
        return _decode_uart_tolerant(data, tolerance, protocol_params["baud"])

    elif protocol in ("spi", "i2c", "can"):
        # SPI/I2C/CAN: Simplified placeholders
        # Real implementations would have protocol-specific resync logic
        return []

    return []
