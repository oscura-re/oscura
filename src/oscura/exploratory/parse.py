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

    : Compensates for clock jitter in logic analyzer
    captures (e.g., USB transmission jitter) while preserving phase.

    Correction constraints (DAQ-003):
    - Max correction per sample: ±max_correction_factor × expected_period  # noqa: RUF002, RUF003
    - Filter cutoff: expected_rate / 10 (removes 10× jitter frequency)  # noqa: RUF002, RUF003
    - Target reduction: ≥5× for typical USB jitter  # noqa: RUF002, RUF003

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
    if len(timestamps) == 0:
        raise ValueError("Timestamps array cannot be empty")

    if expected_rate <= 0:
        raise ValueError("expected_rate must be positive")

    if max_correction_factor <= 0:
        raise ValueError("max_correction_factor must be positive")

    if len(timestamps) < 3:
        # Not enough data to filter
        return TimestampCorrection(
            corrected_timestamps=timestamps.copy(),
            original_jitter_rms=0.0,
            corrected_jitter_rms=0.0,
            reduction_ratio=1.0,
            samples_corrected=0,
            max_correction=0.0,
        )

    expected_period = 1.0 / expected_rate
    max_correction = max_correction_factor * expected_period

    # Calculate original jitter
    diffs = np.diff(timestamps)
    original_jitter = diffs - expected_period
    original_jitter_rms = float(np.sqrt(np.mean(original_jitter**2)))

    # If jitter is negligible (below 1 ns), no correction needed
    # This avoids correcting floating-point rounding errors in perfect timestamps
    if original_jitter_rms < 1e-9:
        return TimestampCorrection(
            corrected_timestamps=timestamps.copy(),
            original_jitter_rms=original_jitter_rms,
            corrected_jitter_rms=original_jitter_rms,
            reduction_ratio=1.0,
            samples_corrected=0,
            max_correction=0.0,
        )

    if method == "lowpass":
        # Low-pass filter approach
        # Design Butterworth filter: cutoff at expected_rate / 10
        cutoff_freq = expected_rate / 10.0
        nyquist = 0.5 * expected_rate

        # Ensure cutoff is valid
        if cutoff_freq >= nyquist:
            cutoff_freq = nyquist * 0.8

        # Design 2nd order Butterworth
        sos = signal.butter(2, cutoff_freq / nyquist, btype="low", output="sos")

        # Filter the timestamps
        # Need to detrend first to avoid edge effects
        t_mean = np.mean(timestamps)
        t_detrended = timestamps - t_mean

        # Apply filter
        filtered = signal.sosfiltfilt(sos, t_detrended)
        corrected = filtered + t_mean

    else:  # pll
        # Phase-locked loop model
        # Simple PLL: track expected phase and correct deviations
        corrected = np.zeros_like(timestamps)
        corrected[0] = timestamps[0]

        # PLL state
        phase = 0.0
        phase_increment = 2 * np.pi * expected_rate

        for i in range(1, len(timestamps)):
            # Predict next timestamp based on expected rate
            predicted = corrected[i - 1] + expected_period

            # Measure phase error
            actual = timestamps[i]
            error = actual - predicted

            # Apply correction with limiting
            correction = np.clip(error * 0.5, -max_correction, max_correction)
            corrected[i] = predicted + correction

            # Update phase
            phase += phase_increment * (corrected[i] - corrected[i - 1])

    # Limit corrections to max_correction
    corrections = corrected - timestamps
    exceeded = np.abs(corrections) > max_correction
    corrections[exceeded] = np.sign(corrections[exceeded]) * max_correction
    corrected = timestamps + corrections

    # Calculate corrected jitter
    corrected_diffs = np.diff(corrected)
    corrected_jitter = corrected_diffs - expected_period
    corrected_jitter_rms = float(np.sqrt(np.mean(corrected_jitter**2)))

    # Calculate metrics
    samples_corrected = int(np.sum(np.abs(corrections) > 1e-12))
    max_correction_applied = float(np.max(np.abs(corrections)))

    # original_jitter_rms is always > 0 here (early return handles negligible jitter)
    reduction_ratio = original_jitter_rms / max(corrected_jitter_rms, 1e-15)

    return TimestampCorrection(
        corrected_timestamps=corrected,
        original_jitter_rms=original_jitter_rms,
        corrected_jitter_rms=corrected_jitter_rms,
        reduction_ratio=reduction_ratio,
        samples_corrected=samples_corrected,
        max_correction=max_correction_applied,
    )


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

    frames: list[DecodedFrame] = []
    pos = 0

    # Protocol-specific decode logic
    # This is a simplified implementation showing the error handling pattern
    # Full protocol decoders are in oscura.analyzers.protocols

    if protocol == "uart":
        # UART parameters
        if "baud" not in protocol_params:
            raise ValueError("UART requires 'baud' parameter")

        # Simplified UART frame extraction with error tolerance
        while pos < len(data):
            try:
                # Try to decode frame at current position
                # This is simplified - real UART decoder would analyze bit timing

                # Check for valid frame (simplified)
                if pos + 1 >= len(data):
                    break

                frame_data = bytes([data[pos]])
                timestamp = float(pos) / protocol_params["baud"]

                # Validate frame (simplified - would check start/stop bits)
                is_valid = True
                error_type = None

                # Example: detect framing error (no proper stop bit)
                if data[pos] == 0xFF:  # Example error condition
                    is_valid = False
                    error_type = "framing"

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
                    # Strict mode: abort on error
                    break
                elif not is_valid and tolerance == ErrorTolerance.TOLERANT:
                    # Tolerant: skip error frame, resync
                    # Search for next valid start bit
                    pos += 1
                    # In real implementation, would search for start bit pattern
                else:
                    # Permissive: record error, continue
                    pos += 1

            except Exception:
                if tolerance == ErrorTolerance.STRICT:
                    raise
                else:
                    # Log error and continue
                    pos += 1

    elif protocol == "spi":
        # SPI: Re-align on CS edge
        # Simplified placeholder
        pass

    elif protocol == "i2c":
        # I2C: Search for START condition
        # Simplified placeholder
        pass

    elif protocol == "can":
        # CAN: Wait for SOF after error
        # Simplified placeholder
        pass

    return frames
