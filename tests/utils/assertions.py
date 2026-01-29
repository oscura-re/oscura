"""Custom test assertions for Oscura tests.

This module provides specialized assertion functions that give
better error messages and handle domain-specific comparisons.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def assert_signals_equal(
    actual: NDArray[np.floating[Any]],
    expected: NDArray[np.floating[Any]],
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    msg: str | None = None,
) -> None:
    """Assert that two signals are equal within tolerance.

    Provides detailed error messages showing where signals differ.

    Args:
        actual: The actual signal array.
        expected: The expected signal array.
        rtol: Relative tolerance (default: 1e-7).
        atol: Absolute tolerance (default: 0.0).
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If signals are not equal within tolerance.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Signal shapes differ: actual={actual.shape}, expected={expected.shape}"
            + (f"\n{msg}" if msg else "")
        )

    # Find differences
    diff = np.abs(actual - expected)
    rel_diff = diff / (np.abs(expected) + 1e-12)  # Avoid division by zero

    # Check tolerance
    within_tolerance = (diff <= atol) | (rel_diff <= rtol)

    if not np.all(within_tolerance):
        # Find the worst violations
        violation_indices = np.where(~within_tolerance)[0]
        num_violations = len(violation_indices)
        worst_idx = violation_indices[np.argmax(diff[violation_indices])]

        error_msg = (
            f"Signals differ at {num_violations} of {len(actual)} samples.\n"
            f"Worst violation at index {worst_idx}:\n"
            f"  actual[{worst_idx}] = {actual[worst_idx]}\n"
            f"  expected[{worst_idx}] = {expected[worst_idx]}\n"
            f"  diff = {diff[worst_idx]}, rel_diff = {rel_diff[worst_idx]}\n"
            f"Max absolute diff: {np.max(diff)}\n"
            f"Max relative diff: {np.max(rel_diff)}"
        )
        if msg:
            error_msg = f"{msg}\n{error_msg}"

        raise AssertionError(error_msg)


def assert_within_tolerance(
    actual: float | int | NDArray[Any],
    expected: float | int | NDArray[Any],
    *,
    tolerance: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    msg: str | None = None,
) -> None:
    """Assert that a value is within tolerance of expected.

    Supports both absolute and relative tolerance specifications.
    For convenience, you can use either 'tolerance' (absolute) or
    'rtol'/'atol' for more control.

    Args:
        actual: The actual value (scalar or array).
        expected: The expected value (scalar or array).
        tolerance: Simple absolute tolerance (shorthand for atol).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If value is not within tolerance.
    """
    # Handle tolerance parameters
    if tolerance is not None and (rtol is not None or atol is not None):
        raise ValueError("Cannot specify both 'tolerance' and 'rtol'/'atol'")

    if tolerance is not None:
        atol = tolerance
        rtol = 0.0
    else:
        atol = atol if atol is not None else 1e-8
        rtol = rtol if rtol is not None else 1e-5

    # Convert to numpy arrays for consistent handling
    actual_arr = np.atleast_1d(np.asarray(actual))
    expected_arr = np.atleast_1d(np.asarray(expected))

    if actual_arr.shape != expected_arr.shape:
        raise AssertionError(
            f"Shape mismatch: actual={actual_arr.shape}, expected={expected_arr.shape}"
            + (f"\n{msg}" if msg else "")
        )

    diff = np.abs(actual_arr - expected_arr)
    threshold = atol + rtol * np.abs(expected_arr)

    if not np.all(diff <= threshold):
        max_diff = np.max(diff)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)

        error_msg = (
            f"Value not within tolerance.\n"
            f"Max diff: {max_diff} (threshold: {threshold.flat[0] if threshold.size == 1 else 'varies'})\n"
            f"At index {max_idx}: actual={actual_arr[max_idx]}, expected={expected_arr[max_idx]}"
        )
        if msg:
            error_msg = f"{msg}\n{error_msg}"

        raise AssertionError(error_msg)


def assert_packet_valid(
    packet: bytes,
    *,
    sync_pattern: bytes | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    expected_fields: dict[str, tuple[int, int, Any]] | None = None,
    msg: str | None = None,
) -> None:
    """Assert that a packet is valid according to format rules.

    Args:
        packet: The packet bytes to validate.
        sync_pattern: Expected sync pattern at start (or None to skip check).
        min_size: Minimum valid packet size (or None to skip check).
        max_size: Maximum valid packet size (or None to skip check).
        expected_fields: Dict mapping field names to (offset, size, expected_value).
            Use None for expected_value to just check the field exists.
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If packet is invalid.

    Example:
        >>> assert_packet_valid(
        ...     packet,
        ...     sync_pattern=b"\\xaa\\x55",
        ...     min_size=8,
        ...     expected_fields={
        ...         "type": (2, 1, 0x01),
        ...         "length": (3, 2, None),  # Just check exists
        ...     }
        ... )
    """
    errors = []

    # Size checks
    if min_size is not None and len(packet) < min_size:
        errors.append(f"Packet too small: {len(packet)} < {min_size}")

    if max_size is not None and len(packet) > max_size:
        errors.append(f"Packet too large: {len(packet)} > {max_size}")

    # Sync pattern check
    if sync_pattern is not None:
        if not packet.startswith(sync_pattern):
            actual_start = packet[: len(sync_pattern)]
            errors.append(
                f"Sync pattern mismatch: got {actual_start.hex()}, expected {sync_pattern.hex()}"
            )

    # Field checks
    if expected_fields:
        for field_name, (offset, size, expected_value) in expected_fields.items():
            if offset + size > len(packet):
                errors.append(
                    f"Field '{field_name}' extends beyond packet end (offset={offset}, size={size})"
                )
                continue

            actual_bytes = packet[offset : offset + size]

            if expected_value is not None:
                if isinstance(expected_value, bytes):
                    if actual_bytes != expected_value:
                        errors.append(
                            f"Field '{field_name}' mismatch: got {actual_bytes.hex()}, "
                            f"expected {expected_value.hex()}"
                        )
                elif isinstance(expected_value, int):
                    # Interpret as little-endian integer
                    actual_int = int.from_bytes(actual_bytes, "little")
                    if actual_int != expected_value:
                        errors.append(
                            f"Field '{field_name}' mismatch: got {actual_int} (0x{actual_int:x}), "
                            f"expected {expected_value} (0x{expected_value:x})"
                        )

    if errors:
        error_msg = "Packet validation failed:\n  - " + "\n  - ".join(errors)
        if msg:
            error_msg = f"{msg}\n{error_msg}"
        raise AssertionError(error_msg)


def assert_frequency_detected(
    signal: NDArray[np.floating[Any]],
    sample_rate: float,
    expected_freq: float,
    *,
    tolerance_hz: float | None = None,
    tolerance_percent: float = 5.0,
    msg: str | None = None,
) -> None:
    """Assert that a signal contains the expected frequency.

    Uses FFT to detect the dominant frequency component.

    Args:
        signal: The signal to analyze.
        sample_rate: Sample rate in Hz.
        expected_freq: Expected dominant frequency in Hz.
        tolerance_hz: Absolute tolerance in Hz (or None to use tolerance_percent).
        tolerance_percent: Relative tolerance as percentage (default 5%).
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If expected frequency is not detected.
    """
    # Compute FFT
    n = len(signal)
    fft_result = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    magnitudes = np.abs(fft_result)

    # Find dominant frequency (excluding DC)
    dc_cutoff = max(1, int(n * 10 / sample_rate))  # Skip first 10 Hz worth
    dominant_idx = dc_cutoff + np.argmax(magnitudes[dc_cutoff:])
    detected_freq = freqs[dominant_idx]

    # Calculate tolerance
    if tolerance_hz is None:
        tolerance_hz = expected_freq * tolerance_percent / 100.0

    diff = abs(detected_freq - expected_freq)

    if diff > tolerance_hz:
        error_msg = (
            f"Frequency detection failed.\n"
            f"Expected: {expected_freq:.2f} Hz (+/- {tolerance_hz:.2f} Hz)\n"
            f"Detected: {detected_freq:.2f} Hz\n"
            f"Difference: {diff:.2f} Hz"
        )
        if msg:
            error_msg = f"{msg}\n{error_msg}"
        raise AssertionError(error_msg)


def assert_edges_at(
    signal: NDArray[np.floating[Any]],
    expected_indices: list[int],
    *,
    threshold: float | None = None,
    tolerance_samples: int = 1,
    rising_only: bool = False,
    falling_only: bool = False,
    msg: str | None = None,
) -> None:
    """Assert that signal edges occur at expected positions.

    Args:
        signal: The signal to analyze.
        expected_indices: List of expected edge positions (sample indices).
        threshold: Threshold for edge detection (or None for mid-point).
        tolerance_samples: Allowed deviation in samples (default 1).
        rising_only: Only check for rising edges.
        falling_only: Only check for falling edges.
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If edges don't match expected positions.
    """
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2

    # Detect edges
    above = signal > threshold
    edges = np.where(np.diff(above.astype(int)) != 0)[0] + 1

    if rising_only:
        edges = edges[np.diff(above.astype(int))[edges - 1] > 0]
    elif falling_only:
        edges = edges[np.diff(above.astype(int))[edges - 1] < 0]

    edges_list = edges.tolist()
    expected_set = set(expected_indices)
    detected_set = set(edges_list)

    # Check each expected edge has a match within tolerance
    missing = []
    for expected in expected_indices:
        found = False
        for detected in edges_list:
            if abs(detected - expected) <= tolerance_samples:
                found = True
                break
        if not found:
            missing.append(expected)

    # Check for unexpected edges
    extra = []
    for detected in edges_list:
        found = False
        for expected in expected_indices:
            if abs(detected - expected) <= tolerance_samples:
                found = True
                break
        if not found:
            extra.append(detected)

    if missing or extra:
        error_msg = "Edge detection mismatch:\n"
        if missing:
            error_msg += f"  Missing edges at: {missing}\n"
        if extra:
            error_msg += f"  Unexpected edges at: {extra}\n"
        error_msg += f"  Expected {len(expected_indices)} edges, detected {len(edges_list)}"
        if msg:
            error_msg = f"{msg}\n{error_msg}"
        raise AssertionError(error_msg)


def assert_checksum_valid(
    data: bytes,
    checksum_offset: int,
    checksum_size: int = 2,
    *,
    algorithm: str = "crc16",
    data_end: int | None = None,
    msg: str | None = None,
) -> None:
    """Assert that a packet's checksum is valid.

    Args:
        data: The complete data including checksum.
        checksum_offset: Byte offset where checksum starts.
        checksum_size: Size of checksum in bytes (default 2).
        algorithm: Checksum algorithm ('crc16', 'sum8', 'xor').
        data_end: End of data to checksum (default: checksum_offset).
        msg: Optional message to include on failure.

    Raises:
        AssertionError: If checksum is invalid.
    """
    if data_end is None:
        data_end = checksum_offset

    payload = data[:data_end]
    stored_checksum = int.from_bytes(
        data[checksum_offset : checksum_offset + checksum_size], "little"
    )

    if algorithm == "crc16":
        calculated = _calculate_crc16(payload)
    elif algorithm == "sum8":
        calculated = sum(payload) & 0xFF
    elif algorithm == "xor":
        calculated = 0
        for b in payload:
            calculated ^= b
    else:
        raise ValueError(f"Unknown checksum algorithm: {algorithm}")

    if calculated != stored_checksum:
        error_msg = (
            f"Checksum validation failed ({algorithm}).\n"
            f"Calculated: 0x{calculated:04x}\n"
            f"Stored: 0x{stored_checksum:04x}"
        )
        if msg:
            error_msg = f"{msg}\n{error_msg}"
        raise AssertionError(error_msg)


def _calculate_crc16(data: bytes) -> int:
    """Calculate CRC-16 (Modbus variant)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF
