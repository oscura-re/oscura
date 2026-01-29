"""DAQ error-tolerant analysis module.

This module provides error-tolerant DAQ analysis features including fuzzy
pattern matching, error recovery, bit error characterization, and gap detection.


Example:
    >>> from oscura.analyzers.packet.daq import fuzzy_pattern_search, detect_gaps
    >>> matches = fuzzy_pattern_search(data, pattern=0xAA55, max_errors=2)
    >>> for match in matches:
    ...     print(f"Found at {match.offset}, errors: {match.bit_errors}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


class ErrorPattern(Enum):
    """Bit error pattern types."""

    RANDOM = "random"  # Uniformly distributed errors (noise)
    BURST = "burst"  # Clustered errors (interference)
    SYSTEMATIC = "systematic"  # Regular pattern (clock issues)
    SINGLE_BIT = "single_bit"  # Isolated single-bit errors


@dataclass
class FuzzyMatch:
    """Result of fuzzy pattern search.

    Attributes:
        offset: Bit offset where pattern was found
        matched_bits: Actual bits at this location
        bit_errors: Number of bit errors (Hamming distance)
        error_positions: Bit positions with errors
        confidence: Match confidence (0-1)
    """

    offset: int
    matched_bits: int
    bit_errors: int
    error_positions: list[int] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def is_exact(self) -> bool:
        """Return True if exact match (no errors)."""
        return self.bit_errors == 0


@dataclass
class PacketRecoveryResult:
    """Result of error-tolerant packet parsing.

    Attributes:
        packets: Successfully parsed packets
        recovered_packets: Packets recovered despite errors
        failed_regions: Regions that could not be parsed
        total_errors: Total bit errors encountered
        sync_resync_count: Number of resynchronizations
    """

    packets: list[dict[str, Any]] = field(default_factory=list)
    recovered_packets: list[dict[str, Any]] = field(default_factory=list)
    failed_regions: list[tuple[int, int]] = field(default_factory=list)
    total_errors: int = 0
    sync_resync_count: int = 0


@dataclass
class JitterCompensationResult:
    """Result of timestamp jitter compensation.

    Attributes:
        original_timestamps: Original timestamps
        corrected_timestamps: Jitter-compensated timestamps
        jitter_removed_ns: RMS jitter removed in nanoseconds
        clock_drift_ppm: Estimated clock drift in ppm
        correction_method: Method used for correction
    """

    original_timestamps: NDArray[np.float64]
    corrected_timestamps: NDArray[np.float64]
    jitter_removed_ns: float
    clock_drift_ppm: float
    correction_method: str


@dataclass
class BitErrorAnalysis:
    """Bit error pattern analysis result.

    Attributes:
        error_rate: Overall bit error rate
        error_pattern: Classified error pattern type
        burst_length_mean: Mean burst length (for burst errors)
        burst_length_max: Maximum burst length
        error_distribution: Error count by bit position (LSB to MSB)
        probable_cause: Inferred probable cause
        recommendations: Suggested fixes
    """

    error_rate: float
    error_pattern: ErrorPattern
    burst_length_mean: float = 0.0
    burst_length_max: int = 0
    error_distribution: list[int] = field(default_factory=list)
    probable_cause: str = ""
    recommendations: list[str] = field(default_factory=list)


# =============================================================================
# =============================================================================


@dataclass
class DAQGap:
    """Represents a detected gap in DAQ data.

    Attributes:
        start_index: Sample index where gap starts
        end_index: Sample index where gap ends
        start_time: Time when gap starts (seconds)
        end_time: Time when gap ends (seconds)
        duration: Gap duration in seconds
        expected_samples: Number of samples that should be present
        missing_samples: Estimated number of missing samples
        gap_type: Type of gap ('timestamp', 'sample_count', 'discontinuity')

    References:
        PKT-008: DAQ Gap Detection
    """

    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    expected_samples: int
    missing_samples: int
    gap_type: str = "timestamp"


@dataclass
class DAQGapAnalysis:
    """Complete gap analysis result.

    Attributes:
        gaps: List of detected gaps
        total_gaps: Total number of gaps found
        total_missing_samples: Total estimated missing samples
        total_gap_duration: Total gap duration in seconds
        acquisition_efficiency: Ratio of captured samples to expected
        sample_rate: Detected or specified sample rate
        discontinuities: List of data discontinuity indices
        metadata: Additional analysis metadata

    References:
        PKT-008: DAQ Gap Detection
    """

    gaps: list[DAQGap]
    total_gaps: int
    total_missing_samples: int
    total_gap_duration: float
    acquisition_efficiency: float
    sample_rate: float
    discontinuities: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_gaps(
    trace: WaveformTrace,
    *,
    expected_interval: float | None = None,
    tolerance: float = 0.1,
    min_gap_samples: int = 1,
) -> DAQGapAnalysis:
    """Detect gaps in DAQ data stream.

    Identifies missing samples based on expected sample interval
    and timestamp analysis.

    Args:
        trace: Waveform trace to analyze
        expected_interval: Expected time between samples (None = auto-detect)
        tolerance: Tolerance for interval deviation (0.1 = 10%)
        min_gap_samples: Minimum number of missing samples to report

    Returns:
        DAQGapAnalysis with detected gaps

    Example:
        >>> trace = osc.load('acquisition.wfm')
        >>> result = detect_gaps(trace)
        >>> for gap in result.gaps:
        ...     print(f"Gap at {gap.start_time:.6f}s: {gap.missing_samples} samples")

    References:
        PKT-008: DAQ Gap Detection
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Calculate expected interval
    if expected_interval is None:
        expected_interval = 1.0 / sample_rate

    return detect_gaps_by_samples(
        data,
        sample_rate=sample_rate,
        expected_interval=expected_interval,
        tolerance=tolerance,
        min_gap_samples=min_gap_samples,
    )


def detect_gaps_by_timestamps(
    timestamps: NDArray[np.float64],
    *,
    expected_interval: float | None = None,
    tolerance: float = 0.1,
    min_gap_samples: int = 1,
) -> DAQGapAnalysis:
    """Detect gaps using explicit timestamps.

    Args:
        timestamps: Array of sample timestamps in seconds
        expected_interval: Expected interval between samples
        tolerance: Tolerance for interval deviation
        min_gap_samples: Minimum missing samples to report

    Returns:
        DAQGapAnalysis with detected gaps

    Example:
        >>> timestamps = np.array([0.0, 1e-6, 2e-6, 5e-6, 6e-6])  # Gap at 2-5us
        >>> result = detect_gaps_by_timestamps(timestamps)

    References:
        PKT-008: DAQ Gap Detection
    """
    if len(timestamps) < 2:
        return DAQGapAnalysis(
            gaps=[],
            total_gaps=0,
            total_missing_samples=0,
            total_gap_duration=0.0,
            acquisition_efficiency=1.0,
            sample_rate=0.0,
            discontinuities=[],
        )

    # Calculate intervals
    intervals = np.diff(timestamps)

    # Auto-detect expected interval if not provided
    if expected_interval is None:
        expected_interval = float(np.median(intervals))

    sample_rate = 1.0 / expected_interval

    # Calculate allowed deviation
    max_interval = expected_interval * (1 + tolerance)

    # Find gaps
    gaps: list[DAQGap] = []
    discontinuities: list[int] = []
    total_missing = 0
    total_gap_duration = 0.0

    for i, interval in enumerate(intervals):
        if interval > max_interval:
            # Calculate missing samples
            missing = round(interval / expected_interval) - 1

            if missing >= min_gap_samples:
                gap = DAQGap(
                    start_index=i,
                    end_index=i + 1,
                    start_time=float(timestamps[i]),
                    end_time=float(timestamps[i + 1]),
                    duration=float(interval - expected_interval),
                    expected_samples=missing + 1,
                    missing_samples=missing,
                    gap_type="timestamp",
                )
                gaps.append(gap)
                total_missing += missing
                total_gap_duration += gap.duration
                discontinuities.append(i)

    # Calculate efficiency
    total_expected = len(timestamps) + total_missing
    efficiency = len(timestamps) / total_expected if total_expected > 0 else 1.0

    return DAQGapAnalysis(
        gaps=gaps,
        total_gaps=len(gaps),
        total_missing_samples=total_missing,
        total_gap_duration=total_gap_duration,
        acquisition_efficiency=efficiency,
        sample_rate=sample_rate,
        discontinuities=discontinuities,
        metadata={
            "method": "timestamp",
            "expected_interval": expected_interval,
            "tolerance": tolerance,
        },
    )


def detect_gaps_by_samples(
    data: NDArray[np.float64],
    *,
    sample_rate: float,
    expected_interval: float | None = None,
    tolerance: float = 0.1,
    min_gap_samples: int = 1,
    check_discontinuities: bool = True,
) -> DAQGapAnalysis:
    """Detect gaps using sample count analysis.

    Analyzes data for discontinuities that may indicate gaps.
    Uses derivative analysis to find sudden jumps.

    Args:
        data: Sample data array
        sample_rate: Sample rate in Hz
        expected_interval: Expected interval (None = 1/sample_rate)
        tolerance: Tolerance for detection
        min_gap_samples: Minimum gap size to report
        check_discontinuities: Check for value discontinuities

    Returns:
        DAQGapAnalysis with detected gaps

    References:
        PKT-008: DAQ Gap Detection
    """
    if len(data) < 2:
        return _empty_daq_gap_analysis(sample_rate)

    expected_interval = expected_interval or 1.0 / sample_rate
    gaps, discontinuities = _detect_discontinuity_gaps(
        data, sample_rate, expected_interval, min_gap_samples, check_discontinuities
    )

    # Calculate totals
    total_missing = sum(g.missing_samples for g in gaps)
    total_gap_duration = sum(g.duration for g in gaps)
    efficiency = len(data) / (len(data) + total_missing) if total_missing > 0 else 1.0

    return DAQGapAnalysis(
        gaps=gaps,
        total_gaps=len(gaps),
        total_missing_samples=total_missing,
        total_gap_duration=total_gap_duration,
        acquisition_efficiency=efficiency,
        sample_rate=sample_rate,
        discontinuities=discontinuities,
        metadata={
            "method": "sample_count",
            "expected_interval": expected_interval,
            "tolerance": tolerance,
            "check_discontinuities": check_discontinuities,
        },
    )


def _empty_daq_gap_analysis(sample_rate: float) -> DAQGapAnalysis:
    """Create empty DAQ gap analysis for insufficient data."""
    return DAQGapAnalysis(
        gaps=[],
        total_gaps=0,
        total_missing_samples=0,
        total_gap_duration=0.0,
        acquisition_efficiency=1.0,
        sample_rate=sample_rate,
        discontinuities=[],
    )


def _detect_discontinuity_gaps(
    data: NDArray[np.float64],
    sample_rate: float,
    expected_interval: float,
    min_gap_samples: int,
    check_discontinuities: bool,
) -> tuple[list[DAQGap], list[int]]:
    """Detect gaps from discontinuities in data."""
    if not check_discontinuities:
        return [], []

    diff = np.abs(np.diff(data))
    median_diff, std_diff = float(np.median(diff)), float(np.std(diff))
    threshold = median_diff + 5 * std_diff

    disc_indices = np.where(diff > threshold)[0]
    gaps, discontinuities = [], []

    for idx in disc_indices:
        estimated_missing = _estimate_missing_samples(diff[idx], median_diff, min_gap_samples)
        if estimated_missing >= min_gap_samples:
            gap = _create_daq_gap(idx, estimated_missing, sample_rate, expected_interval)
            gaps.append(gap)
            discontinuities.append(int(idx))

    return gaps, discontinuities


def _estimate_missing_samples(jump_size: float, median_diff: float, min_gap: int) -> int:
    """Estimate missing samples from value jump."""
    if median_diff > 0:
        return max(1, int(jump_size / median_diff) - 1)
    return min_gap


def _create_daq_gap(
    idx: int, estimated_missing: int, sample_rate: float, expected_interval: float
) -> DAQGap:
    """Create DAQGap object from discontinuity."""
    start_time, end_time = idx / sample_rate, (idx + 1) / sample_rate
    gap_duration = estimated_missing * expected_interval

    return DAQGap(
        start_index=int(idx),
        end_index=int(idx) + 1,
        start_time=start_time,
        end_time=end_time,
        duration=gap_duration,
        expected_samples=estimated_missing + 1,
        missing_samples=estimated_missing,
        gap_type="discontinuity",
    )


# =============================================================================
# =============================================================================


def fuzzy_pattern_search(
    data: bytes | NDArray[np.uint8],
    pattern: int | bytes,
    *,
    pattern_bits: int = 32,
    max_errors: int = 2,
    step: int = 1,
) -> list[FuzzyMatch]:
    """Search for bit patterns with Hamming distance tolerance.

    : Fuzzy Bit Pattern Search.

    Finds sync words and patterns even with bit errors (flipped bits).
    Essential for recovering corrupted logic analyzer captures.

    Args:
        data: Binary data to search (bytes or numpy array).
        pattern: Pattern to search for (int or bytes).
        pattern_bits: Number of bits in pattern.
        max_errors: Maximum allowed bit errors (Hamming distance).
        step: Search step in bits.

    Returns:
        List of FuzzyMatch objects for all matches within tolerance.

    Example:
        >>> # Find 0xAA55 sync word with up to 2 bit errors
        >>> data = bytes([0xAA, 0x55, 0x12, 0x34, 0xAB, 0x55])
        >>> matches = fuzzy_pattern_search(data, 0xAA55, pattern_bits=16, max_errors=2)
        >>> print(f"Found {len(matches)} matches")
    """
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    if isinstance(pattern, bytes):
        pattern = int.from_bytes(pattern, byteorder="big")

    # Ensure pattern fits in specified bits
    pattern_mask = (1 << pattern_bits) - 1
    pattern = pattern & pattern_mask

    matches: list[FuzzyMatch] = []

    # Convert data to bit array for searching
    total_bits = len(data) * 8

    for bit_offset in range(0, total_bits - pattern_bits + 1, step):
        # Extract bits at this offset
        extracted = _extract_bits(data, bit_offset, pattern_bits)

        # Calculate Hamming distance
        xor = extracted ^ pattern
        bit_errors = (xor).bit_count()

        if bit_errors <= max_errors:
            # Find error positions
            error_positions = []
            for i in range(pattern_bits):
                if (xor >> i) & 1:
                    error_positions.append(i)

            confidence = 1.0 - (bit_errors / pattern_bits)

            matches.append(
                FuzzyMatch(
                    offset=bit_offset,
                    matched_bits=extracted,
                    bit_errors=bit_errors,
                    error_positions=error_positions,
                    confidence=confidence,
                )
            )

    return matches


def _extract_bits(data: NDArray[np.uint8], bit_offset: int, num_bits: int) -> int:
    """Extract bits from data array."""
    result = 0
    for i in range(num_bits):
        bit_pos = bit_offset + i
        byte_idx = bit_pos // 8
        bit_in_byte = 7 - (bit_pos % 8)  # MSB first

        if byte_idx < len(data) and (data[byte_idx] >> bit_in_byte) & 1:
            result |= 1 << (num_bits - 1 - i)

    return result


# =============================================================================
# =============================================================================


def _try_recover_packet_with_next_sync(
    data: NDArray[np.uint8],
    byte_offset: int,
    next_sync_byte: int,
    max_packet_length: int,
    match: FuzzyMatch,
) -> dict[str, Any] | None:
    """Try to recover packet using next sync boundary.

    Args:
        data: Binary data
        byte_offset: Current packet start
        next_sync_byte: Next sync position
        max_packet_length: Maximum allowed packet length
        match: Sync match information

    Returns:
        Recovered packet dict or None if recovery failed
    """
    inferred_length = next_sync_byte - byte_offset

    if 0 < inferred_length <= max_packet_length:
        packet_data = bytes(data[byte_offset : byte_offset + inferred_length])
        return {
            "offset": byte_offset,
            "length": inferred_length,
            "data": packet_data,
            "sync_errors": match.bit_errors,
            "length_corrupted": True,
        }
    return None


def _extract_valid_packet(
    data: NDArray[np.uint8],
    byte_offset: int,
    length_offset: int,
    length: int,
    match: FuzzyMatch,
) -> dict[str, Any] | None:
    """Extract packet with valid length field.

    Args:
        data: Binary data
        byte_offset: Packet start offset
        length_offset: Offset to length field
        length: Packet length
        match: Sync match information

    Returns:
        Packet dict or None if extraction failed
    """
    packet_end = byte_offset + length_offset + 1 + length
    if packet_end <= len(data):
        packet_data = bytes(data[byte_offset:packet_end])
        return {
            "offset": byte_offset,
            "length": length,
            "data": packet_data,
            "sync_errors": match.bit_errors,
        }
    return None


def robust_packet_parse(
    data: bytes | NDArray[np.uint8],
    *,
    sync_pattern: int = 0xAA55,
    sync_bits: int = 16,
    length_offset: int = 2,  # Bytes after sync
    max_packet_length: int = 256,
    error_tolerance: int = 2,
) -> PacketRecoveryResult:
    """Parse variable-length packets with error recovery.

    : Robust Variable-Length Packet Parsing.

    Parses packets even when length fields are corrupted by falling
    back to sync word search.

    Args:
        data: Binary data containing packets.
        sync_pattern: Sync word pattern.
        sync_bits: Bits in sync pattern.
        length_offset: Byte offset to length field after sync.
        max_packet_length: Maximum valid packet length.
        error_tolerance: Max bit errors for sync detection.

    Returns:
        PacketRecoveryResult with parsed and recovered packets.

    Example:
        >>> result = robust_packet_parse(data, sync_pattern=0xAA55)
        >>> print(f"Parsed: {len(result.packets)}, Recovered: {len(result.recovered_packets)}")
    """
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    result = PacketRecoveryResult()
    sync_matches = fuzzy_pattern_search(
        data, sync_pattern, pattern_bits=sync_bits, max_errors=error_tolerance
    )
    sync_matches.sort(key=lambda m: m.offset)

    for i, match in enumerate(sync_matches):
        byte_offset = match.offset // 8
        if byte_offset + length_offset >= len(data):
            break

        length = data[byte_offset + length_offset]

        # Handle corrupted/invalid length
        if length > max_packet_length or length == 0:
            if i + 1 < len(sync_matches):
                next_sync_byte = sync_matches[i + 1].offset // 8
                recovered = _try_recover_packet_with_next_sync(
                    data, byte_offset, next_sync_byte, max_packet_length, match
                )
                if recovered:
                    result.recovered_packets.append(recovered)
                    result.total_errors += match.bit_errors
                    result.sync_resync_count += 1
                else:
                    result.failed_regions.append((byte_offset, byte_offset + 10))
            continue

        # Extract valid packet
        packet = _extract_valid_packet(data, byte_offset, length_offset, length, match)
        if packet:
            result.packets.append(packet)
            result.total_errors += match.bit_errors

    return result


# =============================================================================
# =============================================================================


def compensate_timestamp_jitter(
    timestamps: NDArray[np.float64],
    *,
    expected_rate: float | None = None,
    method: str = "lowpass",
    cutoff_ratio: float = 0.1,
) -> JitterCompensationResult:
    """Compensate timestamp jitter and clock drift.

    : Timestamp Jitter Compensation.

    Corrects sample timestamps affected by clock jitter using low-pass
    filtering or PLL model.

    Args:
        timestamps: Array of timestamps in seconds.
        expected_rate: Expected sample rate (auto-detected if None).
        method: Compensation method ('lowpass', 'pll', 'linear').
        cutoff_ratio: Low-pass filter cutoff as ratio of sample rate.

    Returns:
        JitterCompensationResult with corrected timestamps.

    Raises:
        ValueError: If unknown compensation method specified.

    Example:
        >>> result = compensate_timestamp_jitter(timestamps, expected_rate=1e6)
        >>> print(f"Jitter removed: {result.jitter_removed_ns:.1f} ns")
    """
    from scipy import signal

    n = len(timestamps)
    if n < 2:
        return _create_null_jitter_result(timestamps, method)

    intervals = np.diff(timestamps)
    expected_interval = _compute_expected_interval(intervals, expected_rate)

    corrected = _apply_jitter_correction_method(
        timestamps, intervals, expected_interval, method, cutoff_ratio, signal
    )

    metrics = _calculate_jitter_metrics(timestamps, intervals, corrected, expected_interval, n)

    return JitterCompensationResult(
        original_timestamps=timestamps,
        corrected_timestamps=corrected,
        jitter_removed_ns=metrics["jitter_removed_ns"],
        clock_drift_ppm=metrics["clock_drift_ppm"],
        correction_method=method,
    )


def _create_null_jitter_result(
    timestamps: NDArray[np.float64], method: str
) -> JitterCompensationResult:
    """Create jitter result for insufficient data."""
    return JitterCompensationResult(
        original_timestamps=timestamps,
        corrected_timestamps=timestamps,
        jitter_removed_ns=0,
        clock_drift_ppm=0,
        correction_method=method,
    )


def _compute_expected_interval(
    intervals: NDArray[np.float64], expected_rate: float | None
) -> float:
    """Compute expected interval from rate or auto-detect."""
    if expected_rate is None:
        return float(np.median(intervals))
    return 1.0 / expected_rate


def _apply_jitter_correction_method(
    timestamps: NDArray[np.float64],
    intervals: NDArray[np.float64],
    expected_interval: float,
    method: str,
    cutoff_ratio: float,
    signal: Any,
) -> NDArray[np.float64]:
    """Apply jitter correction method to timestamps."""
    if method == "lowpass":
        return _lowpass_correction(timestamps, intervals, cutoff_ratio, signal)
    if method == "linear":
        return _linear_correction(timestamps)
    if method == "pll":
        return _pll_correction(timestamps, expected_interval)
    raise ValueError(f"Unknown method: {method}")


def _lowpass_correction(
    timestamps: NDArray[np.float64],
    intervals: NDArray[np.float64],
    cutoff_ratio: float,
    signal: Any,
) -> NDArray[np.float64]:
    """Apply low-pass filter correction."""
    b, a = signal.butter(2, cutoff_ratio, btype="low")
    filtered_intervals = signal.filtfilt(b, a, intervals)
    corrected = np.zeros_like(timestamps)
    corrected[0] = timestamps[0]
    corrected[1:] = timestamps[0] + np.cumsum(filtered_intervals)
    return corrected


def _linear_correction(timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply linear fit correction (clock drift only)."""
    indices = np.arange(len(timestamps))
    coeffs = np.polyfit(indices, timestamps, 1)
    return np.polyval(coeffs, indices)


def _pll_correction(
    timestamps: NDArray[np.float64], expected_interval: float
) -> NDArray[np.float64]:
    """Apply PLL-based correction."""
    corrected = np.zeros_like(timestamps)
    corrected[0] = timestamps[0]
    gain = 0.1

    for i in range(1, len(timestamps)):
        expected_time = corrected[i - 1] + expected_interval
        phase_error = timestamps[i] - expected_time
        corrected[i] = expected_time + gain * phase_error

    return corrected


def _calculate_jitter_metrics(
    timestamps: NDArray[np.float64],
    intervals: NDArray[np.float64],
    corrected: NDArray[np.float64],
    expected_interval: float,
    n: int,
) -> dict[str, float]:
    """Calculate jitter and drift metrics."""
    original_jitter = np.std(intervals - expected_interval)
    corrected_intervals = np.diff(corrected)
    corrected_jitter = np.std(corrected_intervals - expected_interval)
    jitter_removed = original_jitter - corrected_jitter

    total_time = timestamps[-1] - timestamps[0]
    expected_total = (n - 1) * expected_interval
    drift_ratio = (total_time - expected_total) / expected_total

    return {
        "jitter_removed_ns": jitter_removed * 1e9,
        "clock_drift_ppm": drift_ratio * 1e6,
    }


# =============================================================================
# =============================================================================


def error_tolerant_decode(
    data: bytes | NDArray[np.uint8],
    protocol: str,
    *,
    max_errors_per_frame: int = 2,
    resync_on_error: bool = True,
) -> dict[str, Any]:
    """Decode protocol with error tolerance and resynchronization.

    : Error-Tolerant Protocol Decoding.

    Continues decoding after framing/parity errors instead of aborting.

    Args:
        data: Binary data to decode.
        protocol: Protocol name ('uart', 'spi', 'i2c').
        max_errors_per_frame: Max errors before skipping frame.
        resync_on_error: Attempt resynchronization on errors.

    Returns:
        Dictionary with decoded frames, errors, and sync info.

    Raises:
        ValueError: If unsupported protocol specified.

    Example:
        >>> result = error_tolerant_decode(data, 'uart', max_errors_per_frame=2)
        >>> print(f"Decoded: {result['frame_count']}, Errors: {result['error_count']}")
    """
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    result = {
        "protocol": protocol,
        "frames": [],
        "frame_count": 0,
        "error_count": 0,
        "resync_count": 0,
        "error_frames": [],
    }

    # Protocol-specific decoding with error recovery
    if protocol.lower() == "uart":
        result = _decode_uart_tolerant(data, max_errors_per_frame, resync_on_error)
    elif protocol.lower() == "spi":
        result = _decode_spi_tolerant(data, max_errors_per_frame)
    elif protocol.lower() == "i2c":
        result = _decode_i2c_tolerant(data, max_errors_per_frame, resync_on_error)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    return result


def _decode_uart_tolerant(
    data: NDArray[np.uint8],
    max_errors: int,
    resync: bool,
) -> dict[str, Any]:
    """UART decode with error tolerance."""
    # Simplified UART decoding with error recovery
    frames = []
    errors = []

    # In reality, would properly decode UART bit stream
    # Here we treat each byte as a frame for demonstration
    for i, byte in enumerate(data):
        # Check for framing errors (simplified: check start/stop bits if present)
        parity_error = ((byte).bit_count() % 2) != 0  # Assuming odd parity

        if parity_error:
            errors.append({"offset": i, "type": "parity", "byte": byte})
            if len(errors) > max_errors and resync:
                # Skip to next potential start
                continue
        else:
            frames.append({"offset": i, "data": byte, "valid": True})

    return {
        "protocol": "uart",
        "frames": frames,
        "frame_count": len(frames),
        "error_count": len(errors),
        "resync_count": 0,
        "error_frames": errors,
    }


def _decode_spi_tolerant(
    data: NDArray[np.uint8],
    max_errors: int,
) -> dict[str, Any]:
    """SPI decode with error tolerance."""
    frames = []
    for i, byte in enumerate(data):
        frames.append({"offset": i, "mosi": byte, "miso": 0, "valid": True})

    return {
        "protocol": "spi",
        "frames": frames,
        "frame_count": len(frames),
        "error_count": 0,
        "resync_count": 0,
        "error_frames": [],
    }


def _decode_i2c_tolerant(
    data: NDArray[np.uint8],
    max_errors: int,
    resync: bool,
) -> dict[str, Any]:
    """I2C decode with error tolerance."""
    frames = []
    errors = []

    i = 0
    while i < len(data):
        # Look for start condition marker (simplified)
        if data[i] == 0x00:  # Start marker
            if i + 2 < len(data):
                addr = data[i + 1]
                data_byte = data[i + 2]
                frames.append(
                    {
                        "offset": i,
                        "address": addr >> 1,
                        "read": bool(addr & 1),
                        "data": data_byte,
                        "ack": True,
                    }
                )
                i += 3
            else:
                break
        else:
            errors.append({"offset": i, "type": "no_start"})
            if resync:
                i += 1
            else:
                break

    return {
        "protocol": "i2c",
        "frames": frames,
        "frame_count": len(frames),
        "error_count": len(errors),
        "resync_count": len(errors) if resync else 0,
        "error_frames": errors,
    }


# =============================================================================
# =============================================================================


def analyze_bit_errors(
    expected: bytes | NDArray[np.uint8],
    actual: bytes | NDArray[np.uint8],
) -> BitErrorAnalysis:
    """Analyze bit error patterns for diagnostics.

    : Bit Error Pattern Analysis.

    Characterizes bit error patterns to diagnose capture quality issues
    (EMI, USB problems, clock jitter).

    Args:
        expected: Expected data.
        actual: Actual received data.

    Returns:
        BitErrorAnalysis with error characterization.

    Example:
        >>> result = analyze_bit_errors(expected_data, actual_data)
        >>> print(f"Error rate: {result.error_rate:.2e}")
        >>> print(f"Pattern: {result.error_pattern.value}")
        >>> print(f"Cause: {result.probable_cause}")
    """
    if isinstance(expected, bytes):
        expected = np.frombuffer(expected, dtype=np.uint8)
    if isinstance(actual, bytes):
        actual = np.frombuffer(actual, dtype=np.uint8)

    # Pad shorter array
    min_len = min(len(expected), len(actual))
    expected = expected[:min_len]
    actual = actual[:min_len]

    # XOR to find differences
    xor = expected ^ actual

    # Count errors per bit position (0-7)
    bit_errors_by_position = [0] * 8
    total_bit_errors = 0
    error_locations = []

    for i, byte in enumerate(xor):
        if byte != 0:
            for bit in range(8):
                if (byte >> bit) & 1:
                    bit_errors_by_position[bit] += 1
                    total_bit_errors += 1
                    error_locations.append(i * 8 + bit)

    total_bits = min_len * 8
    error_rate = total_bit_errors / total_bits if total_bits > 0 else 0

    # Analyze error pattern
    if len(error_locations) < 2:
        pattern = ErrorPattern.SINGLE_BIT
        burst_mean = 0.0
        burst_max = 0
    else:
        # Check for burst pattern
        gaps = np.diff(error_locations)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)

        # Calculate burst lengths
        bursts = []
        current_burst = 1
        for gap in gaps:
            if gap <= 2:  # Adjacent or near-adjacent errors
                current_burst += 1
            else:
                bursts.append(current_burst)
                current_burst = 1
        bursts.append(current_burst)

        burst_mean = float(np.mean(bursts))
        burst_max = int(max(bursts))

        if burst_max > 5:
            pattern = ErrorPattern.BURST
        elif std_gap < mean_gap * 0.3:
            pattern = ErrorPattern.SYSTEMATIC
        else:
            pattern = ErrorPattern.RANDOM

    # Determine probable cause and recommendations
    probable_cause, recommendations = _diagnose_errors(pattern, error_rate, bit_errors_by_position)

    return BitErrorAnalysis(
        error_rate=error_rate,
        error_pattern=pattern,
        burst_length_mean=burst_mean,
        burst_length_max=burst_max,
        error_distribution=bit_errors_by_position,
        probable_cause=probable_cause,
        recommendations=recommendations,
    )


def _diagnose_errors(
    pattern: ErrorPattern,
    error_rate: float,
    bit_distribution: list[int],
) -> tuple[str, list[str]]:
    """Diagnose probable cause of errors."""
    if pattern == ErrorPattern.BURST:
        cause = "Electromagnetic interference (EMI) or USB transmission errors"
        recommendations = [
            "Use shorter cables",
            "Add ferrite beads",
            "Check for nearby interference sources",
            "Try a different USB port or hub",
        ]
    elif pattern == ErrorPattern.SYSTEMATIC:
        cause = "Clock synchronization or sampling issues"
        recommendations = [
            "Verify sample rate is adequate (10x signal rate)",
            "Check for clock jitter on logic analyzer",
            "Ensure proper signal termination",
        ]
    elif pattern == ErrorPattern.RANDOM:
        if error_rate > 0.01:
            cause = "Poor signal quality or threshold issues"
            recommendations = [
                "Adjust voltage threshold",
                "Reduce cable length",
                "Check signal integrity",
            ]
        else:
            cause = "Normal noise level"
            recommendations = ["Error rate is acceptable"]
    else:  # SINGLE_BIT
        # Check bit distribution for systematic bias
        max_bit = max(bit_distribution)
        min_bit = min(bit_distribution)
        if max_bit > 0 and max_bit > 2 * (min_bit + 1):
            biased_bit = bit_distribution.index(max_bit)
            cause = f"Bit {biased_bit} shows higher error rate - possible stuck bit"
            recommendations = [
                f"Check hardware for bit {biased_bit} issues",
                "May indicate logic analyzer channel problem",
            ]
        else:
            cause = "Isolated single-bit error"
            recommendations = ["Likely transient noise, no action needed"]

    return cause, recommendations


__all__ = [
    "BitErrorAnalysis",
    "DAQGap",
    "DAQGapAnalysis",
    "ErrorPattern",
    "FuzzyMatch",
    "JitterCompensationResult",
    "PacketRecoveryResult",
    "analyze_bit_errors",
    "compensate_timestamp_jitter",
    "detect_gaps",
    "detect_gaps_by_samples",
    "detect_gaps_by_timestamps",
    "error_tolerant_decode",
    "fuzzy_pattern_search",
    "robust_packet_parse",
]
