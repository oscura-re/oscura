"""Fuzzy synchronization pattern search for corrupted data.


This module provides fuzzy pattern matching for finding sync words and markers
in noisy or corrupted logic analyzer captures, with configurable bit error
tolerance using Hamming distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class RecoveryStrategy(Enum):
    """Error recovery strategies for corrupted packets.

    Attributes:
        NEXT_SYNC: Skip to next sync word when corruption detected
        SKIP_BYTES: Skip fixed byte count and retry parsing
        HEURISTIC: Use statistical packet length model for recovery
    """

    NEXT_SYNC = "next_sync"
    SKIP_BYTES = "skip_bytes"
    HEURISTIC = "heuristic"


@dataclass
class SyncMatch:
    """Result from fuzzy sync pattern search.

    Attributes:
        index: Starting position of match in bits or bytes
        matched_value: The actual value that matched (may differ from pattern)
        hamming_distance: Number of bit errors in the match
        confidence: Match confidence (1.0 - bit_errors/pattern_length)
        pattern_length: Length of pattern in bits
    """

    index: int
    matched_value: int
    hamming_distance: int
    confidence: float
    pattern_length: int


@dataclass
class PacketParseResult:
    """Result from robust packet parsing.

    Attributes:
        packets: List of successfully parsed packet data
        valid: List of validity flags for each packet
        errors: List of error types ('length_corruption', 'sync_lost', None)
        error_positions: Byte positions where errors occurred
        recovery_count: Number of times recovery was triggered
    """

    packets: list[bytes]
    valid: list[bool]
    errors: list[str | None]
    error_positions: list[int]
    recovery_count: int


def hamming_distance(a: int, b: int, pattern_bits: int) -> int:
    """Calculate Hamming distance between two integers.

    Args:
        a: First integer
        b: Second integer
        pattern_bits: Number of bits to compare (8, 16, 32, or 64)

    Returns:
        Number of differing bits

    Examples:
        >>> hamming_distance(0b10101010, 0b10101011, 8)
        1
        >>> hamming_distance(0xAA55, 0xAA54, 16)
        1
    """
    # XOR gives 1s where bits differ
    diff = a ^ b
    # Mask to pattern length
    mask = (1 << pattern_bits) - 1
    diff &= mask
    # Count set bits (population count)
    return (diff).bit_count()


def fuzzy_sync_search(
    data: NDArray[np.uint8],
    pattern: int,
    *,
    pattern_bits: Literal[8, 16, 32, 64] = 8,
    max_errors: int = 2,
    min_confidence: float = 0.85,
) -> list[SyncMatch]:
    """Find sync patterns with bit error tolerance using Hamming distance.

    : Searches for sync words even with bit errors,
    essential for recovering corrupted logic analyzer captures.

    Performance targets (DAQ-001):
    - ≥10 MB/s for max_errors=2
    - ≥5 MB/s for max_errors=4
    - ≥1 MB/s for max_errors=8

    Confidence scoring (DAQ-001):
    - ≥0.95 (0-1 bit errors): Highly reliable
    - 0.85-0.95 (2-4 bit errors): Reliable
    - <0.85 (>4 bit errors): Verify manually

    Args:
        data: Input byte array to search
        pattern: Sync pattern to find (e.g., 0xAA55F0F0 for 32-bit)
        pattern_bits: Pattern length in bits (8, 16, 32, or 64)
        max_errors: Maximum tolerable bit errors (0-8)
        min_confidence: Minimum confidence threshold (0.0-1.0)

    Returns:
        List of SyncMatch objects with position, matched value, distance,
        and confidence score

    Raises:
        ValueError: If pattern_bits not in [8, 16, 32, 64]
        ValueError: If max_errors < 0 or > 8
        ValueError: If min_confidence not in [0.0, 1.0]

    Examples:
        >>> data = np.array([0xAA, 0x55, 0xF0, 0xF0, 0xFF], dtype=np.uint8)
        >>> # Find exact match
        >>> matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16)
        >>> print(matches[0].confidence)
        1.0

        >>> # Find with 1 bit error (0xAA54 instead of 0xAA55)
        >>> data = np.array([0xAA, 0x54, 0x00], dtype=np.uint8)
        >>> matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=2)
        >>> print(matches[0].hamming_distance)
        1

    References:
        DAQ-001: Fuzzy Bit Pattern Search with Hamming Distance Tolerance
    """
    _validate_fuzzy_search_params(pattern_bits, max_errors, min_confidence)

    pattern_bytes = pattern_bits // 8
    if len(data) < pattern_bytes:
        return []

    matches: list[SyncMatch] = []

    for i in range(len(data) - pattern_bytes + 1):
        window = data[i : i + pattern_bytes]
        value = _bytes_to_int(window, pattern_bytes)
        dist = hamming_distance(value, pattern, pattern_bits)

        if dist <= max_errors:
            confidence = 1.0 - (dist / pattern_bits)
            if confidence >= min_confidence:
                matches.append(
                    SyncMatch(
                        index=i,
                        matched_value=value,
                        hamming_distance=dist,
                        confidence=confidence,
                        pattern_length=pattern_bits,
                    )
                )

    return matches


def _validate_fuzzy_search_params(
    pattern_bits: int, max_errors: int, min_confidence: float
) -> None:
    """Validate fuzzy search parameters.

    Args:
        pattern_bits: Pattern length in bits.
        max_errors: Maximum bit errors.
        min_confidence: Minimum confidence threshold.

    Raises:
        ValueError: If parameters invalid.
    """
    if pattern_bits not in (8, 16, 32, 64):
        raise ValueError("pattern_bits must be 8, 16, 32, or 64")
    if max_errors < 0 or max_errors > 8:
        raise ValueError("max_errors must be in range [0, 8]")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in range [0.0, 1.0]")


def _bytes_to_int(window: NDArray[np.uint8], pattern_bytes: int) -> int:
    """Convert byte window to integer (big-endian).

    Args:
        window: Byte array window.
        pattern_bytes: Number of bytes (1, 2, 4, or 8).

    Returns:
        Integer representation.
    """
    if pattern_bytes == 1:
        return int(window[0])
    elif pattern_bytes == 2:
        return (int(window[0]) << 8) | int(window[1])
    elif pattern_bytes == 4:
        return (
            (int(window[0]) << 24) | (int(window[1]) << 16) | (int(window[2]) << 8) | int(window[3])
        )
    else:  # 8 bytes
        value = 0
        for j in range(8):
            value = (value << 8) | int(window[j])
        return value


def _validate_packet_parse_inputs(
    length_size: int,
    recovery_strategy: RecoveryStrategy,
    sync_pattern: int | None,
) -> None:
    """Validate packet parsing input parameters.

    Args:
        length_size: Length field size in bytes
        recovery_strategy: Recovery strategy to use
        sync_pattern: Optional sync pattern

    Raises:
        ValueError: If length_size not 1 or 2
        ValueError: If NEXT_SYNC strategy without sync_pattern
    """
    if length_size not in (1, 2):
        raise ValueError("length_size must be 1 or 2")

    if recovery_strategy == RecoveryStrategy.NEXT_SYNC and sync_pattern is None:
        raise ValueError("NEXT_SYNC strategy requires sync_pattern")


def _has_sufficient_data_for_header(
    pos: int,
    data_length: int,
    length_offset: int,
    length_size: int,
) -> bool:
    """Check if sufficient data remains for packet header.

    Args:
        pos: Current position in data
        data_length: Total data length
        length_offset: Offset to length field
        length_size: Size of length field

    Returns:
        True if enough data for header, False otherwise
    """
    return pos + length_offset + length_size <= data_length


def _extract_length_field(
    data: NDArray[np.uint8],
    pos: int,
    length_offset: int,
    length_size: int,
) -> int:
    """Extract packet length field from data.

    Args:
        data: Input byte array
        pos: Current position
        length_offset: Offset to length field
        length_size: Size of length field (1 or 2 bytes)

    Returns:
        Extracted packet length value
    """
    length_pos = pos + length_offset
    if length_size == 1:
        return int(data[length_pos])
    else:  # 2 bytes
        return (int(data[length_pos]) << 8) | int(data[length_pos + 1])


def _validate_length_field(
    pkt_length: int,
    length_size: int,
    min_packet_size: int,
    max_packet_size: int,
    packet_lengths: list[int],
) -> tuple[bool, str | None]:
    """Validate packet length field for corruption.

    Args:
        pkt_length: Extracted packet length
        length_size: Size of length field (1 or 2)
        min_packet_size: Minimum valid packet size
        max_packet_size: Maximum valid packet size
        packet_lengths: History of valid packet lengths

    Returns:
        Tuple of (is_valid, error_type)
    """
    # Check for obviously corrupted lengths
    if (
        pkt_length == 0
        or pkt_length > max_packet_size
        or (length_size == 2 and (pkt_length & 0xFF00) == 0xFF00)
        or pkt_length < min_packet_size
    ):
        return False, "length_corruption"

    # Check suspiciously large length (heuristic)
    if len(packet_lengths) >= 10:
        p90 = np.percentile(packet_lengths, 90)
        if pkt_length > p90 * 2:
            return False, "length_corruption"

    return True, None


def _try_extract_packet(
    data: NDArray[np.uint8],
    pos: int,
    pkt_length: int,
    packets: list[bytes],
    valid: list[bool],
    errors: list[str | None],
    error_positions: list[int],
    packet_lengths: list[int],
) -> tuple[int | None, bool]:
    """Try to extract complete packet from current position.

    Args:
        data: Input byte array
        pos: Current position
        pkt_length: Expected packet length
        packets: List to append packet to
        valid: List to append validity flag
        errors: List to append error info
        error_positions: List to append error positions
        packet_lengths: List to append packet length

    Returns:
        Tuple of (new_position, continue_parsing)
        - new_position: Next position to parse (None if truncated)
        - continue_parsing: Whether to continue parsing
    """
    packet_end = pos + pkt_length

    if packet_end <= len(data):
        # Successfully extract packet
        packet_data = bytes(data[pos:packet_end])
        packets.append(packet_data)
        valid.append(True)
        errors.append(None)
        packet_lengths.append(pkt_length)
        return packet_end, True
    else:
        # Packet extends beyond data
        errors.append("truncated")
        error_positions.append(pos)
        return None, False


def _apply_recovery_strategy(
    recovery_strategy: RecoveryStrategy,
    pos: int,
    data: NDArray[np.uint8],
    sync_pattern: int | None,
    sync_bits: Literal[8, 16, 32, 64],
    skip_bytes: int,
    packet_lengths: list[int],
    min_packet_size: int,
    errors: list[str | None],
    error_type: str | None,
) -> tuple[int | None, bool]:
    """Apply error recovery strategy.

    Args:
        recovery_strategy: Strategy to apply
        pos: Current position
        data: Input byte array
        sync_pattern: Sync pattern for NEXT_SYNC strategy
        sync_bits: Sync pattern bit length
        skip_bytes: Bytes to skip for SKIP_BYTES strategy
        packet_lengths: History for HEURISTIC strategy
        min_packet_size: Minimum packet size
        errors: List to append error info
        error_type: Error type to append

    Returns:
        Tuple of (new_position, continue_parsing)
    """
    if recovery_strategy == RecoveryStrategy.NEXT_SYNC:
        # Search for next sync word
        assert sync_pattern is not None
        search_start = pos + 1
        search_data = data[search_start:]

        if len(search_data) >= sync_bits // 8:
            matches = fuzzy_sync_search(
                search_data,
                sync_pattern,
                pattern_bits=sync_bits,
                max_errors=0,
            )

            if matches:
                # Found sync, jump to it
                errors.append("sync_lost")
                return search_start + matches[0].index, True
            else:
                # No more syncs found
                return None, False
        else:
            return None, False

    elif recovery_strategy == RecoveryStrategy.SKIP_BYTES:
        # Skip fixed bytes and retry
        errors.append(error_type)
        return pos + skip_bytes, True

    elif recovery_strategy == RecoveryStrategy.HEURISTIC:
        # Use median packet length as guess
        if packet_lengths:
            guess_length = int(np.median(packet_lengths))
            new_pos = pos + guess_length
        else:
            # No history, skip minimal amount
            new_pos = pos + min_packet_size
        errors.append(error_type)
        return new_pos, True

    raise RuntimeError("Unreachable: must either handle error or return None")


def _initialize_parse_state() -> tuple[
    list[bytes], list[bool], list[str | None], list[int], list[int]
]:
    """Initialize packet parsing state containers.

    Returns:
        Tuple of (packets, valid, errors, error_positions, packet_lengths).
    """
    return [], [], [], [], []


def _handle_packet_extraction(
    data: NDArray[np.uint8],
    pos: int,
    pkt_length: int,
    is_valid_length: bool,
    error_type: str | None,
    packets: list[bytes],
    valid: list[bool],
    errors: list[str | None],
    error_positions: list[int],
    packet_lengths: list[int],
    recovery_strategy: RecoveryStrategy,
    sync_pattern: int | None,
    sync_bits: Literal[8, 16, 32, 64],
    skip_bytes: int,
    min_packet_size: int,
) -> tuple[int | None, bool, int]:
    """Handle packet extraction or error recovery.

    Args:
        data: Input byte array.
        pos: Current position.
        pkt_length: Packet length.
        is_valid_length: Whether length is valid.
        error_type: Error type if invalid.
        packets: Packets list.
        valid: Valid flags list.
        errors: Errors list.
        error_positions: Error positions list.
        packet_lengths: Packet lengths list.
        recovery_strategy: Recovery strategy.
        sync_pattern: Sync pattern.
        sync_bits: Sync pattern bits.
        skip_bytes: Bytes to skip.
        min_packet_size: Minimum packet size.

    Returns:
        Tuple of (new_position, should_continue, recovery_increment).
    """
    if is_valid_length:
        new_pos, should_continue = _try_extract_packet(
            data, pos, pkt_length, packets, valid, errors, error_positions, packet_lengths
        )
        return new_pos, should_continue, 0
    else:
        error_positions.append(pos)
        new_pos, should_continue = _apply_recovery_strategy(
            recovery_strategy,
            pos,
            data,
            sync_pattern,
            sync_bits,
            skip_bytes,
            packet_lengths,
            min_packet_size,
            errors,
            error_type,
        )
        return new_pos, should_continue, 1


def parse_variable_length_packets(
    data: NDArray[np.uint8],
    *,
    sync_pattern: int | None = None,
    sync_bits: Literal[8, 16, 32, 64] = 16,
    length_offset: int = 2,
    length_size: int = 2,
    min_packet_size: int = 4,
    max_packet_size: int = 1024,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.NEXT_SYNC,
    skip_bytes: int = 1,
) -> PacketParseResult:
    """Parse variable-length packets with error recovery.

    Robust parsing that continues after corruption, falling back to sync word
    search when length fields are corrupted. Error detection: length=0,
    length>max_packet_size, or length&0xFF00=0xFF00. Recovery strategies:
    next_sync (requires sync_pattern), skip_bytes, or heuristic.

    Args:
        data: Input byte array containing packets
        sync_pattern: Optional sync word to search for on errors
        sync_bits: Sync pattern length in bits
        length_offset: Byte offset to length field from start
        length_size: Length field size in bytes (1 or 2)
        min_packet_size: Minimum valid packet size in bytes
        max_packet_size: Maximum valid packet size in bytes
        recovery_strategy: Strategy to use when corruption detected
        skip_bytes: Number of bytes to skip for SKIP_BYTES strategy

    Returns:
        PacketParseResult with parsed packets and error information

    Raises:
        ValueError: If length_size not 1 or 2
        ValueError: If recovery_strategy is NEXT_SYNC without sync_pattern

    Examples:
        >>> # Simple TLV parsing with sync word
        >>> data = np.array([0xAA, 0x55, 0x00, 0x04, 0x01, 0x02], dtype=np.uint8)
        >>> result = parse_variable_length_packets(
        ...     data, sync_pattern=0xAA55, length_offset=2
        ... )
        >>> len(result.packets)
        1

    References:
        DAQ-002: Robust Variable-Length Packet Parsing with Error Recovery
    """
    _validate_packet_parse_inputs(length_size, recovery_strategy, sync_pattern)
    packets, valid, errors, error_positions, packet_lengths = _initialize_parse_state()
    recovery_count = 0
    pos = 0

    while pos < len(data):
        if not _has_sufficient_data_for_header(pos, len(data), length_offset, length_size):
            break

        pkt_length = _extract_length_field(data, pos, length_offset, length_size)
        is_valid_length, error_type = _validate_length_field(
            pkt_length, length_size, min_packet_size, max_packet_size, packet_lengths
        )

        new_pos, should_continue, recovery_inc = _handle_packet_extraction(
            data,
            pos,
            pkt_length,
            is_valid_length,
            error_type,
            packets,
            valid,
            errors,
            error_positions,
            packet_lengths,
            recovery_strategy,
            sync_pattern,
            sync_bits,
            skip_bytes,
            min_packet_size,
        )

        recovery_count += recovery_inc
        if not should_continue:
            break
        assert new_pos is not None
        pos = new_pos

    return PacketParseResult(
        packets=packets,
        valid=valid,
        errors=errors,
        error_positions=error_positions,
        recovery_count=recovery_count,
    )
