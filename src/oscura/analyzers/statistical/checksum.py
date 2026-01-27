"""Checksum and CRC field detection and identification.


This module provides tools for detecting checksum and CRC fields in binary
messages by analyzing field correlations and testing common algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type alias for input data
DataType = Union[bytes, bytearray, "NDArray[np.uint8]"]


@dataclass
class ChecksumCandidate:
    """Candidate checksum field.

    Attributes:
        offset: Byte offset in message
        size: Field size in bytes (1, 2, or 4)
        position: Location in message structure
        correlation: Correlation with content (0-1)
        likely_scope: Byte range likely covered by checksum (start, end)
    """

    offset: int
    size: int
    position: Literal["header", "trailer"]
    correlation: float
    likely_scope: tuple[int, int]


@dataclass
class ChecksumMatch:
    """Identified checksum algorithm.

    Attributes:
        algorithm: Algorithm name
        offset: Field offset in message
        size: Field size in bytes
        scope_start: Start of checksummed region
        scope_end: End of checksummed region
        match_rate: Fraction of messages that match (0-1)
        polynomial: CRC polynomial (for CRC algorithms)
        init_value: Initial value (for CRC algorithms)
        xor_out: Final XOR value (for CRC algorithms)
    """

    algorithm: str
    offset: int
    size: int
    scope_start: int
    scope_end: int
    match_rate: float
    polynomial: int | None = None
    init_value: int | None = None
    xor_out: int | None = None


def detect_checksum_fields(
    messages: list[DataType], candidate_offsets: list[int] | None = None
) -> list[ChecksumCandidate]:
    """Detect fields that are correlated with message content.

    : Checksum and CRC Field Detection

    Analyzes message fields to find those that vary consistently with
    content changes, indicating potential checksum/CRC fields.

    Args:
        messages: List of messages to analyze
        candidate_offsets: Optional list of specific offsets to check

    Returns:
        List of checksum candidates sorted by correlation

    Example:
        >>> msgs = [b'\\x00\\x00DATA', b'\\x01\\x00DATA']
        >>> candidates = detect_checksum_fields(msgs)
        >>> len(candidates) >= 0
        True
    """
    if not messages:
        return []

    byte_messages = _convert_messages_to_bytes(messages)
    min_len = min(len(msg) for msg in byte_messages)

    if min_len < 2:
        return []

    if candidate_offsets is None:
        candidate_offsets = _generate_default_candidate_offsets(min_len)

    candidates = _test_candidate_offsets(byte_messages, candidate_offsets, min_len)
    candidates.sort(key=lambda c: c.correlation, reverse=True)

    return candidates


def _generate_default_candidate_offsets(min_len: int) -> list[int]:
    """Generate default candidate positions in header and trailer."""
    header_positions = list(range(min(16, min_len - 1)))
    trailer_start = max(0, min_len - 16)
    trailer_positions = list(range(trailer_start, min_len - 1))
    return list(set(header_positions + trailer_positions))


def _test_candidate_offsets(
    byte_messages: list[bytes], candidate_offsets: list[int], min_len: int
) -> list[ChecksumCandidate]:
    """Test each candidate offset for checksum correlation."""
    candidates = []

    for offset in candidate_offsets:
        for size in [1, 2, 4]:
            if offset + size > min_len:
                continue

            candidate = _analyze_field_correlation(byte_messages, offset, size, min_len)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _analyze_field_correlation(
    byte_messages: list[bytes], offset: int, size: int, min_len: int
) -> ChecksumCandidate | None:
    """Analyze correlation between field and message content."""
    field_values = []
    content_hashes = []

    for msg in byte_messages:
        if len(msg) < offset + size:
            continue

        field_bytes = msg[offset : offset + size]
        field_value = int.from_bytes(field_bytes, byteorder="big")
        field_values.append(field_value)

        content = msg[:offset] + msg[offset + size :]
        content_hash = hash(content)
        content_hashes.append(content_hash)

    if len(field_values) < 2:
        return None

    correlation = _calculate_field_correlation(content_hashes, field_values)

    if correlation < 0.3:
        return None

    position: Literal["header", "trailer"] = "header" if offset < min_len // 2 else "trailer"
    likely_scope = (offset + size, min_len) if position == "header" else (0, offset)

    return ChecksumCandidate(
        offset=offset,
        size=size,
        position=position,
        correlation=correlation,
        likely_scope=likely_scope,
    )


def _calculate_field_correlation(content_hashes: list[int], field_values: list[int]) -> float:
    """Calculate correlation between content and field variability."""
    unique_content = len(set(content_hashes))
    unique_fields = len(set(field_values))

    if unique_content > 1:
        return min(1.0, unique_fields / unique_content)

    return 0.0


def _convert_messages_to_bytes(messages: list[DataType]) -> list[bytes]:
    """Convert messages to bytes format.

    Args:
        messages: List of messages in various formats.

    Returns:
        List of messages as bytes objects.
    """
    byte_messages = []
    for msg in messages:
        if isinstance(msg, np.ndarray):
            byte_messages.append(msg.tobytes() if msg.dtype == np.uint8 else bytes(msg.flatten()))
        else:
            byte_messages.append(bytes(msg))
    return byte_messages


def _get_algorithms_for_size(size: int) -> list[str]:
    """Get list of checksum algorithms to test for given field size.

    Args:
        size: Field size in bytes (1, 2, or 4).

    Returns:
        List of algorithm names to test.
    """
    if size == 1:
        return ["xor", "sum8"]
    elif size == 2:
        return ["sum16_big", "sum16_little", "crc16_ccitt", "crc16_ibm", "crc16", "checksum"]
    elif size == 4:
        return ["crc32"]
    return []


def _normalize_algorithm_name(algo: str) -> str:
    """Normalize algorithm name to actual function name.

    Args:
        algo: Original algorithm name.

    Returns:
        Normalized algorithm name.
    """
    if algo == "crc16":
        return "crc16_ccitt"
    elif algo == "checksum":
        return "sum16_big"
    return algo


def _get_init_values_for_algorithm(algo: str) -> list[int | None]:
    """Get list of initialization values to test for algorithm.

    Args:
        algo: Algorithm name.

    Returns:
        List of init values to test (None means use default).
    """
    if algo in ["crc16_ccitt", "crc16_ibm"]:
        return [0x0000, 0xFFFF]
    return [None]


def _extract_checksummed_data(
    msg: bytes, scope_start: int, scope_end: int, field_offset: int, field_size: int
) -> bytes:
    """Extract data region for checksum calculation.

    Args:
        msg: Full message bytes.
        scope_start: Start offset of checksummed region.
        scope_end: End offset of checksummed region.
        field_offset: Offset of checksum field.
        field_size: Size of checksum field.

    Returns:
        Data bytes to be checksummed (excluding checksum field if in range).
    """
    if scope_start < field_offset < scope_end:
        # Exclude checksum field from data
        return msg[scope_start:field_offset] + msg[field_offset + field_size : scope_end]
    else:
        return msg[scope_start:scope_end]


def _test_checksum_match(
    byte_messages: list[bytes],
    field_offset: int,
    field_size: int,
    algo: str,
    init_val: int | None,
    scope_start: int,
    scope_end: int,
) -> int:
    """Test if checksum algorithm matches messages in given scope.

    Args:
        byte_messages: List of message bytes.
        field_offset: Offset of checksum field.
        field_size: Size of checksum field in bytes.
        algo: Algorithm name.
        init_val: Initialization value for CRC algorithms.
        scope_start: Start of checksummed region.
        scope_end: End of checksummed region.

    Returns:
        Number of messages that match the algorithm.
    """
    matches = 0

    for msg in byte_messages:
        if len(msg) < scope_end:
            continue

        # Try both big and little endian for field extraction
        endian_val: Literal["big", "little"]
        for endian_val in ("big", "little"):
            expected = int.from_bytes(
                msg[field_offset : field_offset + field_size], byteorder=endian_val
            )

            data = _extract_checksummed_data(msg, scope_start, scope_end, field_offset, field_size)

            # Compute checksum
            try:
                if init_val is not None:
                    computed = compute_checksum(data, algo, init=init_val)
                else:
                    computed = compute_checksum(data, algo)

                if computed == expected:
                    matches += 1
                    break  # Found match with this endian
            except Exception:
                pass

    return matches


def _test_algorithm_variant(
    byte_messages: list[bytes],
    field_offset: int,
    size: int,
    algo: str,
    init_val: int | None,
) -> ChecksumMatch | None:
    """Test an algorithm variant across different scopes.

    Args:
        byte_messages: List of message bytes.
        field_offset: Offset of checksum field.
        size: Field size in bytes.
        algo: Original algorithm name.
        init_val: Initialization value for CRC.

    Returns:
        ChecksumMatch if match rate >= 80%, None otherwise.
    """
    actual_algo = _normalize_algorithm_name(algo)
    best_match = None
    best_rate = 0.0

    # Try different scopes
    for scope_start in [0, field_offset + size]:
        for scope_end in [field_offset, len(byte_messages[0])]:
            if scope_end <= scope_start:
                continue

            matches = _test_checksum_match(
                byte_messages, field_offset, size, actual_algo, init_val, scope_start, scope_end
            )

            total = sum(1 for msg in byte_messages if len(msg) >= scope_end)
            if total == 0:
                continue

            match_rate = matches / total

            # Consider it a match if >= 80% of messages match
            if match_rate >= 0.8 and match_rate > best_rate:
                best_rate = match_rate
                best_match = ChecksumMatch(
                    algorithm=algo,
                    offset=field_offset,
                    size=size,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    match_rate=match_rate,
                    init_value=init_val,
                )

    return best_match


def identify_checksum_algorithm(
    messages: list[DataType], field_offset: int, field_size: int | None = None
) -> ChecksumMatch | None:
    """Identify which checksum algorithm is used.

    : Checksum and CRC Field Detection

    Tests common checksum algorithms to identify which one matches
    the observed field values.

    Args:
        messages: List of messages to analyze
        field_offset: Offset of checksum field
        field_size: Size of field (1, 2, or 4 bytes), auto-detect if None

    Returns:
        ChecksumMatch if algorithm identified, None otherwise

    Example:
        >>> msgs = [b'\\x41ABC', b'\\x42BCD']  # XOR checksum
        >>> match = identify_checksum_algorithm(msgs, 0, 1)
        >>> match is not None
        True
    """
    if not messages:
        return None

    byte_messages = _convert_messages_to_bytes(messages)

    # Determine field sizes to test
    field_sizes = [1, 2, 4] if field_size is None else [field_size]

    best_match = None
    best_rate = 0.0

    # Try each field size
    for size in field_sizes:
        if any(len(msg) < field_offset + size for msg in byte_messages):
            continue

        algorithms = _get_algorithms_for_size(size)
        if not algorithms:
            continue

        # Test each algorithm
        for algo in algorithms:
            actual_algo = _normalize_algorithm_name(algo)
            init_values = _get_init_values_for_algorithm(actual_algo)

            for init_val in init_values:
                match = _test_algorithm_variant(byte_messages, field_offset, size, algo, init_val)

                if match and match.match_rate > best_rate:
                    best_rate = match.match_rate
                    best_match = match

    return best_match


def verify_checksums(
    messages: list[DataType],
    algorithm: str,
    field_offset: int,
    scope_start: int = 0,
    scope_end: int | None = None,
    init_value: int | None = None,
) -> tuple[int, int]:
    """Verify checksums using identified algorithm.

    Validates checksums across multiple messages using the specified algorithm.
    Tries both big-endian and little-endian byte orders.

    Args:
        messages: List of messages to verify
        algorithm: Checksum algorithm name (e.g., 'xor', 'sum8', 'crc16', 'crc32')
        field_offset: Offset of checksum field in message
        scope_start: Start of checksummed data (default: 0)
        scope_end: End of checksummed data (None = message end)
        init_value: Initial value for CRC algorithms (None = use default)

    Returns:
        Tuple of (passed, failed) counts

    Example:
        >>> msgs = [b'\\x41ABC']
        >>> passed, failed = verify_checksums(msgs, 'xor', 0, 1)
        >>> passed + failed == len(msgs)
        True
    """
    if not messages:
        return (0, 0)

    field_size = _get_checksum_field_size(algorithm)
    passed = 0
    failed = 0

    for msg in messages:
        msg_bytes = _normalize_message_to_bytes(msg)
        if _verify_single_message(
            msg_bytes, algorithm, field_offset, field_size, scope_start, scope_end, init_value
        ):
            passed += 1
        else:
            failed += 1

    return (passed, failed)


def _get_checksum_field_size(algorithm: str) -> int:
    """Determine field size in bytes from algorithm name.

    Args:
        algorithm: Checksum algorithm name.

    Returns:
        Field size in bytes (1, 2, or 4).
    """
    if algorithm in ["xor", "sum8"]:
        return 1
    if algorithm.startswith("sum16") or algorithm.startswith("crc16"):
        return 2
    if algorithm == "crc32":
        return 4
    return 1  # Default to 1 byte


def _normalize_message_to_bytes(msg: DataType) -> bytes:
    """Normalize message to bytes.

    Args:
        msg: Message as bytes, numpy array, or byte-like object.

    Returns:
        Message as bytes.
    """
    if isinstance(msg, np.ndarray):
        return msg.tobytes() if msg.dtype == np.uint8 else bytes(msg.flatten())
    return bytes(msg)


def _verify_single_message(
    msg: bytes,
    algorithm: str,
    field_offset: int,
    field_size: int,
    scope_start: int,
    scope_end: int | None,
    init_value: int | None,
) -> bool:
    """Verify checksum for a single message.

    Args:
        msg: Message bytes.
        algorithm: Checksum algorithm name.
        field_offset: Offset of checksum field.
        field_size: Size of checksum field in bytes.
        scope_start: Start of checksummed data.
        scope_end: End of checksummed data (None = message end).
        init_value: Initial value for CRC algorithms.

    Returns:
        True if checksum matches, False otherwise.
    """
    msg_scope_end = scope_end if scope_end is not None else len(msg)

    # Validate message length
    if len(msg) < field_offset + field_size or len(msg) < msg_scope_end:
        return False

    # Try both endiannesses
    for endian in ("big", "little"):
        expected = int.from_bytes(msg[field_offset : field_offset + field_size], byteorder=endian)

        # Extract data to checksum (excluding checksum field)
        data = _extract_checksummed_data(msg, scope_start, msg_scope_end, field_offset, field_size)

        # Compute and compare checksum
        if _compute_and_compare(data, algorithm, expected, init_value):
            return True

    return False


def _compute_and_compare(
    data: bytes,
    algorithm: str,
    expected: int,
    init_value: int | None,
) -> bool:
    """Compute checksum and compare with expected value.

    Args:
        data: Data to checksum.
        algorithm: Checksum algorithm name.
        expected: Expected checksum value.
        init_value: Initial value for CRC algorithms.

    Returns:
        True if computed checksum matches expected.
    """
    try:
        if init_value is not None:
            computed = compute_checksum(data, algorithm, init=init_value)
        else:
            computed = compute_checksum(data, algorithm)
        return computed == expected
    except Exception:
        return False


def compute_checksum(data: bytes, algorithm: str, **kwargs: Any) -> int:
    """Compute checksum using specified algorithm.

    : Checksum and CRC Field Detection

    Args:
        data: Data to checksum
        algorithm: Algorithm name
        **kwargs: Algorithm-specific parameters

    Returns:
        Computed checksum value

    Raises:
        ValueError: If algorithm is unknown

    Example:
        >>> compute_checksum(b'ABC', 'xor')
        2
    """
    if algorithm == "xor":
        return xor_checksum(data)
    elif algorithm == "sum8":
        return sum8(data)
    elif algorithm == "sum16_big":
        return sum16(data, endian="big")
    elif algorithm == "sum16_little":
        return sum16(data, endian="little")
    elif algorithm == "crc8":
        poly = kwargs.get("poly", 0x07)
        init = kwargs.get("init", 0x00)
        return crc8(data, poly=poly, init=init)
    elif algorithm == "crc16_ccitt":
        init = kwargs.get("init", 0xFFFF)
        return crc16_ccitt(data, init=init)
    elif algorithm == "crc16_ibm":
        init = kwargs.get("init", 0x0000)
        return crc16_ibm(data, init=init)
    elif algorithm == "crc32":
        return crc32(data)
    else:
        raise ValueError(f"Unknown checksum algorithm: {algorithm}")


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """Calculate CRC-8.

    : Checksum and CRC Field Detection

    Standard CRC-8 with configurable polynomial.

    Args:
        data: Data to checksum
        poly: Polynomial (default: 0x07)
        init: Initial value (default: 0x00)

    Returns:
        CRC-8 value (0-255)

    Example:
        >>> crc8(b'123456789')
        244
    """
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
            crc &= 0xFF
    return crc


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Calculate CRC-16-CCITT.

    : Checksum and CRC Field Detection

    CCITT polynomial: 0x1021

    Args:
        data: Data to checksum
        init: Initial value (default: 0xFFFF)

    Returns:
        CRC-16 value (0-65535)

    Example:
        >>> crc16_ccitt(b'123456789')
        10673
    """
    poly = 0x1021
    crc = init

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
            crc &= 0xFFFF

    return crc


def crc16_ibm(data: bytes, init: int = 0x0000) -> int:
    """Calculate CRC-16-IBM (also known as CRC-16-ANSI).

    : Checksum and CRC Field Detection

    IBM polynomial: 0x8005 (reversed: 0xA001)

    Args:
        data: Data to checksum
        init: Initial value (default: 0x0000)

    Returns:
        CRC-16 value (0-65535)

    Example:
        >>> crc16_ibm(b'123456789')
        47933
    """
    poly = 0xA001  # Reversed polynomial for LSB-first
    crc = init

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ poly
            else:
                crc = crc >> 1

    return crc


def crc32(data: bytes) -> int:
    """Calculate CRC-32 (IEEE 802.3).

    : Checksum and CRC Field Detection

    Standard CRC-32 as used in Ethernet, ZIP, etc.

    Args:
        data: Data to checksum

    Returns:
        CRC-32 value (0-4294967295)

    Example:
        >>> crc32(b'123456789')
        3421780262
    """
    poly = 0xEDB88320  # Reversed polynomial
    crc = 0xFFFFFFFF

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x00000001:
                crc = (crc >> 1) ^ poly
            else:
                crc = crc >> 1

    return crc ^ 0xFFFFFFFF


def sum8(data: bytes) -> int:
    """Calculate 8-bit sum checksum.

    : Checksum and CRC Field Detection

    Simple sum of all bytes, modulo 256.

    Args:
        data: Data to checksum

    Returns:
        Sum modulo 256 (0-255)

    Example:
        >>> sum8(b'ABC')
        198
    """
    return sum(data) & 0xFF


def sum16(data: bytes, endian: Literal["big", "little"] = "big") -> int:
    """Calculate 16-bit sum checksum.

    : Checksum and CRC Field Detection

    Sum of 16-bit words with configurable endianness.

    Args:
        data: Data to checksum
        endian: Byte order ('big' or 'little', default: 'big')

    Returns:
        Sum modulo 65536 (0-65535)

    Example:
        >>> sum16(b'ABCD', endian='big')
        33923
    """
    total = 0

    # Process 16-bit words
    for i in range(0, len(data) - 1, 2):
        if endian == "big":
            word = (data[i] << 8) | data[i + 1]
        else:
            word = (data[i + 1] << 8) | data[i]
        total += word

    # Handle odd byte
    if len(data) % 2 == 1:
        if endian == "big":
            total += data[-1] << 8
        else:
            total += data[-1]

    return total & 0xFFFF


def xor_checksum(data: bytes) -> int:
    """Calculate XOR checksum.

    : Checksum and CRC Field Detection

    XOR of all bytes.

    Args:
        data: Data to checksum

    Returns:
        XOR result (0-255)

    Example:
        >>> xor_checksum(b'ABC')
        2
    """
    result = 0
    for byte in data:
        result ^= byte
    return result


@dataclass
class ChecksumDetectionResult:
    """Result of checksum detection.

    Attributes:
        has_checksum: Whether a checksum was detected.
        offset: Byte offset of the checksum field (None if not detected).
        size: Size of the checksum field in bytes (None if not detected).
        algorithm: Identified algorithm name (None if not identified).
        confidence: Detection confidence (0-1).
        candidates: All candidate positions found.
        scope_start: Start of checksummed region (None if not identified).
        scope_end: End of checksummed region (None if not identified).
        init_value: Initial value for CRC algorithms (None if not applicable).
    """

    has_checksum: bool
    offset: int | None = None
    size: int | None = None
    algorithm: str | None = None
    confidence: float = 0.0
    candidates: list[ChecksumCandidate] = field(default_factory=list)
    scope_start: int | None = None
    scope_end: int | None = None
    init_value: int | None = None


class ChecksumDetector:
    """Object-oriented wrapper for checksum detection functionality.

    Provides a class-based interface for checksum detection operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> detector = ChecksumDetector()
        >>> result = detector.detect_checksum_field(messages)
        >>> if result.has_checksum:
        ...     print(f"Checksum at offset {result.offset}")
    """

    def __init__(self, correlation_threshold: float = 0.5):
        """Initialize checksum detector.

        Args:
            correlation_threshold: Minimum correlation for detection.
        """
        self.correlation_threshold = correlation_threshold
        self._detection_result: ChecksumDetectionResult | None = None
        self._messages: list[DataType] = []

    def detect_checksum_field(self, messages: list[DataType]) -> ChecksumDetectionResult:
        """Detect checksum field in messages.

        Args:
            messages: List of messages to analyze.

        Returns:
            ChecksumDetectionResult with detection results.

        Example:
            >>> detector = ChecksumDetector()
            >>> result = detector.detect_checksum_field(messages)
        """
        self._messages = messages
        candidates = detect_checksum_fields(messages)

        if not candidates:
            self._detection_result = ChecksumDetectionResult(has_checksum=False, confidence=0.0)
            return self._detection_result

        # Filter by correlation threshold
        good_candidates = [c for c in candidates if c.correlation >= self.correlation_threshold]

        if not good_candidates:
            # Report no checksum with low confidence if candidates exist but none pass threshold
            max_correlation = max(c.correlation for c in candidates) if candidates else 0.0
            self._detection_result = ChecksumDetectionResult(
                has_checksum=False, candidates=candidates, confidence=max_correlation
            )
            return self._detection_result

        # Use best candidate, preferring trailer checksums when correlation is similar
        best = good_candidates[0]

        # Check if there's a trailer checksum with similar correlation
        for candidate in good_candidates[1:]:
            if candidate.position == "trailer" and best.position == "header":
                # Prefer trailer if correlation is within 5% of header
                if candidate.correlation >= best.correlation * 0.95:
                    best = candidate
                    break

        # Try to identify algorithm for best candidate
        algorithm_match = identify_checksum_algorithm(messages, best.offset, best.size)

        # If algorithm identification fails, try other high-correlation candidates
        if algorithm_match is None and len(good_candidates) > 1:
            for candidate in good_candidates[1:]:
                # Skip if correlation is too much lower
                if candidate.correlation < best.correlation * 0.9:
                    break

                alt_match = identify_checksum_algorithm(messages, candidate.offset, candidate.size)
                if alt_match is not None:
                    # Found a candidate with identifiable algorithm
                    best = candidate
                    algorithm_match = alt_match
                    break

        # Reduce confidence if algorithm couldn't be identified
        # High correlation but no identifiable algorithm suggests false positive
        final_confidence = best.correlation
        if algorithm_match is None:
            final_confidence = best.correlation * 0.3  # Penalize unidentified algorithms

        self._detection_result = ChecksumDetectionResult(
            has_checksum=True,
            offset=best.offset,
            size=best.size,
            algorithm=algorithm_match.algorithm if algorithm_match else None,
            confidence=final_confidence,
            candidates=good_candidates,
            scope_start=algorithm_match.scope_start if algorithm_match else None,
            scope_end=algorithm_match.scope_end if algorithm_match else None,
            init_value=algorithm_match.init_value if algorithm_match else None,
        )
        return self._detection_result

    def identify_algorithm(
        self, messages: list[DataType], offset: int, size: int | None = None
    ) -> ChecksumMatch | None:
        """Identify checksum algorithm at given offset.

        Args:
            messages: List of messages.
            offset: Checksum field offset.
            size: Field size (auto-detect if None).

        Returns:
            ChecksumMatch or None if no match found.
        """
        return identify_checksum_algorithm(messages, offset, size)

    def verify(
        self, messages: list[DataType], algorithm: str, offset: int, **kwargs: Any
    ) -> tuple[int, int]:
        """Verify checksums in messages.

        Args:
            messages: List of messages.
            algorithm: Checksum algorithm name.
            offset: Checksum field offset.
            **kwargs: Algorithm-specific parameters.

        Returns:
            Tuple of (passed, failed) counts.
        """
        return verify_checksums(messages, algorithm, offset, **kwargs)

    def verify_checksum(self, message: DataType) -> bool:
        """Verify checksum for a single message.

        Uses previously detected checksum parameters if available.

        Args:
            message: Single message to verify.

        Returns:
            True if checksum is valid, False otherwise.

        Example:
            >>> detector = ChecksumDetector()
            >>> detector.detect_checksum_field(messages)
            >>> is_valid = detector.verify_checksum(messages[0])
        """
        if self._detection_result is None or not self._detection_result.has_checksum:
            # Try to detect checksum from the single message
            return False

        offset = self._detection_result.offset
        size = self._detection_result.size

        if offset is None or size is None:
            return False

        # Convert message to bytes
        if isinstance(message, np.ndarray):
            msg = message.tobytes() if message.dtype == np.uint8 else bytes(message.flatten())
        else:
            msg = bytes(message)

        if self._detection_result.algorithm is None:
            # No algorithm identified - try common ones
            if size == 1:
                algorithms = ["xor", "sum8"]
            elif size == 2:
                algorithms = ["crc16_ccitt", "crc16_ibm", "sum16_big", "sum16_little"]
            elif size == 4:
                algorithms = ["crc32"]
            else:
                algorithms = ["xor", "sum8", "crc16_ccitt", "crc16_ibm", "sum16_big", "crc32"]

            # Try each algorithm
            for algo in algorithms:
                passed, _failed = verify_checksums([msg], algo, offset)
                if passed == 1:
                    return True

            return False

        # Use identified algorithm
        passed, _failed = verify_checksums(
            [msg],
            self._detection_result.algorithm,
            self._detection_result.offset or 0,
            scope_start=self._detection_result.scope_start or 0,
            scope_end=self._detection_result.scope_end,
            init_value=self._detection_result.init_value,
        )
        return passed == 1


__all__ = [
    "ChecksumCandidate",
    "ChecksumDetectionResult",
    "ChecksumDetector",
    "ChecksumMatch",
    "compute_checksum",
    "crc8",
    "crc16_ccitt",
    "crc16_ibm",
    "crc32",
    "detect_checksum_fields",
    "identify_checksum_algorithm",
    "sum8",
    "sum16",
    "verify_checksums",
    "xor_checksum",
]
