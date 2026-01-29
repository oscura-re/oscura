"""Payload pattern search and delimiter detection.

RE-PAY-002: Payload Pattern Search
RE-PAY-003: Payload Delimiter Detection

This module provides pattern matching, delimiter detection, and
message boundary finding for binary payloads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from oscura.analyzers.packet.payload_extraction import PayloadExtractor


@dataclass
class PatternMatch:
    """Pattern match result.

    Implements RE-PAY-002: Pattern match with location info.

    Attributes:
        pattern_name: Name of matched pattern.
        offset: Byte offset within payload.
        matched: Matched bytes.
        packet_index: Source packet index.
        context: Surrounding bytes for context.
    """

    pattern_name: str
    offset: int
    matched: bytes
    packet_index: int
    context: bytes = b""


@dataclass
class DelimiterResult:
    """Detected delimiter information.

    Implements RE-PAY-003: Delimiter detection result.

    Attributes:
        delimiter: Detected delimiter bytes.
        delimiter_type: Type of delimiter (fixed, length_prefix, pattern).
        confidence: Detection confidence (0-1).
        occurrences: Number of occurrences found.
        positions: List of positions where delimiter found.
    """

    delimiter: bytes
    delimiter_type: Literal["fixed", "length_prefix", "pattern"]
    confidence: float
    occurrences: int
    positions: list[int] = field(default_factory=list)


@dataclass
class LengthPrefixResult:
    """Length prefix detection result.

    Implements RE-PAY-003: Length prefix format detection.

    Attributes:
        detected: Whether length prefix was detected.
        length_bytes: Number of bytes for length field.
        endian: Endianness (big or little).
        offset: Offset of length field from message start.
        includes_length: Whether length includes the length field itself.
        confidence: Detection confidence (0-1).
    """

    detected: bool
    length_bytes: int = 0
    endian: Literal["big", "little"] = "big"
    offset: int = 0
    includes_length: bool = False
    confidence: float = 0.0


@dataclass
class MessageBoundary:
    """Message boundary information.

    Implements RE-PAY-003: Message boundary detection.

    Attributes:
        start: Start offset of message.
        end: End offset of message.
        length: Message length.
        data: Message data.
        index: Message index.
    """

    start: int
    end: int
    length: int
    data: bytes
    index: int


# =============================================================================
# RE-PAY-002: Pattern Search Functions
# =============================================================================


def search_pattern(
    packets: Sequence[dict[str, Any] | bytes],
    pattern: bytes | str,
    pattern_type: Literal["exact", "wildcard", "regex"] = "exact",
    context_bytes: int = 8,
) -> list[PatternMatch]:
    """Search for pattern in packet payloads.

    Implements RE-PAY-002: Payload Pattern Search.

    Args:
        packets: Sequence of packets to search.
        pattern: Pattern to search for.
        pattern_type: Type of pattern matching.
        context_bytes: Number of context bytes around match.

    Returns:
        List of PatternMatch results.

    Example:
        >>> matches = search_pattern(packets, b'\\x00\\x01\\x00\\x00')
        >>> for m in matches:
        ...     print(f"Found at packet {m.packet_index}, offset {m.offset}")
    """
    extractor = PayloadExtractor()
    results = []

    for i, packet in enumerate(packets):
        payload = extractor.extract_payload(packet)
        if isinstance(payload, memoryview | np.ndarray):
            payload = bytes(payload)

        matches = _find_pattern_in_data(payload, pattern, pattern_type)

        for offset, matched in matches:
            # Get context
            start = max(0, offset - context_bytes)
            end = min(len(payload), offset + len(matched) + context_bytes)
            context = payload[start:end]

            results.append(
                PatternMatch(
                    pattern_name=pattern.hex() if isinstance(pattern, bytes) else str(pattern),
                    offset=offset,
                    matched=matched,
                    packet_index=i,
                    context=context,
                )
            )

    return results


def search_patterns(
    packets: Sequence[dict[str, Any] | bytes],
    patterns: dict[str, bytes | str],
    context_bytes: int = 8,
) -> dict[str, list[PatternMatch]]:
    """Search for multiple patterns simultaneously.

    Implements RE-PAY-002: Multi-pattern search.

    Args:
        packets: Sequence of packets to search.
        patterns: Dictionary mapping names to patterns.
        context_bytes: Number of context bytes around match.

    Returns:
        Dictionary mapping pattern names to match lists.

    Example:
        >>> signatures = {
        ...     "header_a": b'\\xAA\\x55',
        ...     "header_b": b'\\xDE\\xAD',
        ... }
        >>> results = search_patterns(packets, signatures)
        >>> for name, matches in results.items():
        ...     print(f"{name}: {len(matches)} matches")
    """
    results: dict[str, list[PatternMatch]] = {name: [] for name in patterns}
    extractor = PayloadExtractor()

    for i, packet in enumerate(packets):
        payload = extractor.extract_payload(packet)
        if isinstance(payload, memoryview | np.ndarray):
            payload = bytes(payload)

        for name, pattern in patterns.items():
            # Detect pattern type
            if isinstance(pattern, bytes):
                if b"??" in pattern or b"\\x??" in pattern:
                    pattern_type = "wildcard"
                else:
                    pattern_type = "exact"
            else:
                pattern_type = "regex"

            matches = _find_pattern_in_data(payload, pattern, pattern_type)

            for offset, matched in matches:
                start = max(0, offset - context_bytes)
                end = min(len(payload), offset + len(matched) + context_bytes)
                context = payload[start:end]

                results[name].append(
                    PatternMatch(
                        pattern_name=name,
                        offset=offset,
                        matched=matched,
                        packet_index=i,
                        context=context,
                    )
                )

    return results


def filter_by_pattern(
    packets: Sequence[dict[str, Any] | bytes],
    pattern: bytes | str,
    pattern_type: Literal["exact", "wildcard", "regex"] = "exact",
) -> list[dict[str, Any] | bytes]:
    """Filter packets that contain a pattern.

    Implements RE-PAY-002: Pattern-based filtering.

    Args:
        packets: Sequence of packets.
        pattern: Pattern to match.
        pattern_type: Type of pattern matching.

    Returns:
        List of packets containing the pattern.
    """
    extractor = PayloadExtractor()
    result = []

    for packet in packets:
        payload = extractor.extract_payload(packet)
        if isinstance(payload, memoryview | np.ndarray):
            payload = bytes(payload)

        matches = _find_pattern_in_data(payload, pattern, pattern_type)
        if len(matches) > 0:
            result.append(packet)

    return result


# =============================================================================
# RE-PAY-003: Delimiter Detection Functions
# =============================================================================


def detect_delimiter(
    payloads: Sequence[bytes] | bytes,
    candidates: list[bytes] | None = None,
) -> DelimiterResult:
    """Automatically detect message delimiter.

    Implements RE-PAY-003: Delimiter detection.

    Args:
        payloads: Payload data or list of payloads.
        candidates: Optional list of candidate delimiters to test.

    Returns:
        DelimiterResult with detected delimiter info.

    Example:
        >>> data = b'msg1\\r\\nmsg2\\r\\nmsg3\\r\\n'
        >>> result = detect_delimiter(data)
        >>> print(f"Delimiter: {result.delimiter!r}")
    """
    data = _combine_payloads(payloads)
    if not data:
        return DelimiterResult(delimiter=b"", delimiter_type="fixed", confidence=0.0, occurrences=0)

    candidates = candidates or _default_delimiter_candidates()

    best_result = None
    best_score = 0.0

    for delim in candidates:
        result, score = _evaluate_delimiter_candidate(data, delim)
        if score > best_score:
            best_score = score
            best_result = result

    return best_result or DelimiterResult(
        delimiter=b"", delimiter_type="fixed", confidence=0.0, occurrences=0
    )


def _combine_payloads(payloads: Sequence[bytes] | bytes) -> bytes:
    """Combine payloads into single bytes object."""
    if isinstance(payloads, list | tuple):
        return b"".join(payloads)
    return cast("bytes", payloads)


def _default_delimiter_candidates() -> list[bytes]:
    """Return default delimiter candidates."""
    return [
        b"\r\n",  # CRLF
        b"\n",  # LF
        b"\x00",  # Null
        b"\r",  # CR
        b"\x0d\x0a",  # CRLF (explicit)
    ]


def _evaluate_delimiter_candidate(
    data: bytes, delim: bytes
) -> tuple[DelimiterResult | None, float]:
    """Evaluate a delimiter candidate and return result with score."""
    if len(delim) == 0:
        return None, 0.0

    count = data.count(delim)
    if count < 2:
        return None, 0.0

    positions = _find_delimiter_positions(data, delim)
    if len(positions) < 2:
        return None, 0.0

    regularity = _calculate_interval_regularity(positions)
    score = count * (0.5 + 0.5 * regularity)
    confidence = min(1.0, regularity * 0.8 + 0.2 * min(1.0, count / 10))

    result = DelimiterResult(
        delimiter=delim,
        delimiter_type="fixed",
        confidence=confidence,
        occurrences=count,
        positions=positions,
    )
    return result, score


def _find_delimiter_positions(data: bytes, delim: bytes) -> list[int]:
    """Find all positions of delimiter in data.

    Args:
        data: Data to search.
        delim: Delimiter bytes to find.

    Returns:
        List of positions where delimiter occurs.

    Raises:
        ValueError: If delimiter is empty.
    """
    if len(delim) == 0:
        raise ValueError("Delimiter cannot be empty")

    positions = []
    pos = 0
    while True:
        pos = data.find(delim, pos)
        if pos == -1:
            break
        positions.append(pos)
        pos += len(delim)
    return positions


def _calculate_interval_regularity(positions: list[int]) -> float:
    """Calculate regularity score from delimiter positions."""
    if len(positions) < 2:
        return 0.0

    intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if len(intervals) == 0:
        return 0.0

    mean_interval = sum(intervals) / len(intervals)
    if mean_interval <= 0:
        return 0.0

    variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
    cv = (variance**0.5) / mean_interval
    regularity: float = 1.0 / (1.0 + cv)
    return regularity


def detect_length_prefix(
    payloads: Sequence[bytes],
    max_length_bytes: int = 4,
) -> LengthPrefixResult:
    """Detect length-prefixed message format.

    Implements RE-PAY-003: Length prefix detection.

    Args:
        payloads: List of payload samples.
        max_length_bytes: Maximum length field size to test.

    Returns:
        LengthPrefixResult with detected format.

    Example:
        >>> result = detect_length_prefix(payloads)
        >>> if result.detected:
        ...     print(f"Length field: {result.length_bytes} bytes, {result.endian}")
    """
    if not payloads:
        return LengthPrefixResult(detected=False)

    # Concatenate payloads for analysis
    data = b"".join(payloads)

    best_result = LengthPrefixResult(detected=False)
    best_score = 0.0

    # Try different length field sizes and offsets
    # IMPORTANT: Prefer larger length_bytes values when scores are equal
    # by iterating in reverse order (4, 2, 1) and using >= for comparison
    for length_bytes in [4, 2, 1]:
        if length_bytes > max_length_bytes:
            continue

        for endian_str in ["big", "little"]:
            endian: Literal["big", "little"] = endian_str  # type: ignore[assignment]
            for offset in range(min(8, len(data) - length_bytes)):
                for includes_length in [False, True]:
                    score, matches = _test_length_prefix(
                        data, length_bytes, endian, offset, includes_length
                    )

                    # Use > to prefer larger length_bytes (tested first) when scores are equal
                    if score > best_score and matches >= 3:
                        best_score = score
                        best_result = LengthPrefixResult(
                            detected=True,
                            length_bytes=length_bytes,
                            endian=endian,
                            offset=offset,
                            includes_length=includes_length,
                            confidence=score,
                        )

    return best_result


def find_message_boundaries(
    payloads: Sequence[bytes] | bytes,
    delimiter: bytes | DelimiterResult | None = None,
    length_prefix: LengthPrefixResult | None = None,
) -> list[MessageBoundary]:
    """Find message boundaries in payload data.

    Implements RE-PAY-003: Message boundary detection.

    Args:
        payloads: Payload data or list of payloads.
        delimiter: Delimiter to use (auto-detect if None).
        length_prefix: Length prefix format (test if None).

    Returns:
        List of MessageBoundary objects.

    Example:
        >>> boundaries = find_message_boundaries(data)
        >>> for b in boundaries:
        ...     print(f"Message {b.index}: {b.length} bytes")
    """
    # Combine payloads if list
    if isinstance(payloads, list | tuple):
        data: bytes = b"".join(payloads)
    else:
        # Type narrowing: payloads is bytes here
        data = cast("bytes", payloads)

    if not data:
        return []

    boundaries = []

    # Try length prefix first
    if length_prefix is None:
        length_prefix = detect_length_prefix([data] if isinstance(data, bytes) else list(payloads))

    if length_prefix.detected:
        boundaries = _extract_length_prefixed_messages(data, length_prefix)
        if len(boundaries) > 0:
            return boundaries

    # Fall back to delimiter
    if delimiter is None:
        delimiter = detect_delimiter(data)

    if isinstance(delimiter, DelimiterResult):
        delim = delimiter.delimiter
    else:
        delim = delimiter

    if not delim:
        # No delimiter found, return whole data as one message
        return [MessageBoundary(start=0, end=len(data), length=len(data), data=data, index=0)]

    # Split by delimiter
    parts = data.split(delim)
    current_offset = 0

    for _i, part in enumerate(parts):
        if part:  # Skip empty parts
            boundaries.append(
                MessageBoundary(
                    start=current_offset,
                    end=current_offset + len(part),
                    length=len(part),
                    data=part,
                    index=len(boundaries),
                )
            )
        current_offset += len(part) + len(delim)

    return boundaries


def segment_messages(
    payloads: Sequence[bytes] | bytes,
    delimiter: bytes | None = None,
    length_prefix: LengthPrefixResult | None = None,
) -> list[bytes]:
    """Segment stream into individual messages.

    Implements RE-PAY-003: Message segmentation.

    Args:
        payloads: Payload data or list of payloads.
        delimiter: Delimiter to use (auto-detect if None).
        length_prefix: Length prefix format (auto-detect if None).

    Returns:
        List of message bytes.
    """
    boundaries = find_message_boundaries(payloads, delimiter, length_prefix)
    return [b.data for b in boundaries]


# =============================================================================
# Helper Functions
# =============================================================================


def _find_pattern_in_data(
    data: bytes,
    pattern: bytes | str,
    pattern_type: str,
    max_matches: int = 100000,
) -> list[tuple[int, bytes]]:
    """Find pattern occurrences in data.

    Args:
        data: Data to search.
        pattern: Pattern to find.
        pattern_type: Type of pattern (exact, wildcard, regex).
        max_matches: Maximum number of matches to return (default 100000).

    Returns:
        List of (offset, matched_bytes) tuples.

    Raises:
        ValueError: If max_matches exceeded (prevents infinite loops) or pattern is empty.
    """
    # Validate pattern is not empty (prevents infinite loops)
    if isinstance(pattern, (str, bytes)):
        if len(pattern) == 0:
            raise ValueError("Pattern cannot be empty")

    matches = []

    if pattern_type == "exact":
        if isinstance(pattern, str):
            pattern = pattern.encode()
        pos = 0
        while True:
            pos = data.find(pattern, pos)
            if pos == -1:
                break
            matches.append((pos, pattern))
            pos += 1

            # Prevent infinite loops from excessive matches
            if len(matches) >= max_matches:
                raise ValueError(
                    f"Pattern match limit exceeded ({max_matches} matches). "
                    "This may indicate a problematic pattern (e.g., empty or too common)."
                )

    elif pattern_type == "wildcard":
        # Convert wildcard pattern to regex
        if isinstance(pattern, bytes):
            # Replace ?? with . for single byte match
            regex_pattern = pattern.replace(b"??", b".")
            try:
                for match in re.finditer(regex_pattern, data, re.DOTALL):
                    matches.append((match.start(), match.group()))
                    if len(matches) >= max_matches:
                        raise ValueError(
                            f"Pattern match limit exceeded ({max_matches} matches). "
                            "Wildcard pattern may be too permissive."
                        )
            except re.error:
                pass

    elif pattern_type == "regex":
        if isinstance(pattern, str):
            pattern = pattern.encode()
        try:
            for match in re.finditer(pattern, data, re.DOTALL):
                matches.append((match.start(), match.group()))
                if len(matches) >= max_matches:
                    raise ValueError(
                        f"Pattern match limit exceeded ({max_matches} matches). "
                        "Regex pattern may be too broad."
                    )
        except re.error:
            pass

    return matches


def _test_length_prefix(
    data: bytes,
    length_bytes: int,
    endian: str,
    offset: int,
    includes_length: bool,
) -> tuple[float, int]:
    """Test if data follows a length-prefix pattern."""
    matches = 0
    pos = 0

    while pos + offset + length_bytes <= len(data):
        # Read length field
        length_data = data[pos + offset : pos + offset + length_bytes]
        if endian == "big":
            length = int.from_bytes(length_data, "big")
        else:
            length = int.from_bytes(length_data, "little")

        if includes_length:
            expected_end = pos + length
        else:
            expected_end = pos + offset + length_bytes + length

        # Check if this makes sense
        if 0 < length < 65536 and expected_end <= len(data):
            matches += 1
            pos = expected_end
        else:
            break

    # Score based on matches and coverage
    coverage = pos / len(data) if len(data) > 0 else 0
    score = min(1.0, matches / 5) * coverage

    return score, matches


def _extract_length_prefixed_messages(
    data: bytes,
    length_prefix: LengthPrefixResult,
) -> list[MessageBoundary]:
    """Extract messages using detected length prefix format."""
    boundaries = []
    pos = 0
    index = 0

    while pos + length_prefix.offset + length_prefix.length_bytes <= len(data):
        # Read length
        length_data = data[
            pos + length_prefix.offset : pos + length_prefix.offset + length_prefix.length_bytes
        ]
        if length_prefix.endian == "big":
            length = int.from_bytes(length_data, "big")
        else:
            length = int.from_bytes(length_data, "little")

        if length_prefix.includes_length:
            end = pos + length
        else:
            end = pos + length_prefix.offset + length_prefix.length_bytes + length

        if end > len(data) or length <= 0:
            break

        msg_data = data[pos:end]
        boundaries.append(
            MessageBoundary(
                start=pos,
                end=end,
                length=end - pos,
                data=msg_data,
                index=index,
            )
        )

        pos = end
        index += 1

    return boundaries


__all__ = [
    "DelimiterResult",
    "LengthPrefixResult",
    "MessageBoundary",
    "PatternMatch",
    "detect_delimiter",
    "detect_length_prefix",
    "filter_by_pattern",
    "find_message_boundaries",
    "search_pattern",
    "search_patterns",
    "segment_messages",
]
