"""Payload extraction and analysis framework for network packets.

    - RE-PAY-001: Payload Extraction Framework
    - RE-PAY-002: Payload Pattern Search
    - RE-PAY-003: Payload Delimiter Detection
    - RE-PAY-004: Payload Field Inference
    - RE-PAY-005: Payload Comparison and Differential Analysis

This module provides comprehensive payload extraction from PCAP packets,
pattern search capabilities, delimiter detection, and comparison tools.
"""

from __future__ import annotations

import logging
import re
import struct
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PayloadInfo:
    """Extracted payload with metadata.

    Implements RE-PAY-001: Payload with preserved metadata.

    Attributes:
        data: Payload bytes.
        packet_index: Index of source packet.
        timestamp: Packet timestamp (optional).
        src_ip: Source IP address (optional).
        dst_ip: Destination IP address (optional).
        src_port: Source port (optional).
        dst_port: Destination port (optional).
        protocol: Protocol name (optional).
        is_fragment: Whether packet is a fragment.
        fragment_offset: Fragment offset if fragmented.
    """

    data: bytes
    packet_index: int
    timestamp: float | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    src_port: int | None = None
    dst_port: int | None = None
    protocol: str | None = None
    is_fragment: bool = False
    fragment_offset: int = 0


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


@dataclass
class PayloadDiff:
    """Difference between two payloads.

    Implements RE-PAY-005: Payload comparison result.

    Attributes:
        common_prefix_length: Length of common prefix.
        common_suffix_length: Length of common suffix.
        differences: List of (offset, byte_a, byte_b) for differences.
        similarity: Similarity score (0-1).
        edit_distance: Levenshtein edit distance.
    """

    common_prefix_length: int
    common_suffix_length: int
    differences: list[tuple[int, int, int]]
    similarity: float
    edit_distance: int


@dataclass
class VariablePositions:
    """Analysis of which byte positions vary across payloads.

    Implements RE-PAY-005: Variable position analysis.

    Attributes:
        constant_positions: Positions that are constant.
        variable_positions: Positions that vary.
        constant_values: Values at constant positions.
        variance_by_position: Variance at each position.
    """

    constant_positions: list[int]
    variable_positions: list[int]
    constant_values: dict[int, int]
    variance_by_position: np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass
class PayloadCluster:
    """Cluster of similar payloads.

    Implements RE-PAY-005: Payload clustering result.

    Attributes:
        cluster_id: Cluster identifier.
        payloads: List of payload data in cluster.
        indices: Original indices of payloads.
        representative: Representative payload (centroid).
        size: Number of payloads in cluster.
    """

    cluster_id: int
    payloads: list[bytes]
    indices: list[int]
    representative: bytes
    size: int


# =============================================================================
# RE-PAY-004: Payload Field Inference
# =============================================================================


@dataclass
class InferredField:
    """Inferred field from binary payload.

    Implements RE-PAY-004: Inferred field structure.

    Attributes:
        name: Field name (auto-generated).
        offset: Byte offset within message.
        size: Field size in bytes.
        inferred_type: Inferred data type.
        endianness: Detected endianness.
        is_constant: Whether field is constant across messages.
        is_sequence: Whether field appears to be a counter/sequence.
        is_checksum: Whether field appears to be a checksum.
        constant_value: Value if constant.
        confidence: Inference confidence (0-1).
        sample_values: Sample values from messages.
    """

    name: str
    offset: int
    size: int
    inferred_type: Literal[
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "bytes",
        "string",
        "unknown",
    ]
    endianness: Literal["big", "little", "n/a"] = "n/a"
    is_constant: bool = False
    is_sequence: bool = False
    is_checksum: bool = False
    constant_value: bytes | None = None
    confidence: float = 0.5
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class MessageSchema:
    """Inferred message schema.

    Implements RE-PAY-004: Complete message schema.

    Attributes:
        fields: List of inferred fields.
        message_length: Total message length.
        fixed_length: Whether all messages have same length.
        length_range: (min, max) length range.
        sample_count: Number of samples analyzed.
        confidence: Overall schema confidence.
    """

    fields: list[InferredField]
    message_length: int
    fixed_length: bool
    length_range: tuple[int, int]
    sample_count: int
    confidence: float


class FieldInferrer:
    """Infer field structure within binary payloads.

    Implements RE-PAY-004: Payload Field Inference.

    Uses statistical analysis, alignment detection, and type inference
    to reconstruct message formats from binary payload samples.

    Example:
        >>> inferrer = FieldInferrer()
        >>> messages = [pkt.data for pkt in udp_packets]
        >>> schema = inferrer.infer_fields(messages)
        >>> for field in schema.fields:
        ...     print(f"{field.name}: {field.inferred_type} at offset {field.offset}")
    """

    def __init__(
        self,
        min_samples: int = 10,
        entropy_threshold: float = 0.5,
        sequence_threshold: int = 3,
    ) -> None:
        """Initialize field inferrer.

        Args:
            min_samples: Minimum samples for reliable inference.
            entropy_threshold: Entropy change threshold for boundary detection.
            sequence_threshold: Minimum consecutive incrementing values for sequence.
        """
        self.min_samples = min_samples
        self.entropy_threshold = entropy_threshold
        self.sequence_threshold = sequence_threshold

    def infer_fields(
        self,
        messages: Sequence[bytes],
        min_samples: int | None = None,
    ) -> MessageSchema:
        """Infer field structure from message samples.

        Implements RE-PAY-004: Complete field inference.

        Args:
            messages: List of binary message samples.
            min_samples: Override minimum sample count.

        Returns:
            MessageSchema with inferred field structure.

        Example:
            >>> schema = inferrer.infer_fields(messages)
            >>> print(f"Detected {len(schema.fields)} fields")
        """
        if not messages:
            return MessageSchema(
                fields=[],
                message_length=0,
                fixed_length=True,
                length_range=(0, 0),
                sample_count=0,
                confidence=0.0,
            )

        min_samples = min_samples or self.min_samples
        lengths = [len(m) for m in messages]
        min_len = min(lengths)
        max_len = max(lengths)
        fixed_length = min_len == max_len

        # Use shortest message length for analysis
        analysis_length = min_len

        # Find field boundaries using entropy transitions
        boundaries = self._detect_field_boundaries(messages, analysis_length)

        # Infer field types for each segment
        fields = []
        for i, (start, end) in enumerate(boundaries):
            field = self._infer_field(messages, start, end, i)
            fields.append(field)

        # Calculate overall confidence
        if fields:
            confidence = sum(f.confidence for f in fields) / len(fields)
        else:
            confidence = 0.0

        return MessageSchema(
            fields=fields,
            message_length=analysis_length,
            fixed_length=fixed_length,
            length_range=(min_len, max_len),
            sample_count=len(messages),
            confidence=confidence,
        )

    def detect_field_types(
        self,
        messages: Sequence[bytes],
        boundaries: list[tuple[int, int]],
    ) -> list[InferredField]:
        """Detect field types for given boundaries.

        Implements RE-PAY-004: Field type detection.

        Args:
            messages: Message samples.
            boundaries: List of (start, end) field boundaries.

        Returns:
            List of InferredField with type information.
        """
        fields = []
        for i, (start, end) in enumerate(boundaries):
            field = self._infer_field(messages, start, end, i)
            fields.append(field)
        return fields

    def find_sequence_fields(
        self,
        messages: Sequence[bytes],
    ) -> list[tuple[int, int]]:
        """Find fields that appear to be sequence/counter values.

        Implements RE-PAY-004: Sequence field detection.

        Args:
            messages: Message samples (should be in order).

        Returns:
            List of (offset, size) for sequence fields.

        Raises:
            ValueError: If messages are too short for field extraction.
        """
        if len(messages) < self.sequence_threshold:
            return []

        min_len = min(len(m) for m in messages)
        sequence_fields = []

        # Check each possible field size at each offset
        for size in [1, 2, 4]:
            for offset in range(min_len - size + 1):
                values = []
                try:
                    for msg in messages:
                        # Validate message length before slicing
                        if len(msg) < offset + size:
                            raise ValueError(
                                f"Message too short: expected at least {offset + size} bytes, "
                                f"got {len(msg)} bytes"
                            )
                        # Try both endianness
                        val_be = int.from_bytes(msg[offset : offset + size], "big")
                        values.append(val_be)

                    if self._is_sequence(values):
                        sequence_fields.append((offset, size))
                except (ValueError, IndexError) as e:
                    # Skip this offset/size combination if extraction fails
                    logger.debug(f"Skipping field at offset={offset}, size={size}: {e}")
                    continue

        return sequence_fields

    def find_checksum_fields(
        self,
        messages: Sequence[bytes],
    ) -> list[tuple[int, int, str]]:
        """Find fields that appear to be checksums.

        Implements RE-PAY-004: Checksum field detection.

        Args:
            messages: Message samples.

        Returns:
            List of (offset, size, algorithm_hint) for checksum fields.

        Raises:
            ValueError: If checksum field validation fails.
        """
        if len(messages) < 5:
            return []

        min_len = min(len(m) for m in messages)
        checksum_fields = []

        # Common checksum sizes and positions
        for size in [1, 2, 4]:
            # Check last position (most common)
            for offset in [min_len - size, 0]:
                if offset < 0:
                    continue

                try:
                    # Validate offset and size before processing
                    if offset + size > min_len:
                        raise ValueError(
                            f"Invalid checksum field: offset={offset} + size={size} exceeds "
                            f"minimum message length={min_len}"
                        )

                    # Extract field values and message content
                    score = self._check_checksum_correlation(messages, offset, size)

                    if score > 0.8:
                        algorithm = self._guess_checksum_algorithm(messages, offset, size)
                        checksum_fields.append((offset, size, algorithm))
                except (ValueError, IndexError) as e:
                    # Skip this offset/size combination if validation fails
                    logger.debug(f"Skipping checksum field at offset={offset}, size={size}: {e}")
                    continue

        return checksum_fields

    def _detect_field_boundaries(
        self,
        messages: Sequence[bytes],
        max_length: int,
    ) -> list[tuple[int, int]]:
        """Detect field boundaries using entropy analysis.

        Args:
            messages: Message samples.
            max_length: Maximum length to analyze.

        Returns:
            List of (start, end) boundaries.
        """
        if max_length == 0:
            return []

        # Calculate per-byte entropy
        byte_entropies = []
        for pos in range(max_length):
            values = [m[pos] for m in messages if len(m) > pos]
            if len(values) < 2:
                byte_entropies.append(0.0)
                continue

            counts = Counter(values)
            total = len(values)
            entropy = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            byte_entropies.append(entropy)

        # Find boundaries at entropy transitions
        boundaries = []
        current_start = 0

        for i in range(1, len(byte_entropies)):
            delta = abs(byte_entropies[i] - byte_entropies[i - 1])

            # Also check for constant vs variable patterns
            if delta > self.entropy_threshold:
                if i > current_start:
                    boundaries.append((current_start, i))
                current_start = i

        # Add final segment
        if max_length > current_start:
            boundaries.append((current_start, max_length))

        # Merge very small segments
        merged: list[tuple[int, int]] = []
        for start, end in boundaries:
            if merged and start - merged[-1][1] == 0 and end - start < 2:
                # Merge with previous
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        return merged if merged else [(0, max_length)]

    def _infer_field(
        self,
        messages: Sequence[bytes],
        start: int,
        end: int,
        index: int,
    ) -> InferredField:
        """Infer type for a single field.

        Args:
            messages: Message samples.
            start: Field start offset.
            end: Field end offset.
            index: Field index for naming.

        Returns:
            InferredField with inferred type.
        """
        size = end - start
        name = f"field_{index}"

        # Extract field values
        values = []
        raw_values = []
        for msg in messages:
            if len(msg) >= end:
                field_bytes = msg[start:end]
                raw_values.append(field_bytes)
                values.append(field_bytes)

        if not values:
            return InferredField(
                name=name,
                offset=start,
                size=size,
                inferred_type="unknown",
                confidence=0.0,
            )

        # Check if constant
        unique_values = set(raw_values)
        is_constant = len(unique_values) == 1

        # Check if sequence
        is_sequence = False
        if not is_constant and size in [1, 2, 4, 8]:
            int_values = [int.from_bytes(v, "big") for v in raw_values]
            is_sequence = self._is_sequence(int_values)

        # Check for checksum patterns
        is_checksum = False
        if start >= min(len(m) for m in messages) - 4:
            score = self._check_checksum_correlation(messages, start, size)
            is_checksum = score > 0.7

        # Infer type
        inferred_type, endianness, confidence = self._infer_type(raw_values, size)

        # Sample values for debugging
        sample_values: list[int | str] = []
        for v in raw_values[:5]:
            if inferred_type.startswith("uint") or inferred_type.startswith("int"):
                try:
                    # Cast endianness to Literal type for type checker
                    byte_order: Literal["big", "little"] = (
                        "big" if endianness == "n/a" else endianness  # type: ignore[assignment]
                    )
                    sample_values.append(int.from_bytes(v, byte_order))
                except Exception:
                    sample_values.append(v.hex())
            elif inferred_type == "string":
                try:
                    sample_values.append(v.decode("utf-8", errors="replace"))
                except Exception:
                    sample_values.append(v.hex())
            else:
                sample_values.append(v.hex())

        # Cast to Literal types for type checker
        inferred_type_literal: Literal[
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
            "bytes",
            "string",
            "unknown",
        ] = inferred_type  # type: ignore[assignment]
        endianness_literal: Literal["big", "little", "n/a"] = endianness  # type: ignore[assignment]

        return InferredField(
            name=name,
            offset=start,
            size=size,
            inferred_type=inferred_type_literal,
            endianness=endianness_literal,
            is_constant=is_constant,
            is_sequence=is_sequence,
            is_checksum=is_checksum,
            constant_value=raw_values[0] if is_constant else None,
            confidence=confidence,
            sample_values=sample_values,
        )

    def _infer_type(
        self,
        values: list[bytes],
        size: int,
    ) -> tuple[str, str, float]:
        """Infer data type from values.

        Args:
            values: Field values.
            size: Field size.

        Returns:
            Tuple of (type, endianness, confidence).
        """
        if not values:
            return "unknown", "n/a", 0.0

        # Check for string (high printable ratio)
        printable_ratio = sum(
            1 for v in values for b in v if 32 <= b <= 126 or b in (9, 10, 13)
        ) / (len(values) * size)

        if printable_ratio > 0.8:
            return "string", "n/a", printable_ratio

        # Check for standard integer sizes
        if size == 1:
            return "uint8", "n/a", 0.9

        elif size == 2:
            # Try to detect endianness
            be_variance = np.var([int.from_bytes(v, "big") for v in values])
            le_variance = np.var([int.from_bytes(v, "little") for v in values])

            if be_variance < le_variance:
                endian = "big"
            else:
                endian = "little"

            return "uint16", endian, 0.8

        elif size == 4:
            # Check for float
            float_valid = 0
            for v in values:
                try:
                    f = struct.unpack(">f", v)[0]
                    if not (np.isnan(f) or np.isinf(f)) and -1e10 < f < 1e10:
                        float_valid += 1
                except Exception:
                    pass

            if float_valid / len(values) > 0.8:
                return "float32", "big", 0.7

            # Otherwise integer
            be_variance = np.var([int.from_bytes(v, "big") for v in values])
            le_variance = np.var([int.from_bytes(v, "little") for v in values])
            endian = "big" if be_variance < le_variance else "little"
            return "uint32", endian, 0.8

        elif size == 8:
            # Check for float64 or uint64
            be_variance = np.var([int.from_bytes(v, "big") for v in values])
            le_variance = np.var([int.from_bytes(v, "little") for v in values])
            endian = "big" if be_variance < le_variance else "little"
            return "uint64", endian, 0.7

        else:
            return "bytes", "n/a", 0.6

    def _is_sequence(self, values: list[int]) -> bool:
        """Check if values form a sequence.

        Args:
            values: Integer values.

        Returns:
            True if values are incrementing/decrementing.
        """
        if len(values) < self.sequence_threshold:
            return False

        # Check for incrementing sequence
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        # Most diffs should be 1 (or consistent)
        counter = Counter(diffs)
        if not counter:
            return False

        most_common_diff, count = counter.most_common(1)[0]
        ratio = count / len(diffs)

        return ratio > 0.8 and most_common_diff in [1, -1, 0]

    def _check_checksum_correlation(
        self,
        messages: Sequence[bytes],
        offset: int,
        size: int,
    ) -> float:
        """Check if field correlates with message content like a checksum.

        Args:
            messages: Message samples.
            offset: Field offset.
            size: Field size.

        Returns:
            Correlation score (0-1).
        """
        # Simple heuristic: checksum fields have high correlation with
        # changes in other parts of the message

        if len(messages) < 5:
            return 0.0

        # Extract checksum values and message content
        checksums = []
        contents = []

        for msg in messages:
            if len(msg) >= offset + size:
                checksums.append(int.from_bytes(msg[offset : offset + size], "big"))
                # Content before checksum
                content = msg[:offset] + msg[offset + size :]
                contents.append(sum(content) % 65536)

        if len(checksums) < 5:
            return 0.0

        # Check if checksum changes correlate with content changes
        unique_contents = len(set(contents))
        unique_checksums = len(set(checksums))

        if unique_contents == 1 and unique_checksums == 1:
            return 0.3  # Both constant - inconclusive

        # Simple correlation check
        if unique_contents > 1 and unique_checksums > 1:
            return 0.8

        return 0.3

    def _guess_checksum_algorithm(
        self,
        messages: Sequence[bytes],
        offset: int,
        size: int,
    ) -> str:
        """Guess the checksum algorithm.

        Args:
            messages: Message samples.
            offset: Checksum offset.
            size: Checksum size.

        Returns:
            Algorithm name hint.
        """
        if size == 1:
            return "xor8_or_sum8"
        elif size == 2:
            return "crc16_or_sum16"
        elif size == 4:
            return "crc32"
        return "unknown"


# =============================================================================
# RE-PAY-004: Convenience functions
# =============================================================================


def infer_fields(messages: Sequence[bytes], min_samples: int = 10) -> MessageSchema:
    """Infer field structure from message samples.

    Implements RE-PAY-004: Payload Field Inference.

    Args:
        messages: List of binary message samples.
        min_samples: Minimum samples for reliable inference.

    Returns:
        MessageSchema with inferred field structure.

    Example:
        >>> messages = [pkt.data for pkt in packets]
        >>> schema = infer_fields(messages)
        >>> for field in schema.fields:
        ...     print(f"{field.name}: {field.inferred_type}")
    """
    inferrer = FieldInferrer(min_samples=min_samples)
    return inferrer.infer_fields(messages)


def detect_field_types(
    messages: Sequence[bytes],
    boundaries: list[tuple[int, int]],
) -> list[InferredField]:
    """Detect field types for given boundaries.

    Implements RE-PAY-004: Field type detection.

    Args:
        messages: Message samples.
        boundaries: List of (start, end) field boundaries.

    Returns:
        List of InferredField with type information.
    """
    inferrer = FieldInferrer()
    return inferrer.detect_field_types(messages, boundaries)


def find_sequence_fields(messages: Sequence[bytes]) -> list[tuple[int, int]]:
    """Find fields that appear to be sequence/counter values.

    Implements RE-PAY-004: Sequence field detection.

    Args:
        messages: Message samples (should be in order).

    Returns:
        List of (offset, size) for sequence fields.
    """
    inferrer = FieldInferrer()
    return inferrer.find_sequence_fields(messages)


def find_checksum_fields(messages: Sequence[bytes]) -> list[tuple[int, int, str]]:
    """Find fields that appear to be checksums.

    Implements RE-PAY-004: Checksum field detection.

    Args:
        messages: Message samples.

    Returns:
        List of (offset, size, algorithm_hint) for checksum fields.
    """
    inferrer = FieldInferrer()
    return inferrer.find_checksum_fields(messages)


class PayloadExtractor:
    """Extract payloads from network packets.

    Implements RE-PAY-001: Payload Extraction Framework.

    Provides zero-copy payload extraction from UDP/TCP packets
    with metadata preservation and fragment handling.

    Example:
        >>> extractor = PayloadExtractor()
        >>> payloads = extractor.extract_all_payloads(packets, protocol="UDP")
        >>> for p in payloads:
        ...     print(f"{p.src_ip}:{p.src_port} -> {len(p.data)} bytes")
    """

    def __init__(
        self,
        include_headers: bool = False,
        zero_copy: bool = True,
        return_type: Literal["bytes", "memoryview", "numpy"] = "bytes",
    ) -> None:
        """Initialize payload extractor.

        Args:
            include_headers: Include protocol headers in payload.
            zero_copy: Use zero-copy memoryview where possible.
            return_type: Type for returned payload data.
        """
        self.include_headers = include_headers
        self.zero_copy = zero_copy
        self.return_type = return_type

    def extract_payload(
        self,
        packet: dict[str, Any] | bytes,
        layer: Literal["ethernet", "ip", "transport", "application"] = "application",
    ) -> bytes | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Extract payload from a single packet.

        Implements RE-PAY-001: Single packet payload extraction.

        Args:
            packet: Packet data (dict with 'data' key or raw bytes).
            layer: OSI layer to extract from.

        Returns:
            Payload data in requested format.

        Example:
            >>> payload = extractor.extract_payload(packet)
            >>> print(f"Payload: {len(payload)} bytes")
        """
        # Handle different packet formats
        if isinstance(packet, dict):
            raw_data = packet.get("data", packet.get("payload", b""))
            if isinstance(raw_data, list | tuple):
                raw_data = bytes(raw_data)
        else:
            raw_data = packet

        if not raw_data:
            return self._format_output(b"")

        # For raw bytes, return as-is
        if layer == "application":
            return self._format_output(raw_data)

        # Layer-based extraction would require protocol parsing
        # For now, return full data
        return self._format_output(raw_data)

    def extract_all_payloads(
        self,
        packets: Sequence[dict[str, Any] | bytes],
        protocol: str | None = None,
        port_filter: tuple[int | None, int | None] | None = None,
    ) -> list[PayloadInfo]:
        """Extract payloads from all packets with metadata.

        Implements RE-PAY-001: Batch payload extraction with metadata.

        Args:
            packets: Sequence of packets.
            protocol: Filter by protocol (e.g., "UDP", "TCP").
            port_filter: (src_port, dst_port) filter tuple.

        Returns:
            List of PayloadInfo with extracted data and metadata.

        Example:
            >>> payloads = extractor.extract_all_payloads(packets, protocol="UDP")
            >>> print(f"Extracted {len(payloads)} payloads")
        """
        results = []

        for i, packet in enumerate(packets):
            if isinstance(packet, dict):
                # Extract metadata from dict
                pkt_protocol = packet.get("protocol", "")
                src_port = packet.get("src_port")
                dst_port = packet.get("dst_port")

                # Apply filters
                if protocol and pkt_protocol.upper() != protocol.upper():
                    continue

                if port_filter:
                    if port_filter[0] is not None and src_port != port_filter[0]:
                        continue
                    if port_filter[1] is not None and dst_port != port_filter[1]:
                        continue

                payload = self.extract_payload(packet)
                if isinstance(payload, memoryview | np.ndarray):
                    payload = bytes(payload)

                info = PayloadInfo(
                    data=payload,
                    packet_index=i,
                    timestamp=packet.get("timestamp"),
                    src_ip=packet.get("src_ip"),
                    dst_ip=packet.get("dst_ip"),
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=pkt_protocol,
                    is_fragment=packet.get("is_fragment", False),
                    fragment_offset=packet.get("fragment_offset", 0),
                )
                results.append(info)
            else:
                # Raw bytes
                payload = bytes(packet)
                info = PayloadInfo(data=payload, packet_index=i)
                results.append(info)

        return results

    def iter_payloads(
        self,
        packets: Sequence[dict[str, Any] | bytes],
    ) -> Iterator[PayloadInfo]:
        """Iterate over payloads for memory-efficient processing.

        Implements RE-PAY-001: Streaming payload iteration.

        Args:
            packets: Sequence of packets.

        Yields:
            PayloadInfo for each packet.
        """
        for i, packet in enumerate(packets):
            payload = self.extract_payload(packet)
            if isinstance(payload, memoryview | np.ndarray):
                payload = bytes(payload)

            if isinstance(packet, dict):
                info = PayloadInfo(
                    data=payload,
                    packet_index=i,
                    timestamp=packet.get("timestamp"),
                    src_ip=packet.get("src_ip"),
                    dst_ip=packet.get("dst_ip"),
                    src_port=packet.get("src_port"),
                    dst_port=packet.get("dst_port"),
                    protocol=packet.get("protocol"),
                )
            else:
                info = PayloadInfo(data=payload, packet_index=i)

            yield info

    def _format_output(
        self, data: bytes
    ) -> bytes | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Format output according to return_type setting."""
        if self.return_type == "bytes":
            return data
        elif self.return_type == "memoryview":
            return memoryview(data)
        # self.return_type == "numpy"
        return np.frombuffer(data, dtype=np.uint8)


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
    # Combine payloads if list
    if isinstance(payloads, list | tuple):
        data: bytes = b"".join(payloads)
    else:
        # Type narrowing: payloads is bytes here
        data = cast("bytes", payloads)

    if not data:
        return DelimiterResult(
            delimiter=b"",
            delimiter_type="fixed",
            confidence=0.0,
            occurrences=0,
        )

    # Default candidates
    if candidates is None:
        candidates = [
            b"\r\n",  # CRLF
            b"\n",  # LF
            b"\x00",  # Null
            b"\r",  # CR
            b"\x0d\x0a",  # CRLF (explicit)
        ]

    best_result = None
    best_score = 0.0

    for delim in candidates:
        if len(delim) == 0:
            continue

        count = data.count(delim)
        if count < 2:
            continue

        # Calculate score based on frequency and regularity
        positions = []
        pos = 0
        while True:
            pos = data.find(delim, pos)
            if pos == -1:
                break
            positions.append(pos)
            pos += len(delim)

        if len(positions) < 2:
            continue

        # Calculate interval regularity
        intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        if len(intervals) > 0:
            mean_interval = sum(intervals) / len(intervals)
            if mean_interval > 0:
                variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                cv = (variance**0.5) / mean_interval if mean_interval > 0 else 1.0
                regularity = 1.0 / (1.0 + cv)
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        # Score combines frequency and regularity
        score = count * (0.5 + 0.5 * regularity)

        if score > best_score:
            best_score = score
            best_result = DelimiterResult(
                delimiter=delim,
                delimiter_type="fixed",
                confidence=min(1.0, regularity * 0.8 + 0.2 * min(1.0, count / 10)),
                occurrences=count,
                positions=positions,
            )

    if best_result is None:
        return DelimiterResult(
            delimiter=b"",
            delimiter_type="fixed",
            confidence=0.0,
            occurrences=0,
        )

    return best_result


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


def diff_payloads(payload_a: bytes, payload_b: bytes) -> PayloadDiff:
    """Compare two payloads and identify differences.

    Implements RE-PAY-005: Payload differential analysis.

    Args:
        payload_a: First payload.
        payload_b: Second payload.

    Returns:
        PayloadDiff with comparison results.

    Example:
        >>> diff = diff_payloads(pkt1.data, pkt2.data)
        >>> print(f"Common prefix: {diff.common_prefix_length} bytes")
        >>> print(f"Different bytes: {len(diff.differences)}")
    """
    # Find common prefix
    common_prefix = 0
    min_len = min(len(payload_a), len(payload_b))
    for i in range(min_len):
        if payload_a[i] == payload_b[i]:
            common_prefix += 1
        else:
            break

    # Find common suffix
    common_suffix = 0
    for i in range(1, min_len - common_prefix + 1):
        if payload_a[-i] == payload_b[-i]:
            common_suffix += 1
        else:
            break

    # Find all differences
    differences = []
    for i in range(min_len):
        if payload_a[i] != payload_b[i]:
            differences.append((i, payload_a[i], payload_b[i]))

    # Add length differences
    if len(payload_a) > len(payload_b):
        for i in range(len(payload_b), len(payload_a)):
            differences.append((i, payload_a[i], -1))
    elif len(payload_b) > len(payload_a):
        for i in range(len(payload_a), len(payload_b)):
            differences.append((i, -1, payload_b[i]))

    # Calculate similarity
    max_len = max(len(payload_a), len(payload_b))
    if max_len == 0:
        similarity = 1.0
    else:
        matching = min_len - len([d for d in differences if d[0] < min_len])
        similarity = matching / max_len

    # Calculate edit distance (simplified Levenshtein)
    edit_distance = _levenshtein_distance(payload_a, payload_b)

    return PayloadDiff(
        common_prefix_length=common_prefix,
        common_suffix_length=common_suffix,
        differences=differences,
        similarity=similarity,
        edit_distance=edit_distance,
    )


def find_common_bytes(payloads: Sequence[bytes]) -> bytes:
    """Find common prefix across all payloads.

    Implements RE-PAY-005: Common byte analysis.

    Args:
        payloads: List of payloads to analyze.

    Returns:
        Common prefix bytes.
    """
    if not payloads:
        return b""

    if len(payloads) == 1:
        return payloads[0]

    # Find minimum length
    min_len = min(len(p) for p in payloads)

    # Find common prefix
    common = bytearray()
    for i in range(min_len):
        byte = payloads[0][i]
        if all(p[i] == byte for p in payloads):
            common.append(byte)
        else:
            break

    return bytes(common)


def find_variable_positions(payloads: Sequence[bytes]) -> VariablePositions:
    """Identify which byte positions vary across payloads.

    Implements RE-PAY-005: Variable position detection.

    Args:
        payloads: List of payloads to analyze.

    Returns:
        VariablePositions with constant and variable position info.

    Example:
        >>> result = find_variable_positions(payloads)
        >>> print(f"Constant positions: {result.constant_positions}")
        >>> print(f"Variable positions: {result.variable_positions}")
    """
    if not payloads:
        return VariablePositions(
            constant_positions=[],
            variable_positions=[],
            constant_values={},
            variance_by_position=np.array([]),
        )

    # Use shortest payload length
    min_len = min(len(p) for p in payloads)

    constant_positions = []
    variable_positions = []
    constant_values = {}
    variances = []

    for i in range(min_len):
        values = [p[i] for p in payloads]
        unique = set(values)

        if len(unique) == 1:
            constant_positions.append(i)
            constant_values[i] = values[0]
            variances.append(0.0)
        else:
            variable_positions.append(i)
            variances.append(float(np.var(values)))

    return VariablePositions(
        constant_positions=constant_positions,
        variable_positions=variable_positions,
        constant_values=constant_values,
        variance_by_position=np.array(variances),
    )


def compute_similarity(
    payload_a: bytes,
    payload_b: bytes,
    metric: Literal["levenshtein", "hamming", "jaccard"] = "levenshtein",
) -> float:
    """Compute similarity between two payloads.

    Implements RE-PAY-005: Similarity computation.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        metric: Similarity metric to use.

    Returns:
        Similarity score (0-1).
    """
    if metric == "levenshtein":
        max_len = max(len(payload_a), len(payload_b))
        if max_len == 0:
            return 1.0
        distance = _levenshtein_distance(payload_a, payload_b)
        return 1.0 - (distance / max_len)

    elif metric == "hamming":
        if len(payload_a) != len(payload_b):
            # Pad shorter one
            max_len = max(len(payload_a), len(payload_b))
            payload_a = payload_a.ljust(max_len, b"\x00")
            payload_b = payload_b.ljust(max_len, b"\x00")

        matches = sum(a == b for a, b in zip(payload_a, payload_b, strict=True))
        return matches / len(payload_a) if payload_a else 1.0

    # metric == "jaccard"
    # Treat bytes as sets
    set_a = set(payload_a)
    set_b = set(payload_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0


def cluster_payloads(
    payloads: Sequence[bytes],
    threshold: float = 0.8,
    algorithm: Literal["greedy", "dbscan"] = "greedy",
) -> list[PayloadCluster]:
    """Cluster similar payloads together.

    Implements RE-PAY-005: Payload clustering.

    Args:
        payloads: List of payloads to cluster.
        threshold: Similarity threshold for clustering.
        algorithm: Clustering algorithm.

    Returns:
        List of PayloadCluster objects.

    Example:
        >>> clusters = cluster_payloads(payloads, threshold=0.85)
        >>> for c in clusters:
        ...     print(f"Cluster {c.cluster_id}: {c.size} payloads")
    """
    if not payloads:
        return []

    if algorithm == "greedy":
        return _cluster_greedy_optimized(payloads, threshold)
    # algorithm == "dbscan"
    return _cluster_dbscan(payloads, threshold)


def correlate_request_response(
    requests: Sequence[PayloadInfo],
    responses: Sequence[PayloadInfo],
    max_delay: float = 1.0,
) -> list[tuple[PayloadInfo, PayloadInfo, float]]:
    """Correlate request payloads with responses.

    Implements RE-PAY-005: Request-response correlation.

    Args:
        requests: List of request PayloadInfo.
        responses: List of response PayloadInfo.
        max_delay: Maximum time between request and response.

    Returns:
        List of (request, response, latency) tuples.
    """
    pairs = []

    for request in requests:
        if request.timestamp is None:
            continue

        best_response = None
        best_latency = float("inf")

        for response in responses:
            if response.timestamp is None:
                continue

            latency = response.timestamp - request.timestamp
            if 0 <= latency <= max_delay and latency < best_latency:
                best_response = response
                best_latency = latency

        if best_response is not None:
            pairs.append((request, best_response, best_latency))

    return pairs


# =============================================================================
# Helper functions
# =============================================================================


def _find_pattern_in_data(
    data: bytes,
    pattern: bytes | str,
    pattern_type: str,
) -> list[tuple[int, bytes]]:
    """Find pattern occurrences in data."""
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

    elif pattern_type == "wildcard":
        # Convert wildcard pattern to regex
        if isinstance(pattern, bytes):
            # Replace ?? with . for single byte match
            regex_pattern = pattern.replace(b"??", b".")
            try:
                for match in re.finditer(regex_pattern, data, re.DOTALL):
                    matches.append((match.start(), match.group()))
            except re.error:
                pass

    elif pattern_type == "regex":
        if isinstance(pattern, str):
            pattern = pattern.encode()
        try:
            for match in re.finditer(pattern, data, re.DOTALL):
                matches.append((match.start(), match.group()))
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


def _levenshtein_distance(a: bytes, b: bytes) -> int:
    """Calculate Levenshtein edit distance between two byte sequences."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)

    if len(b) == 0:
        return len(a)

    previous_row: list[int] = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _fast_similarity(payload_a: bytes, payload_b: bytes, threshold: float) -> float | None:
    """Fast similarity check with early termination.

    Uses length-based filtering and sampling to quickly reject dissimilar payloads.
    Returns None if payloads are likely similar (needs full check),
    or a similarity value if they can be quickly determined.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        threshold: Similarity threshold for clustering.

    Returns:
        Similarity value if quickly determined, None if full check needed.
    """
    len_a = len(payload_a)
    len_b = len(payload_b)

    # Empty payloads
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0

    # Length difference filter: if lengths differ by more than (1-threshold)*max_len,
    # similarity can't exceed threshold
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    _length_diff = max_len - min_len

    # Maximum possible similarity given length difference
    max_possible_similarity = min_len / max_len
    if max_possible_similarity < threshold:
        return max_possible_similarity

    # For same-length payloads, use fast hamming similarity
    if len_a == len_b:
        # Sample comparison for large payloads
        if len_a > 50:
            # Sample first 16, last 16, and some middle bytes
            sample_size = min(48, len_a)
            mismatches = 0

            # First 16 bytes
            for i in range(min(16, len_a)):
                if payload_a[i] != payload_b[i]:
                    mismatches += 1

            # Last 16 bytes
            for i in range(1, min(17, len_a + 1)):
                if payload_a[-i] != payload_b[-i]:
                    mismatches += 1

            # Middle samples
            if len_a > 32:
                step = (len_a - 32) // 16
                if step > 0:
                    for i in range(16, len_a - 16, step):
                        if payload_a[i] != payload_b[i]:
                            mismatches += 1

            # Estimate similarity from sample
            estimated_similarity = 1.0 - (mismatches / sample_size)

            # If sample shows very low similarity, reject early
            if estimated_similarity < threshold * 0.8:
                return estimated_similarity

        # Full hamming comparison for same-length payloads (faster than Levenshtein)
        matches = sum(a == b for a, b in zip(payload_a, payload_b, strict=True))
        return matches / len_a

    # For different-length payloads, use common prefix/suffix heuristic
    common_prefix = 0
    for i in range(min_len):
        if payload_a[i] == payload_b[i]:
            common_prefix += 1
        else:
            break

    common_suffix = 0
    for i in range(1, min_len - common_prefix + 1):
        if payload_a[-i] == payload_b[-i]:
            common_suffix += 1
        else:
            break

    # Estimate similarity from prefix/suffix
    common_bytes = common_prefix + common_suffix
    estimated_similarity = common_bytes / max_len

    # If common bytes suggest low similarity, reject
    if estimated_similarity < threshold * 0.7:
        return estimated_similarity

    # Need full comparison
    return None


def _cluster_greedy_optimized(
    payloads: Sequence[bytes],
    threshold: float,
) -> list[PayloadCluster]:
    """Optimized greedy clustering algorithm.

    Uses fast pre-filtering based on length and sampling to avoid
    expensive Levenshtein distance calculations when possible.

    Args:
        payloads: List of payloads to cluster.
        threshold: Similarity threshold for clustering.

    Returns:
        List of PayloadCluster objects.
    """
    clusters: list[PayloadCluster] = []
    assigned = [False] * len(payloads)

    # Precompute lengths for fast filtering
    lengths = [len(p) for p in payloads]

    for i, payload in enumerate(payloads):
        if assigned[i]:
            continue

        # Start new cluster
        cluster_payloads = [payload]
        cluster_indices = [i]
        assigned[i] = True

        payload_len = lengths[i]

        # Find similar payloads
        for j in range(i + 1, len(payloads)):
            if assigned[j]:
                continue

            other_len = lengths[j]

            # Quick length-based rejection
            max_len = max(payload_len, other_len)
            min_len = min(payload_len, other_len)
            if min_len / max_len < threshold:
                continue

            # Try fast similarity check first
            fast_result = _fast_similarity(payload, payloads[j], threshold)

            if fast_result is not None:
                similarity = fast_result
            else:
                # Fall back to Levenshtein for uncertain cases
                similarity = compute_similarity(payload, payloads[j])

            if similarity >= threshold:
                cluster_payloads.append(payloads[j])
                cluster_indices.append(j)
                assigned[j] = True

        clusters.append(
            PayloadCluster(
                cluster_id=len(clusters),
                payloads=cluster_payloads,
                indices=cluster_indices,
                representative=payload,
                size=len(cluster_payloads),
            )
        )

    return clusters


def _cluster_greedy(
    payloads: Sequence[bytes],
    threshold: float,
) -> list[PayloadCluster]:
    """Greedy clustering algorithm (legacy, uses optimized version)."""
    return _cluster_greedy_optimized(payloads, threshold)


def _cluster_dbscan(
    payloads: Sequence[bytes],
    threshold: float,
) -> list[PayloadCluster]:
    """DBSCAN-style clustering (simplified)."""
    # For simplicity, fall back to greedy
    # Full DBSCAN would require scipy or custom implementation
    return _cluster_greedy_optimized(payloads, threshold)


__all__ = [
    "DelimiterResult",
    "FieldInferrer",
    # RE-PAY-004: Field inference
    "InferredField",
    "LengthPrefixResult",
    "MessageBoundary",
    "MessageSchema",
    "PatternMatch",
    "PayloadCluster",
    "PayloadDiff",
    # Classes
    "PayloadExtractor",
    # Data classes
    "PayloadInfo",
    "VariablePositions",
    "cluster_payloads",
    "compute_similarity",
    "correlate_request_response",
    # RE-PAY-003: Delimiter detection
    "detect_delimiter",
    "detect_field_types",
    "detect_length_prefix",
    # RE-PAY-005: Comparison
    "diff_payloads",
    "filter_by_pattern",
    "find_checksum_fields",
    "find_common_bytes",
    "find_message_boundaries",
    "find_sequence_fields",
    "find_variable_positions",
    "infer_fields",
    # RE-PAY-001: Extraction
    # (via PayloadExtractor methods)
    # RE-PAY-002: Pattern search
    "search_pattern",
    "search_patterns",
    "segment_messages",
]
