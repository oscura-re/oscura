"""Payload field inference and comparison analysis.

RE-PAY-004: Payload Field Inference
RE-PAY-005: Payload Comparison and Differential Analysis

This module provides field structure inference, payload comparison,
similarity computation, and clustering for binary payloads.
"""

from __future__ import annotations

import logging
import struct
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from oscura.analyzers.packet.payload_extraction import PayloadInfo

logger = logging.getLogger(__name__)


# =============================================================================
# RE-PAY-004: Field Inference Data Classes
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


# =============================================================================
# RE-PAY-005: Comparison Data Classes
# =============================================================================


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
# RE-PAY-004: Field Inference Class
# =============================================================================


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
            ValueError: If a message is too short for the expected field size.
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
            ValueError: If checksum field offset and size exceed message length.
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
        raw_values = self._extract_field_values(messages, start, end)

        if not raw_values:
            return InferredField(
                name=name,
                offset=start,
                size=size,
                inferred_type="unknown",
                confidence=0.0,
            )

        # Analyze field properties
        unique_values = set(raw_values)
        is_constant = len(unique_values) == 1
        is_sequence = self._check_sequence(raw_values, size, is_constant)
        is_checksum = self._check_checksum(messages, start, size)

        # Infer type and create sample values
        inferred_type, endianness, confidence = self._infer_type(raw_values, size)
        sample_values = self._create_sample_values(raw_values[:5], inferred_type, endianness)

        # Cast to Literal types for type checker
        type_literal = self._cast_type_literal(inferred_type)
        endianness_literal = self._cast_endianness_literal(endianness)

        return InferredField(
            name=name,
            offset=start,
            size=size,
            inferred_type=type_literal,
            endianness=endianness_literal,
            is_constant=is_constant,
            is_sequence=is_sequence,
            is_checksum=is_checksum,
            constant_value=raw_values[0] if is_constant else None,
            confidence=confidence,
            sample_values=sample_values,
        )

    def _extract_field_values(self, messages: Sequence[bytes], start: int, end: int) -> list[bytes]:
        """Extract field values from messages."""
        return [msg[start:end] for msg in messages if len(msg) >= end]

    def _check_sequence(self, raw_values: list[bytes], size: int, is_constant: bool) -> bool:
        """Check if field values form a sequence."""
        if is_constant or size not in [1, 2, 4, 8]:
            return False
        int_values = [int.from_bytes(v, "big") for v in raw_values]
        return self._is_sequence(int_values)

    def _check_checksum(self, messages: Sequence[bytes], start: int, size: int) -> bool:
        """Check if field appears to be a checksum."""
        if start < min(len(m) for m in messages) - 4:
            return False
        score = self._check_checksum_correlation(messages, start, size)
        return score > 0.7

    def _create_sample_values(
        self, raw_values: list[bytes], inferred_type: str, endianness: str
    ) -> list[int | str]:
        """Create sample values for debugging."""
        sample_values: list[int | str] = []
        for v in raw_values:
            if inferred_type.startswith(("uint", "int")):
                try:
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
        return sample_values

    def _cast_type_literal(
        self, inferred_type: str
    ) -> Literal[
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
    ]:
        """Cast inferred type to Literal for type checker."""
        return inferred_type  # type: ignore[return-value]

    def _cast_endianness_literal(self, endianness: str) -> Literal["big", "little", "n/a"]:
        """Cast endianness to Literal for type checker."""
        return endianness  # type: ignore[return-value]

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

        # Check for string first
        string_result = self._check_string_type(values, size)
        if string_result is not None:
            return string_result

        # Infer based on field size
        if size == 1:
            return "uint8", "n/a", 0.9
        elif size == 2:
            return self._infer_uint16_type(values)
        elif size == 4:
            return self._infer_4byte_type(values)
        elif size == 8:
            return self._infer_uint64_type(values)
        else:
            return "bytes", "n/a", 0.6

    def _check_string_type(self, values: list[bytes], size: int) -> tuple[str, str, float] | None:
        """Check if values represent string data.

        Args:
            values: Field values to check.
            size: Field size.

        Returns:
            Type tuple if string, None otherwise.
        """
        printable_ratio = sum(
            1 for v in values for b in v if 32 <= b <= 126 or b in (9, 10, 13)
        ) / (len(values) * size)

        if printable_ratio > 0.8:
            return "string", "n/a", printable_ratio
        return None

    def _infer_uint16_type(self, values: list[bytes]) -> tuple[str, str, float]:
        """Infer uint16 type and detect endianness.

        Args:
            values: Field values.

        Returns:
            Type tuple with endianness.
        """
        endian = self._detect_endianness(values)
        return "uint16", endian, 0.8

    def _infer_4byte_type(self, values: list[bytes]) -> tuple[str, str, float]:
        """Infer 4-byte type (float32 or uint32).

        Args:
            values: Field values.

        Returns:
            Type tuple with endianness.
        """
        # Check if float32
        if self._is_valid_float32(values):
            return "float32", "big", 0.7

        # Otherwise uint32
        endian = self._detect_endianness(values)
        return "uint32", endian, 0.8

    def _infer_uint64_type(self, values: list[bytes]) -> tuple[str, str, float]:
        """Infer uint64 type and detect endianness.

        Args:
            values: Field values.

        Returns:
            Type tuple with endianness.
        """
        endian = self._detect_endianness(values)
        return "uint64", endian, 0.7

    def _detect_endianness(self, values: list[bytes]) -> str:
        """Detect endianness by comparing variance.

        Args:
            values: Field values.

        Returns:
            Endianness string ("big" or "little").
        """
        be_variance = np.var([int.from_bytes(v, "big") for v in values])
        le_variance = np.var([int.from_bytes(v, "little") for v in values])
        return "big" if be_variance < le_variance else "little"

    def _is_valid_float32(self, values: list[bytes]) -> bool:
        """Check if values are valid float32 numbers.

        Args:
            values: Field values to check.

        Returns:
            True if majority are valid floats.
        """
        float_valid = 0
        for v in values:
            try:
                f = struct.unpack(">f", v)[0]
                if not (np.isnan(f) or np.isinf(f)) and -1e10 < f < 1e10:
                    float_valid += 1
            except Exception:
                pass

        return float_valid / len(values) > 0.8

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
# RE-PAY-004: Convenience Functions
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


# =============================================================================
# RE-PAY-005: Comparison Functions
# =============================================================================


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
    min_len = min(len(payload_a), len(payload_b))

    common_prefix = _find_common_prefix(payload_a, payload_b, min_len)
    common_suffix = _find_common_suffix(payload_a, payload_b, min_len, common_prefix)
    differences = _find_payload_differences(payload_a, payload_b, min_len)

    similarity = _calculate_similarity(payload_a, payload_b, min_len, differences)
    edit_distance = _levenshtein_distance(payload_a, payload_b)

    return PayloadDiff(
        common_prefix_length=common_prefix,
        common_suffix_length=common_suffix,
        differences=differences,
        similarity=similarity,
        edit_distance=edit_distance,
    )


def _find_common_prefix(payload_a: bytes, payload_b: bytes, min_len: int) -> int:
    """Find length of common prefix.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        min_len: Minimum payload length.

    Returns:
        Length of common prefix in bytes.
    """
    for i in range(min_len):
        if payload_a[i] != payload_b[i]:
            return i
    return min_len


def _find_common_suffix(
    payload_a: bytes, payload_b: bytes, min_len: int, common_prefix: int
) -> int:
    """Find length of common suffix.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        min_len: Minimum payload length.
        common_prefix: Length of common prefix.

    Returns:
        Length of common suffix in bytes.
    """
    for i in range(1, min_len - common_prefix + 1):
        if payload_a[-i] != payload_b[-i]:
            return i - 1
    return min_len - common_prefix


def _find_payload_differences(
    payload_a: bytes, payload_b: bytes, min_len: int
) -> list[tuple[int, int, int]]:
    """Find all byte differences between payloads.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        min_len: Minimum payload length.

    Returns:
        List of (offset, byte_a, byte_b) tuples (-1 for missing bytes).
    """
    differences = []

    # Differences in overlapping region
    for i in range(min_len):
        if payload_a[i] != payload_b[i]:
            differences.append((i, payload_a[i], payload_b[i]))

    # Length differences
    if len(payload_a) > len(payload_b):
        for i in range(len(payload_b), len(payload_a)):
            differences.append((i, payload_a[i], -1))
    elif len(payload_b) > len(payload_a):
        for i in range(len(payload_a), len(payload_b)):
            differences.append((i, -1, payload_b[i]))

    return differences


def _calculate_similarity(
    payload_a: bytes, payload_b: bytes, min_len: int, differences: list[tuple[int, int, int]]
) -> float:
    """Calculate payload similarity ratio.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        min_len: Minimum payload length.
        differences: List of differences.

    Returns:
        Similarity ratio (0.0-1.0).
    """
    max_len = max(len(payload_a), len(payload_b))
    if max_len == 0:
        return 1.0

    matching = min_len - len([d for d in differences if d[0] < min_len])
    return matching / max_len


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
    algorithm: Literal["greedy", "dbscan", "lsh"] = "greedy",
) -> list[PayloadCluster]:
    """Cluster similar payloads together.

    Implements RE-PAY-005: Payload clustering.

    Args:
        payloads: List of payloads to cluster.
        threshold: Similarity threshold for clustering.
        algorithm: Clustering algorithm (greedy: O(n²), lsh: O(n log n)).

    Returns:
        List of PayloadCluster objects.

    Example:
        >>> clusters = cluster_payloads(payloads, threshold=0.85)
        >>> for c in clusters:
        ...     print(f"Cluster {c.cluster_id}: {c.size} payloads")

        >>> # For large datasets (>1000 payloads), use LSH for 100-1000x speedup
        >>> clusters = cluster_payloads(payloads, threshold=0.85, algorithm="lsh")
    """
    if not payloads:
        return []

    if algorithm == "lsh":
        # Use LSH for O(n log n) performance on large datasets
        from oscura.utils.performance.lsh_clustering import cluster_payloads_lsh

        return cluster_payloads_lsh(payloads, threshold=threshold)
    elif algorithm == "greedy":
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
# Helper Functions
# =============================================================================


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


def _check_length_similarity(len_a: int, len_b: int, threshold: float) -> float | None:
    """Check if length difference allows similarity threshold.

    Args:
        len_a: Length of first payload.
        len_b: Length of second payload.
        threshold: Similarity threshold.

    Returns:
        Similarity if can be determined from length, None otherwise.
    """
    # Empty payloads
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0

    # Maximum possible similarity given length difference
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    max_possible_similarity = min_len / max_len

    if max_possible_similarity < threshold:
        return max_possible_similarity

    return None


def _sample_hamming_similarity(payload_a: bytes, payload_b: bytes, length: int) -> float:
    """Compute similarity by sampling first 16, last 16, and middle bytes.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        length: Length of payloads (must be equal).

    Returns:
        Estimated similarity based on samples.
    """
    sample_size = min(48, length)
    mismatches = 0

    # First 16 bytes
    for i in range(min(16, length)):
        if payload_a[i] != payload_b[i]:
            mismatches += 1

    # Last 16 bytes
    for i in range(1, min(17, length + 1)):
        if payload_a[-i] != payload_b[-i]:
            mismatches += 1

    # Middle samples (only if length > 32)
    step = (length - 32) // 16
    if step > 0:
        for i in range(16, length - 16, step):
            if payload_a[i] != payload_b[i]:
                mismatches += 1

    return 1.0 - (mismatches / sample_size)


def _prefix_suffix_similarity(
    payload_a: bytes, payload_b: bytes, min_len: int, max_len: int
) -> float:
    """Estimate similarity from common prefix and suffix.

    Args:
        payload_a: First payload.
        payload_b: Second payload.
        min_len: Minimum length.
        max_len: Maximum length.

    Returns:
        Estimated similarity.
    """
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

    common_bytes = common_prefix + common_suffix
    return common_bytes / max_len


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

    # Check length-based similarity
    length_result = _check_length_similarity(len_a, len_b, threshold)
    if length_result is not None:
        return length_result

    # For same-length payloads, use fast hamming similarity
    if len_a == len_b:
        # Sample comparison for large payloads
        if len_a > 50:
            estimated_similarity = _sample_hamming_similarity(payload_a, payload_b, len_a)

            # If sample shows very low similarity, reject early
            if estimated_similarity < threshold * 0.8:
                return estimated_similarity

        # Full hamming comparison for same-length payloads (faster than Levenshtein)
        matches = sum(a == b for a, b in zip(payload_a, payload_b, strict=True))
        return matches / len_a

    # For different-length payloads, use common prefix/suffix heuristic
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    estimated_similarity = _prefix_suffix_similarity(payload_a, payload_b, min_len, max_len)

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
        payloads: Sequence of payload bytes to cluster.
        threshold: Similarity threshold for clustering (0.0 to 1.0).

    Returns:
        List of PayloadCluster objects containing clustered payloads.
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
    "FieldInferrer",
    "InferredField",
    "MessageSchema",
    "PayloadCluster",
    "PayloadDiff",
    "VariablePositions",
    "cluster_payloads",
    "compute_similarity",
    "correlate_request_response",
    "detect_field_types",
    "diff_payloads",
    "find_checksum_fields",
    "find_common_bytes",
    "find_sequence_fields",
    "find_variable_positions",
    "infer_fields",
]
