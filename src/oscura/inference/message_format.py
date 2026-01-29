"""Message format inference using statistical analysis.

Requirements addressed: PSI-001

This module automatically infers message field structure from collections of
similar messages for protocol reverse engineering.

Key capabilities:
- Detect field boundaries via entropy transitions
- Classify fields as constant, variable, or sequential
- Infer field types (integer, counter, timestamp, checksum)
- Detect field dependencies (length fields, checksums)
- Generate message format specifications
- Voting expert ensemble for improved boundary detection (IPART-style)

References:
    IPART: IP Packet Analysis using Random Forests. IEEE ISSRE 2014.
    Discoverer: Automatic Protocol Reverse Engineering. USENIX Security 2007.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from oscura.inference.alignment import align_local


@dataclass
class InferredField:
    """An inferred message field.

    : Field classification and type inference.

    Attributes:
        name: Auto-generated field name
        offset: Byte offset from message start
        size: Field size in bytes
        field_type: Inferred field type classification
        entropy: Shannon entropy of field values
        variance: Statistical variance of field values
        confidence: Confidence score (0-1) for type inference
        values_seen: Sample values for validation
        evidence: Evidence from each expert (for ensemble methods)
    """

    name: str
    offset: int
    size: int
    field_type: Literal["constant", "counter", "timestamp", "length", "checksum", "data", "unknown"]
    entropy: float
    variance: float
    confidence: float
    values_seen: list[Any] = dataclass_field(default_factory=list)  # Sample values
    evidence: dict[str, bool] = dataclass_field(default_factory=dict)  # Expert evidence


@dataclass
class MessageSchema:
    """Inferred message format schema.

    : Complete message format specification.

    Attributes:
        total_size: Total message size in bytes
        fields: List of inferred fields
        field_boundaries: Byte offsets of field starts
        header_size: Detected header size
        payload_offset: Start of payload region
        checksum_field: Detected checksum field if any
        length_field: Detected length field if any
    """

    total_size: int
    fields: list[InferredField]
    field_boundaries: list[int]  # Byte offsets of field starts
    header_size: int  # Detected header size
    payload_offset: int
    checksum_field: InferredField | None
    length_field: InferredField | None


class MessageFormatInferrer:
    """Infer message format from samples.

    : Message format inference using entropy and variance analysis.

    Algorithm:
    1. Detect field boundaries using entropy transitions
    2. Classify fields based on statistical patterns
    3. Detect dependencies between fields
    4. Generate complete schema
    """

    def __init__(self, min_samples: int = 10):
        """Initialize inferrer.

        Args:
            min_samples: Minimum number of message samples required
        """
        self.min_samples = min_samples

    def infer_format(self, messages: list[bytes | NDArray[np.uint8]]) -> MessageSchema:
        """Infer message format from collection of similar messages.

        : Complete format inference workflow.

        Args:
            messages: List of message samples (bytes or np.ndarray)

        Returns:
            MessageSchema with inferred field structure

        Raises:
            ValueError: If insufficient samples or invalid input
        """
        if len(messages) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} messages, got {len(messages)}")

        # Convert to numpy arrays for processing
        msg_arrays = []
        for msg in messages:
            if isinstance(msg, bytes):
                msg_arrays.append(np.frombuffer(msg, dtype=np.uint8))
            elif isinstance(msg, np.ndarray):
                msg_arrays.append(msg.astype(np.uint8))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

        # Check all messages are same length
        lengths = [len(m) for m in msg_arrays]
        if len(set(lengths)) > 1:
            raise ValueError(f"Messages have varying lengths: {set(lengths)}")

        msg_len = lengths[0]

        # Detect field boundaries
        boundaries = self.detect_field_boundaries(msg_arrays, method="combined")

        # Detect field types
        fields = self.detect_field_types(msg_arrays, boundaries)

        # Determine header size (first high-entropy transition or first 4 fields)
        header_size = self._estimate_header_size(fields)

        # Find checksum and length fields
        checksum_field = None
        length_field = None

        for f in fields:
            if f.field_type == "checksum":
                checksum_field = f
            elif f.field_type == "length":
                length_field = f

        # Payload starts after header
        payload_offset = header_size

        schema = MessageSchema(
            total_size=msg_len,
            fields=fields,
            field_boundaries=boundaries,
            header_size=header_size,
            payload_offset=payload_offset,
            checksum_field=checksum_field,
            length_field=length_field,
        )

        return schema

    def _calculate_entropy_boundaries(
        self, messages: list[NDArray[np.uint8]], msg_len: int
    ) -> list[int]:
        """Detect boundaries using entropy transitions.

        Args:
            messages: List of message arrays
            msg_len: Length of messages

        Returns:
            List of boundary positions
        """
        boundaries = []
        entropies = []

        # Calculate entropy at each byte position
        for offset in range(msg_len):
            entropy = self._calculate_byte_entropy(messages, offset)
            entropies.append(entropy)

        # Find transitions (entropy changes > threshold)
        entropy_threshold = 1.5  # bits
        for i in range(1, len(entropies)):
            delta = abs(entropies[i] - entropies[i - 1])
            if delta > entropy_threshold:
                boundaries.append(i)

        return boundaries

    def _calculate_variance_boundaries(
        self, messages: list[NDArray[np.uint8]], msg_len: int
    ) -> list[int]:
        """Detect boundaries using variance transitions.

        Args:
            messages: List of message arrays
            msg_len: Length of messages

        Returns:
            List of boundary positions
        """
        boundaries = []
        variances = []

        # Calculate variance at each byte position
        for offset in range(msg_len):
            values = [msg[offset] for msg in messages]
            variance = np.var(values)
            variances.append(variance)

        # Find variance transitions
        var_threshold = 1000.0
        for i in range(1, len(variances)):
            delta = abs(variances[i] - variances[i - 1])
            if delta > var_threshold:
                boundaries.append(i)

        return boundaries

    def _merge_close_boundaries(self, boundaries: list[int]) -> list[int]:
        """Merge boundaries that are too close together.

        Args:
            boundaries: Sorted list of boundaries

        Returns:
            Merged boundary list
        """
        if not boundaries:
            return []

        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] >= 2:
                merged.append(b)

        return merged

    def detect_field_boundaries(
        self,
        messages: list[NDArray[np.uint8]],
        method: Literal["entropy", "variance", "combined"] = "combined",
    ) -> list[int]:
        """Detect field boundaries using entropy transitions.

        : Boundary detection via statistical transitions.

        Args:
            messages: List of message arrays
            method: Detection method ('entropy', 'variance', or 'combined')

        Returns:
            List of byte offsets marking field starts (always includes 0)
        """
        if not messages:
            return [0]

        msg_len = len(messages[0])
        boundaries = [0]  # Always start at offset 0

        # Detect boundaries using selected method
        if method in ["entropy", "combined"]:
            entropy_boundaries = self._calculate_entropy_boundaries(messages, msg_len)
            boundaries.extend(entropy_boundaries)

        if method in ["variance", "combined"]:
            variance_boundaries = self._calculate_variance_boundaries(messages, msg_len)
            boundaries.extend(variance_boundaries)

        # Merge and sort boundaries
        boundaries = sorted(set(boundaries))
        return self._merge_close_boundaries(boundaries)

    def detect_boundaries_voting(
        self,
        messages: list[bytes],
        min_confidence: float = 0.6,
    ) -> list[int]:
        """Detect field boundaries using voting expert algorithm.

        : IPART-style voting expert for boundary detection.

        Combines multiple detection strategies:
        1. Entropy-based detection
        2. Alignment-based detection (Smith-Waterman)
        3. Statistical variance detection
        4. Byte value distribution analysis
        5. N-gram frequency analysis

        Each "expert" votes on likely boundaries. Boundaries with
        votes >= min_confidence threshold are returned.

        Args:
            messages: List of protocol messages (bytes)
            min_confidence: Minimum vote fraction to accept boundary (0.0-1.0)

        Returns:
            List of byte positions that are likely field boundaries

        References:
            IPART: IP Packet Analysis using Random Forests.
            IEEE ISSRE 2014.
        """
        if not messages:
            return [0]

        # Convert to numpy arrays for processing
        msg_arrays = []
        for msg in messages:
            msg_arrays.append(np.frombuffer(msg, dtype=np.uint8))

        # Run each expert
        experts = [
            self._expert_entropy(msg_arrays),
            self._expert_alignment(messages),
            self._expert_variance(msg_arrays),
            self._expert_distribution(msg_arrays),
            self._expert_ngrams(msg_arrays, n=2),
        ]

        num_experts = len(experts)

        # Collect all possible boundary positions
        all_boundaries = set()
        for expert_boundaries in experts:
            all_boundaries.update(expert_boundaries)

        # Count votes for each boundary
        boundary_votes: dict[int, int] = {}
        for boundary in all_boundaries:
            votes = sum(1 for expert in experts if boundary in expert)
            boundary_votes[boundary] = votes

        # Filter by confidence threshold
        min_votes = int(num_experts * min_confidence)
        accepted_boundaries = [pos for pos, votes in boundary_votes.items() if votes >= min_votes]

        # Always include position 0
        if 0 not in accepted_boundaries:
            accepted_boundaries.append(0)

        # Sort and merge close boundaries
        accepted_boundaries = sorted(accepted_boundaries)

        # Merge boundaries that are too close (< 2 bytes apart)
        merged = [accepted_boundaries[0]]
        for b in accepted_boundaries[1:]:
            if b - merged[-1] >= 2:
                merged.append(b)

        return merged

    def _expert_entropy(self, messages: list[NDArray[np.uint8]]) -> set[int]:
        """Detect boundaries based on entropy changes.

        : Entropy-based boundary expert.

        Args:
            messages: List of message arrays

        Returns:
            Set of boundary positions
        """
        if not messages:
            return {0}

        msg_len = len(messages[0])
        boundaries = {0}

        # Calculate entropy at each byte position
        entropies = []
        for offset in range(msg_len):
            entropy = self._calculate_byte_entropy(messages, offset)
            entropies.append(entropy)

        # Find transitions (entropy changes > threshold)
        entropy_threshold = 1.5  # bits
        for i in range(1, len(entropies)):
            delta = abs(entropies[i] - entropies[i - 1])
            if delta > entropy_threshold:
                boundaries.add(i)

        return boundaries

    def _expert_alignment(self, messages: list[bytes]) -> set[int]:
        """Detect boundaries using Smith-Waterman alignment.

        : Alignment-based boundary expert.

        Uses local alignment to find conserved vs. variable regions.
        Transitions between regions indicate likely boundaries.

        Args:
            messages: List of protocol messages

        Returns:
            Set of boundary positions
        """
        if len(messages) < 2:
            return {0}

        boundaries = {0}

        # Compare first message to several others
        num_comparisons = min(5, len(messages) - 1)
        for i in range(1, num_comparisons + 1):
            result = align_local(messages[0], messages[i])

            # Boundaries at transitions between conserved and variable regions
            for start, _end in result.conserved_regions:
                if start > 0:
                    boundaries.add(start)

            for start, _end in result.variable_regions:
                if start > 0:
                    boundaries.add(start)

        return boundaries

    def _expert_variance(self, messages: list[NDArray[np.uint8]]) -> set[int]:
        """Detect boundaries based on statistical variance.

        : Variance-based boundary expert.

        Args:
            messages: List of message arrays

        Returns:
            Set of boundary positions
        """
        if not messages:
            return {0}

        msg_len = len(messages[0])
        boundaries = {0}

        # Calculate variance at each byte position
        variances = []
        for offset in range(msg_len):
            values = [msg[offset] for msg in messages]
            variance = np.var(values)
            variances.append(variance)

        # Find variance transitions
        var_threshold = 1000.0
        for i in range(1, len(variances)):
            delta = abs(variances[i] - variances[i - 1])
            if delta > var_threshold:
                boundaries.add(i)

        return boundaries

    def _expert_distribution(self, messages: list[NDArray[np.uint8]]) -> set[int]:
        """Detect boundaries from byte value distribution changes.

        : Distribution-based boundary expert.

        Analyzes how the distribution of byte values changes
        across positions. Sharp changes suggest boundaries.

        Args:
            messages: List of message arrays

        Returns:
            Set of boundary positions
        """
        if not messages:
            return {0}

        msg_len = len(messages[0])
        boundaries = {0}

        # Calculate distribution metrics at each position
        distributions = []
        for offset in range(msg_len):
            values = [msg[offset] for msg in messages]
            # Use unique count as distribution metric
            unique_count = len(set(values))
            distributions.append(unique_count)

        # Find sharp changes in distribution
        for i in range(1, len(distributions)):
            # Ratio of change
            if distributions[i - 1] > 0:
                ratio = distributions[i] / distributions[i - 1]
                # Significant change (>2x or <0.5x)
                if ratio > 2.0 or ratio < 0.5:
                    boundaries.add(i)

        return boundaries

    def _expert_ngrams(self, messages: list[NDArray[np.uint8]], n: int = 2) -> set[int]:
        """Detect boundaries using n-gram frequency analysis.

        : N-gram based boundary expert.

        Analyzes how n-gram patterns change across positions.
        Different n-gram distributions suggest different fields.

        Args:
            messages: List of message arrays
            n: N-gram size (default: 2)

        Returns:
            Set of boundary positions
        """
        if not messages or len(messages[0]) < n:
            return {0}

        msg_len = len(messages[0])
        boundaries = {0}

        # Collect n-grams at each position
        ngram_sets = []
        for offset in range(msg_len - n + 1):
            ngrams = set()
            for msg in messages:
                if offset + n <= len(msg):
                    ngram = tuple(msg[offset : offset + n])
                    ngrams.add(ngram)
            ngram_sets.append(ngrams)

        # Find positions where n-gram patterns change significantly
        for i in range(1, len(ngram_sets)):
            # Calculate Jaccard similarity between adjacent positions
            set1 = ngram_sets[i - 1]
            set2 = ngram_sets[i]

            if len(set1) == 0 or len(set2) == 0:
                continue

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            if union > 0:
                similarity = intersection / union
                # Low similarity suggests boundary
                if similarity < 0.3:
                    boundaries.add(i)

        return boundaries

    def infer_format_ensemble(
        self,
        messages: list[bytes | NDArray[np.uint8]],
        min_field_confidence: float = 0.6,
        min_boundary_confidence: float = 0.6,
    ) -> MessageSchema:
        """Infer message format using ensemble of techniques.

        : Ensemble-based format inference with confidence scoring.

        Combines:
        - Voting expert for boundary detection
        - Multiple field type detectors
        - Confidence scoring for each field

        Args:
            messages: List of protocol messages
            min_field_confidence: Minimum confidence to include field
            min_boundary_confidence: Minimum confidence for boundaries

        Returns:
            Message schema with confidence-scored fields

        Raises:
            ValueError: If insufficient messages provided
        """
        if len(messages) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} messages, got {len(messages)}")

        bytes_messages = self._convert_messages_to_bytes(messages)
        msg_len = self._validate_message_lengths(bytes_messages)
        boundaries = self.detect_boundaries_voting(
            bytes_messages, min_confidence=min_boundary_confidence
        )
        msg_arrays = [np.frombuffer(msg, dtype=np.uint8) for msg in bytes_messages]
        fields = self._infer_fields_from_boundaries(
            boundaries, msg_arrays, msg_len, min_field_confidence
        )
        checksum_field, length_field = self._find_special_fields(fields)
        header_size = self._estimate_header_size(fields)

        return MessageSchema(
            total_size=msg_len,
            fields=fields,
            field_boundaries=boundaries,
            header_size=header_size,
            payload_offset=header_size,
            checksum_field=checksum_field,
            length_field=length_field,
        )

    def _convert_messages_to_bytes(self, messages: list[bytes | NDArray[np.uint8]]) -> list[bytes]:
        """Convert messages to bytes format."""
        bytes_messages = []
        for msg in messages:
            if isinstance(msg, bytes):
                bytes_messages.append(msg)
            elif isinstance(msg, np.ndarray):
                bytes_messages.append(msg.tobytes())
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return bytes_messages

    def _validate_message_lengths(self, messages: list[bytes]) -> int:
        """Validate all messages have same length and return that length."""
        lengths = [len(m) for m in messages]
        if len(set(lengths)) > 1:
            raise ValueError(f"Messages have varying lengths: {set(lengths)}")
        return lengths[0]

    def _infer_fields_from_boundaries(
        self,
        boundaries: list[int],
        msg_arrays: list[NDArray[np.uint8]],
        msg_len: int,
        min_confidence: float,
    ) -> list[InferredField]:
        """Infer fields from detected boundaries."""
        fields: list[InferredField] = []
        for i in range(len(boundaries)):
            offset = boundaries[i]
            size = boundaries[i + 1] - offset if i < len(boundaries) - 1 else msg_len - offset
            field_data = self._extract_field_data(msg_arrays, offset, size)
            field_obj = self._classify_field_ensemble(
                field_data, offset, size, msg_len, min_confidence
            )
            if field_obj is not None:
                field_obj.name = f"field_{len(fields)}"
                fields.append(field_obj)
        return fields

    def _classify_field_ensemble(
        self,
        field_data: dict[str, Any],
        offset: int,
        size: int,
        msg_len: int,
        min_confidence: float,
    ) -> InferredField | None:
        """Classify field using ensemble of detectors."""
        entropy_type, entropy_conf = self._detect_type_entropy(field_data)
        pattern_type, pattern_conf = self._detect_type_patterns(field_data, offset, size, msg_len)
        stats_type, stats_conf = self._detect_type_statistics(field_data)
        field_type, confidence, evidence = self._vote_field_type(
            [
                (entropy_type, entropy_conf),
                (pattern_type, pattern_conf),
                (stats_type, stats_conf),
            ]
        )
        if confidence < min_confidence:
            return None
        return InferredField(
            name="",  # Will be set by caller
            offset=offset,
            size=size,
            field_type=field_type,  # type: ignore[arg-type]
            entropy=float(field_data["entropy"]),
            variance=float(field_data["variance"]),
            confidence=confidence,
            values_seen=field_data["values"][:5],
            evidence=evidence,
        )

    def _find_special_fields(
        self, fields: list[InferredField]
    ) -> tuple[InferredField | None, InferredField | None]:
        """Find checksum and length fields."""
        checksum_field = None
        length_field = None
        for f in fields:
            if f.field_type == "checksum":
                checksum_field = f
            elif f.field_type == "length":
                length_field = f
        return checksum_field, length_field

    def _extract_field_data(
        self, messages: list[NDArray[np.uint8]], offset: int, size: int
    ) -> dict[str, Any]:
        """Extract field data for type detection.

        Args:
            messages: List of message arrays
            offset: Field offset
            size: Field size

        Returns:
            Dictionary with field values and statistics
        """
        values: list[int | tuple[int, ...]]
        if size <= 4:
            # Use integer representation for small fields
            int_values: list[int] = []
            for msg in messages:
                if size == 1:
                    val_int = int(msg[offset])
                elif size == 2:
                    val_int = int(msg[offset]) << 8 | int(msg[offset + 1])
                elif size == 4:
                    val_int = (
                        int(msg[offset]) << 24
                        | int(msg[offset + 1]) << 16
                        | int(msg[offset + 2]) << 8
                        | int(msg[offset + 3])
                    )
                else:  # size == 3
                    val_int = (
                        int(msg[offset]) << 16 | int(msg[offset + 1]) << 8 | int(msg[offset + 2])
                    )
                int_values.append(val_int)
            values = list(int_values)
        else:
            # For larger fields, use bytes
            tuple_values: list[tuple[int, ...]] = []
            for msg in messages:
                val_tuple = tuple(int(b) for b in msg[offset : offset + size])
                tuple_values.append(val_tuple)
            values = list(tuple_values)

        # Calculate statistics
        if size > 4:
            # Bytes field - calculate entropy across all bytes
            all_bytes_list: list[int] = []
            for v in values:
                if isinstance(v, tuple):
                    all_bytes_list.extend(v)
            all_bytes = np.array(all_bytes_list, dtype=np.uint8)
            entropy = self._calculate_entropy(all_bytes)
            variance = float(np.var(all_bytes))
        else:
            entropy = self._calculate_entropy(np.array(values, dtype=np.int64))
            variance = float(np.var(values))

        return {
            "values": values,
            "offset": offset,
            "size": size,
            "entropy": entropy,
            "variance": variance,
        }

    def _detect_type_entropy(self, field_data: dict[str, Any]) -> tuple[str, float]:
        """Detect field type using entropy analysis.

        : Entropy-based field type detection.

        Args:
            field_data: Field data dictionary

        Returns:
            Tuple of (field_type, confidence)
        """
        entropy = field_data["entropy"]
        values = field_data["values"]

        # Check if all values are identical (constant)
        if len(set(values)) == 1:
            return ("constant", 1.0)

        # Low entropy suggests constant or semi-constant
        if entropy < 1.0:
            return ("constant", 0.8)
        # Very high entropy suggests random data
        elif entropy > 7.0:
            return ("data", 0.7)
        # Medium entropy could be various types
        else:
            return ("unknown", 0.3)

    def _detect_type_patterns(
        self, field_data: dict[str, Any], offset: int, size: int, msg_len: int
    ) -> tuple[str, float]:
        """Detect field type using pattern matching.

        Detects:
        - Counters (incrementing values)
        - Timestamps (steady large increments)
        - Lengths (correlates with message size)
        - Checksums (end of message)
        - Constants (no variation)

        Args:
            field_data: Field data dictionary with 'values' key.
            offset: Field offset in bytes.
            size: Field size in bytes.
            msg_len: Total message length in bytes.

        Returns:
            Tuple of (field_type, confidence_score).
        """
        values = field_data["values"]

        # Skip tuple values (byte arrays)
        if isinstance(values[0], tuple):
            return ("unknown", 0.3)

        # Extract integer values
        int_values = [v for v in values if isinstance(v, int)]
        if not int_values:
            return ("unknown", 0.3)

        # Check for counter pattern
        if self._detect_counter_field(int_values):
            return ("counter", 0.9)

        # Check for timestamp pattern
        timestamp_result = self._check_timestamp_pattern(int_values)
        if timestamp_result is not None:
            return timestamp_result

        # Check for length field
        if self._is_likely_length_field(offset, size, msg_len, int_values):
            return ("length", 0.6)

        # Check for checksum field
        if self._is_likely_checksum_field(offset, size, msg_len):
            return ("checksum", 0.5)

        return ("unknown", 0.3)

    def _check_timestamp_pattern(self, values: list[int]) -> tuple[str, float] | None:
        """Check if values follow timestamp pattern.

        Args:
            values: Integer values.

        Returns:
            ('timestamp', confidence) or None if not a timestamp.
        """
        if len(values) < 3:
            return None

        # Calculate differences between consecutive values
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        positive_diffs = [d for d in diffs if d > 0]

        # Need mostly increasing values
        if len(positive_diffs) < len(diffs) * 0.7:
            return None

        # Check for large increments (typical of timestamps)
        avg_diff = sum(positive_diffs) / len(positive_diffs)
        if avg_diff > 100:
            return ("timestamp", 0.7)

        return None

    def _is_likely_length_field(
        self, offset: int, size: int, msg_len: int, values: list[int]
    ) -> bool:
        """Check if field is likely a length field.

        Args:
            offset: Field offset.
            size: Field size.
            msg_len: Message length.
            values: Integer values.

        Returns:
            True if likely a length field.
        """
        # Length fields are typically near message start and small
        if offset >= 8 or size > 2:
            return False

        # Values should be reasonable for message lengths
        max_val = max(values)
        return max_val < msg_len * 2

    def _is_likely_checksum_field(self, offset: int, size: int, msg_len: int) -> bool:
        """Check if field is likely a checksum.

        Args:
            offset: Field offset.
            size: Field size.
            msg_len: Message length.

        Returns:
            True if likely a checksum field.
        """
        # Checksums are typically near the end but not the whole message
        return offset + size >= msg_len - 4 and offset > 0

    def _detect_type_statistics(self, field_data: dict[str, Any]) -> tuple[str, float]:
        """Detect field type using statistical properties.

        : Statistics-based field type detection.

        Args:
            field_data: Field data dictionary

        Returns:
            Tuple of (field_type, confidence)
        """
        variance = field_data["variance"]
        entropy = field_data["entropy"]
        values = field_data["values"]

        # Check if all values identical (truly constant)
        if len(set(values)) == 1:
            return ("constant", 0.9)
        # Very low variance suggests constant
        elif variance < 10:
            return ("constant", 0.7)
        # High entropy and variance suggests data
        elif entropy > 6.0 and variance > 1000:
            return ("data", 0.6)
        else:
            return ("unknown", 0.4)

    def _vote_field_type(
        self, detections: list[tuple[str, float]]
    ) -> tuple[str, float, dict[str, bool]]:
        """Vote on field type from multiple detectors.

        : Voting mechanism for field type.

        Args:
            detections: List of (field_type, confidence) tuples from detectors

        Returns:
            Tuple of (field_type, confidence, evidence_dict)
        """
        # Weight votes by confidence
        votes: dict[str, float] = {}
        evidence: dict[str, bool] = {}

        detector_names = ["entropy", "patterns", "statistics"]

        for i, (field_type, confidence) in enumerate(detections):
            detector_name = detector_names[i] if i < len(detector_names) else f"detector_{i}"

            if field_type not in votes:
                votes[field_type] = 0.0

            votes[field_type] += confidence

            # Record evidence
            evidence[f"{detector_name}_voted_{field_type}"] = True

        # Find type with highest vote
        if not votes:
            return ("unknown", 0.0, evidence)

        best_type = max(votes.items(), key=lambda x: x[1])
        field_type = best_type[0]
        total_confidence = best_type[1]

        # Calculate total possible votes
        total_possible = sum(conf for _, conf in detections)

        # Normalize confidence as fraction of total possible votes
        if total_possible > 0:
            normalized_confidence = total_confidence / total_possible
        else:
            normalized_confidence = 0.0

        return (field_type, normalized_confidence, evidence)

    def detect_field_types(
        self, messages: list[NDArray[np.uint8]], boundaries: list[int]
    ) -> list[InferredField]:
        """Classify field types based on value patterns.

        : Field type classification.

        Args:
            messages: List of message arrays
            boundaries: Field boundary offsets

        Returns:
            List of InferredField objects
        """
        fields = []

        for i in range(len(boundaries)):
            offset = boundaries[i]

            # Determine field size
            if i < len(boundaries) - 1:
                size = boundaries[i + 1] - offset
            else:
                size = len(messages[0]) - offset

            # Extract field values
            values: list[int | tuple[int, ...]]
            if size <= 4:
                # Use integer representation for small fields
                int_values: list[int] = []
                for msg in messages:
                    if size == 1:
                        val_int = int(msg[offset])
                    elif size == 2:
                        val_int = int(msg[offset]) << 8 | int(msg[offset + 1])
                    elif size == 4:
                        val_int = (
                            int(msg[offset]) << 24
                            | int(msg[offset + 1]) << 16
                            | int(msg[offset + 2]) << 8
                            | int(msg[offset + 3])
                        )
                    else:  # size == 3
                        val_int = (
                            int(msg[offset]) << 16
                            | int(msg[offset + 1]) << 8
                            | int(msg[offset + 2])
                        )
                    int_values.append(val_int)
                values = list(int_values)
            else:
                # For larger fields, use bytes
                tuple_values: list[tuple[int, ...]] = []
                for msg in messages:
                    val_tuple = tuple(int(b) for b in msg[offset : offset + size])
                    tuple_values.append(val_tuple)
                values = list(tuple_values)

            # Calculate statistics
            if size > 4:
                # Bytes field - calculate entropy across all bytes
                all_bytes_list: list[int] = []
                for v in values:
                    if isinstance(v, tuple):
                        all_bytes_list.extend(v)
                all_bytes = np.array(all_bytes_list, dtype=np.uint8)
                entropy = self._calculate_entropy(all_bytes)
                variance = float(np.var(all_bytes))
            else:
                entropy = self._calculate_entropy(np.array(values, dtype=np.int64))
                variance = float(np.var(values))

            # Classify field type
            field_type, confidence = self._classify_field(values, offset, size, messages)

            # Sample values (first 5)
            sample_values = values[:5]

            field_obj = InferredField(
                name=f"field_{i}",
                offset=offset,
                size=size,
                field_type=field_type,  # type: ignore[arg-type]
                entropy=float(entropy),
                variance=float(variance),
                confidence=confidence,
                values_seen=sample_values,
            )

            fields.append(field_obj)

        return fields

    def find_dependencies(
        self, messages: list[NDArray[np.uint8]], schema: MessageSchema
    ) -> dict[str, str]:
        """Find dependencies between fields (e.g., length->payload).

        : Field dependency detection.

        Args:
            messages: List of message arrays
            schema: Inferred message schema

        Returns:
            Dictionary mapping field names to dependency descriptions
        """
        dependencies = {}

        # Check for length field dependencies
        for field in schema.fields:
            if field.field_type == "length":
                # Check if any field size correlates with this length value
                for msg in messages:
                    _length_val = self._extract_field_value(msg, field)
                    # Look for fields that might be variable length
                    # This is a simplified check
                dependencies[field.name] = "Potential length indicator"

        return dependencies

    def _calculate_byte_entropy(self, messages: list[NDArray[np.uint8]], offset: int) -> float:
        """Calculate entropy at byte offset across messages.

        : Entropy calculation for boundary detection.

        Args:
            messages: List of message arrays
            offset: Byte offset to analyze

        Returns:
            Shannon entropy in bits
        """
        values = [msg[offset] for msg in messages]
        return float(self._calculate_entropy(np.array(values)))

    def _calculate_entropy(self, values: NDArray[np.int_ | np.uint8]) -> float:
        """Calculate Shannon entropy of values.

        Args:
            values: Array of values

        Returns:
            Entropy in bits
        """
        if len(values) == 0:
            return 0.0

        # Count frequencies
        _unique, counts = np.unique(values, return_counts=True)
        probabilities = counts / len(values)

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)

    def _classify_field(
        self,
        values: list[int | tuple[int, ...]],
        offset: int,
        size: int,
        messages: list[NDArray[np.uint8]],
    ) -> tuple[str, float]:
        """Classify field type based on patterns.

        Args:
            values: Field values across all messages.
            offset: Field offset in bytes.
            size: Field size in bytes.
            messages: Original message byte arrays.

        Returns:
            Tuple of (field_type, confidence_score).

        Example:
            >>> field_type, confidence = analyzer._classify_field(values, 0, 1, messages)
        """
        # Handle byte array fields (tuples)
        if isinstance(values[0], tuple):
            return self._classify_byte_array_field(values)

        # Handle scalar fields
        return self._classify_scalar_field(values, offset, size, messages)

    def _classify_byte_array_field(self, values: list[int | tuple[int, ...]]) -> tuple[str, float]:
        """Classify byte array field (multi-byte fields).

        Args:
            values: List of byte tuples.

        Returns:
            Tuple of (field_type, confidence).
        """
        # Check for constant byte arrays
        if len(set(values)) == 1:
            return ("constant", 1.0)

        # Calculate entropy across all bytes
        all_bytes = np.concatenate([np.array(v) for v in values])
        entropy = self._calculate_entropy(all_bytes)

        if entropy < 1.0:
            return ("constant", 0.9)
        elif entropy > 7.0:
            return ("data", 0.6)
        else:
            return ("data", 0.5)

    def _classify_scalar_field(
        self,
        values: list[int | tuple[int, ...]],
        offset: int,
        size: int,
        messages: list[NDArray[np.uint8]],
    ) -> tuple[str, float]:
        """Classify scalar (integer) field.

        Args:
            values: List of integer values.
            offset: Field offset.
            size: Field size.
            messages: Original messages.

        Returns:
            Tuple of (field_type, confidence).
        """
        # Check for constant values
        if len(set(values)) == 1:
            return ("constant", 1.0)

        # Check for counter pattern
        int_values = [v for v in values if isinstance(v, int)]
        if self._detect_counter_field(int_values):
            return ("counter", 0.9)

        # Check for checksum (near end of message)
        msg_len = len(messages[0])
        if offset + size >= msg_len - 4:
            if self._detect_checksum_field(messages, offset, size):
                return ("checksum", 0.8)

        # Check for length field (early, small values)
        if offset < 8 and size <= 2:
            max_val = max(int_values)
            if max_val < msg_len * 2:
                return ("length", 0.6)

        # Classify by variance and entropy
        return self._classify_by_statistics(values)

    def _classify_by_statistics(self, values: list[int | tuple[int, ...]]) -> tuple[str, float]:
        """Classify field by statistical properties.

        Args:
            values: Field values.

        Returns:
            Tuple of (field_type, confidence).
        """
        variance = np.var(values)
        entropy = self._calculate_entropy(np.array(values))

        if variance < 10:
            return ("constant", 0.6)
        elif entropy > 6.0:
            return ("data", 0.7)
        else:
            return ("unknown", 0.5)

    def _detect_counter_field(self, values: list[int]) -> bool:
        """Check if values form a counter sequence.

        : Counter field detection.

        Args:
            values: List of integer values

        Returns:
            True if values appear to be a counter
        """
        if len(values) < 3:
            return False

        # Check for monotonic increase
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        # Allow wrapping
        diffs_filtered = [d for d in diffs if d >= 0]

        # Check if most differences are 1 (counter increments)
        if len(diffs_filtered) < len(diffs) * 0.7:
            return False

        ones = sum(1 for d in diffs_filtered if d == 1)
        return ones >= len(diffs_filtered) * 0.7

    def _detect_checksum_field(
        self, messages: list[NDArray[np.uint8]], field_offset: int, field_size: int
    ) -> bool:
        """Check if field is likely a checksum.

        : Checksum field detection.

        Args:
            messages: List of message arrays
            field_offset: Offset of potential checksum field
            field_size: Size of potential checksum field

        Returns:
            True if field appears to be a checksum
        """
        if field_size not in [1, 2, 4]:
            return False

        # Try simple XOR checksum
        for msg in messages[: min(5, len(messages))]:
            # Calculate XOR of all bytes before checksum
            xor_sum = 0
            for i in range(field_offset):
                xor_sum ^= int(msg[i])

            # Extract checksum value
            if field_size == 1:
                checksum = int(msg[field_offset])
            elif field_size == 2:
                checksum = int(msg[field_offset]) << 8 | int(msg[field_offset + 1])
            else:
                checksum = (
                    int(msg[field_offset]) << 24
                    | int(msg[field_offset + 1]) << 16
                    | int(msg[field_offset + 2]) << 8
                    | int(msg[field_offset + 3])
                )

            # For single-byte, compare
            if field_size == 1 and (xor_sum & 0xFF) == checksum:
                continue
            else:
                return False  # Not a match

        return True  # All matched

    def _estimate_header_size(self, fields: list[InferredField]) -> int:
        """Estimate header size from field patterns.

        Args:
            fields: List of inferred fields

        Returns:
            Estimated header size in bytes
        """
        # Look for transition from low-entropy to high-entropy
        for i, field in enumerate(fields):
            if field.field_type == "data" and field.entropy > 6.0:
                if i > 0:
                    return field.offset

        # Default: first 4 fields or 16 bytes
        if len(fields) >= 5:
            # Header includes first 4 fields, so return offset of 5th field
            return fields[4].offset
        elif len(fields) >= 4:
            # If exactly 4 fields, header is up to end of 4th field
            return fields[3].offset + fields[3].size
        elif fields:
            # Fewer than 4 fields - use offset of last field
            return min(16, fields[-1].offset)
        else:
            return 16

    def _extract_field_value(self, msg: NDArray[np.uint8], field: InferredField) -> int:
        """Extract field value from message.

        Args:
            msg: Message array
            field: Field definition

        Returns:
            Field value as integer
        """
        if field.size == 1:
            return int(msg[field.offset])
        elif field.size == 2:
            return int(msg[field.offset]) << 8 | int(msg[field.offset + 1])
        elif field.size == 4:
            return (
                int(msg[field.offset]) << 24
                | int(msg[field.offset + 1]) << 16
                | int(msg[field.offset + 2]) << 8
                | int(msg[field.offset + 3])
            )
        else:
            # Return first byte for larger fields
            return int(msg[field.offset])


def infer_format(messages: list[bytes | NDArray[np.uint8]], min_samples: int = 10) -> MessageSchema:
    """Convenience function for format inference.

    : Top-level API for message format inference.

    Args:
        messages: List of message samples (bytes or np.ndarray)
        min_samples: Minimum required samples

    Returns:
        MessageSchema with inferred structure
    """
    inferrer = MessageFormatInferrer(min_samples=min_samples)
    return inferrer.infer_format(messages)


def detect_field_types(
    messages: list[bytes | NDArray[np.uint8]] | bytes | NDArray[np.uint8],
    boundaries: list[int] | None = None,
) -> list[InferredField]:
    """Detect field types at boundaries.

    : Field type detection.

    Args:
        messages: List of message samples OR a single message
        boundaries: Field boundary offsets (auto-detected if not provided)

    Returns:
        List of InferredField objects

    Raises:
        ValueError: If message type is invalid.
    """
    inferrer = MessageFormatInferrer()

    # Handle single message case - convert to list
    if isinstance(messages, (bytes, np.ndarray)):
        messages_list: list[bytes | NDArray[np.uint8]] = [messages]
    else:
        messages_list = messages

    # Convert to arrays
    msg_arrays = []
    for msg in messages_list:
        if isinstance(msg, bytes):
            msg_arrays.append(np.frombuffer(msg, dtype=np.uint8))
        elif isinstance(msg, np.ndarray):
            msg_arrays.append(msg.astype(np.uint8))
        else:
            raise ValueError(f"Invalid message type: {type(msg)}")

    # Auto-detect boundaries if not provided
    if boundaries is None:
        boundaries = inferrer.detect_field_boundaries(msg_arrays, method="combined")

    return inferrer.detect_field_types(msg_arrays, boundaries)


def find_dependencies(
    messages: list[bytes | NDArray[np.uint8]], schema: MessageSchema | None = None
) -> dict[str, str]:
    """Find field dependencies.

    : Field dependency analysis.

    Args:
        messages: List of message samples
        schema: Message schema (auto-inferred if not provided)

    Returns:
        Dictionary of dependencies

    Raises:
        ValueError: If message type is invalid.
    """
    inferrer = MessageFormatInferrer()

    # Convert to arrays
    msg_arrays = []
    for msg in messages:
        if isinstance(msg, bytes):
            msg_arrays.append(np.frombuffer(msg, dtype=np.uint8))
        elif isinstance(msg, np.ndarray):
            msg_arrays.append(msg.astype(np.uint8))
        else:
            raise ValueError(f"Invalid message type: {type(msg)}")

    # Auto-infer schema if not provided
    if schema is None:
        # Cast to expected type (msg_arrays contains only NDArray after conversion)
        schema = inferrer.infer_format(msg_arrays)  # type: ignore[arg-type]

    return inferrer.find_dependencies(msg_arrays, schema)
