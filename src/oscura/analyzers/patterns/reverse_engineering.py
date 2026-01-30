"""Comprehensive reverse engineering toolkit for binary data and protocols.

This module provides a complete toolkit for reverse engineering unknown binary
protocols and data formats by integrating pattern analysis, entropy analysis,
field inference, and data classification.

Key capabilities:
- Pattern discovery and motif extraction
- N-gram frequency analysis for fingerprinting
- Signature and delimiter discovery
- Binary regex and multi-pattern search
- Fuzzy/approximate matching
- Anomaly detection
- Entropy analysis and crypto detection
- Data type classification (encrypted, compressed, structured)
- Field boundary inference
- Delimiter and length prefix detection
- Checksum field detection

Example workflow:
    >>> from oscura.analyzers.patterns.reverse_engineering import ReverseEngineer
    >>> re_tool = ReverseEngineer()
    >>>
    >>> # Analyze unknown binary data
    >>> analysis = re_tool.analyze_binary(unknown_data)
    >>> print(f"Data type: {analysis['data_type']}")
    >>> print(f"Entropy: {analysis['entropy']:.2f} bits/byte")
    >>> print(f"Detected signatures: {analysis['signatures']}")
    >>>
    >>> # Infer protocol structure
    >>> messages = [msg1, msg2, msg3, ...]
    >>> structure = re_tool.infer_protocol_structure(messages)
    >>> for field in structure['fields']:
    ...     print(f"Field at offset {field['offset']}: {field['type']}")

Author: Oscura Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Import from existing modules
from oscura.analyzers.entropy import CryptoDetector, EntropyResult
from oscura.analyzers.patterns.discovery import (
    CandidateSignature,
    SignatureDiscovery,
)
from oscura.analyzers.patterns.matching import (
    FuzzyMatcher,
)
from oscura.analyzers.patterns.periodic import detect_period
from oscura.analyzers.patterns.sequences import (
    find_repeating_sequences,
)
from oscura.analyzers.statistical.ngrams import (
    NGramAnalyzer,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FieldDescriptor:
    """Descriptor for an inferred protocol field.

    Attributes:
        offset: Byte offset from start of message.
        length: Field length in bytes.
        field_type: Inferred type (fixed, variable, length, checksum, payload, etc.).
        entropy: Average entropy of field across messages.
        is_constant: True if field has same value across all messages.
        constant_value: Value if is_constant is True.
        description: Human-readable description.
    """

    offset: int
    length: int
    field_type: str
    entropy: float
    is_constant: bool = False
    constant_value: bytes | None = None
    description: str = ""


@dataclass
class ProtocolStructure:
    """Inferred protocol message structure.

    Attributes:
        message_length: Fixed message length, or -1 for variable.
        fields: List of inferred fields.
        delimiter: Detected delimiter bytes, if any.
        length_prefix_offset: Offset of length prefix field, if detected.
        checksum_offset: Offset of checksum field, if detected.
        payload_offset: Offset of variable payload, if detected.
        confidence: Overall confidence in structure inference (0.0-1.0).
    """

    message_length: int
    fields: list[FieldDescriptor] = field(default_factory=list)
    delimiter: bytes | None = None
    length_prefix_offset: int | None = None
    checksum_offset: int | None = None
    payload_offset: int | None = None
    confidence: float = 0.0


@dataclass
class BinaryAnalysisResult:
    """Complete analysis result for binary data.

    Attributes:
        data_type: Classified type (encrypted, compressed, structured, random).
        entropy: Shannon entropy in bits/byte.
        entropy_result: Detailed entropy analysis.
        signatures: Discovered candidate signatures.
        repeating_patterns: Detected repeating sequences.
        ngram_profile: N-gram frequency distribution.
        anomalies: Detected anomaly positions.
        periodic_patterns: Detected periodic patterns.
        confidence: Overall analysis confidence (0.0-1.0).
    """

    data_type: str
    entropy: float
    entropy_result: EntropyResult
    signatures: list[CandidateSignature]
    repeating_patterns: list[dict[str, Any]]
    ngram_profile: dict[bytes, int]
    anomalies: list[int]
    periodic_patterns: list[dict[str, Any]]
    confidence: float


class ReverseEngineer:
    """Comprehensive reverse engineering toolkit for binary data and protocols.

    Integrates multiple analysis techniques to help reverse engineer unknown
    binary formats and protocols:

    - Pattern analysis: motifs, repeating sequences, signatures
    - Entropy analysis: crypto detection, compression detection
    - N-gram analysis: fingerprinting, frequency analysis
    - Field inference: automatic field boundary detection
    - Structure learning: protocol message structure inference

    Example:
        >>> re_tool = ReverseEngineer()
        >>>
        >>> # Quick binary analysis
        >>> result = re_tool.analyze_binary(data)
        >>> print(result.data_type)
        'encrypted'
        >>>
        >>> # Protocol structure inference
        >>> messages = [capture1, capture2, capture3]
        >>> structure = re_tool.infer_protocol_structure(messages)
        >>> for field in structure.fields:
        ...     print(f"{field.field_type} at offset {field.offset}")
    """

    def __init__(
        self,
        min_signature_length: int = 4,
        max_signature_length: int = 16,
        ngram_size: int = 2,
    ):
        """Initialize reverse engineering toolkit.

        Args:
            min_signature_length: Minimum signature length for discovery.
            max_signature_length: Maximum signature length for discovery.
            ngram_size: Default n-gram size for frequency analysis.
        """
        self.crypto_detector = CryptoDetector()
        self.signature_discovery = SignatureDiscovery(
            min_length=min_signature_length,
            max_length=max_signature_length,
        )
        self.ngram_analyzer = NGramAnalyzer(n=ngram_size)
        self.fuzzy_matcher = FuzzyMatcher(max_edit_distance=2)

    def analyze_binary(
        self,
        data: bytes,
        detect_anomalies: bool = True,
        detect_signatures: bool = True,
    ) -> BinaryAnalysisResult:
        """Perform comprehensive analysis of binary data.

        Combines multiple analysis techniques to characterize binary data:
        - Entropy analysis and crypto/compression detection
        - Signature and header discovery
        - Repeating pattern detection
        - N-gram frequency profiling
        - Anomaly detection
        - Periodic pattern detection

        Args:
            data: Binary data to analyze.
            detect_anomalies: Whether to run anomaly detection.
            detect_signatures: Whether to run signature discovery.

        Returns:
            BinaryAnalysisResult with comprehensive analysis.

        Raises:
            ValueError: If data is empty.

        Example:
            >>> data = open('unknown.bin', 'rb').read()
            >>> result = re_tool.analyze_binary(data)
            >>> if result.entropy > 7.5:
            ...     print("Likely encrypted")
            >>> for sig in result.signatures:
            ...     print(f"Signature: {sig.pattern.hex()}")
        """
        if not data:
            raise ValueError("Cannot analyze empty data")

        logger.info(f"Starting comprehensive analysis of {len(data)} bytes")

        # 1. Entropy analysis
        entropy_result = self.crypto_detector.analyze_entropy(data)
        entropy_val = entropy_result.shannon_entropy

        # 2. Classify data type
        if entropy_result.encryption_likelihood > 0.7:
            data_type = "encrypted"
        elif entropy_result.compression_likelihood > 0.7:
            data_type = "compressed"
        elif entropy_val < 3.0:
            data_type = "structured"
        else:
            data_type = "mixed"

        # 3. Signature discovery (skip for encrypted/compressed)
        signatures = []
        if detect_signatures and data_type in ["structured", "mixed"]:
            try:
                signatures = self.signature_discovery.discover_signatures(data)
            except Exception as e:
                logger.warning(f"Signature discovery failed: {e}")

        # 4. Repeating pattern detection
        repeating = []
        try:
            sequences = find_repeating_sequences(data, min_length=4, min_count=3)
            repeating = [
                {
                    "pattern": seq.pattern.hex(),
                    "length": seq.length,
                    "count": seq.count,
                    "frequency": seq.frequency,
                }
                for seq in sequences[:10]  # Top 10
            ]
        except Exception as e:
            logger.warning(f"Repeating pattern detection failed: {e}")

        # 5. N-gram profiling
        ngram_profile = {}
        try:
            ngram_profile = self.ngram_analyzer.analyze(data)
        except Exception as e:
            logger.warning(f"N-gram analysis failed: {e}")

        # 6. Anomaly detection (simple z-score based)
        anomalies = []
        if detect_anomalies:
            try:
                byte_array = np.frombuffer(data, dtype=np.uint8)
                # Simple z-score anomaly detection
                mean = np.mean(byte_array)
                std = np.std(byte_array)
                if std > 0:
                    z_scores = np.abs((byte_array - mean) / std)
                    anomalies = np.where(z_scores > 3.0)[0].tolist()
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        # 7. Periodic pattern detection
        periodic = []
        try:
            byte_data_uint8 = np.frombuffer(data, dtype=np.uint8)
            byte_data_float: NDArray[np.float64] = byte_data_uint8.astype(np.float64)
            period_result = detect_period(byte_data_float)
            if period_result is not None:
                periodic.append(
                    {
                        "period": period_result.period,
                        "confidence": period_result.confidence,
                        "method": period_result.method,
                    }
                )
        except Exception as e:
            logger.warning(f"Period detection failed: {e}")

        # Calculate overall confidence
        confidence = self._calculate_analysis_confidence(
            entropy_result, signatures, repeating, anomalies
        )

        logger.info(
            f"Analysis complete: type={data_type}, "
            f"entropy={entropy_val:.2f}, "
            f"signatures={len(signatures)}, "
            f"confidence={confidence:.2f}"
        )

        return BinaryAnalysisResult(
            data_type=data_type,
            entropy=entropy_val,
            entropy_result=entropy_result,
            signatures=signatures,
            repeating_patterns=repeating,
            ngram_profile=ngram_profile,
            anomalies=anomalies,
            periodic_patterns=periodic,
            confidence=confidence,
        )

    def infer_protocol_structure(
        self,
        messages: list[bytes],
        min_field_size: int = 1,
    ) -> ProtocolStructure:
        """Infer protocol message structure from multiple captures.

        Analyzes a collection of protocol messages to automatically infer:
        - Fixed vs variable length messages
        - Field boundaries and types
        - Header/delimiter bytes
        - Length prefix fields
        - Checksum/CRC fields
        - Encrypted payload regions

        Args:
            messages: List of captured protocol messages.
            min_field_size: Minimum field size to detect.

        Returns:
            ProtocolStructure with inferred fields and metadata.

        Raises:
            ValueError: If messages list is empty.

        Example:
            >>> # Capture multiple protocol messages
            >>> messages = [msg1, msg2, msg3, ...]
            >>>
            >>> # Infer structure
            >>> structure = re_tool.infer_protocol_structure(messages)
            >>>
            >>> # Print discovered fields
            >>> for field in structure.fields:
            ...     print(f"{field.field_type}: offset={field.offset}, "
            ...           f"length={field.length}, entropy={field.entropy:.2f}")
        """
        if not messages:
            raise ValueError("Cannot infer structure from empty message list")

        logger.info(f"Inferring protocol structure from {len(messages)} messages")

        # 1. Determine message length (fixed or variable)
        lengths = [len(msg) for msg in messages]
        is_fixed_length = len(set(lengths)) == 1
        msg_length = lengths[0] if is_fixed_length else -1

        # 2. Detect delimiter (for variable-length protocols)
        delimiter = None
        if not is_fixed_length:
            delimiter = self._detect_delimiter(messages)

        # 3. Group by length for field inference
        if is_fixed_length:
            groups = {msg_length: messages}
        else:
            groups = {}
            for msg in messages:
                groups.setdefault(len(msg), []).append(msg)

        # 4. Infer fields for each length group
        all_fields = []
        for msg_group in groups.values():
            fields = self._infer_fields(msg_group, min_field_size)
            all_fields.extend(fields)

        # 5. Detect special field types
        length_prefix_offset = self._detect_length_prefix(messages) if not is_fixed_length else None
        checksum_offset = self._detect_checksum_field(messages)

        # 6. Detect encrypted payload regions
        payload_offset = None
        crypto_fields = self.crypto_detector.detect_crypto_fields(messages, min_field_size=8)
        if crypto_fields:
            # Mark crypto fields
            for cf in crypto_fields:
                payload_offset = cf["offset"]
                all_fields.append(
                    FieldDescriptor(
                        offset=cf["offset"],
                        length=cf["length"],
                        field_type="encrypted_payload",
                        entropy=cf["entropy"],
                        is_constant=False,
                        description="High entropy region (likely encrypted)",
                    )
                )

        # 7. Calculate confidence
        confidence = self._calculate_structure_confidence(
            all_fields, is_fixed_length, delimiter is not None
        )

        logger.info(
            f"Structure inference complete: "
            f"fields={len(all_fields)}, "
            f"fixed_length={is_fixed_length}, "
            f"confidence={confidence:.2f}"
        )

        return ProtocolStructure(
            message_length=msg_length,
            fields=all_fields,
            delimiter=delimiter,
            length_prefix_offset=length_prefix_offset,
            checksum_offset=checksum_offset,
            payload_offset=payload_offset,
            confidence=confidence,
        )

    def detect_delimiter(self, messages: list[bytes]) -> bytes | None:
        """Detect message delimiter bytes.

        Finds byte sequences that consistently appear at message boundaries
        across multiple messages.

        Args:
            messages: List of messages to analyze.

        Returns:
            Delimiter bytes if found, None otherwise.

        Example:
            >>> messages = [b'START' + data1 + b'END', b'START' + data2 + b'END']
            >>> delim = re_tool.detect_delimiter(messages)
            >>> print(delim)
            b'END'
        """
        return self._detect_delimiter(messages)

    def infer_fields(self, messages: list[bytes], min_field_size: int = 1) -> list[FieldDescriptor]:
        """Infer field boundaries from message samples.

        Analyzes byte-level entropy and variance across messages to detect
        field boundaries. Fields with constant values, high entropy, or
        distinct variance patterns are identified.

        Args:
            messages: List of messages (must all be same length).
            min_field_size: Minimum field size in bytes.

        Returns:
            List of FieldDescriptor objects.

        Raises:
            ValueError: If messages have different lengths.

        Example:
            >>> messages = [msg1, msg2, msg3]  # Same length
            >>> fields = re_tool.infer_fields(messages)
            >>> for field in fields:
            ...     print(f"Field: {field.field_type} at {field.offset}")
        """
        if not messages:
            return []

        # Validate all messages same length
        msg_len = len(messages[0])
        if not all(len(msg) == msg_len for msg in messages):
            raise ValueError("All messages must have same length for field inference")

        return self._infer_fields(messages, min_field_size)

    def detect_length_prefix(self, messages: list[bytes]) -> int | None:
        """Detect length prefix field in variable-length protocol.

        Searches for a field at the beginning of messages that encodes
        the message length. Common encodings: 1-byte, 2-byte LE/BE, varint.

        Args:
            messages: List of variable-length messages.

        Returns:
            Offset of length prefix field, or None if not detected.

        Example:
            >>> messages = [b'\\x05hello', b'\\x07goodbye']  # Length prefix
            >>> offset = re_tool.detect_length_prefix(messages)
            >>> print(offset)
            0
        """
        return self._detect_length_prefix(messages)

    def detect_checksum_field(self, messages: list[bytes]) -> int | None:
        """Detect checksum/CRC field in protocol messages.

        Attempts to identify fields that contain checksums by testing
        common checksum algorithms (CRC8, CRC16, CRC32, simple sum).

        Args:
            messages: List of messages to analyze.

        Returns:
            Offset of checksum field, or None if not detected.

        Example:
            >>> # Messages with CRC at end
            >>> messages = [msg1, msg2, msg3]
            >>> offset = re_tool.detect_checksum_field(messages)
            >>> if offset:
            ...     print(f"Checksum at offset {offset}")
        """
        return self._detect_checksum_field(messages)

    def classify_data_type(self, data: bytes) -> str:
        """Classify binary data type (encrypted, compressed, structured, random).

        Uses entropy analysis and statistical tests to classify data.

        Args:
            data: Binary data to classify.

        Returns:
            Data type string: 'encrypted', 'compressed', 'structured', or 'random'.

        Example:
            >>> encrypted = os.urandom(256)
            >>> print(re_tool.classify_data_type(encrypted))
            'encrypted'
        """
        if not data:
            return "empty"

        result = self.crypto_detector.analyze_entropy(data)

        if result.encryption_likelihood > 0.7:
            return "encrypted"
        elif result.compression_likelihood > 0.7:
            return "compressed"
        elif result.shannon_entropy < 3.0:
            return "structured"
        else:
            return "mixed"

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _detect_delimiter(self, messages: list[bytes]) -> bytes | None:
        """Detect delimiter by finding common endings."""
        if len(messages) < 2:
            return None

        # Look for common suffixes (last 1-4 bytes)
        for delim_len in range(1, 5):
            candidates: dict[bytes, int] = {}
            for msg in messages:
                if len(msg) >= delim_len:
                    suffix = msg[-delim_len:]
                    candidates[suffix] = candidates.get(suffix, 0) + 1

            # Check if any suffix appears in >80% of messages
            for suffix, count in candidates.items():
                if count / len(messages) > 0.8:
                    return suffix

        return None

    def _infer_fields(self, messages: list[bytes], min_field_size: int) -> list[FieldDescriptor]:
        """Infer field boundaries using entropy and variance analysis."""
        if not messages:
            return []

        msg_len = len(messages[0])
        if msg_len < min_field_size:
            return []

        # Compute positional entropy and variance
        position_entropy = np.zeros(msg_len)
        position_variance = np.zeros(msg_len)

        for pos in range(msg_len):
            values = [msg[pos] for msg in messages]
            position_entropy[pos] = self._shannon_entropy_bytes(bytes(values))
            position_variance[pos] = np.var(values)

        # Identify field boundaries (where entropy/variance changes significantly)
        fields = []
        field_start = 0
        field_type = "unknown"

        # Simple field detection: constant fields (entropy ~0) and variable fields
        for pos in range(1, msg_len):
            # Detect boundary if entropy changes significantly
            if abs(position_entropy[pos] - position_entropy[pos - 1]) > 2.0:
                # Create field
                if pos - field_start >= min_field_size:
                    avg_entropy = float(np.mean(position_entropy[field_start:pos]))
                    is_constant = avg_entropy < 0.1

                    constant_val = None
                    if is_constant:
                        # All messages have same value
                        constant_val = messages[0][field_start:pos]
                        field_type = "constant"
                    elif avg_entropy > 6.0:
                        field_type = "high_entropy"
                    else:
                        field_type = "variable"

                    fields.append(
                        FieldDescriptor(
                            offset=field_start,
                            length=pos - field_start,
                            field_type=field_type,
                            entropy=avg_entropy,
                            is_constant=is_constant,
                            constant_value=constant_val,
                        )
                    )
                    field_start = pos

        # Add final field
        if msg_len - field_start >= min_field_size:
            avg_entropy = float(np.mean(position_entropy[field_start:]))
            is_constant = avg_entropy < 0.1
            constant_val = messages[0][field_start:] if is_constant else None

            fields.append(
                FieldDescriptor(
                    offset=field_start,
                    length=msg_len - field_start,
                    field_type="constant" if is_constant else "variable",
                    entropy=avg_entropy,
                    is_constant=is_constant,
                    constant_value=constant_val,
                )
            )

        return fields

    def _detect_length_prefix(self, messages: list[bytes]) -> int | None:
        """Detect length prefix at start of messages."""
        if len(messages) < 3:
            return None

        # Try 1-byte length at offset 0
        if all(len(msg) > 1 for msg in messages):
            if all(msg[0] == len(msg) for msg in messages):
                return 0

        # Try 2-byte little-endian length at offset 0
        if all(len(msg) > 2 for msg in messages):
            matches = 0
            for msg in messages:
                length_field = int.from_bytes(msg[0:2], byteorder="little")
                if length_field == len(msg):
                    matches += 1
            if matches / len(messages) > 0.8:
                return 0

        return None

    def _detect_checksum_field(self, messages: list[bytes]) -> int | None:
        """Detect checksum field (simplified heuristic)."""
        # This is a simplified version - real implementation would test CRC8/16/32
        # For now, just detect if last 1-4 bytes have high variance (likely checksum)
        if len(messages) < 3:
            return None

        msg_len = len(messages[0])
        if not all(len(msg) == msg_len for msg in messages):
            return None

        # Check last few bytes for high variance
        for checksum_len in [1, 2, 4]:
            if msg_len > checksum_len:
                offset = msg_len - checksum_len
                values = [msg[offset:] for msg in messages]
                unique_values = len(set(values))
                # If almost all unique, likely a checksum
                if unique_values / len(messages) > 0.9:
                    return offset

        return None

    def _shannon_entropy_bytes(self, data: bytes) -> float:
        """Calculate Shannon entropy for byte sequence."""
        if not data:
            return 0.0

        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _calculate_analysis_confidence(
        self,
        entropy_result: EntropyResult,
        signatures: list[CandidateSignature],
        repeating: list[dict[str, Any]],
        anomalies: list[int],
    ) -> float:
        """Calculate overall confidence in binary analysis."""
        # Base confidence from entropy analysis
        confidence = entropy_result.confidence

        # Boost if we found signatures
        if signatures:
            confidence = min(1.0, confidence + 0.1 * len(signatures))

        # Boost if we found repeating patterns
        if repeating:
            confidence = min(1.0, confidence + 0.05 * len(repeating))

        return float(confidence)

    def _calculate_structure_confidence(
        self,
        fields: list[FieldDescriptor],
        is_fixed_length: bool,
        has_delimiter: bool,
    ) -> float:
        """Calculate confidence in protocol structure inference."""
        confidence = 0.5  # Base confidence

        # More confidence if we found fields
        if fields:
            confidence += 0.1 * min(len(fields), 5)

        # More confidence for fixed-length protocols
        if is_fixed_length:
            confidence += 0.2

        # More confidence if delimiter found
        if has_delimiter:
            confidence += 0.1

        return min(1.0, float(confidence))


# Convenience functions for common operations


def search_pattern(
    data: bytes,
    pattern: bytes | str,
    fuzzy: bool = False,
    max_distance: int = 2,
) -> list[int]:
    """Search for pattern in binary data with optional fuzzy matching.

    Args:
        data: Binary data to search.
        pattern: Pattern to search for (bytes or hex string).
        fuzzy: Enable fuzzy/approximate matching.
        max_distance: Maximum edit distance for fuzzy matching.

    Returns:
        List of match positions.

    Example:
        >>> positions = search_pattern(data, b'\\xff\\xfe', fuzzy=False)
        >>> print(f"Found at: {positions}")
    """
    # Convert hex string to bytes
    if isinstance(pattern, str):
        pattern = bytes.fromhex(pattern.replace(" ", ""))

    if fuzzy:
        matcher = FuzzyMatcher(max_edit_distance=max_distance)
        matches = matcher.search(data, pattern)
        return [m.offset for m in matches]
    else:
        # Simple exact search
        positions = []
        for i in range(len(data) - len(pattern) + 1):
            if data[i : i + len(pattern)] == pattern:
                positions.append(i)
        return positions


def shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of binary data.

    Convenience wrapper around CryptoDetector entropy calculation.

    Args:
        data: Binary data to analyze.

    Returns:
        Shannon entropy in bits per byte (0.0-8.0).

    Example:
        >>> entropy = shannon_entropy(b'\\x00' * 100)
        >>> print(f"{entropy:.2f}")
        0.00
        >>> entropy = shannon_entropy(os.urandom(100))
        >>> print(f"{entropy:.2f}")
        7.98
    """
    detector = CryptoDetector()
    return detector._shannon_entropy(data)


def byte_frequency_distribution(data: bytes) -> dict[int, int]:
    """Calculate byte frequency distribution.

    Args:
        data: Binary data to analyze.

    Returns:
        Dictionary mapping byte value (0-255) to count.

    Example:
        >>> freq = byte_frequency_distribution(b'AAABBC')
        >>> print(freq[ord('A')])
        3
    """
    byte_array = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(byte_array, minlength=256)
    return {i: int(count) for i, count in enumerate(counts) if count > 0}


def sliding_entropy(
    data: bytes,
    window_size: int = 256,
    stride: int = 64,
) -> list[tuple[int, float]]:
    """Calculate entropy across sliding windows.

    Wrapper around CryptoDetector.sliding_window_entropy.

    Args:
        data: Binary data to analyze.
        window_size: Window size in bytes.
        stride: Step size between windows.

    Returns:
        List of (offset, entropy) tuples.

    Example:
        >>> windows = sliding_entropy(data, window_size=128)
        >>> for offset, ent in windows:
        ...     if ent > 7.5:
        ...         print(f"High entropy at offset {offset}")
    """
    detector = CryptoDetector()
    return detector.sliding_window_entropy(data, window_size=window_size, stride=stride)


def entropy_profile(data: bytes, window_size: int = 256) -> NDArray[np.float64]:
    """Generate entropy profile (entropy over time).

    Args:
        data: Binary data to analyze.
        window_size: Window size for entropy calculation.

    Returns:
        Array of entropy values for each window position.

    Example:
        >>> profile = entropy_profile(data, window_size=128)
        >>> plt.plot(profile)
        >>> plt.ylabel('Entropy (bits/byte)')
        >>> plt.xlabel('Position')
    """
    detector = CryptoDetector()
    windows = detector.sliding_window_entropy(data, window_size=window_size, stride=1)
    return np.array([ent for _, ent in windows])


def detect_encrypted_regions(
    data: bytes,
    window_size: int = 256,
    threshold: float = 7.5,
) -> list[tuple[int, int]]:
    """Detect regions with high entropy (likely encrypted).

    Args:
        data: Binary data to analyze.
        window_size: Window size for analysis.
        threshold: Entropy threshold for encryption (bits/byte).

    Returns:
        List of (start, end) tuples for encrypted regions.

    Example:
        >>> regions = detect_encrypted_regions(data)
        >>> for start, end in regions:
        ...     print(f"Encrypted: {start}-{end}")
    """
    detector = CryptoDetector()
    windows = detector.sliding_window_entropy(data, window_size=window_size, stride=1)

    regions = []
    in_region = False
    region_start = 0

    for offset, entropy in windows:
        if entropy > threshold:
            if not in_region:
                region_start = offset
                in_region = True
        else:
            if in_region:
                regions.append((region_start, offset))
                in_region = False

    # Close final region
    if in_region:
        regions.append((region_start, len(data)))

    return regions


def detect_compressed_regions(
    data: bytes,
    window_size: int = 256,
) -> list[tuple[int, int]]:
    """Detect regions with medium-high entropy (likely compressed).

    Args:
        data: Binary data to analyze.
        window_size: Window size for analysis.

    Returns:
        List of (start, end) tuples for compressed regions.

    Example:
        >>> regions = detect_compressed_regions(data)
        >>> for start, end in regions:
        ...     print(f"Compressed: {start}-{end}")
    """
    detector = CryptoDetector()
    windows = detector.sliding_window_entropy(data, window_size=window_size, stride=1)

    regions = []
    in_region = False
    region_start = 0

    for offset, entropy in windows:
        # Compressed: 6.5-7.5 bits/byte
        if 6.5 < entropy < 7.5:
            if not in_region:
                region_start = offset
                in_region = True
        else:
            if in_region:
                regions.append((region_start, offset))
                in_region = False

    if in_region:
        regions.append((region_start, len(data)))

    return regions


__all__ = [
    "BinaryAnalysisResult",
    "FieldDescriptor",
    "ProtocolStructure",
    "ReverseEngineer",
    # Convenience functions
    "byte_frequency_distribution",
    "detect_compressed_regions",
    "detect_encrypted_regions",
    "entropy_profile",
    "search_pattern",
    "shannon_entropy",
    "sliding_entropy",
]
