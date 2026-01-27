"""Entropy analysis and cryptographic data detection.

This module provides tools for detecting encrypted, compressed, or random data
in protocol messages and binary streams using entropy analysis and statistical tests.

Key capabilities:
- Shannon entropy calculation (information content measurement)
- Chi-squared test for uniform distribution (randomness detection)
- Sliding window entropy analysis (find encrypted regions)
- Crypto field detection across multiple messages
- Compression vs encryption distinction

Typical use cases:
- Identify encrypted payload fields in unknown protocols
- Detect compression in protocol messages
- Find random/high-entropy regions in binary data
- Distinguish structured vs random data

Example:
    >>> from oscura.analyzers.entropy import CryptoDetector
    >>> detector = CryptoDetector()
    >>> result = detector.analyze_entropy(data)
    >>> if result.is_high_entropy:
    ...     print(f"Likely encrypted: {result.encryption_likelihood:.2%}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chisquare

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Results from entropy analysis.

    Attributes:
        shannon_entropy: Shannon entropy in bits per byte (0.0-8.0).
            Higher values indicate more randomness/information content.
        is_high_entropy: True if entropy exceeds threshold for encryption/compression.
        is_random: True if chi-squared test indicates uniform distribution.
        compression_likelihood: Probability data is compressed (0.0-1.0).
        encryption_likelihood: Probability data is encrypted (0.0-1.0).
        confidence: Overall confidence score for classification (0.0-1.0).
            Higher for larger samples with clear characteristics.
        chi_squared_p_value: P-value from chi-squared test (high = random).

    Example:
        >>> result = detector.analyze_entropy(encrypted_data)
        >>> print(f"Entropy: {result.shannon_entropy:.2f} bits/byte")
        >>> print(f"Encrypted: {result.encryption_likelihood:.2%}")
        >>> print(f"Confidence: {result.confidence:.2%}")
    """

    shannon_entropy: float
    is_high_entropy: bool
    is_random: bool
    compression_likelihood: float
    encryption_likelihood: float
    confidence: float
    chi_squared_p_value: float


class CryptoDetector:
    """Detect encrypted, compressed, or random data using entropy analysis.

    This class provides multiple analysis methods for identifying cryptographic
    or compressed data in binary streams. It uses Shannon entropy, chi-squared
    tests, and heuristics to distinguish between different data types.

    Thresholds:
        ENTROPY_THRESHOLD_ENCRYPTED: 7.5 bits/byte - typical for AES/ChaCha20
        ENTROPY_THRESHOLD_COMPRESSED: 6.5 bits/byte - typical for gzip/zlib
        CHI_SQUARED_ALPHA: 0.05 - significance level for randomness test

    Example:
        >>> detector = CryptoDetector()
        >>> # Analyze single message
        >>> result = detector.analyze_entropy(message)
        >>> # Find encrypted regions in mixed data
        >>> windows = detector.sliding_window_entropy(mixed_data, window_size=256)
        >>> # Detect fields across multiple messages
        >>> fields = detector.detect_crypto_fields(messages, min_field_size=16)
    """

    # Entropy thresholds (bits per byte)
    ENTROPY_THRESHOLD_ENCRYPTED = 7.5  # Strong encryption: AES, ChaCha20
    ENTROPY_THRESHOLD_COMPRESSED = 6.5  # Compression: gzip, zlib, deflate
    ENTROPY_THRESHOLD_STRUCTURED = 3.0  # Structured data: text, low-entropy

    # Statistical test parameters
    CHI_SQUARED_ALPHA = 0.05  # Significance level for randomness test
    MIN_SAMPLE_SIZE = 32  # Minimum bytes for reliable entropy analysis

    def analyze_entropy(self, data: bytes, window_size: int | None = None) -> EntropyResult:
        """Analyze entropy and randomness characteristics of data.

        Performs comprehensive entropy analysis including Shannon entropy,
        chi-squared test for randomness, and classification into
        encrypted/compressed/structured categories.

        Args:
            data: Binary data to analyze.
            window_size: Optional window size for localized analysis.
                If None, analyzes entire data block.

        Returns:
            EntropyResult with entropy metrics and classification.

        Raises:
            ValueError: If data is empty or window_size is invalid.

        Example:
            >>> # Analyze encrypted data
            >>> encrypted = os.urandom(256)
            >>> result = detector.analyze_entropy(encrypted)
            >>> assert result.is_high_entropy
            >>> assert result.encryption_likelihood > 0.9
            >>>
            >>> # Analyze structured data
            >>> text = b"Hello World" * 20
            >>> result = detector.analyze_entropy(text)
            >>> assert not result.is_high_entropy
            >>> assert result.encryption_likelihood < 0.1
        """
        if not data:
            raise ValueError("Cannot analyze empty data")

        if window_size is not None and window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")

        # Use full data if no window specified
        if window_size is None or window_size >= len(data):
            analysis_data = data
        else:
            # Take first window for analysis
            analysis_data = data[:window_size]

        # Calculate Shannon entropy
        shannon_ent = self._shannon_entropy(analysis_data)

        # Perform chi-squared test for randomness
        chi_p_value = self._chi_squared_test(analysis_data)
        is_random = chi_p_value > self.CHI_SQUARED_ALPHA

        # Determine if high entropy
        is_high_entropy = shannon_ent > self.ENTROPY_THRESHOLD_ENCRYPTED

        # Calculate compression vs encryption likelihood
        compression_likelihood = self._estimate_compression_likelihood(
            shannon_ent, is_random, chi_p_value
        )
        encryption_likelihood = self._estimate_encryption_likelihood(
            shannon_ent, is_random, chi_p_value
        )

        # Calculate confidence based on sample size
        confidence = self._calculate_confidence(len(analysis_data), shannon_ent)

        logger.debug(
            f"Entropy analysis: {shannon_ent:.2f} bits/byte, "
            f"chi-squared p={chi_p_value:.4f}, "
            f"encrypted={encryption_likelihood:.2%}, "
            f"compressed={compression_likelihood:.2%}"
        )

        return EntropyResult(
            shannon_entropy=shannon_ent,
            is_high_entropy=is_high_entropy,
            is_random=is_random,
            compression_likelihood=compression_likelihood,
            encryption_likelihood=encryption_likelihood,
            confidence=confidence,
            chi_squared_p_value=chi_p_value,
        )

    def sliding_window_entropy(
        self, data: bytes, window_size: int = 256, stride: int = 64
    ) -> list[tuple[int, float]]:
        """Compute entropy across sliding windows to find regions of interest.

        Useful for protocols with mixed plaintext/ciphertext regions:
        - Header: low entropy, structured fields
        - Payload: high entropy, encrypted data
        - Footer: low entropy, checksums/padding

        Args:
            data: Binary data to analyze.
            window_size: Size of sliding window in bytes. Default 256.
            stride: Step size between windows in bytes. Default 64.

        Returns:
            List of (offset, entropy) tuples for each window.

        Raises:
            ValueError: If data is too short, window_size or stride invalid.

        Example:
            >>> # Analyze message with mixed content
            >>> header = b"PROTOCOL_HEADER"
            >>> payload = os.urandom(200)  # Encrypted
            >>> footer = b"END"
            >>> data = header + payload + footer
            >>>
            >>> windows = detector.sliding_window_entropy(data, window_size=64)
            >>> for offset, ent in windows:
            ...     if ent > 7.5:
            ...         print(f"Encrypted region at offset {offset}")
        """
        if not data:
            raise ValueError("Cannot analyze empty data")

        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")

        if stride < 1:
            raise ValueError(f"stride must be positive, got {stride}")

        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")

        results = []
        for offset in range(0, len(data) - window_size + 1, stride):
            window = data[offset : offset + window_size]
            ent = self._shannon_entropy(window)
            results.append((offset, ent))

        logger.debug(f"Sliding window analysis: {len(results)} windows analyzed")

        return results

    def detect_crypto_fields(
        self, messages: list[bytes], min_field_size: int = 8
    ) -> list[dict[str, Any]]:
        """Identify likely encrypted fields by analyzing multiple messages.

        Strategy:
        1. Group messages by length (same protocol message type)
        2. Compute positional entropy (entropy at each byte offset)
        3. Find consecutive high-entropy regions
        4. Return field descriptors with offset, length, and characteristics

        This works because:
        - Plaintext fields vary between messages (low positional entropy)
        - Encrypted fields appear random (high positional entropy)
        - Field boundaries are consistent within a message type

        Args:
            messages: List of protocol messages to analyze.
            min_field_size: Minimum field size in bytes to report. Default 8.

        Returns:
            List of detected crypto field descriptors, each containing:
                - offset: Field start offset in bytes
                - length: Field length in bytes
                - type: Classification (e.g., 'encrypted_payload')
                - entropy: Average entropy across the field
                - message_length: Length of messages containing this field
                - sample_count: Number of messages analyzed

        Raises:
            ValueError: If messages list is empty or min_field_size invalid.

        Example:
            >>> # Capture multiple protocol messages
            >>> messages = [...]  # List of bytes objects
            >>>
            >>> # Detect encrypted fields
            >>> fields = detector.detect_crypto_fields(messages, min_field_size=16)
            >>>
            >>> for field in fields:
            ...     print(f"Encrypted field at offset {field['offset']}, "
            ...           f"length {field['length']} bytes, "
            ...           f"entropy {field['entropy']:.2f}")
        """
        if not messages:
            raise ValueError("Cannot analyze empty message list")

        if min_field_size < 1:
            raise ValueError(f"min_field_size must be positive, got {min_field_size}")

        # Group messages by length (same message type)
        length_groups: dict[int, list[bytes]] = {}
        for msg in messages:
            length_groups.setdefault(len(msg), []).append(msg)

        crypto_fields = []

        for msg_len, msg_group in length_groups.items():
            if msg_len < min_field_size:
                logger.debug(
                    f"Skipping message group with length {msg_len} < "
                    f"min_field_size {min_field_size}"
                )
                continue

            # Compute entropy at each position
            position_entropy = self._compute_positional_entropy(msg_group)

            # Find high-entropy regions
            fields = self._extract_high_entropy_regions(
                position_entropy,
                msg_len,
                min_field_size,
                len(msg_group),
            )

            crypto_fields.extend(fields)

        logger.info(f"Detected {len(crypto_fields)} crypto fields across {len(messages)} messages")

        return crypto_fields

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy in bits per byte.

        Shannon entropy measures the average information content per byte.
        Formula: H = -sum(p(i) * log2(p(i))) for all byte values i

        Args:
            data: Binary data to analyze.

        Returns:
            Entropy in bits per byte (0.0 to 8.0).
                0.0: All bytes identical (no information)
                8.0: Perfect randomness (maximum information)

        Example:
            >>> # All zeros - no entropy
            >>> assert detector._shannon_entropy(b"\x00" * 100) == 0.0
            >>> # Random data - high entropy
            >>> random_data = os.urandom(1000)
            >>> assert detector._shannon_entropy(random_data) > 7.5
        """
        if not data:
            return 0.0

        # Count byte frequencies
        byte_array = np.frombuffer(data, dtype=np.uint8)
        byte_counts = np.bincount(byte_array, minlength=256)

        # Calculate probabilities (exclude zero counts)
        probabilities = byte_counts[byte_counts > 0] / len(data)

        # Shannon entropy: -sum(p * log2(p))
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        return entropy

    def _chi_squared_test(self, data: bytes) -> float:
        """Perform chi-squared test for uniform distribution.

        Tests the null hypothesis: data is uniformly distributed (random).

        Args:
            data: Binary data to test.

        Returns:
            P-value from chi-squared test.
                High p-value (>0.05): Data is likely random/uniform
                Low p-value (<0.05): Data has structure/patterns

        Example:
            >>> # Random data passes test (high p-value)
            >>> random_data = os.urandom(1000)
            >>> p_value = detector._chi_squared_test(random_data)
            >>> assert p_value > 0.05
            >>>
            >>> # Structured data fails test (low p-value)
            >>> text = b"AAAA" * 100
            >>> p_value = detector._chi_squared_test(text)
            >>> assert p_value < 0.05
        """
        if not data:
            return 0.0

        # Count observed byte frequencies
        byte_array = np.frombuffer(data, dtype=np.uint8)
        observed = np.bincount(byte_array, minlength=256)

        # Expected frequencies for uniform distribution
        expected = np.full(256, len(data) / 256.0)

        # Chi-squared test
        _, p_value = chisquare(observed, expected)

        return float(p_value)

    def _estimate_compression_likelihood(
        self, entropy: float, is_random: bool, chi_p_value: float
    ) -> float:
        """Estimate likelihood that data is compressed.

        Compressed data characteristics:
        - Medium-high entropy (6.5-7.5 bits/byte)
        - Less uniformly random than encryption
        - Some residual structure from compression algorithm

        Args:
            entropy: Shannon entropy in bits/byte.
            is_random: Result of chi-squared test.
            chi_p_value: P-value from chi-squared test.

        Returns:
            Compression likelihood (0.0-1.0).
        """
        if entropy < self.ENTROPY_THRESHOLD_COMPRESSED:
            # Too low entropy for compression
            return 0.0

        if entropy > self.ENTROPY_THRESHOLD_ENCRYPTED:
            # Too high entropy - likely encryption, not compression
            # Compression rarely achieves >7.5 bits/byte
            return max(0.0, 1.0 - (entropy - self.ENTROPY_THRESHOLD_ENCRYPTED) * 2.0)

        # Medium entropy range - likely compression
        # Peak likelihood at 7.0 bits/byte
        compression_score = (entropy - self.ENTROPY_THRESHOLD_COMPRESSED) / (
            self.ENTROPY_THRESHOLD_ENCRYPTED - self.ENTROPY_THRESHOLD_COMPRESSED
        )

        # Reduce likelihood if data is too uniformly random
        # (compression has some structure)
        if is_random and chi_p_value > 0.5:
            compression_score *= 0.5

        return min(1.0, max(0.0, compression_score))

    def _estimate_encryption_likelihood(
        self, entropy: float, is_random: bool, chi_p_value: float
    ) -> float:
        """Estimate likelihood that data is encrypted.

        Encrypted data characteristics:
        - Very high entropy (>7.5 bits/byte)
        - Uniformly random distribution
        - High chi-squared p-value

        Args:
            entropy: Shannon entropy in bits/byte.
            is_random: Result of chi-squared test.
            chi_p_value: P-value from chi-squared test.

        Returns:
            Encryption likelihood (0.0-1.0).
        """
        if entropy < self.ENTROPY_THRESHOLD_COMPRESSED:
            # Too low for encryption
            return 0.0

        # Base score from entropy
        if entropy >= self.ENTROPY_THRESHOLD_ENCRYPTED:
            entropy_score = min(1.0, (entropy - 7.0) / 1.0)  # Scale 7.0-8.0 to 0-1
        else:
            # Partial score for medium entropy
            entropy_score = (entropy - self.ENTROPY_THRESHOLD_COMPRESSED) / (
                self.ENTROPY_THRESHOLD_ENCRYPTED - self.ENTROPY_THRESHOLD_COMPRESSED
            )
            entropy_score *= 0.5  # Reduce confidence

        # Boost score if uniformly random
        if is_random:
            # Very uniform = likely encryption
            randomness_boost = min(0.5, chi_p_value)
            encryption_score = min(1.0, entropy_score + randomness_boost)
        else:
            # Not uniform = less likely encryption
            encryption_score = entropy_score * 0.7

        return min(1.0, max(0.0, encryption_score))

    def _calculate_confidence(self, sample_size: int, entropy: float) -> float:
        """Calculate confidence score based on sample size and entropy clarity.

        Args:
            sample_size: Number of bytes analyzed.
            entropy: Shannon entropy value.

        Returns:
            Confidence score (0.0-1.0).
        """
        # Sample size confidence
        if sample_size < self.MIN_SAMPLE_SIZE:
            size_confidence = sample_size / self.MIN_SAMPLE_SIZE
        elif sample_size >= 256:
            size_confidence = 1.0
        else:
            size_confidence = (
                0.5 + (sample_size - self.MIN_SAMPLE_SIZE) / (256 - self.MIN_SAMPLE_SIZE) * 0.5
            )

        # Entropy clarity (distance from thresholds)
        if entropy < self.ENTROPY_THRESHOLD_STRUCTURED:
            # Clearly structured
            clarity = 1.0
        elif entropy > self.ENTROPY_THRESHOLD_ENCRYPTED:
            # Clearly encrypted
            clarity = 1.0
        elif self.ENTROPY_THRESHOLD_COMPRESSED < entropy < self.ENTROPY_THRESHOLD_ENCRYPTED:
            # Ambiguous range (compression vs encryption)
            clarity = 0.6
        else:
            # Between structured and compressed
            clarity = 0.8

        return size_confidence * clarity

    def _compute_positional_entropy(self, messages: list[bytes]) -> NDArray[np.float64]:
        """Compute entropy at each byte position across messages.

        Args:
            messages: List of messages (all same length).

        Returns:
            Array of entropy values, one per byte position.
        """
        msg_len = len(messages[0])
        position_entropy = np.zeros(msg_len, dtype=np.float64)

        for pos in range(msg_len):
            # Extract bytes at this position from all messages
            position_bytes = bytes([msg[pos] for msg in messages])
            position_entropy[pos] = self._shannon_entropy(position_bytes)

        return position_entropy

    def _extract_high_entropy_regions(
        self,
        position_entropy: NDArray[np.float64],
        msg_len: int,
        min_field_size: int,
        sample_count: int,
    ) -> list[dict[str, Any]]:
        """Extract contiguous high-entropy regions as crypto field candidates.

        Args:
            position_entropy: Entropy at each position.
            msg_len: Message length in bytes.
            min_field_size: Minimum field size to report.
            sample_count: Number of messages analyzed.

        Returns:
            List of field descriptors.
        """
        fields = []
        in_crypto = False
        start = 0

        # Use a lower threshold for positional entropy since we're analyzing
        # entropy across limited samples (e.g., 20 messages = 20 bytes per position)
        # Maximum positional entropy with N samples is log2(min(N, 256))
        max_positional_entropy = min(np.log2(sample_count), 8.0) if sample_count > 1 else 0.0
        # Use 70% of max as threshold for high positional entropy
        positional_threshold = max_positional_entropy * 0.7

        for pos in range(msg_len):
            if position_entropy[pos] > positional_threshold:
                if not in_crypto:
                    start = pos
                    in_crypto = True
            else:
                if in_crypto and (pos - start) >= min_field_size:
                    # Found a crypto field
                    fields.append(
                        {
                            "offset": start,
                            "length": pos - start,
                            "type": "encrypted_payload",
                            "entropy": float(np.mean(position_entropy[start:pos])),
                            "message_length": msg_len,
                            "sample_count": sample_count,
                        }
                    )
                in_crypto = False

        # Handle field at end of message
        if in_crypto and (msg_len - start) >= min_field_size:
            fields.append(
                {
                    "offset": start,
                    "length": msg_len - start,
                    "type": "encrypted_payload",
                    "entropy": float(np.mean(position_entropy[start:])),
                    "message_length": msg_len,
                    "sample_count": sample_count,
                }
            )

        return fields


__all__ = ["CryptoDetector", "EntropyResult"]
