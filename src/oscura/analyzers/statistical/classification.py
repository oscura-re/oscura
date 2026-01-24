"""Statistical data type classification.


This module provides tools for classifying binary data regions as text,
binary, compressed, encrypted, or padding using multiple statistical tests
and heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np

from .entropy import shannon_entropy

# Type alias for input data
DataType = Union[bytes, bytearray, "np.ndarray[Any, Any]"]

# Common compression signatures
COMPRESSION_SIGNATURES = {
    b"\x1f\x8b": "gzip",
    b"BZ": "bzip2",
    b"\x50\x4b\x03\x04": "zip",
    b"\x50\x4b\x05\x06": "zip",
    b"\x50\x4b\x07\x08": "zip",
    b"\xfd7zXZ\x00": "xz",
    b"\x28\xb5\x2f\xfd": "zstd",
    b"\x04\x22\x4d\x18": "lz4",
}

# Common executable/binary signatures
BINARY_SIGNATURES = {
    b"\x7fELF": "elf",  # ELF executable
    b"MZ": "pe",  # Windows PE/DOS executable
    b"\xca\xfe\xba\xbe": "macho_fat",  # Mach-O fat binary
    b"\xfe\xed\xfa\xce": "macho_32",  # Mach-O 32-bit
    b"\xfe\xed\xfa\xcf": "macho_64",  # Mach-O 64-bit
    b"\xcf\xfa\xed\xfe": "macho_64_le",  # Mach-O 64-bit little endian
    b"\xce\xfa\xed\xfe": "macho_32_le",  # Mach-O 32-bit little endian
}


@dataclass
class ClassificationResult:
    """Data type classification result.

    Attributes:
        primary_type: Primary classification category
        confidence: Confidence score for classification (0-1)
        entropy: Shannon entropy value
        printable_ratio: Fraction of printable ASCII characters
        null_ratio: Fraction of null bytes
        byte_variance: Variance of byte values
        details: Additional classification details and metadata
    """

    primary_type: Literal["text", "binary", "compressed", "encrypted", "padding", "mixed"]
    confidence: float
    entropy: float
    printable_ratio: float
    null_ratio: float
    byte_variance: float
    details: dict[str, Any] = field(default_factory=dict)

    # Alias for test compatibility
    @property
    def data_type(self) -> str:
        """Alias for primary_type for test compatibility."""
        return self.primary_type


@dataclass
class RegionClassification:
    """Classification of a data region.

    Attributes:
        start: Start offset in bytes
        end: End offset in bytes (exclusive)
        length: Region length in bytes
        classification: Classification result for this region
    """

    start: int
    end: int
    length: int
    classification: ClassificationResult


def classify_data_type(data: DataType) -> ClassificationResult:
    """Classify binary data type using multiple heuristics.

    : Statistical Data Type Classification

    Uses a combination of entropy analysis, printable character ratio,
    byte distribution, and signature detection to classify data.

    Classification logic:
        1. Check for null/padding (null_ratio > 0.9)
        2. Check for executable/binary signatures
        3. Check for compression signatures
        4. Check for encrypted/random (entropy > 7.5, low structure)
        5. Check for text (high printable ratio, medium entropy)
        6. Default to binary/structured

    Args:
        data: Input data as bytes, bytearray, or numpy array

    Returns:
        ClassificationResult with type and confidence

    Raises:
        ValueError: If data is empty

    Example:
        >>> result = classify_data_type(b'Hello, World!')
        >>> result.primary_type
        'text'
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if not data:
        raise ValueError("Cannot classify empty data")

    # Calculate statistics
    entropy_val = shannon_entropy(data)

    # Printable ASCII: 0x20-0x7E plus tab, newline, carriage return
    printable_count = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    printable_ratio = printable_count / len(data)

    # Null byte ratio
    null_count = sum(1 for b in data if b == 0)
    null_ratio = null_count / len(data)

    # Byte variance
    byte_array = np.frombuffer(data, dtype=np.uint8)
    byte_variance = float(np.var(byte_array))

    details = {}

    # Classification logic
    # 1. Padding/null regions
    if null_ratio > 0.9:
        return ClassificationResult(
            primary_type="padding",
            confidence=min(1.0, null_ratio),
            entropy=entropy_val,
            printable_ratio=printable_ratio,
            null_ratio=null_ratio,
            byte_variance=byte_variance,
            details={"reason": "high_null_ratio"},
        )

    # 2. Check for executable/binary signatures (BEFORE compression and encrypted)
    for sig, bin_type in BINARY_SIGNATURES.items():
        if data[: len(sig)] == sig:
            details["binary_type"] = bin_type
            return ClassificationResult(
                primary_type="binary",
                confidence=0.95,
                entropy=entropy_val,
                printable_ratio=printable_ratio,
                null_ratio=null_ratio,
                byte_variance=byte_variance,
                details=details,
            )

    # 3. Check for compression signatures
    for sig, comp_type in COMPRESSION_SIGNATURES.items():
        if data[: len(sig)] == sig:
            details["compression_type"] = comp_type
            return ClassificationResult(
                primary_type="compressed",
                confidence=0.95,
                entropy=entropy_val,
                printable_ratio=printable_ratio,
                null_ratio=null_ratio,
                byte_variance=byte_variance,
                details=details,
            )

    # 4. Text data (high printable ratio) - check BEFORE entropy-based classification
    if printable_ratio > 0.75 and entropy_val < 6.5:
        confidence = min(1.0, printable_ratio * 0.95)
        details["reason"] = "high_printable_ratio"
        return ClassificationResult(
            primary_type="text",
            confidence=confidence,
            entropy=entropy_val,
            printable_ratio=printable_ratio,
            null_ratio=null_ratio,
            byte_variance=byte_variance,
            details=details,
        )

    # 5. Encrypted/random data (high entropy, no structure)
    if entropy_val > 7.5 and byte_variance > 5000:
        # High entropy with high variance suggests random/encrypted
        confidence = min(1.0, (entropy_val - 7.5) / 0.5 + 0.7)
        details["reason"] = "high_entropy_and_variance"
        return ClassificationResult(
            primary_type="encrypted",
            confidence=confidence,
            entropy=entropy_val,
            printable_ratio=printable_ratio,
            null_ratio=null_ratio,
            byte_variance=byte_variance,
            details=details,
        )

    # 6. Compressed data (high entropy, some structure)
    if 6.5 <= entropy_val <= 7.5:
        confidence = 0.7
        details["reason"] = "compression_entropy_range"
        return ClassificationResult(
            primary_type="compressed",
            confidence=confidence,
            entropy=entropy_val,
            printable_ratio=printable_ratio,
            null_ratio=null_ratio,
            byte_variance=byte_variance,
            details=details,
        )

    # 7. Default to binary/structured
    confidence = 0.6
    details["reason"] = "default_binary"
    return ClassificationResult(
        primary_type="binary",
        confidence=confidence,
        entropy=entropy_val,
        printable_ratio=printable_ratio,
        null_ratio=null_ratio,
        byte_variance=byte_variance,
        details=details,
    )


def detect_text_regions(
    data: DataType, min_length: int = 8, min_printable: float = 0.8
) -> list[RegionClassification]:
    """Detect ASCII/UTF-8 text regions.

    : Statistical Data Type Classification

    Scans for contiguous regions with high printable character ratio.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        min_length: Minimum region length in bytes (default: 8)
        min_printable: Minimum printable ratio to consider text (default: 0.8)

    Returns:
        List of detected text regions

    Example:
        >>> data = b'\\x00' * 100 + b'Hello World' + b'\\x00' * 100
        >>> regions = detect_text_regions(data)
        >>> len(regions) > 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    regions = []
    in_region = False
    region_start = 0
    _printable_in_window = 0
    window_size = min_length

    for i, byte in enumerate(data):
        _is_printable = 32 <= byte <= 126 or byte in (9, 10, 13)

        if not in_region:
            # Look for start of text region
            if i >= window_size - 1:
                # Check window
                window = data[i - window_size + 1 : i + 1]
                printable_count = sum(1 for b in window if 32 <= b <= 126 or b in (9, 10, 13))
                if printable_count / window_size >= min_printable:
                    in_region = True
                    region_start = i - window_size + 1
        else:
            # In text region, look for end
            # Use a sliding window to detect when printable ratio drops
            if i >= region_start + window_size:
                window = data[i - window_size + 1 : i + 1]
                printable_count = sum(1 for b in window if 32 <= b <= 126 or b in (9, 10, 13))
                if printable_count / window_size < min_printable:
                    # End of region
                    region_data = data[region_start : i - window_size + 1]
                    if len(region_data) >= min_length:
                        classification = classify_data_type(region_data)
                        regions.append(
                            RegionClassification(
                                start=region_start,
                                end=i - window_size + 1,
                                length=len(region_data),
                                classification=classification,
                            )
                        )
                    in_region = False

    # Handle region extending to end
    if in_region:
        region_data = data[region_start:]
        if len(region_data) >= min_length:
            classification = classify_data_type(region_data)
            regions.append(
                RegionClassification(
                    start=region_start,
                    end=len(data),
                    length=len(region_data),
                    classification=classification,
                )
            )

    return regions


def detect_encrypted_regions(
    data: DataType, min_length: int = 64, min_entropy: float = 7.5
) -> list[RegionClassification]:
    """Detect potentially encrypted regions (high entropy, no structure).

    : Statistical Data Type Classification

    Identifies regions with very high entropy and uniform byte distribution,
    characteristic of encrypted or cryptographically random data.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        min_length: Minimum region length in bytes (default: 64)
        min_entropy: Minimum entropy threshold (default: 7.5)

    Returns:
        List of detected encrypted regions

    Example:
        >>> import os
        >>> random_data = os.urandom(100)
        >>> regions = detect_encrypted_regions(random_data)
        >>> len(regions) >= 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if len(data) < min_length:
        return []

    regions = []
    window_size = min_length
    step = window_size // 4

    i = 0
    while i < len(data) - window_size:
        window = data[i : i + window_size]
        entropy_val = shannon_entropy(window)

        if entropy_val >= min_entropy:
            # Found potential encrypted region, extend it
            region_start = i
            region_end = i + window_size

            # Extend forward
            while region_end < len(data):
                next_window = data[region_end : region_end + window_size]
                if len(next_window) < window_size:
                    break
                if shannon_entropy(next_window) >= min_entropy:
                    region_end += step
                else:
                    break

            # Create region
            region_data = data[region_start:region_end]
            classification = classify_data_type(region_data)
            regions.append(
                RegionClassification(
                    start=region_start,
                    end=region_end,
                    length=len(region_data),
                    classification=classification,
                )
            )

            i = region_end
        else:
            i += step

    return regions


def detect_compressed_regions(data: DataType, min_length: int = 64) -> list[RegionClassification]:
    """Detect compressed data regions (signatures + high entropy).

    : Statistical Data Type Classification

    Identifies compressed regions by looking for compression signatures
    and characteristic entropy patterns.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        min_length: Minimum region length in bytes (default: 64)

    Returns:
        List of detected compressed regions

    Example:
        >>> import gzip
        >>> compressed = gzip.compress(b'Hello World' * 100)
        >>> regions = detect_compressed_regions(compressed)
        >>> len(regions) > 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    regions = []

    # Scan for compression signatures
    for sig, comp_type in COMPRESSION_SIGNATURES.items():
        offset = 0
        while True:
            pos = data.find(sig, offset)
            if pos == -1:
                break

            # Try to determine compressed region size
            # This is heuristic-based since we don't parse the format
            region_start = pos
            region_end = min(pos + min_length, len(data))

            # Extend based on high entropy
            window_size = 256
            while region_end < len(data):
                window = data[region_end : region_end + window_size]
                if len(window) < window_size:
                    break
                entropy_val = shannon_entropy(window)
                if entropy_val >= 6.0:  # Compressed threshold
                    region_end += window_size
                else:
                    break

            if region_end - region_start >= min_length:
                region_data = data[region_start:region_end]
                classification = classify_data_type(region_data)
                classification.details["compression_signature"] = comp_type

                regions.append(
                    RegionClassification(
                        start=region_start,
                        end=region_end,
                        length=len(region_data),
                        classification=classification,
                    )
                )

            offset = region_end

    return regions


def detect_padding_regions(data: DataType, min_length: int = 4) -> list[RegionClassification]:
    """Detect padding/null regions.

    : Statistical Data Type Classification

    Identifies contiguous regions of null bytes or repetitive padding patterns.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        min_length: Minimum region length in bytes (default: 4)

    Returns:
        List of detected padding regions

    Example:
        >>> data = b'DATA' + b'\\x00' * 100 + b'DATA'
        >>> regions = detect_padding_regions(data)
        >>> len(regions) > 0
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    regions = []
    in_padding = False
    padding_start = 0
    padding_byte = None

    for i, byte in enumerate(data):
        if not in_padding:
            # Check if this could be start of padding
            if byte == 0 or byte == 0xFF:
                in_padding = True
                padding_start = i
                padding_byte = byte
        else:
            # In padding region
            if byte != padding_byte:
                # End of padding
                length = i - padding_start
                if length >= min_length:
                    _region_data = data[padding_start:i]
                    classification = ClassificationResult(
                        primary_type="padding",
                        confidence=1.0,
                        entropy=0.0,
                        printable_ratio=0.0,
                        null_ratio=1.0 if padding_byte == 0 else 0.0,
                        byte_variance=0.0,
                        details={"padding_byte": f"0x{padding_byte:02X}"},
                    )
                    regions.append(
                        RegionClassification(
                            start=padding_start, end=i, length=length, classification=classification
                        )
                    )
                in_padding = False

    # Handle padding extending to end
    if in_padding:
        length = len(data) - padding_start
        if length >= min_length:
            _region_data = data[padding_start:]
            classification = ClassificationResult(
                primary_type="padding",
                confidence=1.0,
                entropy=0.0,
                printable_ratio=0.0,
                null_ratio=1.0 if padding_byte == 0 else 0.0,
                byte_variance=0.0,
                details={"padding_byte": f"0x{padding_byte:02X}"},
            )
            regions.append(
                RegionClassification(
                    start=padding_start, end=len(data), length=length, classification=classification
                )
            )

    return regions


def segment_by_type(data: DataType, min_segment: int = 32) -> list[RegionClassification]:
    """Segment data into regions by type.

    : Statistical Data Type Classification

    Divides data into homogeneous regions using a sliding window approach
    and entropy-based segmentation.

    Args:
        data: Input data as bytes, bytearray, or numpy array
        min_segment: Minimum segment size in bytes (default: 32)

    Returns:
        List of classified regions covering the entire input

    Example:
        >>> data = b'Hello' + b'\\x00' * 50 + bytes(range(256))
        >>> segments = segment_by_type(data)
        >>> len(segments) >= 1
        True
    """
    if isinstance(data, np.ndarray):
        data = data.tobytes() if data.dtype == np.uint8 else bytes(data.flatten())

    if len(data) < min_segment:
        # Single segment
        classification = classify_data_type(data)
        return [
            RegionClassification(
                start=0, end=len(data), length=len(data), classification=classification
            )
        ]

    segments = []
    window_size = min_segment
    step = window_size // 2

    current_type = None
    segment_start = 0

    i = 0
    while i < len(data):
        window_end = min(i + window_size, len(data))
        window = data[i:window_end]

        if len(window) < min_segment and i > 0:
            # Last small fragment, merge with previous segment
            break

        classification = classify_data_type(window)
        detected_type = classification.primary_type

        if current_type is None:
            current_type = detected_type
            segment_start = i
        elif detected_type != current_type:
            # Type changed, finalize previous segment
            segment_data = data[segment_start:i]
            if len(segment_data) >= min_segment:
                seg_classification = classify_data_type(segment_data)
                segments.append(
                    RegionClassification(
                        start=segment_start,
                        end=i,
                        length=len(segment_data),
                        classification=seg_classification,
                    )
                )
            current_type = detected_type
            segment_start = i

        i += step

    # Finalize last segment
    segment_data = data[segment_start:]
    if len(segment_data) > 0:
        seg_classification = classify_data_type(segment_data)
        segments.append(
            RegionClassification(
                start=segment_start,
                end=len(data),
                length=len(segment_data),
                classification=seg_classification,
            )
        )

    return segments


class DataClassifier:
    """Object-oriented wrapper for data type classification.

    Provides a class-based interface for data classification operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> classifier = DataClassifier()
        >>> data_type = classifier.classify(b'Hello, World!')
        >>> data_type
        'text'
    """

    def __init__(self, min_segment_size: int = 32):
        """Initialize data classifier.

        Args:
            min_segment_size: Minimum segment size for region detection.
        """
        self.min_segment_size = min_segment_size

    def classify(self, data: DataType) -> str:
        """Classify binary data type.

        Returns the primary type as a string for test compatibility.

        Args:
            data: Input data as bytes, bytearray, or numpy array.

        Returns:
            String data type classification ('text', 'binary', 'compressed',
            'encrypted', 'padding', or 'mixed').

        Example:
            >>> classifier = DataClassifier()
            >>> classifier.classify(b'Hello')
            'text'
        """
        result = classify_data_type(data)
        return result.primary_type

    def classify_detailed(self, data: DataType) -> ClassificationResult:
        """Classify binary data type with full details.

        Args:
            data: Input data as bytes, bytearray, or numpy array.

        Returns:
            ClassificationResult with type, confidence, and metadata.

        Example:
            >>> classifier = DataClassifier()
            >>> result = classifier.classify_detailed(b'Hello')
            >>> result.data_type == 'text'
            True
        """
        return classify_data_type(data)

    def detect_text_regions(
        self, data: DataType, min_length: int = 8, min_printable: float = 0.8
    ) -> list[RegionClassification]:
        """Detect text regions in data.

        Args:
            data: Input data.
            min_length: Minimum region length.
            min_printable: Minimum printable ratio.

        Returns:
            List of text region classifications.
        """
        return detect_text_regions(data, min_length, min_printable)

    def detect_encrypted_regions(
        self, data: DataType, min_length: int = 64, min_entropy: float = 7.5
    ) -> list[RegionClassification]:
        """Detect encrypted regions in data.

        Args:
            data: Input data.
            min_length: Minimum region length.
            min_entropy: Minimum entropy threshold.

        Returns:
            List of encrypted region classifications.
        """
        return detect_encrypted_regions(data, min_length, min_entropy)

    def detect_compressed_regions(
        self, data: DataType, min_length: int = 64
    ) -> list[RegionClassification]:
        """Detect compressed regions in data.

        Args:
            data: Input data.
            min_length: Minimum region length.

        Returns:
            List of compressed region classifications.
        """
        return detect_compressed_regions(data, min_length)

    def detect_padding_regions(
        self, data: DataType, min_length: int = 4
    ) -> list[RegionClassification]:
        """Detect padding regions in data.

        Args:
            data: Input data.
            min_length: Minimum region length.

        Returns:
            List of padding region classifications.
        """
        return detect_padding_regions(data, min_length)

    def segment(self, data: DataType) -> list[RegionClassification]:
        """Segment data by type.

        Args:
            data: Input data.

        Returns:
            List of classified segments.
        """
        return segment_by_type(data, self.min_segment_size)


__all__ = [
    "ClassificationResult",
    "DataClassifier",
    "DataType",
    "RegionClassification",
    "classify_data_type",
    "detect_compressed_regions",
    "detect_encrypted_regions",
    "detect_padding_regions",
    "detect_text_regions",
    "segment_by_type",
]
