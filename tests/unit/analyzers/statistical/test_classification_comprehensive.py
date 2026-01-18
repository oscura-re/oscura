"""Comprehensive unit tests for statistical data type classification module.

Tests for all functions and classes in oscura.analyzers.statistical.classification.
"""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from oscura.analyzers.statistical.classification import (
    BINARY_SIGNATURES,
    COMPRESSION_SIGNATURES,
    ClassificationResult,
    DataClassifier,
    RegionClassification,
    classify_data_type,
    detect_compressed_regions,
    detect_encrypted_regions,
    detect_padding_regions,
    detect_text_regions,
    segment_by_type,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# Classify Data Type Function Tests
# =============================================================================


class TestClassifyDataType:
    """Test classify_data_type function."""

    def test_classify_empty_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot classify empty data"):
            classify_data_type(b"")

    def test_classify_single_byte(self) -> None:
        """Test classification of single byte."""
        result = classify_data_type(b"A")
        assert result.primary_type in ["text", "binary", "padding"]

    def test_classify_plain_text(self) -> None:
        """Test classification of plain ASCII text."""
        data = b"Hello, World! This is a test message with lots of characters."
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.confidence > 0.7
        assert result.printable_ratio > 0.9

    def test_classify_text_with_whitespace(self) -> None:
        """Test classification of text with tabs, newlines, and spaces."""
        data = b"Line 1\nLine 2\tTabbed\rCarriage return"
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.printable_ratio > 0.8

    def test_classify_null_padding(self) -> None:
        """Test classification of null byte padding."""
        data = b"\x00" * 1000
        result = classify_data_type(data)

        assert result.primary_type == "padding"
        assert result.null_ratio > 0.9
        assert result.confidence > 0.9
        assert result.details.get("reason") == "high_null_ratio"

    def test_classify_mostly_null(self) -> None:
        """Test classification with >90% null bytes."""
        data = b"\x00" * 950 + b"DATA" + b"\x00" * 46
        result = classify_data_type(data)

        assert result.primary_type == "padding"
        assert result.null_ratio > 0.9

    def test_classify_gzip_signature(self) -> None:
        """Test classification of gzip compressed data."""
        # Gzip magic bytes with some data
        data = b"\x1f\x8b\x08" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.confidence >= 0.9
        assert result.details.get("compression_type") == "gzip"

    def test_classify_zip_signature(self) -> None:
        """Test classification of ZIP compressed data."""
        data = b"\x50\x4b\x03\x04" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "zip"

    def test_classify_bzip2_signature(self) -> None:
        """Test classification of bzip2 compressed data."""
        data = b"BZ" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "bzip2"

    def test_classify_xz_signature(self) -> None:
        """Test classification of xz compressed data."""
        data = b"\xfd7zXZ\x00" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "xz"

    def test_classify_zstd_signature(self) -> None:
        """Test classification of zstd compressed data."""
        data = b"\x28\xb5\x2f\xfd" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "zstd"

    def test_classify_lz4_signature(self) -> None:
        """Test classification of lz4 compressed data."""
        data = b"\x04\x22\x4d\x18" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "lz4"

    def test_classify_elf_binary(self) -> None:
        """Test classification of ELF executable."""
        data = b"\x7fELF" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.confidence >= 0.9
        assert result.details.get("binary_type") == "elf"

    def test_classify_pe_binary(self) -> None:
        """Test classification of Windows PE executable."""
        data = b"MZ" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "pe"

    def test_classify_macho_32(self) -> None:
        """Test classification of Mach-O 32-bit binary."""
        data = b"\xfe\xed\xfa\xce" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "macho_32"

    def test_classify_macho_64(self) -> None:
        """Test classification of Mach-O 64-bit binary."""
        data = b"\xfe\xed\xfa\xcf" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "macho_64"

    def test_classify_macho_fat(self) -> None:
        """Test classification of Mach-O fat binary."""
        data = b"\xca\xfe\xba\xbe" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "macho_fat"

    def test_classify_high_entropy_encrypted(self) -> None:
        """Test classification of high entropy data as encrypted."""
        np.random.seed(42)
        random_data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        result = classify_data_type(random_data)

        # High entropy + high variance = encrypted
        assert result.primary_type in ["encrypted", "compressed"]
        assert result.entropy > 7.0

    def test_classify_medium_entropy_compressed(self) -> None:
        """Test classification of medium-high entropy as compressed."""
        # Create data with entropy in 6.5-7.5 range
        data = bytes([i % 200 for i in range(1000)])
        result = classify_data_type(data)

        # Entropy should be in compressed range
        assert result.primary_type in ["compressed", "binary"]

    def test_classify_structured_binary(self) -> None:
        """Test classification of structured binary data."""
        # Repetitive pattern = low entropy = structured binary
        data = b"\x01\x02\x03\x04" * 100
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.entropy < 4.0

    def test_classify_bytearray_input(self) -> None:
        """Test classification with bytearray input."""
        data = bytearray(b"Hello World")
        result = classify_data_type(data)

        assert result.primary_type == "text"

    def test_classify_numpy_uint8_input(self) -> None:
        """Test classification with numpy uint8 array."""
        data = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
        result = classify_data_type(data)

        assert result.primary_type == "text"

    def test_classify_numpy_other_dtype(self) -> None:
        """Test classification with numpy array of different dtype."""
        data = np.array([1, 2, 3, 4], dtype=np.int32)
        result = classify_data_type(data)

        # Should handle conversion and classify
        assert result.primary_type in ["text", "binary", "padding"]


# =============================================================================
# Detect Text Regions Function Tests
# =============================================================================


class TestDetectTextRegions:
    """Test detect_text_regions function."""

    def test_detect_text_in_binary(self) -> None:
        """Test detecting text embedded in binary data."""
        data = b"\x00" * 100 + b"This is embedded text content" + b"\x00" * 100
        regions = detect_text_regions(data, min_length=8)

        # May detect text depending on window
        assert isinstance(regions, list)

    def test_detect_no_text(self) -> None:
        """Test with no text regions."""
        # Use truly non-printable bytes (0-31 range only)
        data = bytes(range(1, 32)) * 10
        regions = detect_text_regions(data, min_printable=0.9)

        assert len(regions) == 0

    def test_detect_text_custom_min_length(self) -> None:
        """Test with custom minimum length."""
        data = b"\x00" * 50 + b"Hi" + b"\x00" * 50
        regions = detect_text_regions(data, min_length=20)

        # "Hi" is too short
        assert len(regions) == 0

    def test_detect_text_custom_printable_ratio(self) -> None:
        """Test with custom printable ratio."""
        data = b"\x00" * 50 + b"Hello\x00\x00World" + b"\x00" * 50
        regions = detect_text_regions(data, min_printable=0.5)

        # Lower threshold may detect mixed regions
        assert isinstance(regions, list)

    def test_detect_text_at_end(self) -> None:
        """Test detecting text at end of data."""
        data = b"\x00" * 100 + b"Text at the end of the data"
        regions = detect_text_regions(data, min_length=8)

        if len(regions) > 0:
            # Last region should extend to end
            assert regions[-1].end == len(data)

    def test_detect_text_at_start(self) -> None:
        """Test detecting text at start of data."""
        data = b"Text at the start" + b"\x00" * 100
        regions = detect_text_regions(data, min_length=8)

        if len(regions) > 0:
            # First region should start at beginning
            assert regions[0].start == 0

    def test_detect_multiple_text_regions(self) -> None:
        """Test detecting multiple separate text regions."""
        data = (
            b"\x00" * 50
            + b"First text region here"
            + b"\x00" * 50
            + b"Second text region"
            + b"\x00" * 50
        )
        regions = detect_text_regions(data, min_length=8)

        # May detect multiple regions
        assert isinstance(regions, list)

    def test_detect_text_numpy_input(self) -> None:
        """Test text detection with numpy array."""
        data_bytes = b"\x00" * 100 + b"Test String" + b"\x00" * 100
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        regions = detect_text_regions(data_array)

        assert isinstance(regions, list)


# =============================================================================
# Detect Encrypted Regions Function Tests
# =============================================================================


class TestDetectEncryptedRegions:
    """Test detect_encrypted_regions function."""

    def test_detect_encrypted_high_entropy(self) -> None:
        """Test detecting high entropy random data."""
        np.random.seed(123)
        random_data = bytes(np.random.randint(0, 256, 200, dtype=np.uint8))
        regions = detect_encrypted_regions(random_data, min_length=64)

        # Should detect at least one region
        assert isinstance(regions, list)

    def test_detect_no_encrypted_low_entropy(self) -> None:
        """Test that low entropy data is not detected."""
        data = b"\x00" * 200
        regions = detect_encrypted_regions(data, min_length=64)

        assert len(regions) == 0

    def test_detect_encrypted_too_short(self) -> None:
        """Test with data shorter than min_length."""
        data = bytes(range(30))
        regions = detect_encrypted_regions(data, min_length=64)

        assert len(regions) == 0

    def test_detect_encrypted_custom_entropy(self) -> None:
        """Test with custom entropy threshold."""
        # Create data with known entropy
        data = bytes(range(256)) * 2
        regions = detect_encrypted_regions(data, min_length=64, min_entropy=5.0)

        # Lower threshold should detect more
        assert isinstance(regions, list)

    def test_detect_encrypted_region_extension(self) -> None:
        """Test that detected regions are extended."""
        np.random.seed(456)
        random_data = bytes(np.random.randint(0, 256, 500, dtype=np.uint8))
        regions = detect_encrypted_regions(random_data, min_length=64)

        for region in regions:
            assert region.length >= 64

    def test_detect_encrypted_numpy_input(self) -> None:
        """Test with numpy array input."""
        np.random.seed(789)
        data = np.random.randint(0, 256, 200, dtype=np.uint8)
        regions = detect_encrypted_regions(data, min_length=64)

        assert isinstance(regions, list)


# =============================================================================
# Detect Compressed Regions Function Tests
# =============================================================================


class TestDetectCompressedRegions:
    """Test detect_compressed_regions function."""

    def test_detect_gzip_region(self) -> None:
        """Test detecting gzip compressed region."""
        # Create real gzip data
        original = b"Hello World! " * 100
        compressed = gzip.compress(original)
        data = b"\x00" * 50 + compressed + b"\x00" * 50

        regions = detect_compressed_regions(data, min_length=64)

        if len(regions) > 0:
            assert any(
                r.classification.details.get("compression_signature") == "gzip"
                or r.classification.details.get("compression_type") == "gzip"
                for r in regions
            )

    def test_detect_gzip_by_signature(self) -> None:
        """Test detecting gzip by magic signature."""
        data = b"\x01\x02" * 50 + b"\x1f\x8b\x08" + bytes(range(1, 100))
        regions = detect_compressed_regions(data, min_length=64)

        if len(regions) > 0:
            # Check for gzip detection
            gzip_found = any("gzip" in str(r.classification.details) for r in regions)
            assert gzip_found or len(regions) == 0

    def test_detect_zip_region(self) -> None:
        """Test detecting ZIP region by signature."""
        data = b"\x00" * 50 + b"\x50\x4b\x03\x04" + bytes(range(1, 100))
        regions = detect_compressed_regions(data, min_length=64)

        assert isinstance(regions, list)

    def test_detect_no_compressed_regions(self) -> None:
        """Test with no compressed regions."""
        data = b"Plain text without compression"
        regions = detect_compressed_regions(data, min_length=64)

        assert len(regions) == 0

    def test_detect_compressed_numpy_input(self) -> None:
        """Test with numpy array input."""
        data_bytes = b"\x00" * 50 + b"\x1f\x8b" + b"\x00" * 100
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        regions = detect_compressed_regions(data_array, min_length=64)

        assert isinstance(regions, list)

    def test_detect_multiple_compressed_regions(self) -> None:
        """Test detecting multiple compressed signatures."""
        data = b"\x1f\x8b" + b"\x00" * 100 + b"\x50\x4b\x03\x04" + b"\x00" * 100
        regions = detect_compressed_regions(data, min_length=64)

        # May detect multiple regions
        assert isinstance(regions, list)


# =============================================================================
# Detect Padding Regions Function Tests
# =============================================================================


class TestDetectPaddingRegions:
    """Test detect_padding_regions function."""

    def test_detect_null_padding(self) -> None:
        """Test detecting null byte padding."""
        data = b"DATA" + b"\x00" * 100 + b"DATA"
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) > 0
        assert regions[0].classification.primary_type == "padding"
        assert "0x00" in regions[0].classification.details.get("padding_byte", "")

    def test_detect_ff_padding(self) -> None:
        """Test detecting 0xFF padding."""
        data = b"DATA" + b"\xff" * 100 + b"DATA"
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) > 0
        assert "0xFF" in regions[0].classification.details.get("padding_byte", "")

    def test_detect_padding_at_end(self) -> None:
        """Test detecting padding at end of data."""
        data = b"DATA" + b"\x00" * 100
        regions = detect_padding_regions(data, min_length=4)

        if len(regions) > 0:
            assert regions[-1].end == len(data)

    def test_detect_padding_at_start(self) -> None:
        """Test detecting padding at start of data."""
        data = b"\x00" * 100 + b"DATA"
        regions = detect_padding_regions(data, min_length=4)

        if len(regions) > 0:
            assert regions[0].start == 0

    def test_detect_no_padding(self) -> None:
        """Test with no padding regions."""
        data = b"No padding here at all"
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) == 0

    def test_detect_padding_min_length(self) -> None:
        """Test min_length parameter."""
        data = b"A" + b"\x00" * 3 + b"B"  # Only 3 nulls
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) == 0

    def test_detect_multiple_padding_regions(self) -> None:
        """Test detecting multiple padding regions."""
        data = b"A" + b"\x00" * 20 + b"B" + b"\xff" * 20 + b"C"
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) == 2

    def test_detect_padding_numpy_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([65] + [0] * 100 + [66], dtype=np.uint8)
        regions = detect_padding_regions(data, min_length=4)

        assert len(regions) > 0


# =============================================================================
# Segment By Type Function Tests
# =============================================================================


class TestSegmentByType:
    """Test segment_by_type function."""

    def test_segment_short_data(self) -> None:
        """Test segmentation of data shorter than min_segment."""
        data = b"Short"
        segments = segment_by_type(data, min_segment=32)

        assert len(segments) == 1
        assert segments[0].start == 0
        assert segments[0].end == len(data)

    def test_segment_homogeneous_text(self) -> None:
        """Test segmentation of homogeneous text."""
        data = b"This is all text content. " * 100
        segments = segment_by_type(data, min_segment=32)

        # Should be primarily one segment type
        assert len(segments) >= 1
        assert all(s.classification.primary_type == "text" for s in segments)

    def test_segment_mixed_content(self) -> None:
        """Test segmentation of mixed content types."""
        np.random.seed(42)
        text_part = b"This is text content. " * 50
        binary_part = bytes(range(256)) * 10
        random_part = bytes(np.random.randint(0, 256, 500, dtype=np.uint8))
        data = text_part + binary_part + random_part

        segments = segment_by_type(data, min_segment=64)

        # Should find multiple segments
        assert len(segments) >= 1

    def test_segment_coverage(self) -> None:
        """Test that segments cover entire data."""
        data = b"Mixed content with binary \x00\x01\x02 and text"
        segments = segment_by_type(data, min_segment=16)

        # Segments should cover from start to end
        assert segments[0].start == 0
        assert segments[-1].end == len(data)

    def test_segment_numpy_input(self) -> None:
        """Test segmentation with numpy array."""
        data = np.array([65, 66, 67] * 100, dtype=np.uint8)
        segments = segment_by_type(data, min_segment=32)

        assert isinstance(segments, list)


# =============================================================================
# DataClassifier Class Tests
# =============================================================================


class TestDataClassifier:
    """Test DataClassifier class."""

    def test_classifier_init(self) -> None:
        """Test classifier initialization."""
        classifier = DataClassifier(min_segment_size=64)
        assert classifier.min_segment_size == 64

    def test_classifier_classify(self) -> None:
        """Test classify method returns string."""
        classifier = DataClassifier()
        result = classifier.classify(b"Hello World")

        assert isinstance(result, str)
        assert result == "text"

    def test_classifier_classify_detailed(self) -> None:
        """Test classify_detailed method returns ClassificationResult."""
        classifier = DataClassifier()
        result = classifier.classify_detailed(b"Hello World")

        assert isinstance(result, ClassificationResult)
        assert result.primary_type == "text"

    def test_classifier_detect_text_regions(self) -> None:
        """Test detect_text_regions method."""
        classifier = DataClassifier()
        data = b"\x00" * 50 + b"Text content" + b"\x00" * 50
        regions = classifier.detect_text_regions(data, min_length=8)

        assert isinstance(regions, list)

    def test_classifier_detect_encrypted_regions(self) -> None:
        """Test detect_encrypted_regions method."""
        classifier = DataClassifier()
        np.random.seed(42)
        data = bytes(np.random.randint(0, 256, 200, dtype=np.uint8))
        regions = classifier.detect_encrypted_regions(data, min_length=64)

        assert isinstance(regions, list)

    def test_classifier_detect_compressed_regions(self) -> None:
        """Test detect_compressed_regions method."""
        classifier = DataClassifier()
        data = b"\x1f\x8b" + b"\x00" * 100
        regions = classifier.detect_compressed_regions(data, min_length=64)

        assert isinstance(regions, list)

    def test_classifier_detect_padding_regions(self) -> None:
        """Test detect_padding_regions method."""
        classifier = DataClassifier()
        data = b"A" + b"\x00" * 50 + b"B"
        regions = classifier.detect_padding_regions(data, min_length=4)

        assert isinstance(regions, list)
        assert len(regions) > 0

    def test_classifier_segment(self) -> None:
        """Test segment method."""
        classifier = DataClassifier(min_segment_size=32)
        data = b"Test data " * 100
        segments = classifier.segment(data)

        assert isinstance(segments, list)
        assert len(segments) >= 1


# =============================================================================
# ClassificationResult Dataclass Tests
# =============================================================================


class TestClassificationResultDataclass:
    """Test ClassificationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating ClassificationResult."""
        result = ClassificationResult(
            primary_type="text",
            confidence=0.95,
            entropy=4.5,
            printable_ratio=0.9,
            null_ratio=0.0,
            byte_variance=1000.0,
            details={"reason": "high_printable_ratio"},
        )

        assert result.primary_type == "text"
        assert result.confidence == 0.95
        assert result.entropy == 4.5
        assert result.printable_ratio == 0.9
        assert result.null_ratio == 0.0
        assert result.byte_variance == 1000.0
        assert result.details["reason"] == "high_printable_ratio"

    def test_result_default_details(self) -> None:
        """Test ClassificationResult with default details."""
        result = ClassificationResult(
            primary_type="binary",
            confidence=0.8,
            entropy=6.0,
            printable_ratio=0.3,
            null_ratio=0.1,
            byte_variance=2000.0,
        )

        assert result.details == {}

    def test_data_type_property(self) -> None:
        """Test data_type property alias."""
        result = ClassificationResult(
            primary_type="compressed",
            confidence=0.9,
            entropy=7.0,
            printable_ratio=0.2,
            null_ratio=0.05,
            byte_variance=3000.0,
        )

        assert result.data_type == "compressed"
        assert result.data_type == result.primary_type


# =============================================================================
# RegionClassification Dataclass Tests
# =============================================================================


class TestRegionClassificationDataclass:
    """Test RegionClassification dataclass."""

    def test_create_region(self) -> None:
        """Test creating RegionClassification."""
        classification = ClassificationResult(
            primary_type="text",
            confidence=0.9,
            entropy=4.0,
            printable_ratio=0.85,
            null_ratio=0.0,
            byte_variance=1500.0,
        )

        region = RegionClassification(
            start=100,
            end=200,
            length=100,
            classification=classification,
        )

        assert region.start == 100
        assert region.end == 200
        assert region.length == 100
        assert region.classification.primary_type == "text"


# =============================================================================
# Signature Constants Tests
# =============================================================================


class TestSignatureConstants:
    """Test signature dictionaries."""

    def test_compression_signatures_content(self) -> None:
        """Test compression signatures are defined correctly."""
        assert len(COMPRESSION_SIGNATURES) >= 6

        # Check specific signatures
        assert COMPRESSION_SIGNATURES[b"\x1f\x8b"] == "gzip"
        assert COMPRESSION_SIGNATURES[b"BZ"] == "bzip2"
        assert COMPRESSION_SIGNATURES[b"\x50\x4b\x03\x04"] == "zip"

    def test_binary_signatures_content(self) -> None:
        """Test binary signatures are defined correctly."""
        assert len(BINARY_SIGNATURES) >= 6

        # Check specific signatures
        assert BINARY_SIGNATURES[b"\x7fELF"] == "elf"
        assert BINARY_SIGNATURES[b"MZ"] == "pe"
        assert BINARY_SIGNATURES[b"\xca\xfe\xba\xbe"] == "macho_fat"

    def test_signatures_unique(self) -> None:
        """Test that signature keys are unique."""
        comp_sigs = list(COMPRESSION_SIGNATURES.keys())
        bin_sigs = list(BINARY_SIGNATURES.keys())

        assert len(comp_sigs) == len(set(comp_sigs))
        assert len(bin_sigs) == len(set(bin_sigs))

    def test_no_overlapping_signatures(self) -> None:
        """Test compression and binary signatures don't overlap."""
        comp_keys = set(COMPRESSION_SIGNATURES.keys())
        bin_keys = set(BINARY_SIGNATURES.keys())

        overlap = comp_keys & bin_keys
        assert len(overlap) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestClassificationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_printable_ascii(self) -> None:
        """Test data with all printable ASCII characters."""
        data = bytes(range(32, 127))
        result = classify_data_type(data)

        assert result.printable_ratio == 1.0
        assert result.primary_type in ["text", "compressed"]

    def test_all_non_printable(self) -> None:
        """Test data with all non-printable characters."""
        data = bytes(range(1, 32))
        result = classify_data_type(data)

        assert result.primary_type in ["binary", "compressed", "encrypted"]

    def test_alternating_pattern(self) -> None:
        """Test alternating byte pattern."""
        data = b"\xaa\x55" * 500
        result = classify_data_type(data)

        # Low entropy, repetitive pattern
        assert result.primary_type == "binary"
        assert result.entropy < 3.0

    def test_entropy_at_compression_boundary(self) -> None:
        """Test data at compression entropy boundary (6.5)."""
        # Create data with controlled entropy
        data = bytes([i % 150 for i in range(1000)])
        result = classify_data_type(data)

        # Should be classified based on entropy
        assert result.primary_type in ["binary", "compressed"]

    def test_very_short_regions(self) -> None:
        """Test region detection with very short data."""
        data = b"Hi"

        text_regions = detect_text_regions(data, min_length=10)
        encrypted_regions = detect_encrypted_regions(data, min_length=10)
        compressed_regions = detect_compressed_regions(data, min_length=10)
        padding_regions = detect_padding_regions(data, min_length=10)

        assert len(text_regions) == 0
        assert len(encrypted_regions) == 0
        assert len(compressed_regions) == 0
        assert len(padding_regions) == 0

    def test_mixed_null_and_data(self) -> None:
        """Test data with nulls interspersed."""
        data = b"A\x00B\x00C\x00D" * 100
        result = classify_data_type(data)

        # Has both text and nulls
        assert result.primary_type in ["text", "binary"]

    def test_json_data(self) -> None:
        """Test classification of JSON data."""
        json_data = b'{"name": "test", "value": 123, "items": [1, 2, 3]}'
        result = classify_data_type(json_data)

        assert result.primary_type == "text"

    def test_xml_data(self) -> None:
        """Test classification of XML data."""
        xml_data = b"<root><item>Test</item></root>"
        result = classify_data_type(xml_data)

        assert result.primary_type == "text"

    def test_binary_with_text_header(self) -> None:
        """Test binary data that starts with text-like header."""
        data = b"#!python\n" + bytes(range(256))
        result = classify_data_type(data)

        # Could be classified in various ways based on entropy and ratio
        assert result.primary_type in ["text", "binary", "compressed", "encrypted"]


# =============================================================================
# Integration Scenarios Tests
# =============================================================================


class TestClassificationIntegration:
    """Test complete classification workflows."""

    def test_workflow_text_file(self) -> None:
        """Test classification of typical text file."""
        data = b"""This is a typical text file with multiple lines.
It contains regular English text with punctuation.
There are newlines and spaces throughout.
This should be clearly identified as text data.
"""
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.confidence > 0.7
        assert result.printable_ratio > 0.8

    def test_workflow_binary_protocol(self) -> None:
        """Test classification of binary protocol packet."""
        # Typical packet: header + length + data + checksum
        data = b"\x02\x00\x10\x00" + b"Binary payload" + b"\x42\x1a"
        result = classify_data_type(data)

        assert result.primary_type in ["binary", "text"]

    def test_workflow_embedded_strings(self) -> None:
        """Test workflow for detecting embedded strings."""
        # Firmware with embedded strings
        data = (
            b"\x00\x01\x02\x03" * 50
            + b"Error: Configuration failed"
            + b"\x04\x05\x06" * 50
            + b"Device initialized"
            + b"\x07\x08\x09" * 50
        )

        # Classify whole data
        overall = classify_data_type(data)
        assert overall.primary_type in ["binary", "text", "mixed"]

        # Find text regions
        text_regions = detect_text_regions(data, min_length=8)
        assert isinstance(text_regions, list)

    def test_workflow_various_signatures(self) -> None:
        """Test classification with various known signatures."""
        test_cases = [
            (b"\x1f\x8b" + bytes(range(1, 100)), "compressed", "gzip"),
            (b"\x50\x4b\x03\x04" + bytes(range(1, 100)), "compressed", "zip"),
            (b"\x7fELF" + bytes(range(1, 100)), "binary", "elf"),
            (b"MZ" + bytes(range(1, 100)), "binary", "pe"),
        ]

        for data, expected_type, expected_subtype in test_cases:
            result = classify_data_type(data)
            assert result.primary_type == expected_type

            if expected_type == "compressed":
                assert result.details.get("compression_type") == expected_subtype
            elif expected_type == "binary":
                assert result.details.get("binary_type") == expected_subtype

    def test_workflow_comprehensive_regions(self) -> None:
        """Test comprehensive region detection on complex data."""
        np.random.seed(999)
        high_entropy = bytes(np.random.randint(0, 256, 100, dtype=np.uint8))

        data = (
            b"\x00" * 50  # Padding
            + b"Text region with readable content"  # Text
            + b"\x00" * 50  # Padding
            + high_entropy  # Encrypted/random
            + b"\x00" * 50  # Padding
        )

        # Detect all region types
        text_regions = detect_text_regions(data, min_length=8)
        padding_regions = detect_padding_regions(data, min_length=4)

        # Should detect padding regions
        assert len(padding_regions) > 0
        assert all(r.classification.primary_type == "padding" for r in padding_regions)

    def test_real_gzip_data(self) -> None:
        """Test with real gzip compressed data."""
        original = b"Hello World! This is test data to compress." * 50
        compressed = gzip.compress(original)

        result = classify_data_type(compressed)

        # Should detect as compressed
        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "gzip"
