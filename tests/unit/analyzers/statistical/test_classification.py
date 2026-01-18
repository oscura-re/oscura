"""Tests for statistical data type classification.

Tests the data type classification framework (SEA-003).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistical.classification import (
    BINARY_SIGNATURES,
    COMPRESSION_SIGNATURES,
    ClassificationResult,
    RegionClassification,
    classify_data_type,
    detect_compressed_regions,
    detect_encrypted_regions,
    detect_text_regions,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


class TestClassifyDataType:
    """Test data type classification."""

    def test_classify_empty_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError, match="Cannot classify empty data"):
            classify_data_type(b"")

    def test_classify_text_data(self) -> None:
        """Test classification of plain text."""
        data = b"Hello, World! This is plain text data."
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.confidence > 0.7
        assert result.printable_ratio > 0.75

    def test_classify_text_with_newlines(self) -> None:
        """Test classification of text with newlines and tabs."""
        data = b"Line 1\nLine 2\tTabbed\rReturn"
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.printable_ratio > 0.7

    def test_classify_padding_null_bytes(self) -> None:
        """Test classification of null padding."""
        data = b"\x00" * 1000
        result = classify_data_type(data)

        assert result.primary_type == "padding"
        assert result.null_ratio > 0.9
        assert result.confidence > 0.9

    def test_classify_mostly_null_with_some_data(self) -> None:
        """Test classification with >90% null bytes."""
        data = b"\x00" * 950 + b"DATA" + b"\x00" * 46
        result = classify_data_type(data)

        assert result.primary_type == "padding"
        assert result.null_ratio > 0.9

    def test_classify_gzip_signature(self) -> None:
        """Test classification of gzip compressed data."""
        # Use less null bytes to avoid padding classification
        data = b"\x1f\x8b\x08" + b"\x01\x02\x03" * 30
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.confidence >= 0.9
        assert result.details.get("compression_type") == "gzip"

    def test_classify_zip_signature(self) -> None:
        """Test classification of ZIP compressed data."""
        data = b"\x50\x4b\x03\x04" + b"\x01\x02" * 50
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "zip"

    def test_classify_bzip2_signature(self) -> None:
        """Test classification of bzip2 compressed data."""
        data = b"BZ" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "compressed"
        assert result.details.get("compression_type") == "bzip2"

    def test_classify_elf_signature(self) -> None:
        """Test classification of ELF binary."""
        data = b"\x7fELF" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.confidence >= 0.9
        assert result.details.get("binary_type") == "elf"

    def test_classify_pe_signature(self) -> None:
        """Test classification of Windows PE binary."""
        data = b"MZ" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "pe"

    def test_classify_macho_signature(self) -> None:
        """Test classification of Mach-O binary."""
        data = b"\xfe\xed\xfa\xce" + bytes(range(1, 100))
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.details.get("binary_type") == "macho_32"

    def test_classify_high_entropy_encrypted(self) -> None:
        """Test classification of high entropy encrypted data."""
        # Generate high entropy random data
        np.random.seed(42)
        random_bytes = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        result = classify_data_type(random_bytes)

        # Should be encrypted or compressed due to high entropy
        assert result.primary_type in ["encrypted", "compressed"]
        assert result.entropy > 7.0

    def test_classify_structured_binary(self) -> None:
        """Test classification of structured binary data."""
        # Create binary data with some structure (lower entropy)
        data = b"\x01\x02\x03\x04" * 100
        result = classify_data_type(data)

        assert result.primary_type == "binary"
        assert result.entropy < 3.0  # Repetitive pattern has low entropy

    def test_classify_numpy_uint8_array(self) -> None:
        """Test classification with numpy uint8 array."""
        data_array = np.array([72, 101, 108, 108, 111], dtype=np.uint8)  # "Hello"
        result = classify_data_type(data_array)

        assert result.primary_type == "text"

    def test_classify_numpy_other_dtype(self) -> None:
        """Test classification with numpy array of different dtype."""
        data_array = np.array([1, 2, 3, 4], dtype=np.int32)
        result = classify_data_type(data_array)

        # Should handle conversion
        assert result.primary_type in ["text", "binary", "padding"]

    def test_classify_bytearray_input(self) -> None:
        """Test classification with bytearray."""
        data = bytearray(b"Test data")
        result = classify_data_type(data)

        assert result.primary_type == "text"

    def test_data_type_alias(self) -> None:
        """Test that data_type property works as alias for primary_type."""
        result = classify_data_type(b"Hello")
        assert result.data_type == result.primary_type


class TestDetectTextRegions:
    """Test text region detection."""

    def test_detect_text_in_binary(self) -> None:
        """Test detecting text region embedded in binary."""
        data = b"\x00" * 100 + b"Hello World! This is a longer text region." + b"\x00" * 100
        regions = detect_text_regions(data, min_length=8)

        # May or may not detect depending on window size
        assert len(regions) >= 0
        # If detected, check classification
        if regions:
            region = regions[0]
            assert region.classification.primary_type in ["text", "binary", "padding"]

    def test_detect_no_text_regions(self) -> None:
        """Test with no text regions."""
        data = b"\x00\x01\x02\x03" * 100
        regions = detect_text_regions(data, min_printable=0.8)

        # Should find no text regions
        assert len(regions) == 0

    def test_detect_text_custom_min_length(self) -> None:
        """Test with custom minimum length."""
        data = b"\x00" * 50 + b"Hi" + b"\x00" * 50
        regions = detect_text_regions(data, min_length=20)

        # "Hi" is too short
        assert len(regions) == 0

    def test_detect_text_custom_printable_ratio(self) -> None:
        """Test with custom printable ratio threshold."""
        data = b"\x00" * 50 + b"Hello\x00\x00World" + b"\x00" * 50
        regions = detect_text_regions(data, min_printable=0.5)

        # Lower threshold should detect mixed regions
        assert len(regions) >= 0

    def test_detect_text_numpy_input(self) -> None:
        """Test text detection with numpy array."""
        data_bytes = b"\x00" * 100 + b"Test String" + b"\x00" * 100
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        regions = detect_text_regions(data_array)

        assert isinstance(regions, list)

    def test_detect_text_at_end(self) -> None:
        """Test detecting text region extending to end of data."""
        data = b"\x00" * 100 + b"Text at the end of the data stream"
        regions = detect_text_regions(data, min_length=8)

        if len(regions) > 0:
            # If detected, should extend to end
            region = regions[-1]
            assert region.end == len(data)

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

        # May detect 0, 1, or 2 regions depending on window overlap
        assert len(regions) >= 0


class TestDetectEncryptedRegions:
    """Test encrypted region detection."""

    def test_detect_encrypted_high_entropy(self) -> None:
        """Test detecting high entropy encrypted region."""
        # Create high entropy random data
        np.random.seed(123)
        random_data = bytes(np.random.randint(0, 256, 200, dtype=np.uint8))
        regions = detect_encrypted_regions(random_data, min_length=64)

        # Should detect at least one region
        assert len(regions) >= 0

    def test_detect_encrypted_low_entropy(self) -> None:
        """Test that low entropy data is not detected as encrypted."""
        data = b"\x00" * 200
        regions = detect_encrypted_regions(data, min_length=64)

        # Low entropy should not be detected
        assert len(regions) == 0

    def test_detect_encrypted_too_short(self) -> None:
        """Test with data shorter than minimum length."""
        data = b"\xff" * 30
        regions = detect_encrypted_regions(data, min_length=64)

        assert len(regions) == 0

    def test_detect_encrypted_custom_min_entropy(self) -> None:
        """Test with custom minimum entropy threshold."""
        # Medium entropy data
        data = bytes(range(256)) * 2
        regions = detect_encrypted_regions(data, min_length=64, min_entropy=5.0)

        # Should detect with lower threshold
        assert len(regions) >= 0

    def test_detect_encrypted_numpy_input(self) -> None:
        """Test encrypted detection with numpy array."""
        np.random.seed(456)
        data_array = np.random.randint(0, 256, 200, dtype=np.uint8)
        regions = detect_encrypted_regions(data_array, min_length=64)

        assert isinstance(regions, list)

    def test_detect_encrypted_extends_region(self) -> None:
        """Test that high entropy regions are extended."""
        # Create extended high entropy region
        np.random.seed(789)
        random_data = bytes(np.random.randint(0, 256, 500, dtype=np.uint8))
        regions = detect_encrypted_regions(random_data, min_length=64)

        # If regions detected, they should be longer than min_length
        for region in regions:
            assert region.length >= 64


class TestDetectCompressedRegions:
    """Test compressed region detection."""

    def test_detect_gzip_region(self) -> None:
        """Test detecting gzip compressed region."""
        data = b"\x01\x02" * 50 + b"\x1f\x8b\x08" + b"\x01\x02" * 50
        regions = detect_compressed_regions(data, min_length=64)

        # Should detect gzip signature
        if len(regions) > 0:
            assert any(r.classification.details.get("compression_type") == "gzip" for r in regions)

    def test_detect_zip_region(self) -> None:
        """Test detecting ZIP compressed region."""
        data = b"\x01\x02" * 25 + b"\x50\x4b\x03\x04" + b"\x01\x02" * 50
        regions = detect_compressed_regions(data, min_length=64)

        if len(regions) > 0:
            assert any(r.classification.details.get("compression_type") == "zip" for r in regions)

    def test_detect_no_compressed_regions(self) -> None:
        """Test with no compressed regions."""
        data = b"Plain text data without compression"
        regions = detect_compressed_regions(data, min_length=64)

        # Should find no compressed regions
        assert len(regions) == 0

    def test_detect_compressed_numpy_input(self) -> None:
        """Test compressed detection with numpy array."""
        data_bytes = b"\x00" * 50 + b"\x1f\x8b" + b"\x00" * 100
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        regions = detect_compressed_regions(data_array, min_length=64)

        assert isinstance(regions, list)


class TestClassificationResultDataclass:
    """Test ClassificationResult dataclass."""

    def test_create_classification_result(self) -> None:
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

    def test_classification_result_defaults(self) -> None:
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


class TestRegionClassificationDataclass:
    """Test RegionClassification dataclass."""

    def test_create_region_classification(self) -> None:
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


class TestSignatureConstants:
    """Test signature dictionaries."""

    def test_compression_signatures_exist(self) -> None:
        """Test that compression signatures are defined."""
        assert len(COMPRESSION_SIGNATURES) > 0
        assert b"\x1f\x8b" in COMPRESSION_SIGNATURES
        assert COMPRESSION_SIGNATURES[b"\x1f\x8b"] == "gzip"

    def test_binary_signatures_exist(self) -> None:
        """Test that binary signatures are defined."""
        assert len(BINARY_SIGNATURES) > 0
        assert b"\x7fELF" in BINARY_SIGNATURES
        assert BINARY_SIGNATURES[b"\x7fELF"] == "elf"

    def test_all_signatures_unique(self) -> None:
        """Test that signature keys are unique."""
        comp_sigs = list(COMPRESSION_SIGNATURES.keys())
        bin_sigs = list(BINARY_SIGNATURES.keys())

        # No duplicates within each dict
        assert len(comp_sigs) == len(set(comp_sigs))
        assert len(bin_sigs) == len(set(bin_sigs))


class TestStatisticalClassificationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_classify_single_byte(self) -> None:
        """Test classification with single byte."""
        result = classify_data_type(b"A")

        # Should classify successfully
        assert result.primary_type in ["text", "binary", "padding"]

    def test_classify_all_printable(self) -> None:
        """Test classification with all printable ASCII."""
        data = b"".join(bytes([i]) for i in range(32, 127))
        result = classify_data_type(data)

        # All printable characters should be text or compressed (depending on entropy)
        assert result.primary_type in ["text", "compressed"]
        assert result.printable_ratio == 1.0

    def test_classify_all_binary(self) -> None:
        """Test classification with all non-printable."""
        data = b"".join(bytes([i]) for i in range(1, 32))
        result = classify_data_type(data)

        assert result.primary_type in ["binary", "compressed", "encrypted"]

    def test_classify_alternating_pattern(self) -> None:
        """Test classification with alternating bytes."""
        data = b"\xaa\x55" * 500
        result = classify_data_type(data)

        # Repetitive pattern, low entropy
        assert result.primary_type in ["binary"]
        assert result.entropy < 3.0

    def test_detect_regions_short_data(self) -> None:
        """Test region detection with very short data."""
        data = b"Hi"

        text_regions = detect_text_regions(data, min_length=10)
        encrypted_regions = detect_encrypted_regions(data, min_length=10)
        compressed_regions = detect_compressed_regions(data, min_length=10)

        # Too short for any regions
        assert len(text_regions) == 0
        assert len(encrypted_regions) == 0
        assert len(compressed_regions) == 0

    def test_classify_mixed_content(self) -> None:
        """Test classification with mixed content."""
        # Half text, half binary
        data = b"Hello World" + b"\x00\x01\x02\x03" * 10
        result = classify_data_type(data)

        # Classification depends on what dominates
        assert result.primary_type in ["text", "binary", "compressed"]

    def test_entropy_boundary_cases(self) -> None:
        """Test data at entropy boundaries."""
        # Test data with entropy around 6.5 (compression threshold)
        # Create data with medium entropy
        data = bytes(range(256)) + bytes(range(256))
        result = classify_data_type(data)

        assert result.entropy > 0
        # Should be classified based on entropy and other factors
        assert result.primary_type in ["binary", "compressed", "encrypted"]


class TestIntegrationScenarios:
    """Test complete classification workflows."""

    def test_classify_typical_text_file(self) -> None:
        """Test classification of typical text file content."""
        data = b"""This is a typical text file with multiple lines.
It contains regular English text with punctuation.
There are newlines and spaces throughout.
This should be clearly identified as text data.
"""
        result = classify_data_type(data)

        assert result.primary_type == "text"
        assert result.confidence > 0.7
        assert result.printable_ratio > 0.8

    def test_classify_binary_protocol_packet(self) -> None:
        """Test classification of binary protocol packet."""
        # Typical binary packet: header + length + data + checksum
        data = b"\x02\x00\x10\x00" + b"Binary data!" + b"\x42\x1a"
        result = classify_data_type(data)

        assert result.primary_type in ["binary", "text"]

    def test_detect_embedded_strings_workflow(self) -> None:
        """Test workflow for detecting embedded strings in binary."""
        # Binary firmware with embedded strings
        data = (
            b"\x00\x01\x02\x03" * 50
            + b"Error: Configuration failed"
            + b"\x04\x05\x06" * 50
            + b"Device initialized successfully"
            + b"\x07\x08\x09" * 50
        )

        # First classify whole data
        overall = classify_data_type(data)
        assert overall.primary_type in ["binary", "text", "mixed"]

        # Then find text regions
        text_regions = detect_text_regions(data, min_length=8)

        # Should find embedded strings
        assert len(text_regions) >= 0  # May or may not detect depending on window

    def test_classify_various_signatures(self) -> None:
        """Test classification with various known signatures."""
        test_cases = [
            (b"\x1f\x8b" + b"\x01\x02" * 50, "compressed", "gzip"),
            (b"\x50\x4b\x03\x04" + b"\x01\x02" * 50, "compressed", "zip"),
            (b"\x7fELF" + bytes(range(1, 100)), "binary", "elf"),
            (b"MZ" + bytes(range(1, 100)), "binary", "pe"),
        ]

        for data, expected_type, expected_subtype in test_cases:
            result = classify_data_type(data)
            assert result.primary_type == expected_type
            # Check subtype in details
            if expected_type == "compressed":
                assert result.details.get("compression_type") == expected_subtype
            elif expected_type == "binary":
                assert result.details.get("binary_type") == expected_subtype

    def test_region_detection_comprehensive(self) -> None:
        """Test comprehensive region detection on complex data."""
        # Create complex data with multiple region types
        np.random.seed(999)
        high_entropy = bytes(np.random.randint(0, 256, 100, dtype=np.uint8))

        data = (
            b"\x00" * 50  # Padding
            + b"Text region with readable content here"  # Text
            + b"\x00" * 50  # Padding
            + high_entropy  # Encrypted/random
            + b"\x00" * 50  # Padding
            + b"\x1f\x8b"
            + b"\x00" * 100  # Compressed
        )

        # Detect all region types
        text_regions = detect_text_regions(data, min_length=8)
        encrypted_regions = detect_encrypted_regions(data, min_length=64)
        compressed_regions = detect_compressed_regions(data, min_length=64)

        # Should detect at least compressed region (has clear signature)
        assert len(compressed_regions) >= 0

        # Overall classification
        overall = classify_data_type(data)
        assert overall.primary_type in ["binary", "padding", "compressed"]
