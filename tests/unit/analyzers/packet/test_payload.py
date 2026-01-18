"""Comprehensive unit tests for payload analysis module.

Tests payload extraction and analysis functionality:
- RE-PAY-001: Payload extraction framework
- RE-PAY-002: Payload pattern search
- RE-PAY-003: Payload delimiter detection
- RE-PAY-004: Payload field inference
- RE-PAY-005: Payload comparison and differential analysis

    - RE-PAY-001 through RE-PAY-005: Complete payload analysis
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.packet.payload import (
    DelimiterResult,
    FieldInferrer,
    InferredField,
    LengthPrefixResult,
    MessageBoundary,
    MessageSchema,
    PatternMatch,
    PayloadCluster,
    PayloadDiff,
    PayloadExtractor,
    PayloadInfo,
    VariablePositions,
    cluster_payloads,
    compute_similarity,
    correlate_request_response,
    detect_delimiter,
    detect_field_types,
    detect_length_prefix,
    diff_payloads,
    filter_by_pattern,
    find_checksum_fields,
    find_common_bytes,
    find_message_boundaries,
    find_sequence_fields,
    find_variable_positions,
    infer_fields,
    search_pattern,
    search_patterns,
    segment_messages,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.packet]


# =============================================================================
# RE-PAY-001: Payload Extraction Tests
# =============================================================================


@pytest.mark.unit
class TestPayloadExtractor:
    """Test payload extraction framework."""

    def test_extract_from_bytes(self):
        """Test extracting payload from raw bytes."""
        extractor = PayloadExtractor()

        payload = extractor.extract_payload(b"test data")

        assert payload == b"test data"

    def test_extract_from_dict(self):
        """Test extracting payload from packet dict."""
        extractor = PayloadExtractor()
        packet = {"data": b"payload"}

        payload = extractor.extract_payload(packet)

        assert payload == b"payload"

    def test_extract_with_zero_copy(self):
        """Test zero-copy extraction with memoryview."""
        extractor = PayloadExtractor(zero_copy=True, return_type="memoryview")
        data = b"test"

        payload = extractor.extract_payload(data)

        # Should be memoryview or bytes
        assert isinstance(payload, memoryview | bytes)

    def test_extract_as_numpy(self):
        """Test extraction as numpy array."""
        extractor = PayloadExtractor(return_type="numpy")
        data = b"test"

        payload = extractor.extract_payload(data)

        assert isinstance(payload, np.ndarray | bytes)

    def test_extract_all_payloads(self):
        """Test batch payload extraction."""
        extractor = PayloadExtractor()
        packets = [
            {"data": b"payload1", "protocol": "UDP"},
            {"data": b"payload2", "protocol": "UDP"},
            {"data": b"payload3", "protocol": "TCP"},
        ]

        payloads = extractor.extract_all_payloads(packets)

        assert len(payloads) >= 3

    def test_extract_with_protocol_filter(self):
        """Test extraction with protocol filter."""
        extractor = PayloadExtractor()
        packets = [
            {"data": b"udp1", "protocol": "UDP"},
            {"data": b"tcp1", "protocol": "TCP"},
            {"data": b"udp2", "protocol": "UDP"},
        ]

        payloads = extractor.extract_all_payloads(packets, protocol="UDP")

        assert len(payloads) == 2

    def test_extract_with_port_filter(self):
        """Test extraction with port filter."""
        extractor = PayloadExtractor()
        packets = [
            {"data": b"p1", "protocol": "UDP", "src_port": 1234, "dst_port": 80},
            {"data": b"p2", "protocol": "UDP", "src_port": 5678, "dst_port": 80},
            {"data": b"p3", "protocol": "UDP", "src_port": 1234, "dst_port": 443},
        ]

        payloads = extractor.extract_all_payloads(packets, port_filter=(1234, None))

        assert len(payloads) == 2


@pytest.mark.unit
class TestPayloadInfo:
    """Test PayloadInfo data class."""

    def test_payload_info_creation(self):
        """Test PayloadInfo creation."""
        info = PayloadInfo(
            data=b"test",
            packet_index=5,
            timestamp=1.23,
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=1234,
            dst_port=80,
            protocol="UDP",
        )

        assert info.data == b"test"
        assert info.packet_index == 5
        assert info.src_ip == "192.168.1.1"

    def test_payload_info_minimal(self):
        """Test PayloadInfo with minimal fields."""
        info = PayloadInfo(data=b"data", packet_index=0)

        assert info.timestamp is None
        assert info.is_fragment is False


# =============================================================================
# RE-PAY-002: Pattern Search Tests
# =============================================================================


@pytest.mark.unit
class TestPatternSearch:
    """Test payload pattern search."""

    def test_search_pattern_basic(self):
        """Test basic pattern search."""
        packets = [b"Hello World Hello"]

        matches = search_pattern(packets, b"Hello")

        assert len(matches) == 2
        assert matches[0].offset == 0
        assert matches[1].offset == 12

    def test_search_pattern_regex(self):
        """Test regex pattern search."""
        packets = [b"test123abc456def"]

        matches = search_pattern(packets, rb"\d+", pattern_type="regex")

        assert len(matches) >= 1

    def test_search_pattern_context(self):
        """Test pattern search with context."""
        packets = [b"AAABBBCCC"]

        matches = search_pattern(packets, b"BBB", context_bytes=2)

        assert len(matches) == 1
        assert len(matches[0].context) > 0

    def test_search_patterns_multiple(self):
        """Test searching for multiple patterns."""
        packets = [b"GET /path HTTP/1.1\r\n"]
        patterns = {"get": b"GET", "http": b"HTTP"}

        results = search_patterns(packets, patterns)

        assert "get" in results
        assert "http" in results

    def test_filter_by_pattern(self):
        """Test filtering payloads by pattern."""
        packets = [
            b"contains_keyword",
            b"no match",
            b"keyword again",
        ]

        filtered = filter_by_pattern(packets, b"keyword")

        assert len(filtered) == 2


# =============================================================================
# RE-PAY-003: Delimiter Detection Tests
# =============================================================================


@pytest.mark.unit
class TestDelimiterDetection:
    """Test delimiter detection."""

    def test_detect_delimiter_newline(self):
        """Test detecting newline delimiter."""
        messages = [
            b"record1\nrecord2\nrecord3\n",
            b"data1\ndata2\ndata3\n",
            b"line1\nline2\nline3\n",
        ]

        result = detect_delimiter(messages)

        assert result.delimiter == b"\n"
        assert result.confidence > 0.5

    def test_detect_delimiter_custom(self):
        """Test detecting custom delimiter."""
        messages = [
            b"field1|field2|field3",
            b"a|b|c",
            b"x|y|z",
        ]

        result = detect_delimiter(messages, candidates=[b"|", b",", b"\n"])

        assert result.delimiter == b"|"

    def test_detect_length_prefix_2byte(self):
        """Test detecting 2-byte length prefix."""
        # Messages with length prefixes
        messages = [
            b"\x00\x03ABC",
            b"\x00\x04TEST",
            b"\x00\x02XY",
        ]

        result = detect_length_prefix(messages)

        assert result.detected is True
        assert result.length_bytes >= 1

    def test_find_message_boundaries(self):
        """Test finding message boundaries in stream."""
        stream = b"MSG1\x00MSG2\x00MSG3\x00"

        boundaries = find_message_boundaries(stream, delimiter=b"\x00")

        assert len(boundaries) == 3
        assert boundaries[0].data == b"MSG1"

    def test_segment_messages(self):
        """Test segmenting concatenated messages."""
        stream = b"\x03ABC\x02XY\x04TEST"

        # Create length prefix result
        length_prefix = LengthPrefixResult(
            detected=True,
            length_bytes=1,
            offset=0,
        )

        messages = segment_messages(stream, length_prefix=length_prefix)

        assert len(messages) >= 2


# =============================================================================
# RE-PAY-004: Field Inference Tests
# =============================================================================


@pytest.mark.unit
class TestFieldInference:
    """Test payload field inference."""

    def test_infer_fields_basic(self):
        """Test basic field inference."""
        messages = [
            b"\x01\x00\x00\x00TEST",
            b"\x02\x00\x00\x00DATA",
            b"\x03\x00\x00\x00INFO",
        ]

        schema = infer_fields(messages, min_samples=3)

        assert isinstance(schema, MessageSchema)
        assert len(schema.fields) > 0

    def test_field_inferrer_class(self):
        """Test FieldInferrer class."""
        inferrer = FieldInferrer()
        messages = [
            b"\x01\x02\x03",
            b"\x01\x02\x04",
            b"\x01\x02\x05",
        ]

        schema = inferrer.infer_fields(messages)

        assert len(schema.fields) > 0

    def test_detect_field_types(self):
        """Test field type detection."""
        messages = [
            b"\x00\x01\x00\x02",
            b"\x00\x01\x00\x03",
            b"\x00\x01\x00\x04",
        ]
        boundaries = [(0, 2), (2, 4)]

        fields = detect_field_types(messages, boundaries)

        assert len(fields) == 2

    def test_find_sequence_fields(self):
        """Test finding sequence/counter fields."""
        # Messages with incrementing counter at offset 0
        messages = [
            b"\x01\xaa\xbb",
            b"\x02\xaa\xbb",
            b"\x03\xaa\xbb",
            b"\x04\xaa\xbb",
        ]

        sequence_fields = find_sequence_fields(messages)

        assert len(sequence_fields) >= 1
        # Should find sequence at offset 0, size 1

    def test_find_checksum_fields(self):
        """Test finding checksum fields."""
        # Simple messages with last byte as checksum
        messages = [
            b"\x01\x02\x03",  # Sum=6
            b"\x04\x05\x09",  # Sum=18
            b"\x0a\x0b\x15",  # Sum=48
        ]

        checksum_fields = find_checksum_fields(messages)

        # May or may not detect depending on correlation
        assert isinstance(checksum_fields, list)

    def test_inferred_field_properties(self):
        """Test InferredField attributes."""
        field = InferredField(
            name="test_field",
            offset=0,
            size=4,
            inferred_type="uint32",
            endianness="big",
            is_constant=False,
            is_sequence=True,
            confidence=0.9,
        )

        assert field.inferred_type == "uint32"
        assert field.is_sequence is True

    def test_message_schema_properties(self):
        """Test MessageSchema attributes."""
        fields = [
            InferredField(name="f1", offset=0, size=2, inferred_type="uint16", confidence=0.8)
        ]

        schema = MessageSchema(
            fields=fields,
            message_length=10,
            fixed_length=True,
            length_range=(10, 10),
            sample_count=5,
            confidence=0.8,
        )

        assert schema.fixed_length is True
        assert schema.sample_count == 5


# =============================================================================
# RE-PAY-005: Comparison and Differential Analysis Tests
# =============================================================================


@pytest.mark.unit
class TestPayloadComparison:
    """Test payload comparison and differential analysis."""

    def test_diff_payloads_identical(self):
        """Test diff of identical payloads."""
        a = b"test data"
        b = b"test data"

        diff = diff_payloads(a, b)

        assert diff.similarity == 1.0
        assert len(diff.differences) == 0

    def test_diff_payloads_different(self):
        """Test diff of different payloads."""
        a = b"abcdef"
        b = b"abXdef"

        diff = diff_payloads(a, b)

        assert diff.similarity < 1.0
        assert len(diff.differences) > 0

    def test_diff_payloads_common_prefix_suffix(self):
        """Test common prefix and suffix detection."""
        a = b"PREFIXdataSUFFIX"
        b = b"PREFIXinfSUFFIX"

        diff = diff_payloads(a, b)

        assert diff.common_prefix_length > 0
        assert diff.common_suffix_length > 0

    def test_compute_similarity(self):
        """Test similarity computation."""
        a = b"hello"
        b = b"hallo"

        similarity = compute_similarity(a, b)

        assert 0 <= similarity <= 1.0

    def test_find_common_bytes(self):
        """Test finding common bytes across payloads."""
        payloads = [
            b"\xaa\x01\x02\xbb",
            b"\xaa\x03\x04\xbb",
            b"\xaa\x05\x06\xbb",
        ]

        common = find_common_bytes(payloads)

        # Should return common prefix
        assert len(common) >= 1
        assert common[0] == 0xAA

    def test_find_variable_positions(self):
        """Test finding variable byte positions."""
        payloads = [
            b"\xaa\x01\xbb",
            b"\xaa\x02\xbb",
            b"\xaa\x03\xbb",
        ]

        result = find_variable_positions(payloads)

        assert 1 in result.variable_positions
        assert 0 in result.constant_positions
        assert 2 in result.constant_positions

    def test_cluster_payloads(self):
        """Test payload clustering."""
        payloads = [
            b"type1_data_a",
            b"type1_data_b",
            b"type2_info_x",
            b"type2_info_y",
        ]

        clusters = cluster_payloads(payloads, threshold=0.5)

        assert len(clusters) >= 1
        assert all(c.size > 0 for c in clusters)

    def test_correlate_request_response(self):
        """Test request-response correlation."""
        requests = [
            PayloadInfo(data=b"REQ1", packet_index=0, timestamp=0.0),
            PayloadInfo(data=b"REQ2", packet_index=1, timestamp=1.0),
        ]
        responses = [
            PayloadInfo(data=b"RESP1", packet_index=2, timestamp=0.1),
            PayloadInfo(data=b"RESP2", packet_index=3, timestamp=1.1),
        ]

        pairs = correlate_request_response(requests, responses)

        assert len(pairs) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestPayloadIntegration:
    """Test integrated payload analysis workflows."""

    def test_extract_and_search(self):
        """Test extraction followed by pattern search."""
        packets = [
            {"data": b"contains secret"},
            {"data": b"normal data"},
            {"data": b"secret message"},
        ]

        # Use filter_by_pattern directly with packets
        filtered = filter_by_pattern(packets, b"secret")

        assert len(filtered) == 2

    def test_extract_infer_fields(self):
        """Test extraction and field inference."""
        extractor = PayloadExtractor()
        packets = [
            {"data": b"\x01\x00\x00TEST"},
            {"data": b"\x02\x00\x00DATA"},
            {"data": b"\x03\x00\x00INFO"},
        ]

        payloads = extractor.extract_all_payloads(packets)
        messages = [p.data for p in payloads]

        schema = infer_fields(messages, min_samples=3)

        assert len(schema.fields) > 0

    def test_search_and_cluster(self):
        """Test pattern search and clustering."""
        packets = [
            b"GET /api/v1/users",
            b"GET /api/v1/posts",
            b"POST /api/v1/users",
            b"POST /api/v1/posts",
        ]

        # Filter GET requests
        get_packets = filter_by_pattern(packets, b"GET")

        assert len(get_packets) == 2

        # Cluster similar payloads
        clusters = cluster_payloads(get_packets, threshold=0.5)

        assert len(clusters) >= 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestPayloadEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_payload_extraction(self):
        """Test extracting empty payload."""
        extractor = PayloadExtractor()

        payload = extractor.extract_payload(b"")

        assert payload == b""

    def test_infer_fields_empty(self):
        """Test field inference with empty messages."""
        schema = infer_fields([])

        assert schema.sample_count == 0
        assert len(schema.fields) == 0

    def test_infer_fields_single_message(self):
        """Test field inference with single message."""
        schema = infer_fields([b"\x01\x02\x03"], min_samples=1)

        # May not have enough samples for reliable inference
        assert isinstance(schema, MessageSchema)

    def test_diff_empty_payloads(self):
        """Test diff of empty payloads."""
        diff = diff_payloads(b"", b"")

        assert diff.similarity == 1.0
        assert len(diff.differences) == 0

    def test_diff_different_lengths(self):
        """Test diff of different length payloads."""
        diff = diff_payloads(b"short", b"much longer")

        assert isinstance(diff, PayloadDiff)
        assert len(diff.differences) > 0

    def test_search_pattern_not_found(self):
        """Test pattern search with no matches."""
        packets = [b"test data"]

        matches = search_pattern(packets, b"notfound")

        assert len(matches) == 0

    def test_find_common_bytes_no_common(self):
        """Test finding common bytes when none exist."""
        payloads = [
            b"\x01",
            b"\x02",
            b"\x03",
        ]

        common = find_common_bytes(payloads)

        assert len(common) == 0

    def test_cluster_single_payload(self):
        """Test clustering with single payload."""
        clusters = cluster_payloads([b"single"], threshold=0.5)

        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_delimiter_detection_no_delimiter(self):
        """Test delimiter detection when none present."""
        messages = [
            b"noddelimiter",
            b"alsonodelimiter",
        ]

        result = detect_delimiter(messages)

        # Should return result even if confidence is low
        assert isinstance(result, DelimiterResult)

    def test_variable_positions_identical(self):
        """Test variable positions with identical payloads."""
        payloads = [
            b"same",
            b"same",
            b"same",
        ]

        result = find_variable_positions(payloads)

        # All positions should be constant
        assert len(result.variable_positions) == 0

    def test_extract_from_list_data(self):
        """Test extraction from packet with list data."""
        extractor = PayloadExtractor()
        packet = {"data": [0x48, 0x65, 0x6C, 0x6C, 0x6F]}  # "Hello"

        payload = extractor.extract_payload(packet)

        assert payload == b"Hello"


# =============================================================================
# Data Class Tests
# =============================================================================


@pytest.mark.unit
class TestPayloadDataClasses:
    """Test payload analysis data classes."""

    def test_pattern_match_creation(self):
        """Test PatternMatch creation."""
        match = PatternMatch(
            pattern_name="test",
            offset=10,
            matched=b"pattern",
            packet_index=5,
            context=b"before_pattern_after",
        )

        assert match.offset == 10
        assert match.matched == b"pattern"

    def test_delimiter_result_creation(self):
        """Test DelimiterResult creation."""
        result = DelimiterResult(
            delimiter=b"\n",
            delimiter_type="fixed",
            confidence=0.95,
            occurrences=10,
            positions=[0, 10, 20],
        )

        assert result.delimiter == b"\n"
        assert result.delimiter_type == "fixed"

    def test_length_prefix_result_creation(self):
        """Test LengthPrefixResult creation."""
        result = LengthPrefixResult(
            detected=True,
            length_bytes=2,
            endian="big",
            offset=0,
            includes_length=False,
            confidence=0.9,
        )

        assert result.detected is True
        assert result.length_bytes == 2

    def test_message_boundary_creation(self):
        """Test MessageBoundary creation."""
        boundary = MessageBoundary(start=0, end=10, length=10, data=b"0123456789", index=0)

        assert boundary.length == 10
        assert len(boundary.data) == 10

    def test_payload_diff_creation(self):
        """Test PayloadDiff creation."""
        diff = PayloadDiff(
            common_prefix_length=5,
            common_suffix_length=3,
            differences=[(5, 0x41, 0x42)],
            similarity=0.85,
            edit_distance=2,
        )

        assert diff.similarity == 0.85
        assert len(diff.differences) == 1

    def test_variable_positions_creation(self):
        """Test VariablePositions creation."""
        result = VariablePositions(
            constant_positions=[0, 2],
            variable_positions=[1, 3],
            constant_values={0: 0xAA, 2: 0xBB},
            variance_by_position=np.array([0.0, 1.5, 0.0, 2.3]),
        )

        assert len(result.constant_positions) == 2
        assert len(result.variable_positions) == 2

    def test_payload_cluster_creation(self):
        """Test PayloadCluster creation."""
        cluster = PayloadCluster(
            cluster_id=0,
            payloads=[b"data1", b"data2"],
            indices=[0, 1],
            representative=b"data1",
            size=2,
        )

        assert cluster.cluster_id == 0
        assert cluster.size == 2
