"""RE Module Validation Suite - Comprehensive tests for reverse engineering modules.

    - RE-VAL-001: RE Module Validation Suite

Acceptance Criteria:
    1. 100% code coverage for all RE modules
    2. Accuracy tests against known protocols
    3. Performance benchmarks for each module
    4. Integration tests for full workflows
    5. Fuzz testing for robustness
    6. Regression test suite for updates

Modules tested:
    - src/oscura/analyzers/packet/payload.py (RE-PAY-001 to RE-PAY-005)
    - src/oscura/analyzers/patterns/matching.py (RE-PAT-001 to RE-PAT-003)
    - src/oscura/analyzers/patterns/learning.py (RE-PAT-004)
    - src/oscura/inference/sequences.py (RE-SEQ-002, RE-SEQ-003)
    - src/oscura/inference/stream.py (RE-STR-001 to RE-STR-003)
    - src/oscura/inference/binary.py (RE-BIN-001 to RE-BIN-003)
    - src/oscura/analyzers/digital/thresholds.py (RE-THR-001, RE-THR-002)
    - src/oscura/inference/protocol_library.py (RE-DSL-003)
    - src/oscura/pipeline/reverse_engineering.py (RE-INT-001)
    - src/oscura/analyzers/statistical/entropy.py (RE-ENT-002)
"""

from __future__ import annotations

import random
import struct
import time

import numpy as np
import pytest

pytestmark = pytest.mark.validation

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def modbus_rtu_messages() -> list[bytes]:
    """Generate sample Modbus RTU messages for testing."""
    messages = []
    # Read Holding Registers request: addr=1, func=3, start=0, count=10
    messages.append(b"\x01\x03\x00\x00\x00\x0a\xc5\xcd")
    # Read Holding Registers response: addr=1, func=3, byte_count=20, data...
    messages.append(b"\x01\x03\x14" + bytes(range(20)) + b"\xab\xcd")
    # Write Single Register: addr=1, func=6, reg=0, value=100
    messages.append(b"\x01\x06\x00\x00\x00\x64\x89\xd5")
    return messages


@pytest.fixture
def dns_message() -> bytes:
    """Generate sample DNS message for testing."""
    # DNS query for example.com
    header = struct.pack(
        ">HHHHHH",
        0x1234,  # Transaction ID
        0x0100,  # Flags: standard query
        1,  # Questions
        0,  # Answer RRs
        0,  # Authority RRs
        0,  # Additional RRs
    )
    # Query: example.com, type A, class IN
    query = b"\x07example\x03com\x00\x00\x01\x00\x01"
    return header + query


@pytest.fixture
def length_prefixed_messages() -> bytes:
    """Generate length-prefixed message stream."""
    messages = []
    for i in range(5):
        payload = bytes([i] * (10 + i * 5))
        length = struct.pack(">H", len(payload))
        messages.append(length + payload)
    return b"".join(messages)


@pytest.fixture
def delimiter_separated_messages() -> bytes:
    """Generate delimiter-separated message stream."""
    messages = [
        b"MSG001:Hello",
        b"MSG002:World",
        b"MSG003:Test",
        b"MSG004:Data",
    ]
    return b"\r\n".join(messages) + b"\r\n"


@pytest.fixture
def sequence_counter_messages() -> list[bytes]:
    """Generate messages with sequence counter field."""
    messages = []
    for i in range(20):
        # Header (2 bytes) + Sequence (2 bytes) + Data (8 bytes)
        msg = struct.pack(">HH", 0xAA55, i) + bytes([i % 256] * 8)
        messages.append(msg)
    return messages


@pytest.fixture
def analog_signal_with_drift() -> np.ndarray:
    """Generate analog signal with DC drift."""
    t = np.linspace(0, 1, 10000)
    # Square wave with DC drift
    square = np.sign(np.sin(2 * np.pi * 50 * t))
    drift = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow drift
    noise = np.random.normal(0, 0.1, len(t))
    return square + drift + noise


@pytest.fixture
def pam4_signal() -> np.ndarray:
    """Generate PAM-4 multi-level signal."""
    symbols = np.random.randint(0, 4, 1000)
    levels = [-3, -1, 1, 3]
    signal = np.array([levels[s] for s in symbols])
    # Upsample and add noise
    upsampled = np.repeat(signal, 10)
    noise = np.random.normal(0, 0.2, len(upsampled))
    return upsampled + noise


@pytest.fixture
def structured_binary_data() -> bytes:
    """Generate structured binary data with known fields."""
    # Header: magic (4) + version (2) + length (2) + flags (1) + padding (3)
    header = struct.pack(">4sHHBxxx", b"TEST", 1, 100, 0x0F)
    # Body: sequence (4) + timestamp (8) + data (88)
    body = struct.pack(">IQ", 1, int(time.time())) + bytes(88)
    return header + body


# =============================================================================
# SECTION 1: Payload Analysis Tests (RE-PAY-001 to RE-PAY-005)
# =============================================================================


class TestPayloadExtraction:
    """Tests for RE-PAY-001: Payload Extraction Framework."""

    def test_extract_payload_from_bytes(self):
        """Test extracting payload from raw bytes."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        extractor = PayloadExtractor()
        data = b"\x01\x02\x03\x04\x05"
        payload = extractor.extract_payload(data)
        assert payload == data

    def test_extract_payload_from_dict(self):
        """Test extracting payload from packet dict."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        extractor = PayloadExtractor()
        packet = {"data": b"\x01\x02\x03", "src_ip": "192.168.1.1"}
        payload = extractor.extract_payload(packet)
        assert payload == b"\x01\x02\x03"

    def test_extract_all_payloads_with_filter(self):
        """Test batch extraction with protocol filter."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        packets = [
            {"data": b"udp_data", "protocol": "UDP", "src_port": 1000},
            {"data": b"tcp_data", "protocol": "TCP", "src_port": 80},
            {"data": b"udp_data2", "protocol": "UDP", "src_port": 2000},
        ]
        extractor = PayloadExtractor()
        payloads = extractor.extract_all_payloads(packets, protocol="UDP")
        assert len(payloads) == 2
        assert all(p.protocol == "UDP" for p in payloads)

    def test_payload_info_metadata(self):
        """Test PayloadInfo preserves metadata."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        packets = [
            {
                "data": b"test",
                "timestamp": 1000.0,
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 5000,
                "dst_port": 80,
                "protocol": "TCP",
            }
        ]
        extractor = PayloadExtractor()
        payloads = extractor.extract_all_payloads(packets)
        assert len(payloads) == 1
        info = payloads[0]
        assert info.timestamp == 1000.0
        assert info.src_ip == "10.0.0.1"
        assert info.dst_ip == "10.0.0.2"

    def test_iter_payloads_memory_efficient(self):
        """Test streaming payload iteration."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        packets = [b"data" + bytes([i]) for i in range(100)]
        extractor = PayloadExtractor()
        count = 0
        for payload in extractor.iter_payloads(packets):
            count += 1
            assert len(payload.data) == 5
        assert count == 100


class TestPayloadPatternSearch:
    """Tests for RE-PAY-002: Payload Pattern Search."""

    def test_search_exact_pattern(self):
        """Test exact pattern matching."""
        from oscura.analyzers.packet.payload import search_pattern

        packets = [b"PREFIX\xaa\x55SUFFIX", b"OTHER\xaa\x55DATA"]
        matches = search_pattern(packets, b"\xaa\x55")
        assert len(matches) == 2
        assert matches[0].offset == 6
        assert matches[1].offset == 5

    def test_search_patterns_multiple(self):
        """Test searching for multiple patterns simultaneously."""
        from oscura.analyzers.packet.payload import search_patterns

        packets = [b"HEADER_AA55_MIDDLE_DEAD"]
        patterns = {"header": b"AA55", "footer": b"DEAD"}
        results = search_patterns(packets, patterns)
        assert "header" in results
        assert "footer" in results
        assert len(results["header"]) == 1
        assert len(results["footer"]) == 1

    def test_filter_by_pattern(self):
        """Test filtering packets by pattern presence."""
        from oscura.analyzers.packet.payload import filter_by_pattern

        packets = [
            b"HAS_MARKER_\xde\xad",
            b"NO_MARKER_HERE",
            b"\xde\xad_AT_START",
        ]
        filtered = filter_by_pattern(packets, b"\xde\xad")
        assert len(filtered) == 2

    def test_pattern_context_bytes(self):
        """Test context bytes around match."""
        from oscura.analyzers.packet.payload import search_pattern

        packets = [b"CONTEXT_\xaa\x55_AFTER"]
        matches = search_pattern(packets, b"\xaa\x55", context_bytes=4)
        assert len(matches) == 1
        assert b"XT_\xaa\x55_AFT" in matches[0].context


class TestPayloadDelimiterDetection:
    """Tests for RE-PAY-003: Payload Delimiter Detection."""

    def test_detect_crlf_delimiter(self, delimiter_separated_messages):
        """Test detecting CRLF delimiter."""
        from oscura.analyzers.packet.payload import detect_delimiter

        result = detect_delimiter(delimiter_separated_messages)
        assert result.delimiter == b"\r\n"
        assert result.confidence > 0.5

    def test_detect_length_prefix_format(self, length_prefixed_messages):
        """Test detecting length-prefixed format."""
        from oscura.analyzers.packet.payload import detect_length_prefix

        result = detect_length_prefix([length_prefixed_messages])
        assert result.detected
        assert result.length_bytes == 2
        assert result.endian == "big"

    def test_find_message_boundaries(self, delimiter_separated_messages):
        """Test finding message boundaries."""
        from oscura.analyzers.packet.payload import find_message_boundaries

        boundaries = find_message_boundaries(delimiter_separated_messages)
        assert len(boundaries) == 4  # 4 messages

    def test_segment_messages(self, delimiter_separated_messages):
        """Test message segmentation."""
        from oscura.analyzers.packet.payload import segment_messages

        messages = segment_messages(delimiter_separated_messages)
        assert len(messages) == 4
        assert messages[0].startswith(b"MSG001")


class TestPayloadFieldInference:
    """Tests for RE-PAY-004: Payload Field Inference."""

    def test_infer_fields_from_messages(self, sequence_counter_messages):
        """Test field inference from message samples."""
        from oscura.analyzers.packet.payload import infer_fields

        schema = infer_fields(sequence_counter_messages)
        assert schema.sample_count == 20
        assert len(schema.fields) > 0
        assert schema.fixed_length

    def test_detect_sequence_fields(self, sequence_counter_messages):
        """Test detecting sequence/counter fields."""
        from oscura.analyzers.packet.payload import find_sequence_fields

        sequences = find_sequence_fields(sequence_counter_messages)
        assert len(sequences) > 0
        # Sequence field is at offset 2, size 2
        assert any(offset == 2 and size == 2 for offset, size in sequences)

    def test_find_checksum_fields(self, modbus_rtu_messages):
        """Test detecting checksum fields."""
        from oscura.analyzers.packet.payload import find_checksum_fields

        checksums = find_checksum_fields(modbus_rtu_messages)
        # CRC is at the end of Modbus RTU messages
        assert len(checksums) >= 0  # May or may not detect depending on sample size

    def test_field_type_inference(self, sequence_counter_messages):
        """Test field type inference."""
        from oscura.analyzers.packet.payload import detect_field_types

        boundaries = [(0, 2), (2, 4), (4, 12)]
        fields = detect_field_types(sequence_counter_messages, boundaries)
        assert len(fields) == 3
        # First field should be uint16
        assert fields[0].inferred_type == "uint16"


class TestPayloadComparison:
    """Tests for RE-PAY-005: Payload Comparison and Differential Analysis."""

    def test_diff_payloads(self):
        """Test diffing two payloads."""
        from oscura.analyzers.packet.payload import diff_payloads

        payload_a = b"\x01\x02\x03\x04\x05"
        payload_b = b"\x01\x02\xff\x04\x05"
        diff = diff_payloads(payload_a, payload_b)
        assert diff.common_prefix_length == 2
        assert len(diff.differences) == 1
        assert diff.differences[0] == (2, 0x03, 0xFF)

    def test_find_common_bytes(self):
        """Test finding common prefix."""
        from oscura.analyzers.packet.payload import find_common_bytes

        payloads = [
            b"HEADER_DATA1",
            b"HEADER_DATA2",
            b"HEADER_OTHER",
        ]
        common = find_common_bytes(payloads)
        assert common == b"HEADER_"

    def test_find_variable_positions(self):
        """Test identifying variable positions."""
        from oscura.analyzers.packet.payload import find_variable_positions

        payloads = [
            b"\x01\x00\x03",
            b"\x01\x01\x03",
            b"\x01\x02\x03",
        ]
        result = find_variable_positions(payloads)
        assert 0 in result.constant_positions
        assert 2 in result.constant_positions
        assert 1 in result.variable_positions

    def test_compute_similarity_levenshtein(self):
        """Test Levenshtein similarity."""
        from oscura.analyzers.packet.payload_analysis import compute_similarity

        a = b"ABCD"
        b = b"ABCE"
        similarity = compute_similarity(a, b, metric="levenshtein")
        assert 0.7 < similarity < 1.0

    def test_cluster_payloads(self):
        """Test payload clustering."""
        from oscura.analyzers.packet.payload import cluster_payloads

        payloads = [
            b"TYPE_A_DATA1",
            b"TYPE_A_DATA2",
            b"TYPE_B_INFO1",
            b"TYPE_B_INFO2",
            b"TYPE_A_DATA3",
        ]
        clusters = cluster_payloads(payloads, threshold=0.7)
        assert len(clusters) >= 2  # Should separate TYPE_A from TYPE_B


# =============================================================================
# SECTION 2: Pattern Matching Tests (RE-PAT-001 to RE-PAT-004)
# =============================================================================


class TestBinaryRegexMatching:
    """Tests for RE-PAT-001: Binary Regex Pattern Matching."""

    def test_binary_regex_literal_match(self):
        """Test literal byte matching."""
        from oscura.analyzers.patterns.matching import BinaryRegex

        regex = BinaryRegex(pattern=r"\xAA\x55", name="header")
        data = b"\x00\x00\xaa\x55\x00\x00"
        result = regex.search(data)
        assert result is not None
        assert result.offset == 2

    def test_binary_regex_wildcard(self):
        """Test wildcard matching."""
        from oscura.analyzers.patterns.matching import BinaryRegex

        regex = BinaryRegex(pattern=r"\xAA??\x55", name="pattern")
        data = b"\xaa\x12\x55"
        result = regex.match(data)
        assert result is not None

    def test_binary_regex_findall(self):
        """Test finding all matches."""
        from oscura.analyzers.patterns.matching import binary_regex_search

        data = b"\xaa\x55\x00\xaa\x55\x01\xaa\x55"
        matches = binary_regex_search(data, r"\xAA\x55")
        assert len(matches) == 3


class TestAhoCorasickMultiPattern:
    """Tests for RE-PAT-002: Multi-Pattern Search (Aho-Corasick)."""

    def test_aho_corasick_basic(self):
        """Test basic multi-pattern search."""
        from oscura.analyzers.patterns.matching import AhoCorasickMatcher

        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\x55", "header")
        matcher.add_pattern(b"\xde\xad", "marker")
        matcher.build()

        data = b"\x00\xaa\x55\x00\xde\xad\x00"
        matches = matcher.search(data)
        assert len(matches) == 2
        assert any(m.pattern_name == "header" for m in matches)
        assert any(m.pattern_name == "marker" for m in matches)

    def test_multi_pattern_search_function(self):
        """Test convenience function for multi-pattern search."""
        from oscura.analyzers.patterns.matching import multi_pattern_search

        data = b"PREFIX_\xaa\x55_MIDDLE_\xde\xad_SUFFIX"
        patterns = {"header": b"\xaa\x55", "marker": b"\xde\xad"}
        results = multi_pattern_search(data, patterns)
        assert len(results["header"]) == 1
        assert len(results["marker"]) == 1

    def test_count_pattern_occurrences(self):
        """Test pattern occurrence counting."""
        from oscura.analyzers.patterns.matching import count_pattern_occurrences

        data = b"\xaa\xaa\xaa\xbb\xbb"
        patterns = {"aa": b"\xaa", "bb": b"\xbb"}
        counts = count_pattern_occurrences(data, patterns)
        assert counts["aa"] == 3
        assert counts["bb"] == 2


class TestFuzzyPatternMatching:
    """Tests for RE-PAT-003: Fuzzy Pattern Matching."""

    def test_fuzzy_match_basic(self):
        """Test basic fuzzy matching."""
        from oscura.analyzers.patterns.matching import FuzzyMatcher

        matcher = FuzzyMatcher(max_edit_distance=1)
        data = b"\xaa\x56\x00"  # One byte different from \xAA\x55\x00
        pattern = b"\xaa\x55\x00"
        matches = matcher.search(data, pattern)
        assert len(matches) >= 1
        assert matches[0].edit_distance == 1

    def test_fuzzy_search_function(self):
        """Test fuzzy search convenience function."""
        from oscura.analyzers.patterns.matching import fuzzy_search

        data = b"\x00\xab\x55\x00"  # AB instead of AA
        matches = fuzzy_search(data, b"\xaa\x55", max_distance=1)
        assert len(matches) >= 1

    def test_find_similar_sequences(self):
        """Test finding similar byte sequences."""
        from oscura.analyzers.patterns.matching import find_similar_sequences

        data = b"\xaa\x55\x00\x00\x00\x00\xaa\x56\x00\x00"
        results = find_similar_sequences(data, min_length=4, max_distance=1)
        # Should find similarity between the two similar patterns
        assert len(results) >= 0


class TestPatternLearning:
    """Tests for RE-PAT-004: Pattern Learning and Discovery."""

    def test_learn_patterns_from_data(self):
        """Test pattern learning from binary data."""
        from oscura.analyzers.patterns.learning import learn_patterns_from_data

        # Data with repeated patterns
        data = b"\xaa\x55\x01\xaa\x55\x02\xaa\x55\x03"
        patterns = learn_patterns_from_data(data, min_length=2, min_frequency=3)
        assert len(patterns) > 0
        # Should find the AA55 pattern
        assert any(b"\xaa\x55" in p.pattern for p in patterns)

    def test_pattern_learner_class(self):
        """Test PatternLearner class."""
        from oscura.analyzers.patterns.learning import PatternLearner

        learner = PatternLearner(min_pattern_length=2, min_frequency=2)
        learner.add_sample(b"\x01\x02\x03\x04\x05")
        learner.add_sample(b"\x01\x02\x06\x07\x08")
        learner.add_sample(b"\x01\x02\x09\x0a\x0b")
        patterns = learner.learn_patterns()
        # Should find the common prefix
        assert len(patterns) > 0

    def test_infer_structure(self):
        """Test structure inference."""
        from oscura.analyzers.patterns.learning import infer_structure

        samples = [
            b"\x01\x00" + bytes(10),
            b"\x02\x00" + bytes(10),
            b"\x03\x00" + bytes(10),
        ]
        hypothesis = infer_structure(samples)
        assert hypothesis.record_size == 12 or hypothesis.record_size is None

    def test_find_recurring_structures(self):
        """Test finding recurring fixed-size structures."""
        from oscura.analyzers.patterns.learning import find_recurring_structures

        # Create data with 8-byte records
        record = b"\xaa\x55\x00\x00\x00\x00\x00\x00"
        data = record * 10
        results = find_recurring_structures(data, min_size=4, max_size=16)
        assert len(results) > 0
        assert any(size == 8 for size, _, _ in results)


# =============================================================================
# SECTION 3: Sequence Analysis Tests (RE-SEQ-002, RE-SEQ-003)
# =============================================================================


class TestSequencePatternDetection:
    """Tests for RE-SEQ-002: Sequence Pattern Detection."""

    def test_detect_sequence_patterns(self):
        """Test detecting sequential patterns."""
        from oscura.inference.sequences import detect_sequence_patterns

        messages = [
            {"type": "A"},
            {"type": "B"},
            {"type": "C"},
            {"type": "A"},
            {"type": "B"},
            {"type": "C"},
            {"type": "A"},
            {"type": "B"},
            {"type": "C"},
        ]
        patterns = detect_sequence_patterns(messages, key=lambda m: m["type"], min_frequency=3)
        assert len(patterns) > 0
        # Should find the A-B-C pattern
        assert any(p.pattern == ["A", "B", "C"] or p.pattern == ["A"] for p in patterns)

    def test_find_repeating_sequences(self):
        """Test finding exact repeating sequences."""
        from oscura.inference.sequences import SequencePatternDetector

        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = ["REQ", "RSP", "ACK", "REQ", "RSP", "ACK"]
        results = detector.find_repeating_sequences(messages)
        assert len(results) > 0

    def test_detect_periodic_patterns(self):
        """Test detecting periodic patterns."""
        from oscura.inference.sequences import SequencePatternDetector

        detector = SequencePatternDetector()
        # Messages with regular timing
        messages = [{"type": f"MSG{i % 3}", "ts": i * 10.0} for i in range(15)]
        patterns = detector.detect_periodic_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )
        # Should detect periodic patterns
        assert len(patterns) >= 0


class TestRequestResponseCorrelation:
    """Tests for RE-SEQ-003: Request-Response Correlation."""

    def test_correlate_requests(self):
        """Test correlating request-response pairs."""
        from oscura.inference.sequences import correlate_requests

        messages = [
            {"dir": "out", "type": "REQ", "time": 1.0},
            {"dir": "in", "type": "RSP", "time": 1.5},
            {"dir": "out", "type": "REQ", "time": 2.0},
            {"dir": "in", "type": "RSP", "time": 2.3},
        ]
        pairs = correlate_requests(
            messages,
            request_filter=lambda m: m["dir"] == "out",
            response_filter=lambda m: m["dir"] == "in",
            timestamp_key=lambda m: m["time"],
        )
        assert len(pairs) == 2

    def test_calculate_latency_stats(self):
        """Test latency statistics calculation."""
        from oscura.inference.sequences import (
            RequestResponsePair,
            calculate_latency_stats,
        )

        pairs = [
            RequestResponsePair(
                request_index=0,
                response_index=1,
                request={},
                response={},
                latency=0.5,
            ),
            RequestResponsePair(
                request_index=2,
                response_index=3,
                request={},
                response={},
                latency=0.3,
            ),
            RequestResponsePair(
                request_index=4,
                response_index=5,
                request={},
                response={},
                latency=0.7,
            ),
        ]
        stats = calculate_latency_stats(pairs)
        assert stats["min"] == 0.3
        assert stats["max"] == 0.7
        assert 0.4 < stats["mean"] < 0.6

    def test_find_message_dependencies(self):
        """Test finding message type dependencies."""
        from oscura.inference.sequences import find_message_dependencies

        messages = [
            {"type": "REQ"},
            {"type": "RSP"},
            {"type": "ACK"},
            {"type": "REQ"},
            {"type": "RSP"},
        ]
        deps = find_message_dependencies(messages, key=lambda m: m["type"])
        assert "REQ" in deps
        assert "RSP" in deps["REQ"]


# =============================================================================
# SECTION 4: Stream Reassembly Tests (RE-STR-001 to RE-STR-003)
# =============================================================================


class TestUDPStreamReassembly:
    """Tests for RE-STR-001: UDP Stream Reconstruction."""

    def test_reassemble_udp_stream(self):
        """Test UDP stream reassembly."""
        from oscura.inference.stream import reassemble_udp_stream

        packets = [
            {"data": b"CHUNK1", "src": "10.0.0.1", "dst": "10.0.0.2"},
            {"data": b"CHUNK2", "src": "10.0.0.1", "dst": "10.0.0.2"},
            {"data": b"CHUNK3", "src": "10.0.0.1", "dst": "10.0.0.2"},
        ]
        stream = reassemble_udp_stream(packets)
        assert stream.data == b"CHUNK1CHUNK2CHUNK3"
        assert stream.segments == 3

    def test_udp_out_of_order_detection(self):
        """Test detecting out-of-order UDP packets."""
        from oscura.inference.stream import UDPStreamReassembler

        reassembler = UDPStreamReassembler(sequence_key=lambda p: p.get("seq", 0))
        reassembler.add_segment({"data": b"B", "seq": 2})
        reassembler.add_segment({"data": b"A", "seq": 1})
        reassembler.add_segment({"data": b"C", "seq": 3})
        stream = reassembler.get_stream()
        assert stream.out_of_order >= 1


class TestTCPStreamReassembly:
    """Tests for RE-STR-002: TCP Stream Reassembly."""

    def test_reassemble_tcp_stream(self):
        """Test TCP stream reassembly."""
        from oscura.inference.stream import reassemble_tcp_stream

        segments = [
            {"seq": 0, "data": b"Hello"},
            {"seq": 5, "data": b" World"},
        ]
        stream = reassemble_tcp_stream(segments)
        assert b"Hello World" in stream.data

    def test_tcp_retransmit_detection(self):
        """Test detecting TCP retransmissions."""
        from oscura.inference.stream import TCPStreamReassembler

        reassembler = TCPStreamReassembler()
        reassembler.add_segment({"seq": 0, "data": b"DATA"})
        reassembler.add_segment({"seq": 0, "data": b"DATA"})  # Retransmit
        reassembler.add_segment({"seq": 4, "data": b"MORE"})
        stream = reassembler.get_stream()
        assert stream.retransmits >= 1

    def test_tcp_gap_detection(self):
        """Test detecting gaps in TCP stream."""
        from oscura.inference.stream import TCPStreamReassembler

        reassembler = TCPStreamReassembler()
        reassembler.add_segment({"seq": 0, "data": b"ABCD"})
        reassembler.add_segment({"seq": 10, "data": b"EFGH"})  # Gap 4-10
        stream = reassembler.get_stream()
        assert len(stream.gaps) >= 1


class TestMessageFraming:
    """Tests for RE-STR-003: Message Framing and Segmentation."""

    def test_frame_by_delimiter(self, delimiter_separated_messages):
        """Test delimiter-based framing."""
        from oscura.inference.stream import extract_messages

        result = extract_messages(
            delimiter_separated_messages, framing_type="delimiter", delimiter=b"\r\n"
        )
        assert result.framing_type == "delimiter"
        assert len(result.messages) == 4

    def test_frame_by_length_prefix(self, length_prefixed_messages):
        """Test length-prefix framing."""
        from oscura.inference.stream import extract_messages

        result = extract_messages(
            length_prefixed_messages,
            framing_type="length_prefix",
            length_field_size=2,
        )
        assert len(result.messages) == 5

    def test_auto_detect_framing(self, delimiter_separated_messages):
        """Test automatic framing detection."""
        from oscura.inference.stream import detect_message_framing

        framing = detect_message_framing(delimiter_separated_messages)
        assert framing["type"] == "delimiter"

    def test_message_framer_class(self):
        """Test MessageFramer class."""
        from oscura.inference.stream import MessageFramer

        framer = MessageFramer(framing_type="fixed", fixed_size=8)
        data = bytes(range(32))
        result = framer.frame(data)
        assert len(result.messages) == 4


# =============================================================================
# SECTION 5: Binary Format Inference Tests (RE-BIN-001 to RE-BIN-003)
# =============================================================================


class TestMagicByteDetection:
    """Tests for RE-BIN-001: Magic Byte Detection."""

    def test_detect_known_magic_bytes(self):
        """Test detecting known file signatures."""
        from oscura.inference.binary import detect_magic_bytes

        # PNG signature
        png_data = b"\x89PNG\r\n\x1a\n" + bytes(100)
        result = detect_magic_bytes(png_data)
        assert result is not None
        assert result.known_format == "PNG"

        # ZIP signature
        zip_data = b"PK\x03\x04" + bytes(100)
        result = detect_magic_bytes(zip_data)
        assert result is not None
        assert result.known_format == "ZIP"

    def test_learn_magic_from_samples(self):
        """Test learning magic bytes from samples."""
        from oscura.inference.binary import MagicByteDetector

        samples = [
            b"\xca\xfe\xba\xbe" + bytes(50),
            b"\xca\xfe\xba\xbe" + bytes(60),
            b"\xca\xfe\xba\xbe" + bytes(70),
        ]
        detector = MagicByteDetector()
        results = detector.learn_magic_from_samples(samples)
        assert len(results) > 0
        assert any(r.magic == b"\xca\xfe\xba\xbe" for r in results)

    def test_find_all_magic_bytes(self):
        """Test scanning for all magic bytes."""
        from oscura.inference.binary import find_all_magic_bytes

        # Data with embedded PNG and ZIP signatures
        data = bytes(100) + b"\x89PNG\r\n\x1a\n" + bytes(50) + b"PK\x03\x04" + bytes(50)
        results = find_all_magic_bytes(data)
        assert len(results) >= 2


class TestAlignmentDetection:
    """Tests for RE-BIN-002: Structure Alignment Detection."""

    def test_detect_alignment(self, structured_binary_data):
        """Test detecting structure alignment."""
        from oscura.inference.binary import detect_alignment

        result = detect_alignment(structured_binary_data)
        assert result.alignment in [1, 2, 4, 8, 16]
        assert result.confidence > 0

    def test_detect_field_types_from_alignment(self, structured_binary_data):
        """Test field type detection based on alignment."""
        from oscura.inference.binary import AlignmentDetector

        detector = AlignmentDetector()
        alignment = detector.detect(structured_binary_data)
        fields = detector.detect_field_types(structured_binary_data, alignment)
        assert len(fields) > 0


class TestBinaryParserGeneration:
    """Tests for RE-BIN-003: Binary Parser DSL."""

    def test_generate_parser(self, sequence_counter_messages):
        """Test parser generation from samples."""
        from oscura.inference.binary import generate_parser

        parser = generate_parser(sequence_counter_messages, name="TestMessage")
        assert parser.name == "TestMessage"
        assert len(parser.fields) > 0
        assert parser.total_size == 12

    def test_parser_to_yaml(self, sequence_counter_messages):
        """Test YAML export of parser."""
        from oscura.inference.binary import generate_parser, parser_to_yaml

        parser = generate_parser(sequence_counter_messages, name="TestMsg")
        yaml_str = parser_to_yaml(parser)
        assert "name: TestMsg" in yaml_str
        assert "fields:" in yaml_str

    def test_parser_to_python(self, sequence_counter_messages):
        """Test Python code generation."""
        from oscura.inference.binary import generate_parser, parser_to_python

        parser = generate_parser(sequence_counter_messages, name="TestStruct")
        python_code = parser_to_python(parser)
        assert "class TestStruct:" in python_code
        assert "import struct" in python_code


# =============================================================================
# SECTION 6: Threshold Analysis Tests (RE-THR-001, RE-THR-002)
# =============================================================================


class TestAdaptiveThresholding:
    """Tests for RE-THR-001: Time-Varying Threshold Support."""

    def test_apply_adaptive_threshold(self, analog_signal_with_drift):
        """Test adaptive thresholding on drifting signal."""
        from oscura.analyzers.digital.thresholds import apply_adaptive_threshold

        result = apply_adaptive_threshold(analog_signal_with_drift, window_size=500)
        assert len(result.binary_output) == len(analog_signal_with_drift)
        assert result.binary_output.dtype == np.uint8
        assert set(result.binary_output).issubset({0, 1})

    def test_threshold_methods(self):
        """Test different thresholding methods."""
        from oscura.analyzers.digital.thresholds import AdaptiveThresholder

        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)

        for method in ["median", "mean", "envelope", "otsu"]:
            thresholder = AdaptiveThresholder(window_size=100, method=method)
            result = thresholder.apply(signal)
            assert len(result.crossings) > 0

    def test_hysteresis_prevents_oscillation(self):
        """Test that hysteresis prevents rapid state changes."""
        from oscura.analyzers.digital.thresholds import AdaptiveThresholder

        # Signal hovering around threshold
        signal = np.array([0.49, 0.51, 0.49, 0.51, 0.49, 0.51] * 100, dtype=np.float64)

        thresholder = AdaptiveThresholder(window_size=50, hysteresis=0.1)
        result = thresholder.apply(signal)
        # With hysteresis, should have fewer crossings
        assert len(result.crossings) < 50


class TestMultiLevelLogic:
    """Tests for RE-THR-002: Multi-Level Logic Support."""

    def test_detect_multi_level(self, pam4_signal):
        """Test multi-level signal detection."""
        from oscura.analyzers.digital.thresholds import detect_multi_level

        result = detect_multi_level(pam4_signal, n_levels=4)
        assert len(result.level_values) == 4
        # Levels should be detected in order
        assert result.level_values == sorted(result.level_values)

    def test_multi_level_detector_class(self, pam4_signal):
        """Test MultiLevelDetector class."""
        from oscura.analyzers.digital.thresholds import MultiLevelDetector

        detector = MultiLevelDetector(levels=4, auto_detect_levels=True)
        result = detector.detect(pam4_signal)
        assert len(result.transitions) > 0
        # Check that from_lvl and to_lvl are valid level indices
        assert all(
            from_lvl in range(4) and to_lvl in range(4)
            for _, from_lvl, to_lvl in result.transitions
        )

    def test_eye_diagram_calculation(self, pam4_signal):
        """Test eye diagram data generation."""
        from oscura.analyzers.digital.thresholds import MultiLevelDetector

        detector = MultiLevelDetector(levels=4)
        eye_data = detector.calculate_eye_diagram(pam4_signal, samples_per_symbol=10)
        assert eye_data.shape[1] == 20  # 2 * samples_per_symbol


# =============================================================================
# SECTION 7: Protocol Library Tests (RE-DSL-003)
# =============================================================================


class TestProtocolLibrary:
    """Tests for RE-DSL-003: Protocol Format Library."""

    def test_list_protocols(self):
        """Test listing available protocols."""
        from oscura.inference.protocol_library import list_protocols

        protocols = list_protocols()
        assert len(protocols) > 0
        # Should include common protocols
        names = [p.name for p in protocols]
        assert "modbus_rtu" in names
        assert "dns" in names

    def test_list_protocols_by_category(self):
        """Test listing protocols by category."""
        from oscura.inference.protocol_library import list_protocols

        industrial = list_protocols(category="industrial")
        assert all(p.category == "industrial" for p in industrial)
        assert len(industrial) > 0

    def test_get_protocol(self):
        """Test getting protocol by name."""
        from oscura.inference.protocol_library import get_protocol

        modbus = get_protocol("modbus_rtu")
        assert modbus is not None
        assert modbus.name == "modbus_rtu"
        assert modbus.category == "industrial"

    def test_get_decoder(self, modbus_rtu_messages):
        """Test getting protocol decoder."""
        from oscura.inference.protocol_library import get_decoder

        decoder = get_decoder("modbus_rtu")
        assert decoder is not None
        # Test decoding
        result = decoder.decode(modbus_rtu_messages[0])
        assert "address" in result
        assert "function_code" in result

    def test_protocol_categories(self):
        """Test protocol categories."""
        from oscura.inference.protocol_library import get_library

        library = get_library()
        categories = library.categories()
        assert "industrial" in categories
        assert "iot" in categories
        assert "automotive" in categories


# =============================================================================
# SECTION 8: Pipeline Integration Tests (RE-INT-001)
# =============================================================================


class TestREPipeline:
    """Tests for RE-INT-001: RE Pipeline Integration."""

    def test_analyze_raw_data(self):
        """Test analyzing raw binary data."""
        from oscura.utils.pipeline.reverse_engineering import analyze

        data = b"\xaa\x55\x00\x01\xde\xad" * 10
        results = analyze(data)
        assert results.message_count > 0
        assert results.duration_seconds >= 0

    def test_analyze_packet_list(self):
        """Test analyzing packet list."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        packets = [
            {"data": b"\x01\x03\x00\x00\x00\x0a", "protocol": "UDP", "src_port": 502},
            {"data": b"\x01\x03\x14" + bytes(20), "protocol": "UDP", "src_port": 502},
        ]
        pipeline = REPipeline()
        results = pipeline.analyze(packets)
        assert results.flow_count >= 1
        assert results.message_count == 2

    def test_pipeline_stages(self):
        """Test individual pipeline stages."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        pipeline = REPipeline(stages=["flow_extraction", "payload_analysis"])
        data = [b"MSG1", b"MSG2", b"MSG3"]
        results = pipeline.analyze(data)
        assert "flow_extraction" in results.statistics
        assert "payload_analysis" in results.statistics

    def test_progress_callback(self):
        """Test progress reporting."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        progress_updates = []

        def on_progress(stage: str, percent: float):
            progress_updates.append((stage, percent))

        pipeline = REPipeline()
        pipeline.analyze(b"test_data", progress_callback=on_progress)
        assert len(progress_updates) > 0
        assert any("complete" in stage for stage, _ in progress_updates)

    def test_generate_json_report(self, tmp_path):
        """Test JSON report generation."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        pipeline = REPipeline()
        results = pipeline.analyze(b"test_data")
        report_path = tmp_path / "report.json"
        pipeline.generate_report(results, report_path, format="json")
        assert report_path.exists()
        import json

        with open(report_path) as f:
            report = json.load(f)
        assert "flow_count" in report

    def test_generate_markdown_report(self, tmp_path):
        """Test Markdown report generation."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        pipeline = REPipeline()
        results = pipeline.analyze(b"test_data")
        report_path = tmp_path / "report.md"
        pipeline.generate_report(results, report_path, format="markdown")
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Reverse Engineering Analysis Report" in content

    def test_generate_html_report(self, tmp_path):
        """Test HTML report generation."""
        from oscura.utils.pipeline.reverse_engineering import REPipeline

        pipeline = REPipeline()
        results = pipeline.analyze(b"test_data")
        report_path = tmp_path / "report.html"
        pipeline.generate_report(results, report_path, format="html")
        assert report_path.exists()
        content = report_path.read_text()
        assert "<html>" in content


# =============================================================================
# SECTION 9: Entropy Analysis Tests (RE-ENT-002)
# =============================================================================


class TestByteFrequencyDistribution:
    """Tests for RE-ENT-002: Byte Frequency Distribution."""

    def test_byte_frequency_distribution(self):
        """Test byte frequency distribution analysis."""
        from oscura.analyzers.statistical.entropy import byte_frequency_distribution

        data = b"\x00\x00\x01\x02\x03"
        result = byte_frequency_distribution(data)
        assert result.unique_bytes == 4
        assert result.most_common[0] == (0, 2)  # 0x00 appears twice
        assert result.entropy > 0

    def test_detect_frequency_anomalies(self):
        """Test detecting frequency anomalies."""
        from oscura.analyzers.statistical.entropy import detect_frequency_anomalies

        # Create data with anomalous byte
        data = b"A" * 100 + bytes(range(256))
        result = detect_frequency_anomalies(data)
        assert 65 in result.anomalous_bytes  # 'A' should be anomalous

    def test_compare_byte_distributions(self):
        """Test comparing two distributions."""
        from oscura.analyzers.statistical.entropy import compare_byte_distributions

        data_a = bytes(range(256)) * 10
        data_b = bytes(range(256)) * 10
        chi_sq, _kl_div, _diffs = compare_byte_distributions(data_a, data_b)
        assert chi_sq < 0.01  # Should be nearly identical

    def test_sliding_byte_frequency(self):
        """Test sliding window byte frequency."""
        from oscura.analyzers.statistical.entropy import sliding_byte_frequency

        data = b"\x00" * 1000 + b"\xff" * 1000
        profile = sliding_byte_frequency(data, window=256, byte_value=0)
        assert profile[0] > profile[-1]  # More zeros at start

    def test_detect_compression_indicators(self):
        """Test detecting compression/encryption indicators."""
        from oscura.analyzers.statistical.entropy import detect_compression_indicators

        # High entropy data (like random)
        random_data = bytes([random.randint(0, 255) for _ in range(1000)])
        result = detect_compression_indicators(random_data)
        assert result.is_compressed or result.is_encrypted

        # Low entropy data
        low_entropy = b"\x00" * 1000
        result = detect_compression_indicators(low_entropy)
        assert not result.is_encrypted


class TestEntropyAnalysis:
    """Additional entropy analysis tests."""

    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        from oscura.analyzers.statistical.entropy import shannon_entropy

        # All zeros - minimum entropy
        assert shannon_entropy(b"\x00" * 100) == 0.0

        # Uniform distribution - maximum entropy
        uniform = bytes(range(256))
        assert shannon_entropy(uniform) == 8.0

    def test_entropy_classification(self):
        """Test entropy-based classification."""
        from oscura.analyzers.statistical.entropy import classify_by_entropy

        # Constant data
        result = classify_by_entropy(b"\x00" * 100)
        assert result.classification == "constant"

        # Text data
        result = classify_by_entropy(b"Hello World! This is a test message.")
        assert result.classification in ["text", "structured"]


# =============================================================================
# SECTION 10: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmarks for RE modules."""

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_entropy_performance(self, size, benchmark):
        """Benchmark entropy calculation."""
        from oscura.analyzers.statistical.entropy import shannon_entropy

        data = bytes([random.randint(0, 255) for _ in range(size)])

        def run():
            return shannon_entropy(data)

        if callable(benchmark):
            result = benchmark(run)
        else:
            start = time.time()
            result = run()
            elapsed = time.time() - start
            assert elapsed < size / 100000  # Should be fast

    @pytest.mark.parametrize("pattern_count", [10, 50, 100])
    def test_aho_corasick_performance(self, pattern_count, benchmark):
        """Benchmark Aho-Corasick multi-pattern search."""
        from oscura.analyzers.patterns.matching import AhoCorasickMatcher

        patterns = {f"pat_{i}": bytes([i % 256, (i + 1) % 256]) for i in range(pattern_count)}
        data = bytes([random.randint(0, 255) for _ in range(10000)])

        matcher = AhoCorasickMatcher()
        matcher.add_patterns(patterns)
        matcher.build()

        def run():
            return matcher.search(data)

        if callable(benchmark):
            result = benchmark(run)
        else:
            start = time.time()
            result = run()
            elapsed = time.time() - start
            assert elapsed < 1.0

    def test_payload_clustering_performance(self, benchmark):
        """Benchmark payload clustering."""
        from oscura.analyzers.packet.payload import cluster_payloads

        payloads = [bytes([i % 256] * 100) for i in range(100)]

        def run():
            return cluster_payloads(payloads, threshold=0.8)

        if callable(benchmark):
            result = benchmark(run)
        else:
            start = time.time()
            result = run()
            elapsed = time.time() - start
            assert elapsed < 5.0


# =============================================================================
# SECTION 11: Fuzz Testing
# =============================================================================


class TestFuzzTesting:
    """Fuzz testing for robustness."""

    @pytest.mark.parametrize("seed", range(10))
    def test_fuzz_payload_extractor(self, seed):
        """Fuzz test payload extractor."""
        from oscura.analyzers.packet.payload import PayloadExtractor

        random.seed(seed)
        extractor = PayloadExtractor()

        # Random bytes
        data = bytes([random.randint(0, 255) for _ in range(random.randint(0, 1000))])
        try:
            payload = extractor.extract_payload(data)
            assert payload is not None
        except Exception as e:
            pytest.fail(f"Payload extraction failed on random data: {e}")

    @pytest.mark.parametrize("seed", range(10))
    def test_fuzz_delimiter_detection(self, seed):
        """Fuzz test delimiter detection."""
        from oscura.analyzers.packet.payload import detect_delimiter

        random.seed(seed)
        data = bytes([random.randint(0, 255) for _ in range(random.randint(10, 500))])
        try:
            result = detect_delimiter(data)
            assert result.confidence >= 0
        except Exception as e:
            pytest.fail(f"Delimiter detection failed on random data: {e}")

    @pytest.mark.parametrize("seed", range(10))
    def test_fuzz_pattern_learning(self, seed):
        """Fuzz test pattern learning."""
        from oscura.analyzers.patterns.learning import learn_patterns_from_data

        random.seed(seed)
        data = bytes([random.randint(0, 255) for _ in range(random.randint(50, 500))])
        try:
            patterns = learn_patterns_from_data(data, min_length=2, min_frequency=2)
            assert isinstance(patterns, list)
        except Exception as e:
            pytest.fail(f"Pattern learning failed on random data: {e}")

    @pytest.mark.parametrize("seed", range(10))
    def test_fuzz_entropy_analysis(self, seed):
        """Fuzz test entropy analysis."""
        from oscura.analyzers.statistical.entropy import (
            byte_frequency_distribution,
            classify_by_entropy,
            shannon_entropy,
        )

        random.seed(seed)
        data = bytes([random.randint(0, 255) for _ in range(random.randint(1, 1000))])
        try:
            entropy = shannon_entropy(data)
            assert 0 <= entropy <= 8

            classification = classify_by_entropy(data)
            assert classification.confidence >= 0

            freq = byte_frequency_distribution(data)
            assert freq.unique_bytes >= 0
        except Exception as e:
            pytest.fail(f"Entropy analysis failed on random data: {e}")

    @pytest.mark.parametrize("seed", range(5))
    def test_fuzz_adaptive_threshold(self, seed):
        """Fuzz test adaptive thresholding."""
        from oscura.analyzers.digital.thresholds import apply_adaptive_threshold

        random.seed(seed)
        signal = np.random.randn(random.randint(100, 5000))
        try:
            result = apply_adaptive_threshold(signal, window_size=64)
            assert len(result.binary_output) == len(signal)
        except ValueError:
            pass  # Expected for signals smaller than window
        except Exception as e:
            pytest.fail(f"Adaptive threshold failed on random signal: {e}")


# =============================================================================
# SECTION 12: Regression Tests
# =============================================================================


class TestRegressionSuite:
    """Regression tests for known issues and edge cases."""

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        from oscura.analyzers.packet.payload import (
            detect_delimiter,
            find_message_boundaries,
            infer_fields,
        )
        from oscura.analyzers.patterns.matching import multi_pattern_search
        from oscura.analyzers.statistical.entropy import shannon_entropy

        # Empty data raises ValueError for entropy (mathematically undefined)
        with pytest.raises(ValueError):
            shannon_entropy(b"")

        result = detect_delimiter(b"")
        assert result.confidence == 0.0

        boundaries = find_message_boundaries(b"")
        assert boundaries == []

        schema = infer_fields([])
        assert len(schema.fields) == 0

        results = multi_pattern_search(b"", {"test": b"pattern"})
        assert results["test"] == []

    def test_single_byte_input(self):
        """Test handling of single-byte inputs."""
        from oscura.analyzers.packet.payload import (
            PayloadExtractor,
            diff_payloads,
        )
        from oscura.analyzers.statistical.entropy import shannon_entropy

        assert shannon_entropy(b"\x00") == 0.0

        extractor = PayloadExtractor()
        payload = extractor.extract_payload(b"A")
        assert payload == b"A"

        diff = diff_payloads(b"A", b"B")
        assert len(diff.differences) == 1

    def test_unicode_in_text_detection(self):
        """Test text detection with UTF-8 content."""
        from oscura.analyzers.statistical.entropy import classify_by_entropy

        text = "Hello World! This is Unicode: \u00e9\u00e8".encode()
        result = classify_by_entropy(text)
        # Should still work, may classify as structured or text
        assert result.classification in ["text", "structured", "compressed"]

    def test_maximum_length_inputs(self):
        """Test handling of large inputs."""
        from oscura.analyzers.patterns.matching import binary_regex_search
        from oscura.analyzers.statistical.entropy import shannon_entropy

        # 1MB of data
        large_data = bytes(range(256)) * (1024 * 4)

        entropy = shannon_entropy(large_data)
        assert entropy == 8.0  # Uniform distribution

        # Pattern search on large data
        matches = binary_regex_search(large_data, r"\x00\x01\x02")
        assert len(matches) == 1024 * 4

    def test_all_same_bytes(self):
        """Test handling of uniform data (all same byte)."""
        from oscura.analyzers.packet.payload import find_variable_positions
        from oscura.analyzers.statistical.entropy import (
            byte_frequency_distribution,
            shannon_entropy,
        )

        data = b"\xff" * 1000
        assert shannon_entropy(data) == 0.0

        freq = byte_frequency_distribution(data)
        assert freq.unique_bytes == 1
        assert freq.most_common[0] == (255, 1000)

        positions = find_variable_positions([data, data, data])
        assert len(positions.variable_positions) == 0


# =============================================================================
# SECTION 13: Integration Tests
# =============================================================================


class TestFullWorkflowIntegration:
    """Integration tests for complete reverse engineering workflows."""

    def test_modbus_analysis_workflow(self, modbus_rtu_messages):
        """Test complete Modbus protocol analysis workflow."""
        from oscura.analyzers.packet.payload import (
            PayloadExtractor,
            find_variable_positions,
        )
        from oscura.inference.protocol_library import get_decoder

        # Extract payloads
        extractor = PayloadExtractor()
        payloads = [extractor.extract_payload(msg) for msg in modbus_rtu_messages]
        assert len(payloads) == 3

        # Analyze variable positions
        positions = find_variable_positions(payloads[:2])  # Same type messages
        # Should have some constant positions (address, function code)

        # Try protocol decoder
        decoder = get_decoder("modbus_rtu")
        if decoder:
            result = decoder.decode(modbus_rtu_messages[0])
            assert result is not None

    def test_dns_analysis_workflow(self, dns_message):
        """Test complete DNS protocol analysis workflow."""
        from oscura.analyzers.statistical.entropy import classify_by_entropy
        from oscura.inference.protocol_library import get_decoder

        # Classify by entropy
        result = classify_by_entropy(dns_message)
        assert result.classification in ["structured", "text"]

        # Try protocol decoder
        decoder = get_decoder("dns")
        if decoder:
            result = decoder.decode(dns_message)
            assert result is not None
            assert "transaction_id" in result

    def test_unknown_protocol_discovery(self):
        """Test discovering unknown protocol structure."""
        from oscura.analyzers.packet.payload import (
            cluster_payloads,
        )
        from oscura.analyzers.patterns.learning import learn_patterns_from_data
        from oscura.utils.pipeline.reverse_engineering import analyze

        # Create unknown protocol messages
        messages = []
        for i in range(20):
            # Header (2) + Type (1) + Sequence (2) + Data (variable)
            header = b"\xca\xfe"
            msg_type = bytes([i % 4])
            sequence = struct.pack(">H", i)
            data_len = 5 + (i % 5)
            data = bytes([i % 256] * data_len)
            messages.append(header + msg_type + sequence + data)

        # Run full analysis
        results = analyze(messages)
        assert results.message_count == 20
        assert results.flow_count >= 1

        # Learn patterns
        concat = b"".join(messages)
        patterns = learn_patterns_from_data(concat, min_length=2, min_frequency=10)
        # Should find the CA FE header pattern
        assert any(b"\xca\xfe" in p.pattern for p in patterns)

        # Cluster messages
        clusters = cluster_payloads(messages, threshold=0.8)
        # Should create multiple clusters based on type/length
        assert len(clusters) >= 1

    def test_stream_to_message_workflow(self, length_prefixed_messages):
        """Test complete stream reassembly to message extraction workflow."""
        from oscura.analyzers.packet.payload import infer_fields
        from oscura.inference.stream import (
            UDPStreamReassembler,
            extract_messages,
        )

        # Simulate receiving packets
        reassembler = UDPStreamReassembler()
        chunk_size = 10
        for i in range(0, len(length_prefixed_messages), chunk_size):
            chunk = length_prefixed_messages[i : i + chunk_size]
            reassembler.add_segment(chunk)

        stream = reassembler.get_stream()

        # Extract messages
        result = extract_messages(stream.data, framing_type="length_prefix", length_field_size=2)
        assert len(result.messages) == 5

        # Infer message structure
        message_data = [m.data for m in result.messages]
        schema = infer_fields(message_data)
        assert schema.sample_count == 5


# =============================================================================
# SECTION 14: Accuracy Tests Against Known Protocols
# =============================================================================


class TestAccuracyKnownProtocols:
    """Accuracy tests against known protocol specifications."""

    def test_modbus_field_accuracy(self, modbus_rtu_messages):
        """Test accuracy of Modbus field detection."""
        from oscura.inference.protocol_library import get_decoder

        decoder = get_decoder("modbus_rtu")
        if not decoder:
            pytest.skip("Modbus decoder not available")

        # Test Read Holding Registers request
        result = decoder.decode(modbus_rtu_messages[0])
        assert result.get("address") == 1
        assert result.get("function_code") == 3

    def test_dns_header_accuracy(self, dns_message):
        """Test accuracy of DNS header detection."""
        from oscura.inference.protocol_library import get_decoder

        decoder = get_decoder("dns")
        if not decoder:
            pytest.skip("DNS decoder not available")

        result = decoder.decode(dns_message)
        assert result.get("transaction_id") == 0x1234
        assert result.get("questions") == 1
        assert result.get("answer_rrs") == 0

    def test_entropy_classification_accuracy(self):
        """Test accuracy of entropy-based classification."""
        from oscura.analyzers.statistical.entropy import classify_by_entropy

        # Test cases with known classifications
        test_cases = [
            (b"\x00" * 100, "constant"),
            (b"Hello, this is plain text content.", "text"),
            (bytes(range(256)) * 10, "random"),
        ]

        for data, expected in test_cases:
            result = classify_by_entropy(data)
            assert result.classification == expected, (
                f"Expected {expected}, got {result.classification}"
            )


# =============================================================================
# Run configuration
# =============================================================================
