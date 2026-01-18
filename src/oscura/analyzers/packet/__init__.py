"""Packet and DAQ analysis package.

This module provides packet parsing, streaming analysis, and error-tolerant
DAQ features.

- RE-PAY-001: Payload Extraction Framework
- RE-PAY-002: Payload Pattern Search
- RE-PAY-003: Payload Delimiter Detection
- RE-PAY-004: Payload Field Inference
- RE-PAY-005: Payload Comparison and Differential Analysis
"""

from oscura.analyzers.packet.daq import (
    BitErrorAnalysis,
    DAQGap,
    DAQGapAnalysis,
    ErrorPattern,
    FuzzyMatch,
    JitterCompensationResult,
    PacketRecoveryResult,
    analyze_bit_errors,
    compensate_timestamp_jitter,
    detect_gaps,
    detect_gaps_by_samples,
    detect_gaps_by_timestamps,
    error_tolerant_decode,
    fuzzy_pattern_search,
    robust_packet_parse,
)
from oscura.analyzers.packet.metrics import (
    JitterResult,
    LatencyResult,
    LossResult,
    PacketInfo,
    ThroughputResult,
    jitter,
    latency,
    loss_rate,
    throughput,
    windowed_throughput,
)
from oscura.analyzers.packet.parser import (
    BinaryParser,
    parse_tlv,
)

# RE-PAY-001 through RE-PAY-005: Payload analysis
from oscura.analyzers.packet.payload import (
    DelimiterResult,
    FieldInferrer,
    # RE-PAY-004: Field inference
    InferredField,
    LengthPrefixResult,
    MessageBoundary,
    MessageSchema,
    PatternMatch,
    PayloadCluster,
    PayloadDiff,
    # Classes
    PayloadExtractor,
    # Data classes
    PayloadInfo,
    VariablePositions,
    cluster_payloads,
    compute_similarity,
    correlate_request_response,
    # RE-PAY-003: Delimiter detection
    detect_delimiter,
    detect_field_types,
    detect_length_prefix,
    # RE-PAY-005: Comparison
    diff_payloads,
    filter_by_pattern,
    find_checksum_fields,
    find_common_bytes,
    find_message_boundaries,
    find_sequence_fields,
    find_variable_positions,
    infer_fields,
    # RE-PAY-002: Pattern search
    search_pattern,
    search_patterns,
    segment_messages,
)
from oscura.analyzers.packet.stream import (
    StreamPacket,
    batch,
    pipeline,
    skip,
    stream_delimited,
    stream_file,
    stream_packets,
    stream_records,
    take,
)

__all__ = [
    # Parser (PKT-001, PKT-002)
    "BinaryParser",
    "BitErrorAnalysis",
    # DAQ Gap Detection (PKT-008)
    "DAQGap",
    "DAQGapAnalysis",
    "DelimiterResult",
    "ErrorPattern",
    "FieldInferrer",
    # DAQ (DAQ-001 through DAQ-005)
    "FuzzyMatch",
    # RE-PAY-004: Field inference
    "InferredField",
    "JitterCompensationResult",
    "JitterResult",
    "LatencyResult",
    "LengthPrefixResult",
    "LossResult",
    "MessageBoundary",
    "MessageSchema",
    # Metrics (PKT-005, PKT-006, PKT-007, PKT-009)
    "PacketInfo",
    "PacketRecoveryResult",
    "PatternMatch",
    "PayloadCluster",
    "PayloadDiff",
    # Other payload exports
    "PayloadExtractor",
    # RE-PAY-001 through RE-PAY-005: Payload analysis
    "PayloadInfo",
    # Stream (PKT-003)
    "StreamPacket",
    "ThroughputResult",
    "VariablePositions",
    "analyze_bit_errors",
    "batch",
    "cluster_payloads",
    "compensate_timestamp_jitter",
    "compute_similarity",
    "correlate_request_response",
    "detect_delimiter",
    "detect_field_types",
    "detect_gaps",
    "detect_gaps_by_samples",
    "detect_gaps_by_timestamps",
    "detect_length_prefix",
    "diff_payloads",
    "error_tolerant_decode",
    "filter_by_pattern",
    "find_checksum_fields",
    "find_common_bytes",
    "find_message_boundaries",
    "find_sequence_fields",
    "find_variable_positions",
    "fuzzy_pattern_search",
    "infer_fields",
    "jitter",
    "latency",
    "loss_rate",
    "parse_tlv",
    "pipeline",
    "robust_packet_parse",
    "search_pattern",
    "search_patterns",
    "segment_messages",
    "skip",
    "stream_delimited",
    "stream_file",
    "stream_packets",
    "stream_records",
    "take",
    "throughput",
    "windowed_throughput",
]
