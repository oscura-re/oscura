"""Payload extraction and analysis framework for network packets.

    - RE-PAY-001: Payload Extraction Framework
    - RE-PAY-002: Payload Pattern Search
    - RE-PAY-003: Payload Delimiter Detection
    - RE-PAY-004: Payload Field Inference
    - RE-PAY-005: Payload Comparison and Differential Analysis

This module provides comprehensive payload extraction from PCAP packets,
pattern search capabilities, delimiter detection, and comparison tools.

This is the public API module that re-exports functionality from specialized modules:
- payload_analysis: Field inference, diff, clustering
- payload_patterns: Pattern search, delimiters, boundaries
- payload_extraction: Payload extraction utilities
"""

from __future__ import annotations

# RE-PAY-004 & RE-PAY-005: Field Inference and Comparison
from oscura.analyzers.packet.payload_analysis import (
    FieldInferrer,
    InferredField,
    MessageSchema,
    PayloadCluster,
    PayloadDiff,
    VariablePositions,
    cluster_payloads,
    compute_similarity,
    correlate_request_response,
    detect_field_types,
    diff_payloads,
    find_checksum_fields,
    find_common_bytes,
    find_sequence_fields,
    find_variable_positions,
    infer_fields,
)

# RE-PAY-001: Payload Extraction
from oscura.analyzers.packet.payload_extraction import (
    PayloadExtractor,
    PayloadInfo,
)

# RE-PAY-002 & RE-PAY-003: Pattern Search and Delimiter Detection
from oscura.analyzers.packet.payload_patterns import (
    DelimiterResult,
    LengthPrefixResult,
    MessageBoundary,
    PatternMatch,
    detect_delimiter,
    detect_length_prefix,
    filter_by_pattern,
    find_message_boundaries,
    search_pattern,
    search_patterns,
    segment_messages,
)

__all__ = [
    # RE-PAY-003: Delimiter Detection
    "DelimiterResult",
    "FieldInferrer",
    # RE-PAY-004: Field Inference
    "InferredField",
    "LengthPrefixResult",
    "MessageBoundary",
    "MessageSchema",
    # RE-PAY-002: Pattern Search
    "PatternMatch",
    "PayloadCluster",
    # RE-PAY-005: Payload Comparison
    "PayloadDiff",
    "PayloadExtractor",
    # RE-PAY-001: Payload Extraction
    "PayloadInfo",
    "VariablePositions",
    "cluster_payloads",
    "compute_similarity",
    "correlate_request_response",
    "detect_delimiter",
    "detect_field_types",
    "detect_length_prefix",
    "diff_payloads",
    "filter_by_pattern",
    "find_checksum_fields",
    "find_common_bytes",
    "find_message_boundaries",
    "find_sequence_fields",
    "find_variable_positions",
    "infer_fields",
    "search_pattern",
    "search_patterns",
    "segment_messages",
]
