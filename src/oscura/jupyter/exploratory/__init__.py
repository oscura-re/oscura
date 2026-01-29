"""Exploratory analysis for unknown and legacy signals.

This package provides tools for analyzing signals from unknown systems,
legacy hardware, and corrupted or noisy data.

- UNKNOWN-001: Binary Field Detection
- UNKNOWN-002: Protocol Auto-Detection with Fuzzy Matching
- UNKNOWN-003: Unknown Signal Characterization
- UNKNOWN-004: Pattern Frequency Analysis
- UNKNOWN-005: Reverse Engineering Workflow
"""

from oscura.jupyter.exploratory.error_recovery import (
    ErrorContext,
    partial_decode,
    recover_corrupted_data,
    retry_with_adjustment,
)
from oscura.jupyter.exploratory.fuzzy import (
    fuzzy_pattern_match,
    fuzzy_protocol_detect,
    fuzzy_timing_match,
)
from oscura.jupyter.exploratory.fuzzy_advanced import (
    AlignmentResult,
    PositionAnalysis,
    VariantCharacterization,
    align_sequences,
    align_two_sequences,
    characterize_variants,
    compute_conservation_scores,
)
from oscura.jupyter.exploratory.legacy import (
    assess_signal_quality,
    characterize_test_points,
    cross_correlate_multi_reference,
    detect_logic_families_multi_channel,
)
from oscura.jupyter.exploratory.parse import (
    DecodedFrame,
    ErrorTolerance,
    TimestampCorrection,
    correct_timestamp_jitter,
    decode_with_error_tolerance,
)
from oscura.jupyter.exploratory.recovery import (
    ErrorAnalysis,
    ErrorPattern,
    analyze_bit_errors,
    generate_error_visualization_data,
)
from oscura.jupyter.exploratory.sync import (
    PacketParseResult,
    RecoveryStrategy,
    SyncMatch,
    fuzzy_sync_search,
    parse_variable_length_packets,
)
from oscura.jupyter.exploratory.unknown import (
    analyze_pattern_frequency,
    characterize_unknown_signal,
    detect_binary_fields,
    reverse_engineer_protocol,
)

__all__ = [
    # Advanced fuzzy (FUZZY-004, FUZZY-005)
    "AlignmentResult",
    "DecodedFrame",
    "ErrorAnalysis",
    # Error recovery
    "ErrorContext",
    # DAQ - Bit error analysis (DAQ-005)
    "ErrorPattern",
    # DAQ - Error-tolerant parsing (DAQ-003, DAQ-004)
    "ErrorTolerance",
    "PacketParseResult",
    "PositionAnalysis",
    "RecoveryStrategy",
    # DAQ - Fuzzy sync search (DAQ-001, DAQ-002)
    "SyncMatch",
    "TimestampCorrection",
    "VariantCharacterization",
    "align_sequences",
    "align_two_sequences",
    "analyze_bit_errors",
    # Unknown signal analysis
    "analyze_pattern_frequency",
    # Legacy analysis
    "assess_signal_quality",
    "characterize_test_points",
    "characterize_unknown_signal",
    "characterize_variants",
    "compute_conservation_scores",
    "correct_timestamp_jitter",
    "cross_correlate_multi_reference",
    "decode_with_error_tolerance",
    "detect_binary_fields",
    "detect_logic_families_multi_channel",
    # Fuzzy matching
    "fuzzy_pattern_match",
    "fuzzy_protocol_detect",
    "fuzzy_sync_search",
    "fuzzy_timing_match",
    "generate_error_visualization_data",
    "parse_variable_length_packets",
    "partial_decode",
    "recover_corrupted_data",
    "retry_with_adjustment",
    "reverse_engineer_protocol",
]
